using StaticArrays
using FFTW

# helpers 
_box_length(mesh::Trixi.TreeMesh) = mesh.tree.length_level_0

function _cells_per_dim(mesh::Trixi.TreeMesh)
    cell_ids = Trixi.leaf_cells(mesh.tree)
    level = mesh.tree.levels[first(cell_ids)]
    return 2^level
end

#lookup table so we can go from (cx,cy,cz) -> element index
function _build_cell_lookup(mesh::Trixi.TreeMesh, cache)
    N_cells = _cells_per_dim(mesh)
    dx_cell = _box_length(mesh) / N_cells
    x0 = mesh.tree.center_level_0[1] - _box_length(mesh) / 2  # domain left edge

    lookup = zeros(Int, N_cells, N_cells, N_cells)
    elems = cache.elements
    n_elems = length(elems.cell_ids)

    for e in 1:n_elems
        cid = elems.cell_ids[e]
        center = Trixi.cell_coordinates(mesh.tree, cid)
        # convert physical coords to integer cell indices
        ci = clamp(floor(Int, (center[1] - x0) / dx_cell) + 1, 1, N_cells)
        cj = clamp(floor(Int, (center[2] - x0) / dx_cell) + 1, 1, N_cells)
        ck = clamp(floor(Int, (center[3] - x0) / dx_cell) + 1, 1, N_cells)
        lookup[ci, cj, ck] = e
    end

    return lookup, x0, dx_cell
end

function measure_power_spectrum(semi, u_ode)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, semi)

    L = _box_length(mesh)
    N_cells = _cells_per_dim(mesh)
    k = Trixi.polydeg(solver)
    N_grid = N_cells * (k + 1)  # uniform grid resolution

    # interpolate the DG solution onto a uniform grid first
    vx, vy, vz = _vel_to_grid(u, mesh, equations, solver, cache, N_grid, L)

    # then FFT each velocity component
    fvx = rfft(vx)
    fvy = rfft(vy)
    fvz = rfft(vz)

    k_centers, Ek, n_modes, bin_edges = _shell_spectrum(fvx, fvy, fvz, L, 2000, N_grid)
    return k_centers, Ek, n_modes, bin_edges
end

function _vel_to_grid(u, mesh::Trixi.TreeMesh, equations, solver, cache,
                      N_grid::Int, L::Float64)
    vx = zeros(Float64, N_grid, N_grid, N_grid)
    vy = zeros(Float64, N_grid, N_grid, N_grid)
    vz = zeros(Float64, N_grid, N_grid, N_grid)

    nodes = solver.basis.nodes
    bary_w = Trixi.barycentric_weights(nodes)
    n_nodes = length(nodes)
    N_cells = _cells_per_dim(mesh)

    lookup, x0, dx_cell = _build_cell_lookup(mesh, cache)

    # for each output point we need the 1D Lagrange weights along each axis
    # the sub-points are cell-centered within each DG node interval
    half_cell = dx_cell / 2
    lag_w = Vector{Vector{Float64}}(undef, n_nodes)
    for s in 0:(n_nodes - 1)
        # map sub-point to reference coords [-1, 1]
        xi = ((s + 0.5) / n_nodes * dx_cell - half_cell) / half_cell
        lag_w[s + 1] = Trixi.lagrange_interpolating_polynomials(xi, nodes, bary_w)
    end

    # precompute the full 3D tensor product weights
    w3d = Array{Array{Float64, 3}, 3}(undef, n_nodes, n_nodes, n_nodes)
    for sz in 0:(n_nodes - 1), sy in 0:(n_nodes - 1), sx in 0:(n_nodes - 1)
        w = zeros(Float64, n_nodes, n_nodes, n_nodes)
        @inbounds for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
            w[i, j, k] = lag_w[sx + 1][i] * lag_w[sy + 1][j] * lag_w[sz + 1][k]
        end
        w3d[sx + 1, sy + 1, sz + 1] = w
    end

    # for each cell, interpolate u onto the uniform sub-grid points
    @inbounds for cz in 1:N_cells, cy in 1:N_cells, cx in 1:N_cells
        eid = lookup[cx, cy, cz]

        for sz in 0:(n_nodes - 1)
            iz = (cz - 1) * n_nodes + sz + 1
            for sy in 0:(n_nodes - 1)
                iy = (cy - 1) * n_nodes + sy + 1
                for sx in 0:(n_nodes - 1)
                    ix = (cx - 1) * n_nodes + sx + 1

                    w = w3d[sx + 1, sy + 1, sz + 1]
                    rho = rv1 = rv2 = rv3 = 0.0

                    # dot the DG nodal values with the interpolation weights
                    for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
                        ww = w[i, j, k]
                        rho += u[1, i, j, k, eid] * ww
                        rv1 += u[2, i, j, k, eid] * ww
                        rv2 += u[3, i, j, k, eid] * ww
                        rv3 += u[4, i, j, k, eid] * ww
                    end

                    # conservative -> primitive
                    vx[ix, iy, iz] = rv1 / rho
                    vy[ix, iy, iz] = rv2 / rho
                    vz[ix, iy, iz] = rv3 / rho
                end
            end
        end
    end

    return vx, vy, vz
end

function _shell_spectrum(fvx, fvy, fvz, L::Float64, n_bins::Int, N_grid::Int)
    dk = 2 * pi / L
    k_nyq = pi * N_grid / L
    k_max = sqrt(3) * k_nyq  # corner of the Fourier cube

    kx_vals = collect(0:div(N_grid, 2))
    ky_vals = fftfreq(N_grid, N_grid)
    kz_vals = fftfreq(N_grid, N_grid)

    # log-spaced bins
    bin_edges = 10.0 .^ range(log10(0.5 * dk), log10(k_max); length = n_bins + 1)

    E_bins = zeros(Float64, n_bins)
    n_modes = zeros(Int, n_bins)
    n_full = zeros(Int, n_bins)  # counts both conjugate pairs

    norm = L^3 / N_grid^6
    n_kx = div(N_grid, 2) + 1

    for ikz in 1:N_grid
        kz = kz_vals[ikz] * dk
        for iky in 1:N_grid
            ky = ky_vals[iky] * dk
            for ikx in 1:n_kx
                kx = kx_vals[ikx] * dk
                k_mag = sqrt(kx^2 + ky^2 + kz^2)
                k_mag == 0.0 && continue

                # rfft only keeps kx >= 0; modes with 0 < kx < N/2 each represent
                # two conjugate modes in the full DFT, so we weight them by 2
                mult = (ikx == 1 || (iseven(N_grid) && ikx == n_kx)) ? 1 : 2

                P = (abs2(fvx[ikx, iky, ikz]) + abs2(fvy[ikx, iky, ikz]) +
                     abs2(fvz[ikx, iky, ikz])) * norm * mult

                b = searchsortedfirst(bin_edges, k_mag) - 1
                if 1 <= b <= n_bins
                    E_bins[b] += P
                    n_modes[b] += 1
                    n_full[b] += mult
                end
            end
        end
    end

    k_centers = zeros(Float64, n_bins)
    Ek = zeros(Float64, n_bins)
    for i in 1:n_bins
        k_centers[i] = sqrt(bin_edges[i] * bin_edges[i + 1])
        if n_full[i] > 0
            # divide by bin width to get spectral density E(k)
            Ek[i] = E_bins[i] / (bin_edges[i + 1] - bin_edges[i])
        end
    end

    return k_centers, Ek, n_modes, bin_edges
end

function rebin_spectrum(k_centers, Ek, n_modes, bin_edges;
                        min_modes_low_k::Int = 1,
                        min_modes_high_k::Int = 10,
                        k_split::Union{Nothing, Float64} = nothing)
    n = length(k_centers)

    # default split point: geometric mean of the populated k range
    if isnothing(k_split)
        k_valid = k_centers[n_modes .> 0]
        isempty(k_valid) && return k_centers, Ek, n_modes
        k_split = sqrt(first(k_valid) * last(k_valid))
        println(k_split)
    end

    k_out = Float64[]
    E_out = Float64[]
    cnt = Int[]

    i = 1
    while i <= n
        n_modes[i] == 0 && (i += 1; continue)

        min_m = k_centers[i] < k_split ? min_modes_low_k : min_modes_high_k

        sum_Edk = 0.0
        dk_tot = 0.0
        m_tot = 0
        i0 = i
        j = i

        # accumulate bins until we have enough modes
        while j <= n && m_tot < min_m
            if n_modes[j] > 0
                dkj = bin_edges[j + 1] - bin_edges[j]
                sum_Edk += Ek[j] * dkj
                dk_tot += dkj
                m_tot += n_modes[j]
            end
            j += 1
        end

        if m_tot > 0 && dk_tot > 0
            push!(k_out, sqrt(bin_edges[i0] * bin_edges[j]))
            push!(E_out, sum_Edk / dk_tot)
            push!(cnt, m_tot)
        end

        i = j
    end

    return k_out, E_out, cnt
end