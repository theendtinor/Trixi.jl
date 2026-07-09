using StaticArrays
using FFTW
using LinearAlgebra: mul!
using Base.Threads

# --- helpers ---

_box_length(mesh::Trixi.TreeMesh) = mesh.tree.length_level_0

function _cells_per_dim(mesh::Trixi.TreeMesh)
    cell_ids = Trixi.leaf_cells(mesh.tree)
    level = mesh.tree.levels[first(cell_ids)]
    return 2^level
end

function _build_cell_lookup(mesh::Trixi.TreeMesh, cache)
    N_cells = _cells_per_dim(mesh)
    dx_cell = _box_length(mesh) / N_cells
    x0 = mesh.tree.center_level_0[1] - _box_length(mesh) / 2

    lookup = zeros(Int, N_cells, N_cells, N_cells)
    elems = cache.elements
    n_elems = length(elems.cell_ids)

    for e in 1:n_elems
        cid = elems.cell_ids[e]
        center = Trixi.cell_coordinates(mesh.tree, cid)
        ci = clamp(floor(Int, (center[1] - x0) / dx_cell) + 1, 1, N_cells)
        cj = clamp(floor(Int, (center[2] - x0) / dx_cell) + 1, 1, N_cells)
        ck = clamp(floor(Int, (center[3] - x0) / dx_cell) + 1, 1, N_cells)
        lookup[ci, cj, ck] = e
    end

    return lookup, x0, dx_cell
end

# workspace holding the velocity grids, FFT plan and the Fourier arrays, so we
# don't re-allocate this stuff for every snapshot

struct PSWorkspace{P}
    vx::Array{Float64, 3}
    vy::Array{Float64, 3}
    vz::Array{Float64, 3}
    fvx::Array{ComplexF64, 3}
    fvy::Array{ComplexF64, 3}
    fvz::Array{ComplexF64, 3}
    plan::P
    N_grid::Int
    L::Float64
end

function PSWorkspace(N_grid::Int, L::Float64)
    vx = zeros(Float64, N_grid, N_grid, N_grid)
    vy = zeros(Float64, N_grid, N_grid, N_grid)
    vz = zeros(Float64, N_grid, N_grid, N_grid)

    # MEASURE plan costs a few seconds up front but pays off over many snapshots
    plan = plan_rfft(vx; flags = FFTW.MEASURE)

    n_kx = div(N_grid, 2) + 1
    fvx = zeros(ComplexF64, n_kx, N_grid, N_grid)
    fvy = zeros(ComplexF64, n_kx, N_grid, N_grid)
    fvz = zeros(ComplexF64, n_kx, N_grid, N_grid)

    return PSWorkspace(vx, vy, vz, fvx, fvy, fvz, plan, N_grid, L)
end

# caches mesh/solver geometry (nodes, lookup table, lag1d) so we don't need to
# trixi_include on every run -- only depends on (N_cells, polydeg, L).
# save/load logic lives in ps_geometry_cache.jl, struct is defined here so the
# measure_PS overloads below can use it

struct PSGeometryCache
    nodes::Vector{Float64}       # GLL nodes on [-1,1], length = polydeg+1
    bary_w::Vector{Float64}      # barycentric weights
    lag1d::Matrix{Float64}       # (n_nodes × n_nodes): lag1d[s+1, i] = L_i(xi_s)
    lookup::Array{Int, 3}        # (N_cells, N_cells, N_cells) cell→elem-id
    N_cells::Int
    N_grid::Int                  # = N_cells * (polydeg + 1)
    L::Float64
end

function allocate_workspace(semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    L = _box_length(mesh)
    N_cells = _cells_per_dim(mesh)
    k = Trixi.polydeg(solver)
    N_grid = N_cells * (k + 1)
    return PSWorkspace(N_grid, L)
end

# same thing but from a cache, no semi needed
function allocate_workspace(cache::PSGeometryCache)
    return PSWorkspace(cache.N_grid, cache.L)
end

# tensor-product interpolation done as 3 separate 1D passes instead of one
# big 3D contraction, much faster

function _vel_to_grid_fast!(ws::PSWorkspace, u, mesh::Trixi.TreeMesh,
                            equations, solver, cache)
    N_grid = ws.N_grid
    vx, vy, vz = ws.vx, ws.vy, ws.vz

    fill!(vx, 0.0)
    fill!(vy, 0.0)
    fill!(vz, 0.0)

    nodes = solver.basis.nodes
    bary_w = Trixi.barycentric_weights(nodes)
    n = length(nodes)   # n_nodes = polydeg + 1
    N_cells = _cells_per_dim(mesh)

    lookup, x0, dx_cell = _build_cell_lookup(mesh, cache)

    # --- precompute 1D interpolation weights for each sub-point ----
    # lag1d[s+1][i]  =  L_i(xi_s),  s in 0:n-1,  i in 1:n
    half_cell = dx_cell / 2
    lag1d = Vector{Vector{Float64}}(undef, n)
    for s in 0:(n - 1)
        xi = ((s + 0.5) / n * dx_cell - half_cell) / half_cell
        lag1d[s + 1] = Trixi.lagrange_interpolating_polynomials(xi, nodes, bary_w)
    end

    # --- threaded loop over cells ---
    total_cells = N_cells^3
    @threads for linear_idx in 1:total_cells
        rem1 = linear_idx - 1
        cz_m1, rem2 = divrem(rem1, N_cells^2)
        cy_m1, cx_m1 = divrem(rem2, N_cells)
        cx = cx_m1 + 1
        cy = cy_m1 + 1
        cz = cz_m1 + 1

        eid = lookup[cx, cy, cz]

        # ---- factored interpolation: 3 passes per sub-point ----
        # We interpolate 4 conservative vars (rho, rho*v1, rho*v2, rho*v3)
        # then convert to primitive at the very end.
        #
        # pass 1 (over i): tmp1[j, k, var] = sum_i  L_i(xi_sx) * u[var, i, j, k, eid]
        # pass 2 (over j): tmp2[k, var]     = sum_j  L_j(xi_sy) * tmp1[j, k, var]
        # pass 3 (over k): val[var]          = sum_k  L_k(xi_sz) * tmp2[k, var]

        # stack-allocate small work arrays
        tmp1 = MArray{Tuple{4, n, n}, Float64}(undef)  # [var, j, k]

        for sz in 0:(n - 1)
            iz = (cz - 1) * n + sz + 1
            wz = lag1d[sz + 1]

            for sy in 0:(n - 1)
                iy = (cy - 1) * n + sy + 1
                wy = lag1d[sy + 1]

                for sx in 0:(n - 1)
                    ix = (cx - 1) * n + sx + 1
                    wx = lag1d[sx + 1]

                    # --- pass 1: contract over i ---
                    @inbounds for kk in 1:n, jj in 1:n
                        r = 0.0
                        m1 = 0.0
                        m2 = 0.0
                        m3 = 0.0
                        for ii in 1:n
                            w = wx[ii]
                            r += u[1, ii, jj, kk, eid] * w
                            m1 += u[2, ii, jj, kk, eid] * w
                            m2 += u[3, ii, jj, kk, eid] * w
                            m3 += u[4, ii, jj, kk, eid] * w
                        end
                        tmp1[1, jj, kk] = r
                        tmp1[2, jj, kk] = m1
                        tmp1[3, jj, kk] = m2
                        tmp1[4, jj, kk] = m3
                    end

                    # --- pass 2: contract over j ---
                    r2 = 0.0
                    m12 = 0.0
                    m22 = 0.0
                    m32 = 0.0
                    # we fold pass 2 + pass 3 together for the k loop
                    @inbounds for kk in 1:n
                        r_j = 0.0
                        m1_j = 0.0
                        m2_j = 0.0
                        m3_j = 0.0
                        for jj in 1:n
                            w = wy[jj]
                            r_j += tmp1[1, jj, kk] * w
                            m1_j += tmp1[2, jj, kk] * w
                            m2_j += tmp1[3, jj, kk] * w
                            m3_j += tmp1[4, jj, kk] * w
                        end
                        # --- pass 3: contract over k ---
                        wk = wz[kk]
                        r2 += r_j * wk
                        m12 += m1_j * wk
                        m22 += m2_j * wk
                        m32 += m3_j * wk
                    end

                    # conservative -> primitive
                    inv_rho = 1.0 / r2
                    @inbounds vx[ix, iy, iz] = m12 * inv_rho
                    @inbounds vy[ix, iy, iz] = m22 * inv_rho
                    @inbounds vz[ix, iy, iz] = m32 * inv_rho
                end
            end
        end
    end

    return nothing
end

# same idea, but reads primitive vars straight from HDF5 instead of a live u

function _vel_to_grid_from_prim!(ws::PSWorkspace, prim_data,
                                 mesh::Trixi.TreeMesh, solver, cache)
    N_grid = ws.N_grid
    vx, vy, vz = ws.vx, ws.vy, ws.vz
    fill!(vx, 0.0)
    fill!(vy, 0.0)
    fill!(vz, 0.0)

    nodes = solver.basis.nodes
    bary_w = Trixi.barycentric_weights(nodes)
    n = length(nodes)
    N_cells = _cells_per_dim(mesh)

    lookup, x0, dx_cell = _build_cell_lookup(mesh, cache)

    half_cell = dx_cell / 2
    lag1d = Vector{Vector{Float64}}(undef, n)
    for s in 0:(n - 1)
        xi = ((s + 0.5) / n * dx_cell - half_cell) / half_cell
        lag1d[s + 1] = Trixi.lagrange_interpolating_polynomials(xi, nodes, bary_w)
    end

    nn = n  # nodes per dim
    nn3 = nn^3

    # prim_data indices: 1=rho (unused for velocity), 2=v1, 3=v2, 4=v3
    v1_raw = prim_data[2]
    v2_raw = prim_data[3]
    v3_raw = prim_data[4]

    total_cells = N_cells^3
    @threads for linear_idx in 1:total_cells
        rem1 = linear_idx - 1
        cz_m1, rem2 = divrem(rem1, N_cells^2)
        cy_m1, cx_m1 = divrem(rem2, N_cells)
        cx = cx_m1 + 1
        cy = cy_m1 + 1
        cz = cz_m1 + 1

        eid = lookup[cx, cy, cz]
        base = (eid - 1) * nn3

        # small stack buffers for the 1D contraction
        tmp1 = MArray{Tuple{3, nn, nn}, Float64}(undef)  # [var, j, k]

        for sz in 0:(nn - 1)
            iz = (cz - 1) * nn + sz + 1
            wz = lag1d[sz + 1]
            for sy in 0:(nn - 1)
                iy = (cy - 1) * nn + sy + 1
                wy = lag1d[sy + 1]
                for sx in 0:(nn - 1)
                    ix = (cx - 1) * nn + sx + 1
                    wx = lag1d[sx + 1]

                    # pass 1: contract i
                    @inbounds for kk in 1:nn, jj in 1:nn
                        a1 = 0.0
                        a2 = 0.0
                        a3 = 0.0
                        for ii in 1:nn
                            idx = base + ii + (jj - 1) * nn + (kk - 1) * nn^2
                            w = wx[ii]
                            a1 += v1_raw[idx] * w
                            a2 += v2_raw[idx] * w
                            a3 += v3_raw[idx] * w
                        end
                        tmp1[1, jj, kk] = a1
                        tmp1[2, jj, kk] = a2
                        tmp1[3, jj, kk] = a3
                    end

                    # pass 2+3: contract j then k
                    r1 = 0.0
                    r2 = 0.0
                    r3 = 0.0
                    @inbounds for kk in 1:nn
                        s1 = 0.0
                        s2 = 0.0
                        s3 = 0.0
                        for jj in 1:nn
                            w = wy[jj]
                            s1 += tmp1[1, jj, kk] * w
                            s2 += tmp1[2, jj, kk] * w
                            s3 += tmp1[3, jj, kk] * w
                        end
                        wk = wz[kk]
                        r1 += s1 * wk
                        r2 += s2 * wk
                        r3 += s3 * wk
                    end

                    @inbounds vx[ix, iy, iz] = r1
                    @inbounds vy[ix, iy, iz] = r2
                    @inbounds vz[ix, iy, iz] = r3
                end
            end
        end
    end

    return nothing
end

# same, but using a PSGeometryCache instead of semi

function _vel_to_grid_from_prim!(ws::PSWorkspace, prim_data,
                                 cache::PSGeometryCache)
    N_grid = ws.N_grid
    vx, vy, vz = ws.vx, ws.vy, ws.vz
    fill!(vx, 0.0)
    fill!(vy, 0.0)
    fill!(vz, 0.0)

    nodes = cache.nodes
    bary_w = cache.bary_w
    nn = length(nodes)          # nodes per dim = polydeg + 1
    nn3 = nn^3
    N_cells = cache.N_cells
    lookup = cache.lookup

    # lag1d[s+1, i] already stored as a matrix in the cache
    lag1d_mat = cache.lag1d   # (nn × nn)

    v1_raw = prim_data[2]
    v2_raw = prim_data[3]
    v3_raw = prim_data[4]

    total_cells = N_cells^3
    @threads for linear_idx in 1:total_cells
        rem1 = linear_idx - 1
        cz_m1, rem2 = divrem(rem1, N_cells^2)
        cy_m1, cx_m1 = divrem(rem2, N_cells)
        cx = cx_m1 + 1
        cy = cy_m1 + 1
        cz = cz_m1 + 1

        eid = lookup[cx, cy, cz]
        base = (eid - 1) * nn3

        tmp1 = MArray{Tuple{3, nn, nn}, Float64}(undef)

        for sz in 0:(nn - 1)
            iz = (cz - 1) * nn + sz + 1
            wz = @view lag1d_mat[sz + 1, :]
            for sy in 0:(nn - 1)
                iy = (cy - 1) * nn + sy + 1
                wy = @view lag1d_mat[sy + 1, :]
                for sx in 0:(nn - 1)
                    ix = (cx - 1) * nn + sx + 1
                    wx = @view lag1d_mat[sx + 1, :]

                    # pass 1: contract i
                    @inbounds for kk in 1:nn, jj in 1:nn
                        a1 = 0.0
                        a2 = 0.0
                        a3 = 0.0
                        for ii in 1:nn
                            idx = base + ii + (jj - 1) * nn + (kk - 1) * nn^2
                            w = wx[ii]
                            a1 += v1_raw[idx] * w
                            a2 += v2_raw[idx] * w
                            a3 += v3_raw[idx] * w
                        end
                        tmp1[1, jj, kk] = a1
                        tmp1[2, jj, kk] = a2
                        tmp1[3, jj, kk] = a3
                    end

                    # pass 2+3: contract j then k
                    r1 = 0.0
                    r2 = 0.0
                    r3 = 0.0
                    @inbounds for kk in 1:nn
                        s1 = 0.0
                        s2 = 0.0
                        s3 = 0.0
                        for jj in 1:nn
                            w = wy[jj]
                            s1 += tmp1[1, jj, kk] * w
                            s2 += tmp1[2, jj, kk] * w
                            s3 += tmp1[3, jj, kk] * w
                        end
                        wk = wz[kk]
                        r1 += s1 * wk
                        r2 += s2 * wk
                        r3 += s3 * wk
                    end

                    @inbounds vx[ix, iy, iz] = r1
                    @inbounds vy[ix, iy, iz] = r2
                    @inbounds vz[ix, iy, iz] = r3
                end
            end
        end
    end

    return nothing
end

# bins |k| into shells and sums power per shell, threaded over kz slices

function _shell_spectrum_fast!(ws::PSWorkspace, n_bins::Int;
                               ignore_mixed_parity::Bool = false)
    (; fvx, fvy, fvz, N_grid, L) = ws

    dk = 2pi / L
    k_nyq = pi * N_grid / L
    k_max = sqrt(3) * k_nyq

    # log-spaced bins
    log_lo = log10(0.5 * dk)
    log_hi = log10(k_max)
    bin_edges = 10.0 .^ range(log_lo, log_hi; length = n_bins + 1)

    inv_dk_log = n_bins / (log_hi - log_lo)

    kx_vals = collect(0:div(N_grid, 2))
    ky_vals = fftfreq(N_grid, N_grid)
    kz_vals = fftfreq(N_grid, N_grid)

    norm = L^3 / N_grid^6
    n_kx = div(N_grid, 2) + 1

    # thread-local accumulators (maxthreadid covers all threadpools in Julia 1.9+)
    nt = Threads.maxthreadid()
    E_local = [zeros(Float64, n_bins) for _ in 1:nt]
    n_local = [zeros(Int, n_bins) for _ in 1:nt]
    nf_local = [zeros(Int, n_bins) for _ in 1:nt]

    @threads for ikz in 1:N_grid
        tid = threadid()
        E_t = E_local[tid]
        n_t = n_local[tid]
        nf_t = nf_local[tid]
        kz = kz_vals[ikz] * dk

        @inbounds for iky in 1:N_grid
            ky = ky_vals[iky] * dk
            for ikx in 1:n_kx
                kx = kx_vals[ikx] * dk
                k_mag = sqrt(kx^2 + ky^2 + kz^2)
                k_mag == 0.0 && continue

                if ignore_mixed_parity
                    r2_val = round(Int, kx_vals[ikx]^2 + ky_vals[iky]^2 + kz_vals[ikz]^2)
                    r2_mod = r2_val % 4
                    if r2_mod == 1 || r2_mod == 2
                        continue
                    end
                end

                mult = (ikx == 1 || (iseven(N_grid) && ikx == n_kx)) ? 1 : 2

                P = (abs2(fvx[ikx, iky, ikz]) +
                     abs2(fvy[ikx, iky, ikz]) +
                     abs2(fvz[ikx, iky, ikz])) * norm * mult

                # analytic bin index for log-spaced bins
                b = floor(Int, (log10(k_mag) - log_lo) * inv_dk_log) + 1
                if 1 <= b <= n_bins
                    E_t[b] += P
                    n_t[b] += 1
                    nf_t[b] += mult
                end
            end
        end
    end

    # reduce across threads
    E_bins = sum(E_local)
    n_modes = sum(n_local)
    n_full = sum(nf_local)

    # convert to spectral density
    k_centers = zeros(Float64, n_bins)
    Ek = zeros(Float64, n_bins)
    for i in 1:n_bins
        k_centers[i] = sqrt(bin_edges[i] * bin_edges[i + 1])
        if n_full[i] > 0
            Ek[i] = E_bins[i] / (bin_edges[i + 1] - bin_edges[i])
        end
    end

    return k_centers, Ek, n_modes, bin_edges
end

# grid + FFT + bin, straight from a live u_ode
function measure_power_spectrum(ws::PSWorkspace, semi, u_ode; n_bins::Int = 2000,
                                ignore_mixed_parity::Bool = false)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, semi)

    _vel_to_grid_fast!(ws, u, mesh, equations, solver, cache)

    # in-place FFT using the pre-planned transform
    mul!(ws.fvx, ws.plan, ws.vx)
    mul!(ws.fvy, ws.plan, ws.vy)
    mul!(ws.fvz, ws.plan, ws.vz)

    return _shell_spectrum_fast!(ws, n_bins; ignore_mixed_parity = ignore_mixed_parity)
end

# same, from primitive vars loaded from disk
function measure_power_spectrum_prim(ws::PSWorkspace, prim_data, semi;
                                     n_bins::Int = 2000, ignore_mixed_parity::Bool = false)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    _vel_to_grid_from_prim!(ws, prim_data, mesh, solver, cache)

    mul!(ws.fvx, ws.plan, ws.vx)
    mul!(ws.fvy, ws.plan, ws.vy)
    mul!(ws.fvz, ws.plan, ws.vz)

    return _shell_spectrum_fast!(ws, n_bins; ignore_mixed_parity = ignore_mixed_parity)
end

# same, but with a PSGeometryCache instead of semi (skips trixi_include)
function measure_power_spectrum_prim(ws::PSWorkspace, prim_data, cache::PSGeometryCache;
                                     n_bins::Int = 2000, ignore_mixed_parity::Bool = false)
    _vel_to_grid_from_prim!(ws, prim_data, cache)

    mul!(ws.fvx, ws.plan, ws.vx)
    mul!(ws.fvy, ws.plan, ws.vy)
    mul!(ws.fvz, ws.plan, ws.vz)

    return _shell_spectrum_fast!(ws, n_bins; ignore_mixed_parity = ignore_mixed_parity)
end

# merges neighboring bins that have too few modes so the spectrum isn't noisy at low k

function rebin_spectrum(k_centers, Ek, n_modes, bin_edges;
                        min_modes_low_k::Int = 1,
                        min_modes_high_k::Int = 10,
                        k_split::Union{Nothing, Float64} = nothing)
    n = length(k_centers)

    if isnothing(k_split)
        k_valid = k_centers[n_modes .> 0]
        isempty(k_valid) && return k_centers, Ek, n_modes
        k_split = sqrt(first(k_valid) * last(k_valid))
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
