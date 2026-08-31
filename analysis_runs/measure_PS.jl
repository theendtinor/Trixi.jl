using FFTW
using LinearAlgebra: mul!
using Base.Threads

# Spectrum convention:
#
#   E_v(k) = (L / 2pi)^3 |v_hat(k)|^2      (single Fourier mode)
#   E(k)   = 4 pi k^2 <E_v(k)>             (shell-averaged spectrum)


# number of k-space lattice modes per unit |k| in a thin shell at k
_modes_per_dk(k, L) = 4 * pi * k^2 * (L / (2 * pi))^3

# E(k) from the power summed over a shell and the number of modes it contains
function _spectral_density(k, power_sum, n_modes, L)
    n_modes == 0 && return 0.0
    return _modes_per_dk(k, L) * power_sum / n_modes
end

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

# workspace holding the velocity grids, FFT plan and the Fourier arrays

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

    plan = plan_rfft(vx; flags = FFTW.MEASURE)

    n_kx = div(N_grid, 2) + 1
    fvx = zeros(ComplexF64, n_kx, N_grid, N_grid)
    fvy = zeros(ComplexF64, n_kx, N_grid, N_grid)
    fvz = zeros(ComplexF64, n_kx, N_grid, N_grid)

    return PSWorkspace(vx, vy, vz, fvx, fvy, fvz, plan, N_grid, L)
end

# caches mesh/solver geometry (nodes, lookup table, lag1d), only depends on (N_cells, polydeg, L).
# save/load logic lives in ps_geometry_cache.jl

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

# same thing but from a cache
function allocate_workspace(cache::PSGeometryCache)
    return PSWorkspace(cache.N_grid, cache.L)
end

# Tensor-product Lagrange interpolation of one DG cell onto its nn^3 uniform
# sub-cell points. The interpolation is
#
#   out(sx, sy, sz) = sum_{i,j,k} lag[sx,i] lag[sy,j] lag[sz,k] * u[i,j,k],
#
# which we evaluate as three successive 1D contractions (sum factorisation), one
# per direction, instead of the full triple sum. A and B are scratch buffers for the two intermediate stages.
function _interpolate_cell!(grid, u, base, cx, cy, cz, nn, lag, A, B)
    # pass 1: contract i  ->  A[sx, j, k]
    @inbounds for k in 1:nn, j in 1:nn, sx in 1:nn
        acc = 0.0
        for i in 1:nn
            acc += lag[sx, i] * u[base + i + (j - 1) * nn + (k - 1) * nn^2]
        end
        A[sx, j, k] = acc
    end

    # pass 2: contract j  ->  B[sx, sy, k]
    @inbounds for k in 1:nn, sy in 1:nn, sx in 1:nn
        acc = 0.0
        for j in 1:nn
            acc += lag[sy, j] * A[sx, j, k]
        end
        B[sx, sy, k] = acc
    end

    # pass 3: contract k and scatter into the uniform grid. cx, cy, cz are the
    # 0-based cell indices, so this cell fills grid[cx*nn+1 : cx*nn+nn, ...].
    @inbounds for sz in 1:nn, sy in 1:nn, sx in 1:nn
        acc = 0.0
        for k in 1:nn
            acc += lag[sz, k] * B[sx, sy, k]
        end
        grid[cx * nn + sx, cy * nn + sy, cz * nn + sz] = acc
    end

    return nothing
end

# Interpolate the primitive velocities from the GLL nodes onto the uniform grid
# the FFT runs on. The three velocity components are handled independently, one _interpolate_cell! call each.

function _vel_to_grid_from_prim!(ws::PSWorkspace, prim_data,
                                 cache::PSGeometryCache)
    vx, vy, vz = ws.vx, ws.vy, ws.vz
    fill!(vx, 0.0)
    fill!(vy, 0.0)
    fill!(vz, 0.0)

    nn = length(cache.nodes)          # nodes per dim = polydeg + 1
    N_cells = cache.N_cells
    lookup = cache.lookup
    lag = cache.lag1d               

    grids = (vx, vy, vz)
    velocities = (prim_data[2], prim_data[3], prim_data[4])

    # one scratch pair per thread, reused across all cells that thread handles
    nt = Threads.maxthreadid()
    scratch_A = [Array{Float64}(undef, nn, nn, nn) for _ in 1:nt]
    scratch_B = [Array{Float64}(undef, nn, nn, nn) for _ in 1:nt]

    @threads for cell in 1:N_cells^3
        cz, r = divrem(cell - 1, N_cells^2)
        cy, cx = divrem(r, N_cells)

        eid = lookup[cx + 1, cy + 1, cz + 1]
        base = (eid - 1) * nn^3

        A = scratch_A[threadid()]
        B = scratch_B[threadid()]

        for c in 1:3
            _interpolate_cell!(grids[c], velocities[c], base, cx, cy, cz, nn, lag,
                               A, B)
        end
    end

    return nothing
end

# bins |k| into shells and sums power per shell, threaded over kz slices

function _shell_spectrum_fast!(ws::PSWorkspace, n_bins::Int;
                               k_cutoff::Symbol = :nyquist,
                               ignore_mixed_parity::Bool = false)
    (; fvx, fvy, fvz, N_grid, L) = ws

    dk = 2pi / L
    k_nyq = pi * N_grid / L
    # shells past the Nyquist wavenumber are only partially covered by the grid and
    # therefore under-report the power; :cube keeps them out to the corner of the
    # k-cube, which is what an exact Parseval check needs
    k_max = k_cutoff === :nyquist ? k_nyq :
            k_cutoff === :cube ? sqrt(3) * k_nyq :
            error("k_cutoff must be :nyquist or :cube")

    # log-spaced bins
    log_lo = log10(0.5 * dk)
    log_hi = log10(k_max)
    bin_edges = 10.0 .^ range(log_lo, log_hi; length = n_bins + 1)

    inv_dk_log = n_bins / (log_hi - log_lo)

    kx_vals = collect(0:div(N_grid, 2))
    ky_vals = fftfreq(N_grid, N_grid)
    kz_vals = fftfreq(N_grid, N_grid)

    # per-mode power |v_hat|^2 from FFTW's unnormalised output
    norm = 1.0 / N_grid^6
    n_kx = div(N_grid, 2) + 1

    # thread-local accumulators 
    nt = Threads.maxthreadid()
    E_local = [zeros(Float64, n_bins) for _ in 1:nt]
    modes_local = [zeros(Int, n_bins) for _ in 1:nt]

    @threads for ikz in 1:N_grid
        tid = threadid()
        E_t = E_local[tid]
        modes_t = modes_local[tid]
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
                    modes_t[b] += mult
                end
            end
        end
    end

    # reduce across threads
    E_bins = sum(E_local)
    n_modes = sum(modes_local)

    # convert to the spectral density E(k) = 4 pi k^2 <E_v(k)>
    k_centers = zeros(Float64, n_bins)
    Ek = zeros(Float64, n_bins)
    for i in 1:n_bins
        k_centers[i] = sqrt(bin_edges[i] * bin_edges[i + 1])
        Ek[i] = _spectral_density(k_centers[i], E_bins[i], n_modes[i], L)
    end

    # n_modes counts the modes the sums actually contain, i.e. both halves of every
    # conjugate pair 
    return k_centers, Ek, n_modes, bin_edges
end

# grid + FFT + bin
function measure_power_spectrum_prim(ws::PSWorkspace, prim_data, cache::PSGeometryCache;
                                     n_bins::Int = 2000, k_cutoff::Symbol = :nyquist,
                                     ignore_mixed_parity::Bool = false)
   
   _vel_to_grid_from_prim!(ws, prim_data, cache)

    mul!(ws.fvx, ws.plan, ws.vx)
    mul!(ws.fvy, ws.plan, ws.vy)
    mul!(ws.fvz, ws.plan, ws.vz)

    return _shell_spectrum_fast!(ws, n_bins; k_cutoff = k_cutoff,
                                 ignore_mixed_parity = ignore_mixed_parity)
end

# merges neighboring bins that have too few modes so the spectrum isn't noisy at low k

function rebin_spectrum(k_centers, Ek, n_modes, bin_edges; L::Float64,
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
        if n_modes[i] == 0
            i += 1
            continue
        end

        min_m = k_centers[i] < k_split ? min_modes_low_k : min_modes_high_k

        # merge the summed shell power and the mode counts, not the E values, so the
        # merged bin obeys the same E(k) = 4 pi k^2 <E_v> definition exactly
        power_tot = 0.0
        k_weighted = 0.0
        m_tot = 0
        j = i

        while j <= n && m_tot < min_m
            if n_modes[j] > 0
                power_tot += Ek[j] * n_modes[j] / _modes_per_dk(k_centers[j], L)
                k_weighted += n_modes[j] * k_centers[j]
                m_tot += n_modes[j]
            end
            j += 1
        end

        if m_tot > 0
            # represent the merged bin by the mean |k| of the modes it holds.
            k_new = k_weighted / m_tot
            push!(k_out, k_new)
            push!(E_out, _spectral_density(k_new, power_tot, m_tot, L))
            push!(cnt, m_tot)
        end

        i = j
    end

    return k_out, E_out, cnt
end
