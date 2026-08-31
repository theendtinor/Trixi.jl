using HDF5

# save/load for PSGeometryCache, so we don't have to trixi_include just to get
# node coords and interpolation lookups every time we run an analysis script

const GEOCACHE_K_DRIV = 2.0

function _geocache_filename(N_cells::Int, polydeg::Int, L::Float64)
    return "geocache_cells$(N_cells)_pd$(polydeg)_L$(L).h5"
end

function save_geometry_cache(path::String, cache::PSGeometryCache)
    h5open(path, "w") do f
        attributes(f)["N_cells"] = cache.N_cells
        attributes(f)["N_grid"]  = cache.N_grid
        attributes(f)["L"]       = cache.L

        f["nodes"]  = cache.nodes
        f["bary_w"] = cache.bary_w
        f["lag1d"]  = cache.lag1d      # stored as (n_nodes × n_nodes) matrix
        f["lookup"] = cache.lookup     # stored as (N_cells × N_cells × N_cells)
    end
    println("  [geocache] Saved to $path")
    return nothing
end

function load_geometry_cache(path::String)
    h5open(path, "r") do f
        N_cells = read(attributes(f)["N_cells"])
        N_grid  = read(attributes(f)["N_grid"])
        L       = read(attributes(f)["L"])

        nodes  = read(f["nodes"])
        bary_w = read(f["bary_w"])
        lag1d  = read(f["lag1d"])
        lookup = read(f["lookup"])

        return PSGeometryCache(nodes, bary_w, lag1d, lookup, N_cells, N_grid, L)
    end
end

function build_geometry_from_semi(semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    L       = _box_length(mesh)
    N_cells = _cells_per_dim(mesh)
    k       = Trixi.polydeg(solver)
    N_grid  = N_cells * (k + 1)

    nodes  = collect(solver.basis.nodes)
    bary_w = collect(Trixi.barycentric_weights(nodes))

    n = length(nodes)
    dx_cell   = L / N_cells
    half_cell = dx_cell / 2

    # lag1d[s+1, i] = L_i(xi_s), xi_s = sub-cell s center mapped to [-1,1]
    lag1d = Matrix{Float64}(undef, n, n)
    for s in 0:(n - 1)
        xi = ((s + 0.5) / n * dx_cell - half_cell) / half_cell
        w  = Trixi.lagrange_interpolating_polynomials(xi, nodes, bary_w)
        for i in 1:n
            lag1d[s + 1, i] = w[i]
        end
    end

    lookup, _, _ = _build_cell_lookup(mesh, cache)

    return PSGeometryCache(nodes, bary_w, lag1d, lookup, N_cells, N_grid, L)
end

# loads the cache for (N_cells, polydeg, L) if we've built it before, otherwise
# runs elixir_fn() (expected to trixi_include and leave `semi` as a global),
# builds the geometry from that and saves it for next time
function get_or_build_geometry(N_cells::Int, polydeg::Int, L::Float64;
                               cache_dir::String,
                               elixir_fn)
    mkpath(cache_dir)
    fname = _geocache_filename(N_cells, polydeg, L)
    fpath = joinpath(cache_dir, fname)

    if isfile(fpath)
        println("  [geocache] cache hit, loading $fname")
        return load_geometry_cache(fpath)
    end

    println("  [geocache] cache miss, running trixi_include to build geometry...")
    elixir_fn()
    # trixi_include defines `semi` in Main in a newer world age than this function
    # was compiled in, so we reach it through invokelatest to avoid a world-age error.
    semi_global = Base.invokelatest(() -> Main.semi)
    geo = Base.invokelatest(build_geometry_from_semi, semi_global)
    save_geometry_cache(fpath, geo)
    return geo
end

# same t_decay formula as TurbGen.init_driving!, just without needing to load Trixi
function compute_t_turnover(L::Float64, v_turb::Float64;
                            k_driv::Float64 = GEOCACHE_K_DRIV)
    return (L / k_driv) / v_turb
end
