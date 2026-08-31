using Plots
using HDF5

include(joinpath(@__DIR__, "measure_PS.jl"))
include(joinpath(@__DIR__, "ps_geometry_cache.jl"))

# --- config ---

desktop_run_dir = joinpath(homedir(), "Desktop", "Runs")

output_dirs = [
    joinpath(homedir(), "Desktop", "Runs",
             "out_ns_subcell_v0.5_sol1.0_cells32_L1.0_pd3_mu5.0e-7_pr0.72_verns_subcell_lim_coeff")
]
N_avg = 16

const GEOCACHE_DIR = joinpath(@__DIR__, "geocache")

function parse_outdir(path)
    dname = basename(rstrip(path, ['/', '\\']))

    physics = occursin("euler", dname) ? "euler" : "navierstokes"
    if occursin("subcell", dname)
        method = "subcell"
    elseif occursin("pureDG", dname)
        method = "pureDG"
    else
        method = "shockcapturing"
    end
    elixir = "elixir_$(physics)_turbgen_$(method).jl"

    # core fields v/sol/cells, then whatever's left up to _ver{name_add}
    m = match(r"v([\d.]+)_sol([\d.]+)_cells(\d+)(.*)_ver(.*)$", dname)
    if m === nothing
        error("Couldn't parse dir name: $dname")
    end

    v_turb = parse(Float64, m[1])
    sw = parse(Float64, m[2])
    nc = parse(Int, m[3])
    middle = String(m[4])
    name_add = String(m[5])
    ref = round(Int, log2(nc))

    # optional params, may or may not be in the dirname
    pd_match = match(r"_pd(\d+)", middle)
    mu_match = match(r"_mu([\d.eE+-]+)", middle)
    pr_match = match(r"_pr([\d.]+)", middle)
    L_match = match(r"_L([\d.]+)", middle)

    pd = pd_match !== nothing ? parse(Int, pd_match[1]) : nothing
    mu_val = mu_match !== nothing ? parse(Float64, mu_match[1]) : nothing
    pr_val = pr_match !== nothing ? parse(Float64, pr_match[1]) : nothing
    L_val = L_match !== nothing ? parse(Float64, L_match[1]) : nothing

    return (; v_turb, sw, nc, ref, name_add, elixir, polydeg = pd, mu = mu_val, pr = pr_val,
            L = L_val)
end

function load_prim_data(fpath)
    h5open(fpath, "r") do f
        nvars = read(attributes(f)["n_vars"])
        return [read(f["variables_$v"]) for v in 1:nvars]
    end
end

# Process a single run directory
function process_run(output_dir::String; N_avg::Int = 12)
    println("\n", "="^72)
    println("Processing: $output_dir")
    println("="^72)

    p = parse_outdir(output_dir)
    println("v_turb=$(p.v_turb)  sw=$(p.sw)  cells=$(p.nc) (2^$(p.ref))  " *
            "L=$(p.L)  mu=$(p.mu)  pr=$(p.pr)  polydeg=$(p.polydeg)  elixir=$(p.elixir)")

    L = something(p.L, 1.0)
    polydeg = something(p.polydeg, 3)

    # t_turnover: reproduced from TurbGen.jl init_driving! (k_driv = GEOCACHE_K_DRIV)
    t_turnover = compute_t_turnover(L, p.v_turb)
    println("  t_turnover = $t_turnover  (L=$L, v_turb=$(p.v_turb), k_driv=$(GEOCACHE_K_DRIV))")

    elixir_path = joinpath(@__DIR__, "..", "examples", "DrivingGen", p.elixir)

    # cache hit skips trixi_include entirely
    geo = get_or_build_geometry(p.nc, polydeg, L;
                                cache_dir = GEOCACHE_DIR,
                                elixir_fn = () -> begin
                                    kwargs = Dict{Symbol, Any}(:tspan => (0.0, 0.0),
                                                               :turb_velocity => p.v_turb,
                                                               :cell => p.ref,
                                                               :name_addition => p.name_add,
                                                               :polydeg => polydeg)
                                    p.mu !== nothing && (kwargs[:mu] = p.mu)
                                    # variable name for the box size differs between
                                    # elixirs: navierstokes_{subcell,shockcapturing} use
                                    # domain_length, everything else uses domain_size
                                    if p.L !== nothing
                                        domain_key = p.elixir in
                                                     ("elixir_navierstokes_turbgen_subcell.jl",
                                                      "elixir_navierstokes_turbgen_shockcapturing.jl") ?
                                                     :domain_length : :domain_size
                                        kwargs[domain_key] = p.L
                                    end
                                    trixi_include(elixir_path; kwargs...)
                                end)

    _process_run_inner(output_dir, N_avg, geo, t_turnover, p)
end

function _process_run_inner(output_dir::String, N_avg::Int,
                            geo::PSGeometryCache, t_turnover::Float64, p)
    t_start = 3.0 * t_turnover

    all_files = sort(filter(f -> startswith(basename(f), "solution_") &&
                                endswith(f, ".h5"),
                            readdir(output_dir, join = true)))

    println("Found $(length(all_files)) total files in $output_dir, filtering t >= $t_start...")

    file_times = Tuple{String, Float64}[]
    for fpath in all_files
        t = h5open(fpath, "r") do f
            read(attributes(f)["time"])
        end
        t >= t_start && push!(file_times, (fpath, t))
    end
    sort!(file_times, by = x -> x[2])

    if length(file_times) < N_avg
        println("Warning: Only $(length(file_times)) files with t >= $t_start, need $N_avg. Skipping.")
        return
    end

    idx = round.(Int, range(1, length(file_times), length = N_avg))
    selected = file_times[idx]

    println("Selected $N_avg snapshots:")
    for (i, (f, t)) in enumerate(selected)
        println("  [$i] t=$(round(t, digits=4))  $(basename(f))")
    end

    ws = allocate_workspace(geo)
    println("Workspace allocated: N_grid=$(ws.N_grid), L=$(ws.L)")
    println("FFTW plan created with MEASURE flag (optimised)")
    println("Using $(Threads.nthreads()) Julia threads")

    Ek_sum_raw = nothing
    Ek_sum_rb = nothing
    k_ref_raw = nothing
    k_ref_rb = nothing
    cnt_raw = nothing
    cnt_rb = nothing

    for (i, (fpath, t)) in enumerate(selected)
        println("[$i/$N_avg] $(basename(fpath))  t=$(round(t, digits=4))")

        prim_data = load_prim_data(fpath)
        k_r, Ek_r, c_r, edges = measure_power_spectrum_prim(ws, prim_data, geo)
        k_b, Ek_b, c_b = rebin_spectrum(k_r, Ek_r, c_r, edges; L = ws.L,
                                        min_modes_low_k = 20,
                                        min_modes_high_k = 2000)

        if i == 1
            k_ref_raw, k_ref_rb = k_r, k_b
            Ek_sum_raw, Ek_sum_rb = copy(Ek_r), copy(Ek_b)
            cnt_raw, cnt_rb = c_r, c_b
        else
            Ek_sum_raw .+= Ek_r
            # rebinned bins have the same length for all snapshots (same N_grid, L)
            Ek_sum_rb .+= Ek_b
        end
    end

    Ek_avg_raw = Ek_sum_raw ./ N_avg
    Ek_avg_rb = Ek_sum_rb ./ N_avg

    ok = Ek_avg_raw .> 0
    ok_rb = Ek_avg_rb .> 0
    println("raw bins: $(sum(ok)), rebinned: $(sum(ok_rb))")

    println("\nMode counts per rebinned bin (k < 100):")
    for (i, (k, Ek, cnt)) in enumerate(zip(k_ref_rb, Ek_avg_rb, cnt_rb))
        k < 100 || continue
        println("  bin $i  k=$(round(k, digits=2))  modes=$cnt  E=$(round(Ek, sigdigits=4))")
    end

    # reference slopes, anchored to the measured spectrum at k ~ 110
    k_fit = k_ref_rb[ok_rb]
    Ek_fit = Ek_avg_rb[ok_rb]
    a_idx = argmin(abs.(k_fit .- 110.0))
    k0, E0 = k_fit[a_idx], Ek_fit[a_idx]

    kolm = E0 .* (k_fit ./ k0) .^ (-5 / 3)
    burg = E0 .* (k_fit ./ k0) .^ (-2)

    kd_lo = 1.0 * 2pi
    kd_hi = 2.0 * 2pi

    # tick every 100 in k, covering the plotted range
    k_tick_max = maximum(k_ref_rb[ok_rb])
    k_ticks = collect(100:100:k_tick_max)
    xticks = (k_ticks, string.(k_ticks))

    plt = plot(k_ref_rb[ok_rb], Ek_avg_rb[ok_rb];
               xscale = :log10, yscale = :log10,
               label = "rebinned (avg, N=$N_avg)",
               marker = :diamond, ms = 1.5, lw = 1.5, color = :crimson,
               xlabel = "k", ylabel = "E(k)", xticks = xticks,
               title = "Averaged Velocity Power Spectrum",
               legend = :bottomleft, size = (800, 600), dpi = 200)

    plot!(plt, k_fit, kolm; label = "k^{-5/3}", ls = :dash, lw = 1.5, color = :gray40)
    plot!(plt, k_fit, burg; label = "k^{-2}", ls = :dash, lw = 1.5, color = :black)

    vline!(plt, [kd_lo]; label = "k_drive_min", ls = :dot, lw = 1.5, color = :darkorange)
    vline!(plt, [kd_hi]; label = "k_drive_max", ls = :dot, lw = 1.5, color = :darkgreen)

    plots_dir = replace(output_dir, "out_" => "Analysis_")
    mkpath(plots_dir)

    out = joinpath(plots_dir, "power_spectrum_averaged.png")
    savefig(plt, out)
    println("\nSaved to $out")
    display(plt)

    h5_out = joinpath(plots_dir, "averaged_power_spectrum.h5")
    println("Saving data to $h5_out")

    h5open(h5_out, "w") do f
        # Metadata
        attributes(f)["v_turb"] = p.v_turb
        attributes(f)["sw"] = p.sw
        attributes(f)["cells"] = p.nc
        attributes(f)["ref"] = p.ref
        attributes(f)["name_addition"] = p.name_add
        attributes(f)["elixir"] = p.elixir
        attributes(f)["polydeg"] = p.polydeg === nothing ? -1 : p.polydeg
        attributes(f)["mu"] = p.mu === nothing ? 0.0 : p.mu
        attributes(f)["pr"] = p.pr === nothing ? 0.0 : p.pr
        attributes(f)["L"] = p.L === nothing ? 1.0 : p.L
        attributes(f)["N_avg"] = N_avg

        # Raw Data
        g_raw = create_group(f, "raw")
        g_raw["k"] = k_ref_raw
        g_raw["Ek_avg"] = Ek_avg_raw
        g_raw["modes"] = cnt_raw

        # Rebinned Data
        g_rb = create_group(f, "rebinned")
        g_rb["k"] = k_ref_rb
        g_rb["Ek_avg"] = Ek_avg_rb
        g_rb["modes"] = cnt_rb
    end
end

for target in output_dirs
    process_run(target; N_avg = N_avg)
end
