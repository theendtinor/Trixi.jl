using Plots
using DelimitedFiles
using Printf
using Statistics

# point at the out_ dir (or analysis.dat) of the isothermal run
target_runs = [
    joinpath(homedir(), "Desktop", "Runs",
             "out_euler_subcell_v3.2_sol1.0_cells32_L1.0_pd3_verit_dm0"),
    joinpath(homedir(), "Desktop", "Runs",
             "out_euler_subcell_v3.2_sol1.0_cells32_L1.0_pd3_verit_sf0_sup"),
    joinpath(homedir(), "Desktop", "Runs",
             "out_euler_subcell_v3.2_sol1.0_cells32_L1.0_pd4_verit_sf0_sup")
]

function cumulative_trapz(t, y)
    n = length(t)
    result = zeros(n)
    for i in 2:n
        result[i] = result[i - 1] + 0.5 * (y[i] + y[i - 1]) * (t[i] - t[i - 1])
    end
    return result
end

# makes 3 plots for an isothermal run: cumulative energy budget, Mach number vs time, and the instantaneous rates
function plot_energy_budget_isothermal(filename::String)
    if isfile(filename) && endswith(filename, "analysis.dat")
        data_file = filename
        run_dir = dirname(filename)
    elseif isdir(filename) && isfile(joinpath(filename, "analysis.dat"))
        data_file = joinpath(filename, "analysis.dat")
        run_dir = filename
    else
        println("Warning: analysis.dat not found for '$filename'. Skipping.")
        return
    end

    plots_dir = replace(run_dir, "out_" => "Analysis_")
    if !occursin("Analysis_", plots_dir) && !occursin("out_", run_dir)
        plots_dir = joinpath(run_dir, "Analysis_Plots")
    end
    mkpath(plots_dir)

    header_line = readline(data_file)
    column_names = split(replace(header_line, "#" => ""))
    data_raw = readdlm(data_file, skipstart = 1)

    col_idx(name) = findfirst(isequal(name), column_names)
    function get_col(name)
        idx = col_idx(name)
        (idx !== nothing && idx <= size(data_raw, 2)) ? data_raw[:, idx] : nothing
    end

    time = get_col("time")
    e_kinetic = get_col("e_kinetic")
    P_inject = get_col("P_inject")
    eps_diss = get_col("eps_diss")
    eps_cool = get_col("eps_cool")
    M_sq = get_col("mach_squared")

    required = Dict("time" => time,
                    "e_kinetic" => e_kinetic,
                    "P_inject" => P_inject,
                    "eps_diss" => eps_diss,
                    "eps_cool" => eps_cool,
                    "mach_squared" => M_sq)
    missing_cols = [k for (k, v) in required if v === nothing]
    if !isempty(missing_cols)
        mc_str = join(missing_cols, ", ")
        all_str = join(column_names, ", ")
        println("Warning: Missing columns $mc_str in $data_file. Skipping.")
        println("  Available: $all_str")
        return
    end

    eps_net = eps_diss .- eps_cool
    E_inject_cum = cumulative_trapz(time, P_inject)
    E_diss_cum = cumulative_trapz(time, eps_diss)
    E_cool_cum = cumulative_trapz(time, eps_cool)
    E_net_cum = cumulative_trapz(time, eps_net)
    ΔE_kinetic = e_kinetic .- e_kinetic[1]
    mach_rms = sqrt.(abs.(M_sq))   # rms Mach 

    run_name = basename(run_dir)
    short_label = replace(run_name, r"^out_" => "")

    gr(size = (950, 560))

    # plot 1: cumulative budget. dE_kin + int(eps_net) should be close to int(P_inject)
    p_budget = plot(time, E_inject_cum,
                    label = "∫P_inject dt  (total injected)",
                    xlabel = "Time",
                    ylabel = "Cumulative energy",
                    linewidth = 2.5,
                    color = :dodgerblue,
                    legend = :topleft,
                    title = "Cumulative energy budget\n$short_label",
                    grid = true,
                    gridalpha = 0.3,
                    framestyle = :box)

    plot!(p_budget, time, E_net_cum,
          label = "∫ε_net dt  (net extracted: diss − cool)",
          linewidth = 2.0,
          color = :purple,
          linestyle = :dot)

    plot!(p_budget, time, ΔE_kinetic,
          label = "ΔE_kinetic  (stored kinetic energy)",
          linewidth = 2.0,
          color = :forestgreen,
          linestyle = :dash)

    plot!(p_budget, time, ΔE_kinetic .+ E_net_cum,
          label = "ΔE_kin + ∫ε_net  (should ≈ ∫P_inject)",
          linewidth = 1.8,
          color = :black,
          linestyle = :dot,
          alpha = 0.75)

    savefig(p_budget, joinpath(plots_dir, "energy_budget_isothermal.png"))
    println("Saved: $(joinpath(plots_dir, "energy_budget_isothermal.png"))")

    # plot 2: Mach number vs time
    p_mach = plot(time, mach_rms,
                  label = "M_rms = √⟨M²⟩",
                  xlabel = "Time",
                  ylabel = "rms Mach number",
                  linewidth = 2.5,
                  color = :teal,
                  legend = :bottomright,
                  title = "Mach-number evolution\n$short_label",
                  grid = true,
                  gridalpha = 0.3,
                  framestyle = :box)

    M_mean = mean(mach_rms)
    hline!(p_mach, [M_mean],
           label = @sprintf("⟨M_rms⟩ = %.3f", M_mean),
           linewidth = 1.8,
           color = :teal,
           linestyle = :dash,
           alpha = 0.7)

    savefig(p_mach, joinpath(plots_dir, "mach_evolution_isothermal.png"))
    println("Saved: $(joinpath(plots_dir, "mach_evolution_isothermal.png"))")

    # plot 3: cumulative injected/extracted (left axis) plus Mach (right axis)
    p_rates = plot(time, E_inject_cum,
                   label = "∫P_inject dt  (cumulative injected)",
                   xlabel = "Time",
                   ylabel = "Cumulative energy",
                   linewidth = 2.0,
                   color = :dodgerblue,
                   linestyle = :dash,
                   legend = :topleft,
                   title = "Cumulative injected vs extracted · Mach\n$short_label",
                   grid = true,
                   gridalpha = 0.3,
                   framestyle = :box)

    plot!(p_rates, time, E_net_cum,
          label = "∫ε_net dt  (cumulative extracted)",
          linewidth = 2.0,
          color = :purple,
          linestyle = :dot)

    p_mach_ax = twinx(p_rates)
    plot!(p_mach_ax, time, mach_rms,
          label = "M_rms = √⟨M²⟩",
          ylabel = "Mach number",
          linewidth = 2.0,
          color = :teal,
          legend = :bottomright)

    savefig(p_rates, joinpath(plots_dir, "rates_isothermal.png"))
    println("Saved: $(joinpath(plots_dir, "rates_isothermal.png"))")

    mean_Pinj = mean(P_inject)
    mean_diss = mean(eps_diss)
    mean_cool = mean(eps_cool)
    mean_net = mean(eps_net)
    cool_frac = mean_cool / max(mean_diss, 1e-30)
    residual = abs(mean_Pinj - mean_net) / max(abs(mean_Pinj), 1e-30)
    mean_mach = mean(mach_rms)

    println()
    println("═"^64)
    println("  Energy budget — entire run")
    println("  Run: $short_label")
    println("─"^64)
    @printf("  ⟨P_inject⟩            = %+.4e\n", mean_Pinj)
    @printf("  ⟨ε_diss⟩              = %+.4e\n", mean_diss)
    @printf("  ⟨ε_cool⟩              = %+.4e\n", mean_cool)
    @printf("  ⟨ε_net⟩               = %+.4e\n", mean_net)
    @printf("  cool / diss           =  %.4f  %s\n", cool_frac,
            cool_frac < 0.1 ? "✓  (< 10 %, clip negligible)" :
            cool_frac < 0.5 ? "⚠  (10–50 %, worth checking)" :
            "~  (≈ 1 expected for isothermal reset)")
    @printf("  |P_inject − ε_net| / P_inject = %.4f\n", residual)
    @printf("  ⟨M_rms⟩               =  %.4f\n", mean_mach)
    println("═"^64)
    println()
end

if length(ARGS) > 0
    for arg in ARGS
        plot_energy_budget_isothermal(arg)
    end
else
    println("No arguments provided, processing $(length(target_runs)) default target(s):")
    for (i, t) in enumerate(target_runs)
        println("  [$i] $t")
    end
    for t in target_runs
        plot_energy_budget_isothermal(t)
    end
end
