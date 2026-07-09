using Plots
using DelimitedFiles

target_runs = [
    joinpath(homedir(), "Desktop", "Runs",
             "out_ns_subcell_v0.5_sol1.0_cells32_L1.0_pd2_mu5.0e-7_pr0.72_ver")
]

# integral of y(t), same length as t, starts at 0
function cumulative_trapz(t, y)
    n = length(t)
    result = zeros(n)
    for i in 2:n
        result[i] = result[i - 1] + 0.5 * (y[i] + y[i - 1]) * (t[i] - t[i - 1])
    end
    return result
end

function plot_energy_balance(filename::String)
    if isfile(filename) && endswith(filename, "analysis.dat")
        data_file = filename
        run_dir = dirname(filename)
    elseif isdir(filename) && isfile(joinpath(filename, "analysis.dat"))
        data_file = joinpath(filename, "analysis.dat")
        run_dir = filename
    else
        println("Warning: Data file not found for $filename. Skipping.")
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
        if idx !== nothing && idx <= size(data_raw, 2)
            return data_raw[:, idx]
        else
            return nothing
        end
    end

    time = get_col("time")
    e_kinetic = get_col("e_kinetic")
    P_inject = get_col("P_inject")
    M_sq = get_col("mach_squared")

    if any(x -> x === nothing, [time, e_kinetic, P_inject, M_sq])
        missing_cols = String[]
        time === nothing && push!(missing_cols, "time")
        e_kinetic === nothing && push!(missing_cols, "e_kinetic")
        P_inject === nothing && push!(missing_cols, "P_inject")
        M_sq === nothing && push!(missing_cols, "mach_squared")
        println("Warning: Missing columns $(join(missing_cols, ", ")) in $data_file. Skipping.")
        println("  Available columns: $(join(column_names, ", "))")
        return
    end

    E_inject_cum = cumulative_trapz(time, P_inject)
    E_kinetic = e_kinetic .- e_kinetic[1]   # relative to t=0 (starts at rest)

    # whatever got injected but isn't showing up as kinetic energy must have dissipated
    E_dissipated_cum = E_inject_cum .- E_kinetic

    mach_number = sqrt.(abs.(M_sq))

    run_name = basename(run_dir)
    short_label = replace(run_name, r"^out_" => "")

    # left axis energy, right axis Mach number
    gr(size = (800, 500))

    p = plot(time, E_inject_cum,
             label = "Cumulative injected energy",
             xlabel = "Time",
             ylabel = "Cumulative energy",
             linewidth = 2.5,
             color = :dodgerblue,
             legend = :topleft,
             title = "Energy balance – $short_label",
             grid = true,
             gridalpha = 0.3,
             framestyle = :box)

    plot!(p, time, E_dissipated_cum,
          label = "Cumulative dissipated energy",
          linewidth = 2.5,
          color = :crimson,
          linestyle = :dash)

    p2 = twinx(p)
    plot!(p2, time, mach_number,
          label = "Mach number",
          ylabel = "Mach number",
          linewidth = 2.0,
          color = :forestgreen,
          linestyle = :dot,
          legend = :right)

    savefig(p, joinpath(plots_dir, "energy_balance.png"))
    println("Saved energy balance plot to $(joinpath(plots_dir, "energy_balance.png"))")

    # instantaneous rates too: eps = P_inject - dE_kinetic/dt
    dEk_dt = zeros(length(time))
    for i in 2:(length(time) - 1)
        dEk_dt[i] = (e_kinetic[i + 1] - e_kinetic[i - 1]) / (time[i + 1] - time[i - 1])
    end
    dEk_dt[1] = (e_kinetic[2] - e_kinetic[1]) / (time[2] - time[1])
    dEk_dt[end] = (e_kinetic[end] - e_kinetic[end - 1]) / (time[end] - time[end - 1])

    dissipation_rate = P_inject .- dEk_dt

    p_rates = plot(time, P_inject,
                   label = "Injection rate P_inject",
                   xlabel = "Time",
                   ylabel = "Power (energy / time)",
                   linewidth = 2.0,
                   color = :dodgerblue,
                   legend = :topright,
                   title = "Injection & dissipation rates – $short_label",
                   grid = true,
                   gridalpha = 0.3,
                   framestyle = :box)
    plot!(p_rates, time, dissipation_rate,
          label = "Dissipation rate ε",
          linewidth = 2.0,
          color = :crimson,
          linestyle = :dash)
    plot!(p_rates, time, dEk_dt,
          label = "dE_kinetic/dt",
          linewidth = 1.5,
          color = :gray,
          linestyle = :dot,
          alpha = 0.7)

    savefig(p_rates, joinpath(plots_dir, "energy_rates.png"))
    println("Saved energy rates plot to $(joinpath(plots_dir, "energy_rates.png"))")
end

if length(ARGS) > 0
    for arg in ARGS
        plot_energy_balance(arg)
    end
else
    println("No arguments provided, processing $(length(target_runs)) default target(s):")
    for (i, target) in enumerate(target_runs)
        println("  [$i] $target")
    end
    for target in target_runs
        plot_energy_balance(target)
    end
end
