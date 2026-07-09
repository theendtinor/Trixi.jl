using Plots
using DelimitedFiles

target_runs = [
    joinpath(homedir(), "Desktop", "Runs",
             "out_euler_subcell_v3.2_sol1.0_cells32_L1.0_pd3_verit_dm0"),
    joinpath(homedir(), "Desktop", "Runs",
             "out_euler_subcell_v3.2_sol1.0_cells32_L1.0_pd3_verit_sf0_sup"),
    joinpath(homedir(), "Desktop", "Runs",
             "out_euler_subcell_v3.2_sol1.0_cells32_L1.0_pd4_verit_sf0_sup")
]

function make_plots_for_run(filename::String)
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

    get_col(name) = begin
        idx = col_idx(name)
        if idx !== nothing && idx <= size(data_raw, 2)
            return data_raw[:, idx]
        else
            return nothing
        end
    end

    time = get_col("time")
    dt = get_col("dt")
    e_kinetic = get_col("e_kinetic")
    e_internal = get_col("e_internal")
    rho_v1_squared = get_col("rho_v1_squared")
    rho_v2_squared = get_col("rho_v2_squared")
    rho_v3_squared = get_col("rho_v3_squared")
    density = get_col("density")
    pressure = get_col("pressure")
    mach_squared = get_col("mach_squared")
    entropy = get_col("entropy")

    default(linewidth = 2, markersize = 4, legend = :best)

    plots_list = []

    # Plot 1: Time step size (skip first entry, log scale)
    if dt !== nothing && length(dt) > 1
        p1 = plot(time[2:(end - 1)], dt[2:(end - 1)], xlabel = "Time", ylabel = "dt",
                  title = "Time Step Size",
                  marker = :circle, yscale = :log10)
        push!(plots_list, p1)
        savefig(p1, joinpath(plots_dir, "dt.png"))
    end

    # Plot 2: Kinetic energy
    if e_kinetic !== nothing
        p2 = plot(time, e_kinetic, xlabel = "Time", ylabel = "E_kinetic",
                  title = "Kinetic Energy", marker = :circle, color = :red)
        push!(plots_list, p2)
        savefig(p2, joinpath(plots_dir, "kinetic_energy.png"))
    end

    # Plot 3: Velocity components
    if rho_v1_squared !== nothing && rho_v2_squared !== nothing &&
       rho_v3_squared !== nothing
        p3 = plot(time, rho_v1_squared, label = "ρv₁²", xlabel = "Time", ylabel = "ρvᵢ²",
                  title = "Velocity Components", marker = :circle)
        plot!(p3, time, rho_v2_squared, label = "ρv₂²", marker = :square)
        plot!(p3, time, rho_v3_squared, label = "ρv₃²", marker = :diamond)
        push!(plots_list, p3)
        savefig(p3, joinpath(plots_dir, "velocity_components.png"))
    end

    # Plot 4: Mach number
    if mach_squared !== nothing
        p4 = plot(time, sqrt.(abs.(mach_squared)), xlabel = "Time", ylabel = "Mach Number",
                  title = "Local Mach Number", marker = :circle, color = :purple)
        push!(plots_list, p4)
        savefig(p4, joinpath(plots_dir, "mach_number.png"))
    end

    # Plot 5: Density (skip first entry)
    if density !== nothing && length(density) > 1
        p5 = plot(time[2:end], density[2:end] .- 1, xlabel = "Time", ylabel = "ρ - 1",
                  title = "Density Deviation from 1", marker = :circle, color = :green)
        push!(plots_list, p5)
        savefig(p5, joinpath(plots_dir, "density.png"))
    end

    # Plot 6: Entropy
    if entropy !== nothing
        p6 = plot(time, entropy, xlabel = "Time", ylabel = "Entropy", title = "Entropy",
                  marker = :circle, color = :orange)
        push!(plots_list, p6)
        savefig(p6, joinpath(plots_dir, "entropy.png"))
    end

    # Plot 7: Internal Energy
    if e_internal !== nothing
        p7 = plot(time, e_internal, xlabel = "Time", ylabel = "E_internal",
                  title = "Internal Energy", marker = :circle, color = :blue)
        push!(plots_list, p7)
        savefig(p7, joinpath(plots_dir, "internal_energy.png"))
    end

    # Plot 8: Pressure
    if pressure !== nothing
        p8 = plot(time, pressure, xlabel = "Time", ylabel = "Pressure", title = "Pressure",
                  marker = :circle, color = :cyan)
        push!(plots_list, p8)
        savefig(p8, joinpath(plots_dir, "pressure.png"))
    end

    if length(plots_list) > 0
        n_plots = length(plots_list)
        rows = ceil(Int, n_plots / 2)
        combined = plot(plots_list..., layout = (rows, 2), size = (1000, 300 * rows))
        savefig(combined, joinpath(plots_dir, "simulation_plots.png"))
    end

    println("Plots saved to $plots_dir")
end

if length(ARGS) > 0
    for arg in ARGS
        make_plots_for_run(arg)
    end
else
    println("No arguments provided, processing $(length(target_runs)) default target(s):")
    for (i, target) in enumerate(target_runs)
        println("  [$i] $target")
    end
    for target in target_runs
        make_plots_for_run(target)
    end
end
