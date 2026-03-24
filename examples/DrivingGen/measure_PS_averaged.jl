using Plots
using HDF5

# directory with the solution files
output_dir = joinpath(@__DIR__, "..", "..",
                      "out_turbgen_subcell_v1.0_sol1.0_cells32_ver2303")

N_avg = 16

# parse turb velocity, sol weight, cell count etc. from the directory name
# expected: out_turbgen_[subcell_]v{vel}_sol{sw}_cells{nc}_ver{na}
function parse_outdir(path)
    dname = basename(rstrip(path, ['/', '\\']))

    elixir = occursin("subcell", dname) ?
             "elixir_euler_turbgen_subcell.jl" :
             "elixir_euler_turbgen_shockcapturing.jl"

    m = match(r"v([\d.]+)_sol([\d.]+)_cells(\d+)_ver(.+)$", dname)
    m === nothing && error("Couldn't parse dir name: $dname")

    v_turb = parse(Float64, m[1])
    sw = parse(Float64, m[2])
    nc = parse(Int, m[3])
    name_add = String(m[4])

    ref = round(Int, log2(nc))

    return (; v_turb, sw, nc, ref, name_add, elixir)
end

p = parse_outdir(output_dir)
println("v_turb=$(p.v_turb)  sw=$(p.sw)  cells=$(p.nc) (2^$(p.ref))  elixir=$(p.elixir)")

# set up semi without running anything
trixi_include(joinpath(@__DIR__, p.elixir);
              tspan = (0.0, 0.0),
              turb_velocity = p.v_turb,
              cell = p.ref,
              name_addition = p.name_add)

t_start = 3.0 * t_turnover

# grab all solution files and keep only those past 3*t_turnover
all_files = sort(filter(f -> startswith(basename(f), "solution_") &&
                            endswith(f, ".h5"),
                        readdir(output_dir, join = true)))

println("Found $(length(all_files)) total files, filtering t >= $t_start...")

file_times = Tuple{String, Float64}[]
for fpath in all_files
    t = h5open(fpath, "r") do f
        read(attributes(f)["time"])
    end
    t >= t_start && push!(file_times, (fpath, t))
end
sort!(file_times, by = x -> x[2])

length(file_times) < N_avg &&
    error("Only $(length(file_times)) files with t >= $t_start, need $N_avg")

# pick N_avg evenly spaced snapshots
idx = round.(Int, range(1, length(file_times), length = N_avg))
selected = file_times[idx]

println("Selected $N_avg snapshots:")
for (i, (f, t)) in enumerate(selected)
    println("  [$i] t=$(round(t, digits=4))  $(basename(f))")
end

# load prim vars from h5 and convert back to cons
function load_sol(semi, fpath)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u_ode = Trixi.allocate_coefficients(mesh, equations, solver, cache)
    u = Trixi.wrap_array(u_ode, semi)

    h5open(fpath, "r") do f
        nvars = read(attributes(f)["n_vars"])
        raw = [read(f["variables_$v"]) for v in 1:nvars]
        nn = Trixi.nnodes(solver)
        ne = Trixi.nelements(solver, cache)

        for elem in 1:ne
            for k in 1:nn, j in 1:nn, i in 1:nn
                idx = i + (j - 1) * nn + (k - 1) * nn^2 + (elem - 1) * nn^3
                pv = SVector(raw[1][idx], raw[2][idx], raw[3][idx],
                             raw[4][idx], raw[5][idx])
                cv = Trixi.prim2cons(pv, equations)
                for v in 1:nvars
                    u[v, i, j, k, elem] = cv[v]
                end
            end
        end
    end
    return u_ode
end

include(joinpath(@__DIR__, "measure_PS.jl"))

# accumulate spectra across snapshots
Ek_sum_raw = nothing
Ek_sum_rb = nothing
k_ref_raw = nothing
k_ref_rb = nothing
cnt_raw = nothing
cnt_rb = nothing

for (i, (fpath, t)) in enumerate(selected)
    global Ek_sum_raw, Ek_sum_rb, k_ref_raw, k_ref_rb, cnt_raw, cnt_rb

    println("[$i/$N_avg] $(basename(fpath))  t=$(round(t, digits=4))")
    u_i = load_sol(semi, fpath)

    k_r, Ek_r, c_r, edges = measure_power_spectrum(semi, u_i)
    k_b, Ek_b, c_b = rebin_spectrum(k_r, Ek_r, c_r, edges;
                                    min_modes_low_k = 35, min_modes_high_k = 4000)

    if i == 1
        k_ref_raw, k_ref_rb = k_r, k_b
        Ek_sum_raw, Ek_sum_rb = copy(Ek_r), copy(Ek_b)
        cnt_raw, cnt_rb = c_r, c_b
    else
        Ek_sum_raw .+= Ek_r
        Ek_sum_rb .+= Ek_b
    end
end

Ek_avg_raw = Ek_sum_raw ./ N_avg
Ek_avg_rb = Ek_sum_rb ./ N_avg

ok = Ek_avg_raw .> 0
ok_rb = Ek_avg_rb .> 0
println("raw bins: $(sum(ok)), rebinned: $(sum(ok_rb))")

# check mode counts for low-k bins
println("\nMode counts per rebinned bin (k < 100):")
for (i, (k, Ek, cnt)) in enumerate(zip(k_ref_rb, Ek_avg_rb, cnt_rb))
    k < 100 || continue
    println("  bin $i  k=$(round(k, digits=2))  modes=$cnt  E=$(round(Ek, sigdigits=4))")
end

# reference lines , anchor in ca. middle
k_fit = k_ref_rb[ok_rb]
Ek_fit = Ek_avg_rb[ok_rb]
a_idx = argmin(abs.(k_fit .- 65.0))
k0, E0 = k_fit[a_idx], Ek_fit[a_idx]

kolm = E0 .* (k_fit ./ k0) .^ (-5 / 3)
burg = E0 .* (k_fit ./ k0) .^ (-2)

# driving range
kd_lo = 1.0 * 2 * pi
kd_hi = 2.0 * 2 * pi

plt = plot(k_ref_rb[ok_rb], Ek_avg_rb[ok_rb];
           xscale = :log10, yscale = :log10,
           label = "rebinned (avg, N=$N_avg)",
           marker = :diamond, ms = 1.5, lw = 1.5, color = :crimson,
           xlabel = "k", ylabel = "E(k)",
           title = "Averaged Velocity Power Spectrum",
           legend = :bottomleft, size = (800, 600), dpi = 200)

plot!(plt, k_fit, kolm; label = "k^{-5/3}", ls = :dash, lw = 1.5, color = :gray40)
plot!(plt, k_fit, burg; label = "k^{-2}", ls = :dash, lw = 1.5, color = :black)

vline!(plt, [kd_lo]; label = "k_drive_min", ls = :dot, lw = 1.5, color = :darkorange)
vline!(plt, [kd_hi]; label = "k_drive_max", ls = :dot, lw = 1.5, color = :darkgreen)

run_name = replace(basename(rstrip(output_dir, ['/', '\\'])), "out_turbgen_" => "")
plots_dir = joinpath(@__DIR__, "..", "..", "plots", run_name)
mkpath(plots_dir)

out = joinpath(plots_dir, "power_spectrum_averaged.png")
savefig(plt, out)
println("Saved to $out")
display(plt)