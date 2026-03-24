using OrdinaryDiffEq: DiscreteCallback
using Plots
using Printf

include(joinpath(@__DIR__, "TurbGen.jl"))
using .TurbGen

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

function initial_condition_uniform(x, t, equations::CompressibleEulerEquations3D)
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    p = 1.0 / equations.gamma
    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_uniform

###############################################################################
# turbulence generator (Ornstein-Uhlenbeck forcing)

turb_velocity = 1.0       # target Mach number
turb_sol_weight = 1.0     # solenoidal weight (1.0 = purely solenoidal)
cell = 5
name_addition = "lowDR"
num_cells = 2^cell
polydeg = 3

run_label = "v$(turb_velocity)_sol$(turb_sol_weight)_cells$(num_cells)_ver$(name_addition)"
analysis_outdir = "Analysis_turbgen_subcell_$(run_label)"
solution_outdir = "out_turbgen_subcell_$(run_label)"
slices_outdir = "Slices_turbgen_subcell_$(run_label)"

turb_gen = TurbGen.TurbGenGenerator(seed = 42)
TurbGen.init_driving!(turb_gen,
                      Dict{String, Any}("L" => [1.0, 1.0, 1.0],
                                        "velocity" => turb_velocity,
                                        "k_driv" => 1,
                                        "k_min" => 0.5,
                                        "k_max" => 1.5,
                                        "spectral_slope" => -5.0 / 3.0,
                                        "sol_weight" => turb_sol_weight,
                                        "random_seed" => 42,
                                        "nsteps_per_t_turb" => 10,
                                        "ampl_factor" => [1.0, 1.0, 1.0]))
t_turnover = turb_gen.t_decay

###############################################################################
# source terms

mutable struct TurbulentForcing{Gen <: TurbGen.TurbGenGenerator}
    generator::Gen
    accel_cache::Array{SVector{3, Float64}, 4}
    cached_step::Int
end

function TurbulentForcing(generator)
    TurbulentForcing(generator, Array{SVector{3, Float64}, 4}(undef, 0, 0, 0, 0), -1)
end

# source term (point-wise fallback)
function (source::TurbulentForcing)(u, x, t, equations::CompressibleEulerEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u
    v1, v2, v3 = rho_v1 / rho, rho_v2 / rho, rho_v3 / rho

    accel = TurbGen.get_turb_vector(source.generator, SVector(x[1], x[2], x[3]))
    ax, ay, az = accel[1], accel[2], accel[3]

    return SVector(zero(eltype(u)),
                   rho * ax, rho * ay, rho * az,
                   rho * (v1 * ax + v2 * ay + v3 * az))
end

function Trixi.calc_sources!(du, u, t, source_terms::TurbulentForcing,
                             equations::CompressibleEulerEquations3D, dg::Trixi.DG, cache)
    node_coordinates = cache.elements.node_coordinates
    nn = size(node_coordinates, 2)
    ne = Trixi.nelements(dg, cache)

    # Resize cache on first call
    if size(source_terms.accel_cache) != (nn, nn, nn, ne)
        source_terms.accel_cache = Array{SVector{3, Float64}, 4}(undef, nn, nn, nn, ne)
        source_terms.cached_step = -1
    end

    # Recompute acceleration field only when the OU process took a step
    if source_terms.cached_step != source_terms.generator.step
        Trixi.@threaded for element in Trixi.eachelement(dg, cache)
            for k in Trixi.eachnode(dg), j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
                x_local = Trixi.get_node_coords(node_coordinates, equations, dg, i, j, k,
                                                element)
                accel = TurbGen.get_turb_vector(source_terms.generator, x_local)
                source_terms.accel_cache[i, j, k, element] = SVector(accel[1], accel[2],
                                                                     accel[3])
            end
        end
        source_terms.cached_step = source_terms.generator.step
    end

    # Apply the source terms using the cached acceleration
    Trixi.@threaded for element in Trixi.eachelement(dg, cache)
        for k in Trixi.eachnode(dg), j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            u_local = Trixi.get_node_vars(u, equations, dg, i, j, k, element)
            rho, rho_v1, rho_v2, rho_v3, rho_e = u_local
            v1, v2, v3 = rho_v1 / rho, rho_v2 / rho, rho_v3 / rho

            ax, ay, az = source_terms.accel_cache[i, j, k, element]

            du_local = SVector(zero(eltype(u_local)),
                               rho * ax, rho * ay, rho * az,
                               rho * (v1 * ax + v2 * ay + v3 * az))

            Trixi.add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
        end
    end

    return nothing
end
source_terms = TurbulentForcing(turb_gen)

###############################################################################
# custom analysis integrals for turbulence diagnostics

# for RMS rho_v ^2
rho_v1_squared(u, eq::CompressibleEulerEquations3D) = u[2]^2 / u[1]
rho_v2_squared(u, eq::CompressibleEulerEquations3D) = u[3]^2 / u[1]
rho_v3_squared(u, eq::CompressibleEulerEquations3D) = u[4]^2 / u[1]

# for density variance
density_squared(u, eq::CompressibleEulerEquations3D) = u[1]^2

# Mach number M = |v|/c_s
function mach_number_local(u, equations::CompressibleEulerEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_mag = sqrt(v1^2 + v2^2 + v3^2)
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho)
    c_s = sqrt(equations.gamma * p / rho)
    return v_mag / c_s
end

# M^2 = v^2/c_s^2
function mach_squared(u, equations::CompressibleEulerEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_mag = sqrt(v1^2 + v2^2 + v3^2)
    v_sq = v_mag^2
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho)
    cs_sq = equations.gamma * p / rho
    return v_sq / cs_sq
end

###############################################################################
# solver with subcell IDP shock capturing

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure])
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# mesh

coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (1.0, 1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = cell,
                n_cells_max = 1_000_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 8.0 * t_turnover)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     output_directory = analysis_outdir,
                                     analysis_errors = Symbol[],
                                     extra_analysis_integrals = (energy_kinetic,
                                                                 rho_v1_squared,
                                                                 rho_v2_squared,
                                                                 rho_v3_squared,
                                                                 Trixi.density,
                                                                 density_squared,
                                                                 mach_number_local,
                                                                 mach_squared,
                                                                 entropy))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 200,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = solution_outdir,
                                     extra_node_variables = (:limiting_coefficient,))

stepsize_callback = StepsizeCallback(cfl = 0.5)

# Update the OU process each timestep
turbulence_callback = DiscreteCallback((u, t, integrator) -> true,
                                       integrator -> (TurbGen.check_for_update!(turb_gen,
                                                                                integrator.t);
                                                      nothing);
                                       save_positions = (false, false))

# Save z=0.5 slice plots of primitive variables at each analysis interval
mkpath(slices_outdir)
function save_slice_plot(plot_data, variable_names;
                         show_mesh = true, plot_arguments = Dict{Symbol, Any}(),
                         time = nothing, timestep = nothing)
    plots = [Plots.plot(plot_data[v]; title = v, plot_arguments...) for v in variable_names]
    cols = ceil(Int, sqrt(length(plots)))
    rows = div(length(plots), cols, RoundUp)
    Plots.plot(plots..., layout = (rows, cols), size = (400 * cols, 350 * rows))
    Plots.savefig(joinpath(slices_outdir, @sprintf("slice_z05_%09d.png", timestep)))
end

visualization_callback = VisualizationCallback(semi,
                                               (u, semi; kwargs...) -> PlotData2D(u, semi;
                                                                                  slice = :xy,
                                                                                  point = (0.0,
                                                                                           0.0,
                                                                                           0.5),
                                                                                  kwargs...);
                                               interval = 200,
                                               solution_variables = cons2prim,
                                               show_mesh = false,
                                               plot_creator = save_slice_plot)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        visualization_callback,
                        turbulence_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # overwritten by stepsize_callback
                  callback = callbacks);
