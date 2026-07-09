using OrdinaryDiffEqLowStorageRK: DiscreteCallback
using Printf
#using Plots

include(joinpath(@__DIR__, "TurbGen.jl"))
using .TurbGen

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 5.0e-6

equations = CompressibleEulerEquations3D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

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

turb_velocity = 0.5       # target Mach number
turb_sol_weight = 1.0     # solenoidal weight (1.0 = purely solenoidal)
cell = 4
name_addition = ""
num_cells = 2^cell
polydeg = 4
domain_length = 1.0
prefix = "../../scratch/output/test/"

run_label = "v$(turb_velocity)_sol$(turb_sol_weight)_cells$(num_cells)_L$(domain_length)_pd$(polydeg)_mu$(mu)_pr$(prandtl_number())_ver$(name_addition)"

analysis_outdir = joinpath(prefix, "Analysis_ns_subcell_$(run_label)")
solution_outdir = joinpath(prefix, "out_ns_subcell_$(run_label)")
turb_gen = TurbGen.TurbGenGenerator(seed = 42)
TurbGen.init_driving!(turb_gen,
                      Dict{String, Any}("L" => [domain_length, domain_length, domain_length],
                                        "velocity" => turb_velocity,
                                        "k_driv" => 1.5,
                                        "k_min" => 1.0,
                                        "k_max" => 2.0,
                                        "spectral_slope" => -5.0 / 3.0,
                                        "sol_weight" => turb_sol_weight,
                                        "spect_form" => 2,
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

# fills accel_cache for every node, threaded over elements
function _compute_turbgen_accel!(source_terms, dg, cache, node_coordinates, equations)
    Trixi.@threaded for element in Trixi.eachelement(dg, cache)
        for k in Trixi.eachnode(dg), j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            x_local = Trixi.get_node_coords(node_coordinates, equations, dg, i, j, k,
                                            element)
            accel = TurbGen.get_turb_vector(source_terms.generator, x_local)
            source_terms.accel_cache[i, j, k, element] = SVector(accel[1], accel[2],
                                                                  accel[3])
        end
    end
    return nothing
end

# adds the cached acceleration into du, threaded over elements
function _apply_turbgen_sources!(du, u, source_terms, equations, dg, cache)
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
        Trixi.@trixi_timeit Trixi.timer() "turbgen accel field" _compute_turbgen_accel!(
            source_terms, dg, cache, node_coordinates, equations)
        source_terms.cached_step = source_terms.generator.step
    end

    # Apply the source terms using the cached acceleration
    Trixi.@trixi_timeit Trixi.timer() "turbgen source application" _apply_turbgen_sources!(
        du, u, source_terms, equations, dg, cache)

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

# power injected by the forcing, rho*(v.a). should roughly balance dissipation
# once things settle into a statistically steady state
struct InjectedPowerIntegral{S}
    source::S
end

function (ip::InjectedPowerIntegral)(u, equations::CompressibleEulerEquations3D)
    return zero(eltype(u))
end

# uses the cached acceleration instead of recomputing it per node
function Trixi.analyze(ip::InjectedPowerIntegral, du, u, t,
                       semi::Trixi.AbstractSemidiscretization)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    # equations is a tuple (hyp, par) here since this is the NS case
    eqs_hyp = equations isa Tuple ? equations[1] : equations

    src = ip.source
    nn = Trixi.nnodes(solver)
    ne = Trixi.nelements(solver, cache)

    # Make sure the acceleration cache is populated
    if size(src.accel_cache) != (nn, nn, nn, ne)
        return zero(real(eltype(u)))
    end

    w = solver.basis.weights
    inv_jacobian = cache.elements.inverse_jacobian

    total_power = zero(real(eltype(u)))
    total_volume = zero(real(eltype(u)))
    for element in Trixi.eachelement(solver, cache)
        abs_J = abs(inv(inv_jacobian[element]))
        for k in Trixi.eachnode(solver), j in Trixi.eachnode(solver),
            i in Trixi.eachnode(solver)

            u_local = Trixi.get_node_vars(u, eqs_hyp, solver, i, j, k, element)
            rho = u_local[1]
            v1 = u_local[2] / rho
            v2 = u_local[3] / rho
            v3 = u_local[4] / rho

            ax, ay, az = src.accel_cache[i, j, k, element]

            dV = w[i] * w[j] * w[k] * abs_J
            total_power += rho * (v1 * ax + v2 * ay + v3 * az) * dV
            total_volume += dV
        end
    end

    # divide by volume so this lines up with the other volume-averaged integrals
    return total_power / total_volume
end

# Pretty names for the analysis output file
Trixi.pretty_form_utf(::InjectedPowerIntegral) = "∑P_inject"
Trixi.pretty_form_ascii(::InjectedPowerIntegral) = "P_inject"
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
coordinates_max = (domain_length, domain_length, domain_length)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = cell,
                n_cells_max = 1_000_000,
                periodicity = true)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             source_terms = source_terms,
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 8.0 * t_turnover)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
injected_power = InjectedPowerIntegral(source_terms)
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     output_directory = analysis_outdir,
                                     analysis_errors = Symbol[],
                                     extra_analysis_integrals = (energy_total,
                                                                 energy_kinetic,
                                                                 energy_internal,
                                                                 injected_power,
                                                                 rho_v1_squared,
                                                                 rho_v2_squared,
                                                                 rho_v3_squared,
                                                                 Trixi.density,
                                                                 density_squared,
                                                                 pressure,
                                                                 mach_squared,
                                                                 entropy))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 800,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = solution_outdir,
                                     extra_node_variables = (:limiting_coefficient,))

stepsize_callback = StepsizeCallback(cfl = 0.5)

# Update the OU process each timestep
turbulence_callback = DiscreteCallback((u, t, integrator) -> true,
                                       integrator -> (Trixi.@trixi_timeit Trixi.timer() "turbgen OU update" TurbGen.check_for_update!(turb_gen,
                                                                                                                                       integrator.t);
                                                       nothing);
                                       save_positions = (false, false))

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        turbulence_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # overwritten by stepsize_callback
                  callback = callbacks);
