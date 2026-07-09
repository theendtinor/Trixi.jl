using OrdinaryDiffEqLowStorageRK: DiscreteCallback
using Printf
#using Plots

include(joinpath(@__DIR__, "..", "TurbGen.jl"))
using .TurbGen

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 5.0e-7

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

turb_velocity = 0.3       # target Mach number (M ~ 0.3)
turb_sol_weight = 1.0     # solenoidal weight (1.0 = purely solenoidal)
cell = 5
name_addition = "_isothermal_M0.3"
num_cells = 2^cell
polydeg = 3
domain_length = 1.0
prefix = "../../scratch/output/test/"

run_label = "v$(turb_velocity)_sol$(turb_sol_weight)_cells$(num_cells)_L$(domain_length)_pd$(polydeg)_mu$(mu)_pr$(prandtl_number())_ver$(name_addition)"

analysis_outdir = joinpath(prefix, "Analysis_ns_subcell_$(run_label)")
solution_outdir = joinpath(prefix, "out_ns_subcell_$(run_label)")
turb_gen = TurbGen.TurbGenGenerator(seed = 42)
TurbGen.init_driving!(turb_gen,
                      Dict{String, Any}("L" => [domain_length, domain_length, domain_length],
                                        "velocity" => turb_velocity,
                                        # picked k_driv so t_decay = ts = 1 (k_driv = L/(ts*v) ~ 3.33)
                                        "k_driv" => 1.0 / (1.0 * turb_velocity),
                                        "k_min" => 1.0,   # kmin = 2*pi*1
                                        "k_max" => 2.0,   # kmax = 2*pi*2
                                        "spectral_slope" => -5.0 / 3.0,
                                        "sol_weight" => turb_sol_weight,
                                        "spect_form" => 2,
                                        "random_seed" => 42,
                                        "nsteps_per_t_turb" => 200,
                                        "ampl_factor" => [1.0, 1.0, 1.0]))
# t_decay (OU memory) and t_eddy (actual turnover time) are different things and
# shouldn't be confused -- we want ~8 turnover times regardless of what t_decay is.
# t_eddy uses the midpoint of [k_min,k_max] as the driving scale, ~2.22 here
t_decay = turb_gen.t_decay
k_driv_band = 0.5 * (1.0 + 2.0)
t_eddy = (domain_length / k_driv_band) / turb_velocity

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
        Trixi.@trixi_timeit Trixi.timer() "turbgen accel field" _compute_turbgen_accel!(source_terms,
                                                                                        dg,
                                                                                        cache,
                                                                                        node_coordinates,
                                                                                        equations)
        source_terms.cached_step = source_terms.generator.step
    end

    # Apply the source terms using the cached acceleration
    Trixi.@trixi_timeit Trixi.timer() "turbgen source application" _apply_turbgen_sources!(du,
                                                                                           u,
                                                                                           source_terms,
                                                                                           equations,
                                                                                           dg,
                                                                                           cache)

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

    # equations is a tuple (hyp, par)
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

# tracking how much energy the isothermal reset pulls out (following the
# dissipation definition used in Bauer & Springel 2012). every step we reset
# rho_e back to the target, so eps_diss = energy removed, eps_cool = energy
# added back in the (frequent) cases where the reset goes the other way.

mutable struct DissipationAccumulator
    total::Float64           # running volume-integral of dissipated energy since t=0
    at_last_report::Float64  # value at the previous analysis call (for rate)
    t_last_report::Float64   # simulation time at the previous analysis call
    cool::Float64            # cumulative discarded cooling ∫ max(0,-Δe_int) dV
    cool_at_last_report::Float64
    t_cool_last_report::Float64
end
DissipationAccumulator() = DissipationAccumulator(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

struct DissipatedPowerIntegral
    accum::DissipationAccumulator
    domain_volume::Float64
end

# unused, Trixi.analyze below is what actually gets called
function (dp::DissipatedPowerIntegral)(u, equations::CompressibleEulerEquations3D)
    return zero(eltype(u))
end

# mean rate since the last analysis call, then move the reference point up
function Trixi.analyze(dp::DissipatedPowerIntegral, du, u, t,
                       semi::Trixi.AbstractSemidiscretization)
    dt_interval = t - dp.accum.t_last_report
    if dt_interval <= 0
        return zero(real(eltype(u)))
    end
    rate = (dp.accum.total - dp.accum.at_last_report) /
           (dt_interval * dp.domain_volume)
    dp.accum.at_last_report = dp.accum.total
    dp.accum.t_last_report = t
    return rate
end

Trixi.pretty_form_utf(::DissipatedPowerIntegral) = "ε_diss"
Trixi.pretty_form_ascii(::DissipatedPowerIntegral) = "eps_diss"

struct CoolingPowerIntegral
    accum::DissipationAccumulator
    domain_volume::Float64
end

function (cp::CoolingPowerIntegral)(u, equations::CompressibleEulerEquations3D)
    return zero(eltype(u))
end

function Trixi.analyze(cp::CoolingPowerIntegral, du, u, t,
                       semi::Trixi.AbstractSemidiscretization)
    dt_interval = t - cp.accum.t_cool_last_report
    if dt_interval <= 0
        return zero(real(eltype(u)))
    end
    rate = (cp.accum.cool - cp.accum.cool_at_last_report) /
           (dt_interval * cp.domain_volume)
    cp.accum.cool_at_last_report = cp.accum.cool
    cp.accum.t_cool_last_report = t
    return rate
end

Trixi.pretty_form_utf(::CoolingPowerIntegral) = "ε_cool"
Trixi.pretty_form_ascii(::CoolingPowerIntegral) = "eps_cool"

function integration_measure(semi)
    _, _, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    w = solver.basis.weights
    inv_jac = cache.elements.inverse_jacobian
    V = 0.0
    for element in Trixi.eachelement(solver, cache)
        absJ = abs(inv(inv_jac[element]))
        for k in Trixi.eachnode(solver), j in Trixi.eachnode(solver),
            i in Trixi.eachnode(solver)

            V += w[i] * w[j] * w[k] * absJ
        end
    end
    return V
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

# 8 turnover times total, ~3 to spin up + 5 to collect statistics
tspan = (0.0, 8.0 * t_eddy)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
injected_power = InjectedPowerIntegral(source_terms)
diss_accum = DissipationAccumulator()
V_meas = integration_measure(semi)
dissipated_power = DissipatedPowerIntegral(diss_accum, V_meas)
cooling_power = CoolingPowerIntegral(diss_accum, V_meas)
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     output_directory = analysis_outdir,
                                     analysis_errors = Symbol[],
                                     extra_analysis_integrals = (energy_total,
                                                                 energy_kinetic,
                                                                 energy_internal,
                                                                 injected_power,
                                                                 dissipated_power,
                                                                 cooling_power,
                                                                 rho_v1_squared,
                                                                 rho_v2_squared,
                                                                 rho_v3_squared,
                                                                 Trixi.density,
                                                                 density_squared,
                                                                 pressure,
                                                                 mach_squared,
                                                                 entropy))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 400,
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

###############################################################################

# isothermal reset: every step, overwrite rho_e so the local sound speed is
# cs_target everywhere (density/momentum untouched). cheaper than an actual
# isothermal Riemann solver, same idea as Bauer & Springel 2012.
# p_reset = rho*cs_target^2/gamma, rho_e = p_reset/(gamma-1) + kinetic part

# with rho=1, p=1/gamma initially, cs^2 = gamma*p/rho = 1 already, so cs_target=1
const cs_target = 1.0

isothermal_reset_callback = DiscreteCallback((u, t, integrator) -> true,
                                             integrator -> begin
                                                 Trixi.@trixi_timeit Trixi.timer() "isothermal reset" begin
                                                     u_ode = integrator.u
                                                     mesh, eqs, solver, cache = Trixi.mesh_equations_solver_cache(semi)
                                                     # eqs is a Tuple for HyperbolicParabolic semidiscretizations
                                                     eqs_hyp = eqs isa Tuple ? eqs[1] : eqs

                                                     e_int_specific_target = cs_target^2 /
                                                                             (eqs_hyp.gamma *
                                                                              (eqs_hyp.gamma -
                                                                               1))

                                                     # reshape is a view, so writes here go straight back into u_ode
                                                     ne_loc = Trixi.nelements(solver, cache)
                                                     nn = Trixi.nnodes(solver)
                                                     nvars = Trixi.nvariables(eqs_hyp)
                                                     u = reshape(u_ode, nvars, nn, nn, nn,
                                                                 ne_loc)

                                                     diss_per_elem = zeros(Float64, ne_loc)
                                                     cool_per_elem = zeros(Float64, ne_loc)
                                                     w = solver.basis.weights
                                                     inv_jacobian = cache.elements.inverse_jacobian

                                                     Trixi.@threaded for element in Trixi.eachelement(solver,
                                                                                                      cache)
                                                         abs_J = abs(inv(inv_jacobian[element]))
                                                         local_diss = 0.0
                                                         local_cool = 0.0
                                                         for k in Trixi.eachnode(solver),
                                                             j in Trixi.eachnode(solver),
                                                             i in Trixi.eachnode(solver)

                                                             u_node = Trixi.get_node_vars(u,
                                                                                          eqs_hyp,
                                                                                          solver,
                                                                                          i,
                                                                                          j,
                                                                                          k,
                                                                                          element)
                                                             rho, rho_v1, rho_v2, rho_v3,
                                                             rho_e = u_node

                                                             # kinetic part stays as is
                                                             rho_e_kin = 0.5 *
                                                                         (rho_v1^2 +
                                                                          rho_v2^2 +
                                                                          rho_v3^2) / rho

                                                             # this is the actual reset, keeps p = rho*cs_target^2/gamma
                                                             rho_e_reset = rho *
                                                                           e_int_specific_target +
                                                                           rho_e_kin

                                                             # track how much energy this removes/adds so we can report it later
                                                             dV = w[i] * w[j] * w[k] * abs_J
                                                             dev = rho_e - rho_e_reset
                                                             local_diss += max(0.0, dev) *
                                                                           dV
                                                             local_cool += max(0.0, -dev) *
                                                                           dV

                                                             u_new = SVector(rho, rho_v1,
                                                                             rho_v2, rho_v3,
                                                                             rho_e_reset)
                                                             Trixi.set_node_vars!(u,
                                                                                  u_new,
                                                                                  eqs_hyp,
                                                                                  solver, i,
                                                                                  j, k,
                                                                                  element)
                                                         end
                                                         diss_per_elem[element] = local_diss
                                                         cool_per_elem[element] = local_cool
                                                     end
                                                     diss_accum.total += sum(diss_per_elem)
                                                     diss_accum.cool += sum(cool_per_elem)
                                                 end
                                                 nothing
                                             end;
                                             save_positions = (false, false))

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        turbulence_callback,
                        isothermal_reset_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # overwritten by stepsize_callback
                  callback = callbacks);
