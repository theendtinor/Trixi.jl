using OrdinaryDiffEqLowStorageRK: DiscreteCallback

include(joinpath(@__DIR__, "..", "TurbGen.jl"))
using .TurbGen

include(joinpath(@__DIR__, "..", "run_logging.jl"))
using .RunLog

###############################################################################

equations = CompressibleEulerEquations3D(1.0001)

function initial_condition_uniform(x, t, equations::CompressibleEulerEquations3D)
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    # p = rho * cs^2 / gamma  so that cs^2 = gamma * p / rho = 1.0
    p = 1.0 / equations.gamma
    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_uniform

###############################################################################

turb_velocity = 0.5
turb_sol_weight = 1.0
cell = 5
name_addition = "isothermal_rs1"
num_cells = 2^cell
polydeg = 3
domain_length = 1.0

run_label = "v$(turb_velocity)_sol$(turb_sol_weight)_cells$(num_cells)_L$(domain_length)_pd$(polydeg)_ver$(name_addition)"
analysis_outdir = "Analysis_euler_subcell_$(run_label)"
solution_outdir = "out_euler_subcell_$(run_label)"

turb_gen = TurbGen.TurbGenGenerator()
TurbGen.init_driving!(turb_gen,
                      Dict{String, Any}("L" => [domain_length, domain_length, domain_length],
                                        "velocity" => turb_velocity,
                                        "k_driv" => 2.0,       # = k_max, so t_decay = T
                                        "k_min" => 1.0,        # fundamental mode 2*pi/L
                                        "k_max" => 2.0,        # first harmonic 4*pi/L
                                        "spectral_slope" => -5.0 / 3.0,
                                        "sol_weight" => turb_sol_weight,
                                        "spect_form" => 0,
                                        "random_seed" => 1,
                                        "nsteps_per_t_turb" => 100,
                                        "ampl_factor" => [1.0, 1.0, 1.0]))

t_turnover = turb_gen.t_decay

###############################################################################
# source terms

mutable struct TurbulentForcing{Gen <: TurbGen.TurbGenGenerator}
    generator::Gen
    accel_cache::Array{SVector{3, Float64}, 4}
    cached_step::Int
end

# never matches generator.step, so the first calc_sources! call fills the cache
const STEP_NEVER_CACHED = typemin(Int)

function TurbulentForcing(generator)
    TurbulentForcing(generator, Array{SVector{3, Float64}, 4}(undef, 0, 0, 0, 0),
                     STEP_NEVER_CACHED)
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

    # allocate the cache once the node and element counts are known
    if size(source_terms.accel_cache) != (nn, nn, nn, ne)
        source_terms.accel_cache = Array{SVector{3, Float64}, 4}(undef, nn, nn, nn, ne)
        source_terms.cached_step = STEP_NEVER_CACHED
    end

    # the acceleration field only changes when the OU process takes a step
    if source_terms.cached_step != source_terms.generator.step
        Trixi.@trixi_timeit Trixi.timer() "turbgen accel field" begin
            _compute_turbgen_accel!(source_terms, dg, cache, node_coordinates, equations)
        end
        source_terms.cached_step = source_terms.generator.step
    end

    Trixi.@trixi_timeit Trixi.timer() "turbgen source application" begin
        _apply_turbgen_sources!(du, u, source_terms, equations, dg, cache)
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

# M^2 = v^2/c_s^2
function mach_squared(u, equations::CompressibleEulerEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_sq = v1^2 + v2^2 + v3^2
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho)
    cs_sq = equations.gamma * p / rho
    return v_sq / cs_sq
end

# power injected by the forcing, rho*(v.a). should roughly balance dissipation
# once things settle into a statistically steady state
struct InjectedPowerIntegral{S}
    source::S
end

# no node position here; the real work is in analyze() below
function (ip::InjectedPowerIntegral)(u, equations::CompressibleEulerEquations3D)
    return zero(eltype(u))
end

# uses the cached acceleration instead of recomputing it per node
function _injected_power(src, u, eqs_hyp, solver, cache)
    nn = Trixi.nnodes(solver)
    ne = Trixi.nelements(solver, cache)

    # cache not allocated yet (analyze can run before the first calc_sources!)
    if size(src.accel_cache) != (nn, nn, nn, ne)
        return zero(real(eltype(u)))
    end

    w = solver.basis.weights
    inv_jacobian = cache.elements.inverse_jacobian

    total_power = zero(real(eltype(u)))
    total_volume = zero(real(eltype(u)))
    for element in Trixi.eachelement(solver, cache)
        # TreeMesh: constant Jacobian per element
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

function Trixi.analyze(ip::InjectedPowerIntegral, du, u, t,
                       semi::Trixi.AbstractSemidiscretization)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    return Trixi.@trixi_timeit Trixi.timer() "turbgen injected power" begin
        _injected_power(ip.source, u, equations, solver, cache)
    end
end

# Pretty names for the analysis output file
Trixi.pretty_form_utf(::InjectedPowerIntegral) = "P_inject"
Trixi.pretty_form_ascii(::InjectedPowerIntegral) = "P_inject"

###############################################################################
# Dissipation tracking (energy extracted by the isothermal reset).

mutable struct DissipationAccumulator
    total::Float64           # running volume-integral of dissipated energy since t=0
    at_last_report::Float64  # value at the previous analysis call (for rate)
    t_last_report::Float64   # simulation time at the previous analysis call
    cool::Float64            # cumulative discarded cooling int max(0,-de_int) dV
    cool_at_last_report::Float64
    t_cool_last_report::Float64
end
DissipationAccumulator() = DissipationAccumulator(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

struct DissipatedPowerIntegral
    accum::DissipationAccumulator
    domain_volume::Float64
end

function (dp::DissipatedPowerIntegral)(u, equations::CompressibleEulerEquations3D)
    return zero(eltype(u))
end

# Called at every analysis_interval: return the mean volume-averaged rate
# over the last interval and reset the reference point.
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

Trixi.pretty_form_utf(::DissipatedPowerIntegral) = "eps_diss"
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

Trixi.pretty_form_utf(::CoolingPowerIntegral) = "eps_cool"
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

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

n_turnovers = 8.0
tspan = (0.0, n_turnovers * t_turnover)
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

save_solution_interval = 250
save_solution = SaveSolutionCallback(interval = save_solution_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = solution_outdir,
                                     extra_node_variables = (:limiting_coefficient,))

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

# advance the OU process once per timestep
function update_turbulence!(integrator)
    Trixi.@trixi_timeit Trixi.timer() "turbgen OU update" begin
        TurbGen.check_for_update!(turb_gen, integrator.t)
    end
    return nothing
end

turbulence_callback = DiscreteCallback((u, t, integrator) -> true, update_turbulence!;
                                       save_positions = (false, false))

###############################################################################
# isothermal reset: overwrite rho_e every step so cs stays at cs_target.
# gamma is ~1.0001 here so p ~ rho*cs^2, that's basically what this enforces

const cs_target = 1.0   # target isothermal sound speed

# Overwrite rho_e at every step so the internal energy matches cs_target while
# the kinetic energy is left untouched. With gamma ~ 1 this enforces
# p ~ rho*cs^2, i.e. an isothermal equation of state. The energy the reset
# removes is tracked as dissipation, the energy it adds back (when the reset
# raises rho_e) as cooling.
function apply_isothermal_reset!(integrator, semi, accum, cs_target)
    Trixi.@trixi_timeit Trixi.timer() "isothermal reset" begin
        _, eqs, solver, cache = Trixi.mesh_equations_solver_cache(semi)
        if eqs isa Tuple
            # Navier-Stokes returns (hyperbolic, parabolic); take the hyperbolic part
            eqs = eqs[1]
        end
        # wrap_array returns a view onto integrator.u, so the writes below update
        # the state in place
        u = Trixi.wrap_array(integrator.u, semi)

        # cs^2 = gamma*(gamma-1)*e_int  ->  target specific internal energy
        e_int_specific_target = cs_target^2 / (eqs.gamma * (eqs.gamma - 1))

        w = solver.basis.weights
        inv_jacobian = cache.elements.inverse_jacobian

        total_diss = 0.0
        total_cool = 0.0
        for element in Trixi.eachelement(solver, cache)
            abs_J = abs(inv(inv_jacobian[element]))
            for k in Trixi.eachnode(solver), j in Trixi.eachnode(solver),
                i in Trixi.eachnode(solver)

                u_node = Trixi.get_node_vars(u, eqs, solver, i, j, k, element)
                rho, rho_v1, rho_v2, rho_v3, rho_e = u_node

                # keep the kinetic part, reset the internal part to the target
                rho_e_kin = 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho
                rho_e_reset = rho * e_int_specific_target + rho_e_kin

                # how much energy the reset removes (diss) or adds back (cool)
                dV = w[i] * w[j] * w[k] * abs_J
                dev = rho_e - rho_e_reset
                total_diss += max(0.0, dev) * dV
                total_cool += max(0.0, -dev) * dV

                u_new = SVector(rho, rho_v1, rho_v2, rho_v3, rho_e_reset)
                Trixi.set_node_vars!(u, u_new, eqs, solver, i, j, k, element)
            end
        end
        accum.total += total_diss
        accum.cool += total_cool
    end
    return nothing
end

isothermal_reset_callback = DiscreteCallback((u, t, integrator) -> true,
                                             integrator -> apply_isothermal_reset!(integrator,
                                                                                   semi,
                                                                                   diss_accum,
                                                                                   cs_target);
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

console_log = RunLog.start_console_log(joinpath(solution_outdir, "console_output.txt"))

# the SummaryCallback already prints mesh, equations and solver
RunLog.print_settings("Run: " * basename(@__FILE__);
                      RunLog.environment()..., run_label, analysis_outdir,
                      solution_outdir, analysis_interval, save_solution_interval,
                      gamma = equations.gamma, cs_target, domain_length, num_cells,
                      polydeg, cfl, n_turnovers, tspan)
RunLog.print_settings("Turbulence driving"; RunLog.turbgen_settings(turb_gen)...)

try
    global sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                             dt = 1.0, # overwritten by stepsize_callback
                             callback = callbacks)
finally
    RunLog.stop_console_log(console_log)
end
