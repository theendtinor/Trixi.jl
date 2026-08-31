# Reference implementation: https://github.com/chfeder/turbulence_generator

module TurbGen

using Random

export TurbGenGenerator,
       init_driving!,
       check_for_update!,
       get_turb_vector

const X, Y, Z = 1, 2, 3

const MAX_NMODES = 100_000  

mutable struct TurbGenGenerator
    L::Vector{Float64}    # box side lengths

    # wave vectors, mode[dim][mode_index]
    mode::Vector{Vector{Float64}}

    # Fourier coefficients after the Helmholtz decomposition (aka real, akb imaginary)
    aka::Vector{Vector{Float64}}
    akb::Vector{Vector{Float64}}

    # OU phases, flattened: OUphases[6*(m-1) + 2*(d-1) + ir], ir=1 real, ir=2 imag
    OUphases::Vector{Float64}

    ampl::Vector{Float64}  # spectral weight per mode

    nmodes::Int
    random_seed::Int

    kmin::Float64            # driven range in k, both already scaled by 2pi/L
    kmax::Float64
    spectral_slope::Float64  # exponent of E(k) ~ k^slope, only used by spect_form 2

    sol_weight::Float64       # 1 = purely solenoidal, 0 = purely compressive forcing
    sol_weight_norm::Float64  # keeps RMS forcing independent of sol_weight (Eq. 9)
    spect_form::Int           # 0 = band, 1 = parabola, 2 = power law

    velocity::Float64  # target RMS velocity of the driven turbulence
    t_decay::Float64   # correlation time of the OU process, one eddy turnover
    dt::Float64        # time between two OU updates
    energy::Float64    # energy injection rate
    OUvar::Float64     # standard deviation of the OU phases

    nsteps_per_t_turb::Int  # OU updates per turnover time, sets dt
    step::Int               # index of the last OU update, -1 before the first one

    ampl_factor::Vector{Float64}  # per-direction scaling of the forcing amplitude
    ampl_auto_adjust::Bool        # rescale ampl_factor towards the target velocity

    rng::MersenneTwister
end

function TurbGenGenerator(; seed::Int = 140281)
    TurbGenGenerator([1.0, 1.0, 1.0],
                     [Float64[], Float64[], Float64[]],
                     [Float64[], Float64[], Float64[]],
                     [Float64[], Float64[], Float64[]],
                     Float64[],
                     Float64[],
                     0,
                     seed,
                     0.0, 0.0,
                     -5 / 3,
                     2,  # spect_form
                     0.5, 1.0,
                     1.0, 1.0, 0.1, 1.0, 1.0,
                     10, -1,
                     [1.0, 1.0, 1.0],
                     false,
                     MersenneTwister(seed))
end

# normalisation
function set_solenoidal_weight_normalisation!(gen::TurbGenGenerator)
    zeta = gen.sol_weight
    gen.sol_weight_norm = sqrt(3.0) / sqrt(1 - 2 * zeta + 3 * zeta^2)
end


function init_modes!(gen::TurbGenGenerator)
    (; kmin, kmax, spectral_slope, spect_form, L) = gen

    kc = spect_form == 1 ? 0.5 * (kmin + kmax) : kmin  # ref wavenumber
    parab_prefact = kmax > kmin ? -4.0 / (kmax - kmin)^2 : 0.0
    ikmax = 256


    nmodes_count = 0
    for ikx in (-ikmax):ikmax
        kx = 2pi * ikx / L[X]
        for iky in (-ikmax):ikmax
            ky = 2pi * iky / L[Y]
            for ikz in (-ikmax):ikmax
                kz = 2pi * ikz / L[Z]
                ka = sqrt(kx^2 + ky^2 + kz^2)
                if ka >= kmin && ka <= kmax
                    nmodes_count += 1
                end
            end
        end
    end

    if nmodes_count > MAX_NMODES
        error("Too many stirring modes: nmodes = $nmodes_count > $MAX_NMODES")
    end

    for d in 1:3
        empty!(gen.mode[d])
    end
    empty!(gen.ampl)
    gen.nmodes = 0

    for ikx in (-ikmax):ikmax
        kx = 2 * pi * ikx / L[X]
        for iky in (-ikmax):ikmax
            ky = 2 * pi * iky / L[Y]
            for ikz in (-ikmax):ikmax
                kz = 2 * pi * ikz / L[Z]
                ka = sqrt(kx^2 + ky^2 + kz^2)

                if ka >= kmin && ka <= kmax
                    if spect_form == 0          # band / flat
                        amplitude = 1.0
                    elseif spect_form == 1      # parabola
                        amplitude = abs(parab_prefact * (ka - kc)^2 + 1.0)
                    else                        # power law
                        amplitude = (ka / kc)^spectral_slope
                    end

                    amplitude = sqrt(amplitude) * (kc / ka)

                    push!(gen.ampl, amplitude)
                    push!(gen.mode[X], kx)
                    push!(gen.mode[Y], ky)
                    push!(gen.mode[Z], kz)
                    gen.nmodes += 1
                end
            end
        end
    end
end

# draw the initial OU phases from a Gaussian with standard deviation OUvar
function OU_noise_init!(gen::TurbGenGenerator)
    (; nmodes, OUvar) = gen
    resize!(gen.OUphases, nmodes * 3 * 2)
    for m in 1:nmodes
        for d in 1:3
            for ir in 1:2
                idx = 6 * (m - 1) + 2 * (d - 1) + ir
                gen.OUphases[idx] = OUvar * randn(gen.rng)
            end
        end
    end
end

# one Euler-Maruyama-exact step of the OU process,
# x_{n+1} = f*x_n + sigma*sqrt(1-f^2)*N(0,1), f = exp(-dt/t_decay)
function OU_noise_update!(gen::TurbGenGenerator)
    (; nmodes, dt, t_decay, OUvar) = gen

    f = exp(-dt / t_decay)
    noise_scale = sqrt(1 - f^2)

    for m in 1:nmodes
        for d in 1:3
            for ir in 1:2
                idx = 6 * (m - 1) + 2 * (d - 1) + ir
                gen.OUphases[idx] = f * gen.OUphases[idx] +
                                    noise_scale * OUvar * randn(gen.rng)
            end
        end
    end
end

# Helmholtz decomposition of the OU vector: split it into a divergence-free and a
# curl-free part in Fourier space and mix the two with the weight sol_weight
function get_decomposition_coeffs!(gen::TurbGenGenerator)
    (; nmodes, sol_weight, mode, OUphases) = gen

    for d in 1:3
        resize!(gen.aka[d], nmodes)
        resize!(gen.akb[d], nmodes)
    end

    for m in 1:nmodes
        ka = 0.0
        kb = 0.0
        kk = 0.0

        for d in 1:3
            k_d = mode[d][m]
            kk += k_d^2
            ka += k_d * OUphases[6 * (m - 1) + 2 * (d - 1) + 2]
            kb += k_d * OUphases[6 * (m - 1) + 2 * (d - 1) + 1]
        end

        for d in 1:3
            k_d = mode[d][m]
            ir = 6 * (m - 1) + 2 * (d - 1) + 1
            ii = 6 * (m - 1) + 2 * (d - 1) + 2

            diva = k_d * ka / kk
            divb = k_d * kb / kk
            curla = OUphases[ir] - divb
            curlb = OUphases[ii] - diva

            gen.aka[d][m] = sol_weight * curla + (1 - sol_weight) * divb
            gen.akb[d][m] = sol_weight * curlb + (1 - sol_weight) * diva
        end
    end
end

# set up box, spectrum and OU parameters from a parameter dict and build the initial
# mode set. must run before the first check_for_update!
function init_driving!(gen::TurbGenGenerator, params::Dict{String, Any};
                       time::Float64 = 0.0)
    L = get(params, "L", [1.0, 1.0, 1.0])
    gen.L[X] = L[1]
    gen.L[Y] = L[2]
    gen.L[Z] = L[3]

    gen.velocity = get(params, "velocity", 1.0)
    k_driv = get(params, "k_driv", 2.0)
    k_min = get(params, "k_min", 1.0)
    k_max = get(params, "k_max", 2.0)
    gen.spectral_slope = get(params, "spectral_slope", -5 / 3)
    gen.sol_weight = get(params, "sol_weight", 0.5)
    gen.spect_form = get(params, "spect_form", 2)
    gen.random_seed = get(params, "random_seed", 140281)
    gen.nsteps_per_t_turb = get(params, "nsteps_per_t_turb", 10)

    ampl_factor_input = get(params, "ampl_factor", [1.0, 1.0, 1.0])
    if length(ampl_factor_input) == 1
        gen.ampl_factor .= ampl_factor_input[1]
    else
        gen.ampl_factor .= ampl_factor_input
    end
    gen.ampl_auto_adjust = get(params, "ampl_auto_adjust", false)

    gen.rng = MersenneTwister(gen.random_seed)

    # nudge kmin/kmax by eps so modes sitting right on the boundary don't get dropped
    gen.kmin = (k_min - eps()) * 2pi / gen.L[X]
    gen.kmax = (k_max + eps()) * 2pi / gen.L[X]

    gen.t_decay = (gen.L[X] / k_driv) / gen.velocity
    gen.dt = gen.t_decay / gen.nsteps_per_t_turb
    gen.step = -1

    ampl_coeff = 0.15  # empirical prefactor
    gen.energy = (ampl_coeff * gen.velocity)^3 / gen.L[X]
    gen.OUvar = sqrt(gen.energy / gen.t_decay)

    for d in 1:3
        gen.ampl_factor[d] = gen.ampl_factor[d]^1.5
    end

    set_solenoidal_weight_normalisation!(gen)
    init_modes!(gen)
    OU_noise_init!(gen)
    get_decomposition_coeffs!(gen)

    return nothing
end

# advance the OU process up to `time` and refresh the Fourier coeffs if a step
# actually happened. returns whether anything changed
function check_for_update!(gen::TurbGenGenerator, time::Float64;
                           v_turb::Union{Vector{Float64}, Nothing} = nothing)
    step_requested = floor(Int, time / gen.dt)

    if step_requested <= gen.step
        return false
    end

    if gen.ampl_auto_adjust && v_turb !== nothing && v_turb[X] > 0
        if time > 0.1 * gen.t_decay
            for d in 1:3
                ampl_factor_new = gen.ampl_factor[d] *
                                  (gen.velocity / v_turb[d] / sqrt(3))^1.5

                ampl_adjust_timescale = sqrt(ampl_factor_new / gen.ampl_factor[d]) *
                                        gen.t_decay

                f = exp(-gen.dt / ampl_adjust_timescale)
                gen.ampl_factor[d] = f * gen.ampl_factor[d] + (1 - f) * ampl_factor_new
            end
        end
    end

    for _ in (gen.step):(step_requested - 1)
        OU_noise_update!(gen)
        gen.step += 1
    end

    get_decomposition_coeffs!(gen)

    return true
end

# direct Fourier sum for the forcing field at `pos`. the factor of 2 comes from
# only summing over half the modes 
function get_turb_vector(gen::TurbGenGenerator, pos)
    (; nmodes, sol_weight_norm, mode, ampl, aka, akb, ampl_factor) = gen

    vx = 0.0
    vy = 0.0
    vz = 0.0

    @inbounds ax = ampl_factor[X]
    @inbounds ay = ampl_factor[Y]
    @inbounds az = ampl_factor[Z]

    @inbounds @simd for m in 1:nmodes
        sinx, cosx = sincos(mode[X][m] * pos[1])
        siny, cosy = sincos(mode[Y][m] * pos[2])
        sinz, cosz = sincos(mode[Z][m] * pos[3])

        A = 2 * sol_weight_norm * ampl[m]

        # real and imaginary part of exp(i k.x), expanded from the three sincos pairs
        real_part = (cosx * cosy - sinx * siny) * cosz - (sinx * cosy + cosx * siny) * sinz
        imag_part = cosx * (cosy * sinz + siny * cosz) + sinx * (cosy * cosz - siny * sinz)

        vx += A * (aka[X][m] * real_part - akb[X][m] * imag_part) * ax
        vy += A * (aka[Y][m] * real_part - akb[Y][m] * imag_part) * ay
        vz += A * (aka[Z][m] * real_part - akb[Z][m] * imag_part) * az
    end

    return (vx, vy, vz)
end

end
