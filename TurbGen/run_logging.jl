
# Console logging for the TurbGen elixirs. start_console_log redirects stdout and
# stderr into a text file next to the solution output, so the run settings, the
# SummaryCallback boxes, the analysis output and the final timer table all end
# up in one file.

module RunLog

using Dates: Dates
using Logging: ConsoleLogger, global_logger

export start_console_log, stop_console_log, print_settings, turbgen_settings, environment

struct ConsoleLog
    path::String
    file::IOStream
    stdout_orig::IO
    stderr_orig::IO
    logger_orig::Any
end

function start_console_log(path)
    mkpath(dirname(path))
    file = open(path, "w")

    stdout_orig, stderr_orig = stdout, stderr
    redirect_stdout(file)
    redirect_stderr(file)

    # the default logger keeps writing to the stream it was built with, so point
    # it at the file too
    logger_orig = global_logger(ConsoleLogger(file))

    return ConsoleLog(abspath(path), file, stdout_orig, stderr_orig, logger_orig)
end

function stop_console_log(log::ConsoleLog)
    flush(stdout)
    flush(stderr)

    global_logger(log.logger_orig)
    redirect_stdout(log.stdout_orig)
    redirect_stderr(log.stderr_orig)
    close(log.file)

    println("Console output written to ", log.path)
    return nothing
end

function print_settings(title; settings...)
    width = maximum(length(string(key)) for key in keys(settings); init = 0)
    println("\n", title)
    for (key, value) in pairs(settings)
        println("  ", rpad(replace(string(key), '_' => ' '), width), " : ", value)
    end
    println()
    flush(stdout)
    return nothing
end

# scalar parameters of the generator; the Fourier coefficients (mode, aka, akb,
# OUphases, ampl), the update counter and the RNG are evolving state, not settings
function turbgen_settings(gen)
    return (L = gen.L, nmodes = gen.nmodes, random_seed = gen.random_seed,
            kmin = gen.kmin, kmax = gen.kmax, spectral_slope = gen.spectral_slope,
            sol_weight = gen.sol_weight, sol_weight_norm = gen.sol_weight_norm,
            spect_form = gen.spect_form, velocity = gen.velocity,
            t_decay = gen.t_decay, dt = gen.dt, energy = gen.energy,
            OUvar = gen.OUvar, nsteps_per_t_turb = gen.nsteps_per_t_turb,
            ampl_factor = gen.ampl_factor, ampl_auto_adjust = gen.ampl_auto_adjust)
end

function environment()
    commit = try
        readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
    catch
        "unavailable"
    end

    return (date = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
            hostname = gethostname(),
            julia = string(VERSION),
            threads = Threads.nthreads(),
            git_commit = commit)
end

end
