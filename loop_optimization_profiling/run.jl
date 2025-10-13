using Mooncake
using Mooncake: CoDual, primal, tangent, zero_codual, fdata, zero_rdata
using BenchmarkTools
using Profile
using Dates

include("benchmarks.jl")

# Helper functions from Issue #156
function prep_test_case(fargs...)
    rule = Mooncake.build_rrule(fargs...)
    coduals = map(zero_codual, fargs)
    return rule, coduals
end

function to_benchmark(__rrule!!::R, dx::Vararg{CoDual,N}) where {R,N}
    dx_f = Mooncake.tuple_map(x -> CoDual(primal(x), Mooncake.fdata(tangent(x))), dx)
    out, pb!! = __rrule!!(dx_f...)
    return pb!!(Mooncake.zero_rdata(primal(out)))
end

function run_many_times(N, f, args::Vararg{Any, P}) where {P}
    @inbounds for _ in 1:N
        f(args...)
    end
    return nothing
end

# Run benchmark and save results
function run_benchmark(rule, coduals; profile_iters=1_000_000)
    # Warm-up
    to_benchmark(rule, coduals...)

    # Benchmark
    result = @benchmark to_benchmark($rule, $coduals...)

    # Profile
    Profile.clear()
    @profile run_many_times(profile_iters, to_benchmark, rule, coduals...)

    return result
end

# Main
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
env_name = basename(dirname(Base.active_project()))
results_dir = joinpath(@__DIR__, "results", "$(env_name)_$(timestamp)")
mkpath(results_dir)

println("Mooncake: ", pkgversion(Mooncake))
println("Results: $results_dir")

results = Dict()
for (name, f, args) in get_benchmarks()
    println("\n$name")

    rule, coduals = prep_test_case(f, args...)
    results[name] = run_benchmark(rule, coduals)

    # Save profile
    open(joinpath(results_dir, "profile_$(name).txt"), "w") do io
        Profile.print(io; maxdepth=20)
    end
end

# Save summary
open(joinpath(results_dir, "summary.txt"), "w") do io
    println(io, "Mooncake version: ", pkgversion(Mooncake))
    println(io, "Timestamp: ", timestamp)
    println(io)
    for (name, result) in results
        println(io, name, ":")
        println(io, result)
        println(io)
    end
end

println("\nDone")
