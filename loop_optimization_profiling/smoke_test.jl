using Mooncake
using Mooncake: CoDual, primal, tangent, zero_codual, zero_rdata, fdata, tuple_map
using BenchmarkTools
using Random

include("benchmarks.jl")

sum_wrapper(x) = _sum(identity, x)

function gradient_smoke_test(; atol = 1e-8, seed = 0xBEEF)
    rng = MersenneTwister(seed)
    x = randn(rng, 1_000)

    cache = Mooncake.prepare_gradient_cache(sum_wrapper, x)
    val, grads = Mooncake.value_and_gradient!!(cache, sum_wrapper, x)
    dx = grads[2]

    max_err = maximum(abs.(dx .- 1))
    isfinite(max_err) && max_err ≤ atol || error("Gradient check failed: max |dx-1| = $(max_err)")

    println("✓ gradient smoke test (max |dx-1| = $(max_err))")
    return nothing
end

function prep_test_case(f::F, args...) where {F}
    rule = Mooncake.build_rrule(f, args...)
    coduals = map(zero_codual, (f, args...))
    return rule, coduals
end

function to_benchmark(rule, coduals...)
    dual_args = tuple_map(x -> CoDual(primal(x), fdata(tangent(x))), coduals)
    out, pb!! = rule(dual_args...)
    return pb!!(zero_rdata(primal(out)))
end

function benchmark_smoke_test(; threshold_ns = 10_000, seed = 0xFEED)
    rng = MersenneTwister(seed)
    data = randn(rng, 1_000)

    rule, coduals = prep_test_case(_sum, identity, data)
    trial = @benchmark to_benchmark($rule, $coduals...) samples = 50 evals = 1

    median_ns = BenchmarkTools.median(trial).time
    allocs = BenchmarkTools.minimum(trial).allocs

    median_ns ≤ threshold_ns || error("Benchmark regression: median $(median_ns) ns exceeds threshold $(threshold_ns) ns")
    allocs == 0 || error("Benchmark regression: allocations detected (minimum allocs = $(allocs))")

    println("✓ benchmark smoke test (median = $(median_ns) ns)")
    return nothing
end

gradient_smoke_test()
benchmark_smoke_test()
