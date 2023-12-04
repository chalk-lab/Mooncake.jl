using Pkg
Pkg.develop(path=joinpath(@__DIR__, ".."))

using BenchmarkTools, Random, Taped, Test

using Taped:
    CoDual,
    generate_hand_written_rrule!!_test_cases,
    generate_derived_rrule!!_test_cases

using Taped.TestUtils: _deepcopy

function generate_rrule!!_benchmarks(rng::AbstractRNG, args)

    # Generate CoDuals etc.
    primals = map(x -> x isa CoDual ? primal(x) : x, args)
    dargs = map(x -> x isa CoDual ? tangent(x) : randn_tangent(rng, x), args)
    cd_args = map(CoDual, primals, dargs)
    suite = BenchmarkGroup()

    # Benchmark primal.
    suite["primal"] = @benchmarkable(
        (a[1])((a[2:end])...);
        setup=(a = ($primals[1], _deepcopy($primals[2:end])...)),
        evals=1,    
    )

    # Benchmark forwards-pass.
    suite["forwards"] = @benchmarkable(
        Taped.rrule!!(ca...);
        setup=(ca = ($cd_args[1], _deepcopy($(cd_args)[2:end])...)),
        evals=1,
    )

    # Benchmark pullback.
    suite["pullback"] = @benchmarkable(
        x[2]((tangent(x[1])), map(tangent, ca)...),
        setup=(ca = ($cd_args[1], _deepcopy($cd_args[2:end])...); x = Taped.rrule!!(ca...)),
        evals=1,
    )

    return suite
end

function generate_hand_written_cases(rng_ctor, v::Val)
    test_cases, memory = generate_hand_written_rrule!!_test_cases(rng_ctor, v)
    ranges = map(x -> x[3], test_cases)
    return map(x -> x[4:end], test_cases), memory, ranges
end

default_hand_written_ratios() = (lb=1e-3, ub=10.0)

function benchmark_hand_written_rrules!!(rng_ctor)
    rng = rng_ctor(123)

    # Benchmark the performance of all benchmarks.
    test_case_data = map([
        :avoiding_non_differentiable_code,
        :blas,
        :builtins,
        :foreigncall,
        :iddict,
        :lapack,
        :low_level_maths,
        :misc,
        :umlaut_internals_rules,
        :unrolled_function
    ]) do s
        generate_hand_written_cases(Xoshiro, Val(s))
    end
    test_cases = reduce(vcat, map(first, test_case_data))
    memory = map(x -> x[2], test_case_data)
    ranges = reduce(vcat, map(x -> x[3], test_case_data))

    GC.@preserve memory begin
        results = map(enumerate(test_cases)) do (n, x)
            @info "$n / $(length(test_cases))", x
            suite = generate_rrule!!_benchmarks(rng, x)
            return (x, BenchmarkTools.run(suite; verbose=true, seconds=3))
        end
    end

    # Compute performance ratio for all cases.
    return combine_results.(results, ranges, Ref(default_hand_written_ratios()))
end

default_derived_ratios() = (lb=100, ub=100_000)

function benchmark_derived_rrules!!(rng_ctor)
    rng = rng_ctor(123)

    # We only run a subset of the test cases, because it will take far too long to run
    # then all. Moreover, it seems unlikely that there is presently a need for all tests,
    # and that the :unrolled_function tests will be sufficient.
    test_case_data = map([
        # :avoiding_non_differentiable_code,
        # :blas,
        # :builtins,
        # :foreigncall,
        # :iddict,
        # :lapack,
        # :low_level_maths,
        # :misc,
        # :umlaut_internals_rules,
        :unrolled_function
    ]) do s
        test_cases, memory = generate_derived_rrule!!_test_cases(rng_ctor, Val(s))
        unrolled_test_cases = map(test_cases) do test_case
            f, x... = test_case[3:end]
            f_t = last(Taped.trace_recursive_tape!!(f, map(_deepcopy, x)...))
            return Any[f_t, f, x...]
        end
        ranges = map(x -> x[2], test_cases)
        return unrolled_test_cases, memory, ranges
    end
    test_cases = reduce(vcat, map(first, test_case_data))
    memory = map(x -> x[2], test_case_data)
    ranges = reduce(vcat, map(x -> x[3], test_case_data))

    GC.@preserve memory begin
        results = map(enumerate(test_cases)) do (n, args)
            @info "$n / $(length(test_cases))", args

            # Generate CoDuals etc.
            primals = map(x -> x isa CoDual ? primal(x) : x, args[2:end])
            unrolled_primals = map(x -> x isa CoDual ? primal(x) : x, args)
            dargs = map(x -> x isa CoDual ? tangent(x) : randn_tangent(rng, x), args)
            cd_args = map(CoDual, unrolled_primals, dargs)
            suite = BenchmarkGroup()

            # Benchmark primal.
            suite["primal"] = @benchmarkable(
                (a[1])((a[2:end])...);
                setup=(a = ($primals[1], _deepcopy($primals[2:end])...)),
                evals=1,    
            )

            # Benchmark forwards-pass.
            suite["forwards"] = @benchmarkable(
                Taped.rrule!!(ca...);
                setup=(ca = ($cd_args[1], _deepcopy($(cd_args)[2:end])...)),
                evals=1,
            )

            # Benchmark pullback.
            suite["pullback"] = @benchmarkable(
                x[2]((tangent(x[1])), map(tangent, ca)...),
                setup=(ca = ($cd_args[1], _deepcopy($cd_args[2:end])...); x = Taped.rrule!!(ca...)),
                evals=1,
            )

            return (args, BenchmarkTools.run(suite; verbose=true, seconds=3))
        end
    end

    # Compute performance ratio for all cases.
    return combine_results.(results, ranges, Ref(default_derived_ratios()))
end

function combine_results(result, _range, default_range)
    result_dict = result[2]
    primal_time = time(minimum(result_dict["primal"]))
    forwards_time = time(minimum(result_dict["forwards"]))
    pullback_time = time(minimum(result_dict["pullback"]))
    return (
        tag=result[1],
        forwards_range=_range === nothing ? default_range : _range,
        pullback_range=_range === nothing ? default_range : _range,
        primal_time=primal_time,
        forwards_time=forwards_time,
        pullback_time=pullback_time,
        forwards_ratio=forwards_time / primal_time,
        pullback_ratio=pullback_time / primal_time,
    )
end

between(x, (lb, ub)) = lb < x && x < ub

function flag_concerning_performance(ratios)
    @testset "detect concerning performance" begin
        @testset for ratio in ratios
            @test between(ratio.forwards_ratio, ratio.forwards_range)
            @test between(ratio.pullback_ratio, ratio.pullback_range)
        end
    end
end

const perf_group = get(ENV, "PERF_GROUP", "derived")

if perf_group == "hand_written"
    flag_concerning_performance(benchmark_hand_written_rrules!!(Xoshiro))
elseif perf_group == "derived"
    flag_concerning_performance(benchmark_derived_rrules!!(Xoshiro))
else
    throw(error("perf_group=$(perf_group) is not recognised"))
end
