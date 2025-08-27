using Pkg
Pkg.develop(; path=joinpath(@__DIR__, ".."))

using AbstractGPs,
    Chairmarks,
    CSV,
    DataFrames,
    Distributions,
    DynamicPPL,
    Enzyme,
    KernelFunctions,
    LinearAlgebra,
    Plots,
    PrettyTables,
    Random,
    ReverseDiff,
    Mooncake,
    Test,
    Zygote

using Mooncake:
    Dual,
    CoDual,
    generate_hand_written_rrule!!_test_cases,
    generate_derived_rrule!!_test_cases,
    TestUtils,
    _typeof,
    primal,
    tangent,
    zero_dual,
    zero_codual

using Mooncake.TestUtils: _deepcopy

to_benchmark(__frule!!::R, dx::Vararg{Dual,N}) where {R,N} = __frule!!(dx...)

function to_benchmark(__rrule!!::R, dx::Vararg{CoDual,N}) where {R,N}
    dx_f = Mooncake.tuple_map(x -> CoDual(primal(x), Mooncake.fdata(tangent(x))), dx)
    out, pb!! = __rrule!!(dx_f...)
    return pb!!(Mooncake.zero_rdata(primal(out)))
end

function zygote_to_benchmark(ctx, x::Vararg{Any,N}) where {N}
    out, pb = Zygote._pullback(ctx, x...)
    return pb(out)
end

function rd_to_benchmark!(result, tape, x)
    return ReverseDiff.gradient!(result, tape, x)
end

should_run_benchmark(args...) = true

# Test out the performance of a hand-written sum function, so we can be confident that there
# is no rule. Note that ReverseDiff has a (seemingly not fantastic) hand-written rule for
# sum.
function _sum(f::F, x::AbstractArray{<:Real}) where {F}
    y = 0.0
    n = 0
    while n < length(x)
        n += 1
        y += f(x[n])
    end
    return y
end

# Zygote has rules for both sum and kron, so it's interesting to compare this against the
# other frameworks because they don't have rules for kron, and maybe for not for sum.
_kron_sum(x::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real}) = sum(kron(x, y))

# Zygote (should) use the ChainRules projection functionality to handle the more interesting
# types surrounding the arrays.
function _kron_view_sum(x::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real})
    return _kron_sum(view(x, 1:20, 1:20), UpperTriangular(y))
end

# No one has a rule for this.
function _naive_map_sin_cos_exp(x::AbstractArray{<:Real})
    y = similar(x)
    for n in eachindex(x)
        y[n] = sin(cos(exp(x[n])))
    end
    return sum(y)
end

should_run_benchmark(::Val{:zygote}, ::typeof(_naive_map_sin_cos_exp), x) = false

# RD and Zygote have a rule for this.
_map_sin_cos_exp(x::AbstractArray{<:Real}) = sum(map(x -> sin(cos(exp(x))), x))

# Only Zygote has a rule for this.
_broadcast_sin_cos_exp(x::AbstractArray{<:Real}) = sum(sin.(cos.(exp.(x))))

# Different frameworks have rules for this to differing degrees. Zygote has rules for just
# about all of the operations.
_simple_mlp(W2, W1, Y, X) = sum(abs2, Y - W2 * map(x -> x * (0 <= x), W1 * X))

# Only Zygote and Mooncake can actually handle this. Note that Mooncake only has rules for
# BLAS and LAPACK stuff, not explicit rules for things like the squared euclidean distance.
# Consequently, Zygote is at a major advantage.
_gp_lml(x, y, s) = logpdf(GP(SEKernel())(x, s), y)

should_run_benchmark(::Val{:reverse_diff}, ::typeof(_gp_lml), x...) = false

function _generate_gp_inputs()
    x = collect(range(0.0; step=0.2, length=128))
    s = 1.0
    y = rand(GP(SEKernel())(x, s))
    return x, y, s
end

@model broadcast_demo(x) = begin
    μ ~ truncated(Normal(1, 2), 0.1, 10)
    σ ~ truncated(Normal(1, 2), 0.1, 10)
    x .~ LogNormal(μ, σ)
end

function build_dynamicppl_problem()
    rng = Xoshiro(123)
    model = broadcast_demo(rand(LogNormal(1.5, 0.5), 100_000))
    vi = DynamicPPL.SimpleVarInfo(model)
    vi_linked = DynamicPPL.link(vi, model)
    ldp = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi_linked)
    test_function = Base.Fix1(DynamicPPL.LogDensityProblems.logdensity, ldp)
    d = DynamicPPL.LogDensityProblems.dimension(ldp)
    return test_function, randn(rng, d)
end

run_dynamicppl_problem(f::F, x::X) where {F,X} = f(x)

function should_run_benchmark(
    ::Val{:zygote}, ::Base.Fix1{<:typeof(DynamicPPL.LogDensityProblems.logdensity)}, x...
)
    return false
end
function should_run_benchmark(
    ::Val{:enzyme}, ::Base.Fix1{<:typeof(DynamicPPL.LogDensityProblems.logdensity)}, x...
)
    return false
end

@inline g(x, a, ::Val{N}) where {N} = N > 0 ? g(x * a, a, Val(N - 1)) : x

large_single_block(x::AbstractVector{<:Real}) = g(x[1], x[2], Val(400))

"""
    generate_inter_framework_tests()

Constructs a set of benchmarks which can be used to compare between AD frameworks.
Outputs a vector of tuples. Each tuples comprises a function (first element) and arguments
at which its value and pullback should be computed (remaining elements).

Arguments must comprise only scalars and arrays, and output must be either a scalar or
an array.
"""
function generate_inter_framework_tests()
    return Any[
        ("sum_1000", (sum, randn(1_000))),
        ("_sum_1000", (x -> _sum(identity, x), randn(1_000))),
        ("sum_sin_1000", (x -> sum(sin, x), randn(1_000))),
        ("_sum_sin_1000", (x -> _sum(sin, x), randn(1_000))),
        ("kron_sum", (_kron_sum, randn(20, 20), randn(40, 40))),
        ("kron_view_sum", (_kron_view_sum, randn(40, 30), randn(40, 40))),
        ("naive_map_sin_cos_exp", (_naive_map_sin_cos_exp, randn(10, 10))),
        ("map_sin_cos_exp", (_map_sin_cos_exp, randn(10, 10))),
        ("broadcast_sin_cos_exp", (_broadcast_sin_cos_exp, randn(10, 10))),
        (
            "simple_mlp",
            (_simple_mlp, randn(128, 256), randn(256, 128), randn(128, 70), randn(128, 70)),
        ),
        ("gp_lml", (_gp_lml, _generate_gp_inputs()...)),
        ("turing_broadcast_benchmark", build_dynamicppl_problem()),
        ("large_single_block", (large_single_block, [0.9, 0.99])),
    ]
end

function benchmark_rules!!(test_case_data, default_ratios, include_other_frameworks::Bool)
    test_cases = reduce(vcat, map(first, test_case_data))
    memory = map(x -> x[2], test_case_data)
    ranges = reduce(vcat, map(x -> x[3], test_case_data))
    tags = reduce(vcat, map(x -> x[4], test_case_data))
    GC.@preserve memory begin
        return map(enumerate(test_cases)) do (n, args)
            @info "$n / $(length(test_cases))", _typeof(args)
            suite = Dict()

            # Benchmark primal.
            @info "Primal"
            primals = map(x -> x isa CoDual ? primal(x) : x, args)
            include_other_frameworks && GC.gc(true)
            suite["primal"] = Chairmarks.benchmark(
                () -> primals,
                primals -> (primals[1], _deepcopy(primals[2:end])),
                (a -> a[1]((a[2]...))),
                _ -> true;
                evals=1,
            )

            # Benchmark AD via Mooncake.
            @info "Mooncake"
            rule = Mooncake.build_rrule(args...)
            coduals = map(x -> x isa CoDual ? x : zero_codual(x), args)
            to_benchmark(rule, coduals...)
            include_other_frameworks && GC.gc(true)
            suite["mooncake"] = Chairmarks.benchmark(
                () -> (rule, coduals),
                identity,
                a -> to_benchmark(a[1], a[2]...),
                _ -> true;
                evals=1,
            )

            # Benchmark AD via Mooncake.
            @info "Mooncake (Forward)"
            rule = Mooncake.build_frule(args...)
            duals = map(x -> x isa CoDual ? Dual(x.x, x.dx) : zero_dual(x), args)
            to_benchmark(rule, duals...)
            include_other_frameworks && GC.gc(true)
            suite["mooncake_fwd"] = Chairmarks.benchmark(
                () -> (rule, duals),
                identity,
                a -> to_benchmark(a[1], a[2]...),
                _ -> true;
                evals=1,
            )

            if include_other_frameworks
                if should_run_benchmark(Val(:zygote), args...)
                    @info "Zygote"
                    GC.gc(true)
                    suite["zygote"] = @be(
                        _,
                        _,
                        zygote_to_benchmark($(Zygote.Context()), $primals...),
                        _,
                        evals = 1,
                    )
                end

                if should_run_benchmark(Val(:reverse_diff), args...)
                    @info "ReverseDiff"
                    tape = ReverseDiff.GradientTape(primals[1], primals[2:end])
                    compiled_tape = ReverseDiff.compile(tape)
                    result = map(x -> randn(size(x)), primals[2:end])
                    GC.gc(true)
                    suite["rd"] = @be(
                        _,
                        _,
                        rd_to_benchmark!($result, $compiled_tape, $primals[2:end]),
                        _,
                        evals = 1,
                    )
                end

                if should_run_benchmark(Val(:enzyme), args...)
                    @info "Enzyme"
                    _rand_similiar(x) = x isa Real ? randn() : randn(size(x))
                    dup_args = map(x -> Duplicated(x, _rand_similiar(x)), primals[2:end])
                    GC.gc(true)
                    suite["enzyme"] = @be(
                        _,
                        _,
                        autodiff(ReverseWithPrimal, $primals[1], Active, $dup_args...),
                        _,
                        evals = 1,
                    )
                end
            end

            return combine_results((args, suite), tags[n], ranges[n], default_ratios)
        end
    end
end

function combine_results(result, tag, _range, default_range)
    d = result[2]
    primal_time = minimum(d["primal"]).time
    mooncake_time = minimum(d["mooncake"]).time
    mooncake_fwd_time = minimum(d["mooncake_fwd"]).time
    zygote_time = in("zygote", keys(d)) ? minimum(d["zygote"]).time : missing
    rd_time = in("rd", keys(d)) ? minimum(d["rd"]).time : missing
    ez_time = in("enzyme", keys(d)) ? minimum(d["enzyme"]).time : missing
    fallback_tag = string((result[1][1], map(Mooncake._typeof, result[1][2:end])...))
    return (
        tag=tag === nothing ? fallback_tag : tag,
        primal_time=primal_time,
        mooncake_time=mooncake_time,
        Mooncake=mooncake_time / primal_time,
        mooncake_fwd_time=mooncake_fwd_time,
        MooncakeFwd=mooncake_fwd_time / primal_time,
        zygote_time=zygote_time,
        Zygote=zygote_time / primal_time,
        rd_time=rd_time,
        ReverseDiff=rd_time / primal_time,
        enzyme_time=ez_time,
        Enzyme=ez_time / primal_time,
        range=_range === nothing ? default_range : _range,
    )
end

function benchmark_hand_written_rrules!!(rng_ctor)
    test_case_data = map([
        :avoiding_non_differentiable_code,
        :blas,
        :builtins,
        :fastmath,
        :foreigncall,
        :iddict,
        :lapack,
        :low_level_maths,
        :misc,
        :new,
    ]) do s
        test_cases, memory = generate_hand_written_rrule!!_test_cases(rng_ctor, Val(s))
        ranges = map(x -> x[3], test_cases)
        tags = fill(nothing, length(test_cases))
        return map(x -> x[4:end], test_cases), memory, ranges, tags
    end
    return benchmark_rules!!(test_case_data, (lb=1e-3, ub=50.0), false)
end

function benchmark_derived_rrules!!(rng_ctor)
    test_case_data = map([:test_resources]) do s
        test_cases, memory = generate_derived_rrule!!_test_cases(rng_ctor, Val(s))
        ranges = map(x -> x[3], test_cases)
        tags = fill(nothing, length(test_cases))
        return map(x -> x[4:end], test_cases), memory, ranges, tags
    end
    return benchmark_rules!!(test_case_data, (lb=1e-3, ub=200), false)
end

function benchmark_inter_framework_rules()
    test_case_data = generate_inter_framework_tests()
    tags = map(first, test_case_data)
    test_cases = map(last, test_case_data)
    memory = []
    ranges = fill(nothing, length(test_cases))
    return benchmark_rules!!([(test_cases, memory, ranges, tags)], (lb=0.1, ub=200), true)
end

function flag_concerning_performance(ratios)
    @testset "detect concerning performance" begin
        @testset for ratio in ratios
            @test ratio.range.lb < ratio.Mooncake < ratio.range.ub
        end
    end
end

"""
    plot_ratio_histogram!(df::DataFrame)

Constructs a histogram of the `mooncake_ratio` field of `df`, with formatting that is
well-suited to the numbers typically found in this field.
"""
function plot_ratio_histogram!(df::DataFrame)
    bin = 10.0 .^ (-1.0:0.05:4.0)
    xlim = extrema(bin)
    return histogram(df.Mooncake; xscale=:log10, xlim, bin, title="log", label="")
end

fix_sig_fig(t) = string.(round(t; sigdigits=3))

function format_time(t::Float64)
    t < 1e-6 && return fix_sig_fig(t * 1e9) * " ns"
    t < 1e-3 && return fix_sig_fig(t * 1e6) * " μs"
    t < 1 && return fix_sig_fig(t * 1e3) * " ms"
    return fix_sig_fig(t) * " s"
end

function create_inter_ad_benchmarks()
    results = benchmark_inter_framework_rules()
    tools = [:Mooncake, :MooncakeFwd, :Zygote, :ReverseDiff, :Enzyme]
    df = DataFrame(results)[:, [:tag, :primal_time, tools...]]

    # Plot graph of results.
    plt = plot(; yscale=:log10, legend=:topright, title="AD Time / Primal Time (Log Scale)")
    for label in string.(tools)
        plot!(plt, df.tag, df[:, label]; label, marker=:circle, xrotation=45)
    end
    Plots.savefig(plt, "bench/benchmark_results.png")

    # Write table of results.
    formatted_ts = format_time.(df.primal_time)
    formatted_cols = map(t -> t => fix_sig_fig.(df[:, t]), tools)
    df_formatted = DataFrame(:Label => df.tag, :Primal => formatted_ts, formatted_cols...)
    return open(
        io -> pretty_table(io, df_formatted), "bench/benchmark_results.txt"; write=true
    )
end

function main()
    perf_group = get(ENV, "PERF_GROUP", "hand_written")
    @info perf_group
    println(perf_group)
    if perf_group == "hand_written"
        flag_concerning_performance(benchmark_hand_written_rrules!!(Xoshiro))
    elseif perf_group == "derived"
        flag_concerning_performance(benchmark_derived_rrules!!(Xoshiro))
    elseif perf_group == "comparison"
        create_inter_ad_benchmarks()
    else
        throw(error("perf_group=$(perf_group) is not recognised"))
    end
end
