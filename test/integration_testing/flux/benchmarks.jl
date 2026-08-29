#!/usr/bin/env julia
#
# Benchmark Zygote vs Mooncake CPU gradient evaluation on the Flux integration models.
#
# Usage (from test/integration_testing/flux):
#   julia --project=. benchmarks.jl                # all models; update README.md
#   julia --project=. benchmarks.jl 1              # model 1 only; print to stdout
#   julia --project=. benchmarks.jl --stdout-only  # all models; do not update README.md
#
# First-gradient timings run in isolated Julia processes. Warm Mooncake timings reuse the
# prepared gradient cache. Both backends use the same model and input realization.
#
# From the REPL:
#   include("benchmarks.jl")
#   results = run()
#   write_readme(results)
#   run(; model_indices=[1])

const IS_WORKER = !isempty(ARGS) && first(ARGS) == "worker"

if !IS_WORKER
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
end

using Flux, Mooncake
using LinearAlgebra: BLAS
using Printf: @sprintf
using Random: seed!

seed!(23)
include("models.jl")

const BACKENDS = (:zygote, :mooncake)
const DEFAULT_BENCHMARK_SECONDS = 2.0
const README_PATH = joinpath(@__DIR__, "README.md")

if IS_WORKER
    length(ARGS) == 4 || error("invalid worker arguments: $(ARGS)")
    const WORKER_MODEL_INDEX = parse(Int, ARGS[2])
    1 <= WORKER_MODEL_INDEX <= length(FLUX_MODELS) ||
        error("invalid model index: $WORKER_MODEL_INDEX")
    const BENCHMARK_INPUT = FLUX_MODELS[WORKER_MODEL_INDEX][3]
    benchmark_loss(model) = sum(abs2, model(BENCHMARK_INPUT))
end

fix_sig_fig(t) = string(round(t; sigdigits=3))
function format_time(t::Float64)
    t < 1e-6 && return fix_sig_fig(t * 1e9) * " ns"
    t < 1e-3 && return fix_sig_fig(t * 1e6) * " us"
    t < 1 && return fix_sig_fig(t * 1e3) * " ms"
    return fix_sig_fig(t) * " s"
end

function format_ratio(numerator::Float64, denominator::Float64)
    @sprintf("%.2fx", numerator / denominator)
end

function median_time(samples::Vector{Float64})
    sort!(samples)
    n = length(samples)
    middle = div(n, 2)
    return isodd(n) ? samples[middle + 1] : (samples[middle] + samples[middle + 1]) / 2
end

function benchmark_warm(f, seconds::Float64)
    f()
    f()
    samples = Float64[]
    start = time_ns()
    budget = round(UInt64, seconds * 1e9)
    while length(samples) < 10 || time_ns() - start < budget
        push!(samples, @elapsed f())
    end
    return median_time(samples)
end

function benchmark_backend(::Val{:zygote}, model, seconds::Float64)
    GC.gc(true)
    first_gradient = @elapsed Flux.Zygote.gradient(benchmark_loss, model)
    warm = benchmark_warm(() -> Flux.Zygote.gradient(benchmark_loss, model), seconds)
    return (; first_gradient, warm)
end

function benchmark_backend(::Val{:mooncake}, model, seconds::Float64)
    local cache
    GC.gc(true)
    first_gradient = @elapsed begin
        cache = Mooncake.prepare_gradient_cache(benchmark_loss, model)
        Mooncake.value_and_gradient!!(cache, benchmark_loss, model)
    end
    warm = benchmark_warm(
        () -> Mooncake.value_and_gradient!!(cache, benchmark_loss, model), seconds
    )
    return (; first_gradient, warm)
end

function worker(model_index::Int, backend::Symbol, seconds::Float64)
    1 <= model_index <= length(FLUX_MODELS) || error("invalid model index: $model_index")
    backend in BACKENDS || error("invalid backend: $backend")
    model_index == WORKER_MODEL_INDEX || error("worker model index mismatch")
    _, model, _, _ = FLUX_MODELS[model_index]

    result = benchmark_backend(Val(backend), model, seconds)
    println(result.first_gradient, '\t', result.warm)
    return nothing
end

function run_worker(model_index::Int, backend::Symbol, seconds::Float64)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__)`
    command = `$command $(@__FILE__) worker $model_index $backend $seconds`
    output = read(command, String)
    values = split(strip(output), '\t')
    length(values) == 2 || error("unexpected worker output: $(repr(output))")
    return (; first_gradient=parse(Float64, values[1]), warm=parse(Float64, values[2]))
end

function print_results(io::IO, results)
    isempty(results) && return println(io, "No benchmark results obtained.")

    name_w = maximum(textwidth(result.name) for result in results) + 2
    column_w = 11
    group_w = 3 * column_w
    gap = "  "
    total_w = name_w + 2 * group_w + 2 * textwidth(gap)
    center(s, w) = lpad(rpad(s, div(w + textwidth(s), 2)), w)

    println(io, repeat("=", total_w))
    group_header = string(
        rpad("", name_w),
        gap,
        center("first gradient", group_w),
        gap,
        center("warm gradient", group_w),
    )
    println(io, rstrip(group_header))
    println(io, rpad("", name_w), gap, repeat("-", group_w), gap, repeat("-", group_w))
    labels = ("Zygote", "Mooncake", "Mc / Zyg", "Zygote", "Mooncake", "Mc / Zyg")
    println(
        io,
        rpad("Model", name_w),
        gap,
        join(lpad.(labels[1:3], column_w)),
        gap,
        join(lpad.(labels[4:6], column_w)),
    )
    println(io, repeat("-", total_w))
    for result in results
        values = (
            format_time(result.zygote.first_gradient),
            format_time(result.mooncake.first_gradient),
            format_ratio(result.mooncake.first_gradient, result.zygote.first_gradient),
            format_time(result.zygote.warm),
            format_time(result.mooncake.warm),
            format_ratio(result.mooncake.warm, result.zygote.warm),
        )
        println(
            io,
            rpad(result.name, name_w),
            gap,
            join(lpad.(values[1:3], column_w)),
            gap,
            join(lpad.(values[4:6], column_w)),
        )
    end
    println(io, repeat("=", total_w))
    return nothing
end

print_results(results) = print_results(stdout, results)

function geometric_mean_ratio(results, field::Symbol)
    log_ratios = map(results) do result
        return log(getproperty(result.mooncake, field) / getproperty(result.zygote, field))
    end
    return exp(sum(log_ratios) / length(log_ratios))
end

function cpu_models()
    names = unique(info.model for info in Sys.cpu_info())
    lscpu = Sys.which("lscpu")
    if names == ["unknown"] && !isnothing(lscpu)
        lines = filter(line -> startswith(line, "Model name:"), readlines(`$lscpu`))
        names = unique(strip(last(split(line, ':'; limit=2))) for line in lines)
    end
    return join(sort!(names), " + ")
end

function write_readme(
    results; seconds::Float64=DEFAULT_BENCHMARK_SECONDS, path::String=README_PATH
)
    first_gradient_ratio = geometric_mean_ratio(results, :first_gradient)
    warm_ratio = geometric_mean_ratio(results, :warm)
    thread_suffix = Threads.nthreads() == 1 ? "" : "s"

    open(path, "w") do io
        println(io, "# Flux CPU gradient benchmarks")
        println(io)
        println(
            io,
            "- Julia $VERSION, Flux $(pkgversion(Flux)), Mooncake $(pkgversion(Mooncake))",
        )
        println(io, "- $(Sys.MACHINE)")
        println(io, "- CPU: $(cpu_models()) ($(Sys.CPU_THREADS) logical threads)")
        println(io, "- $(Threads.nthreads()) Julia thread$thread_suffix")
        println(io, "- BLAS: $(BLAS.vendor()), $(BLAS.get_num_threads()) thread(s)")
        println(
            io,
            "- a $(fix_sig_fig(seconds))-second warm sampling budget per model and backend",
        )
        println(io)
        println(
            io,
            "Each first-gradient measurement ran in a fresh Julia process after packages " *
            "and model values had loaded. Mooncake's first-gradient time includes " *
            "`prepare_gradient_cache` and the first `value_and_gradient!!` call. Warm " *
            "Mooncake measurements reuse the prepared cache. `Mc / Zyg` is Mooncake time " *
            "divided by Zygote time, so values below one favour Mooncake.",
        )
        println(io)
        println(io, "```text")
        print_results(io, results)
        println(io, "```")

        println(io)
        return println(
            io,
            "Across the $(length(results)) models, the geometric-mean Mooncake/Zygote " *
            "ratio is $(@sprintf("%.2f", first_gradient_ratio)) for first-gradient time " *
            "and $(@sprintf("%.3f", warm_ratio)) for warm-gradient time. Equivalently, " *
            "Mooncake is $(@sprintf("%.2f", inv(warm_ratio))) times faster on warm " *
            "gradients by geometric mean.",
        )
    end
    return path
end

function run(;
    model_indices=eachindex(FLUX_MODELS), seconds::Float64=DEFAULT_BENCHMARK_SECONDS
)
    indices = collect(Int, model_indices)
    all(index -> 1 <= index <= length(FLUX_MODELS), indices) ||
        error("model indices must be between 1 and $(length(FLUX_MODELS))")
    results = []
    total = length(indices) * length(BACKENDS)
    benchmark_index = 0
    for model_index in indices
        _, _, _, name = FLUX_MODELS[model_index]
        timings = Dict{Symbol,NamedTuple}()
        for backend in BACKENDS
            benchmark_index += 1
            @info "$benchmark_index / $total", name, backend
            timings[backend] = run_worker(model_index, backend, seconds)
            @info "  first gradient = $(format_time(timings[backend].first_gradient))"
            @info "  warm gradient  = $(format_time(timings[backend].warm))"
        end
        push!(results, (; name, zygote=timings[:zygote], mooncake=timings[:mooncake]))
    end

    print_results(results)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    if IS_WORKER
        worker(parse(Int, ARGS[2]), Symbol(ARGS[3]), parse(Float64, ARGS[4]))
    elseif isempty(ARGS)
        path = write_readme(run())
        @info "updated benchmark report", path
    elseif ARGS == ["--stdout-only"]
        run()
    elseif length(ARGS) == 1 && tryparse(Int, only(ARGS)) !== nothing
        run(; model_indices=[parse(Int, only(ARGS))])
    else
        error("invalid arguments: $(ARGS)")
    end
end
