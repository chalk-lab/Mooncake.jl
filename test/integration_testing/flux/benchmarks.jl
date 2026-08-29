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
#   run()
#   run(; model_indices=[1], update_readme=false)

const IS_WORKER = !isempty(ARGS) && first(ARGS) == "worker"

if !IS_WORKER
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
end

using Flux, Mooncake
using Printf: @sprintf
using Random: seed!

seed!(23)
include("test_models.jl")

const BACKENDS = (:zygote, :mooncake)
const DEFAULT_BENCHMARK_SECONDS = 2.0
const README_PATH = joinpath(@__DIR__, "README.md")

if IS_WORKER
    length(ARGS) == 4 || error("invalid worker arguments: $(ARGS)")
    const WORKER_MODEL_INDEX = parse(Int, ARGS[2])
    1 <= WORKER_MODEL_INDEX <= length(TEST_MODELS) ||
        error("invalid model index: $WORKER_MODEL_INDEX")
    const BENCHMARK_INPUT = TEST_MODELS[WORKER_MODEL_INDEX][3]
    benchmark_loss(model) = sum(abs2, model(BENCHMARK_INPUT))
end

fix_sig_fig(t) = string(round(t; sigdigits=3))
function format_time(t::Float64)
    t < 1e-6 && return fix_sig_fig(t * 1e9) * " ns"
    t < 1e-3 && return fix_sig_fig(t * 1e6) * " us"
    t < 1 && return fix_sig_fig(t * 1e3) * " ms"
    return fix_sig_fig(t) * " s"
end
format_time(::Missing) = "err"

format_ratio(x::Float64) = @sprintf("%.2fx", x)
format_ratio(::Missing) = "err"

ratio(numerator::Float64, denominator::Float64) = numerator / denominator
ratio(::Any, ::Any) = missing

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
    cold = @elapsed Flux.Zygote.gradient(benchmark_loss, model)
    warm = benchmark_warm(() -> Flux.Zygote.gradient(benchmark_loss, model), seconds)
    return (; cold, warm)
end

function benchmark_backend(::Val{:mooncake}, model, seconds::Float64)
    local cache
    GC.gc(true)
    cold = @elapsed begin
        cache = Mooncake.prepare_gradient_cache(benchmark_loss, model)
        Mooncake.value_and_gradient!!(cache, benchmark_loss, model)
    end
    warm = benchmark_warm(
        () -> Mooncake.value_and_gradient!!(cache, benchmark_loss, model), seconds
    )
    return (; cold, warm)
end

function worker(model_index::Int, backend::Symbol, seconds::Float64)
    1 <= model_index <= length(TEST_MODELS) || error("invalid model index: $model_index")
    backend in BACKENDS || error("invalid backend: $backend")
    model_index == WORKER_MODEL_INDEX || error("worker model index mismatch")
    _, model, _, _ = TEST_MODELS[model_index]

    result = benchmark_backend(Val(backend), model, seconds)
    println(result.cold, '\t', result.warm)
    return nothing
end

function run_worker(model_index::Int, backend::Symbol, seconds::Float64)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__)`
    command = `$command $(@__FILE__) worker $model_index $backend $seconds`
    output = read(command, String)
    values = split(strip(output), '\t')
    length(values) == 2 || error("unexpected worker output: $(repr(output))")
    return (; cold=parse(Float64, values[1]), warm=parse(Float64, values[2]))
end

function print_results(io::IO, results)
    isempty(results) && return println(io, "No benchmark results obtained.")

    rows = map(results) do result
        zygote = result.zygote
        mooncake = result.mooncake
        return (
            name=result.name,
            zygote_cold=format_time(zygote.cold),
            mooncake_cold=format_time(mooncake.cold),
            cold_ratio=format_ratio(ratio(mooncake.cold, zygote.cold)),
            zygote_warm=format_time(zygote.warm),
            mooncake_warm=format_time(mooncake.warm),
            warm_ratio=format_ratio(ratio(mooncake.warm, zygote.warm)),
        )
    end

    name_w = max(length("Model"), maximum(textwidth(row.name) for row in rows)) + 1
    column_specs = [
        (key=:zygote_cold, label="Zygote"),
        (key=:mooncake_cold, label="Mooncake"),
        (key=:cold_ratio, label="Mc / Zyg"),
        (key=:zygote_warm, label="Zygote"),
        (key=:mooncake_warm, label="Mooncake"),
        (key=:warm_ratio, label="Mc / Zyg"),
    ]
    widths = [
        max(
            length(spec.label),
            maximum(textwidth(getproperty(row, spec.key)) for row in rows),
        ) + 2 for spec in column_specs
    ]

    gap = "  "
    cold_w = sum(widths[1:3]) + 2 * textwidth(gap)
    warm_w = sum(widths[4:6]) + 2 * textwidth(gap)
    total_w = name_w + textwidth(gap) + cold_w + textwidth(gap) + warm_w
    center(s, w) = lpad(rpad(s, div(w + textwidth(s), 2)), w)

    println(io, repeat("=", total_w))
    group_header =
        rpad("", name_w) *
        gap *
        center("first gradient", cold_w) *
        gap *
        center("warm gradient", warm_w)
    println(io, rstrip(group_header))
    println(io, rpad("", name_w) * gap * repeat("-", cold_w) * gap * repeat("-", warm_w))
    header =
        rpad("Model", name_w) *
        gap *
        join((lpad(spec.label, width) for (spec, width) in zip(column_specs, widths)), gap)
    println(io, header)
    println(io, repeat("-", total_w))
    for row in rows
        values = (getproperty(row, spec.key) for spec in column_specs)
        println(
            io,
            rpad(row.name, name_w) *
            gap *
            join((lpad(value, width) for (value, width) in zip(values, widths)), gap),
        )
    end
    println(io, repeat("=", total_w))
    return nothing
end

print_results(results) = print_results(stdout, results)

function geometric_mean_ratio(results, field::Symbol)
    ratios = Float64[]
    for result in results
        zygote_time = getproperty(result.zygote, field)
        mooncake_time = getproperty(result.mooncake, field)
        if !ismissing(zygote_time) && !ismissing(mooncake_time)
            push!(ratios, mooncake_time / zygote_time)
        end
    end
    return isempty(ratios) ? missing : exp(sum(log, ratios) / length(ratios))
end

function print_wrapped(io::IO, text::String; width::Int=100)
    line = ""
    for word in split(text)
        if isempty(line)
            line = word
        elseif length(line) + length(word) + 1 <= width
            line *= " " * word
        else
            println(io, line)
            line = word
        end
    end
    isempty(line) || println(io, line)
    return nothing
end

function write_readme(results; seconds::Float64, path::String=README_PATH)
    cold_ratio = geometric_mean_ratio(results, :cold)
    warm_ratio = geometric_mean_ratio(results, :warm)
    thread_suffix = Threads.nthreads() == 1 ? "" : "s"

    open(path, "w") do io
        println(io, "# Flux CPU gradient benchmarks")
        println(io)
        println(io, "- Julia $VERSION")
        println(io, "- Flux $(pkgversion(Flux))")
        println(io, "- Mooncake $(pkgversion(Mooncake))")
        println(io, "- $(Sys.MACHINE)")
        println(io, "- $(Threads.nthreads()) Julia thread$thread_suffix")
        println(
            io,
            "- a $(fix_sig_fig(seconds))-second warm sampling budget per model and backend",
        )
        println(io)
        print_wrapped(
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

        if !ismissing(cold_ratio) && !ismissing(warm_ratio)
            println(io)
            print_wrapped(
                io,
                "Across the $(length(results)) models, the geometric-mean Mooncake/Zygote " *
                "ratio is $(@sprintf("%.2f", cold_ratio)) for first-gradient time and " *
                "$(@sprintf("%.3f", warm_ratio)) for warm-gradient time. Equivalently, " *
                "Mooncake is $(@sprintf("%.2f", inv(warm_ratio))) times faster on warm " *
                "gradients by geometric mean.",
            )
        end
    end
    return path
end

"""
    run(; model_indices=eachindex(TEST_MODELS), seconds=2.0, update_readme=true)

Compare Zygote and Mooncake on every CPU model in `TEST_MODELS`. Each first-gradient
measurement runs in a fresh Julia process after packages and model values have loaded. The
Mooncake measurement includes both `prepare_gradient_cache` and the first
`value_and_gradient!!` call. Warm measurements reuse the prepared cache and report the median
sample collected over `seconds`.
"""
function run(;
    model_indices=eachindex(TEST_MODELS),
    seconds::Float64=DEFAULT_BENCHMARK_SECONDS,
    update_readme::Bool=true,
)
    indices = collect(Int, model_indices)
    all(index -> 1 <= index <= length(TEST_MODELS), indices) ||
        error("model indices must be between 1 and $(length(TEST_MODELS))")
    results = []
    total = length(indices) * length(BACKENDS)
    benchmark_index = 0
    for model_index in indices
        _, _, _, name = TEST_MODELS[model_index]
        timings = Dict{Symbol,NamedTuple}()
        for backend in BACKENDS
            benchmark_index += 1
            @info "$benchmark_index / $total", name, backend
            timings[backend] = try
                run_worker(model_index, backend, seconds)
            catch error
                @info "  errored: $(sprint(showerror, error))"
                (; cold=missing, warm=missing)
            end
            @info "  first gradient = $(format_time(timings[backend].cold))"
            @info "  warm gradient  = $(format_time(timings[backend].warm))"
        end
        push!(results, (; name, zygote=timings[:zygote], mooncake=timings[:mooncake]))
    end

    print_results(results)
    if update_readme
        path = write_readme(results; seconds)
        @info "updated benchmark report", path
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    if IS_WORKER
        worker(parse(Int, ARGS[2]), Symbol(ARGS[3]), parse(Float64, ARGS[4]))
    elseif isempty(ARGS)
        run()
    elseif ARGS == ["--stdout-only"]
        run(; update_readme=false)
    elseif length(ARGS) == 1 && tryparse(Int, only(ARGS)) !== nothing
        run(; model_indices=[parse(Int, only(ARGS))], update_readme=false)
    else
        error("invalid arguments: $(ARGS)")
    end
end
