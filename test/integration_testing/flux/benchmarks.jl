#!/usr/bin/env julia
#
# Benchmark Zygote vs Mooncake CPU and GPU gradients on the Flux integration models.
#
# Usage (from test/integration_testing/flux):
#   julia --project=. benchmarks.jl         # all models/devices; update README.md
#   julia --project=. benchmarks.jl cpu     # all CPU models
#   julia --project=. benchmarks.jl gpu 1   # model 1 on the GPU
#
# First-gradient timings run in isolated Julia processes. Warm Mooncake timings reuse the
# prepared gradient cache. Both backends use the same model and input realization, and
# compute gradients with respect to both so that Zygote's lazy input cotangents are forced.
#
# From the REPL:
#   include("benchmarks.jl")
#   cpu_results, gpu_results = run_all()
#   run_all(; model_indices=[1])

const IS_WORKER = !isempty(ARGS) && first(ARGS) == "worker"
const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

if !IS_WORKER
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(; path=REPOSITORY_ROOT)
end

using CUDA, cuDNN, Flux, Mooncake
using LinearAlgebra: BLAS
using Printf: @sprintf
using Random: seed!

seed!(23)
include("models.jl")

const BACKENDS = (:zygote, :mooncake)
const DEVICES = (:cpu, :gpu)
const DEFAULT_BENCHMARK_SECONDS = 2.0
const BENCHMARK_WORKLOAD_MULTIPLIER = 32
const README_PATH = joinpath(@__DIR__, "README.md")
BLAS.set_num_threads(1)

function benchmark_input(model_index::Int)
    _, model, input, _ = FLUX_MODELS[model_index]
    if input isa AbstractVector
        return if model isa Flux.Scale
            repeat(reshape(input, :, 1, 1), 1, 1, BENCHMARK_WORKLOAD_MULTIPLIER)
        else
            repeat(input, 1, BENCHMARK_WORKLOAD_MULTIPLIER)
        end
    end
    outer = ntuple(
        dim -> dim == ndims(input) ? BENCHMARK_WORKLOAD_MULTIPLIER : 1, ndims(input)
    )
    return repeat(input; outer)
end

if IS_WORKER
    length(ARGS) == 5 || error("invalid worker arguments: $(ARGS)")
    const WORKER_MODEL_INDEX = parse(Int, ARGS[2])
    1 <= WORKER_MODEL_INDEX <= length(FLUX_MODELS) ||
        error("invalid model index: $WORKER_MODEL_INDEX")
    const WORKER_DEVICE = Symbol(ARGS[4])
    const BENCHMARK_INPUT = if WORKER_DEVICE === :gpu
        cu(benchmark_input(WORKER_MODEL_INDEX))
    else
        benchmark_input(WORKER_MODEL_INDEX)
    end
    benchmark_loss(model, input) = sum(abs2, model(input))
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

sync(::Val{:cpu}) = nothing
sync(::Val{:gpu}) = CUDA.synchronize()

function check_seconds(seconds::Float64)
    isfinite(seconds) && seconds > 0 ||
        throw(ArgumentError("benchmark duration must be finite and positive"))
    return nothing
end

function benchmark_warm(f, device::Val, seconds::Float64)
    f()
    f()
    sync(device)
    samples = Float64[]
    start = time_ns()
    budget = round(UInt64, seconds * 1e9)
    while length(samples) < 10 || time_ns() - start < budget
        push!(samples, @elapsed begin
            f()
            sync(device)
        end)
    end
    return median_time(samples)
end

function benchmark_backend(::Val{:zygote}, model, device::Val, seconds::Float64)
    GC.gc(true)
    first_gradient = @elapsed begin
        Flux.Zygote.gradient(benchmark_loss, model, BENCHMARK_INPUT)
        sync(device)
    end
    warm = benchmark_warm(
        () -> Flux.Zygote.gradient(benchmark_loss, model, BENCHMARK_INPUT), device, seconds
    )
    return (; first_gradient, warm)
end

function benchmark_backend(::Val{:mooncake}, model, device::Val, seconds::Float64)
    local cache
    GC.gc(true)
    first_gradient = @elapsed begin
        cache = Mooncake.prepare_gradient_cache(benchmark_loss, model, BENCHMARK_INPUT)
        Mooncake.value_and_gradient!!(cache, benchmark_loss, model, BENCHMARK_INPUT)
        sync(device)
    end
    warm = benchmark_warm(
        () -> Mooncake.value_and_gradient!!(cache, benchmark_loss, model, BENCHMARK_INPUT),
        device,
        seconds,
    )
    return (; first_gradient, warm)
end

function worker(model_index::Int, backend::Symbol, device::Symbol, seconds::Float64)
    1 <= model_index <= length(FLUX_MODELS) || error("invalid model index: $model_index")
    backend in BACKENDS || error("invalid backend: $backend")
    device in DEVICES || error("invalid device: $device")
    check_seconds(seconds)
    model_index == WORKER_MODEL_INDEX || error("worker model index mismatch")
    _, model, _, _ = FLUX_MODELS[model_index]
    Flux.testmode!(model)
    device === :gpu && (model = Flux.gpu(model))

    result = benchmark_backend(Val(backend), model, Val(device), seconds)
    println(result.first_gradient, '\t', result.warm)
    return nothing
end

function run_worker(model_index::Int, backend::Symbol, device::Symbol, seconds::Float64)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__)`
    command = `$command $(@__FILE__) worker $model_index $backend $device $seconds`
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
        values = if isnothing(result.zygote)
            ntuple(_ -> "n/a", 6)
        else
            (
                format_time(result.zygote.first_gradient),
                format_time(result.mooncake.first_gradient),
                format_ratio(result.mooncake.first_gradient, result.zygote.first_gradient),
                format_time(result.zygote.warm),
                format_time(result.mooncake.warm),
                format_ratio(result.mooncake.warm, result.zygote.warm),
            )
        end
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
    supported = filter(result -> !isnothing(result.zygote), results)
    log_ratios = map(supported) do result
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

function repository_revision()
    try
        revision = readchomp(`git -C $REPOSITORY_ROOT rev-parse --short HEAD`)
        dirty = !isempty(read(`git -C $REPOSITORY_ROOT status --short`, String))
        return revision * (dirty ? "-dirty" : "")
    catch
        return "unknown"
    end
end

function print_environment(io::IO=stdout; seconds::Float64=DEFAULT_BENCHMARK_SECONDS)
    gpu_name = CUDA.functional() ? CUDA.name(CUDA.device()) : "unavailable"
    println(
        io,
        "Julia $VERSION, Flux $(pkgversion(Flux)), " *
        "Mooncake $(pkgversion(Mooncake)) ($(repository_revision())), " *
        "CUDA $(pkgversion(CUDA)), cuDNN $(pkgversion(cuDNN))",
    )
    println(io, "$(Sys.MACHINE); CPU: $(cpu_models()); GPU: $gpu_name")
    println(
        io,
        "$(Threads.nthreads()) Julia thread(s); BLAS $(BLAS.vendor()), " *
        "$(BLAS.get_num_threads()) thread(s)",
    )
    return println(
        io,
        "$(fix_sig_fig(seconds))-second warm budget; " *
        "$(BENCHMARK_WORKLOAD_MULTIPLIER)x input workload",
    )
end

function write_readme(
    cpu_results,
    gpu_results;
    seconds::Float64=DEFAULT_BENCHMARK_SECONDS,
    path::String=README_PATH,
)
    check_seconds(seconds)
    environment = sprint(io -> print_environment(io; seconds))
    open(path, "w") do io
        println(io, "# Flux gradient benchmarks")
        println(io)
        println(
            io,
            "This README is generated automatically by `benchmarks.jl`; do not edit " *
            "it directly.",
        )
        println(io)
        println(io, "Run the complete benchmark from the repository root:")
        println(io)
        println(io, "```sh")
        println(io, "julia --project=test/integration_testing/flux \\")
        println(io, "    test/integration_testing/flux/benchmarks.jl")
        println(io, "```")
        println(io)
        println(
            io,
            "Pass `cpu` or `gpu` to select one device. A following integer selects " *
            "one model, for example `benchmarks.jl cpu 1`.",
        )
        println(io)
        println(io, "## Environment")
        println(io)
        println(io, "```text")
        print(io, environment)
        println(io, "```")
        println(io)
        println(io, "## Method")
        println(io)
        println(
            io,
            "Each first-gradient measurement runs in a fresh Julia process after " *
            "package and model loading. Mooncake's measurement includes " *
            "`prepare_gradient_cache` and the first `value_and_gradient!!` call. " *
            "Warm measurements reuse the prepared cache. Both backends differentiate " *
            "the same model and input with respect to both arguments.",
        )
        println(io)
        println(
            io,
            "Inputs are enlarged by a factor of $BENCHMARK_WORKLOAD_MULTIPLIER, and " *
            "Flux layers run in test mode. `Mc / Zyg` is Mooncake time divided by " *
            "Zygote time, so values below one favour Mooncake. GPU runs use " *
            "`Flux.gpu` and `CUDA.cu`, which convert floating-point arrays to `Float32`.",
        )
        for (label, results) in (("CPU", cpu_results), ("GPU", gpu_results))
            println(io)
            println(io, "## $label results")
            println(io)
            println(io, "```text")
            print_results(io, results)
            println(io, "```")
            println(io)
            print_summary(io, results)
        end
    end
    return path
end

function print_summary(io::IO, results)
    supported = count(result -> !isnothing(result.zygote), results)
    iszero(supported) &&
        return println(io, "No models in this selection support this device.")
    first_gradient_ratio = geometric_mean_ratio(results, :first_gradient)
    warm_ratio = geometric_mean_ratio(results, :warm)
    println(
        io,
        "Across the $supported supported models, the geometric-mean Mooncake/Zygote " *
        "ratio is $(@sprintf("%.2f", first_gradient_ratio)) for first-gradient time " *
        "and $(@sprintf("%.3f", warm_ratio)) for warm-gradient time.",
    )
    comparison = if warm_ratio <= 1
        "$(@sprintf("%.2f", inv(warm_ratio))) times faster"
    else
        "$(@sprintf("%.2f", warm_ratio)) times slower"
    end
    return println(io, "Mooncake is $comparison on warm gradients by geometric mean.")
end

function run(
    device::Symbol;
    model_indices=eachindex(FLUX_MODELS),
    seconds::Float64=DEFAULT_BENCHMARK_SECONDS,
)
    device in DEVICES || error("invalid device: $device")
    check_seconds(seconds)
    device === :gpu && !CUDA.functional() && error("CUDA is not functional")
    indices = collect(Int, model_indices)
    all(index -> 1 <= index <= length(FLUX_MODELS), indices) ||
        error("model indices must be between 1 and $(length(FLUX_MODELS))")
    supports(index) = begin
        gpu_supported = first(FLUX_MODELS[index])
        device === :cpu || gpu_supported
    end
    results = []
    total = count(supports, indices) * length(BACKENDS)
    benchmark_index = 0
    for model_index in indices
        _, _, _, name = FLUX_MODELS[model_index]
        if !supports(model_index)
            push!(results, (; name, zygote=nothing, mooncake=nothing))
            continue
        end
        timings = Dict{Symbol,NamedTuple}()
        for backend in BACKENDS
            benchmark_index += 1
            @info "$device: $benchmark_index / $total", name, backend
            timings[backend] = run_worker(model_index, backend, device, seconds)
            @info "  first gradient = $(format_time(timings[backend].first_gradient))"
            @info "  warm gradient  = $(format_time(timings[backend].warm))"
        end
        push!(results, (; name, zygote=timings[:zygote], mooncake=timings[:mooncake]))
    end

    println("\n## $(uppercase(string(device)))\n")
    print_results(results)
    print_summary(stdout, results)
    return results
end

function run_all(; kwargs...)
    cpu_results = run(:cpu; kwargs...)
    gpu_results = run(:gpu; kwargs...)
    return cpu_results, gpu_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    if IS_WORKER
        worker(
            parse(Int, ARGS[2]), Symbol(ARGS[3]), Symbol(ARGS[4]), parse(Float64, ARGS[5])
        )
    elseif isempty(ARGS)
        print_environment()
        path = write_readme(run_all()...)
        @info "updated benchmark report", path
    elseif length(ARGS) == 1 && tryparse(Int, only(ARGS)) !== nothing
        print_environment()
        run_all(; model_indices=[parse(Int, only(ARGS))])
    elseif length(ARGS) in (1, 2) && Symbol(first(ARGS)) in DEVICES
        print_environment()
        device = Symbol(first(ARGS))
        indices = length(ARGS) == 1 ? eachindex(FLUX_MODELS) : [parse(Int, ARGS[2])]
        run(device; model_indices=indices)
    else
        error("invalid arguments: $(ARGS)")
    end
end
