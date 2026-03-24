"""
    threaded_map!(f, output::Vector{Tout}, inputs::Vector{<:IEEEFloat}...) -> output

Apply `f` element-wise to corresponding elements of `inputs`, storing results in `output`,
using `Threads.@threads`.

For reverse-mode AD via Mooncake to be correct and race-free, two conditions must hold:

1. `output` must have element type `Tout <: IEEEFloat` and each input must have element
   type `Ti <: IEEEFloat`. The element types need not be the same across vectors. Since
   each element is a bits-type scalar with an independent cotangent slot, per-element
   gradient writes during the reverse pass are race-free.

2. `f` must satisfy `fdata_type(tangent_type(F)) == NoFData`, i.e. it carries no mutable
   differentiable state. This covers plain (non-closure) functions and closures that capture
   only `IEEEFloat` scalars. Closures capturing mutable containers are not supported.

See https://github.com/chalk-lab/Mooncake.jl/issues/791.
"""
function threaded_map!(f::F, output::Vector{<:IEEEFloat}, inputs::Vector{<:IEEEFloat}...) where {F}
    isempty(inputs) && throw(ArgumentError("threaded_map!: at least one input vector required"))
    n = minimum(length, inputs)
    length(output) >= n || throw(
        ArgumentError(
            "threaded_map!: output length $(length(output)) < input length $n"
        ),
    )
    N = length(inputs)
    Threads.@threads for i in 1:n
        output[i] = f(ntuple(j -> inputs[j][i], Val(N))...)
    end
    return output
end

@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(threaded_map!),F,Vector{<:IEEEFloat},Vararg{Vector{<:IEEEFloat}}
} where {F}

# Cache pre-compiled scalar rules keyed by (F, input_Ts) so build_rrule is paid once per
# type combination. Accessed serially (rrule!! runs on the caller's thread before the
# parallel sections), so no lock is needed.
const _threaded_map_rule_cache = IdDict{Any,Any}()

@inline function rrule!!(
    ::CoDual{typeof(threaded_map!)},
    f::CoDual{F},
    output::CoDual{Vector{Tout}, Vector{Tout}},
    inputs::CoDual{<:Vector{<:IEEEFloat}, <:Vector{<:IEEEFloat}}...,
) where {F, Tout<:IEEEFloat}
    fp  = primal(f)
    xps = map(primal, inputs)   # NTuple of Vector{Ti} — primals (Ti may differ)
    xds = map(tangent, inputs)  # NTuple of Vector{Ti} — cotangent accumulators
    yp  = primal(output)
    yd  = tangent(output)
    N   = length(inputs)
    n   = minimum(length, xps)

    # Get or build the scalar rule for f applied to (T1, T2, ..., TN).
    # Cache key is (F, input_element_types) to handle heterogeneous inputs.
    input_Ts = map(eltype, xps)  # e.g. (Float32, Float64) for mixed inputs
    f_rule = get!(_threaded_map_rule_cache, (F, input_Ts)) do
        build_rrule(Tuple{F, input_Ts...})
    end

    # Save current output state for restoration in the pullback (undo-tape semantics).
    old_yp = copy(yp)
    old_yd = copy(yd)

    # Handle empty input vectors: nothing to compute.
    if n == 0
        pb_noop(::NoRData) = begin
            copyto!(yp, old_yp); copyto!(yd, old_yd)
            return NoRData(), NoRData(), NoRData(), ntuple(_ -> NoRData(), Val(N))...
        end
        return output, pb_noop
    end

    # Forward pass: run element 1 serially to determine the concrete pullback type,
    # then run elements 2:n in parallel. This gives Vector{PBType} instead of
    # Vector{Any}, eliminating boxing and dynamic dispatch in the reverse pass.
    xi1 = ntuple(j -> CoDual(@inbounds(xps[j][1]), NoFData()), Val(N))
    yi1_codual, pb1 = f_rule(zero_fcodual(fp), xi1...)
    @inbounds yp[1] = primal(yi1_codual)
    @inbounds yd[1] = zero_tangent(primal(yi1_codual))
    PBType = typeof(pb1)
    pullbacks = Vector{PBType}(undef, n)
    @inbounds pullbacks[1] = pb1

    Threads.@threads for i in 2:n
        xi = ntuple(j -> CoDual(@inbounds(xps[j][i]), NoFData()), Val(N))
        yi_codual, pb = f_rule(zero_fcodual(fp), xi...)
        @inbounds yp[i] = primal(yi_codual)
        @inbounds yd[i] = zero_tangent(primal(yi_codual))
        @inbounds pullbacks[i] = pb
    end

    function threaded_map_pb!!(::NoRData)
        # rdata type for f: NoRData for plain functions; RData{...} for closures
        # capturing IEEEFloat scalars.
        FRData = rdata_type(tangent_type(F))

        # Option A: per-element f-rdata storage (race-free: thread i writes only
        # f_rdatas[i]). Fold serially after the parallel loop.
        # TODO: Option B — per-thread accumulators with Threads.@threads :static +
        # threadid() for O(nthreads) memory. Deferred: :static has composability concerns.
        f_rdatas = FRData === NoRData ? nothing : Vector{FRData}(undef, n)

        # Reverse pass: read cotangents accumulated in yd[i] by the upstream and propagate
        # them back to each input's cotangent accumulator.
        Threads.@threads for i in 1:n
            @inbounds dy_i = yd[i]
            @inbounds pb_i = pullbacks[i]
            rdata_tuple = pb_i(dy_i)
            FRData !== NoRData && @inbounds(f_rdatas[i] = rdata_tuple[1])
            for j in 1:N
                @inbounds xds[j][i] += rdata_tuple[j + 1]
            end
        end

        f_rdata = FRData === NoRData ? NoRData() :
            foldl((a, b) -> increment_internal!!(NoCache(), a, b), f_rdatas; init=zero_rdata(fp))

        # Restore output to its pre-call state.
        copyto!(yp, old_yp)
        copyto!(yd, old_yd)
        return NoRData(), f_rdata, NoRData(), ntuple(_ -> NoRData(), Val(N))...
    end

    return output, threaded_map_pb!!
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:threads})
    test_cases = Any[
        # Single input, Float64 — plain function (NoRData for f)
        (false, :none, nothing, threaded_map!, sin, zeros(Float64, 4), randn(Float64, 4)),
        # Single input, Float32 — plain function
        (false, :none, nothing, threaded_map!, exp, zeros(Float32, 3), abs.(randn(Float32, 3))),
        # Two inputs, Float64 — plain function
        (false, :none, nothing, threaded_map!, +, zeros(Float64, 4), randn(Float64, 4), randn(Float64, 4)),
        # Single input, Float64 — isbits closure (non-trivial RData for f)
        (false, :none, nothing, threaded_map!, (let a = 2.0; x -> a * x; end), zeros(Float64, 4), randn(Float64, 4)),
        # Heterogeneous element types: Float32 input, Float64 output
        (false, :none, nothing, threaded_map!, Float64, zeros(Float64, 4), abs.(randn(Float32, 4))),
        # Heterogeneous element types: Float32 + Float64 inputs, Float64 output
        (false, :none, nothing, threaded_map!, (x, y) -> x + y, zeros(Float64, 4), abs.(randn(Float32, 4)), randn(Float64, 4)),
    ]
    memory = Any[]
    return test_cases, memory
end
