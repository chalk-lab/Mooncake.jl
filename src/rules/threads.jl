"""
    threaded_map!(f, output::Vector{T}, inputs::Vector{T}...) -> output

Apply `f` element-wise to corresponding elements of `inputs`, storing results in `output`,
using `Threads.@threads`.

For reverse-mode AD via Mooncake to be correct and race-free, two conditions must hold:

1. All vectors must have the same element type `T <: IEEEFloat` (`Float16`, `Float32`, or
   `Float64`). Since each element is a bits-type scalar with an independent cotangent slot,
   per-element gradient writes during the reverse pass are race-free.

2. `f` must carry no mutable differentiable state (e.g. a non-closure function or a closure
   that captures only immutable data) so that the compiled scalar rule for `f` can be shared
   across threads without a race on its fdata.

See https://github.com/chalk-lab/Mooncake.jl/issues/791.
"""
function threaded_map!(f::F, output::Vector{T}, inputs::Vector{T}...) where {F,T}
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
    typeof(threaded_map!),F,Vector{T},Vararg{Vector{T}}
} where {F,T<:IEEEFloat}

# Cache pre-compiled scalar rules keyed by (F, N, T) so build_rrule is paid once per type
# combination. Accessed serially (rrule!! runs on the caller's thread before the parallel
# sections), so no lock is needed.
const _threaded_map_rule_cache = IdDict{Any,Any}()

@inline function rrule!!(
    ::CoDual{typeof(threaded_map!)},
    f::CoDual{F},
    output::CoDual{Vector{T},Vector{T}},
    inputs::CoDual{Vector{T},Vector{T}}...,
) where {F,T<:IEEEFloat}
    fp  = primal(f)
    xps = map(primal, inputs)   # NTuple{N, Vector{T}} — primals
    xds = map(tangent, inputs)  # NTuple{N, Vector{T}} — cotangent accumulators
    yp  = primal(output)
    yd  = tangent(output)
    N   = length(inputs)
    n   = minimum(length, xps)

    # Get or build the scalar rule for f applied to N arguments of type T.
    f_rule = get!(_threaded_map_rule_cache, (F, N, T)) do
        build_rrule(Tuple{F,Vararg{T,N}})
    end

    # Save current output state for restoration in the pullback (undo-tape semantics).
    old_yp = copy(yp)
    old_yd = copy(yd)

    # Storage for per-element pullbacks.
    pullbacks = Vector{Any}(undef, n)

    # Forward pass: run f element-wise in parallel.
    # Zero yd[i] because we overwrite output[i] (mirrors isbits_arrayset_rrule).
    Threads.@threads for i in 1:n
        xi_coduals = ntuple(j -> CoDual(xps[j][i], NoFData()), Val(N))
        yi_codual, pb = f_rule(zero_fcodual(fp), xi_coduals...)
        yp[i] = primal(yi_codual)
        yd[i] = zero_tangent(primal(yi_codual))
        pullbacks[i] = pb
    end

    function threaded_map_pb!!(::NoRData)
        # Reverse pass: read cotangents accumulated in yd[i] by the upstream and propagate
        # them back to each input's cotangent accumulator.
        Threads.@threads for i in 1:n
            dy_i = yd[i]
            # pullbacks[i](dy_i) returns (NoRData(), dx1, dx2, ..., dxN)
            rdata_tuple = pullbacks[i](dy_i)
            for j in 1:N
                xds[j][i] += rdata_tuple[j + 1]
            end
        end
        # Restore output to its pre-call state.
        copyto!(yp, old_yp)
        copyto!(yd, old_yd)
        return NoRData(), NoRData(), NoRData(), ntuple(_ -> NoRData(), Val(N))...
    end

    return output, threaded_map_pb!!
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:threads})
    test_cases = Any[
        # Single input, Float64
        (false, :none, nothing, threaded_map!, sin, zeros(Float64, 4), randn(Float64, 4)),
        # Single input, Float32
        (false, :none, nothing, threaded_map!, exp, zeros(Float32, 3), abs.(randn(Float32, 3))),
        # Two inputs, Float64
        (false, :none, nothing, threaded_map!, +, zeros(Float64, 4), randn(Float64, 4), randn(Float64, 4)),
    ]
    memory = Any[]
    return test_cases, memory
end
