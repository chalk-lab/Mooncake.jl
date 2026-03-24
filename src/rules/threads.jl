"""
    threaded_map!(f, output::Vector{T1}, input::Vector{T2}) -> output

Apply `f` element-wise from `input` to `output` using `Threads.@threads`, and return
`output`.

For reverse-mode AD via Mooncake to be correct and race-free, two conditions must hold:

1. Both `T1` and `T2` must be subtypes of `IEEEFloat` (`Float16`, `Float32`, or `Float64`).
   Since each element is a bits-type scalar with an independent cotangent slot in the
   gradient vector, per-element writes during the reverse pass are race-free.

2. `f` must carry no mutable differentiable state. Specifically, `f` should be a
   non-closure function (e.g. `sin`, `exp`, a user-defined non-closure) so that the
   compiled rule for `f` can be shared across threads without a race on its fdata.

See https://github.com/chalk-lab/Mooncake.jl/issues/791.
"""
function threaded_map!(f::F, output::Vector{T1}, input::Vector{T2}) where {F,T1,T2}
    length(output) >= length(input) || throw(
        ArgumentError(
            "threaded_map!: output length $(length(output)) < input length $(length(input))"
        ),
    )
    Threads.@threads for i in eachindex(input)
        @inbounds output[i] = f(@inbounds input[i])
    end
    return output
end

@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(threaded_map!),F,Vector{T1},Vector{T2}
} where {F,T1<:IEEEFloat,T2<:IEEEFloat}

# Cache pre-compiled scalar rules: (F, T2) -> rule, so build_rrule is paid once per type
# combination rather than once per call. Access is serial (rrule!! is called on the main
# thread before the parallel section), so no lock is needed.
const _threaded_map_rule_cache = IdDict{Any,Any}()

@inline function rrule!!(
    ::CoDual{typeof(threaded_map!)},
    f::CoDual{F},
    output::CoDual{Vector{T1},Vector{T1}},
    input::CoDual{Vector{T2},Vector{T2}},
) where {F,T1<:IEEEFloat,T2<:IEEEFloat}
    fp = primal(f)
    xp = primal(input)
    yp = primal(output)
    xd = tangent(input)   # cotangent accumulator for input elements
    yd = tangent(output)  # cotangent accumulator for output elements
    n = length(xp)

    # Get or build the scalar rule for f (one per (F, T2) type pair).
    f_rule = get!(_threaded_map_rule_cache, (F, T2)) do
        build_rrule(Tuple{F,T2})
    end

    # Save old output values for restoration in the pullback (undo-tape semantics).
    old_yp = copy(yp)
    old_yd = copy(yd)

    # Storage for per-element pullbacks.
    pullbacks = Vector{Any}(undef, n)

    # Forward pass: run f element-wise in parallel.
    # Zero yd[i] because we overwrite output[i] (mirrors isbits_arrayset_rrule).
    Threads.@threads for i in 1:n
        xi = @inbounds xp[i]
        yi_codual, pb = f_rule(zero_fcodual(fp), CoDual(xi, NoFData()))
        @inbounds yp[i] = primal(yi_codual)
        @inbounds yd[i] = zero_tangent(primal(yi_codual))
        @inbounds pullbacks[i] = pb
    end

    function threaded_map_pb!!(::NoRData)
        # Reverse pass: propagate cotangents from output back to input in parallel.
        # After the upstream pullbacks run, yd[i] holds the cotangent for output[i].
        Threads.@threads for i in 1:n
            dy_i = @inbounds yd[i]
            _, dx_i = @inbounds pullbacks[i](dy_i)
            @inbounds xd[i] += dx_i
        end
        # Restore output to its pre-call state (primal and cotangent).
        copyto!(yp, old_yp)
        copyto!(yd, old_yd)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return output, threaded_map_pb!!
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:threads})
    test_cases = Any[
        (false, :none, nothing, threaded_map!, sin, zeros(Float64, 4), randn(Float64, 4)),
        (false, :none, nothing, threaded_map!, exp, zeros(Float32, 3), abs.(randn(Float32, 3))),
    ]
    memory = Any[]
    return test_cases, memory
end
