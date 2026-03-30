"""
    threaded_map!(f, output::Vector, inputs::Vector...) -> output

Apply `f` element-wise to corresponding elements of `inputs`, storing results in `output`,
using `Threads.@threads`.

All element types must be isbits (checked at runtime); element types need not be identical
across vectors. For reverse-mode AD via Mooncake to be race-free, `f` must also satisfy
`fdata_type(tangent_type(F)) == NoFData`, i.e. it carries no mutable differentiable state.
This covers plain functions and closures capturing only isbits scalars.

See https://github.com/chalk-lab/Mooncake.jl/issues/791.
"""
function threaded_map!(f::F, output::Vector, inputs::Vector...) where {F}
    n = _check_threaded_map!(output, inputs)
    N    = length(inputs)
    valN = Val(N)  # Val{N} computed at outer scope where N is a compile-time constant;
                   # capturing valN::Val{N} in the closure lets ntuple unroll correctly.
    Threads.@threads for i in 1:n
        output[i] = f(ntuple(j -> inputs[j][i], valN)...)
    end
    return output
end

# Shared validation for both primal and AD paths. Returns the effective length (minimum
# of input lengths, upper-bounded by output length).
function _check_threaded_map!(output::Vector, inputs)
    isempty(inputs) && throw(ArgumentError("threaded_map!: at least one input vector required"))
    all(v -> isbitstype(eltype(v)), (output, inputs...)) || throw(ArgumentError(
        "threaded_map!: all element types must be isbits; " *
        "got $(map(eltype, (output, inputs...)))"
    ))
    n = minimum(length, inputs)
    length(output) >= n || throw(
        ArgumentError(
            "threaded_map!: output length $(length(output)) < input length $n"
        ),
    )
    return n
end

@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(threaded_map!),F,Vector,Vararg{Vector}
} where {F}

@inline function rrule!!(
    ::CoDual{typeof(threaded_map!)},
    f::CoDual{F},
    output::CoDual{Vector{Tout}, Vector{Tout}},
    inputs::CoDual...,
) where {F, Tout}
    fp  = primal(f)
    xps = map(primal, inputs)
    xds = map(tangent, inputs)  # Vector{Ti} for differentiable Ti, NoFData for others
    yp  = primal(output)
    yd  = tangent(output)
    N    = length(inputs)
    valN = Val(N)  # Val{N} computed at outer scope where N is a compile-time constant;
                   # capturing valN::Val{N} in closures lets ntuple unroll correctly.
    n   = _check_threaded_map!(yp, xps)

    # Save current primal output state for restoration in the pullback (undo-tape semantics).
    old_yp = copy(yp)

    # Forward pass: run f element-wise in parallel, storing per-element pullbacks.
    # Zero yd[i] because we overwrite output[i] (mirrors isbits_arrayset_rrule).
    # Calls rrule!! directly — Julia's method dispatch caches the specialisation.
    # Infer the concrete pullback type via Core.Compiler.return_type (same technique
    # as Base.Broadcast.combine_eltypes) to avoid a Vector{Any}.
    PBType = pullback_type(typeof(rrule!!), (F, map(eltype, xps)...))
    pullbacks = Vector{PBType}(undef, n)
    Threads.@threads for i in 1:n
        xi = ntuple(j -> CoDual(xps[j][i], NoFData()), valN)
        yi_codual, pb = rrule!!(zero_fcodual(fp), xi...)
        yp[i] = primal(yi_codual)
        yd[i] = zero_tangent(primal(yi_codual))
        pullbacks[i] = pb
    end

    function threaded_map_pb!!(::NoRData)
        # rdata type for f: NoRData for plain functions; RData{...} for closures
        # capturing isbits scalars.
        FRData = rdata_type(tangent_type(F))

        # Option A: per-element f-rdata storage (race-free: thread i writes only
        # f_rdatas[i]). Fold serially after the parallel loop.
        # TODO: Option B — per-thread accumulators with Threads.@threads :static +
        # threadid() for O(nthreads) memory. Deferred: :static has composability concerns.
        f_rdatas = FRData === NoRData ? nothing : Vector{FRData}(undef, n)

        # Reverse pass: read cotangents accumulated in yd[i] by the upstream and propagate
        # them back to each input's cotangent accumulator (skip non-differentiable inputs).
        # Zero out yd[i] after reading — the forward pass set it to zero, so zeroing
        # restores it to its post-forward state without needing a saved copy.
        Threads.@threads for i in 1:n
            dy_i = yd[i]
            yd[i] = zero_tangent(yp[i])
            rdata_tuple = pullbacks[i](dy_i)
            FRData !== NoRData && (f_rdatas[i] = rdata_tuple[1])
            for j in 1:N
                xd = xds[j]
                xd isa NoFData || (xd[i] = increment_rdata!!(xd[i], rdata_tuple[j + 1]))
            end
        end

        f_rdata = FRData === NoRData ? NoRData() :
            foldl((a, b) -> increment_internal!!(NoCache(), a, b), f_rdatas; init=zero_rdata(fp))

        # Restore primal output to its pre-call state.
        copyto!(yp, old_yp)
        return NoRData(), f_rdata, NoRData(), ntuple(_ -> NoRData(), valN)...
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
