"""
    threaded_map!(f, output::Vector, inputs::Vector...) -> output

Apply `f` element-wise to corresponding elements of `inputs`, storing results in `output`,
using `Threads.@threads`.

All element types must be isbits (checked at runtime); element types need not be identical
across vectors. In addition, `f` must satisfy
`fdata_type(tangent_type(typeof(f))) == NoFData`, i.e. it carries no mutable differentiable
state. This restriction is enforced for all calls to `threaded_map!` and covers plain
functions and closures capturing only isbits scalars.
"""
function threaded_map!(f, output::Vector, inputs::Vector...)
    n = _check_threaded_map!(f, output, inputs)
    N = length(inputs)
    valN = Val(N)  # Val{N} computed at outer scope where N is a compile-time constant;
    # capturing valN::Val{N} in the closure lets ntuple unroll correctly.
    Threads.@threads for i in 1:n
        output[i] = f(ntuple(j -> inputs[j][i], valN)...)
    end
    return output
end

# Shared validation for both primal and AD paths. Returns the effective length (the common
# prefix over `output` and all inputs) — same semantics as `Base.map!`.
function _check_threaded_map!(f, output::Vector, inputs)
    isempty(inputs) &&
        throw(ArgumentError("threaded_map!: at least one input vector required"))
    all(v -> isbitstype(eltype(v)), (output, inputs...)) || throw(
        ArgumentError(
            "threaded_map!: all element types must be isbits; " *
            "got $(map(eltype, (output, inputs...)))",
        ),
    )
    fdata_type(tangent_type(typeof(f))) == NoFData || throw(
        ArgumentError(
            "threaded_map!: f must satisfy fdata_type(tangent_type(typeof(f))) == NoFData " *
            "(no mutable differentiable state); got typeof(f) = $(typeof(f))",
        ),
    )
    n = min(length(output), minimum(length, inputs))
    return n
end

@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(threaded_map!),F,Vector,Vararg{Vector}
} where {F}

@inline function _rrule_output_codual_type(Trule, arg_types)
    return _rrule_output_codual_type(
        Core.Compiler.return_type(Tuple{Trule,map(fcodual_type, arg_types)...})
    )
end

_rrule_output_codual_type(::Core.TypeofBottom) = Any
_rrule_output_codual_type(T::DataType) = T <: Tuple ? T.parameters[1] : Any
function _rrule_output_codual_type(T::Union)
    Union{_rrule_output_codual_type(T.a),_rrule_output_codual_type(T.b)}
end

_codual_primal_type(::Type{<:CoDual{P}}) where {P} = P
_codual_primal_type(::Type) = Any
_codual_primal_type(T::Union) = Union{_codual_primal_type(T.a),_codual_primal_type(T.b)}

@inline _thread_local_rule(rule::Function) = rule
@inline _thread_local_rule(rule) = _copy(rule)

struct ThreadedMapElementPullback{TConvertPB,TElementPB}
    convert_pb::TConvertPB
    element_pb::TElementPB
end

@inline function (pb::ThreadedMapElementPullback)(dy)
    # The conversion rule is called as `convert(T, x)`, so its last pullback return is the
    # cotangent for the value argument `x`, which is what the element pullback consumes.
    return pb.element_pb(last(pb.convert_pb(dy)))
end

struct ThreadedMapReverseRule{TElementRule,TConvertRule,TPB,N}
    element_rule::TElementRule
    convert_rule::TConvertRule
end

function build_primitive_rrule(
    sig::Type{<:Tuple{typeof(threaded_map!),F,O,Vararg{Any,N}}}
) where {F,O<:Vector,N}
    element_f_type = sig.parameters[2]
    input_vec_types = sig.parameters[4:end]
    all(T -> T <: Vector, input_vec_types) || throw(
        ArgumentError(
            "threaded_map! primitive rule expects Vector inputs, got $input_vec_types"
        ),
    )
    element_arg_types = map(eltype, input_vec_types)
    element_sig = Tuple{element_f_type,element_arg_types...}
    element_rule = build_rrule(element_sig)
    element_pb_type = pullback_type(
        _typeof(element_rule), (element_f_type, element_arg_types...)
    )
    element_output_type = _codual_primal_type(
        _rrule_output_codual_type(
            _typeof(element_rule), (element_f_type, element_arg_types...)
        ),
    )
    # A concrete element output type is required to build a concrete `convert` rule for
    # the array write; accepting abstract / union outputs would silently degrade precision.
    isconcretetype(element_output_type) || throw(
        ArgumentError(
            "threaded_map! reverse rule requires a concrete element output type; got $element_output_type for $element_sig",
        ),
    )
    # Differentiate the actual array write `output[i] = value`, which lowers through
    # `convert(eltype(output), value)`.
    convert_sig = Tuple{typeof(convert),Type{eltype(O)},element_output_type}
    convert_rule = build_rrule(convert_sig)
    convert_pb_type = pullback_type(
        _typeof(convert_rule), (typeof(convert), Type{eltype(O)}, element_output_type)
    )
    pb_type = ThreadedMapElementPullback{convert_pb_type,element_pb_type}
    return ThreadedMapReverseRule{typeof(element_rule),typeof(convert_rule),pb_type,N}(
        element_rule, convert_rule
    )
end

@inline function (rule::ThreadedMapReverseRule{TElementRule,TConvertRule,TPB,N})(
    ::CoDual{typeof(threaded_map!)},
    f::CoDual{F},
    output::CoDual{<:Vector},
    inputs::CoDual{<:Vector}...,
) where {TElementRule,TConvertRule,TPB,N,F}
    fp = primal(f)
    xps = map(primal, inputs)
    xds = map(tangent, inputs)
    yp = primal(output)
    yd = tangent(output)
    valN = Val(N)
    n = _check_threaded_map!(fp, yp, xps)

    # Save the destination primal/tangent so the pullback can restore the overwritten
    # prefix for non-aliased outputs. Snapshotting `yd` adds another array-sized copy, but
    # the returned value aliases `output`, so higher-order AD expects the destination
    # tangent buffer to be reset after the pullback.
    old_yp = copy(yp)
    old_yd = copy(yd)
    output_tangent_aliases_input = any(xd -> !(xd isa NoFData) && xd === yd, xds)

    # Forward pass: run f element-wise in parallel, storing per-element pullbacks.
    # These rules are materialized per element, not per thread: reverse DerivedRules keep
    # mutable pullback state, so reusing one across multiple element calls would alias the
    # stored pullbacks.
    pullbacks = Vector{TPB}(undef, n)
    Threads.@threads for i in 1:n
        thread_rule = _thread_local_rule(rule.element_rule)
        convert_rule = _thread_local_rule(rule.convert_rule)
        xi = ntuple(j -> CoDual(xps[j][i], NoFData()), valN)
        yi_codual, element_pb = thread_rule(zero_fcodual(fp), xi...)
        # `output[i] = ...` lowers through `convert(eltype(output), value)`, so the reverse
        # pass must compose the write conversion pullback before the element pullback.
        zi_codual, convert_pb = convert_rule(
            zero_fcodual(convert), zero_fcodual(eltype(yp)), yi_codual
        )
        yp[i] = primal(zi_codual)
        yd[i] = zero_tangent(primal(zi_codual))
        pullbacks[i] = ThreadedMapElementPullback(convert_pb, element_pb)
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
        # Consume yd[i] after reading it, then restore the full destination tangent buffer
        # from the pre-call snapshot once the pullback is finished.
        # Use increment_rdata!! rather than += : for isbits structs, xd[i]::Tangent{...}
        # but the pullback returns RData{...} — different types; increment_rdata!! handles
        # all tangent types uniformly.
        Threads.@threads for i in 1:n
            dy_i = yd[i]
            rdata_tuple = pullbacks[i](dy_i)
            # Element pullbacks return `(f_rdata, input_1_rdata, input_2_rdata, ...)`.
            FRData !== NoRData && (f_rdatas[i] = rdata_tuple[1])
            for j in 1:N
                xd = xds[j]
                xd isa NoFData || (xd[i] = increment_rdata!!(xd[i], rdata_tuple[j + 1]))
            end
        end

        f_rdata = if FRData === NoRData
            NoRData()
        else
            foldl(
                (a, b) -> increment_internal!!(NoCache(), a, b),
                f_rdatas;
                init=zero_rdata(fp),
            )
        end

        # Restore the overwritten destination region for non-aliased outputs. If `output`
        # aliases an input tangent buffer, the prefix now holds that input's adjoints and
        # must be preserved. The untouched suffix was never overwritten, so any adjoints
        # accumulated there should also remain available after the pullback.
        copyto!(yp, old_yp)
        if !output_tangent_aliases_input
            copyto!(view(yd, 1:n), view(old_yd, 1:n))
        end
        return NoRData(), f_rdata, NoRData(), ntuple(_ -> NoRData(), valN)...
    end

    return output, threaded_map_pb!!
end

struct ThreadedScale{T}
    a::T
end

(f::ThreadedScale)(x) = f.a * x

function hand_written_rule_test_cases(rng_ctor, ::Val{:threads})
    alias_vec = randn(Float64, 4)
    test_cases = Any[
        # Single input, Float64 — plain function (NoRData for f)
        (false, :none, nothing, threaded_map!, sin, zeros(Float64, 0), randn(Float64, 0)),
        (false, :none, nothing, threaded_map!, sin, zeros(Float64, 4), randn(Float64, 4)),
        # Single input, Float32 — plain function
        (
            false,
            :none,
            nothing,
            threaded_map!,
            exp,
            zeros(Float32, 3),
            abs.(randn(Float32, 3)),
        ),
        # Two inputs, Float64 — plain function
        (
            false,
            :none,
            nothing,
            threaded_map!,
            +,
            zeros(Float64, 4),
            randn(Float64, 4),
            randn(Float64, 4),
        ),
        # Single input, Float64 — isbits closure (non-trivial RData for f)
        (
            false,
            :none,
            nothing,
            threaded_map!,
            ThreadedScale(2.0),
            zeros(Float64, 4),
            randn(Float64, 4),
        ),
        # Heterogeneous element types: Float32 input, Float64 output
        (
            false,
            :none,
            nothing,
            threaded_map!,
            Float64,
            zeros(Float64, 4),
            abs.(randn(Float32, 4)),
        ),
        # Destination can be the shortest vector, same as Base.map!
        (false, :none, nothing, threaded_map!, sin, zeros(Float64, 2), randn(Float64, 4)),
        # Longer destination leaves a suffix untouched; its returned adjoint must survive.
        (false, :none, nothing, threaded_map!, sin, zeros(Float64, 6), randn(Float64, 4)),
        # Heterogeneous element types: Float32 + Float64 inputs, Float64 output
        (
            false,
            :none,
            nothing,
            threaded_map!,
            (x, y) -> x + y,
            zeros(Float64, 4),
            abs.(randn(Float32, 4)),
            randn(Float64, 4),
        ),
        # Destination conversion: Float64 result written into ComplexF64 output
        (
            false,
            :none,
            nothing,
            threaded_map!,
            sin,
            zeros(ComplexF64, 4),
            randn(Float64, 4),
        ),
        # In-place aliasing output === input should keep the input adjoint.
        (false, :none, nothing, threaded_map!, sin, alias_vec, alias_vec),
    ]
    memory = Any[]
    return test_cases, memory
end

# Primitive rule coverage lives in this file; there are no derived threads-specific cases.
derived_rule_test_cases(rng_ctor, ::Val{:threads}) = Any[], Any[]
