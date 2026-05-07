# Primal-mode forward AD: single lifted OC for both primal (Val(0)) and dual (Val(N))
# execution.

# Construct a PrimalMode interpreter with BugPatchInterpreter in meta, giving the
# inliner access to the native Julia code cache (where frule!! bodies live).
function _make_primal_mode_interp(::Type{C}) where {C}
    return MooncakeInterpreter(C, PrimalMode; meta=BugPatchInterpreter())
end

# Initialize PrimalMode interpreter (deferred from abstract_interpretation.jl because
# BugPatchInterpreter is defined in patch_for_319.jl, loaded after abstract_interpretation).
GLOBAL_INTERPRETERS[PrimalMode] = _make_primal_mode_interp(DefaultCtx)

# _prim_call dispatches primitive calls by tangent_mode:
# Val(0) → direct primal call, Val(N) → frule!! call.
@inline _prim_call(::Val{0}, f, args...) = f(args...)
@inline _prim_call(::Val{N}, f, args...) where {N} = frule!!(f, args...)

# _primal_of extracts the primal boolean for control flow.
@inline _primal_of(x::Bool) = x
@inline _primal_of(x) = primal(x)

# __get_primal for Dual values — reverse_mode.jl defines the CoDual overload and
# the generic fallback; this adds the Dual case needed by __fwds_pass_no_ad!.
# The NDual overload is in NfwdMooncake.jl (NDual is defined after this file loads).
__get_primal(x::Dual) = primal(x)
__get_primal(x::Tuple) = map(__get_primal, x)

# Check if a type contains Union{} (bottom type) anywhere in its structure.
# This can happen with unreachable code or failed type inference.
@inline contains_bottom_type(T) = _contains_bottom_type(T, Base.IdSet{Any}())

function _contains_bottom_type(T, seen::Base.IdSet{Any})
    T === Union{} && return true
    if T isa Union
        return _contains_bottom_type(T.a, seen) || _contains_bottom_type(T.b, seen)
    elseif T isa TypeVar
        T in seen && return false
        push!(seen, T)
        return _contains_bottom_type(T.ub, seen)
    elseif T isa UnionAll
        T in seen && return false
        push!(seen, T)
        return _contains_bottom_type(T.body, seen)
    elseif T isa DataType
        T in seen && return false
        push!(seen, T)
        for p in T.parameters
            _contains_bottom_type(p, seen) && return true
        end
        return false
    else
        return false
    end
end

@inline get_capture(captures::T, n::Int) where {T} = captures[n]

"""
    const_dual!(captures::Vector{Any}, stmt, width=Val(1))

Build a `Lifted{P, N, V}` (Phase 4 width-N path) from `stmt`, with zero /
uninitialised tangent. If the resulting value is a bits type, then it is
returned. If it is not, then it is put into captures, and its location in
`captures` returned. `Val(0)` is the primal passthrough.

Whether or not the value is a literal, or an index into the captures, can
be determined from the return type.
"""
function const_dual!(captures::Vector{Any}, stmt, width::Val=Val(1))
    v = get_const_primal_value(stmt)
    x = _uninit_dual(width, v)
    if safe_for_literal(v)
        return x
    else
        push!(captures, x)
        return length(captures)
    end
end

# Width-aware uninit_dual dispatcher for IR-gen-time constants.
#
# Returns a `Lifted{P, N, V}` matching `lifted_type(Val(N), P)` — IR-emit
# threads Lifted through OC slot boundaries. Rule call sites unwrap via
# `_unlift` before invoking the still-bare rule body. `Val(0)` is the
# primal passthrough — `lifted_type(Val(0), P) === P`, so constants flow
# through as bare values.
@inline _uninit_dual(::Val{0}, v) = v
@inline function _uninit_dual(w::Val{N}, v::T) where {T<:IEEEFloat,N}
    return Lifted{T,N}(dual_type(w, T)(v))
end
@inline function _uninit_dual(w::Val{N}, v::Complex{T}) where {T<:IEEEFloat,N}
    inner = Complex(dual_type(w, T)(real(v)), dual_type(w, T)(imag(v)))
    return Lifted{Complex{T},N}(inner)
end
# Concrete Tuple/NamedTuple primal: lift element-wise. Recurse via
# `_uninit_inner` to skip the per-element `Lifted` wrap-then-unwrap.
@inline function _uninit_dual(w::Val{N}, v::Tuple) where {N}
    isconcretetype(_typeof(v)) || return _uninit_dual_fallback(w, v)
    return Lifted{_typeof(v),N}(_uninit_inner(w, v))
end
@inline function _uninit_dual(w::Val{N}, v::NamedTuple) where {N}
    isconcretetype(_typeof(v)) || return _uninit_dual_fallback(w, v)
    return Lifted{_typeof(v),N}(_uninit_inner(w, v))
end

# `_uninit_inner` returns the bare inner V for the Tuple/NamedTuple recursion,
# avoiding the `Lifted` wrapper that the recursive caller would only unwrap.
@inline _uninit_inner(::Val{0}, v) = v
@inline _uninit_inner(w::Val{N}, v::T) where {T<:IEEEFloat,N} = dual_type(w, T)(v)
@inline function _uninit_inner(w::Val{N}, v::Complex{T}) where {T<:IEEEFloat,N}
    return Complex(dual_type(w, T)(real(v)), dual_type(w, T)(imag(v)))
end
@inline _uninit_inner(w::Val, v::Tuple) = map(vi -> _uninit_inner(w, vi), v)
@inline _uninit_inner(w::Val, v::NamedTuple{names}) where {names} = NamedTuple{names}(
    map(vi -> _uninit_inner(w, vi), values(v))
)
@inline _uninit_inner(::Val{N}, v) where {N} = uninit_dual(v)

# Strict width-N fallback: containers with `_count_slots > 0` must register an
# explicit `_uninit_dual(::Val{N}, ::T)` overload in `NfwdMooncake.jl` —
# silently downgrading to width-1 here would produce wrong tangents.
function _uninit_dual(w::Val{N}, v) where {N}
    return _uninit_dual_fallback(w, v)
end
function _uninit_dual_fallback(w::Val{N}, v) where {N}
    if _count_slots(v) > 0
        throw(
            ArgumentError(
                "_uninit_dual: missing width-N overload for `$(_typeof(v))` " *
                "(found $(_count_slots(v)) differentiable scalar slot(s)). " *
                "Add a method to `NfwdMooncake.jl` so that primal-mode forward " *
                "AD can lift this container type into an NDual representation.",
            ),
        )
    end
    return Lifted{_typeof(v),N}(uninit_dual(v))
end

# Extract the integer N from `Val{N}`. Used at IR-emit time to construct
# `Lifted{P, N}` parametric types when wrapping rule results.
@inline _val_n(::Val{N}) where {N} = N

# A "lifted width" is one that uses the `Lifted{P, N, V}` boundary at the OC
# (i.e. `Val{N}` for `N >= 1`). The primal-passthrough `Val{0}` keeps the OC
# slot types bare.
@inline _is_lifted_width(::Val{0}) = false
@inline _is_lifted_width(::Val{N}) where {N} = true

"""
    _is_lifted_aware(sig::Type{<:Tuple})

Trait function — returns `true` for primitive call signatures whose
`frule!!` accepts `Lifted{P, N, V}` arguments directly (Phase 4 migration),
`false` (default) for legacy primitives that still expect bare slot values.

The IR-emit consults this at each primitive rule-call site: when `true`, it
passes `Lifted` args straight through and trusts the rule to return a
`Lifted`; when `false`, it inserts the Phase-3 unwrap/wrap scaffolding
(unlift each arg, call the bare rule, wrap the bare return back into
`Lifted{primal_retype, N}`).

Each rule file migrated under Phase 4 registers its sigs by adding a
specialised method that returns `true`. The default catch-all keeps every
non-migrated rule on the legacy bare path, so partial migration is safe.
"""
@inline _is_lifted_aware(::Type) = false

"""
    frule!!(f::Lifted{F, N}, args::Vararg{Lifted, M})

Generic Lifted-aware `frule!!` adapter — used by Phase 4 mechanical
migrations whose rule body has no benefit from `Lifted` dispatch. Unwraps
each `Lifted` arg via `_unlift`, calls the bare `frule!!`, and re-wraps
the bare result in `Lifted{P_out, N}` where `P_out` is recovered from
the bare result via `__get_primal`.

This generic catch-all is invoked when a rule has registered itself in
`_is_lifted_aware` but provides no specific `Lifted{typeof(op), N}`
overload. Rules with meaningful `Lifted`-dispatch benefit (e.g. the
`tuple` frule's three-branch collapse) override this by defining their
own specific `Lifted`-typed `frule!!` method.
"""
@inline function frule!!(f::Lifted{F,N}, args::Vararg{Lifted,M}) where {F,N,M}
    bare_f = _unlift(f)
    bare_args = ntuple(i -> _unlift(args[i]), Val(M))
    bare_result = frule!!(bare_f, bare_args...)
    P_out = _typeof(__get_primal(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end

# Wrap a bare frule result back into `Lifted{P_out, N, V}`. The OC's slot
# type is `lifted_type(Val(N), P_out)` which fixes V to `dual_type(Val(N),
# P_out)`. The bare rule may return values whose actual V differs from the
# canonical (e.g. `getfield` returns `Dual{Float64, Float64}` from
# `_dual_or_ndual` even when the canonical V at width=Val(1) is
# `NDual{Float64, 1}`). To canonicalise, route bare `Dual{P, T}` results
# through the 2-arg `Lifted{P_out, N}(primal, tangent)` ctor — that calls
# `dual_type(Val(N), P_out)(primal, tangent)`, producing the canonical V.
# Other shapes (NDual, Complex{<:NDual}, Array{<:NDual}, Dual{P, NTangent}
# at width N>=2, etc.) are already canonical and use the 1-arg form.
@inline function _wrap_rule_result(::Type{P}, ::Val{N}, x::Dual) where {P,N}
    # `Lifted{P, N}(primal, tangent)` 2-arg routes through `dual_type(Val(N),
    # P)(primal, tangent)` to canonicalise V. For abstract P (Union, UnionAll,
    # Any) `dual_type` returns an abstract type that's not callable, so fall
    # back to the 1-arg form (V inferred from the bare result's type).
    return isconcretetype(P) ? Lifted{P,N}(primal(x), tangent(x)) : Lifted{P,N}(x)
end

# Tuple-primal result: bare frule may return a Tuple of bare `Dual{P_i, T_i}`
# values (e.g. from `_dual_or_ndual` for IEEEFloat fields). Canonicalise each
# element to the slot's expected V_i = `dual_type(Val(N), P_i)`.
@inline function _wrap_rule_result(::Type{P}, w::Val{N}, x::Tuple) where {P<:Tuple,N}
    if isconcretetype(P) && fieldcount(P) == length(x)
        InnerT = dual_type(Val(N), P)
        if InnerT isa DataType && InnerT <: Tuple
            return Lifted{P,N,InnerT}(_canonicalise_tuple_inner(InnerT, x))
        end
    end
    return Lifted{P,N}(x)
end

@inline _wrap_rule_result(::Type{P}, ::Val{N}, x) where {P,N} = Lifted{P,N}(x)

# Build an inner element-wise tuple of inner duals from a tuple of bare
# rule-emitted values. Each element is routed through `Vi(primal, tangent)`
# when it's a non-canonical `Dual`, otherwise passes through. Shared by
# `_wrap_rule_result(::Type{<:Tuple})` and the `tuple` Lifted-aware frule.
@inline function _canonicalise_tuple_inner(::Type{InnerT}, x::Tuple) where {InnerT<:Tuple}
    return ntuple(Val(fieldcount(InnerT))) do i
        Vi = fieldtype(InnerT, i)
        elem = x[i]
        if elem isa Dual && typeof(elem) !== Vi && isconcretetype(Vi)
            Vi(primal(elem), tangent(elem))
        else
            elem
        end
    end
end

# Canonicalise a runtime `Lifted` value to match a target P at the OC return
# boundary. `Lifted`'s `P` parameter is invariant, so a runtime `Lifted{Any}`
# is not a subtype of the OC's `Lifted{P_target}` return slot — PhiNode
# merges and abstract-slot rules (dynamic getfield through `RefValue{Any}`)
# can leave widened P / legacy V in the runtime value.
@inline _canon_return(::Val{N}, ::Type{P_target}, x) where {N,P_target} = x
@inline function _canon_return(
    ::Val{N}, ::Type{P_target}, x::Lifted{P,N}
) where {N,P_target,P}
    P === P_target && return x
    inner = _unlift(x)
    if inner isa Dual && isconcretetype(P_target)
        return Lifted{P_target,N}(primal(inner), tangent(inner))
    end
    return Lifted{P_target,N,typeof(inner)}(inner)
end

# Insert an SSA `_unlift(arg)` at the current position, returning the new SSA.
# Only used when `info.width !== nothing`; for runtime references (`Argument` /
# `SSAValue`) inside an OC whose slot type is `Lifted{P, N, V}`.
@inline function _insert_unlift!(lifted_ir::IRCode, ssa::SSAValue, arg)
    return CC.insert_node!(
        lifted_ir, ssa, new_inst(Expr(:call, _unlift, arg)), ATTACH_BEFORE
    )
end

const ATTACH_AFTER = true
const ATTACH_BEFORE = false

"""
    __unflatten_dual_varargs(isva::Bool, args, ::Val{nargs}, width::Val=Val(1)) where {nargs}

If isva and nargs=2, then inputs `(NDual(5.0, (0.0,)), NDual(4.0, (0.0,)),
NDual(3.0, (0.0,)))` are grouped element-wise into the trailing varargs slot.
"""
function __unflatten_dual_varargs(
    isva::Bool, args, ::Val{nargs}, width::Val=Val(1)
) where {nargs}
    isva || return args
    rest = args[nargs:end]
    group_primal = map(primal, rest)
    if tangent_type(_typeof(group_primal)) == NoTangent
        grouped_args = zero_dual(group_primal)
    else
        grouped_args = _group_vararg_dual(width, group_primal, rest)
    end
    return (args[1:(nargs - 1)]..., grouped_args)
end

function _group_vararg_dual(::Val{1}, group_primal, rest)
    # Width 1: bare per-element tangent tuple (no `NTangent` wrap), so the
    # resulting `Dual` matches the canonical `dual_type(Val(1), Tuple{...})`
    # which uses bare T at N=1.
    return Dual(group_primal, map(tangent, rest))
end

function _group_vararg_dual(::Val{N}, group_primal, rest) where {N}
    per_dir = ntuple(Val(N)) do i
        map(x -> _partial_i(x, i), rest)
    end
    return Dual(group_primal, NTangent(per_dir))
end

_partial_i(x::Dual, ::Int) = tangent(x)

get_forward_primal_type(ir::CC.IRCode, a::Argument) = ir.argtypes[a.n]
get_forward_primal_type(ir::CC.IRCode, ssa::SSAValue) = get_ir(ir, ssa, :type)
get_forward_primal_type(::CC.IRCode, x::QuoteNode) = _typeof(x.value)
get_forward_primal_type(::CC.IRCode, x) = _typeof(x)
function get_forward_primal_type(::CC.IRCode, x::GlobalRef)
    return isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.binding.ty
end
function get_forward_primal_type(::CC.IRCode, x::Expr)
    x.head === :boundscheck && return Bool
    return error("Unrecognised expression $x found in argument slot.")
end

# Public entry point

function build_frule(args...; debug_mode=false, silence_debug_messages=true)
    sig = _typeof(TestUtils.__get_primals(args))
    interp = get_interpreter(ForwardMode)
    return build_frule(interp, sig; debug_mode, silence_debug_messages)
end

struct PrimalRuleInfo
    isva::Bool
    nargs::Int
    lifted_ret_type::Type
end

"""
    build_frule(
        interp::MooncakeInterpreter{C},
        sig_or_mi;
        debug_mode=false,
        silence_debug_messages=true,
        skip_world_age_check=false,
    ) where {C}

Returns a function which performs forward-mode AD for `sig_or_mi`. Will derive a rule if
`sig_or_mi` is not a primitive.

Set `skip_world_age_check=true` when the interpreter's world age is intentionally older
than the current world (e.g., when building rules for MistyClosure which uses its own world).
"""
function build_frule(
    interp::MooncakeInterpreter{C},
    sig_or_mi,
    width=Val(1);
    debug_mode=false,
    silence_debug_messages=true,
    skip_world_age_check=false,
) where {C}
    @nospecialize sig_or_mi

    # To avoid segfaults, ensure that we bail out if the interpreter's world age is greater
    # than the current world age.
    if !skip_world_age_check && Base.get_world_counter() > interp.world
        throw(
            ArgumentError(
                "World age associated to interp is behind current world age. Please " *
                "create a new interpreter for the current world age.",
            ),
        )
    end

    # If we're compiling in debug mode, let the user know by default.
    if !silence_debug_messages && debug_mode
        @info "Compiling frule for $sig_or_mi in debug mode. Disable for best performance."
    end

    # If we have a hand-coded rule, just use that.
    sig = _get_sig(sig_or_mi)
    if is_primitive(C, ForwardMode, sig, interp.world)
        rule = build_primitive_frule(sig)
        return debug_mode ? DebugFRule(rule) : rule
    end

    # We don't have a hand-coded rule, so derive one.
    lock(MOONCAKE_INFERENCE_LOCK)
    try
        # If we've already derived the OpaqueClosures and info, do not re-derive, just
        # create a copy and pass in new shared data.
        oc_cache_key = ClosureCacheKey(
            interp.world, (sig_or_mi, debug_mode, :forward, width)
        )
        if haskey(interp.oc_cache, oc_cache_key)
            return interp.oc_cache[oc_cache_key]
        else
            # Derive forward-pass IR, and shove in a `MistyClosure`.
            lifted_ir, captures, info = generate_lifted_ir(
                interp, sig_or_mi, width; debug_mode
            )
            # Soundness check on the post-inference IR. Uses `typeintersect`
            # non-empty (not strict subtype) so legitimate widenings pass and
            # only truly disjoint joins fire.
            verify_phi_soundness(lifted_ir, sig_or_mi, width)
            lifted_oc = misty_closure(
                info.lifted_ret_type, lifted_ir, captures...; do_compile=true
            )
            sig = flatten_va_sig(sig, info.isva, info.nargs)
            _W = typeof(width)
            raw_rule = DerivedPrimal{sig,typeof(lifted_oc),info.isva,info.nargs,_W}(
                lifted_oc, width
            )
            rule = debug_mode ? DebugFRule(raw_rule) : raw_rule
            interp.oc_cache[oc_cache_key] = rule
            return rule
        end
    catch e
        rethrow(e)
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

# DerivedPrimal — compiled forward rule from lifted OC

struct DerivedPrimal{primal_sig,Tlifted_oc,isva,nargs,W}
    lifted_oc::Tlifted_oc
    width::W
end

@inline function (fwd::DerivedPrimal{primal_sig,sig,isva,nargs})(
    args::Vararg{Any,N}
) where {primal_sig,sig,N,isva,nargs}
    width = fwd.width
    flat_args = __unflatten_dual_varargs(isva, args, Val(nargs), width)
    if _is_lifted_width(width)
        # OC boundary: at width=Val(N>=1), the OC argtypes are
        # `Lifted{P_i, N, V_i}` (per `lifted_type`). Wrap each bare slot value
        # before invoking the OC, and unwrap the `Lifted` return so the
        # caller still sees a bare slot value.
        lifted_args = _wrap_oc_args(width, primal_sig, flat_args, isva, Val(nargs))
        return _unlift(fwd.lifted_oc(lifted_args...))
    else
        # Val{0} primal-passthrough — OC argtypes are bare primal `P_i`.
        return fwd.lifted_oc(flat_args...)
    end
end

# Wrap each entry of `flat_args` in `Lifted{P_i, N}` using the primal types
# from the rule's `primal_sig` (a `Tuple{P_1, P_2, ...}`). Vararg flattening
# already happened in `__unflatten_dual_varargs`, so for `isva=true` the
# trailing packed arg's primal type is recovered from the value itself
# rather than from `primal_sig`.
@inline function _wrap_oc_args(
    w::Val{N}, ::Type{primal_sig}, flat_args::Tuple, isva::Bool, ::Val{nargs}
) where {N,primal_sig,nargs}
    return ntuple(Val(length(flat_args))) do i
        if isva && i == nargs
            P_i = _typeof(__get_primal(flat_args[i]))
        else
            P_i = fieldtype(primal_sig, i)
        end
        _wrap_arg(w, P_i, flat_args[i])
    end
end

# Construct `Lifted{P_i, N, V}` for a single OC arg slot. For a bare canonical
# inner (NDual / Complex{<:NDual} / Array{<:NDual} / Dual{P, NTangent}), the
# 1-arg `Lifted{P, N}(value)` infers V from `typeof(value)` and produces the
# OC's expected slot type. For a legacy `Dual{P, T}` (test_rule constructs
# these via `dual_type(P)`), route through the 2-arg `Lifted{P, N}(primal,
# tangent)` so `dual_type(Val(N), P)(primal, tangent)` builds the canonical V.
@inline _wrap_arg(::Val{N}, ::Type{P}, x) where {N,P} = Lifted{P,N}(x)
@inline function _wrap_arg(::Val{N}, ::Type{P}, x::Dual{P}) where {N,P}
    return Lifted{P,N}(primal(x), tangent(x))
end
# Passthrough for callers that already hold the slot type (option (c) of the
# boundary contract: caller built a `Lifted{P, N}` directly via `zero_lifted`
# / `uninit_lifted` / a direct ctor, so no repack is needed). This is the path
# that lets in-place mutation propagate from the rule body back to the
# caller's storage — `Vector{NDual}` shares no storage with `Dual{Vector,
# Vector}`, so the only way a `mul!`-style write reaches user-visible state
# is if the user passed the slot directly.
@inline _wrap_arg(::Val{N}, ::Type{P}, x::Lifted{P,N}) where {N,P} = x

# On Julia 1.10, restore type stability lost to the inferencebarrier in __call_rule by
# asserting the return type, which is encoded in the MistyClosure type parameter.
@static if VERSION < v"1.11-"
    @inline function __call_rule(
        rule::DerivedPrimal{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}, args
    ) where {P,A,R,isva,nargs}
        return __call_rule_erased!(Base.inferencebarrier(rule), args)::R
    end
end

# Copy forward rule with recursively copied captures
function _copy(x::P) where {P<:DerivedPrimal}
    return P(replace_captures(x.lifted_oc, _copy(x.lifted_oc.oc.captures)), x.width)
end

_isva(::DerivedPrimal{P,T,isva,nargs}) where {P,T,isva,nargs} = isva
_nargs(::DerivedPrimal{P,T,isva,nargs}) where {P,T,isva,nargs} = nargs

# Extends functionality defined in debug_mode.jl.
function verify_args(r::DerivedPrimal{sig}, x) where {sig}
    Tx = Tuple{
        map(
            _typeof ∘ primal, __unflatten_dual_varargs(_isva(r), x, Val(_nargs(r)), r.width)
        )...,
    }
    Tx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tx do not subtype rule signature, $sig"))
end

# LazyPrimal — deferred rule for known invoke sites

mutable struct LazyPrimal{primal_sig,Trule}
    debug_mode::Bool
    mi::Core.MethodInstance
    width  # Val{N} or nothing
    rule::Trule
    function LazyPrimal(mi::Core.MethodInstance, debug_mode::Bool, width::Val=Val(1))
        interp = get_interpreter(ForwardMode)
        return new{mi.specTypes,primal_rule_type(interp, mi, width;debug_mode)}(
            debug_mode, mi, width
        )
    end
    function LazyPrimal{Tprimal_sig,Trule}(
        mi::Core.MethodInstance, debug_mode::Bool, width::Val=Val(1)
    ) where {Tprimal_sig,Trule}
        return new{Tprimal_sig,Trule}(debug_mode, mi, width)
    end
end

# Create new lazy rule with same method instance, debug mode, and width
_copy(x::P) where {P<:LazyPrimal} = P(x.mi, x.debug_mode, x.width)

# On Julia 1.10, the generic __call_rule fallback is @stable-checked and returns Any for
# LazyPrimal, triggering TypeInstabilityError when dispatch_doctor_mode = "error".
@static if VERSION < v"1.11-"
    @inline function __call_rule(
        rule::LazyPrimal{sig,DerivedPrimal{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}},
        args,
    ) where {sig,P,A,R,isva,nargs}
        return rule(args...)::R
    end
    @inline function __call_rule(
        rule::LazyPrimal{
            sig,DebugFRule{DerivedPrimal{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}}
        },
        args,
    ) where {sig,P,A,R,isva,nargs}
        return rule(args...)::R
    end
end

@inline function (rule::LazyPrimal)(args::Vararg{Any,N}) where {N}
    return isdefined(rule, :rule) ? __call_rule(rule.rule, args) : _build_rule!(rule, args)
end

@noinline function _build_rule!(rule::LazyPrimal{sig,Trule}, args) where {sig,Trule}
    interp = get_interpreter(ForwardMode)
    rule.rule = build_frule(interp, rule.mi, rule.width; debug_mode=rule.debug_mode)
    return __call_rule(rule.rule, args)
end

# DynamicPrimal — dict-cached rule for dynamic callsites

struct DynamicPrimal{V,W}
    cache::V
    debug_mode::Bool
    width::W
end

function DynamicPrimal(debug_mode::Bool, width::Val=Val(1))
    return DynamicPrimal(Dict{Any,Any}(), debug_mode, width)
end

# Create new dynamic rule with empty cache and same debug mode/width
_copy(x::P) where {P<:DynamicPrimal} = P(Dict{Any,Any}(), x.debug_mode, x.width)

function (dynamic_rule::DynamicPrimal)(args::Vararg{Any,N}) where {N}
    sig = Tuple{map(Base._stable_typeof ∘ primal, args)...}
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        interp = get_interpreter(ForwardMode)
        rule = build_frule(
            interp, sig, dynamic_rule.width; debug_mode=dynamic_rule.debug_mode
        )
        dynamic_rule.cache[sig] = rule
    end
    return __call_rule(rule, args)
end

# IR generation — lifted primal IR

struct LiftedInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
    is_used::Vector{Bool}
    debug_mode::Bool
    width  # Val{N} or nothing (nothing = legacy width-1 Dual path)
end

function generate_lifted_ir(
    interp::MooncakeInterpreter,
    sig_or_mi,
    width::Val=Val(1);
    debug_mode=false,
    do_inline=true,
    do_optimize=true,
)
    # Reset id count. This ensures that the IDs generated are the same each time this
    # function runs.
    seed_id!()

    # Grab code associated to the primal.
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    nargs = length(primal_ir.argtypes)

    # Normalise the IR.
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)

    # Keep a copy of the primal IR with the insertions
    lifted_ir = CC.copy(primal_ir)

    # Modify lifted argument types:
    # - add one for the captures in the first position, with placeholder type for now
    # - convert the rest to dual types
    for (a, P) in enumerate(primal_ir.argtypes)
        lifted_ir.argtypes[a] = lifted_type(width, CC.widenconst(P))
    end
    pushfirst!(lifted_ir.argtypes, Any)

    # Data structure for captures.
    captures = Any[]

    is_used = characterised_used_ssas(stmt(primal_ir.stmts))
    info = LiftedInfo(primal_ir, interp, is_used, debug_mode, width)
    for (n, inst) in enumerate(lifted_ir.stmts)
        ssa = SSAValue(n)
        modify_primal_stmts!(stmt(inst), lifted_ir, ssa, captures, info)
    end

    # Process new nodes etc.
    lifted_ir = CC.compact!(lifted_ir)

    CC.verify_ir(lifted_ir)

    # Now that the captured values are known, replace the placeholder value given for the
    # first argument type with the actual type.
    captures_tuple = (captures...,)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)

    # Inspection tools need the pre-optimization lifted IR, while the AD pipeline still
    # wants the optimized form by default.
    lifted_ir = do_optimize ? optimise_ir!(lifted_ir; do_inline) : lifted_ir
    return lifted_ir,
    captures_tuple,
    PrimalRuleInfo(isva, nargs, lifted_ret_type(primal_ir, width))
end

function lifted_ret_type(primal_ir::IRCode, width::Val=Val(1))
    return lifted_type(width, compute_ir_rettype(primal_ir))
end

function primal_rule_type(
    interp::MooncakeInterpreter{C}, mi::CC.MethodInstance, width::Val=Val(1); debug_mode
) where {C}
    sig = _get_sig(mi)
    if is_primitive(C, ForwardMode, sig, interp.world)
        rule = build_primitive_frule(sig)
        return debug_mode ? DebugFRule{typeof(rule)} : typeof(rule)
    end
    ir, _ = lookup_ir(interp, mi)
    nargs = length(ir.argtypes)
    isva, _ = is_vararg_and_sparam_names(mi)
    arg_types = map(CC.widenconst, ir.argtypes)
    sig = Tuple{arg_types...}
    dual_args_type = Tuple{map(Base.Fix1(lifted_type, width), arg_types)...}
    closure_type = RuleMC{dual_args_type,lifted_ret_type(ir, width)}
    Tderived = DerivedPrimal{sig,closure_type,isva,nargs,typeof(width)}
    return debug_mode ? DebugFRule{Tderived} : Tderived
end

# Statement modification — lifted primal IR rewrite

modify_primal_stmts!(::Nothing, ::IRCode, ::SSAValue, ::Vector{Any}, ::LiftedInfo) = nothing

function modify_primal_stmts!(::GotoNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::LiftedInfo)
    return nothing
end

function modify_primal_stmts!(
    stmt::GotoIfNot,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::LiftedInfo,
)
    # Extract primal boolean via _primal_of for control flow.
    Mooncake.replace_call!(lifted_ir, ssa, Expr(:call, _primal_of, inc_args(stmt).cond))

    # reinsert the GotoIfNot right after the call to _primal_of
    new_gotoifnot_inst = new_inst(Core.GotoIfNot(ssa, stmt.dest))
    CC.insert_node!(lifted_ir, ssa, new_gotoifnot_inst, ATTACH_AFTER)
    return nothing
end

function modify_primal_stmts!(
    stmt::GlobalRef,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::LiftedInfo,
)
    if isconst(stmt)
        d = const_dual!(captures, stmt, info.width)
        if d isa Int
            Mooncake.replace_call!(lifted_ir, ssa, Expr(:call, get_capture, Argument(1), d))
        else
            Mooncake.replace_call!(lifted_ir, ssa, Expr(:call, identity, d))
        end
    else
        new_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(stmt), ATTACH_BEFORE)
        zero_call = if _is_lifted_width(info.width)
            Expr(:call, Mooncake.zero_lifted, info.width, new_ssa)
        else
            Expr(:call, Mooncake.zero_dual, new_ssa)
        end
        Mooncake.replace_call!(lifted_ir, ssa, zero_call)
    end

    return nothing
end

function modify_primal_stmts!(
    stmt::ReturnNode,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::LiftedInfo,
)
    # undefined `val` field means that stmt is unreachable.
    isdefined(stmt, :val) || return nothing

    # stmt is an Argument, then already a dual, and must just be incremented.
    if stmt.val isa Union{Argument,SSAValue}
        ret_val = __inc(stmt.val)
        if _is_lifted_width(info.width)
            P_target = CC.widenconst(compute_ir_rettype(info.primal_ir))
            canon_call = Expr(:call, _canon_return, info.width, P_target, ret_val)
            ret_val = CC.insert_node!(lifted_ir, ssa, new_inst(canon_call), ATTACH_BEFORE)
        end
        Mooncake.replace_call!(lifted_ir, ssa, ReturnNode(ret_val))
        return nothing
    end

    # stmt is a const, so we have to turn it into a dual.
    d = const_dual!(captures, stmt.val, info.width)
    if d isa Int
        get_dual = Expr(:call, get_capture, Argument(1), d)
        get_dual_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(get_dual), ATTACH_BEFORE)
        Mooncake.replace_call!(lifted_ir, ssa, ReturnNode(get_dual_ssa))
    else
        Mooncake.replace_call!(lifted_ir, ssa, ReturnNode(d))
    end
    return nothing
end

function modify_primal_stmts!(
    stmt::PhiNode, lifted_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::LiftedInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = _uninit_dual(info.width, get_const_primal_value(stmt.values[n]))
    end
    set_stmt!(lifted_ir, ssa, inc_args(stmt))
    set_ir!(
        lifted_ir,
        ssa,
        :type,
        lifted_type(info.width, CC.widenconst(get_ir(lifted_ir, ssa, :type))),
    )
    return nothing
end

function modify_primal_stmts!(
    stmt::PiNode, lifted_ir::IRCode, ssa::SSAValue, ::Vector{Any}, info::LiftedInfo
)
    if stmt.val isa Union{Argument,SSAValue}
        v = __inc(stmt.val)
    else
        v = _uninit_dual(info.width, get_const_primal_value(stmt.val))
    end
    # PiNode is an "assume" — Julia treats the value as the narrowed type at
    # this code path. For the legacy bare-`Dual` path, narrowing `Dual`
    # (UnionAll) → `Dual{P_narrow, T}` is sound by UnionAll subtyping. For the
    # `Lifted{P, N, V}` boundary, however, `Lifted`'s `P` parameter is
    # invariant: a runtime `Lifted{Any, 1, V}` value is **not** a subtype of
    # the narrowed `Lifted{Int, 1, V'}`, so the assertion would be wrong and
    # the OC's compiled code emits LLVM `unreachable` that gets executed at
    # runtime ("Illegal instruction" in `pi_node_tester`). Drop the narrowing
    # for lifted slots: use `Any` as the assertion. We lose the narrowing's
    # optimization benefit at lifted widths, but correctness holds because
    # the underlying `V` already encodes the runtime concrete type.
    new_typ = if _is_lifted_width(info.width)
        Any
    else
        lifted_type(info.width, CC.widenconst(stmt.typ))
    end
    replace_call!(lifted_ir, ssa, PiNode(v, new_typ))
    return nothing
end

function modify_primal_stmts!(
    stmt::UpsilonNode,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::LiftedInfo,
)
    if !(stmt.val isa Union{Argument,SSAValue})
        stmt = UpsilonNode(_uninit_dual(info.width, get_const_primal_value(stmt.val)))
    end
    set_stmt!(lifted_ir, ssa, inc_args(stmt))
    set_ir!(
        lifted_ir,
        ssa,
        :type,
        lifted_type(info.width, CC.widenconst(get_ir(lifted_ir, ssa, :type))),
    )
    return nothing
end

function modify_primal_stmts!(
    stmt::PhiCNode,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::LiftedInfo,
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = _uninit_dual(info.width, get_const_primal_value(stmt.values[n]))
    end
    set_stmt!(lifted_ir, ssa, inc_args(stmt))
    set_ir!(
        lifted_ir,
        ssa,
        :type,
        lifted_type(info.width, CC.widenconst(get_ir(lifted_ir, ssa, :type))),
    )
    return nothing
end

@static if isdefined(Core, :EnterNode)
    function modify_primal_stmts!(
        ::Core.EnterNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::LiftedInfo
    )
        return nothing
    end
end

## Modification of IR nodes - expressions

function modify_primal_stmts!(
    stmt::Expr, lifted_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::LiftedInfo
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        sig_types = map(raw_args) do x
            t = CC.widenconst(get_forward_primal_type(info.primal_ir, x))
            return contains_bottom_type(t) ? Any : t
        end
        sig = Tuple{sig_types...}
        mi = isexpr(stmt, :invoke) ? get_mi(stmt.args[1]) : missing
        args = map(__inc, raw_args)

        # Special case: if the result of a call to getfield is un-used, then leave the
        # primal statement alone (just increment arguments as usual).
        if !info.is_used[ssa.id] && get_const_primal_value(args[1]) == getfield
            fwds = new_inst(Expr(:call, __fwds_pass_no_ad!, args...))
            replace_call!(lifted_ir, ssa, fwds)
            return nothing
        end

        # Dual-ise arguments. At width=nothing these are bare `Dual{...}`; at
        # width=Val(N) they are `Lifted{P, N, V}` — see `_uninit_dual` above.
        dual_args = map(args) do arg
            arg isa Union{Argument,SSAValue} && return arg
            return _uninit_dual(info.width, get_const_primal_value(arg))
        end

        # Resolve the rule callable and decide which dispatch path to use.
        # `is_lifted_aware` is the Phase-4 trait: the primitive's `frule!!`
        # accepts `Lifted` args and returns `Lifted` directly. When false, the
        # IR-emit inserts the Phase-3 unwrap/wrap scaffolding around the call
        # so the still-bare rule sees bare slot values.
        interp = info.interp
        is_prim = is_primitive(context_type(interp), ForwardMode, sig, interp.world)
        is_lifted_aware = is_prim && _is_lifted_aware(sig)
        needs_scaffold = _is_lifted_width(info.width) && !is_lifted_aware

        if is_prim
            rule = build_primitive_frule(sig)
            if safe_for_literal(rule)
                rule_callable = rule
            else
                push!(captures, rule)
                get_rule = Expr(:call, get_capture, Argument(1), length(captures))
                rule_callable = CC.insert_node!(
                    lifted_ir, ssa, new_inst(get_rule), ATTACH_BEFORE
                )
            end
        else
            dm = info.debug_mode
            push!(
                captures,
                if isexpr(stmt, :invoke)
                    LazyPrimal(mi, dm, info.width)
                else
                    DynamicPrimal(dm, info.width)
                end,
            )
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_callable = CC.insert_node!(
                lifted_ir, ssa, new_inst(get_rule), ATTACH_BEFORE
            )
        end

        # Unwrap Lifted args only when the rule isn't Lifted-aware. Lifted-aware
        # primitives receive `Lifted` slots directly. `Argument`/`SSAValue`
        # references get a runtime `_unlift` SSA; constants (already-built
        # `Lifted` values) are unwrapped at IR-emit time.
        rule_args = if needs_scaffold
            map(dual_args) do arg
                arg isa Union{Argument,SSAValue} &&
                    return _insert_unlift!(lifted_ir, ssa, arg)
                return _unlift(arg)
            end
        else
            dual_args
        end

        # Emit the rule call. Lifted-aware primitives and non-lifted widths
        # replace the original SSA directly. Otherwise, wrap the bare rule
        # result via `_wrap_rule_result`, which canonicalises V to match
        # `dual_type(Val(N), primal_retype)` (e.g. routes bare
        # `Dual{Float64, Float64}` through the 2-arg ctor to produce
        # `Lifted{Float64, N, NDual{Float64, N}}` instead of
        # `Lifted{Float64, N, Dual{Float64, Float64}}`).
        if needs_scaffold
            primal_retype = let t = CC.widenconst(get_ir(info.primal_ir, ssa, :type))
                # Fallback to `Any` when `t` is `Bottom` or contains an
                # unbound `TypeVar` (e.g. a `Type{AbstractArray{a, 1}}`
                # constructed via `Core.apply_type` / `UnionAll`); Julia's
                # static parameter binding chokes on TypeVars in dispatch.
                if contains_bottom_type(t) || Base.has_free_typevars(t)
                    Any
                else
                    t
                end
            end
            rule_call_inst = new_inst(Expr(:call, rule_callable, rule_args...))
            rule_result_ssa = CC.insert_node!(lifted_ir, ssa, rule_call_inst, ATTACH_BEFORE)
            replace_call!(
                lifted_ir,
                ssa,
                Expr(:call, _wrap_rule_result, primal_retype, info.width, rule_result_ssa),
            )
        else
            replace_call!(lifted_ir, ssa, Expr(:call, rule_callable, rule_args...))
        end
    elseif isexpr(stmt, :boundscheck)
        # Keep the boundscheck, but wrap it as a slot value (Dual at legacy
        # width, Lifted at Val(N)).
        inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        bc_ssa = CC.insert_node!(lifted_ir, ssa, inst, ATTACH_BEFORE)
        zero_call = if _is_lifted_width(info.width)
            Expr(:call, zero_lifted, info.width, bc_ssa)
        else
            Expr(:call, zero_dual, bc_ssa)
        end
        replace_call!(lifted_ir, ssa, zero_call)
    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(lifted_ir, ssa, nothing)
    elseif Meta.isexpr(stmt, :copyast)
        new_copyast_inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        new_copyast_ssa = CC.insert_node!(lifted_ir, ssa, new_copyast_inst, ATTACH_BEFORE)
        zero_call = if _is_lifted_width(info.width)
            Expr(:call, zero_lifted, info.width, new_copyast_ssa)
        else
            Expr(:call, zero_dual, new_copyast_ssa)
        end
        replace_call!(lifted_ir, ssa, zero_call)
    elseif Meta.isexpr(stmt, :loopinfo)
        # Leave this node alone.
    elseif isexpr(stmt, :throw_undef_if_not)
        # args[1] is a Symbol, args[2] is the condition which must be primalized
        primal_cond = Expr(:call, _primal_of, inc_args(stmt).args[2])
        replace_call!(lifted_ir, ssa, primal_cond)
        new_undef_inst = new_inst(Expr(:throw_undef_if_not, stmt.args[1], ssa))
        CC.insert_node!(lifted_ir, ssa, new_undef_inst, ATTACH_AFTER)
    elseif isexpr(stmt, :enter)
        # Leave this node alone
    elseif isexpr(stmt, :leave)
        # Leave this node alone
    elseif isexpr(stmt, :pop_exception)
        # Leave this node alone
    else
        msg = "Expressions of type `:$(stmt.head)` are not yet supported in forward mode"
        throw(ArgumentError(msg))
    end
    return nothing
end
