module MooncakeFunctionWrappersExt

using Random
using FunctionWrappers: FunctionWrapper
import Mooncake:
    TestUtils,
    _add_to_primal_internal,
    _zero_dual_internal,
    _uninit_dual_internal,
    _randn_dual_internal,
    tangent_to_primal_internal!!,
    primal_to_tangent_internal!!,
    _dot_internal,
    _scale_internal,
    __verify_fdata_value,
    fdata,
    has_equal_data_internal,
    increment_internal!!,
    randn_tangent_internal,
    set_to_zero_internal!!,
    zero_tangent_internal,
    build_frule,
    build_rrule,
    dual_type,
    fcodual_type,
    fdata_type,
    frule!!,
    get_interpreter,
    increment_rdata!!,
    primal,
    pullback_type,
    rdata_type,
    tangent,
    tangent_type,
    to_cr_tangent,
    rdata,
    rrule!!,
    zero_tangent,
    @foldable,
    @is_primitive,
    CoDual,
    ForwardMode,
    IncCache,
    Lifted,
    MaybeCache,
    MinimalCtx,
    NoRData,
    SetToZeroCache,
    Stack,
    lift,
    lifted_type,
    randn_lifted,
    uninit_lifted,
    zero_lifted

# Tangent type for FunctionWrapper. Also serves as fdata since FunctionWrapper is mutable.
# Fields:
# - fwds_wrapper: reverse-mode forward pass (OpaqueClosure capturing rrule)
# - frule_wrapper: forward-mode rule (OpaqueClosure capturing frule)
# - dobj_ref: tangent of the wrapped callable
mutable struct FunctionWrapperTangent{Tfwds_oc,Tfrule_oc}
    fwds_wrapper::Tfwds_oc
    frule_wrapper::Tfrule_oc
    dobj_ref::Ref
end

function _construct_rrule_types(R, A)
    # Convert signature into a tuple of types.
    primal_arg_types = (A.parameters...,)

    # Signature and OpaqueClosure type for reverse pass.
    rvs_sig = Tuple{rdata_type(tangent_type(R))}
    primal_rdata_sig = Tuple{map(rdata_type ∘ tangent_type, primal_arg_types)...}
    pb_ret_type = Tuple{NoRData,primal_rdata_sig.parameters...}
    rvs_oc_type = Core.OpaqueClosure{rvs_sig,pb_ret_type}

    # Signature and OpaqueClosure type for forwards pass.
    fwd_sig = Tuple{map(fcodual_type, primal_arg_types)...}
    fwd_oc_type = Core.OpaqueClosure{fwd_sig,Tuple{fcodual_type(R),rvs_oc_type}}
    return fwd_oc_type, rvs_oc_type, fwd_sig, rvs_sig
end

function _construct_frule_types(R, A, (::Val{N})=Val(1)) where {N}
    # Convert signature into a tuple of types.
    primal_arg_types = (A.parameters...,)

    # Signature and OpaqueClosure type for the forward pass. `build_frule` returns a
    # Lifted-dispatched callable, so the OC signature uses width-`N` lifted_type for each arg
    # and return. The reverse tangent uses N = 1 (its `frule_wrapper` field is never run);
    # the forward V (`dual_type(Val(N), ...)`) uses the chunk width N.
    fwd_sig = Tuple{map(P -> lifted_type(Val(N), P), primal_arg_types)...}
    fwd_ret_type = lifted_type(Val(N), R)
    fwd_oc_type = Core.OpaqueClosure{fwd_sig,fwd_ret_type}
    return fwd_oc_type, fwd_sig, fwd_ret_type
end

@foldable function tangent_type(::Type{FunctionWrapper{R,A}}) where {R,A<:Tuple}
    rrule_oc_type = _construct_rrule_types(R, A)[1]
    frule_oc_type = _construct_frule_types(R, A)[1]
    return FunctionWrapperTangent{rrule_oc_type,frule_oc_type}
end

# Forward-mode canonical V reuses the `FunctionWrapperTangent` struct (it serves as its own
# fdata and abstracts the wrapper's `Ptr`/`captures::Any` fields behind OpaqueClosures), but at
# chunk width N its `frule_wrapper` OpaqueClosure is built at `chunk_size = N`. So the forward V
# `dual_type(Val(N), …)` is `FunctionWrapperTangent{rrule_oc, frule_oc_N}` — a distinct concrete
# type from the width-1 reverse `tangent_type` (whose `frule_wrapper` field is never run). The
# forward seed factories build it (carved out below so the generic structural lift does not recurse
# into the wrapper's `Ptr` fields); its `frule_wrapper` closes over the obj's width-N forward slot.
@foldable function dual_type(::Val{N}, ::Type{FunctionWrapper{R,A}}) where {N,R,A<:Tuple}
    rrule_oc_type = _construct_rrule_types(R, A)[1]
    frule_oc_type = _construct_frule_types(R, A, Val(N))[1]
    return FunctionWrapperTangent{rrule_oc_type,frule_oc_type}
end
@foldable dual_type(::Val{N}, ::Type{P}) where {N,P<:FunctionWrapper} =
    FunctionWrapperTangent
lift(x::FunctionWrapper, ẋ::FunctionWrapperTangent) = lift(x, ẋ, nothing)
function lift(x::FunctionWrapper, ẋ::FunctionWrapperTangent, ::Union{Nothing,IdDict})
    return Lifted{typeof(x),1,typeof(ẋ)}(x, ẋ)
end
# A `FunctionWrapperTangent` wraps an OpaqueClosure tangent with all `N` lanes baked in,
# so it cannot be decomposed per lane; at `N == 1` the whole tangent IS lane 1. Wider
# slots fail loudly rather than hand every lane the full width-`N` tangent.
@inline function tangent(
    x::Lifted{P,N,<:FunctionWrapperTangent}, ::Integer
) where {P<:FunctionWrapper,N}
    N == 1 || throw(
        ArgumentError(
            "per-lane tangent extraction is unsupported for a width-$N FunctionWrapper " *
            "slot: its FunctionWrapperTangent bakes all $N lanes into one OpaqueClosure.",
        ),
    )
    return x.value
end

import .TestUtils: has_equal_data_internal
function has_equal_data_internal(
    p::P, q::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P<:FunctionWrapper}
    return has_equal_data_internal(p.obj, q.obj, equal_undefs, d)
end
function has_equal_data_internal(
    t::T, s::T, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T<:FunctionWrapperTangent}
    return has_equal_data_internal(t.dobj_ref[], s.dobj_ref[], equal_undefs, d)
end

# Shared reverse-pass construction: the rrule forwards-pass OpaqueClosure (which also closes over
# the reverse-pass) and the obj-tangent ref. Deliberately does NOT build the frule — both callers
# build their own (`_function_wrapper_tangent` at width 1, `_function_wrapper_forward_tangent` at
# width N), so the forward path no longer wastes a discarded width-1 frule build on every tangent.
function _function_wrapper_reverse_parts(R, obj::Tobj, A, obj_tangent) where {Tobj}

    # Analyse types for rrule.
    rrule_fwd_oc_type, rvs_oc_type, _, _ = _construct_rrule_types(R, A)
    (fwd_sig, fwd_ret) = rrule_fwd_oc_type.parameters
    (rvs_sig, rvs_ret) = rvs_oc_type.parameters

    # Construct reference to obj_tangent that we can read / write-to.
    obj_tangent_ref = Ref{tangent_type(Tobj)}(obj_tangent)

    # Construct rrule for `obj`, applied to its declared argument types.
    rrule = build_rrule(Tuple{Tobj,A.parameters...})

    # Construct stack which can hold pullbacks generated by `rrule`. The forwards-pass will
    # run `rrule` and push the pullback to `pb_stack`. The reverse-pass will pop and run it.
    pb_stack = Stack{pullback_type(typeof(rrule), (Tobj, A.parameters...))}()

    # Construct reverse-pass. Note: this closes over `pb_stack`.
    @static if VERSION ≥ v"1.12-"
        run_rvs_pass = Base.Experimental.@opaque rvs_sig -> rvs_ret dy -> begin
            obj_rdata, dx... = pop!(pb_stack)(dy)
            obj_tangent_ref[] = increment_rdata!!(obj_tangent_ref[], obj_rdata)
            return NoRData(), dx...
        end
    else
        run_rvs_pass = Base.Experimental.@opaque rvs_sig dy -> begin
            obj_rdata, dx... = pop!(pb_stack)(dy)
            obj_tangent_ref[] = increment_rdata!!(obj_tangent_ref[], obj_rdata)
            return NoRData(), dx...
        end
    end

    # Construct fowards-pass for rrule. Note: this closes over the reverse-pass and `pb_stack`.
    @static if VERSION ≥ v"1.12-"
        run_fwds_pass = Base.Experimental.@opaque fwd_sig -> fwd_ret (x...) -> begin
            y, pb = rrule(CoDual(obj, fdata(obj_tangent_ref[])), x...)
            push!(pb_stack, pb)
            return y, run_rvs_pass
        end
    else
        run_fwds_pass = Base.Experimental.@opaque fwd_sig (x...) -> begin
            y, pb = rrule(CoDual(obj, fdata(obj_tangent_ref[])), x...)
            push!(pb_stack, pb)
            return y, run_rvs_pass
        end
    end

    return run_fwds_pass, obj_tangent_ref
end

# Reverse (and FD) tangent: shared reverse parts plus a *width-1* frule. The `frule_wrapper` field
# is never run by reverse mode, but it is read (copied) by `_scale_internal`, so it must be a valid
# OpaqueClosure of the declared type — hence the width-1 build here.
function _function_wrapper_tangent(R, obj::Tobj, A, obj_tangent) where {Tobj}
    run_fwds_pass, obj_tangent_ref = _function_wrapper_reverse_parts(R, obj, A, obj_tangent)

    # Forward-pass wrapper for the width-1 frule. Post-cutover `frule` is Lifted-dispatched; wrap
    # the closure obj + current obj_tangent into a width-1 Lifted slot via the boundary helper.
    _, frule_sig, frule_ret = _construct_frule_types(R, A)
    frule = build_frule(get_interpreter(ForwardMode), Tuple{Tobj,A.parameters...})
    @static if VERSION ≥ v"1.12-"
        run_frule = Base.Experimental.@opaque frule_sig -> frule_ret (x...) -> begin
            return frule(lift(obj, obj_tangent_ref[]), x...)
        end
    else
        run_frule = Base.Experimental.@opaque frule_sig (x...) -> begin
            return frule(lift(obj, obj_tangent_ref[]), x...)
        end
    end

    t = FunctionWrapperTangent(run_fwds_pass, run_frule, obj_tangent_ref)
    return t, obj_tangent_ref
end

# Forward-mode V builder for chunk width N. `obj_fwd_slot` is the wrapped object's width-N forward
# slot — supplied with N independent lane directions either by a forward seed factory
# (`zero/uninit/randn_lifted(Val(N), obj)`) or by the FunctionWrapper-construction `frule!!` (where it
# is the lifted constructor argument). The frule OpaqueClosure is built at `chunk_size = N` and closes
# over `obj_fwd_slot`. The reverse fields (`fwds_wrapper`, `dobj_ref`) are reused from the width-1
# builder so the V type matches `dual_type`; `dobj_ref` is the slot's lane-1 direction, keeping width-1
# FD consistent with AD (at width 1 the slot has exactly that one direction).
function _function_wrapper_forward_tangent(
    R, A, obj_fwd_slot::Lifted{Tobj,N}, ::Val{N}
) where {Tobj,N}
    obj = primal(obj_fwd_slot)
    # Use the shared reverse parts directly (not `_function_wrapper_tangent`) so we do not build a
    # width-1 frule only to discard it — this path builds the width-N frule below.
    run_fwds_pass, obj_tangent_ref = _function_wrapper_reverse_parts(
        R, obj, A, tangent(obj_fwd_slot, 1)
    )
    _, frule_sig, frule_ret = _construct_frule_types(R, A, Val(N))
    frule = build_frule(
        get_interpreter(ForwardMode), Tuple{Tobj,A.parameters...}; chunk_size=N
    )
    run_frule = @static if VERSION ≥ v"1.12-"
        Base.Experimental.@opaque frule_sig -> frule_ret (x...) -> frule(obj_fwd_slot, x...)
    else
        Base.Experimental.@opaque frule_sig (x...) -> frule(obj_fwd_slot, x...)
    end
    return FunctionWrapperTangent(run_fwds_pass, run_frule, obj_tangent_ref)
end

function zero_tangent_internal(p::FunctionWrapper{R,A}, dict::MaybeCache) where {R,A}

    # If we've seen this primal before, then we must return that tangent.
    haskey(dict, p) && return dict[p]::tangent_type(typeof(p))

    # We have not seen this primal before, create it and log it.
    obj_tangent = zero_tangent_internal(p.obj[], dict)
    t, _ = _function_wrapper_tangent(R, p.obj[], A, obj_tangent)
    dict === nothing || setindex!(dict, t, p)
    return t
end

function randn_tangent_internal(
    rng::AbstractRNG, p::FunctionWrapper{R,A}, dict::MaybeCache
) where {R,A}

    # If we've seen this primal before, then we must return that tangent.
    haskey(dict, p) && return dict[p]::tangent_type(typeof(p))

    # We have not seen this primal before, create it and log it.
    obj_tangent = randn_tangent_internal(rng, p.obj[], dict)
    t, _ = _function_wrapper_tangent(R, p.obj[], A, obj_tangent)
    dict === nothing || setindex!(dict, t, p)
    return t
end

# Forward seed factories: build the width-N forward V. The wrapped object's width-N forward slot
# (with N independent lane directions) comes from the matching lifted seed factory on `obj`.
function _zero_dual_internal(::Val{N}, p::FunctionWrapper{R,A}, d::MaybeCache) where {N,R,A}
    haskey(d, p) && return d[p]::dual_type(Val(N), typeof(p))
    t = _function_wrapper_forward_tangent(R, A, zero_lifted(Val(N), p.obj[]), Val(N))
    d[p] = t
    return t
end
function _uninit_dual_internal(
    ::Val{N}, p::FunctionWrapper{R,A}, d::MaybeCache
) where {N,R,A}
    haskey(d, p) && return d[p]::dual_type(Val(N), typeof(p))
    t = _function_wrapper_forward_tangent(R, A, uninit_lifted(Val(N), p.obj[]), Val(N))
    d[p] = t
    return t
end
function _randn_dual_internal(
    ::Val{N}, rng::AbstractRNG, p::FunctionWrapper{R,A}, d::MaybeCache
) where {N,R,A}
    haskey(d, p) && return d[p]::dual_type(Val(N), typeof(p))
    t = _function_wrapper_forward_tangent(R, A, randn_lifted(Val(N), rng, p.obj[]), Val(N))
    d[p] = t
    return t
end

function increment_internal!!(c::IncCache, t::T, s::T) where {T<:FunctionWrapperTangent}
    t.dobj_ref[] = increment_internal!!(c, t.dobj_ref[], s.dobj_ref[])
    return t
end

function set_to_zero_internal!!(c::SetToZeroCache, t::FunctionWrapperTangent)
    t.dobj_ref[] = set_to_zero_internal!!(c, t.dobj_ref[])
    return t
end

function _add_to_primal_internal(
    c::MaybeCache, p::FunctionWrapper, t::FunctionWrapperTangent, unsafe::Bool
)
    return typeof(p)(_add_to_primal_internal(c, p.obj[], t.dobj_ref[], unsafe))
end

function tangent_to_primal_internal!!(
    p::FunctionWrapper, t::FunctionWrapperTangent, c::MaybeCache
)
    haskey(c, p) && return c[p]::typeof(p)
    c[p] = p
    p.obj[] = tangent_to_primal_internal!!(p.obj[], t.dobj_ref[], c)
    return p
end
function primal_to_tangent_internal!!(
    t::FunctionWrapperTangent, p::FunctionWrapper, c::MaybeCache
)
    haskey(c, p) && return c[p]::typeof(t)
    c[p] = t
    t.dobj_ref[] = primal_to_tangent_internal!!(t.dobj_ref[], p.obj[], c)
    return t
end

function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:FunctionWrapperTangent}
    return _dot_internal(c, t.dobj_ref[], s.dobj_ref[])::Float64
end

function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:FunctionWrapperTangent}
    return T(t.fwds_wrapper, t.frule_wrapper, Ref(_scale_internal(c, a, t.dobj_ref[])))
end

# `FunctionWrapperTangent` is not field-parallel to `FunctionWrapper` (its fields are opaque
# closures + a `dobj_ref`, not the wrapper's `ptr`/`obj`/… fields), so the standardised
# field-access interaction tests (`getfield`/`_new_`/`setfield!`/…) do not apply — AD of a
# FunctionWrapper happens at the call/construction level, never via field access.
TestUtils.supports_field_access_interactions(::Type{<:FunctionWrapper}) = false

function TestUtils.populate_address_map_internal(
    m::TestUtils.AddressMap, p::FunctionWrapper, t::FunctionWrapperTangent
)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    return m
end

fdata_type(T::Type{<:FunctionWrapperTangent}) = T
rdata_type(::Type{FunctionWrapperTangent}) = NoRData
@foldable tangent_type(F::Type{<:FunctionWrapperTangent}, ::Type{NoRData}) = F
tangent(f::FunctionWrapperTangent, ::NoRData) = f

function __verify_fdata_value(
    ::IdDict{Any,Nothing}, p::FunctionWrapper, t::FunctionWrapperTangent
)
    return nothing
end

# Will: to the best of my knowledge, no one has ever actually worked with FunctionWrappers
# before in the ChainRules ecosystem. Consequently, it shouldn't matter what type we use
# here. We might need to revise this is people start making use of FunctionWrappers in a
# meaningful way inside of ChainRules, but it seems unlikely that this will ever happen.
to_cr_tangent(t::FunctionWrapperTangent) = t

@is_primitive MinimalCtx Tuple{Type{<:FunctionWrapper},Any}
function rrule!!(::CoDual{Type{FunctionWrapper{R,A}}}, obj::CoDual{P}) where {R,A,P}
    t, obj_tangent_ref = _function_wrapper_tangent(R, obj.x, A, zero_tangent(obj.x, obj.dx))
    function_wrapper_pb(::NoRData) = NoRData(), rdata(obj_tangent_ref[])
    return CoDual(FunctionWrapper{R,A}(obj.x), t), function_wrapper_pb
end

function frule!!(::Lifted{Type{FunctionWrapper{R,A}},N}, obj::Lifted{P,N}) where {R,A,P,N}
    # `obj` is already the wrapped object's width-N forward slot, so the constructed
    # FunctionWrapper's forward V closes its frule over it directly.
    y = FunctionWrapper{R,A}(primal(obj))
    return Lifted{typeof(y),N}(y, _function_wrapper_forward_tangent(R, A, obj, Val(N)))
end

@is_primitive MinimalCtx Tuple{<:FunctionWrapper,Vararg}
function rrule!!(f::CoDual{<:FunctionWrapper}, x::Vararg{CoDual})
    y, pb = f.dx.fwds_wrapper(x...)
    function_wrapper_eval_pb(dy) = pb(dy)
    return y, function_wrapper_eval_pb
end

function frule!!(f::Lifted{FunctionWrapper{R,A},N}, x::Vararg{Lifted,M}) where {R,A,N,M}
    return tangent(f).frule_wrapper(x...)
end

end
