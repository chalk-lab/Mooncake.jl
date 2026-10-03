module MooncakeFunctionWrappersExt

using Random
using Mooncake: Mooncake
using FunctionWrappers: FunctionWrapper, reinit_wrapper
import Mooncake:
    TestUtils,
    _tangent_fields,
    _basis_seed!!,
    _materialise_lane,
    friendly_tangent_cache,
    FriendlyTangentCache,
    AsRaw,
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
    lifted_type

# Tangent type for FunctionWrapper. Also serves as fdata since FunctionWrapper is mutable.
# Fields:
# - fwds_wrapper: reverse-mode forward pass (OpaqueClosure capturing rrule)
# - frule_wrapper: forward-mode rule, absent for native tangents
# - dobj_ref: native tangent, or the cached forward slot, of the wrapped callable
# - frule / rrule / pb_type: compiled rules and pullback type retained for reuse
mutable struct FunctionWrapperTangent{R,A,N,Tfwds_oc,Tfrule_oc}
    fwds_wrapper::Tfwds_oc
    frule_wrapper::Tfrule_oc
    dobj_ref::Ref
    frule::Any
    rrule::Any
    pb_type::Type
end
function FunctionWrapperTangent{R,A,N}(fwds, fwd, ref, frule, rrule, pb_type) where {R,A,N}
    return FunctionWrapperTangent{R,A,N,typeof(fwds),typeof(fwd)}(
        fwds, fwd, ref, frule, rrule, pb_type
    )
end

# Native storage never changes representation; forward slots are made by lift.
mutable struct CallableTangentRef{P,T} <: Ref{T}
    obj::P
    value::T
end
_tangent_fields(::Type{<:CallableTangentRef}) = (:value,)
Base.getindex(r::CallableTangentRef) = r.value
Base.setindex!(r::CallableTangentRef, value) = (r.value = value)

function Mooncake._slot_primal(
    ::FunctionWrapper, ::FunctionWrapperTangent{R,A,0}
) where {R,A}
    throw(
        ArgumentError("native FunctionWrapper tangents cannot be forward storage; use lift")
    )
end

# Keep callable type recovery outside the opaque closure for Julia 1.11 inference.
@inline function _function_wrapper_object(
    fw, ::Union{CallableTangentRef{P},Lifted{P}}
) where {P}
    ref = fw.obj
    ref isa Base.RefValue{P} && return ref[]
    obj = ref[]
    obj isa P || throw(
        ArgumentError(
            "cannot rebind FunctionWrapper storage to a different callable type; use lift with the new primal",
        ),
    )
    return obj::P
end

_tangent_fields(::Type{<:FunctionWrapperTangent}) = (:dobj_ref,)

function Mooncake.tangent_dim(
    v::FunctionWrapperTangent{R,A,N}, seen::IdDict{Any,Any}
) where {R,A,N}
    haskey(seen, v) && return 0
    seen[v] = nothing
    return Mooncake.tangent_dim(N == 0 ? v.dobj_ref : tangent(v.dobj_ref[], 1), seen)
end

function Mooncake._check_tangent_storage!(
    seen::IdDict, expected::FunctionWrapperTangent{R,A,N}, supplied
) where {R,A,N}
    supplied isa typeof(expected) || throw(
        ArgumentError(
            "incompatible FunctionWrapper tangent storage; use lift for forward slots and native tangents for directions",
        ),
    )
    if N != 0 && haskey(seen, expected) && seen[expected] !== supplied
        # Only zero-dimensionality matters here; do not walk the callable's primal metadata.
        Mooncake.tangent_dim(tangent(expected.dobj_ref[], 1)) == 0 ||
            Mooncake._throw_shared_input_tangent_error()
        return nothing
    end
    return invoke(
        Mooncake._check_tangent_storage!, Tuple{IdDict,Any,Any}, seen, expected, supplied
    )
end

# The primal's pointer and type metadata have no friendly derivative representation.
friendly_tangent_cache(::FunctionWrapper) = FriendlyTangentCache{AsRaw}(nothing)

function _basis_seed!!(
    v::FunctionWrapperTangent{R,A,W}, slots::NTuple{N,Int}, cursor, dict
) where {R,A,W,N}
    W == 0 &&
        throw(ArgumentError("use lift before seeding FunctionWrapper forward storage"))
    haskey(dict, v) && return dict[v]
    dict[v] = v
    ref = v.dobj_ref
    haskey(dict, ref) && return v
    dict[ref] = ref
    ref[] = _basis_seed!!(ref[], slots, cursor, dict)
    return v
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
    fwd_sig = Tuple{FunctionWrapper{R,A},map(fcodual_type, primal_arg_types)...}
    fwd_oc_type = Core.OpaqueClosure{fwd_sig,Tuple{fcodual_type(R),rvs_oc_type}}
    return fwd_oc_type, rvs_oc_type, fwd_sig, rvs_sig
end

function _construct_frule_types(R, A, (::Val{N})=Val(1)) where {N}
    # Convert signature into a tuple of types.
    primal_arg_types = (A.parameters...,)

    fwd_sig = Tuple{
        FunctionWrapper{R,A},map(P -> lifted_type(Val(N), P), primal_arg_types)...
    }
    fwd_ret_type = lifted_type(Val(N), R)
    fwd_oc_type = Core.OpaqueClosure{fwd_sig,fwd_ret_type}
    return fwd_oc_type, fwd_sig, fwd_ret_type
end

@foldable function tangent_type(::Type{FunctionWrapper{R,A}}) where {R,A<:Tuple}
    rrule_oc_type = _construct_rrule_types(R, A)[1]
    return FunctionWrapperTangent{R,A,0,rrule_oc_type,Nothing}
end

# Forward V hides Ptr/capture fields behind OpaqueClosures, as the reverse tangent does,
# but its frule_wrapper closes over a width-N slot and has a width-N signature.
@foldable function dual_type(::Val{N}, ::Type{FunctionWrapper{R,A}}) where {N,R,A<:Tuple}
    rrule_oc_type = _construct_rrule_types(R, A)[1]
    frule_oc_type = _construct_frule_types(R, A, Val(N))[1]
    return FunctionWrapperTangent{R,A,N,rrule_oc_type,frule_oc_type}
end
@foldable dual_type(::Val{N}, ::Type{P}) where {N,P<:FunctionWrapper} =
    FunctionWrapperTangent
lift(x::FunctionWrapper, ẋ::FunctionWrapperTangent) = lift(x, ẋ, nothing)
function lift(
    x::FunctionWrapper{R,A}, ẋ::FunctionWrapperTangent, c::Union{Nothing,IdDict}
) where {R,A}
    ẋ isa FunctionWrapperTangent{R,A,0} || throw(
        ArgumentError(
            "lift requires a native FunctionWrapper tangent; use tangent(slot, lane) to materialise one",
        ),
    )
    d = c === nothing ? IdDict() : c
    haskey(d, x) && return d[x]::lifted_type(Val(1), typeof(x))
    obj = lift(x.obj[], ẋ.dobj_ref[], d)
    v = _function_wrapper_forward_tangent(R, A, obj, Val(1), ẋ)
    return d[x] = Lifted{typeof(x),1}(x, v)::lifted_type(Val(1), typeof(x))
end
# Materialise a native tangent from the selected lane of derivative storage.
@inline function tangent(
    x::Lifted{P,N,<:FunctionWrapperTangent}, lane::Integer
) where {P<:FunctionWrapper,N}
    return _materialise_lane(x, lane, IdDict{Any,Any}())
end
function _materialise_lane(
    x::Lifted{FunctionWrapper{R,A},N,<:FunctionWrapperTangent}, lane::Integer, cache::IdDict
) where {R,A,N}
    haskey(cache, primal(x)) && return cache[primal(x)]
    obj = x.rep.dobj_ref[]
    t = _function_wrapper_tangent(R, primal(obj), A, _materialise_lane(obj, lane, cache))[1]
    return cache[primal(x)] = t
end

import .TestUtils: has_equal_data_internal
function has_equal_data_internal(
    p::P, q::P, equal_undefs::Bool, d::IdDict{Any,Bool}
) where {P<:FunctionWrapper}
    return has_equal_data_internal(p.obj, q.obj, equal_undefs, d)
end
function has_equal_data_internal(
    t::T, s::T, equal_undefs::Bool, d::IdDict{Any,Bool}
) where {T<:FunctionWrapperTangent}
    return has_equal_data_internal(t.dobj_ref[], s.dobj_ref[], equal_undefs, d)
end

# Before Julia 1.12, @opaque cannot specify return bounds. Reconstruct a mismatched
# closure with equal bounds, so abstract signatures still have canonical storage types.
function _wrapper_closure(c::Core.OpaqueClosure{A}, ::Type{R}) where {A,R}
    src, _ = Mooncake.CC.typeinf_code(
        Mooncake.CC.NativeInterpreter(UInt(c.world)),
        c.source,
        Tuple{typeof(c.captures),A.parameters...},
        Core.svec(),
        true,
    )
    return Base.Experimental.generate_opaque_closure(
        A, R, R, src, Int(c.source.nargs) - 1, c.source.isva, c.captures...
    )::Core.OpaqueClosure{A,R}
end

# Share reverse parts without building a frule: callers choose width 1 or N.
@inline function _function_wrapper_reverse_parts(
    ::Type{R}, obj::Tobj, ::Type{A}, obj_tangent, source=nothing
) where {R,Tobj,A}

    # Analyse types for rrule.
    rrule_fwd_oc_type, rvs_oc_type, _, _ = _construct_rrule_types(R, A)
    (fwd_sig, fwd_ret) = rrule_fwd_oc_type.parameters
    (rvs_sig, rvs_ret) = rvs_oc_type.parameters

    # Construct reference to obj_tangent that we can read / write-to.
    obj_tangent_ref = CallableTangentRef(obj, obj_tangent)

    rrule = source === nothing ? build_rrule(Tuple{Tobj,A.parameters...}) : source.rrule
    PB = if source === nothing
        pullback_type(typeof(rrule), (Tobj, A.parameters...))
    else
        source.pb_type
    end

    # Forward-only wrappers never need a pullback stack. Allocate it on the first
    # reverse-mode call, then retain it across calls.
    pb_stack = Ref{Stack{PB}}()

    # Construct reverse-pass. Note: this closes over `pb_stack`.
    @static if VERSION ≥ v"1.12-"
        run_rvs_pass = Base.Experimental.@opaque rvs_sig -> rvs_ret dy -> begin
            obj_rdata, dx... = pop!(pb_stack[])(dy)
            obj_tangent_ref[] = increment_rdata!!(obj_tangent_ref[], obj_rdata)
            return NoRData(), dx...
        end
    else
        run_rvs_pass = Base.Experimental.@opaque rvs_sig dy -> begin
            obj_rdata, dx... = pop!(pb_stack[])(dy)
            obj_tangent_ref[] = increment_rdata!!(obj_tangent_ref[], obj_rdata)
            return NoRData(), dx...
        end
    end

    canonical_rvs_pass = if run_rvs_pass isa rvs_oc_type
        run_rvs_pass
    else
        _wrapper_closure(run_rvs_pass, rvs_ret)
    end

    # Construct fowards-pass for rrule. Note: this closes over the reverse-pass and `pb_stack`.
    @static if VERSION ≥ v"1.12-"
        run_fwds_pass = Base.Experimental.@opaque fwd_sig -> fwd_ret (args...) -> begin
            y, pb = rrule(
                CoDual(
                    _function_wrapper_object(args[1], obj_tangent_ref),
                    fdata(obj_tangent_ref[]),
                ),
                Base.tail(args)...,
            )
            isassigned(pb_stack) || (pb_stack[] = eltype(pb_stack)())
            push!(pb_stack[], pb)
            return y, canonical_rvs_pass
        end
    else
        run_fwds_pass = Base.Experimental.@opaque fwd_sig (args...) -> begin
            y, pb = rrule(
                CoDual(
                    _function_wrapper_object(args[1], obj_tangent_ref),
                    fdata(obj_tangent_ref[]),
                ),
                Base.tail(args)...,
            )
            isassigned(pb_stack) || (pb_stack[] = eltype(pb_stack)())
            push!(pb_stack[], pb)
            return y, canonical_rvs_pass
        end
    end

    canonical_fwds_pass = if run_fwds_pass isa rrule_fwd_oc_type
        run_fwds_pass
    else
        _wrapper_closure(run_fwds_pass, fwd_ret)
    end
    return canonical_fwds_pass, obj_tangent_ref, rrule, PB
end

function _function_wrapper_tangent(
    ::Type{R}, obj::Tobj, ::Type{A}, obj_tangent, source=nothing
) where {R,Tobj,A}
    run_fwds_pass, obj_tangent_ref, rrule, PB = _function_wrapper_reverse_parts(
        R, obj, A, obj_tangent, source
    )
    frule = if source === nothing
        build_frule(get_interpreter(ForwardMode), Tuple{Tobj,A.parameters...})
    else
        source.frule
    end
    t = FunctionWrapperTangent{R,A,0}(
        run_fwds_pass, nothing, obj_tangent_ref, frule, rrule, PB
    )
    return t, obj_tangent_ref
end

# A reused V keeps its partials, but scalar dual values must follow the slot's callable.
@inline function _callable_slot(obj::P, slot::Lifted{P,N}) where {P,N}
    obj === primal(slot) && return slot
    return Lifted{P,N}(obj, _callable_rep(obj, primal(slot), tangent(slot)))
end
@inline function _callable_rep(p, old, v)
    p === old && return v
    throw(
        ArgumentError(
            "cannot rebind this FunctionWrapper capture; use lift with the new primal"
        ),
    )
end
@inline _callable_rep(p, old, v::Mooncake.NoDual) = v
@inline _callable_rep(p, old, v::Mooncake.Nfwd.NDual) = typeof(v)(p, v.partials)
@inline _callable_rep(p, old, v::Mooncake.Nfwd.NDualArray) = typeof(v)(
    p, getfield(v, :partials_block)
)
@inline _callable_rep(p, old, v::Mooncake.Nfwd.NDualRef) = v
@inline function _callable_rep(p, old, v::Union{Tuple,NamedTuple})
    return map((p, old, v) -> _callable_rep(p, old, v), p, old, v)
end
@inline function _callable_rep(
    p, old, v::Union{Mooncake.ImmutableDual,Mooncake.MutableDual}
)
    p === old && return v
    fields = map(keys(v.fields)) do name
        _callable_rep(getfield(p, name), getfield(old, name), getfield(v.fields, name))
    end
    nt = NamedTuple{keys(v.fields)}(fields)
    v isa Mooncake.ImmutableDual && return typeof(v)(nt)
    v.fields = nt
    return v
end

# Keep the slot in dobj_ref so basis seeding updates the storage read by the rule.
# Native lifting can reuse its compiled rule and reverse closures.
function _function_wrapper_forward_tangent(
    ::Type{R}, ::Type{A}, obj_fwd_slot::Lifted{Tobj,N}, ::Val{N}, source=nothing
) where {R,A,Tobj,N}
    obj = primal(obj_fwd_slot)
    run_fwds_pass, _, rrule, PB = if source === nothing
        _function_wrapper_reverse_parts(R, obj, A, tangent(obj_fwd_slot, 1))
    else
        (source.fwds_wrapper, nothing, source.rrule, source.pb_type)
    end
    obj_tangent_ref = Ref(obj_fwd_slot)
    frule_type, frule_sig, frule_ret = _construct_frule_types(R, A, Val(N))
    frule = if source === nothing
        build_frule(get_interpreter(ForwardMode), Tuple{Tobj,A.parameters...}; chunk_size=N)
    else
        source.frule
    end
    run_frule = @static if VERSION ≥ v"1.12-"
        Base.Experimental.@opaque frule_sig -> frule_ret (x...) -> begin
            obj_tangent_ref[] = _callable_slot(
                _function_wrapper_object(x[1], obj_tangent_ref[]), obj_tangent_ref[]
            )
            frule(obj_tangent_ref[], Base.tail(x)...)
        end
    else
        Base.Experimental.@opaque frule_sig (x...) -> begin
            obj_tangent_ref[] = _callable_slot(
                _function_wrapper_object(x[1], obj_tangent_ref[]), obj_tangent_ref[]
            )
            frule(obj_tangent_ref[], Base.tail(x)...)
        end
    end
    return FunctionWrapperTangent{R,A,N}(
        run_fwds_pass,
        run_frule isa frule_type ? run_frule : _wrapper_closure(run_frule, frule_ret),
        obj_tangent_ref,
        frule,
        rrule,
        PB,
    )
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

# Thread the seed cache into captures so aliases elsewhere in the seed share tangents.
# Calling zero/uninit/randn_lifted here would create a fresh cache.
for internal in (:_zero_dual_internal, :_uninit_dual_internal)
    @eval function $internal(::Val{N}, p::FunctionWrapper{R,A}, d::MaybeCache) where {N,R,A}
        haskey(d, p) && return d[p]::dual_type(Val(N), typeof(p))
        obj = p.obj[]
        slot = Lifted{typeof(obj),N}(obj, $internal(Val(N), obj, d))
        t = _function_wrapper_forward_tangent(R, A, slot, Val(N))
        d[p] = t
        return t
    end
end
function _randn_dual_internal(
    ::Val{N}, rng::AbstractRNG, p::FunctionWrapper{R,A}, d::MaybeCache
) where {N,R,A}
    haskey(d, p) && return d[p]::dual_type(Val(N), typeof(p))
    obj = p.obj[]
    slot = Lifted{typeof(obj),N}(obj, _randn_dual_internal(Val(N), rng, obj, d))
    t = _function_wrapper_forward_tangent(R, A, slot, Val(N))
    d[p] = t
    return t
end

function increment_internal!!(
    c::IncCache, t::T, s::T
) where {R,A,N,T<:FunctionWrapperTangent{R,A,N}}
    N == 0 || throw(
        ArgumentError("materialise native tangents before incrementing forward storage")
    )
    (t === s || haskey(c, t)) && return t
    c[t] = true
    t.dobj_ref[] = increment_internal!!(c, t.dobj_ref[], s.dobj_ref[])
    return t
end

function set_to_zero_internal!!(
    c::SetToZeroCache, t::FunctionWrapperTangent{R,A,N}
) where {R,A,N}
    N == 0 || throw(ArgumentError("use basis_lifted!! to reseed forward storage"))
    t.dobj_ref[] = set_to_zero_internal!!(c, t.dobj_ref[])
    return t
end

function _add_to_primal_internal(
    c::MaybeCache, p::FunctionWrapper, t::FunctionWrapperTangent, unsafe::Bool
)
    key = (p, t, unsafe)
    haskey(c, key) && return c[key]::typeof(p)
    q = typeof(p)(_add_to_primal_internal(c, p.obj[], t.dobj_ref[], unsafe))
    c[key] = q
    return q
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
    key = (t, s)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    return _dot_internal(c, t.dobj_ref[], s.dobj_ref[])::Float64
end

function _scale_internal(
    c::MaybeCache, a::Float64, t::T
) where {R,A,T<:FunctionWrapperTangent{R,A,0}}
    haskey(c, t) && return c[t]::T
    ref = t.dobj_ref
    s = _function_wrapper_tangent(R, ref.obj, A, _scale_internal(c, a, ref[]), t)[1]
    return c[t] = s::T
end

function _scale_internal(::MaybeCache, ::Float64, ::FunctionWrapperTangent)
    throw(
        ArgumentError(
            "materialise a native tangent with tangent(slot, lane) before scaling"
        ),
    )
end

# Type piracy: Base copies objptr verbatim, leaving it pointing at the original callable.
# Reconstruct the wrapper so its object pointer uses the copied callable.
function Base.deepcopy_internal(x::FunctionWrapper, dict::IdDict)
    haskey(dict, x) && return dict[x]::typeof(x)
    y = typeof(x)(x.obj[])
    dict[x] = y
    dict[x.obj] = y.obj
    y.obj[] = Base.deepcopy_internal(x.obj[], dict)
    reinit_wrapper(y)
    return y
end

# Opaque closures are not deep-copied. Rebind them to the copied derivative storage.
function Base.deepcopy_internal(
    t::T, dict::IdDict
) where {R,A,N,T<:FunctionWrapperTangent{R,A,N}}
    haskey(dict, t) && return dict[t]::T
    ref = t.dobj_ref
    copied = if N == 0
        obj = Base.deepcopy_internal(ref.obj, dict)
        seed = Base.deepcopy_internal(ref[], dict)
        _function_wrapper_tangent(R, obj, A, seed, t)[1]
    else
        slot = Base.deepcopy_internal(ref[], dict)
        _function_wrapper_forward_tangent(R, A, slot, Val(N))
    end
    return dict[t] = copied::T
end

# Opaque closures and dobj_ref are not field-parallel to FunctionWrapper; test calls and
# construction rather than structural field interactions.
TestUtils.supports_field_access_interactions(::Type{<:FunctionWrapper}) = false

function TestUtils._chunked_v_invariant(
    p::FunctionWrapper, v::FunctionWrapperTangent, c::IdDict
)
    haskey(c, v) && return true
    c[v] = nothing
    return TestUtils._chunked_v_invariant(p.obj[], tangent(v.dobj_ref[]), c)
end

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
    y = FunctionWrapper{R,A}(primal(obj))
    return Lifted{typeof(y),N}(y, _function_wrapper_forward_tangent(R, A, obj, Val(N)))
end

@is_primitive MinimalCtx Tuple{<:FunctionWrapper,Vararg}
function rrule!!(f::CoDual{<:FunctionWrapper}, x::Vararg{CoDual})
    y, pb = f.dx.fwds_wrapper(primal(f), x...)
    function_wrapper_eval_pb(dy) = pb(dy)
    return y, function_wrapper_eval_pb
end

function frule!!(f::Lifted{FunctionWrapper{R,A},N}, x::Vararg{Lifted}) where {R,A,N}
    tangent(f) isa dual_type(Val(N), typeof(primal(f))) ||
        throw(ArgumentError("incompatible FunctionWrapper forward storage; use lift"))
    return tangent(f).frule_wrapper(primal(f), x...)
end

end
