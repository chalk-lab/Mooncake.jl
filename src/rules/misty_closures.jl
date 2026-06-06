"""
    MistyClosureTangent(captures_tangent::Any, dual_callable::Any)

The tangent type for `MistyClosure`. `captures_tangent` contains the tangent to the captured
variables, and `dual_callable` contains a callable object which performs forwards-mode AD.

That the field type of `captures_tangent` is `Any` is unavoidable since the `captures`
field of an `OpaqueClosure` has field type `Any`.

That the field type of `dual_callable` is `Any` is a limitation of the current
implementation. The concrete type of `dual_callable` might be one of a couple of things,
notably either `typeof(frule!!)` or `DerivedFRule`. It might be possible to figure out which
it is, and use this information to improve the type stability of this function.
"""
struct MistyClosureTangent
    captures_tangent::Any
    dual_callable::Any
end

# Degree-of-freedom count (forward gradient/Jacobian seeding) of a MistyClosure tangent: only
# the differentiable `captures_tangent` carries scalar DOFs. `dual_callable` is the compiled
# dual rule (an OpaqueClosure/MistyClosure), not a tangent — walking it generically recurses
# unboundedly into compiled IR (e.g. via the HVP `grad_f`), so it is skipped.
@inline dof(t::MistyClosureTangent, seen::IdDict{Any,Any}) = dof(
    getfield(t, :captures_tangent), seen
)

# Build a forward-mode rule for a MistyClosure using its original world age.
#
# We cannot use the current world age because the MistyClosure's IR (p.ir[]) has a
# valid_worlds range set at creation time. On Julia 1.12+, generate_dual_ir calls
# set_valid_world!(ir, interp.world), which throws if the world is outside this range.
# If methods were defined after the MistyClosure was created, the current world would
# fall outside valid_worlds and cause an error.
#
# Using the original world age is safe because lookup_ir for MistyClosure returns mc.ir[]
# directly, bypassing method table lookups. Nested non-primitive calls use LazyFRule or
# DynamicFRule, which obtain a current-world interpreter via get_interpreter() at runtime.
# We pass skip_world_age_check=true since build_frule's safety check would incorrectly
# reject our intentionally-older interpreter.
#
function _dual_mc(p::MistyClosure)
    @static if VERSION > v"1.12-"
        # Use the IR's valid_worlds.max_world instead of oc.world to avoid world age mismatch.
        # The oc.world can be slightly newer than valid_worlds.max_world if methods were
        # defined between IR generation and OpaqueClosure creation. Using max_world ensures
        # we're within the valid range while still having access to all methods the IR needs.
        mc_world = UInt(p.ir[].valid_worlds.max_world)
    else
        mc_world = UInt(p.oc.world)
    end
    interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=mc_world)
    return build_frule(interp, p; skip_world_age_check=true)
end

tangent_type(::Type{<:MistyClosure}) = MistyClosureTangent

# Forward-mode V for a MistyClosure is its `MistyClosureTangent` — NOT the
# generic structural lift of the closure's IR. In the *forward* slot the
# `captures_tangent` field holds the already-lifted forward captures slot
# (`Lifted{captures, V}`), built here at lift time rather than per `frule!!`
# call. Building once is required for forward-over-reverse: a reverse rule's
# `fwds_oc` and `pb_oc` share their captures, so they must share the forward
# tangent buffer — which they do when both are lifted within one operation that
# threads the aliasing cache `c` (keyed by the primal captures identity). The
# `dual_callable` is the forward rule built by `_dual_mc`.
@foldable @inline dual_type(::Val{N}, ::Type{<:MistyClosure}) where {N} =
    MistyClosureTangent
lift(x::MistyClosure, ẋ::MistyClosureTangent) = lift(x, ẋ, nothing)
function lift(x::MistyClosure, ẋ::MistyClosureTangent, c::Union{Nothing,IdDict})
    lifted_captures = _lift_mc_captures(x.oc.captures, ẋ.captures_tangent, c)
    return Lifted{typeof(x),1,MistyClosureTangent}(
        x, MistyClosureTangent(lifted_captures, ẋ.dual_callable)
    )
end
@inline _lift_mc_captures(captures, ct, ::Nothing) = lift(captures, ct)
@inline _lift_mc_captures(captures, ct, c::IdDict) = get!(
    () -> lift(captures, ct, c), c, captures
)

# Per-lane tangent: only `captures_tangent` (itself a `Lifted` captures slot) carries
# DOFs, so recurse into it for lane `lane` and carry `dual_callable` through unchanged.
@inline function tangent(
    x::Lifted{P,N,MistyClosureTangent}, lane::Integer
) where {P<:MistyClosure,N}
    v = x.value
    return MistyClosureTangent(tangent(v.captures_tangent, lane), v.dual_callable)
end

function zero_tangent_internal(p::MistyClosure, d::MaybeCache)
    return MistyClosureTangent(zero_tangent_internal(p.oc.captures, d), _dual_mc(p))
end

function randn_tangent_internal(rng::AbstractRNG, p::MistyClosure, d::MaybeCache)
    return MistyClosureTangent(randn_tangent_internal(rng, p.oc.captures, d), _dual_mc(p))
end

# Forward-mode cache-aware seed factories. Like Complex/Memory, a MistyClosure has
# fields but its own canonical V (`MistyClosureTangent`), not the generic structural
# lift — so the cache-aware `_*_dual_internal` must build the tangent here rather than
# recurse into the closure's internals (the `OpaqueClosure`'s `Ptr` and `captures::Any`
# fields, which have no coherent V). Mirrors the reverse factories above; the forward
# `captures_tangent` is a `Lifted` slot (as built by `lift`), cached by captures identity
# so a reverse rule's shared `fwds_oc`/`pb_oc` captures get one tangent buffer.
for internal in (:_zero_dual_internal, :_uninit_dual_internal)
    @eval function $internal(w::Val{N}, p::MistyClosure, d::MaybeCache) where {N}
        cap = p.oc.captures
        haskey(d, cap) && return MistyClosureTangent(d[cap], _dual_mc(p))
        lc = Lifted{typeof(cap),N}(cap, $internal(w, cap, d))
        d[cap] = lc
        return MistyClosureTangent(lc, _dual_mc(p))
    end
end
function _randn_dual_internal(
    w::Val{N}, rng::AbstractRNG, p::MistyClosure, d::MaybeCache
) where {N}
    cap = p.oc.captures
    haskey(d, cap) && return MistyClosureTangent(d[cap], _dual_mc(p))
    lc = Lifted{typeof(cap),N}(cap, _randn_dual_internal(w, rng, cap, d))
    d[cap] = lc
    return MistyClosureTangent(lc, _dual_mc(p))
end

function increment_internal!!(c::IncCache, t::T, s::T) where {T<:MistyClosureTangent}
    new_captures_tangent = increment_internal!!(c, t.captures_tangent, s.captures_tangent)
    return MistyClosureTangent(new_captures_tangent, t.dual_callable)
end

function set_to_zero_internal!!(c::SetToZeroCache, t::MistyClosureTangent)
    new_captures_tangent = set_to_zero_internal!!(c, t.captures_tangent)
    return MistyClosureTangent(new_captures_tangent, t.dual_callable)
end

function _add_to_primal_internal(
    c::MaybeCache, p::MistyClosure, t::MistyClosureTangent, unsafe::Bool
)
    new_captures = _add_to_primal_internal(c, p.oc.captures, t.captures_tangent, unsafe)
    return replace_captures(p, new_captures)
end

function tangent_to_primal_internal!!(
    p::MistyClosure, t::MistyClosureTangent, c::MaybeCache
)
    new_captures = tangent_to_primal_internal!!(p.oc.captures, t.captures_tangent, c)
    return replace_captures(p, new_captures)
end
function primal_to_tangent_internal!!(
    t::MistyClosureTangent, p::MistyClosure, c::MaybeCache
)
    new_captures_tangent = primal_to_tangent_internal!!(
        t.captures_tangent, p.oc.captures, c
    )
    return MistyClosureTangent(new_captures_tangent, t.dual_callable)
end

function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:MistyClosureTangent}
    return _dot_internal(c, t.captures_tangent, s.captures_tangent)
end

function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:MistyClosureTangent}
    captures_tangent = _scale_internal(c, a, t.captures_tangent)
    return T(captures_tangent, t.dual_callable)
end

import .TestUtils: populate_address_map_internal, AddressMap, has_equal_data_internal
function populate_address_map_internal(
    m::AddressMap, p::MistyClosure, t::MistyClosureTangent
)
    return populate_address_map_internal(m, p.oc.captures, t.captures_tangent)
end

function has_equal_data_internal(
    x::MistyClosureTangent,
    y::MistyClosureTangent,
    equal_undefs::Bool,
    d::Dict{Tuple{UInt,UInt},Bool},
)
    # Only compare captures_tangent. The dual_callable field is a forward-mode rule
    # built on-demand by _dual_mc, which creates a new interpreter each time. Different
    # interpreter instances produce different rule objects, even for the same MistyClosure.
    # Since dual_callable is just a computational tool (not part of the tangent's value),
    # two tangents with identical captures_tangent are mathematically equal.
    return has_equal_data_internal(x.captures_tangent, y.captures_tangent, equal_undefs, d)
end

struct MistyClosureFData
    captures_fdata::Any
    dual_callable::Any
end

struct MistyClosureRData{Tr}
    captures_rdata::Tr
end

# Deep copy the captures data for misty closures
_copy(r::MistyClosureRData) = MistyClosureRData(deepcopy(r.captures_rdata))

fdata_type(::Type{<:MistyClosureTangent}) = MistyClosureFData
function fdata(t::MistyClosureTangent)
    return MistyClosureFData(fdata(t.captures_tangent), t.dual_callable)
end

rdata_type(::Type{<:MistyClosureTangent}) = MistyClosureRData
rdata(t::MistyClosureTangent) = MistyClosureRData(rdata(t.captures_tangent))

@foldable function tangent_type(::Type{<:MistyClosureFData}, ::Type{<:MistyClosureRData})
    return MistyClosureTangent
end
function tangent(f::MistyClosureFData, r::MistyClosureRData)
    return MistyClosureTangent(tangent(f.captures_fdata, r.captures_rdata), f.dual_callable)
end

function __verify_fdata_value(::IdDict{Any,Nothing}, p::MistyClosure, t::MistyClosureFData)
    return nothing
end
_verify_rdata_value(p::MistyClosure, r::MistyClosureRData) = nothing

zero_rdata(p::MistyClosure) = MistyClosureRData(zero_rdata(p.oc.captures))

function increment!!(x::MistyClosureFData, y::MistyClosureFData)
    return MistyClosureFData(
        increment!!(x.captures_fdata, y.captures_fdata), x.dual_callable
    )
end

function increment_internal!!(c::IncCache, x::MistyClosureRData, y::MistyClosureRData)
    return MistyClosureRData(increment_internal!!(c, x.captures_rdata, y.captures_rdata))
end

function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{P,F}, ::CoDual{Val{f}}
) where {P<:MistyClosure,F<:MistyClosureFData,f}
    misty_closure_getfield_rrule_exception()
end

function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{P,F}, ::CoDual{Val{f}}, ::CoDual{Val{order}}
) where {P<:MistyClosure,F<:MistyClosureFData,f,order}
    misty_closure_getfield_rrule_exception()
end

function misty_closure_getfield_rrule_exception()
    msg =
        "rrule!! for `lgetfield` and `getfield` not implemented for " *
        "`MistyClosure`s. That is, you cannot currently query a field of a " *
        "`MistyClosure` in code which you differentiate. If this is a " *
        "problem for your use-case, please open an issue on the Mooncake.jl " *
        "repository."
    throw(UnhandledLanguageFeatureException(msg))
end

function rrule!!(::CoDual{typeof(_new_)}, p::CoDual{<:MistyClosure}, x::Vararg{CoDual})
    misty_closure_getfield_rrule_exception()
end

function misty_closure_new_rrule_exception()
    msg =
        "rrule!! for `_new_` not implemented for `MistyClosure`. That is, " *
        "you cannot currently construct a `MistyClosure` in code that you " *
        "differentiate. If this is a problem for your use-case, please open " *
        "an issue on the Mooncake.jl repository."
    throw(UnhandledLanguageFeatureException(msg))
end

@is_primitive MinimalCtx Tuple{MistyClosure,Vararg{Any,N}} where {N}
# The forward-slot `captures_tangent` already holds the lifted (and, in
# forward-over-reverse, shared) forward captures slot built at lift time, so
# forward it directly to the `_dual_mc`-built callable. Re-lifting here would
# allocate a fresh, unshared buffer and silently zero the HVP.
function frule!!(f::Lifted{<:MistyClosure,1}, x::Vararg{Lifted,M}) where {M}
    t = tangent(f)
    return t.dual_callable(t.captures_tangent, x...)
end
function rrule!!(f::CoDual{<:MistyClosure}, x::CoDual...)
    msg =
        "Attempted to compute the adjoint associated to a `MistyClosure`. " *
        "This is not currently supported. Please open an issue if you need " *
        "this functionality."
    throw(ArgumentError(msg))
end
