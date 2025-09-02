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

_dual_mc(p::MistyClosure) = build_frule(get_interpreter(ForwardMode), p)

tangent_type(::Type{<:MistyClosure}) = MistyClosureTangent

function zero_tangent_internal(p::MistyClosure, d::MaybeCache)
    return MistyClosureTangent(zero_tangent_internal(p.oc.captures, d), _dual_mc(p))
end

function randn_tangent_internal(rng::AbstractRNG, p::MistyClosure, d::MaybeCache)
    return MistyClosureTangent(randn_tangent_internal(rng, p.oc.captures, d), _dual_mc(p))
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

function _diff_internal(c::MaybeCache, p::P, q::P) where {P<:MistyClosure}
    # Just assumes that the code associated to `p` is the same as that of `q`.
    captures_tangent = _diff_internal(c, p.oc.captures, q.oc.captures)
    return MistyClosureTangent(captures_tangent, _dual_mc(p))
end

function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:MistyClosureTangent}
    return _dot_internal(c, t.captures_tangent, s.captures_tangent)
end

function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:MistyClosureTangent}
    captures_tangent = _scale_internal(c, a, t.captures_tangent)
    return T(captures_tangent, t.dual_callable)
end

import .TestUtils: populate_address_map_internal, AddressMap
function populate_address_map_internal(
    m::AddressMap, p::MistyClosure, t::MistyClosureTangent
)
    return populate_address_map_internal(m, p.oc.captures, t.captures_tangent)
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
function frule!!(f::Dual{<:MistyClosure}, x::Dual...)
    dual_captures = Dual(primal(f).oc.captures, tangent(f).captures_tangent)
    return tangent(f).dual_callable(dual_captures, x...)
end
function rrule!!(f::CoDual{<:MistyClosure}, x::CoDual...)
    msg =
        "Attempted to compute the adjoint associated to a `MistyClosure`. " *
        "This is not currently supported. Please open an issue if you need " *
        "this functionality."
    throw(ArgumentError(msg))
end
