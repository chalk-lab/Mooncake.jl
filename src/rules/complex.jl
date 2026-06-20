# NOTE: Type aliasing does not reliably preserve parameter constraints.
# For example:
#   const ComplexFloat{P<:IEEEFloat} = Complex{P}
#   const CF{P} = ComplexFloat{P}
#
# Methods like
#   tangent_type(::Type{CF{P}}) where {P}
#   tangent_type(::Type{NoFData}, ::Type{CF{P}}) where {P}
#
# and even uses of `CF{P}` itself will accept `P = Int`, despite the original
# `P<:IEEEFloat` constraint on `ComplexFloat{P}`. Aliases do not enforce the
# bounds of the types they refer to.
#
# Therefore, we use forms like `Complex{<:IEEEFloat}` or a non-parametric alias
# `const CF = Complex{<:IEEEFloat}`.

const CF = Complex{<:IEEEFloat}

@foldable tangent_type(::Type{Complex{P}}) where {P<:IEEEFloat} = Complex{tangent_type(P)}
@foldable tangent_type(::Type{NoFData}, ::Type{Complex{P}}) where {P<:IEEEFloat} = Complex{
    P
}
@foldable fdata_type(::Type{T}) where {T<:CF} = NoFData
@foldable rdata_type(::Type{T}) where {T<:CF} = T

tangent(::NoFData, t::CF) = t

zero_tangent_internal(p::Complex{P}, ::MaybeCache) where {P<:IEEEFloat} = zero(p)

zero_rdata(p::CF) = zero_tangent(p)
zero_rdata_from_type(P::Type{<:CF}) = zero(P)

set_to_zero_internal!!(::SetToZeroCache, p::CF) = zero(p)

randn_tangent_internal(rng::AbstractRNG, p::CF, ::MaybeCache) = randn(rng, typeof(p))

__verify_fdata_value(::IdDict{Any,Nothing}, ::P, ::P) where {P<:CF} = nothing
_verify_rdata_value(::P, ::P) where {P<:CF} = nothing

increment_internal!!(::IncCache, t::T, s::T) where {T<:CF} = t + s

function increment_field!!(
    x::Complex{P}, re_or_im::P, ::Val{FieldName}
) where {P<:IEEEFloat,FieldName}
    return if (FieldName === :re) || (FieldName == 1)
        complex(real(x) + re_or_im, imag(x))
    elseif (FieldName === :im) || (FieldName == 2)
        complex(real(x), re_or_im + imag(x))
    else
        throw(ArgumentError(lazy"Unknown field `$FieldName` for type `$(Complex{P})`)"))
    end
end

tangent_to_primal_internal!!(x::P, t::P, ::MaybeCache) where {P<:CF} = t
primal_to_tangent_internal!!(t::P, x::P, ::MaybeCache) where {P<:CF} = x

_add_to_primal_internal(::MaybeCache, x::T, t::T, ::Bool) where {T<:CF} = x + t

function _dot_internal(::MaybeCache, t::T, s::T) where {T<:CF}
    _dot(real(t), real(s)) + _dot(imag(t), imag(s))
end

_scale_internal(::MaybeCache, a::Float64, t::T) where {T<:CF} = T(a * t)

TestUtils.populate_address_map_internal(m::TestUtils.AddressMap, ::P, ::P) where {P<:CF} = m

# `lgetfield(::Complex, ::Val)` is already a primitive via the generic
# `Tuple{typeof(lgetfield),Any,Val}` declaration in `misc.jl`; this Complex-specific frule only
# refines the forward V, so it needs no `@is_primitive` of its own.
function frule!!(
    ::Lifted{typeof(lgetfield),N},
    x::Lifted{Complex{P},N,Complex{NDual{P,N}}},
    ::Lifted{Val{FieldName},N},
) where {N,P<:IEEEFloat,FieldName}
    y = getfield(primal(x), FieldName)
    dy = getfield(tangent(x), FieldName)
    return Lifted{P,N}(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    obj_cd::CoDual{<:CF,<:CF},
    field_name_cd::CoDual{Val{FieldName}},
) where {FieldName}
    a = primal(obj_cd)
    a_tangent = tangent(obj_cd)

    value_primal = getfield(a, FieldName)
    actual_field_tangent_value = if FieldName === :re
        a_tangent.re
    elseif FieldName === :im
        a_tangent.im
    else
        throw(ArgumentError(lazy"lgetfield: Unknown field '$FieldName' for type $(typeof(a))."))
    end

    value_output_fdata = fdata(actual_field_tangent_value)
    y_cd = CoDual(value_primal, value_output_fdata)

    function lgetfield_Complex_pullback(Δy_rdata)
        if FieldName === :re
            Δx = complex(Δy_rdata, zero(Δy_rdata))
        elseif FieldName === :im
            Δx = complex(zero(Δy_rdata), Δy_rdata)
        end
        return NoRData(), Δx, NoRData()
    end
    return y_cd, lgetfield_Complex_pullback
end

# `_new_(Type{Complex{P}}, re, im)` is already a primitive via the generic `Tuple{typeof(_new_),Vararg}`
# declaration in `new.jl`; this Complex-specific frule only refines the forward construction, so it
# needs no `@is_primitive` of its own. (The previous `Tuple{typeof(_new_),<:Complex{P},P,P}` declaration
# matched nothing — `_new_`'s second argument is the *type* `Type{Complex{P}}`, not a `Complex` value.)
function frule!!(
    ::Lifted{typeof(_new_),N},
    ::Lifted{Type{Complex{P}},N},
    re::Lifted{P,N,NDual{P,N}},
    im::Lifted{P,N,NDual{P,N}},
) where {N,P<:IEEEFloat}
    x = _new_(Complex{P}, primal(re), primal(im))
    dx = _new_(Complex{NDual{P,N}}, tangent(re), tangent(im))
    return Lifted{Complex{P},N}(x, dx)
end
function rrule!!(
    ::CoDual{typeof(_new_)}, ::CoDual{Type{Complex{P}}}, re::CoDual{P}, im::CoDual{P}
) where {P<:IEEEFloat}
    _new_complex_pb(dy::Complex{P}) = NoRData(), NoRData(), real(dy), imag(dy)
    return zero_fcodual(_new_(Complex{P}, re.x, im.x)), _new_complex_pb
end

# Complex-scalar `lgetfield` (field read) and `_new_` (construction) are primitives (via the generic
# lgetfield/_new_ declarations), so register them as hand-written cases to get the chunked widths the
# derived cases below skip (is_primitive=false).
function hand_written_rule_test_cases(rng_ctor, ::Val{:complex})
    (
        Any[
            (false, :stability_and_allocs, nothing, lgetfield, 1.5 - 0.5im, Val(:re)),
            (false, :stability_and_allocs, nothing, lgetfield, 1.5 - 0.5im, Val(:im)),
            (false, :stability_and_allocs, nothing, _new_, ComplexF64, 1.5, -0.5),
        ],
        Any[],
    )
end

function derived_rule_test_cases(rng_ctor, ::Val{:complex})
    test_cases = Any[
        (false, :none, nothing, real, 1.0 + 2.0im),
        (false, :none, nothing, imag, 1.0 + 2.0im),
        (false, :none, nothing, z -> z.re * z.im, 1.5 - 0.5im),
        (false, :none, nothing, (a, b) -> Complex(a, b), 1.0, 2.0),
        (false, :none, nothing, (a, b) -> abs2(Complex(a, b)), 1.0, 2.0),
    ]
    return test_cases, Any[]
end
