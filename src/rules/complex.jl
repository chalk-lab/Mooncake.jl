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

_scale_internal(::MaybeCache, a::IEEEFloat, t::T) where {T<:CF} = T(a * t)

TestUtils.populate_address_map_internal(m::TestUtils.AddressMap, ::P, ::P) where {P<:CF} = m

@is_primitive MinimalCtx Tuple{typeof(lgetfield),Complex{P},Val} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{<:CF}, ::Dual{Val{FieldName}}
) where {FieldName}
    y = getfield(primal(x), FieldName)
    dy = get_tangent_field(tangent(x), FieldName)
    return Dual(y, dy)
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
        throw(ArgumentError(lazy"lgetfield: Unknown field '$FieldName' for type Complex{$T}."))
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

@is_primitive MinimalCtx Tuple{typeof(_new_),<:Complex{P},P,P} where {P<:IEEEFloat}
@inline function _complex_width_aware_tangent(re_t, im_t, ::Val{1})
    return complex(
        Mooncake._ntangent_basis_dir(re_t, Val(1)),
        Mooncake._ntangent_basis_dir(im_t, Val(1)),
    )
end
@inline function _complex_width_aware_tangent(re_t, im_t, ::Val{N}) where {N}
    return NTangent(
        ntuple(
            i -> complex(
                Mooncake._ntangent_basis_dir(re_t, Val(i)),
                Mooncake._ntangent_basis_dir(im_t, Val(i)),
            ),
            Val(N),
        ),
    )
end

function frule!!(
    ::Dual{typeof(_new_)}, ::Dual{Type{Complex{P}}}, re::Dual{P,Tr}, im::Dual{P,Ti}
) where {P<:IEEEFloat,Tr,Ti}
    x = _new_(Complex{P}, primal(re), primal(im))
    N = max(Mooncake._dual_width(re), Mooncake._dual_width(im))
    dx = _complex_width_aware_tangent(tangent(re), tangent(im), Val(N))
    return Mooncake.dual_type(Val(N), typeof(x))(x, dx)
end
function frule!!(
    ::Dual{typeof(_new_)}, ::Dual{Type{Complex{P}}}, re::Dual, im::Dual
) where {P<:IEEEFloat}
    x = _new_(Complex{P}, P(primal(re)), P(primal(im)))
    N = max(Mooncake._dual_width(re), Mooncake._dual_width(im))
    dx = _complex_width_aware_tangent(tangent(re), tangent(im), Val(N))
    return Mooncake.dual_type(Val(N), typeof(x))(x, dx)
end
function frule!!(
    ::Dual{typeof(_new_)}, ::Dual{Type{Complex{P}}}, re::Any, im::Any
) where {P<:IEEEFloat}
    all(verify_dual_type, (re, im)) || error_if_incorrect_dual_types(re, im)
    x = _new_(Complex{P}, primal(re), primal(im))
    N = max(Mooncake._dual_width(re), Mooncake._dual_width(im))
    dx = _complex_width_aware_tangent(tangent(re), tangent(im), Val(N))
    return Mooncake.dual_type(Val(N), typeof(x))(x, dx)
end
function rrule!!(
    ::CoDual{typeof(_new_)}, ::CoDual{Type{Complex{P}}}, re::CoDual{P}, im::CoDual{P}
) where {P<:IEEEFloat}
    _new_complex_pb(dy::Complex{P}) = NoRData(), NoRData(), real(dy), imag(dy)
    return zero_fcodual(_new_(Complex{P}, re.x, im.x)), _new_complex_pb
end
