const ComplexFloat{P<:IEEEFloat} = Complex{P}
const CF{P} = ComplexFloat{P}

@foldable tangent_type(::Type{CF{P}}) where {P} = Complex{tangent_type(P)}
@foldable tangent_type(::Type{NoFData}, ::Type{CF{P}}) where {P} = Complex{P}
@foldable fdata_type(::Type{T}) where {T<:ComplexFloat} = NoFData
@foldable rdata_type(::Type{T}) where {T<:ComplexFloat} = T

tangent(::NoFData, t::CF) = t

zero_tangent_internal(p::CF{P}, ::MaybeCache) where {P} = zero(p)

zero_rdata(p::CF) = zero_tangent(p)
zero_rdata_from_type(P::Type{<:CF}) = zero(P)

set_to_zero_internal!!(::SetToZeroCache, p::CF) = zero(p)

randn_tangent_internal(rng::AbstractRNG, p::CF, ::MaybeCache) = randn(rng, typeof(p))

__verify_fdata_value(::IdDict{Any,Nothing}, ::P, ::P) where {P<:CF} = nothing
_verify_rdata_value(::P, ::P) where {P<:CF} = nothing

increment!!(t::T, s::T) where {T<:CF} = t + s
increment_internal!!(::IncCache, t::T, s::T) where {T<:CF} = t + s

function increment_field!!(x::CF{P}, re_or_im::P, ::Val{FieldName}) where {P,FieldName}
    return if FieldName === :re
        complex(real(x) + re_or_im, imag(x))
    elseif FieldName === :im
        complex(real(x), re_or_im + imag(x))
    else
        throw(ArgumentError(lazy"Unkown field `$Fieldname` for type `$(CF{P})`)"))
    end
end

tangent_to_primal_internal!!(x::P, t::P, ::MaybeCache) where {P<:CF} = t
primal_to_tangent_internal!!(t::P, x::P, ::MaybeCache) where {P<:CF} = x

_add_to_primal_internal(::MaybeCache, x::T, t::T, ::Bool) where {T<:CF} = x + t

function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:CF}
    _dot(real(t), real(s)) + _dot(imag(t), imag(s))
end

_scale_internal(::MaybeCache, a::Float64, t::T) where {T<:CF} = T(a * t)

populate_address_map_internal(m::AddressMap, ::P, ::P) where {P<:CF} = m

@is_primitive MinimalCtx Tuple{typeof(lgetfield),CF{T},Val} where {T}
function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{<:CF,<:CF}, ::Dual{Val{FieldName}}
) where {FieldName}
    y = getfield(primal(x), FieldName)
    dy = getfield(tangent(x), FieldName)
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

@is_primitive MinimalCtx Tuple{typeof(_new_),<:CF{T},T,T} where {T<:IEEEFloat}
function frule!!(
    ::Dual{typeof(_new_)}, ::Dual{Type{CF{P}}}, re::Dual{P}, im::Dual{P}
) where {P<:IEEEFloat}
    x = _new_(CF{P}, primal(re), primal(im))
    dx = _new_(CF{P}, tangent(re), tangent(im))
    return Dual(x, dx)
end
function rrule!!(
    ::CoDual{typeof(_new_)}, ::CoDual{Type{CF{P}}}, re::CoDual{P}, im::CoDual{P}
) where {P<:IEEEFloat}
    _new_complex_pb(dy::CF{P}) = NoRData(), NoRData(), real(dy), imag(dy)
    return zero_fcodual(_new_(CF{P}, re.x, im.x)), _new_complex_pb
end
