module MooncakeSpecialFunctionsExt

using SpecialFunctions, Mooncake
using Base: IEEEFloat
import ChainRulesCore as CRC
import Mooncake:
    DefaultCtx,
    @from_chainrules,
    @from_rrule,
    @zero_derivative,
    @is_primitive,
    Dual,
    Lifted,
    NDual,
    frule!!,
    Tangent,
    primal,
    tangent,
    notimplemented_tangent_guard,
    ForwardMode,
    extract

@from_chainrules DefaultCtx Tuple{typeof(airyai),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airyaix),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airyaiprime),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airyaiprimex),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airybi),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airybiprime),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(besselj0),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(besselj1),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(bessely0),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(bessely1),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(dawson),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(digamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erf),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erf),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfc),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logerfc),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfcinv),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfcx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logerfcx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfi),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfinv),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(gamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(invdigamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(trigamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(polygamma),Integer,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(beta),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logbeta),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logabsgamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(loggamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(expint),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(expintx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(expinti),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(sinint),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(cosint),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(ellipk),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(ellipe),IEEEFloat}

@zero_derivative DefaultCtx Tuple{typeof(logfactorial),Integer}

"""
## Handling `ChainRulesCore.NotImplemented` Tangents in Imported Rules

Mooncake uses a *masking* trick to handle
`ChainRulesCore.NotImplemented` partial derivatives.

**NOTE:**
A missing partial derivative is irrelevant if it is multiplied by the zero
element of the corresponding tangent or cotangent space.

The result is therefore either correct or explicitly marked as unknown
(`NaN`).

### Forward mode (pushforward)

For `f(x, y)` with unimplemented `∂f/∂x`:

    ḟ = (∂f/∂x)·ẋ + (∂f/∂y)·ẏ

- If `ẋ == 0`, the contribution from `x` vanishes:
  
      ḟ = (∂f/∂y)·ẏ

- If `ẋ != 0`, the missing derivative is required ⇒ `ḟ = NaN`.

### Reverse mode (pullback)

Given upstream cotangent `f̄`:

    x̄ = (∂f/∂x)'·f̄
    ȳ = (∂f/∂y)'·f̄

- If `f̄ == 0`, then `x̄ = 0` even if `∂f/∂x` is not implemented.
- If `f̄ != 0`, then `x̄ = NaN`, while `ȳ` is computed normally.

### Notes

- “Zero” refers to the additive identity of the tangent/cotangent space.
- This trick relies on `NaN` and therefore applies only to
  floating-point tangent spaces and their compositions
  (e.g. arrays of floats).
  This restriction exists because `NaN`, `NaN32`, and `NaN16`
  live exclusively in `AbstractFloat` spaces.
- `NotImplemented` indicates a missing AD rule, not a non-differentiable
  function. Any resulting `NaN`s are handled using the same masking
  principle: they affect the result only when multiplied by a nonzero
  tangent or cotangent, and are otherwise safely ignored.

### Outcome

Correct derivatives when possible, and explicit `NaN` only when
an unimplemented partial is mathematically required.
"""
#
# Standard Bessel & Hankel functions
#
@from_rrule DefaultCtx Tuple{typeof(besseli),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselj),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselk),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(bessely),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh1),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh2),IEEEFloat,Union{IEEEFloat,<:Complex}}

# Scaled bessel-i,j,k,y & hankelh1, hankelh2 rrules
@from_rrule DefaultCtx Tuple{typeof(besselix),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besseljx),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselkx),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselyx),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh1x),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh2x),IEEEFloat,Union{IEEEFloat,<:Complex}}

# Gamma and exponential integral rrules
@from_rrule DefaultCtx Tuple{
    typeof(gamma),Union{IEEEFloat,<:Complex},Union{IEEEFloat,<:Complex}
}
@from_rrule DefaultCtx Tuple{
    typeof(loggamma),Union{IEEEFloat,<:Complex},Union{IEEEFloat,<:Complex}
}
@from_rrule DefaultCtx Tuple{
    typeof(expint),Union{IEEEFloat,<:Complex},Union{IEEEFloat,<:Complex}
}
@from_rrule DefaultCtx Tuple{
    typeof(expintx),Union{IEEEFloat,<:Complex},Union{IEEEFloat,<:Complex}
}

@from_rrule DefaultCtx Tuple{typeof(gamma_inc),IEEEFloat,IEEEFloat,Integer}

# Ensure the frule return type matches the primal type.
function real_or_complex_valued(y::L, primal_eltype, dy_val) where {L<:IEEEFloat}
    return Dual(y, primal_eltype(dy_val))
end
function real_or_complex_valued(y::Complex{L}, primal_eltype, dy_val) where {L<:IEEEFloat}
    return Dual(y, Complex(primal_eltype(real(dy_val)), primal_eltype(imag(dy_val))))
end

function real_or_complex_valued(y::L, primal_eltype, dy_val) where {L<:Complex}
    return Dual(
        y,
        Mooncake.Tangent((re=primal_eltype(real(dy_val)), im=primal_eltype(imag(dy_val)))),
    )
end

# Lifted analogues — per-lane partial extraction and result packing.
@inline _sf_lane(v::NDual, lane::Integer) = v.partials[lane]
@inline function _sf_lane(v::Complex{<:NDual}, lane::Integer)
    return Complex(real(v).partials[lane], imag(v).partials[lane])
end

# Wrap (y, per-lane dy values) into the canonical Lifted slot for a
# scalar (real or complex) result.
@inline function _lifted_scalar_result(
    y::L, primal_eltype, dy_lanes::NTuple{Nw}, ::Val{Nw}
) where {L<:IEEEFloat,Nw}
    parts = ntuple(k -> primal_eltype(dy_lanes[k]), Val(Nw))
    return Lifted{L,Nw}(y, NDual{L,Nw}(y, parts))
end
@inline function _lifted_scalar_result(
    y::Complex{L}, primal_eltype, dy_lanes::NTuple{Nw}, ::Val{Nw}
) where {L<:IEEEFloat,Nw}
    re_parts = ntuple(k -> primal_eltype(real(dy_lanes[k])), Val(Nw))
    im_parts = ntuple(k -> primal_eltype(imag(dy_lanes[k])), Val(Nw))
    re_nd = NDual{L,Nw}(real(y), re_parts)
    im_nd = NDual{L,Nw}(imag(y), im_parts)
    return Lifted{Complex{L},Nw}(y, Complex{NDual{L,Nw}}(re_nd, im_nd))
end

# 3-arg `gamma_inc` (first-argument gradient is `NotImplemented`)
@is_primitive DefaultCtx ForwardMode Tuple{typeof(gamma_inc),IEEEFloat,IEEEFloat,Integer}

function frule!!(
    ::Lifted{typeof(gamma_inc),Nw},
    _a::Lifted{T,Nw,NDual{T,Nw}},
    _x::Lifted{P,Nw,NDual{P,Nw}},
    _IND::Lifted{I},
) where {T<:IEEEFloat,P<:IEEEFloat,I<:Integer,Nw}
    a = primal(_a)
    x = primal(_x)
    IND = primal(_IND)
    y = gamma_inc(a, x, IND)
    primal_eltype = eltype(y)
    z = exp((a - 1) * log(x) - x - loggamma(a))
    a_parts = tangent(_a).partials
    x_parts = tangent(_x).partials
    p_lanes = ntuple(Val(Nw)) do k
        ∂a = notimplemented_tangent_guard(a_parts[k])
        primal_eltype(∂a + x_parts[k] * z)
    end
    q_lanes = ntuple(Val(Nw)) do k
        ∂a = notimplemented_tangent_guard(a_parts[k])
        primal_eltype(∂a + x_parts[k] * -z)
    end
    p_nd = NDual{primal_eltype,Nw}(y[1], p_lanes)
    q_nd = NDual{primal_eltype,Nw}(y[2], q_lanes)
    return Lifted{typeof(y),Nw}(y, (p_nd, q_nd))
end

# 2-arg Gamma and exponential integrals (first-argument gradient is `NotImplemented`)
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(gamma),
    Union{IEEEFloat,Complex{<:IEEEFloat}},
    Union{IEEEFloat,Complex{<:IEEEFloat}},
}
function frule!!(
    ::Lifted{typeof(gamma),Nw}, _a::Lifted{T,Nw}, _x::Lifted{P,Nw}
) where {Nw,L<:IEEEFloat,T<:Union{L,Complex{L}},P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    a = primal(_a)
    x = primal(_x)
    y = gamma(a, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = -exp((a - 1) * log(x) - x)
    a_v = tangent(_a)
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(_sf_lane(a_v, k)) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(loggamma),
    Union{IEEEFloat,Complex{<:IEEEFloat}},
    Union{IEEEFloat,Complex{<:IEEEFloat}},
}
function frule!!(
    ::Lifted{typeof(loggamma),Nw}, _a::Lifted{T,Nw}, _x::Lifted{P,Nw}
) where {Nw,L<:IEEEFloat,T<:Union{L,Complex{L}},P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    a = primal(_a)
    x = primal(_x)
    y = loggamma(a, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = -exp((a - 1) * log(x) - x - loggamma(a, x))
    a_v = tangent(_a)
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(_sf_lane(a_v, k)) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(expint),
    Union{IEEEFloat,Complex{<:IEEEFloat}},
    Union{IEEEFloat,Complex{<:IEEEFloat}},
}
function frule!!(
    ::Lifted{typeof(expint),Nw}, _a::Lifted{T,Nw}, _x::Lifted{P,Nw}
) where {Nw,L<:IEEEFloat,T<:Union{L,Complex{L}},P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    a = primal(_a)
    x = primal(_x)
    y = expint(a, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = -expint(a - 1, x)
    a_v = tangent(_a)
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(_sf_lane(a_v, k)) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(expintx),
    Union{IEEEFloat,Complex{<:IEEEFloat}},
    Union{IEEEFloat,Complex{<:IEEEFloat}},
}
function frule!!(
    ::Lifted{typeof(expintx),Nw}, _a::Lifted{T,Nw}, _x::Lifted{P,Nw}
) where {Nw,L<:IEEEFloat,T<:Union{L,Complex{L}},P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    a = primal(_a)
    x = primal(_x)
    y = expintx(a, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = y - expintx(a - 1, x)
    a_v = tangent(_a)
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(_sf_lane(a_v, k)) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

# 2-arg standard Bessel and Hankel functions
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselj),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(besselj),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = besselj(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = (besselj(v - 1, x) - besselj(v + 1, x)) / 2
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(v_parts[k]) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(bessely),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(bessely),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = bessely(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = (bessely(v - 1, x) - bessely(v + 1, x)) / 2
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(v_parts[k]) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besseli),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(besseli),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = besseli(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = (besseli(v - 1, x) + besseli(v + 1, x)) / 2
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(v_parts[k]) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselk),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(besselk),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = besselk(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = -(besselk(v - 1, x) + besselk(v + 1, x)) / 2
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(v_parts[k]) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(hankelh1),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(hankelh1),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = hankelh1(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = (hankelh1(v - 1, x) - hankelh1(v + 1, x)) / 2
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(v_parts[k]) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(hankelh2),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(hankelh2),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = hankelh2(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = (hankelh2(v - 1, x) - hankelh2(v + 1, x)) / 2
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(v_parts[k]) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

#
# Non Holomorphic functions
#

# 2-arg scaled Bessel functions
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselix),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(besselix),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = besselix(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x_1 = (besselix(v - 1, x) + besselix(v + 1, x)) / 2
    ∂x_2 = -sign(real(x)) * y
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        dx_k = _sf_lane(x_v, k)
        notimplemented_tangent_guard(v_parts[k]) + ∂x_1 * dx_k + ∂x_2 * real(dx_k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselkx),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(besselkx),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = besselkx(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = -(besselkx(v - 1, x) + besselkx(v + 1, x)) / 2 + y
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(v_parts[k]) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besseljx),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(besseljx),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = besseljx(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x_1 = (besseljx(v - 1, x) - besseljx(v + 1, x)) / 2
    ∂x_2 = -sign(imag(x)) * y
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        dx_k = _sf_lane(x_v, k)
        notimplemented_tangent_guard(v_parts[k]) + ∂x_1 * dx_k + ∂x_2 * imag(dx_k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselyx),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(besselyx),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = besselyx(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x_1 = (besselyx(v - 1, x) - besselyx(v + 1, x)) / 2
    ∂x_2 = -sign(imag(x)) * y
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        dx_k = _sf_lane(x_v, k)
        notimplemented_tangent_guard(v_parts[k]) + ∂x_1 * dx_k + ∂x_2 * imag(dx_k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

# Scaled Hankel functions
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(hankelh1x),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(hankelh1x),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = hankelh1x(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = (hankelh1x(v - 1, x) - hankelh1x(v + 1, x)) / 2 - im * y
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(v_parts[k]) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(hankelh2x),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Lifted{typeof(hankelh2x),Nw}, _v::Lifted{T,Nw,NDual{T,Nw}}, _x::Lifted{P,Nw}
) where {Nw,T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v = primal(_v)
    x = primal(_x)
    y = hankelh2x(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)
    ∂x = (hankelh2x(v - 1, x) - hankelh2x(v + 1, x)) / 2 + im * y
    v_parts = tangent(_v).partials
    x_v = tangent(_x)
    dy_lanes = ntuple(Val(Nw)) do k
        notimplemented_tangent_guard(v_parts[k]) + ∂x * _sf_lane(x_v, k)
    end
    return _lifted_scalar_result(y, primal_eltype, dy_lanes, Val(Nw))
end

# ── NDual overloads for SpecialFunctions ──────────────────────────────────────
#
# These let NDual-typed inputs (used by nfwd / Hessian) propagate through
# the special-function calls that appear inside distribution logpdfs (Beta,
# Gamma, Chi, Dirichlet, …).
#
# Each method evaluates the primal at the Float64 value, then propagates the
# N partial slots using the known scalar derivative:
#
#   d/dx loggamma(x)     = digamma(x)
#   d/dx digamma(x)      = trigamma(x)
#   d/dx trigamma(x)     = polygamma(2, x)
#   d/dx polygamma(n, x) = polygamma(n+1, x)
#   d/dx logbeta(x, y)   = digamma(x) - digamma(x+y)
#   d/dy logbeta(x, y)   = digamma(y) - digamma(x+y)
#   d/dx beta(x, y)      = beta(x, y) * (digamma(x) - digamma(x+y))
#   d/dx gamma(x)        = gamma(x) * digamma(x)
#   d/dx erf(x)          = 2/√π · exp(-x²)
#   d/dx erfc(x)         = -2/√π · exp(-x²)
#   d/dx erfinv(x)       = √π/2 · exp(erfinv(x)²)
#   d/dx besselk(ν, x)   = -(besselk(ν-1,x) + besselk(ν+1,x)) / 2

using Mooncake.Nfwd: NDual

@inline function SpecialFunctions.loggamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = SpecialFunctions.digamma(v)
    return NDual{T,N}(SpecialFunctions.loggamma(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.logabsgamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    lv, sv = SpecialFunctions.logabsgamma(v)
    dv = SpecialFunctions.digamma(v)
    return (NDual{T,N}(lv, ntuple(k -> dv * x.partials[k], Val(N))), sv)
end

@inline function SpecialFunctions.digamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = SpecialFunctions.trigamma(v)
    return NDual{T,N}(SpecialFunctions.digamma(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.trigamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = SpecialFunctions.polygamma(2, v)
    return NDual{T,N}(SpecialFunctions.trigamma(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.polygamma(
    n::Integer, x::NDual{T,N}
) where {T<:IEEEFloat,N}
    v = x.value
    dv = SpecialFunctions.polygamma(n + 1, v)
    return NDual{T,N}(
        SpecialFunctions.polygamma(n, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

@inline function SpecialFunctions.gamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    gv = SpecialFunctions.gamma(v)
    dv = gv * SpecialFunctions.digamma(v)
    return NDual{T,N}(gv, ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.logbeta(
    x::NDual{T,N}, y::NDual{T,N}
) where {T<:IEEEFloat,N}
    xv, yv = x.value, y.value
    ψx = SpecialFunctions.digamma(xv)
    ψy = SpecialFunctions.digamma(yv)
    ψxy = SpecialFunctions.digamma(xv + yv)
    return NDual{T,N}(
        SpecialFunctions.logbeta(xv, yv),
        ntuple(k -> (ψx - ψxy) * x.partials[k] + (ψy - ψxy) * y.partials[k], Val(N)),
    )
end

@inline function SpecialFunctions.logbeta(x::NDual{T,N}, y::Real) where {T<:IEEEFloat,N}
    xv, yv = x.value, T(y)
    ψx = SpecialFunctions.digamma(xv)
    ψxy = SpecialFunctions.digamma(xv + yv)
    return NDual{T,N}(
        SpecialFunctions.logbeta(xv, yv), ntuple(k -> (ψx - ψxy) * x.partials[k], Val(N))
    )
end

@inline function SpecialFunctions.logbeta(x::Real, y::NDual{T,N}) where {T<:IEEEFloat,N}
    xv, yv = T(x), y.value
    ψy = SpecialFunctions.digamma(yv)
    ψxy = SpecialFunctions.digamma(xv + yv)
    return NDual{T,N}(
        SpecialFunctions.logbeta(xv, yv), ntuple(k -> (ψy - ψxy) * y.partials[k], Val(N))
    )
end

@inline function SpecialFunctions.beta(x::NDual{T,N}, y::NDual{T,N}) where {T<:IEEEFloat,N}
    xv, yv = x.value, y.value
    bv = SpecialFunctions.beta(xv, yv)
    ψx = SpecialFunctions.digamma(xv)
    ψy = SpecialFunctions.digamma(yv)
    ψxy = SpecialFunctions.digamma(xv + yv)
    return NDual{T,N}(
        bv,
        ntuple(k -> bv * ((ψx - ψxy) * x.partials[k] + (ψy - ψxy) * y.partials[k]), Val(N)),
    )
end

@inline function SpecialFunctions.erf(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = T(2 / sqrt(π)) * exp(-v^2)
    return NDual{T,N}(SpecialFunctions.erf(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.erfc(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = -T(2 / sqrt(π)) * exp(-v^2)
    return NDual{T,N}(SpecialFunctions.erfc(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.erfinv(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    inv_v = SpecialFunctions.erfinv(v)
    dv = T(sqrt(π) / 2) * exp(inv_v^2)
    return NDual{T,N}(inv_v, ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline _ndual_partials_are_zero(partials::NTuple) = all(iszero, partials)

@noinline function _throw_ndual_notimplemented(name::Symbol, argname::Symbol)
    throw(
        ArgumentError(
            "SpecialFunctions.$name does not support nfwd differentiation with respect to " *
            "`$argname`; pass a constant parameter or a promoted NDual with zero partials.",
        ),
    )
end

# Helper: extract the scalar (Float64/Float32) value from ν, which may be an NDual when the
# Julia promote machinery wraps an order into the same type as x. Active order tangents are
# not supported here and must fail loudly rather than being silently dropped.
function _bessel_nu(ν::NDual)
    _ndual_partials_are_zero(ν.partials) || _throw_ndual_notimplemented(:bessel, :ν)
    return ν.value
end
_bessel_nu(ν::Real) = ν

# d/dx besselk(ν, x) = -(besselk(ν-1, x) + besselk(ν+1, x)) / 2
@inline function SpecialFunctions.besselk(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    dv = -(SpecialFunctions.besselk(νv - 1, v) + SpecialFunctions.besselk(νv + 1, v)) / 2
    return NDual{T,N}(
        SpecialFunctions.besselk(νv, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

# d/dx besseli(ν, x) = (besseli(ν-1, x) + besseli(ν+1, x)) / 2
# Without NDual overloads the generic `bessel*(nu::Real, x::Real) = bessel*(nu, float(x))`
# path recurses infinitely because `float(x::NDual) = x`.
@inline function SpecialFunctions.besseli(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    dv = (SpecialFunctions.besseli(νv - 1, v) + SpecialFunctions.besseli(νv + 1, v)) / 2
    return NDual{T,N}(
        SpecialFunctions.besseli(νv, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

# besselix(ν, x) = besseli(ν, x) * exp(-|x|)  (exponentially scaled).
# VonMises stores besselix(0, κ) in the constructor, making this the hot path.
# d/dx besselix(ν,x) = (besselix(ν-1,x)+besselix(ν+1,x))/2 - sign(x) * besselix(ν,x)
@inline function SpecialFunctions.besselix(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    yv = SpecialFunctions.besselix(νv, v)
    dv =
        (SpecialFunctions.besselix(νv - 1, v) + SpecialFunctions.besselix(νv + 1, v)) / 2 -
        sign(v) * yv
    return NDual{T,N}(yv, ntuple(k -> dv * x.partials[k], Val(N)))
end

# besselkx(ν, x) = besselk(ν, x) * exp(x)  (exponentially scaled).
# d/dx besselkx(ν,x) = besselkx(ν,x) - (besselkx(ν-1,x)+besselkx(ν+1,x))/2  (x>0)
@inline function SpecialFunctions.besselkx(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    yv = SpecialFunctions.besselkx(νv, v)
    dv =
        yv -
        (SpecialFunctions.besselkx(νv - 1, v) + SpecialFunctions.besselkx(νv + 1, v)) / 2
    return NDual{T,N}(yv, ntuple(k -> dv * x.partials[k], Val(N)))
end

# d/dx bessely(ν, x) = (bessely(ν-1, x) - bessely(ν+1, x)) / 2
@inline function SpecialFunctions.bessely(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    dv = (SpecialFunctions.bessely(νv - 1, v) - SpecialFunctions.bessely(νv + 1, v)) / 2
    return NDual{T,N}(
        SpecialFunctions.bessely(νv, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

# d/dx besselj(ν, x) = (besselj(ν-1, x) - besselj(ν+1, x)) / 2
@inline function SpecialFunctions.besselj(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    dv = (SpecialFunctions.besselj(νv - 1, v) - SpecialFunctions.besselj(νv + 1, v)) / 2
    return NDual{T,N}(
        SpecialFunctions.besselj(νv, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

# beta_inc(a, b, x) = (I_x(a,b), 1 - I_x(a,b)) — regularized incomplete beta function.
# Implements the x-partial: ∂I_x(a,b)/∂x = x^(a-1)·(1-x)^(b-1) / B(a,b)  (the Beta PDF).
# Derivatives w.r.t. the shape parameters a, b are not implemented; callers that promote
# Float64 shape params to NDual (zero partials) get the correct result, but differentiating
# w.r.t. a or b directly will give wrong answers.

# Case: a and b are plain Reals (not NDual), only x varies.
function SpecialFunctions.beta_inc(a::Real, b::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    av, bv, xv = T(a), T(b), x.value
    Iv, Qv = SpecialFunctions.beta_inc(av, bv, xv)
    d_dx = exp(
        (av - 1) * log(xv) + (bv - 1) * log(1 - xv) - SpecialFunctions.logbeta(av, bv)
    )
    return (
        NDual{T,N}(Iv, ntuple(k -> d_dx * x.partials[k], Val(N))),
        NDual{T,N}(Qv, ntuple(k -> -d_dx * x.partials[k], Val(N))),
    )
end

# Case: all three args are NDual (e.g. when StatsFuns promotes Float64 shape params).
# Only the x-partial is computed; zero-partial promoted shape parameters are supported,
# but active shape-parameter tangents must fail loudly rather than being ignored.
function SpecialFunctions.beta_inc(
    a::NDual{T,N}, b::NDual{T,N}, x::NDual{T,N}
) where {T<:IEEEFloat,N}
    _ndual_partials_are_zero(a.partials) || _throw_ndual_notimplemented(:beta_inc, :a)
    _ndual_partials_are_zero(b.partials) || _throw_ndual_notimplemented(:beta_inc, :b)
    av, bv, xv = a.value, b.value, x.value
    Iv, Qv = SpecialFunctions.beta_inc(av, bv, xv)
    d_dx = exp(
        (av - 1) * log(xv) + (bv - 1) * log(1 - xv) - SpecialFunctions.logbeta(av, bv)
    )
    return (
        NDual{T,N}(Iv, ntuple(k -> d_dx * x.partials[k], Val(N))),
        NDual{T,N}(Qv, ntuple(k -> -d_dx * x.partials[k], Val(N))),
    )
end

end
