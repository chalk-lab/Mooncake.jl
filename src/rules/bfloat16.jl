# Tangent infrastructure and rules for Core.BFloat16 (available in Julia >= 1.11).
# Full arithmetic requires BFloat16s.jl at runtime. The tangent space is BFloat16 itself,
# mirroring the IEEEFloat treatment in tangents.jl and fwds_rvs_data.jl.

@static if isdefined(Core, :BFloat16)

@foldable tangent_type(::Type{Core.BFloat16}) = Core.BFloat16

zero_tangent_internal(::Core.BFloat16, ::MaybeCache) = zero(Core.BFloat16)

randn_tangent_internal(rng::AbstractRNG, ::Core.BFloat16, ::MaybeCache) =
    Core.BFloat16(randn(rng, Float32))

increment_internal!!(::IncCache, x::Core.BFloat16, y::Core.BFloat16) = x + y

set_to_zero_internal!!(::SetToZeroCache, ::Core.BFloat16) = zero(Core.BFloat16)

_scale_internal(::MaybeCache, a::Float64, t::Core.BFloat16) =
    Core.BFloat16(Float32(a) * Float32(t))

_dot_internal(::MaybeCache, t::Core.BFloat16, s::Core.BFloat16) =
    Float64(Float32(t) * Float32(s))

_add_to_primal_internal(::MaybeCache, x::Core.BFloat16, t::Core.BFloat16, ::Bool) = x + t

tangent_to_primal_internal!!(::Core.BFloat16, tx, ::MaybeCache) = tx

primal_to_tangent_internal!!(tx, x::Core.BFloat16, ::MaybeCache) = x

fdata_type(::Type{Core.BFloat16}) = NoFData

rdata_type(::Type{Core.BFloat16}) = Core.BFloat16

@foldable tangent_type(::Type{NoFData}, ::Type{Core.BFloat16}) = Core.BFloat16

tangent(::NoFData, r::Core.BFloat16) = r

__verify_fdata_value(::IdDict{Any,Nothing}, ::Core.BFloat16, ::NoFData) = nothing

_verify_rdata_value(::Core.BFloat16, ::Core.BFloat16) = nothing

zero_rdata(::Core.BFloat16) = zero(Core.BFloat16)

@foldable can_produce_zero_rdata_from_type(::Type{Core.BFloat16}) = true

zero_rdata_from_type(::Type{Core.BFloat16}) = zero(Core.BFloat16)

@inline nan_tangent_guard(dy::Core.BFloat16, t::Core.BFloat16) =
    iszero(dy) ? zero(Core.BFloat16) : t

# Conversions

@is_primitive MinimalCtx Tuple{Type{Float32},Core.BFloat16}
function frule!!(::Dual{Type{Float32}}, x::Dual{Core.BFloat16})
    return Dual(Float32(primal(x)), Float32(tangent(x)))
end
function rrule!!(::CoDual{Type{Float32}}, x::CoDual{Core.BFloat16})
    pb(dy::Float32) = NoRData(), Core.BFloat16(dy)
    return zero_fcodual(Float32(primal(x))), pb
end

@is_primitive MinimalCtx Tuple{Type{Float64},Core.BFloat16}
function frule!!(::Dual{Type{Float64}}, x::Dual{Core.BFloat16})
    return Dual(Float64(primal(x)), Float64(tangent(x)))
end
function rrule!!(::CoDual{Type{Float64}}, x::CoDual{Core.BFloat16})
    pb(dy::Float64) = NoRData(), Core.BFloat16(Float32(dy))
    return zero_fcodual(Float64(primal(x))), pb
end

@is_primitive MinimalCtx Tuple{Type{Core.BFloat16},Float32}
function frule!!(::Dual{Type{Core.BFloat16}}, x::Dual{Float32})
    return Dual(Core.BFloat16(primal(x)), Core.BFloat16(tangent(x)))
end
function rrule!!(::CoDual{Type{Core.BFloat16}}, x::CoDual{Float32})
    pb(dy::Core.BFloat16) = NoRData(), Float32(dy)
    return zero_fcodual(Core.BFloat16(primal(x))), pb
end

@is_primitive MinimalCtx Tuple{Type{Core.BFloat16},Float64}
function frule!!(::Dual{Type{Core.BFloat16}}, x::Dual{Float64})
    return Dual(Core.BFloat16(Float32(primal(x))), Core.BFloat16(Float32(tangent(x))))
end
function rrule!!(::CoDual{Type{Core.BFloat16}}, x::CoDual{Float64})
    pb(dy::Core.BFloat16) = NoRData(), Float64(Float32(dy))
    return zero_fcodual(Core.BFloat16(Float32(primal(x)))), pb
end

# Math rules — mirror low_level_maths.jl for IEEEFloat.

@is_primitive MinimalCtx Tuple{typeof(sqrt),Core.BFloat16}
function frule!!(::Dual{typeof(sqrt)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    y = sqrt(_x)
    return Dual(y, nan_tangent_guard(dx, dx / (2 * y)))
end
function rrule!!(::CoDual{typeof(sqrt)}, x::CoDual{Core.BFloat16})
    y = sqrt(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), nan_tangent_guard(dy, dy / (2 * y))
    return zero_fcodual(y), pb
end

@is_primitive MinimalCtx Tuple{typeof(cbrt),Core.BFloat16}
function frule!!(::Dual{typeof(cbrt)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    y = cbrt(_x)
    return Dual(y, nan_tangent_guard(dx, dx / (3 * y^2)))
end
function rrule!!(::CoDual{typeof(cbrt)}, x::CoDual{Core.BFloat16})
    y = cbrt(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), nan_tangent_guard(dy, dy / (3 * y^2))
    return zero_fcodual(y), pb
end

@is_primitive MinimalCtx Tuple{typeof(exp),Core.BFloat16}
function frule!!(::Dual{typeof(exp)}, x::Dual{Core.BFloat16})
    y = exp(primal(x))
    return Dual(y, tangent(x) * y)
end
function rrule!!(::CoDual{typeof(exp)}, x::CoDual{Core.BFloat16})
    y = exp(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), dy * y
    return zero_fcodual(y), pb
end

@is_primitive MinimalCtx Tuple{typeof(exp2),Core.BFloat16}
function frule!!(::Dual{typeof(exp2)}, x::Dual{Core.BFloat16})
    y = exp2(primal(x))
    return Dual(y, tangent(x) * y * Core.BFloat16(log(2f0)))
end
function rrule!!(::CoDual{typeof(exp2)}, x::CoDual{Core.BFloat16})
    y = exp2(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), dy * y * Core.BFloat16(log(2f0))
    return zero_fcodual(y), pb
end

@is_primitive MinimalCtx Tuple{typeof(exp10),Core.BFloat16}
function frule!!(::Dual{typeof(exp10)}, x::Dual{Core.BFloat16})
    y = exp10(primal(x))
    return Dual(y, tangent(x) * y * Core.BFloat16(log(10f0)))
end
function rrule!!(::CoDual{typeof(exp10)}, x::CoDual{Core.BFloat16})
    y = exp10(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), dy * y * Core.BFloat16(log(10f0))
    return zero_fcodual(y), pb
end

@is_primitive MinimalCtx Tuple{typeof(expm1),Core.BFloat16}
function frule!!(::Dual{typeof(expm1)}, x::Dual{Core.BFloat16})
    y = expm1(primal(x))
    return Dual(y, tangent(x) * (y + one(Core.BFloat16)))
end
function rrule!!(::CoDual{typeof(expm1)}, x::CoDual{Core.BFloat16})
    y = expm1(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), dy * (y + one(Core.BFloat16))
    return zero_fcodual(y), pb
end

@is_primitive MinimalCtx Tuple{typeof(log),Core.BFloat16}
function frule!!(::Dual{typeof(log)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(log(_x), nan_tangent_guard(dx, dx / _x))
end
function rrule!!(::CoDual{typeof(log)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) = NoRData(), nan_tangent_guard(dy, dy / _x)
    return zero_fcodual(log(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(log2),Core.BFloat16}
function frule!!(::Dual{typeof(log2)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(log2(_x), nan_tangent_guard(dx, dx / (_x * Core.BFloat16(log(2f0)))))
end
function rrule!!(::CoDual{typeof(log2)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) =
        NoRData(), nan_tangent_guard(dy, dy / (_x * Core.BFloat16(log(2f0))))
    return zero_fcodual(log2(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(log10),Core.BFloat16}
function frule!!(::Dual{typeof(log10)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(log10(_x), nan_tangent_guard(dx, dx / (_x * Core.BFloat16(log(10f0)))))
end
function rrule!!(::CoDual{typeof(log10)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) =
        NoRData(), nan_tangent_guard(dy, dy / (_x * Core.BFloat16(log(10f0))))
    return zero_fcodual(log10(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(log1p),Core.BFloat16}
function frule!!(::Dual{typeof(log1p)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(log1p(_x), nan_tangent_guard(dx, dx / (one(Core.BFloat16) + _x)))
end
function rrule!!(::CoDual{typeof(log1p)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) =
        NoRData(), nan_tangent_guard(dy, dy / (one(Core.BFloat16) + _x))
    return zero_fcodual(log1p(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(sin),Core.BFloat16}
function frule!!(::Dual{typeof(sin)}, x::Dual{Core.BFloat16})
    s, c = sincos(primal(x))
    return Dual(s, tangent(x) * c)
end
function rrule!!(::CoDual{typeof(sin)}, x::CoDual{Core.BFloat16})
    s, c = sincos(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), dy * c
    return zero_fcodual(s), pb
end

@is_primitive MinimalCtx Tuple{typeof(cos),Core.BFloat16}
function frule!!(::Dual{typeof(cos)}, x::Dual{Core.BFloat16})
    s, c = sincos(primal(x))
    return Dual(c, -tangent(x) * s)
end
function rrule!!(::CoDual{typeof(cos)}, x::CoDual{Core.BFloat16})
    s, c = sincos(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), -dy * s
    return zero_fcodual(c), pb
end

@is_primitive MinimalCtx Tuple{typeof(tan),Core.BFloat16}
function frule!!(::Dual{typeof(tan)}, x::Dual{Core.BFloat16})
    y = tan(primal(x))
    return Dual(y, tangent(x) * (one(Core.BFloat16) + y^2))
end
function rrule!!(::CoDual{typeof(tan)}, x::CoDual{Core.BFloat16})
    y = tan(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), dy * (one(Core.BFloat16) + y^2)
    return zero_fcodual(y), pb
end

@is_primitive MinimalCtx Tuple{typeof(asin),Core.BFloat16}
function frule!!(::Dual{typeof(asin)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(asin(_x), nan_tangent_guard(dx, dx / sqrt(one(Core.BFloat16) - _x^2)))
end
function rrule!!(::CoDual{typeof(asin)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) =
        NoRData(), nan_tangent_guard(dy, dy / sqrt(one(Core.BFloat16) - _x^2))
    return zero_fcodual(asin(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(acos),Core.BFloat16}
function frule!!(::Dual{typeof(acos)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(acos(_x), nan_tangent_guard(dx, -dx / sqrt(one(Core.BFloat16) - _x^2)))
end
function rrule!!(::CoDual{typeof(acos)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) =
        NoRData(), nan_tangent_guard(dy, -dy / sqrt(one(Core.BFloat16) - _x^2))
    return zero_fcodual(acos(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(atan),Core.BFloat16}
function frule!!(::Dual{typeof(atan)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(atan(_x), dx / (one(Core.BFloat16) + _x^2))
end
function rrule!!(::CoDual{typeof(atan)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) = NoRData(), dy / (one(Core.BFloat16) + _x^2)
    return zero_fcodual(atan(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(sinh),Core.BFloat16}
function frule!!(::Dual{typeof(sinh)}, x::Dual{Core.BFloat16})
    _x = primal(x)
    return Dual(sinh(_x), tangent(x) * cosh(_x))
end
function rrule!!(::CoDual{typeof(sinh)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) = NoRData(), dy * cosh(_x)
    return zero_fcodual(sinh(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(cosh),Core.BFloat16}
function frule!!(::Dual{typeof(cosh)}, x::Dual{Core.BFloat16})
    _x = primal(x)
    return Dual(cosh(_x), tangent(x) * sinh(_x))
end
function rrule!!(::CoDual{typeof(cosh)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) = NoRData(), dy * sinh(_x)
    return zero_fcodual(cosh(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(tanh),Core.BFloat16}
function frule!!(::Dual{typeof(tanh)}, x::Dual{Core.BFloat16})
    y = tanh(primal(x))
    return Dual(y, tangent(x) * (one(Core.BFloat16) - y^2))
end
function rrule!!(::CoDual{typeof(tanh)}, x::CoDual{Core.BFloat16})
    y = tanh(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), dy * (one(Core.BFloat16) - y^2)
    return zero_fcodual(y), pb
end

@is_primitive MinimalCtx Tuple{typeof(asinh),Core.BFloat16}
function frule!!(::Dual{typeof(asinh)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(asinh(_x), dx / sqrt(one(Core.BFloat16) + _x^2))
end
function rrule!!(::CoDual{typeof(asinh)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) = NoRData(), dy / sqrt(one(Core.BFloat16) + _x^2)
    return zero_fcodual(asinh(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(acosh),Core.BFloat16}
function frule!!(::Dual{typeof(acosh)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(acosh(_x), nan_tangent_guard(dx, dx / sqrt(_x^2 - one(Core.BFloat16))))
end
function rrule!!(::CoDual{typeof(acosh)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) =
        NoRData(), nan_tangent_guard(dy, dy / sqrt(_x^2 - one(Core.BFloat16)))
    return zero_fcodual(acosh(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(atanh),Core.BFloat16}
function frule!!(::Dual{typeof(atanh)}, x::Dual{Core.BFloat16})
    _x, dx = extract(x)
    return Dual(atanh(_x), nan_tangent_guard(dx, dx / (one(Core.BFloat16) - _x^2)))
end
function rrule!!(::CoDual{typeof(atanh)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) =
        NoRData(), nan_tangent_guard(dy, dy / (one(Core.BFloat16) - _x^2))
    return zero_fcodual(atanh(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(sinpi),Core.BFloat16}
function frule!!(::Dual{typeof(sinpi)}, x::Dual{Core.BFloat16})
    s, c = sincospi(primal(x))
    return Dual(s, tangent(x) * Core.BFloat16(Float32(π)) * c)
end
function rrule!!(::CoDual{typeof(sinpi)}, x::CoDual{Core.BFloat16})
    s, c = sincospi(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), dy * Core.BFloat16(Float32(π)) * c
    return zero_fcodual(s), pb
end

@is_primitive MinimalCtx Tuple{typeof(cospi),Core.BFloat16}
function frule!!(::Dual{typeof(cospi)}, x::Dual{Core.BFloat16})
    s, c = sincospi(primal(x))
    return Dual(c, -tangent(x) * Core.BFloat16(Float32(π)) * s)
end
function rrule!!(::CoDual{typeof(cospi)}, x::CoDual{Core.BFloat16})
    s, c = sincospi(primal(x))
    pb(dy::Core.BFloat16) = NoRData(), -dy * Core.BFloat16(Float32(π)) * s
    return zero_fcodual(c), pb
end

@is_primitive MinimalCtx Tuple{typeof(hypot),Core.BFloat16,Core.BFloat16}
function frule!!(::Dual{typeof(hypot)}, x::Dual{Core.BFloat16}, y::Dual{Core.BFloat16})
    _x, _y = primal(x), primal(y)
    h = hypot(_x, _y)
    dh = nan_tangent_guard(tangent(x), _x * tangent(x)) +
         nan_tangent_guard(tangent(y), _y * tangent(y))
    return Dual(h, dh / h)
end
function rrule!!(
    ::CoDual{typeof(hypot)}, x::CoDual{Core.BFloat16}, y::CoDual{Core.BFloat16}
)
    _x, _y = primal(x), primal(y)
    h = hypot(_x, _y)
    pb(dh::Core.BFloat16) = NoRData(),
        nan_tangent_guard(dh, dh * _x / h),
        nan_tangent_guard(dh, dh * _y / h)
    return zero_fcodual(h), pb
end

@is_primitive MinimalCtx Tuple{typeof(^),Core.BFloat16,Core.BFloat16}
function frule!!(::Dual{typeof(^)}, x::Dual{Core.BFloat16}, y::Dual{Core.BFloat16})
    _x, _y = primal(x), primal(y)
    z = _x^_y
    dz = _y * _x^(_y - one(Core.BFloat16)) * tangent(x) + z * log(_x) * tangent(y)
    return Dual(z, dz)
end
function rrule!!(::CoDual{typeof(^)}, x::CoDual{Core.BFloat16}, y::CoDual{Core.BFloat16})
    _x, _y = primal(x), primal(y)
    z = _x^_y
    function pow_pb(dz::Core.BFloat16)
        return NoRData(), dz * _y * _x^(_y - one(Core.BFloat16)), dz * z * log(_x)
    end
    return zero_fcodual(z), pow_pb
end

@is_primitive MinimalCtx Tuple{typeof(atan),Core.BFloat16,Core.BFloat16}
function frule!!(::Dual{typeof(atan)}, y::Dual{Core.BFloat16}, x::Dual{Core.BFloat16})
    _y, _x = primal(y), primal(x)
    r2 = _x^2 + _y^2
    return Dual(atan(_y, _x), (tangent(y) * _x - tangent(x) * _y) / r2)
end
function rrule!!(::CoDual{typeof(atan)}, y::CoDual{Core.BFloat16}, x::CoDual{Core.BFloat16})
    _y, _x = primal(y), primal(x)
    r2 = _x^2 + _y^2
    pb(dz::Core.BFloat16) = NoRData(), dz * _x / r2, -dz * _y / r2
    return zero_fcodual(atan(_y, _x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(max),Core.BFloat16,Core.BFloat16}
function frule!!(::Dual{typeof(max)}, x::Dual{Core.BFloat16}, y::Dual{Core.BFloat16})
    _x, _y = primal(x), primal(y)
    return Dual(max(_x, _y), _x >= _y ? tangent(x) : tangent(y))
end
function rrule!!(::CoDual{typeof(max)}, x::CoDual{Core.BFloat16}, y::CoDual{Core.BFloat16})
    _x, _y = primal(x), primal(y)
    _x_wins = _x >= _y
    pb(dz::Core.BFloat16) = NoRData(),
        _x_wins ? dz : zero(Core.BFloat16),
        _x_wins ? zero(Core.BFloat16) : dz
    return zero_fcodual(max(_x, _y)), pb
end

@is_primitive MinimalCtx Tuple{typeof(min),Core.BFloat16,Core.BFloat16}
function frule!!(::Dual{typeof(min)}, x::Dual{Core.BFloat16}, y::Dual{Core.BFloat16})
    _x, _y = primal(x), primal(y)
    return Dual(min(_x, _y), _x <= _y ? tangent(x) : tangent(y))
end
function rrule!!(::CoDual{typeof(min)}, x::CoDual{Core.BFloat16}, y::CoDual{Core.BFloat16})
    _x, _y = primal(x), primal(y)
    _x_wins = _x <= _y
    pb(dz::Core.BFloat16) = NoRData(),
        _x_wins ? dz : zero(Core.BFloat16),
        _x_wins ? zero(Core.BFloat16) : dz
    return zero_fcodual(min(_x, _y)), pb
end

@is_primitive MinimalCtx Tuple{typeof(abs),Core.BFloat16}
function frule!!(::Dual{typeof(abs)}, x::Dual{Core.BFloat16})
    _x = primal(x)
    return Dual(abs(_x), _x >= zero(Core.BFloat16) ? tangent(x) : -tangent(x))
end
function rrule!!(::CoDual{typeof(abs)}, x::CoDual{Core.BFloat16})
    _x = primal(x)
    pb(dy::Core.BFloat16) = NoRData(), _x >= zero(Core.BFloat16) ? dy : -dy
    return zero_fcodual(abs(_x)), pb
end

@is_primitive MinimalCtx Tuple{typeof(Base.eps),Core.BFloat16}
function frule!!(::Dual{typeof(Base.eps)}, x::Dual{Core.BFloat16})
    return Dual(eps(primal(x)), zero(Core.BFloat16))
end
function rrule!!(::CoDual{typeof(Base.eps)}, x::CoDual{Core.BFloat16})
    pb(::Core.BFloat16) = NoRData(), zero(Core.BFloat16)
    return zero_fcodual(eps(primal(x))), pb
end

@is_primitive MinimalCtx Tuple{typeof(nextfloat),Core.BFloat16}
function frule!!(::Dual{typeof(nextfloat)}, x::Dual{Core.BFloat16})
    return Dual(nextfloat(primal(x)), tangent(x))
end
function rrule!!(::CoDual{typeof(nextfloat)}, x::CoDual{Core.BFloat16})
    pb(dy::Core.BFloat16) = (NoRData(), dy)
    return zero_fcodual(nextfloat(primal(x))), pb
end

@is_primitive MinimalCtx Tuple{typeof(prevfloat),Core.BFloat16}
function frule!!(::Dual{typeof(prevfloat)}, x::Dual{Core.BFloat16})
    return Dual(prevfloat(primal(x)), tangent(x))
end
function rrule!!(::CoDual{typeof(prevfloat)}, x::CoDual{Core.BFloat16})
    pb(dy::Core.BFloat16) = (NoRData(), dy)
    return zero_fcodual(prevfloat(primal(x))), pb
end

end # @static if isdefined(Core, :BFloat16)

function hand_written_rule_test_cases(rng_ctor, ::Val{:bfloat16})
    @static if !isdefined(Core, :BFloat16)
        return Any[], Any[]
    end
    P = Core.BFloat16
    cases = [
        (Float32, P(0.5)),
        (Float64, P(0.5)),
        (P, 0.5f0),
        (P, 0.5),
        (sqrt, P(0.5)),
        (cbrt, P(0.4)),
        (exp, P(1.1)),
        (exp2, P(1.12)),
        (exp10, P(0.55)),
        (expm1, P(-0.3)),
        (log, P(0.1)),
        (log2, P(0.15)),
        (log10, P(0.1)),
        (log1p, P(0.95)),
        (sin, P(1.1)),
        (cos, P(1.1)),
        (tan, P(0.5)),
        (sinpi, P(1.5)),
        (cospi, P(-0.5)),
        (asin, P(0.77)),
        (acos, P(0.53)),
        (atan, P(0.77)),
        (sinh, P(-0.56)),
        (cosh, P(0.4)),
        (tanh, P(0.25)),
        (asinh, P(1.45)),
        (acosh, P(1.56)),
        (atanh, P(-0.44)),
        (hypot, P(4.0), P(5.0)),
        (^, P(4.0), P(2.0)),
        (atan, P(4.3), P(0.23)),
        (max, P(1.5), P(0.5)),
        (max, P(0.45), P(1.1)),
        (min, P(1.5), P(0.5)),
        (min, P(0.45), P(1.1)),
        (abs, P(0.5)),
        (abs, P(-0.5)),
        (Base.eps, P(1.0)),
        (nextfloat, P(0.25)),
        (prevfloat, P(1.0)),
    ]
    return map(case -> (false, :none, nothing, case...), cases), Any[]
end

derived_rule_test_cases(rng_ctor, ::Val{:bfloat16}) = Any[], Any[]
