module MooncakeBFloat16sExt

using Mooncake
using Random: AbstractRNG

#! format: off
using BFloat16s: BFloat16s

using Mooncake: @foldable

import Mooncake:
    MaybeCache,
    IncCache,
    SetToZeroCache,
    zero_tangent_internal,
    randn_tangent_internal,
    increment_internal!!,
    set_to_zero_internal!!,
    _scale_internal,
    _dot_internal,
    _add_to_primal_internal,
    tangent_to_primal_internal!!,
    primal_to_tangent_internal!!,
    zero_rdata,
    zero_rdata_from_type,
    can_produce_zero_rdata_from_type,
    nan_tangent_guard,
    NoFData,
    NoRData,
    CoDual,
    Dual,
    Lifted,
    NoTangent,
    NTangent,
    primal,
    tangent,
    zero_fcodual,
    MinimalCtx

# Core.BFloat16 requires Julia >= 1.11.
# BFloat16s.BFloat16 === Core.BFloat16 is not guaranteed on all platforms.
@static if VERSION >= v"1.11-" && BFloat16s.BFloat16 === Core.BFloat16

# On x86_64 with LLVM >= 15, BFloat16s.BFloat16 === Core.BFloat16.
# On other platforms, BFloat16s.BFloat16 is a distinct type.
# All methods below are defined on Core.BFloat16 directly (always available on Julia >= 1.11).

const P = Core.BFloat16

# zero(P) calls P(0), which requires BFloat16s.jl to define convert(Core.BFloat16, ::Int).
# These therefore live here rather than in src/rules/bfloat16.jl.
zero_tangent_internal(::P, ::MaybeCache) = zero(P)

randn_tangent_internal(rng::AbstractRNG, ::P, ::MaybeCache) = P(randn(rng, Float32))

increment_internal!!(::IncCache, x::P, y::P) = x + y

set_to_zero_internal!!(::SetToZeroCache, ::P) = zero(P)

_scale_internal(::MaybeCache, a::Float64, t::P) = P(a * Float64(t))

# Must return Float64: _dot_internal is always accumulated into a Float64 scalar.
_dot_internal(::MaybeCache, t::P, s::P) = Float64(t) * Float64(s)

_add_to_primal_internal(::MaybeCache, x::P, t::P, ::Bool) = x + t

tangent_to_primal_internal!!(::P, tx, ::MaybeCache) = tx

primal_to_tangent_internal!!(tx, x::P, ::MaybeCache) = x

zero_rdata(::P) = zero(P)

zero_rdata_from_type(::Type{P}) = zero(P)

@foldable can_produce_zero_rdata_from_type(::Type{P}) = true

@inline nan_tangent_guard(dy::P, t::P) = iszero(dy) ? zero(P) : t

# Bare-Dual entry points removed; the AD transform always supplies
# Lifted-wrapped args (see comment in `src/rules/rules_via_nfwd.jl`).

# Conversions: each `(Pout, Pin)` pair has the same per-lane apply-fwd template.
# `fwd` produces `Pout` from `Pin`; `rev` is the rrule's reverse-direction
# convert used to map an upstream `Pout` cotangent back to a `Pin` cotangent.
# `Float64→P` and `P→Float64` route through Float32 because BFloat16s doesn't
# define direct converts between `Float64` and `BFloat16`.

@inline function _bf16_convert_frule(
    ::Type{Pout}, x::Lifted{Pin,N}, fwd::F
) where {Pout,Pin,N,F}
    y = fwd(primal(x))
    lanes = tangent(x).lanes
    new_lanes = ntuple(n -> fwd(lanes[n]), Val(N))
    return Lifted{Pout,N}(y, new_lanes)
end

for (Pout, Pin, fwd, rev) in (
    (:Float32, :P, :(x -> Float32(x)), :(dy -> P(dy))),
    (:Float64, :P, :(x -> Float64(x)), :(dy -> P(Float32(dy)))),
    (:P, :Float32, :(x -> P(x)), :(dy -> Float32(dy))),
    (:P, :Float64, :(x -> P(Float32(x))), :(dy -> Float64(Float32(dy)))),
)
    @eval begin
        Mooncake.@is_primitive MinimalCtx Tuple{Type{$Pout},$Pin}
        @inline Mooncake.frule!!(
            ::Lifted{Type{$Pout},N}, x::Lifted{$Pin,N}
        ) where {N} = _bf16_convert_frule($Pout, x, $fwd)
        function Mooncake.rrule!!(::CoDual{Type{$Pout}}, x::CoDual{$Pin})
            pb(dy::$Pout) = (NoRData(), ($rev)(dy))
            return zero_fcodual(($fwd)(primal(x))), pb
        end
    end
end

# Math rules
#
# Most unary scalar primitives follow one of two templates: a "guarded" form
# that wraps the per-lane tangent product in `nan_tangent_guard`, and a
# "plain" form that does not. Both compute the derivative coefficient from
# `(_x, y) = (primal(x), f(primal(x)))` via a per-op closure passed to the
# shared helper. The frule/rrule pair is produced by `_register_bf16_unary`
# below to keep each primitive's source to a single line.

@inline function _bf16_unary_frule_guarded(
    f::F, x::Lifted{P,N}, deriv::D
) where {F,D,N}
    _x = primal(x)
    y = f(_x)
    d = deriv(_x, y)
    lanes = tangent(x).lanes
    new_lanes = ntuple(n -> nan_tangent_guard(lanes[n], lanes[n] * d), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end

@inline function _bf16_unary_frule_plain(
    f::F, x::Lifted{P,N}, deriv::D
) where {F,D,N}
    _x = primal(x)
    y = f(_x)
    d = deriv(_x, y)
    lanes = tangent(x).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end

# Per-op registration via @eval loop. Each entry is
# `(op, deriv_expr, guarded::Bool)` where `deriv_expr` is a 2-arg
# function-like expression `(_x, y) -> coeff` computing the derivative
# coefficient. `guarded` controls whether the tangent product is wrapped in
# `nan_tangent_guard` (true for ops like sqrt/cbrt/asin/acos that have
# singular derivative at boundary inputs).
for (op, deriv, guarded) in (
    (:sqrt, :((_x, y) -> inv(2 * y)), true),
    (:cbrt, :((_x, y) -> inv(3 * y^2)), true),
    (:exp, :((_x, y) -> y), false),
    (:exp2, :((_x, y) -> y * P(log(2.0f0))), false),
    (:exp10, :((_x, y) -> y * P(log(10.0f0))), false),
    (:expm1, :((_x, y) -> y + one(P)), false),
)
    fr_helper = guarded ? :_bf16_unary_frule_guarded : :_bf16_unary_frule_plain
    pb_body = guarded ? :(NoRData(), nan_tangent_guard(dy, dy * d)) :
                        :(NoRData(), dy * d)
    @eval begin
        Mooncake.@is_primitive MinimalCtx Tuple{typeof($op),P}
        @inline Mooncake.frule!!(
            ::Lifted{typeof($op),N}, x::Lifted{P,N}
        ) where {N} = $fr_helper($op, x, $deriv)
        function Mooncake.rrule!!(::CoDual{typeof($op)}, x::CoDual{P})
            _x = primal(x)
            y = $op(_x)
            d = ($deriv)(_x, y)
            pb(dy::P) = $pb_body
            return zero_fcodual(y), pb
        end
    end
end

# Extended unary registrations. `sin`/`cos` use separate `sin`/`cos` calls
# rather than `sincos(::BFloat16)` because `sincos` is infinitely recursive
# in Julia 1.12.
for (op, deriv, guarded) in (
    (:log, :((_x, y) -> inv(_x)), true),
    (:log2, :((_x, y) -> inv(_x * P(log(2.0f0)))), true),
    (:log10, :((_x, y) -> inv(_x * P(log(10.0f0)))), true),
    (:log1p, :((_x, y) -> inv(one(P) + _x)), true),
    (:sin, :((_x, y) -> cos(_x)), false),
    (:cos, :((_x, y) -> -sin(_x)), false),
    (:tan, :((_x, y) -> one(P) + y^2), false),
    (:asin, :((_x, y) -> inv(sqrt(one(P) - _x^2))), true),
    (:acos, :((_x, y) -> -inv(sqrt(one(P) - _x^2))), true),
    (:atan, :((_x, y) -> inv(one(P) + _x^2)), false),
    (:sinh, :((_x, y) -> cosh(_x)), false),
    (:cosh, :((_x, y) -> sinh(_x)), false),
    (:tanh, :((_x, y) -> one(P) - y^2), false),
    (:asinh, :((_x, y) -> inv(sqrt(one(P) + _x^2))), false),
    (:acosh, :((_x, y) -> inv(sqrt(_x^2 - one(P)))), true),
    (:atanh, :((_x, y) -> inv(one(P) - _x^2)), true),
    # abs's derivative is sign(_x) (locally constant ±1; non-differentiable at 0).
    (:abs, :((_x, y) -> _x >= zero(P) ? one(P) : -one(P)), false),
    # eps is non-differentiable in the standard sense; rule returns zero tangent.
    (:(Base.eps), :((_x, y) -> zero(P)), false),
    # nextfloat/prevfloat are step functions: locally constant primal,
    # tangent passes through unchanged (derivative = 1).
    (:nextfloat, :((_x, y) -> one(P)), false),
    (:prevfloat, :((_x, y) -> one(P)), false),
)
    fr_helper = guarded ? :_bf16_unary_frule_guarded : :_bf16_unary_frule_plain
    pb_body = guarded ? :(NoRData(), nan_tangent_guard(dy, dy * d)) :
                        :(NoRData(), dy * d)
    @eval begin
        Mooncake.@is_primitive MinimalCtx Tuple{typeof($op),P}
        @inline Mooncake.frule!!(
            ::Lifted{typeof($op),N}, x::Lifted{P,N}
        ) where {N} = $fr_helper($op, x, $deriv)
        function Mooncake.rrule!!(::CoDual{typeof($op)}, x::CoDual{P})
            _x = primal(x)
            y = $op(_x)
            d = ($deriv)(_x, y)
            pb(dy::P) = $pb_body
            return zero_fcodual(y), pb
        end
    end
end


Mooncake.@is_primitive MinimalCtx Tuple{typeof(hypot),P,P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(hypot),N}, x::Lifted{P,N}, y::Lifted{P,N}
) where {N}
    _x, _y = primal(x), primal(y)
    h = hypot(_x, _y)
    inv_h = inv(h)
    new_lanes = ntuple(Val(N)) do n
        xn, yn = tangent(x, n), tangent(y, n)
        (nan_tangent_guard(xn, _x * xn) + nan_tangent_guard(yn, _y * yn)) * inv_h
    end
    return Lifted{typeof(h),N}(h, new_lanes)
end
function Mooncake.rrule!!(::CoDual{typeof(hypot)}, x::CoDual{P}, y::CoDual{P})
    _x, _y = primal(x), primal(y)
    h = hypot(_x, _y)
    pb(dh::P) = NoRData(),
    nan_tangent_guard(dh, dh * _x / h),
    nan_tangent_guard(dh, dh * _y / h)
    return zero_fcodual(h), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(^),P,P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(^),N}, x::Lifted{P,N}, y::Lifted{P,N}
) where {N}
    _x, _y = primal(x), primal(y)
    z = _x^_y
    # `_x^(_y-1)` is computed separately from `z` to handle `_x=0`:
    # `z/_x = 0/0 = NaN`, whereas `_x^(_y-1)` gives the correct value at the
    # boundary (e.g. Inf when 0 < _y < 1, since d/dx(x^y)|_{x=0} diverges).
    dx_coeff = _y * _x^(_y - one(P))
    # `z*log(_x)` (not `tangent(y)`) is guarded: when `_x=0`, `z=0` and
    # `log(_x)=-Inf`, so `z*log(_x)*tangent(y) = 0*(-Inf)*tangent(y) = NaN`.
    dy_base = z * log(_x)
    new_lanes = ntuple(Val(N)) do n
        xn, yn = tangent(x, n), tangent(y, n)
        nan_tangent_guard(xn, dx_coeff * xn) + nan_tangent_guard(z, dy_base * yn)
    end
    return Lifted{typeof(z),N}(z, new_lanes)
end
function Mooncake.rrule!!(::CoDual{typeof(^)}, x::CoDual{P}, y::CoDual{P})
    _x, _y = primal(x), primal(y)
    z = _x^_y
    function pow_pb(dz::P)
        return NoRData(),
            nan_tangent_guard(dz, dz * _y * _x^(_y - one(P))),
            # Inner guard on z: prevents 0*(-Inf)=NaN when _x=0 (z=0, log(_x)=-Inf).
            # Outer guard on dz: standard upstream-zero mask (dz=0 → zero gradient).
            nan_tangent_guard(dz, nan_tangent_guard(z, dz * z * log(_x)))
    end
    return zero_fcodual(z), pow_pb
end

# max / min share a comparison-based branching tangent: the winner's tangent
# propagates through. `cmp` is the primal comparison (`>=` for max, `<=` for
# min); the same `cmp` selects the winning tangent in the rrule.
for (op, _cmp) in ((:max, :>=), (:min, :<=))
    @eval begin
        Mooncake.@is_primitive MinimalCtx Tuple{typeof($op),P,P}
        @inline function Mooncake.frule!!(
            ::Lifted{typeof($op),N}, x::Lifted{P,N}, y::Lifted{P,N}
        ) where {N}
            _x, _y = primal(x), primal(y)
            z = $op(_x, _y)
            new_lanes = $_cmp(_x, _y) ? tangent(x).lanes : tangent(y).lanes
            return Lifted{typeof(z),N}(z, new_lanes)
        end
        function Mooncake.rrule!!(::CoDual{typeof($op)}, x::CoDual{P}, y::CoDual{P})
            _x_wins = $_cmp(primal(x), primal(y))
            pb(dz::P) = NoRData(), _x_wins ? dz : zero(P), _x_wins ? zero(P) : dz
            return zero_fcodual($op(primal(x), primal(y))), pb
        end
    end
end

end # @static if BFloat16s.BFloat16 === Core.BFloat16
#! format: on

end # module MooncakeBFloat16sExt
