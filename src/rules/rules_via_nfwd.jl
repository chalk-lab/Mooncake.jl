#
# nfwd-backed primitive rules for scalar functions.
#
# Each entry registers direct primitive `frule!!` / `rrule!!` wrappers backed by
# the nfwd engine, which avoids hand-coding pullbacks for well-supported scalar
# operations.
#
# MinimalCtx is used throughout here rather than DefaultCtx: several of these
# functions (e.g. tanpi, sincosd, sincospi) contain try/catch internally, which
# Mooncake's IR-transform-based AD cannot handle.  Registering as MinimalCtx
# primitives ensures that the nfwd rule is dispatched directly, bypassing
# any problematic IR transforms.
#
# This file intentionally uses direct primitive wrappers rather than
# `build_primitive_*`: Mooncake still has direct primitive call sites, notably
# public `rrule!!` / `frule!!` examples and rule-to-rule forwarding paths such
# as `getfield -> lgetfield` and `setfield! -> lsetfield!`.
#
# Warning: avoid using `Rule` / `RRule` as hidden cached state for primitive rules.
# Those wrapper types own mutable workspace and are safe to reuse only when the caller
# explicitly owns the instance. Primitive rules are entered through ordinary dispatch, so
# caching a wrapper here would hide shared mutable state behind a plain rule method and
# make thread-safety hazards much less obvious.
#
# Special nfwd-backed wrappers below:
#   single-input:
#     tanpi (scalar output, DOF=1)
#     sincosd, sincospi, modf (tuple output, DOF=1)
#   multi-input:
#     atan(y, x) (scalar output, DOF=2)
#     clamp(x, lo, hi) (scalar output, DOF=3)
#

# ── nfwd-backed unary scalar rules ─────────────────────────────────────────────
for f in (
    exp,
    exp2,
    exp10,
    expm1,
    log,
    log10,
    log2,
    log1p,
    sqrt,
    cbrt,
    sin,
    cos,
    cospi,
    tan,
    sec,
    csc,
    cot,
    sind,
    cosd,
    tand,
    secd,
    cscd,
    cotd,
    sinpi,
    asin,
    acos,
    atan,
    asec,
    acsc,
    acot,
    asind,
    acosd,
    atand,
    asecd,
    acscd,
    acotd,
    sinh,
    cosh,
    tanh,
    sech,
    csch,
    coth,
    asinh,
    acosh,
    atanh,
    asech,
    acsch,
    acoth,
    sinc,
    deg2rad,
    rad2deg,
    mod2pi,
    Base.eps,
    nextfloat,
    prevfloat,
    Base.FastMath.exp_fast,
    Base.FastMath.exp2_fast,
    Base.FastMath.exp10_fast,
    Base.FastMath.atan_fast,
    Base.FastMath.sincos,
)
    # Call the primitive nfwd entrypoints directly here rather than constructing
    # `Rule{...}()` / `RRule{...}()` on every call. These wrappers sit on hot scalar paths,
    # so avoiding per-invocation wrapper construction keeps them allocation-free. See the
    # file-level warning above for why hidden cached Rule/RRule state is also a bad fit
    # for primitive rules.
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        # Direct low-level rule callers still pass bare `Dual` slots.
        @inline function frule!!(f::Dual{typeof($f)}, x::Dual{P}) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
        end
        @inline function frule!!(
            ::Mooncake.Lifted{typeof($f),N}, x::Mooncake.Lifted
        ) where {N}
            inner = _unlift(x)
            return Mooncake.Lifted{_typeof(primal(x)),N}($f(inner))
        end
        # Arity-specific registration so this unary loop doesn't overwrite the
        # binary loop's registration for `atan` (which appears in both lists).
        Mooncake._is_lifted_aware(::Type{<:Tuple{typeof($f),Any}}) = true
        function rrule!!(fcodual::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), fcodual, x)
        end
    end
end

@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base.eps),N}, x::Mooncake.Lifted{P,N}
) where {N,P<:IEEEFloat}
    return Mooncake.zero_lifted(Val(N), eps(primal(x)))
end

# ── tanpi ─────────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(tanpi),P} where {P<:IEEEFloat}
@inline function frule!!(f::Dual{typeof(tanpi)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
end
@inline function frule!!(::Mooncake.Lifted{typeof(tanpi),N}, x::Mooncake.Lifted) where {N}
    return Mooncake.Lifted{_typeof(primal(x)),N}(tanpi(_unlift(x)))
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(tanpi),Any}}) = true
function rrule!!(f::CoDual{typeof(tanpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── nfwd-backed fixed-arity scalar rules ──────────────────────────────────────
for f in (atan, Base.FastMath.atan_fast, log, ^, mod, max, min)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P} where {P<:IEEEFloat}
        @inline function frule!!(
            f::Dual{typeof($f)}, a::Dual{P}, b::Dual{P}
        ) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, a, b)
        end
        @inline function frule!!(
            ::Mooncake.Lifted{typeof($f),N}, a::Mooncake.Lifted, b::Mooncake.Lifted
        ) where {N}
            return Mooncake.Lifted{_typeof(primal(a)),N}($f(_unlift(a), _unlift(b)))
        end
        Mooncake._is_lifted_aware(::Type{<:Tuple{typeof($f),Any,Any}}) = true
        function rrule!!(
            fcodual::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}
        ) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_rrule_call(Val(2), fcodual, x1, x2)
        end
    end
end

# Integer-power fastmath rules share the same local derivative as scalar `pow_fast`,
# but only the floating-point base is differentiable.
@is_primitive MinimalCtx Tuple{
    typeof(Base.FastMath.pow_fast),P,I
} where {P<:IEEEFloat,I<:Integer}
# Lifted slots carry the integer-power derivative through
# `Base.FastMath.pow_fast(::NDual{P}, ::Integer)`'s NDual operator overload in
# `Nfwd.jl`, which handles the partials internally.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base.FastMath.pow_fast),N},
    x::Mooncake.Lifted{P,N},
    n::Mooncake.Lifted{I,N},
) where {N,P<:IEEEFloat,I<:Integer}
    return Mooncake.Lifted{P,N}(Base.FastMath.pow_fast(_unlift(x), _unlift(n)))
end
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(Base.FastMath.pow_fast),Any,Any}}) =
    true
function rrule!!(
    ::CoDual{typeof(Base.FastMath.pow_fast)}, x::CoDual{P}, n::CoDual{I}
) where {P<:IEEEFloat,I<:Integer}
    _x = primal(x)
    _n = primal(n)
    y = Base.FastMath.pow_fast(_x, _n)
    function pow_fast_pb!!(dy::P)
        return NoRData(), Nfwd._nfwd_pow_grad_x(_x, P(_n), float(y)) * dy, NoRData()
    end
    return zero_fcodual(y), pow_fast_pb!!
end

for f in (clamp,)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P,P} where {P<:IEEEFloat}
        @inline function frule!!(
            f::Dual{typeof($f)}, x1::Dual{P}, x2::Dual{P}, x3::Dual{P}
        ) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x1, x2, x3)
        end
        @inline function frule!!(
            ::Mooncake.Lifted{typeof($f),N},
            x1::Mooncake.Lifted,
            x2::Mooncake.Lifted,
            x3::Mooncake.Lifted,
        ) where {N}
            return Mooncake.Lifted{_typeof(primal(x1)),N}(
                $f(_unlift(x1), _unlift(x2), _unlift(x3))
            )
        end
        Mooncake._is_lifted_aware(::Type{<:Tuple{typeof($f),Any,Any,Any}}) = true
        function rrule!!(
            fcodual::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}, x3::CoDual{P}
        ) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_rrule_call(Val(3), fcodual, x1, x2, x3)
        end
    end
end

# ── sincosd ───────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincosd),P} where {P<:IEEEFloat}
@inline function frule!!(f::Dual{typeof(sincosd)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
end
@inline frule!!(::Dual{typeof(sincosd)}, x::NDual) = sincosd(x)
@inline function frule!!(
    ::Mooncake.Lifted{typeof(sincosd),N}, x::Mooncake.Lifted{P,N}
) where {N,P<:IEEEFloat}
    return Mooncake.Lifted{Tuple{P,P},N}(sincosd(_unlift(x)))
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(sincosd),Any}}) = true
function rrule!!(f::CoDual{typeof(sincosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── sincospi ──────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincospi),P} where {P<:IEEEFloat}
@inline function frule!!(f::Dual{typeof(sincospi)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
end
@inline frule!!(::Dual{typeof(sincospi)}, x::NDual) = sincospi(x)
@inline function frule!!(
    ::Mooncake.Lifted{typeof(sincospi),N}, x::Mooncake.Lifted{P,N}
) where {N,P<:IEEEFloat}
    return Mooncake.Lifted{Tuple{P,P},N}(sincospi(_unlift(x)))
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(sincospi),Any}}) = true
function rrule!!(f::CoDual{typeof(sincospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── modf ──────────────────────────────────────────────────────────────────────
# modf(x) = (frac, int) where frac = x - trunc(x); d(frac)/dx = 1, d(int)/dx = 0.

# angle_fast is constant on real inputs, so dispatch directly to the zero-derivative path.
@zero_derivative MinimalCtx Tuple{typeof(Base.FastMath.angle_fast),P} where {P<:IEEEFloat}

@is_primitive MinimalCtx Tuple{typeof(modf),P} where {P<:IEEEFloat}
@inline function frule!!(f::Dual{typeof(modf)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
end
@inline frule!!(::Dual{typeof(modf)}, x::NDual) = modf(x)
@inline function frule!!(
    ::Mooncake.Lifted{typeof(modf),N}, x::Mooncake.Lifted{P,N}
) where {N,P<:IEEEFloat}
    return Mooncake.Lifted{Tuple{P,P},N}(modf(_unlift(x)))
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(modf),Any}}) = true
function rrule!!(f::CoDual{typeof(modf)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── hypot(x, xs...) ───────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(hypot),P,Vararg{P}} where {P<:IEEEFloat}
@inline function frule!!(
    f::Dual{typeof(hypot)}, x::Dual{P}, xs::Vararg{Dual{P},M}
) where {P<:IEEEFloat,M}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x, xs...)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(hypot),N},
    x::Mooncake.Lifted{P,N},
    xs::Vararg{Mooncake.Lifted,M},
) where {N,P<:IEEEFloat,M}
    bare = ntuple(i -> i == 1 ? _unlift(x) : _unlift(xs[i - 1]), Val(M + 1))
    return Mooncake.Lifted{P,N}(hypot(bare...))
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(hypot),Vararg}}) = true
function rrule!!(
    f::CoDual{typeof(hypot)}, x::CoDual{P}, xs::Vararg{CoDual{P},M}
) where {P<:IEEEFloat,M}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(M + 1), f, x, xs...)
end

# ── NDual frule!! methods ─────────────────────────────────────────────────────
#
# All NDual-dispatched frule!! rules live here, organised into sections
# mirroring the files under src/rules/.

# ── builtins (intrinsic wrappers) ─────────────────────────────────────────────
using .IntrinsicsWrappers:
    abs_float,
    add_float,
    add_float_fast,
    sub_float,
    sub_float_fast,
    mul_float,
    mul_float_fast,
    div_float,
    div_float_fast,
    neg_float,
    neg_float_fast,
    sqrt_llvm,
    sqrt_llvm_fast,
    copysign_float,
    fma_float,
    muladd_float,
    fpext,
    fptrunc

frule!!(::Dual{typeof(abs_float)}, x::NDual) = abs(x)
frule!!(::Dual{typeof(neg_float)}, x::NDual) = -x
frule!!(::Dual{typeof(neg_float_fast)}, x::NDual) = -x
frule!!(::Dual{typeof(sqrt_llvm)}, x::NDual) = sqrt(x)
frule!!(::Dual{typeof(sqrt_llvm_fast)}, x::NDual) = sqrt(x)

# Audit Todo 4: `_lane1_ndual` and `_lane1_dual` are width-1-only by name.
# They bridge between the legacy bare `Dual{P, T}` shape and the canonical
# width-1 `NDual{T, 1}` shape used by Nfwd. The bare-Dual `frule!!(::Dual{$op},
# x::Dual{P, <:IEEEFloat})` overload below uses these to delegate into the
# NDual path; chunk-N callers go through the Lifted-aware overload (which
# uses `_unlift(x)::NDual{T, N}` directly and is width-N correct).
@inline _lane1_ndual(x::Dual{P,<:IEEEFloat}) where {P<:IEEEFloat} = NDual{P,1}(
    primal(x), (convert(P, tangent(x)),)
)
@inline _lane1_dual(x::NDual) = Dual(primal(x), x.partials[1])

# Lifted-typed overloads for unary float intrinsics — same canonical-V
# principle: `_unlift` produces an NDual on which the pointwise operator is
# defined, then re-wrap as Lifted to skip the `_wrap_rule_result` allocation.
for (op_sym, op_fn) in (
    (:abs_float, :abs),
    (:neg_float, :-),
    (:neg_float_fast, :-),
    (:sqrt_llvm, :sqrt),
    (:sqrt_llvm_fast, :sqrt),
)
    @eval begin
        @inline function frule!!(
            f::Dual{typeof($op_sym)}, x::Dual{P,<:IEEEFloat}
        ) where {P<:IEEEFloat}
            return _lane1_dual(frule!!(f, _lane1_ndual(x)))
        end
        @inline function frule!!(
            ::Mooncake.Lifted{typeof($op_sym),N}, x::Mooncake.Lifted
        ) where {N}
            return Mooncake.Lifted{_typeof(primal(x)),N}($op_fn(_unlift(x)))
        end
    end
end

# Mixed `(NDual, Dual{<:IEEEFloat})` cases arise when an `@inactive_intrinsic`
# (e.g. `sitofp(Float64, 2)`) emits a width-1 `Dual{Float64}` alongside an
# `NDual` user input. Unwrapping the `Dual` to its primal is sound because the
# inactive frule produces `Dual(_, zero_tangent(_))`, contributing nothing.
for (op_sym, op_fn) in (
    (:add_float, :+),
    (:add_float_fast, :+),
    (:sub_float, :-),
    (:sub_float_fast, :-),
    (:mul_float, :*),
    (:mul_float_fast, :*),
    (:div_float, :/),
    (:div_float_fast, :/),
    (:copysign_float, :copysign),
)
    @eval begin
        @inline frule!!(::Dual{typeof($op_sym)}, a::NDual, b::NDual) = $op_fn(a, b)
        @inline function frule!!(
            f::Dual{typeof($op_sym)}, a::Dual{P,<:IEEEFloat}, b::Dual{P,<:IEEEFloat}
        ) where {P<:IEEEFloat}
            return _lane1_dual(frule!!(f, _lane1_ndual(a), _lane1_ndual(b)))
        end
        @inline function frule!!(
            ::Dual{typeof($op_sym)}, a::NDual{T,N}, b::Dual{<:IEEEFloat}
        ) where {T<:IEEEFloat,N}
            return $op_fn(a, primal(b))
        end
        @inline function frule!!(
            ::Dual{typeof($op_sym)}, a::Dual{<:IEEEFloat}, b::NDual{T,N}
        ) where {T<:IEEEFloat,N}
            return $op_fn(primal(a), b)
        end
        # Lifted-typed overload — accepts canonical V from `_unlift`. Per
        # AGENTS.md "no inner-V branching", the body just calls the operator
        # which dispatches on the canonical V (NDual / Dual / mixed).
        @inline function frule!!(
            ::Mooncake.Lifted{typeof($op_sym),N}, a::Mooncake.Lifted, b::Mooncake.Lifted
        ) where {N}
            return Mooncake.Lifted{_typeof(primal(a)),N}($op_fn(_unlift(a), _unlift(b)))
        end
    end
end

# Ternary float intrinsics: NDual×NDual×NDual plus the (≥1 NDual, rest Dual)
# mixes that Nfwd already supports natively.
for (op_sym, op_fn) in ((:fma_float, :fma), (:muladd_float, :muladd))
    @eval begin
        @inline frule!!(::Dual{typeof($op_sym)}, x::NDual, y::NDual, z::NDual) = $op_fn(
            x, y, z
        )
        @inline function frule!!(
            f::Dual{typeof($op_sym)},
            x::Dual{P,<:IEEEFloat},
            y::Dual{P,<:IEEEFloat},
            z::Dual{P,<:IEEEFloat},
        ) where {P<:IEEEFloat}
            return _lane1_dual(
                frule!!(f, _lane1_ndual(x), _lane1_ndual(y), _lane1_ndual(z))
            )
        end
        @inline function frule!!(
            ::Dual{typeof($op_sym)}, x::NDual{T,N}, y::NDual{T,N}, z::Dual{<:IEEEFloat}
        ) where {T<:IEEEFloat,N}
            return $op_fn(x, y, primal(z))
        end
        @inline function frule!!(
            ::Dual{typeof($op_sym)}, x::NDual{T,N}, y::Dual{<:IEEEFloat}, z::NDual{T,N}
        ) where {T<:IEEEFloat,N}
            return $op_fn(x, primal(y), z)
        end
        @inline function frule!!(
            ::Dual{typeof($op_sym)}, x::Dual{<:IEEEFloat}, y::NDual{T,N}, z::NDual{T,N}
        ) where {T<:IEEEFloat,N}
            return $op_fn(primal(x), y, z)
        end
        # Lifted-typed overload — canonical V via `_unlift`.
        @inline function frule!!(
            ::Mooncake.Lifted{typeof($op_sym),N},
            x::Mooncake.Lifted,
            y::Mooncake.Lifted,
            z::Mooncake.Lifted,
        ) where {N}
            return Mooncake.Lifted{_typeof(primal(x)),N}(
                $op_fn(_unlift(x), _unlift(y), _unlift(z))
            )
        end
    end
end

function frule!!(
    ::Dual{typeof(fpext)}, ::Dual{Type{Pext}}, x::NDual{P,N}
) where {Pext<:IEEEFloat,P<:IEEEFloat,N}
    return convert(NDual{Pext,N}, x)
end
@inline function _convert_dual_primal_tangent(
    ::Type{Pout}, x::Dual{<:IEEEFloat,<:IEEEFloat}
) where {Pout<:IEEEFloat}
    return Dual(convert(Pout, primal(x)), convert(Pout, tangent(x)))
end
function frule!!(
    ::Dual{typeof(fpext)}, ::Dual{Type{Pext}}, x::Dual{P,<:IEEEFloat}
) where {Pext<:IEEEFloat,P<:IEEEFloat}
    return _convert_dual_primal_tangent(Pext, x)
end
function frule!!(
    ::Dual{typeof(fptrunc)}, ::Dual{Type{Ptrunc}}, x::NDual{P,N}
) where {Ptrunc<:IEEEFloat,P<:IEEEFloat,N}
    return convert(NDual{Ptrunc,N}, x)
end
function frule!!(
    ::Dual{typeof(fptrunc)}, ::Dual{Type{Ptrunc}}, x::Dual{P,<:IEEEFloat}
) where {Ptrunc<:IEEEFloat,P<:IEEEFloat}
    return _convert_dual_primal_tangent(Ptrunc, x)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(fpext),N}, ::Mooncake.Lifted{Type{Pext}}, x::Mooncake.Lifted{P}
) where {N,Pext<:IEEEFloat,P<:IEEEFloat}
    return Mooncake.Lifted{Pext,N}(convert(NDual{Pext,N}, _unlift(x)))
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(fptrunc),N},
    ::Mooncake.Lifted{Type{Ptrunc}},
    x::Mooncake.Lifted{P},
) where {N,Ptrunc<:IEEEFloat,P<:IEEEFloat}
    return Mooncake.Lifted{Ptrunc,N}(convert(NDual{Ptrunc,N}, _unlift(x)))
end
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(fpext),Type{<:IEEEFloat},<:IEEEFloat}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(fptrunc),Type{<:IEEEFloat},<:IEEEFloat}}
) = true

@static if VERSION >= v"1.12.0-rc2"
    using .IntrinsicsWrappers: max_float, max_float_fast, min_float, min_float_fast
    frule!!(::Dual{typeof(max_float)}, a::NDual, b::NDual) = max(a, b)
    frule!!(::Dual{typeof(max_float_fast)}, a::NDual, b::NDual) = max(a, b)
    frule!!(::Dual{typeof(min_float)}, a::NDual, b::NDual) = min(a, b)
    frule!!(::Dual{typeof(min_float_fast)}, a::NDual, b::NDual) = min(a, b)
    for (op_sym, op_fn) in (
        (:max_float, :max),
        (:max_float_fast, :max),
        (:min_float, :min),
        (:min_float_fast, :min),
    )
        @eval @inline function frule!!(
            ::Mooncake.Lifted{typeof($op_sym),N}, a::Mooncake.Lifted, b::Mooncake.Lifted
        ) where {N}
            return Mooncake.Lifted{_typeof(primal(a)),N}($op_fn(_unlift(a), _unlift(b)))
        end
    end
end

# ── scalar_math ───────────────────────────────────────────────────────────────
# Tuple-returning functions (sincosd, sincospi, modf) are skipped:
# their output type is Tuple, not IEEEFloat.

for f in (
    exp,
    exp2,
    exp10,
    expm1,
    log,
    log10,
    log2,
    log1p,
    sqrt,
    cbrt,
    sin,
    cos,
    cospi,
    tan,
    sec,
    csc,
    cot,
    sind,
    cosd,
    tand,
    secd,
    cscd,
    cotd,
    sinpi,
    asin,
    acos,
    atan,
    asec,
    acsc,
    acot,
    asind,
    acosd,
    atand,
    asecd,
    acscd,
    acotd,
    sinh,
    cosh,
    tanh,
    sech,
    csch,
    coth,
    asinh,
    acosh,
    atanh,
    asech,
    acsch,
    acoth,
    sinc,
    deg2rad,
    rad2deg,
    mod2pi,
    nextfloat,
    prevfloat,
    tanpi,
)
    @eval function frule!!(::Dual{typeof($f)}, x::NDual{T,N}) where {T<:IEEEFloat,N}
        return $f(x)
    end
end

# Binary functions
for f in (atan, log, ^, mod, max, min)
    @eval function frule!!(
        ::Dual{typeof($f)}, x::NDual{T,N}, y::NDual{T,N}
    ) where {T<:IEEEFloat,N}
        return $f(x, y)
    end
end

# Ternary: clamp
function frule!!(
    ::Dual{typeof(clamp)}, x::NDual{T,N}, lo::NDual{T,N}, hi::NDual{T,N}
) where {T<:IEEEFloat,N}
    return clamp(x, lo, hi)
end

# Vararg: hypot
function frule!!(
    ::Dual{typeof(hypot)}, x::NDual{T,N}, xs::Vararg{NDual{T,N},M}
) where {T<:IEEEFloat,N,M}
    return hypot(x, xs...)
end
