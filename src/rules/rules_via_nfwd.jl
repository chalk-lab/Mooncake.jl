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

# ── Centralised bare-Dual / bare-NDual → Lifted dispatch adapters ────────────
#
# For primitive rules that have a Lifted-typed `frule!!` body and are registered
# via `_is_lifted_aware`, these unary adapters route bare-Dual and bare-NDual
# calls through the Lifted path. Per-op `frule!!(::Dual{op}, ::Dual{P})` and
# `frule!!(::Dual{op}, ::NDual{T,N})` duplicates that delegate to the same
# Lifted body are then redundant and can be removed. Specific bare-shape rules
# dispatch first when present; these adapters only fire when no specific rule
# exists.

@inline function frule!!(f::Dual{F}, x::Dual{P}) where {F,P<:IEEEFloat}
    Mooncake._is_lifted_aware(Tuple{F,P}) || throw(MethodError(frule!!, (f, x)))
    return Mooncake._ndual_output_to_width1(
        frule!!(
            Mooncake.Lifted{F,1}(primal(f), tangent(f)),
            Mooncake.Lifted{P,1}(primal(x), tangent(x)),
        ),
    )
end

@inline function frule!!(f::Dual{F}, a::Dual{P}, b::Dual{P}) where {F,P<:IEEEFloat}
    Mooncake._is_lifted_aware(Tuple{F,P,P}) || throw(MethodError(frule!!, (f, a, b)))
    return Mooncake._ndual_output_to_width1(
        frule!!(
            Mooncake.Lifted{F,1}(primal(f), tangent(f)),
            Mooncake.Lifted{P,1}(primal(a), tangent(a)),
            Mooncake.Lifted{P,1}(primal(b), tangent(b)),
        ),
    )
end

# Binary adapter covering `Union{IEEEFloat, Complex{<:IEEEFloat}}` with
# independent type parameters per arg. Used by SpecialFunctions rules
# (e.g. `gamma`, `loggamma`, `besselj`, ...) whose registered signatures
# mix IEEEFloat and Complex variants. The `P=Q<:IEEEFloat` case is
# strictly covered by the more-specific adapter above; this method
# fires only when at least one arg is Complex, or when P ≠ Q.
@inline function frule!!(
    f::Dual{F}, a::Dual{P}, b::Dual{Q}
) where {
    F,P<:Union{IEEEFloat,Complex{<:IEEEFloat}},Q<:Union{IEEEFloat,Complex{<:IEEEFloat}}
}
    Mooncake._is_lifted_aware(Tuple{F,P,Q}) || throw(MethodError(frule!!, (f, a, b)))
    return Mooncake._ndual_output_to_width1(
        frule!!(
            Mooncake.Lifted{F,1}(primal(f), tangent(f)),
            Mooncake.Lifted{P,1}(primal(a), tangent(a)),
            Mooncake.Lifted{Q,1}(primal(b), tangent(b)),
        ),
    )
end

@inline function frule!!(
    f::Dual{F}, a::Dual{P}, b::Dual{P}, c::Dual{P}
) where {F,P<:IEEEFloat}
    Mooncake._is_lifted_aware(Tuple{F,P,P,P}) || throw(MethodError(frule!!, (f, a, b, c)))
    return Mooncake._ndual_output_to_width1(
        frule!!(
            Mooncake.Lifted{F,1}(primal(f), tangent(f)),
            Mooncake.Lifted{P,1}(primal(a), tangent(a)),
            Mooncake.Lifted{P,1}(primal(b), tangent(b)),
            Mooncake.Lifted{P,1}(primal(c), tangent(c)),
        ),
    )
end

# Ternary adapter for `(IEEEFloat, IEEEFloat, Integer)` signatures.
# Disjoint from the all-IEEEFloat ternary above (the Integer constraint
# rules out IEEEFloat). Used by `SpecialFunctions.gamma_inc` and any
# similar mixed-Integer primitive.
@inline function frule!!(
    f::Dual{F}, a::Dual{P}, b::Dual{Q}, c::Dual{R}
) where {F,P<:IEEEFloat,Q<:IEEEFloat,R<:Integer}
    Mooncake._is_lifted_aware(Tuple{F,P,Q,R}) || throw(MethodError(frule!!, (f, a, b, c)))
    return Mooncake._ndual_output_to_width1(
        frule!!(
            Mooncake.Lifted{F,1}(primal(f), tangent(f)),
            Mooncake.Lifted{P,1}(primal(a), tangent(a)),
            Mooncake.Lifted{Q,1}(primal(b), tangent(b)),
            Mooncake.Lifted{R,1}(primal(c), tangent(c)),
        ),
    )
end

@inline function frule!!(f::Dual{F}, x::NDual{T,N}) where {F,T<:IEEEFloat,N}
    Mooncake._is_lifted_aware(Tuple{F,T}) || throw(MethodError(frule!!, (f, x)))
    return Mooncake._unlift(
        frule!!(Mooncake.Lifted{F,N}(primal(f), tangent(f)), Mooncake.Lifted{T,N}(x))
    )
end

@inline function frule!!(f::Dual{F}, a::NDual{T,N}, b::NDual{T,N}) where {F,T<:IEEEFloat,N}
    Mooncake._is_lifted_aware(Tuple{F,T,T}) || throw(MethodError(frule!!, (f, a, b)))
    return Mooncake._unlift(
        frule!!(
            Mooncake.Lifted{F,N}(primal(f), tangent(f)),
            Mooncake.Lifted{T,N}(a),
            Mooncake.Lifted{T,N}(b),
        ),
    )
end

@inline function frule!!(
    f::Dual{F}, a::NDual{T,N}, b::NDual{T,N}, c::NDual{T,N}
) where {F,T<:IEEEFloat,N}
    Mooncake._is_lifted_aware(Tuple{F,T,T,T}) || throw(MethodError(frule!!, (f, a, b, c)))
    return Mooncake._unlift(
        frule!!(
            Mooncake.Lifted{F,N}(primal(f), tangent(f)),
            Mooncake.Lifted{T,N}(a),
            Mooncake.Lifted{T,N}(b),
            Mooncake.Lifted{T,N}(c),
        ),
    )
end

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
    # One-for-one Lifted-typed rule per op; bare-Dual and bare-NDual entry
    # points are routed through the centralised adapters at the top of the file.
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
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
# Migrated to one-for-one Lifted-typed rule. Bare-Dual and bare-NDual entry
# points are routed through the centralised adapters above; the Lifted-typed
# body is the single source of truth for the derivative.

@is_primitive MinimalCtx Tuple{typeof(tanpi),P} where {P<:IEEEFloat}
@inline function frule!!(::Mooncake.Lifted{typeof(tanpi),N}, x::Mooncake.Lifted) where {N}
    return Mooncake.Lifted{_typeof(primal(x)),N}(tanpi(_unlift(x)))
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(tanpi),Any}}) = true
function rrule!!(f::CoDual{typeof(tanpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── nfwd-backed fixed-arity scalar rules ──────────────────────────────────────
# One-for-one Lifted-typed rule per op; bare-Dual and bare-NDual entry points
# are routed through the centralised binary adapters at the top of the file.
for f in (atan, Base.FastMath.atan_fast, log, ^, mod, max, min)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P} where {P<:IEEEFloat}
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

# Ternary: bare-Dual and bare-NDual go through the centralised adapters.
for f in (clamp,)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P,P} where {P<:IEEEFloat}
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

# ── sincosd / sincospi / modf ─────────────────────────────────────────────────
# Tuple-output unary scalars. Bare-Dual and bare-NDual entry points are routed
# through the centralised unary adapters; the Lifted-typed body is the single
# source of truth.

for (f, P_out) in
    ((sincosd, :(Tuple{P,P})), (sincospi, :(Tuple{P,P})), (modf, :(Tuple{P,P})))
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        @inline function frule!!(
            ::Mooncake.Lifted{typeof($f),N}, x::Mooncake.Lifted{P,N}
        ) where {N,P<:IEEEFloat}
            return Mooncake.Lifted{$P_out,N}($f(_unlift(x)))
        end
        Mooncake._is_lifted_aware(::Type{<:Tuple{typeof($f),Any}}) = true
        function rrule!!(f::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
        end
    end
end

# angle_fast is constant on real inputs, so dispatch directly to the zero-derivative path.
@zero_derivative MinimalCtx Tuple{typeof(Base.FastMath.angle_fast),P} where {P<:IEEEFloat}

# ── hypot(x, xs...) ───────────────────────────────────────────────────────────
# hypot is vararg, so it gets its own bare-Dual and bare-NDual adapters at this
# signature (the generic unary/binary/ternary adapters above don't cover Vararg).
# The Lifted-typed body remains the single source of truth.

@is_primitive MinimalCtx Tuple{typeof(hypot),P,Vararg{P}} where {P<:IEEEFloat}
@inline function frule!!(
    f::Dual{typeof(hypot)}, x::Dual{P}, xs::Vararg{Dual{P},M}
) where {P<:IEEEFloat,M}
    return Mooncake._ndual_output_to_width1(
        frule!!(
            Mooncake.Lifted{typeof(hypot),1}(primal(f), tangent(f)),
            Mooncake.Lifted{P,1}(primal(x), tangent(x)),
            ntuple(i -> Mooncake.Lifted{P,1}(primal(xs[i]), tangent(xs[i])), Val(M))...,
        ),
    )
end
@inline function frule!!(
    f::Dual{typeof(hypot)}, x::NDual{T,N}, xs::Vararg{NDual{T,N},M}
) where {T<:IEEEFloat,N,M}
    return Mooncake._unlift(
        frule!!(
            Mooncake.Lifted{typeof(hypot),N}(primal(f), tangent(f)),
            Mooncake.Lifted{T,N}(x),
            ntuple(i -> Mooncake.Lifted{T,N}(xs[i]), Val(M))...,
        ),
    )
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

# Unary float intrinsics: one-for-one Lifted-typed rules. Bare-Dual and
# bare-NDual entry points are routed through the centralised unary adapters
# at the top of the file.
for (op_sym, op_fn) in (
    (:abs_float, :abs),
    (:neg_float, :-),
    (:neg_float_fast, :-),
    (:sqrt_llvm, :sqrt),
    (:sqrt_llvm_fast, :sqrt),
)
    @eval begin
        @inline function frule!!(
            ::Mooncake.Lifted{typeof($op_sym),N}, x::Mooncake.Lifted
        ) where {N}
            return Mooncake.Lifted{_typeof(primal(x)),N}($op_fn(_unlift(x)))
        end
        Mooncake._is_lifted_aware(::Type{<:Tuple{typeof($op_sym),<:IEEEFloat}}) = true
    end
end

# Binary float intrinsics: one-for-one Lifted-typed rules. Same-shape bare-Dual
# and bare-NDual entry points are routed through the centralised binary adapters
# at the top of the file. Mixed `(NDual, Dual{<:IEEEFloat})` and
# `(Dual{<:IEEEFloat}, NDual)` retains specific overloads because the adapters
# require homogeneous arg shapes — these mixed cases arise when an
# `@inactive_intrinsic` (e.g. `sitofp(Float64, 2)`) emits a width-1
# `Dual{Float64}` alongside an `NDual` user input. Unwrapping the `Dual` to its
# primal is sound because the inactive frule produces
# `Dual(_, zero_tangent(_))`, contributing nothing.
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
        Mooncake._is_lifted_aware(
            ::Type{<:Tuple{typeof($op_sym),<:IEEEFloat,<:IEEEFloat}}
        ) = true
    end
end

# Ternary float intrinsics: one-for-one Lifted-typed rules. Same-shape entry
# points route through the centralised ternary adapters. The (≥1 NDual, rest
# Dual) mixes that arise from `@inactive_intrinsic` callers keep their specific
# overloads.
for (op_sym, op_fn) in ((:fma_float, :fma), (:muladd_float, :muladd))
    @eval begin
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
        Mooncake._is_lifted_aware(
            ::Type{<:Tuple{typeof($op_sym),<:IEEEFloat,<:IEEEFloat,<:IEEEFloat}}
        ) = true
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
    for (op_sym, op_fn) in (
        (:max_float, :max),
        (:max_float_fast, :max),
        (:min_float, :min),
        (:min_float_fast, :min),
    )
        @eval begin
            @inline function frule!!(
                ::Mooncake.Lifted{typeof($op_sym),N}, a::Mooncake.Lifted, b::Mooncake.Lifted
            ) where {N}
                return Mooncake.Lifted{_typeof(primal(a)),N}($op_fn(_unlift(a), _unlift(b)))
            end
            Mooncake._is_lifted_aware(
                ::Type{<:Tuple{typeof($op_sym),<:IEEEFloat,<:IEEEFloat}}
            ) = true
        end
    end
end

# ── scalar_math ───────────────────────────────────────────────────────────────
# Bare-NDual entry points for unary, binary, ternary, and vararg scalar_math ops
# are routed through the centralised adapters defined at the top of this file,
# so per-op bare-NDual rules are no longer needed.
