#
# nforward-backed primitive rules for Tier 1 scalar functions.
#
# Each entry registers both forward (`frule!!`, chunk_size=1) and reverse-mode
# (`build_primitive_rrule`, chunk_size=DOF) rules via the nforward engine, which
# avoids hand-coding pullbacks for well-supported scalar operations.
#
# MinimalCtx is used throughout here rather than DefaultCtx: several of these
# functions (e.g. tanpi, sincosd, sincospi) contain try/catch internally, which
# Mooncake's IR-transform-based AD cannot handle.  Registering as MinimalCtx
# primitives ensures that the nforward rule is dispatched directly, bypassing
# any problematic IR transforms.
#
# Tier 1 — single-input scalars (DOF=1):
#   tanpi, sincosd, sincospi, modf
# Tier 1 — multi-input scalars (DOF=3):
#   clamp(x, lo, hi)
#
# Note: sincos is already registered in fastmath.jl (Base.FastMath.sincos ===
# sincos), so it is intentionally omitted here to avoid a duplicate @is_primitive.
#
# Tuple-output functions (sincosd, sincospi, modf) return
# `Dual{Tuple{P,P}}` from frule!! and a cotangent tuple from the pullback.
#

# ── tanpi ─────────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(tanpi),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(tanpi)}, x::Dual{P}) where {P<:IEEEFloat}
    return NForwardRule{Tuple{typeof(tanpi),P},1}()(f, x)
end
function build_primitive_rrule(sig::Type{<:Tuple{typeof(tanpi),<:IEEEFloat}})
    return nforward_build_rrule(sig; chunk_size=1)
end

# ── clamp(x, lo, hi) ──────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(clamp),P,P,P} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(clamp)}, x::Dual{P}, lo::Dual{P}, hi::Dual{P}
) where {P<:IEEEFloat}
    return NForwardRule{Tuple{typeof(clamp),P,P,P},1}()(f, x, lo, hi)
end
function build_primitive_rrule(sig::Type{<:Tuple{typeof(clamp),P,P,P}}) where {P<:IEEEFloat}
    # chunk_size=3 == DOF: all three inputs (x, lo, hi) are simultaneously differentiable.
    # Using chunk_size=3 computes ∂/∂x, ∂/∂lo, and ∂/∂hi in a single forward pass.  When
    # lo/hi are known constants at the call site, inference will fold their partials to zero
    # at no extra runtime cost.
    return nforward_build_rrule(sig; chunk_size=3)
end

# ── sincosd ───────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincosd),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(sincosd)}, x::Dual{P}) where {P<:IEEEFloat}
    return NForwardRule{Tuple{typeof(sincosd),P},1}()(f, x)
end
function build_primitive_rrule(sig::Type{<:Tuple{typeof(sincosd),<:IEEEFloat}})
    return nforward_build_rrule(sig; chunk_size=1)
end

# ── sincospi ──────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincospi),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(sincospi)}, x::Dual{P}) where {P<:IEEEFloat}
    return NForwardRule{Tuple{typeof(sincospi),P},1}()(f, x)
end
function build_primitive_rrule(sig::Type{<:Tuple{typeof(sincospi),<:IEEEFloat}})
    return nforward_build_rrule(sig; chunk_size=1)
end

# ── modf ──────────────────────────────────────────────────────────────────────
# modf(x) = (frac, int) where frac = x - trunc(x); d(frac)/dx = 1, d(int)/dx = 0.

@is_primitive MinimalCtx Tuple{typeof(modf),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(modf)}, x::Dual{P}) where {P<:IEEEFloat}
    return NForwardRule{Tuple{typeof(modf),P},1}()(f, x)
end
function build_primitive_rrule(sig::Type{<:Tuple{typeof(modf),<:IEEEFloat}})
    return nforward_build_rrule(sig; chunk_size=1)
end
