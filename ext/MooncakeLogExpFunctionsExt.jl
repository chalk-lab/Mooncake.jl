module MooncakeLogExpFunctionsExt

using LinearAlgebra: dot
using LogExpFunctions
using Base: IEEEFloat
import Mooncake:
    DefaultCtx,
    @from_chainrules,
    frule!!,
    rrule!!,
    Dual,
    CoDual,
    primal,
    tangent,
    @is_primitive,
    zero_fcodual,
    NoRData,
    extract,
    arrayify
using Mooncake.Nfwd: NDual

# ── NDual performance fixes ───────────────────────────────────────────────────
# logistic(x::Real) = inv(exp(-x) + one(x)) produces a zero-partial NDual from
# one(x), making the + dispatch to NDual+NDual and generating `fadd 0.0, p[i]`
# per partial slot that LLVM cannot fold (IEEE -0.0).  Use one(T) (plain scalar)
# so the + hits the NDual+Real path whose partials are just copied, not added.
@inline function LogExpFunctions.logistic(x::NDual{T,N}) where {T<:IEEEFloat,N}
    return inv(exp(-x) + one(T))
end

# _log1pexp_thresholds(x::Real) computes branch thresholds using oftype(x, logtwo),
# producing zero-partial NDuals and running NDual arithmetic on pure constants.
# Float64/Float32 have hardcoded overloads; redirect NDual to those.
@inline function LogExpFunctions._log1pexp_thresholds(x::NDual{T,N}) where {T<:IEEEFloat,N}
    return LogExpFunctions._log1pexp_thresholds(x.value)
end

# log1pexp, _log1pexp, and logcosh are not @inline in LogExpFunctions, so the
# compiler cannot fuse their branch structure with the caller's loop body.
# Specialize here so the entire computation inlines at the call site.
@inline function LogExpFunctions._log1pexp(x::NDual{T,N}) where {T<:IEEEFloat,N}
    x1, x2, x3, x4 = LogExpFunctions._log1pexp_thresholds(x)
    x < x1 && return zero(x)
    x < x2 && return exp(x)
    x < x3 && return log1p(exp(x))
    x < x4 && return x + exp(-x)
    return x
end

@inline function LogExpFunctions.logcosh(x::NDual{T,N}) where {T<:IEEEFloat,N}
    abs_x = abs(x)
    # Call _log1pexp directly (inline) rather than log1pexp, to avoid inlining the
    # 5-branch body into logaddexp (which also calls log1pexp) and bloating that path.
    return abs_x + LogExpFunctions._log1pexp(-2 * abs_x) -
           T(LogExpFunctions.IrrationalConstants.logtwo)
end

# logsumexp(x::AbstractVector{NDual}): direct scalar-then-differentiate implementation.
# The generic LogExpFunctions path runs _logsumexp_onepass_reduce which calls reduce()
# with a Tuple{NDual,NDual} accumulator — one _logsumexp_onepass_op call per element
# that internally does isnan checks, real() calls, and exp() on full NDual values.
# This implementation extracts the scalar primal values once, computes the primal
# logsumexp in scalar T, then propagates partials in a single additional pass.
# Result: same answer, no Tuple boxing, and the inner loop operates on plain T and
# NTuple{N,T} fields rather than full NDual dispatch.
#
# _nf_logsumexp_accum is a separate @inline function (not a closure) so that `grad` is a
# function parameter rather than a variable that is both captured and reassigned.  Julia
# would otherwise box `grad` in a Core.Box, forcing heap reads for every grad[k] access
# inside the ntuple closure.
@inline function _nf_logsumexp_accum(
    grad::NTuple{N,T}, w::T, partials::NTuple{N,T}
) where {N,T}
    return ntuple(k -> grad[k] + w * partials[k], Val(N))
end

@inline function _nf_logsumexp_scale(grad::NTuple{N,T}, inv_sw::T) where {N,T}
    return ntuple(k -> grad[k] * inv_sw, Val(N))
end

@inline function _nf_logsumexp_inf(x::AbstractVector{NDual{T,N}}, u::T) where {T,N}
    count_u = 0
    grad = ntuple(_ -> zero(T), Val(N))
    @inbounds for xi in x
        if xi.value == u
            count_u += 1
            grad = _nf_logsumexp_accum(grad, one(T), xi.partials)
        end
    end
    return NDual{T,N}(u, _nf_logsumexp_scale(grad, inv(T(count_u))))
end

function LogExpFunctions.logsumexp(x::AbstractVector{NDual{T,N}}) where {T<:IEEEFloat,N}
    isempty(x) && return NDual{T,N}(typemin(T))
    # Pass 1: find maximum primal value for numerical stability.
    u = @inbounds x[begin].value
    @inbounds for i in (firstindex(x) + 1):lastindex(x)
        v = x[i].value
        v > u && (u = v)
    end
    isinf(u) && return _nf_logsumexp_inf(x, u)
    # Pass 2: accumulate sum(exp(xᵢ − u)) and partial-slot weighted sums.
    # Both _nf_logsumexp_accum and _nf_logsumexp_scale take grad as a function parameter
    # rather than capturing it as a closure variable.  If any ntuple closure captured grad
    # while it is also reassigned in this scope, Julia would box grad in a Core.Box and
    # force heap access on every grad[k] read.
    sum_w = zero(T)
    grad = ntuple(_ -> zero(T), Val(N))
    @inbounds for xi in x
        w = exp(xi.value - u)
        sum_w += w
        grad = _nf_logsumexp_accum(grad, w, xi.partials)
    end
    y_val = u + log(sum_w)
    inv_sw = inv(sum_w)
    return NDual{T,N}(y_val, _nf_logsumexp_scale(grad, inv_sw))
end

# xlogx(x) = x == 0 ? zero(x*log(x)) : x*log(x).  The generic implementation computes
# x*log(x) speculatively over the full NDual before the iszero branch can discard the
# result.  Specialise to: (1) early-exit on x.value (scalar branch, no NDual work),
# (2) compute log once at scalar level, (3) apply the chain rule d/dx[x log x] = log(x)+1
# with a single scalar multiply per partial slot, saving one NDual multiply and one NDual log.
@inline function LogExpFunctions.xlogx(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    iszero(v) && return zero(x)
    lv = log(v)
    d = lv + one(T)  # derivative: d/dx[x log x] = log(x) + 1
    return NDual(v * lv, ntuple(i -> x.partials[i] * d, Val(N)))
end

# Importing these rules provides improved numerical stability for `logistic`, and avoids
# incorrect derivatives arising from a 'fast branch' in `logaddexp(x1, x2)` where x1 == x2
# (similar to that for `logsumexp` below). The other chain rules for LogExpFunctions were
# investigated and found to be no better than Mooncake's derived rules in terms of
# performance or numerical stability, so are not imported here.
@from_chainrules DefaultCtx Tuple{typeof(logistic),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logaddexp),IEEEFloat,IEEEFloat}

# logsumexp and logsumexp! need a custom rule to avoid incorrect derivatives due to
# branching in the primal implementation. (In principle, the forward-mode rule for logsumexp
# could be imported from ChainRulesCore, but that leads to extra allocations, so we
# reimplement them.)
@is_primitive DefaultCtx Tuple{typeof(logsumexp),AbstractArray{<:IEEEFloat}}
@is_primitive DefaultCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),AbstractArray{<:IEEEFloat}
}
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    kwargs::Dual{<:NamedTuple},
    ::Dual{typeof(logsumexp)},
    x::Dual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y = logsumexp(_x; primal(kwargs)...)
    dy = sum(_dx .* (exp.(_x .- y)); primal(kwargs)...)
    return Dual(y, dy)
end
function frule!!(
    ::Dual{typeof(logsumexp)}, x::Dual{<:AbstractArray{P}}
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y = logsumexp(_x)
    dy = zero(P)
    # same as dy = dot(_dx, exp.(_x .- y)) but manually looped over to avoid allocations
    for i in eachindex(_dx)
        @inbounds dy += _dx[i] * exp(_x[i] - y)
    end
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs::CoDual{<:NamedTuple{(:dims,),<:Tuple{Colon}}},
    ::CoDual{typeof(logsumexp)},
    x::CoDual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y = logsumexp(_x; primal(kwargs)...)
    function logsumexp_pb!!(dy::P)
        _dx .+= dy .* exp.(_x .- y)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), logsumexp_pb!!
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    # all other dim arguments - i.e. ints or tuples thereof
    kwargs::CoDual{<:NamedTuple{(:dims,),<:Tuple{Any}}},
    ::CoDual{typeof(logsumexp)},
    x::CoDual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y = logsumexp(_x; primal(kwargs)...)
    dy = zero(y)
    function logsumexp_pb!!(::NoRData)
        _dx .+= dy .* exp.(_x .- y)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), logsumexp_pb!!
end
function rrule!!(
    ::CoDual{typeof(logsumexp)}, x::CoDual{<:AbstractArray{P}}
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y = logsumexp(_x)
    function logsumexp_pb!!(dy::P)
        _dx .+= dy .* exp.(_x .- y)
        return NoRData(), NoRData()
    end
    return zero_fcodual(y), logsumexp_pb!!
end

@is_primitive DefaultCtx Tuple{
    typeof(logsumexp!),AbstractArray{P},AbstractArray{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logsumexp!)}, out::Dual{<:AbstractArray{P}}, x::Dual{<:AbstractArray{P}}
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y, _dy = arrayify(out)
    logsumexp!(y, _x)
    sum!(_dy, _dx .* exp.(_x .- y))
    return out
end
function rrule!!(
    ::CoDual{typeof(logsumexp!)},
    out::CoDual{<:AbstractArray{P}},
    x::CoDual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y, _dy = arrayify(out)
    old_out = copy(y)
    logsumexp!(y, _x)
    function logsumexp!_pb!!(::NoRData)
        _dx .+= _dy .* exp.(_x .- y)
        copyto!(y, old_out)
        fill!(_dy, zero(P))
        return NoRData(), NoRData(), NoRData()
    end
    return out, logsumexp!_pb!!
end

end
