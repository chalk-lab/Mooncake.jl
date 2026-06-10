module MooncakeLogExpFunctionsExt

using LinearAlgebra: dot
using LogExpFunctions
using Base: IEEEFloat
using LinearAlgebra.BLAS: BlasFloat
import Mooncake:
    DefaultCtx,
    @from_chainrules,
    frule!!,
    rrule!!,
    CoDual,
    primal,
    tangent,
    @is_primitive,
    zero_fcodual,
    NoRData,
    extract,
    arrayify,
    Lifted,
    ImmutableDual,
    NDualArray
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
# Handles both scalar (Colon dims) and array (Int/Tuple dims) result
# shapes. Per-lane scalar/array reduction.
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kwargs::Lifted{<:NamedTuple,Nw},
    ::Lifted{typeof(logsumexp),Nw},
    x::Lifted{Array{P,D},Nw,NDualArray{P,Nw,D,Array{P,D},NDual{P,Nw}}},
) where {Nw,P<:IEEEFloat,D}
    _x = primal(x)
    kw = primal(kwargs)
    y = logsumexp(_x; kw...)
    if y isa AbstractArray
        # Array result — each lane's partial is the same shape as y.
        dy_partials = ntuple(
            lane -> sum(tangent(x).partials[lane] .* exp.(_x .- y); kw...), Val(Nw)
        )
        return Lifted{typeof(y),Nw}(y, NDualArray{P,Nw,ndims(y),typeof(y)}(y, dy_partials))
    else
        # Scalar result.
        dy_lanes = ntuple(
            lane -> sum(tangent(x).partials[lane] .* exp.(_x .- y); kw...), Val(Nw)
        )
        return Lifted{P,Nw}(y, NDual{P,Nw}(y, dy_lanes))
    end
end
# Per-lane scalar accumulation `dy_lane = sum(_dx_lane[i] * exp(_x[i] - y))`.
function frule!!(
    ::Lifted{typeof(logsumexp),Nw},
    x::Lifted{Array{P,D},Nw,NDualArray{P,Nw,D,Array{P,D},NDual{P,Nw}}},
) where {Nw,P<:IEEEFloat,D}
    _x = primal(x)
    y = logsumexp(_x)
    dy_lanes = ntuple(Val(Nw)) do lane
        _dx = tangent(x).partials[lane]
        s = zero(P)
        @inbounds for i in eachindex(_dx)
            s += _dx[i] * exp(_x[i] - y)
        end
        s
    end
    return Lifted{P,Nw}(y, NDual{P,Nw}(y, dy_lanes))
end
# Wrapped-input variants (e.g. a `view`/`SubArray`, whose forward V is an `ImmutableDual`): canonicalise
# each lane to a dense tangent via `arrayify` (mirroring the reverse rrules, which also use `arrayify`),
# then reuse the same per-lane reductions as the dense methods above. The dense `Array`/`NDualArray`
# methods are strictly more specific. Restricted to `BlasFloat` (what `arrayify` supports) — a *wrapped*
# non-`BlasFloat` array (e.g. a `Float16` `SubArray`) therefore has no matching forward rule and fails
# loudly with a `MethodError`; dense `Array{Float16}` is still covered by the `IEEEFloat` methods above.
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kwargs::Lifted{<:NamedTuple,Nw},
    ::Lifted{typeof(logsumexp),Nw},
    x::Lifted{<:AbstractArray{P},Nw,<:ImmutableDual},
) where {Nw,P<:BlasFloat}
    kw = primal(kwargs)
    px, dxs = arrayify(x)
    y = logsumexp(px; kw...)
    if y isa AbstractArray
        dy_partials = ntuple(lane -> sum(dxs[lane] .* exp.(px .- y); kw...), Val(Nw))
        return Lifted{typeof(y),Nw}(y, NDualArray{P,Nw,ndims(y),typeof(y)}(y, dy_partials))
    else
        dy_lanes = ntuple(lane -> sum(dxs[lane] .* exp.(px .- y); kw...), Val(Nw))
        return Lifted{P,Nw}(y, NDual{P,Nw}(y, dy_lanes))
    end
end
# Plain scalar logsumexp on a wrapped input: 0-alloc manual loop (mirrors the dense scalar method).
function frule!!(
    ::Lifted{typeof(logsumexp),Nw}, x::Lifted{<:AbstractArray{P},Nw,<:ImmutableDual}
) where {Nw,P<:BlasFloat}
    px, dxs = arrayify(x)
    y = logsumexp(px)
    dy_lanes = ntuple(Val(Nw)) do lane
        _dx = dxs[lane]
        s = zero(P)
        @inbounds for i in eachindex(_dx)
            s += _dx[i] * exp(px[i] - y)
        end
        s
    end
    return Lifted{P,Nw}(y, NDual{P,Nw}(y, dy_lanes))
end
# In-place logsumexp! where either argument is wrapped (or they differ in wrapper): arrayify both,
# mirroring the dense frule + reverse rrule. The dense `Array`/`NDualArray` (both args) frule above is
# strictly more specific and wins for the dense/dense case; this covers the mixed/wrapped combinations.
function frule!!(
    ::Lifted{typeof(logsumexp!),Nw},
    out::Lifted{<:AbstractArray{P},Nw},
    x::Lifted{<:AbstractArray{P},Nw},
) where {Nw,P<:BlasFloat}
    px, dxs = arrayify(x)
    y, dys = arrayify(out)
    logsumexp!(y, px)
    for lane in 1:Nw
        sum!(dys[lane], dxs[lane] .* exp.(px .- y))
    end
    return out
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
# Per-lane in-place `sum!` into `tangent(out).partials[lane]`.
function frule!!(
    ::Lifted{typeof(logsumexp!),Nw},
    out::Lifted{Array{P,Do},Nw,NDualArray{P,Nw,Do,Array{P,Do},NDual{P,Nw}}},
    x::Lifted{Array{P,Dx},Nw,NDualArray{P,Nw,Dx,Array{P,Dx},NDual{P,Nw}}},
) where {Nw,P<:IEEEFloat,Do,Dx}
    _x = primal(x)
    y = primal(out)
    logsumexp!(y, _x)
    for lane in 1:Nw
        sum!(tangent(out).partials[lane], tangent(x).partials[lane] .* exp.(_x .- y))
    end
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
