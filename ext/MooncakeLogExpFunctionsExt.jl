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
    tangent_view,
    @is_primitive,
    MinimalCtx,
    ForwardMode,
    densify_tangent,
    increment_densified_tangent!!,
    zero_fcodual,
    NoRData,
    extract,
    nan_tangent_guard,
    arrayify,
    Lifted,
    ImmutableDual,
    NDualArray
using Mooncake.Nfwd: NDual, _lane_views, _promote_matching_nduals

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
    # At an infinite maximum, scalar-NDual CUDA kernels use a uniform argmax subgradient
    # to avoid poisoning reductions with NaN. NDualArray frules and reverse rrules instead
    # return NaN consistently to flag the singularity.
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

# The zero-value branch preserves the primal limit, not its derivative.
function xlogx_value_and_partial(x)
    lx = log(x)
    z = x * lx
    return iszero(x) ? zero(z) : z, lx + one(x)
end

@inline function LogExpFunctions.xlogx(x::NDual{T,N}) where {T<:IEEEFloat,N}
    z, d = xlogx_value_and_partial(x.value)
    return NDual(z, ntuple(i -> scale_partial(d, x.partials[i]), Val(N)))
end

@is_primitive DefaultCtx Tuple{typeof(xlogx),IEEEFloat}
function rrule!!(::CoDual{typeof(xlogx)}, x::CoDual{<:IEEEFloat})
    z, d = xlogx_value_and_partial(primal(x))
    xlogx_pb!!(dz) = (NoRData(), scale_partial(d, dz))
    return zero_fcodual(z), xlogx_pb!!
end

# These rules avoid saturation in logistic and the equal-input branch in logaddexp.
@from_chainrules DefaultCtx Tuple{typeof(logistic),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logaddexp),IEEEFloat,IEEEFloat}

# Preserve the quotients at regular points, including x=0, to retain mixed derivatives.
xlogy_partials(x, y, z) = (log(y), iszero(x) && iszero(y) ? zero(x / y) : x / y)
function xlog1py_partials(x, y, z)
    v = one(y) + y
    return log1p(y), iszero(x) && iszero(v) ? zero(x / v) : x / v
end
xexpy_partials(x, y, z) = (exp(y), z)

@inline scale_partial(p, d) = isfinite(p) ? p * d : nan_tangent_guard(d, p * d)

# Forward-only timings against the transform: log1psq 1.66x/1.90x and log2mexp
# 1.52x/1.13x at widths 1/8. Reverse stays derived; log1pexp, log1mexp and
# logexpm1 measured at parity (1.02x-1.12x), so get no rule.
for f in (:log1psq, :log2mexp)
    @eval @is_primitive MinimalCtx ForwardMode Tuple{typeof($f),P} where {P<:IEEEFloat}
end
for f in (:xlogx, :log1psq, :log2mexp)
    @eval function frule!!(
        ::Lifted{typeof($f),Nw}, x::Lifted{P,Nw,NDual{P,Nw}}
    ) where {Nw,P<:IEEEFloat}
        dy = $f(tangent(x))
        y = dy.value
        return Lifted{typeof(y),Nw}(y, dy)
    end
end

# The zero-multiplier branches require rules; evaluate the original primal separately.
for f in (:xlogy, :xlog1py, :xexpy)
    partials = Symbol(f, :_partials)
    @eval begin
        @is_primitive DefaultCtx Tuple{typeof($f),IEEEFloat,Union{IEEEFloat,Integer}}
        function frule!!(
            ::Lifted{typeof($f),Nw},
            _x::Lifted{T,Nw,NDual{T,Nw}},
            _y::Lifted{S,Nw,NDual{S,Nw}},
        ) where {T<:IEEEFloat,S<:IEEEFloat,Nw}
            x, y = primal(_x), primal(_y)
            z = $f(x, y)
            a, b = $partials(x, y, z)
            xp, yp = tangent(_x).partials, tangent(_y).partials
            P = typeof(z)
            dz = ntuple(k -> P(scale_partial(a, xp[k]) + scale_partial(b, yp[k])), Val(Nw))
            return Lifted{P,Nw}(z, NDual{P,Nw}(z, dz))
        end
        # Integer `y` carries no derivative, so only the `x` partial contributes.
        function frule!!(
            ::Lifted{typeof($f),Nw}, _x::Lifted{T,Nw,NDual{T,Nw}}, _y::Lifted{<:Integer}
        ) where {T<:IEEEFloat,Nw}
            x, y = primal(_x), primal(_y)
            z = $f(x, y)
            a, _ = $partials(x, y, z)
            xp = tangent(_x).partials
            P = typeof(z)
            dz = ntuple(k -> P(scale_partial(a, xp[k])), Val(Nw))
            return Lifted{P,Nw}(z, NDual{P,Nw}(z, dz))
        end
        function rrule!!(
            ::CoDual{typeof($f)}, _x::CoDual{T}, _y::CoDual{S}
        ) where {T<:IEEEFloat,S<:Union{IEEEFloat,Integer}}
            x, y = primal(_x), primal(_y)
            z = $f(x, y)
            a, b = $partials(x, y, z)
            function pb!!(dz)
                dy = S <: Integer ? NoRData() : S(scale_partial(b, dz))
                return NoRData(), T(scale_partial(a, dz)), dy
            end
            return zero_fcodual(z), pb!!
        end
        @inline function LogExpFunctions.$f(x::NDual{T,N}, y::NDual{S,M}) where {T,S,N,M}
            xp, yp = _promote_matching_nduals($(QuoteNode(f)), x, y)
            z = $f(x.value, y.value)
            a, b = $partials(x.value, y.value, z)
            dz = ntuple(
                i -> scale_partial(a, xp.partials[i]) + scale_partial(b, yp.partials[i]),
                Val(N),
            )
            return NDual(z, dz)
        end
        @inline function LogExpFunctions.$f(x::NDual{T,N}, y::Real) where {T,N}
            z = $f(x.value, y)
            a, _ = $partials(x.value, y, z)
            return NDual(z, ntuple(i -> scale_partial(a, x.partials[i]), Val(N)))
        end
        @inline function LogExpFunctions.$f(x::Real, y::NDual{T,N}) where {T,N}
            z = $f(x, y.value)
            _, b = $partials(x, y.value, z)
            return NDual(z, ntuple(i -> scale_partial(b, y.partials[i]), Val(N)))
        end
    end
end

# logsumexp and logsumexp! need a custom rule to avoid incorrect derivatives due to
# branching in the primal implementation. (In principle, the forward-mode rule for logsumexp
# could be imported from ChainRulesCore, but that leads to extra allocations, so we
# reimplement them.)
@is_primitive DefaultCtx Tuple{typeof(logsumexp),AbstractArray{<:IEEEFloat}}
@is_primitive DefaultCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),AbstractArray{<:IEEEFloat}
}
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kwargs::Lifted{<:NamedTuple,Nw},
    ::Lifted{typeof(logsumexp),Nw},
    x::Lifted{A,Nw,<:NDualArray{P,Nw,D,A,NDual{P,Nw}}},
) where {Nw,P<:IEEEFloat,D,A<:AbstractArray{P,D}}
    _x = primal(x)
    kw = primal(kwargs)
    y = logsumexp(_x; kw...)
    w = exp.(_x .- y)  # softmax weights, lane-independent — computed once, not per lane
    tmp = similar(_x)  # scratch reused across lanes
    dy = ntuple(Val(Nw)) do lane
        tmp .= tangent_view(x, lane) .* w
        sum(tmp; kw...)
    end
    if y isa AbstractArray
        return Lifted{typeof(y),Nw}(y, NDualArray{P,Nw,ndims(y),typeof(y)}(y, dy))
    else
        return Lifted{P,Nw}(y, NDual{P,Nw}(y, dy))
    end
end
function frule!!(
    ::Lifted{typeof(logsumexp),Nw},
    x::Lifted{Array{P,D},Nw,<:NDualArray{P,Nw,D,Array{P,D},NDual{P,Nw}}},
) where {Nw,P<:IEEEFloat,D}
    _x = primal(x)
    y = logsumexp(_x)
    parts = _lane_views(tangent(x))
    # Share each softmax weight across lanes without a weight array. `foldl` keeps
    # the ntuple accumulator out of a captured, reassigned variable, avoiding boxing.
    grad = foldl(eachindex(_x); init=ntuple(_ -> zero(P), Val(Nw))) do g, i
        wi = exp(@inbounds(_x[i]) - y)
        ntuple(lane -> g[lane] + @inbounds(parts[lane][i]) * wi, Val(Nw))
    end
    return Lifted{P,Nw}(y, NDual{P,Nw}(y, grad))
end
# Dense non-`Array` storage (e.g. `CuArray`): the same per-lane reduction via broadcast,
# since scalar indexing is unavailable. The `Array` loop method above is strictly more
# specific and keeps the 0-alloc CPU path.
function frule!!(
    ::Lifted{typeof(logsumexp),Nw}, x::Lifted{A,Nw,<:NDualArray{P,Nw,D,A,NDual{P,Nw}}}
) where {Nw,P<:IEEEFloat,D,A<:AbstractArray{P,D}}
    _x = primal(x)
    y = logsumexp(_x)
    w = exp.(_x .- y)  # softmax weights, lane-independent — computed once, not per lane
    dy_lanes = ntuple(lane -> dot(tangent_view(x, lane), w), Val(Nw))
    return Lifted{P,Nw}(y, NDual{P,Nw}(y, dy_lanes))
end
# Wrapped inputs require arrayify, which supports only BlasFloat: wrapped Float16
# fails with MethodError, while dense Float16 uses the IEEEFloat methods above.
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kwargs::Lifted{<:NamedTuple,Nw},
    ::Lifted{typeof(logsumexp),Nw},
    x::Lifted{<:AbstractArray{P},Nw,<:ImmutableDual},
) where {Nw,P<:BlasFloat}
    kw = primal(kwargs)
    px, dxs = arrayify(x)
    y = logsumexp(px; kw...)
    # `dot` conjugates, so keep `sum(dxs .* w)` here (P may be Complex); w is lane-independent.
    w = exp.(px .- y)  # softmax weights, computed once, not per lane
    tmp = similar(px)  # scratch reused across lanes
    dy = ntuple(Val(Nw)) do lane
        tmp .= dxs[lane] .* w
        sum(tmp; kw...)
    end
    if y isa AbstractArray
        return Lifted{typeof(y),Nw}(y, NDualArray{P,Nw,ndims(y),typeof(y)}(y, dy))
    else
        return Lifted{P,Nw}(y, NDual{P,Nw}(y, dy))
    end
end
function frule!!(
    ::Lifted{typeof(logsumexp),Nw}, x::Lifted{<:AbstractArray{P},Nw,<:ImmutableDual}
) where {Nw,P<:BlasFloat}
    px, dxs = arrayify(x)
    y = logsumexp(px)
    # As above, foldl shares weights across lanes without boxing the accumulator.
    grad = foldl(eachindex(px); init=ntuple(_ -> zero(P), Val(Nw))) do g, i
        wi = exp(@inbounds(px[i]) - y)
        ntuple(lane -> g[lane] + @inbounds(dxs[lane][i]) * wi, Val(Nw))
    end
    return Lifted{P,Nw}(y, NDual{P,Nw}(y, grad))
end
# Canonicalise mixed/wrapped arguments; the dense/dense method is more specific.
function frule!!(
    ::Lifted{typeof(logsumexp!),Nw},
    out::Lifted{<:AbstractArray{P},Nw},
    x::Lifted{<:AbstractArray{P},Nw},
) where {Nw,P<:BlasFloat}
    px, dxs = arrayify(x)
    y, dys = arrayify(out)
    logsumexp!(y, px)
    w = exp.(px .- y)  # softmax weights, lane-independent — computed once, not per lane
    tmp = similar(px)  # scratch reused across lanes
    for lane in 1:Nw
        tmp .= dxs[lane] .* w
        sum!(dys[lane], tmp)
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
        dense = densify_tangent(_dx)
        dense .+= dy .* exp.(_x .- y)
        increment_densified_tangent!!(_dx, dense)
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
        dense = densify_tangent(_dx)
        dense .+= dy .* exp.(_x .- y)
        increment_densified_tangent!!(_dx, dense)
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
        dense = densify_tangent(_dx)
        dense .+= dy .* exp.(_x .- y)
        increment_densified_tangent!!(_dx, dense)
        return NoRData(), NoRData()
    end
    return zero_fcodual(y), logsumexp_pb!!
end

@is_primitive DefaultCtx Tuple{
    typeof(logsumexp!),AbstractArray{P},AbstractArray{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(logsumexp!),Nw},
    out::Lifted{Ao,Nw,<:NDualArray{P,Nw,Do,Ao,NDual{P,Nw}}},
    x::Lifted{Ax,Nw,<:NDualArray{P,Nw,Dx,Ax,NDual{P,Nw}}},
) where {Nw,P<:IEEEFloat,Do,Dx,Ao<:AbstractArray{P,Do},Ax<:AbstractArray{P,Dx}}
    _x = primal(x)
    y = primal(out)
    logsumexp!(y, _x)
    w = exp.(_x .- y)  # softmax weights, lane-independent — computed once, not per lane
    tmp = similar(_x)  # scratch reused across lanes
    for lane in 1:Nw
        tmp .= tangent_view(x, lane) .* w
        sum!(tangent_view(out, lane), tmp)
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
        dense = densify_tangent(_dx)
        dense .+= _dy .* exp.(_x .- y)
        increment_densified_tangent!!(_dx, dense)
        copyto!(y, old_out)
        fill!(_dy, zero(P))
        return NoRData(), NoRData(), NoRData()
    end
    return out, logsumexp!_pb!!
end

end
