module MooncakeLogExpFunctionsExt

using LinearAlgebra: dot
using LogExpFunctions
using Base: IEEEFloat
import Mooncake:
    DefaultCtx,
    frule!!,
    rrule!!,
    Dual,
    CoDual,
    primal,
    tangent,
    @is_primitive,
    densify_tangent,
    increment_densified_tangent!!,
    zero_fcodual,
    NoRData,
    extract,
    nan_tangent_guard,
    arrayify
using Mooncake.Nfwd: NDual, _promote_matching_nduals

# ── NDual performance fixes ───────────────────────────────────────────────────
# logistic(x::Real) = inv(exp(-x) + one(x)) produces a zero-partial NDual from
# one(x), making the + dispatch to NDual+NDual and generating `fadd 0.0, p[i]`
# per partial slot that LLVM cannot fold (IEEE -0.0).  Use one(T) (plain scalar)
# so the + hits the NDual+Real path whose partials are just copied, not added.
@inline function LogExpFunctions.logistic(x::NDual{T,N}) where {T<:IEEEFloat,N}
    return inv(exp(-x) + one(T))
end

# Scalar log1pexp delegation avoids dependence on private approximation thresholds.
@inline function LogExpFunctions.log1pexp(x::NDual{T,N}) where {T<:IEEEFloat,N}
    z = log1pexp(x.value)
    d = logistic(x.value)
    return NDual(z, ntuple(i -> d * x.partials[i], Val(N)))
end

@inline function LogExpFunctions.logcosh(x::NDual{T,N}) where {T<:IEEEFloat,N}
    z = logcosh(x.value)
    d = tanh(x.value)
    return NDual(z, ntuple(i -> d * x.partials[i], Val(N)))
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
function frule!!(::Dual{typeof(xlogx)}, x::Dual{<:IEEEFloat})
    z, d = xlogx_value_and_partial(primal(x))
    return Dual(z, scale_partial(d, tangent(x)))
end
function rrule!!(::CoDual{typeof(xlogx)}, x::CoDual{<:IEEEFloat})
    z, d = xlogx_value_and_partial(primal(x))
    xlogx_pb!!(dz) = (NoRData(), scale_partial(d, dz))
    return zero_fcodual(z), xlogx_pb!!
end

# Compute partials from inputs: rounded logistic tails lose their slope,
# and rounded logaddexp ties lose their normaliser.
logistic_partial(x) = logistic(x) * logistic(-x)
function logaddexp_partials(x, y)
    d = x == y && isinf(x) ? zero(x - y) : x - y
    return logistic(d), logistic(-d)
end

@is_primitive DefaultCtx Tuple{typeof(logistic),IEEEFloat}
function frule!!(::Dual{typeof(logistic)}, x::Dual{<:IEEEFloat})
    z = logistic(primal(x))
    return Dual(z, scale_partial(logistic_partial(primal(x)), tangent(x)))
end
function rrule!!(::CoDual{typeof(logistic)}, x::CoDual{<:IEEEFloat})
    d = logistic_partial(primal(x))
    logistic_pb!!(dz) = NoRData(), scale_partial(d, dz)
    return zero_fcodual(logistic(primal(x))), logistic_pb!!
end

@is_primitive DefaultCtx Tuple{typeof(logaddexp),IEEEFloat,IEEEFloat}
function frule!!(::Dual{typeof(logaddexp)}, x::Dual{<:IEEEFloat}, y::Dual{<:IEEEFloat})
    z = logaddexp(primal(x), primal(y))
    a, b = logaddexp_partials(primal(x), primal(y))
    return Dual(z, oftype(z, scale_partial(a, tangent(x)) + scale_partial(b, tangent(y))))
end
function rrule!!(
    ::CoDual{typeof(logaddexp)}, x::CoDual{P}, y::CoDual{Q}
) where {P<:IEEEFloat,Q<:IEEEFloat}
    a, b = logaddexp_partials(primal(x), primal(y))
    logaddexp_pb!!(dz) = NoRData(), P(scale_partial(a, dz)), Q(scale_partial(b, dz))
    return zero_fcodual(logaddexp(primal(x), primal(y))), logaddexp_pb!!
end

# Preserve the quotients at regular points, including x=0, to retain mixed derivatives.
xlogy_partials(x, y, z) = (log(y), iszero(x) && iszero(y) ? zero(x / y) : x / y)
function xlog1py_partials(x, y, z)
    v = one(y) + y
    return log1p(y), iszero(x) && iszero(v) ? zero(x / v) : x / v
end
xexpy_partials(x, y, z) = (exp(y), z)

@inline scale_partial(p, d) = isfinite(p) ? p * d : nan_tangent_guard(d, p * d)

# The zero-multiplier branches require rules; evaluate the original primal separately.
for f in (:xlogy, :xlog1py, :xexpy)
    partials = Symbol(f, :_partials)
    @eval begin
        @is_primitive DefaultCtx Tuple{typeof($f),IEEEFloat,Union{IEEEFloat,Integer}}
        function frule!!(
            ::Dual{typeof($f)}, _x::Dual{T}, _y::Dual{S}
        ) where {T<:IEEEFloat,S<:Union{IEEEFloat,Integer}}
            x, dx = extract(_x)
            y, dy = extract(_y)
            z = $f(x, y)
            a, b = $partials(x, y, z)
            dz = scale_partial(a, dx) + (S <: Integer ? zero(z) : scale_partial(b, dy))
            return Dual(z, typeof(z)(dz))
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

# Compute the normaliser independently of rounded `logsumexp(x)`.
# The shift cancels from weight derivatives, including at ties.
function logsumexp_shift(x, y::Number)
    isempty(x) && return y, wide(y)
    m = maximum(x)
    return m, isfinite(m) ? log(sum(xi -> exp(wide(xi) - wide(m)), x)) : zero(wide(m))
end
function logsumexp_shift(x, y::AbstractArray)
    isempty(x) && return y, wide.(y)
    m = maximum!(similar(y), x)
    l = sum!(wide.(m), exp.(wide.(x) .- wide.(m)))
    l .= ifelse.(isfinite.(m), log.(l), zero.(l))
    return m, l
end
# A Float16 sum stops growing at 2048 unit terms, so Float16 weights and sums use Float32.
wide(x) = x
wide(x::Float16) = Float32(x)
# At an infinite maximum the weight is NaN; a zero seed still contributes zero.
scaled_weight(d, x, m, l) = nan_tangent_guard(d, d * exp((x - m) - l))

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
    m, l = logsumexp_shift(_x, y)
    dy = oftype.(y, sum(scaled_weight.(_dx, _x, m, l); primal(kwargs)...))
    return Dual(y, dy)
end
function frule!!(
    ::Dual{typeof(logsumexp)}, x::Dual{<:AbstractArray{P}}
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y = logsumexp(_x)
    m = wide(isempty(_x) ? y : maximum(_x))
    s, dy, l = zero(m), zero(m), zero(m)
    # Accumulate without allocations or repeated exponentials; normalise first only
    # when directions near `floatmax` overflow the unnormalised sum.
    if isfinite(m)
        for i in eachindex(_dx)
            e = exp(wide(@inbounds(_x[i])) - m)
            s += e
            dy += wide(@inbounds(_dx[i])) * e
        end
        isfinite(dy) && return Dual(y, oftype(y, dy / s))
        dy, l = zero(m), log(s)
    end
    for i in eachindex(_dx)
        @inbounds dy += scaled_weight(wide(_dx[i]), wide(_x[i]), m, l)
    end
    return Dual(y, oftype(y, dy))
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs::CoDual{<:NamedTuple{(:dims,),<:Tuple{Colon}}},
    ::CoDual{typeof(logsumexp)},
    x::CoDual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y = logsumexp(_x; primal(kwargs)...)
    m, l = logsumexp_shift(_x, y)
    function logsumexp_pb!!(dy::P)
        dense = densify_tangent(_dx)
        dense .+= scaled_weight.(dy, _x, m, l)
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
    m, l = logsumexp_shift(_x, y)
    dy = zero(y)
    function logsumexp_pb!!(::NoRData)
        dense = densify_tangent(_dx)
        dense .+= scaled_weight.(dy, _x, m, l)
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
    m, l = logsumexp_shift(_x, y)
    function logsumexp_pb!!(dy::P)
        dense = densify_tangent(_dx)
        dense .+= scaled_weight.(dy, _x, m, l)
        increment_densified_tangent!!(_dx, dense)
        return NoRData(), NoRData()
    end
    return zero_fcodual(y), logsumexp_pb!!
end

# Read all of `x` and its tangent before writing aliased `out`.
@is_primitive DefaultCtx Tuple{
    typeof(logsumexp!),AbstractArray{P},AbstractArray{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logsumexp!)}, out::Dual{<:AbstractArray{P}}, x::Dual{<:AbstractArray{P}}
) where {P<:IEEEFloat}
    _x, _dx = arrayify(x)
    y, _dy = arrayify(out)
    r = logsumexp!(similar(y), _x)
    m, l = logsumexp_shift(_x, r)
    _dy .= sum!(similar(l), scaled_weight.(_dx, _x, m, l))
    y .= r
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
    r = logsumexp!(similar(y), _x)
    m, l = logsumexp_shift(_x, r)
    y .= r
    function logsumexp!_pb!!(::NoRData)
        copyto!(y, old_out)
        dy = copy(_dy)
        fill!(_dy, zero(P))
        dense = densify_tangent(_dx)
        dense .+= scaled_weight.(dy, _x, m, l)
        increment_densified_tangent!!(_dx, dense)
        return NoRData(), NoRData(), NoRData()
    end
    return out, logsumexp!_pb!!
end

end
