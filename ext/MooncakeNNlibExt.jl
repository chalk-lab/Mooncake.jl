module MooncakeNNlibExt

using NNlib, Random, Mooncake
import NNlib.GPUArraysCore: AbstractGPUArray
using Base: IEEEFloat
using LinearAlgebra
using NNlib: conv, depthwiseconv, logsoftmax, softmax, logsumexp, dropout
using Mooncake.Nfwd: NDual

import Mooncake:
    @from_rrule,
    DefaultCtx,
    MinimalCtx,
    @is_primitive,
    rrule!!,
    CoDual,
    NoRData,
    zero_fcodual,
    primal,
    tangent,
    arrayify,
    frule!!,
    Dual

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

function NNlib.logsumexp(x::AbstractVector{NDual{T,N}}) where {T<:IEEEFloat,N}
    isempty(x) && return NDual{T,N}(typemin(T))
    u = @inbounds x[begin].value
    @inbounds for i in (firstindex(x) + 1):lastindex(x)
        v = x[i].value
        v > u && (u = v)
    end
    isinf(u) && return _nf_logsumexp_inf(x, u)
    sum_w = zero(T)
    grad = ntuple(_ -> zero(T), Val(N))
    @inbounds for xi in x
        w = exp(xi.value - u)
        sum_w += w
        grad = _nf_logsumexp_accum(grad, w, xi.partials)
    end
    y_val = u + log(sum_w)
    return NDual{T,N}(y_val, _nf_logsumexp_scale(grad, inv(sum_w)))
end

# Array types which we test rules against, so are confident work.
# Parametric on both element type P and dimensionality N.
const SupportedArray{P,N} = Union{
    Array{P,N},
    AbstractGPUArray{P,N},
    Adjoint{P,<:Union{Array{P,N},AbstractGPUArray{P,N}}},
    Transpose{P,<:Union{Array{P,N},AbstractGPUArray{P,N}}},
}

# On Julia ≤ 1.11, `maximum(x::Adjoint/Transpose; dims, init)` routes through
# `LinearAlgebra.mapreducedim! → switch_dim12 → PermutedDimsArray`, leaving
# type parameters unresolved and causing JET type-stability failures.
# Collecting CPU-backed wrappers to a plain Array avoids that path.
@static if VERSION < v"1.12"
    function _maximum(
        x::Tx, dims, init
    ) where {T<:IEEEFloat,A<:Array{T},Tx<:Union{Adjoint{T,A},Transpose{T,A}}}
        return maximum(collect(x); dims, init)
    end
end
_maximum(x, dims, init) = maximum(x; dims, init)

@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(batched_mul),
        Union{Array{P,3},AbstractGPUArray{P,3}},
        Union{Array{P,3},AbstractGPUArray{P,3}},
    } where {P<:IEEEFloat},
)
@from_rrule(
    MinimalCtx,
    Tuple{typeof(dropout),AbstractRNG,SupportedArray{P,N},P} where {P<:IEEEFloat,N},
    true,
)

# logsoftmax rrules
@is_primitive MinimalCtx Tuple{
    typeof(logsoftmax),SupportedArray{T,N}
} where {T<:IEEEFloat,N}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsoftmax),SupportedArray{T,N}
} where {T<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsoftmax)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:IEEEFloat,N}
    xp = primal(x)
    y = logsoftmax(xp)
    res = zero_fcodual(y)
    function logsoftmax_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        # TODO: Drop the fallback once NNlib >= 0.9.37 is more widely supported.
        # Until then, use the public softmax backpass API when available and delegate
        # NNlib < 0.9.37 to the legacy `_data` helpers.
        # See https://github.com/chalk-lab/Mooncake.jl/pull/1229 for more context.
        @static if hasmethod(NNlib.∇logsoftmax, Tuple{AbstractArray,AbstractArray})
            dx .+= NNlib.∇logsoftmax(dy, y; dims=1)
        else
            dx .+= NNlib.∇logsoftmax_data(dy, y; dims=1)
        end
        return NoRData(), NoRData()
    end
    return res, logsoftmax_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(logsoftmax)},
    x::CoDual{<:SupportedArray{T,N}},
) where {T<:IEEEFloat,N}
    dims = primal(kw).dims
    xp = primal(x)
    y = logsoftmax(xp; dims)
    res = zero_fcodual(y)
    function logsoftmax_kw_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        @static if hasmethod(NNlib.∇logsoftmax, Tuple{AbstractArray,AbstractArray})
            dx .+= NNlib.∇logsoftmax(dy, y; dims)
        else
            dx .+= NNlib.∇logsoftmax_data(dy, y; dims)
        end
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, logsoftmax_kw_pb!!
end

# softmax rrules
@is_primitive MinimalCtx Tuple{typeof(softmax),SupportedArray{T,N}} where {T<:IEEEFloat,N}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(softmax),SupportedArray{T,N}
} where {T<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(softmax)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:IEEEFloat,N}
    xp = primal(x)
    y = softmax(xp)
    res = zero_fcodual(y)
    function softmax_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        @static if hasmethod(NNlib.∇softmax, Tuple{AbstractArray,AbstractArray})
            dx .+= NNlib.∇softmax(dy, y; dims=1)
        else
            dx .+= NNlib.∇softmax_data(dy, y; dims=1)
        end
        return NoRData(), NoRData()
    end
    return res, softmax_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(softmax)},
    x::CoDual{<:SupportedArray{T,N}},
) where {T<:IEEEFloat,N}
    dims = primal(kw).dims
    xp = primal(x)
    y = softmax(xp; dims)
    res = zero_fcodual(y)
    function softmax_kw_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        @static if hasmethod(NNlib.∇softmax, Tuple{AbstractArray,AbstractArray})
            dx .+= NNlib.∇softmax(dy, y; dims)
        else
            dx .+= NNlib.∇softmax_data(dy, y; dims)
        end
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, softmax_kw_pb!!
end

# logsumexp rrules
@is_primitive MinimalCtx Tuple{typeof(logsumexp),SupportedArray{T,N}} where {T<:IEEEFloat,N}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),SupportedArray{T,N}
} where {T<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsumexp)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:IEEEFloat,N}
    xp = primal(x)
    max_ = maximum(xp; init=typemin(T))
    @fastmath tmp = exp.(xp .- max_)
    s = sum(tmp)
    @fastmath y = max_ + log(s)
    res = zero_fcodual(y)
    function logsumexp_pb!!(dy::T)
        _, dx = arrayify(x)
        dx .+= dy .* tmp ./ s
        return NoRData(), NoRData()
    end
    return res, logsumexp_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(logsumexp)},
    x::CoDual{<:SupportedArray{T,N}},
) where {T<:IEEEFloat,N}
    dims = primal(kw).dims
    xp = primal(x)
    max_ = _maximum(xp, dims, typemin(T))
    # avoids Inf instability when xp[i]==max_==Inf
    @fastmath tmp = ifelse.(xp .== max_, one(T), exp.(xp .- max_))
    s = sum(tmp; dims)
    @fastmath y = max_ .+ log.(s)
    res = zero_fcodual(y)
    function logsumexp_kw_pb!!(::NoRData)
        _, dx = arrayify(x)
        dx .+= tangent(res) .* tmp ./ s
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, logsumexp_kw_pb!!
end

@from_rrule(
    MinimalCtx,
    Tuple{typeof(upsample_nearest),SupportedArray{<:IEEEFloat,N},NTuple{M,Int}} where {N,M},
)
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(NNlib.fold),SupportedArray{<:IEEEFloat,N},NTuple{M,Int},DenseConvDims
    } where {N,M},
)
@from_rrule(
    MinimalCtx,
    Tuple{typeof(NNlib.unfold),SupportedArray{<:IEEEFloat,N},DenseConvDims} where {N},
)
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(NNlib.scatter),
        Any,
        SupportedArray{P,N},
        SupportedArray{<:Union{Integer,Tuple},M},
    } where {P,N,M},
    true,
)

# ChainRules defines an `rrule` only for the in-place `gather!`, not `gather`, so without
# this AD would trace `gather`'s scalar per-index loop (see
# https://github.com/chalk-lab/Mooncake.jl/issues/1234). The pullback is `∇gather_src`
# accumulating straight into `src`'s fdata, avoiding its intermediate allocation.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(NNlib.gather),SupportedArray{P,N},SupportedArray{<:Union{Integer,Tuple},M}
    } where {P<:IEEEFloat,N,M},
)
function Mooncake.rrule!!(
    ::CoDual{typeof(NNlib.gather)},
    src::CoDual{<:SupportedArray{P,N}},
    idx::CoDual{<:SupportedArray{<:Union{Integer,Tuple},M}},
) where {P<:IEEEFloat,N,M}
    pidx = primal(idx)
    res = zero_fcodual(NNlib.gather(primal(src), pidx))
    function gather_pb!!(::NoRData)
        _, dsrc = arrayify(src)
        NNlib.scatter!(+, dsrc, tangent(res), pidx)
        return NoRData(), NoRData(), NoRData()
    end
    return res, gather_pb!!
end
for conv in [:conv, :depthwiseconv]
    local ∇conv_data, ∇conv_filter = Symbol.(:∇, conv, [:_data, :_filter])

    @eval @from_rrule(
        MinimalCtx,
        Tuple{
            typeof($conv),SupportedArray{P,N},SupportedArray{P,M},ConvDims
        } where {P<:IEEEFloat,N,M},
        true,
    )
    @eval @from_rrule(
        MinimalCtx,
        Tuple{
            typeof($∇conv_data),SupportedArray{P,N},SupportedArray{P,M},ConvDims
        } where {P<:IEEEFloat,N,M},
        true,
    )
end
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(∇conv_filter),SupportedArray{P,N},SupportedArray{P,M},ConvDims
    } where {P<:IEEEFloat,N,M},
    true,
)
for pool in [:maxpool, :meanpool]
    @eval @from_rrule(
        MinimalCtx,
        Tuple{typeof($pool),SupportedArray{<:IEEEFloat,N},PoolDims} where {N},
        true,
    )
end
@from_rrule(
    MinimalCtx, Tuple{typeof(pad_constant),SupportedArray{P,N},Any,Any} where {P,N}, true,
)

# Direct rules for bias_act!(identity, x, b) on CPU and GPU arrays.
# bias_act! modifies x in-place (x .+= b), so we save x's primal before mutation,
# compute in-place, return x as output, and restore x's primal in the pullback.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(bias_act!),
        typeof(identity),
        SupportedArray{<:IEEEFloat,N} where {N},
        SupportedArray{<:IEEEFloat,M} where {M},
    },
)
function frule!!(
    ::Dual{typeof(bias_act!)},
    ::Dual{typeof(identity)},
    x::Dual{<:SupportedArray{<:IEEEFloat,N}},
    b::Dual{<:SupportedArray{<:IEEEFloat,M}},
) where {N,M}
    primal(x) .+= primal(b)
    tangent(x) .+= tangent(b)
    return x
end
function rrule!!(
    ::CoDual{typeof(bias_act!)},
    ::CoDual{typeof(identity)},
    x::CoDual{<:SupportedArray{P}},
    b::CoDual{<:SupportedArray{<:IEEEFloat}},
) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    pb, db = arrayify(b)
    px_copy = copy(px)
    px .+= pb
    # Dims over which b is broadcast (size 1 in b but potentially larger in x).
    broadcast_dims = Tuple(filter(d -> size(pb, d) == 1, 1:ndims(px)))
    function bias_act_id_pb!!(::NoRData)
        if isempty(broadcast_dims)
            db .+= dx
        else
            db .+= reshape(sum(dx; dims=broadcast_dims), size(pb))
        end
        copyto!(px, px_copy)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return x, bias_act_id_pb!!
end

# The logistic sigmoid, `σ(x) = 1 / (1 + exp(-x))`, and its clamped variant `sigmoid_fast`.
# Both are smooth, so neither needs a rule for the derivative to exist — but two numerical
# traps make tracing the primal give the wrong answer, at opposite ends of the input range.
#
# Near zero: the primal is written for overflow safety, branching on the sign of `x` and
# routing through `abs(x)`. AD therefore picks up a factor of `sign(x)`, which Julia defines
# as `0` at `x == 0`, and reports a derivative of `0` there instead of `1/4`. The kink that
# `abs` introduces cancels between the two branches analytically, but not through the chain
# rule. Exactly-zero arguments are common (zero-initialised biases, dead ReLU units).
#
# For saturated `x`: writing `t = exp(-abs(x))`, the derivative `σ(x) * (1 - σ(x))` equals
# `t / (1 + t)^2` for either sign of `x`. The rules use that form because it holds the small
# quantity in `t`'s exponent, where it keeps full relative precision. The textbook form does
# not: floats near `1.0` are spaced `eps` apart, so forming `σ(x) = 1 - δ` destroys any
# `δ < eps/2`, and the later `1 - σ(x)` — exact in itself — recovers only what survived.
# That quantises the derivative to one ulp and then to exactly `0` (Float64 `x ≳ 37`,
# Float32 `x ≳ 17`, Float16 `x ≳ 8`) while the true value is still a normal float. `t ≤ 1`,
# so the quotient cannot overflow.
#
# `t` and the value are computed separately, so `exp` runs twice — the primal recomputes it
# internally. Deriving the value from `t` would cut the rule body from 5.5 to 3.5 ns, about
# a tenth of a gradient through a large broadcast, and is bit-identical over [-5, 5]. It is
# not done because it means restating the primal's branch here, and `sigmoid_fast`'s clamps
# with it: 2 ns against a rule whose *value* could then drift from the function it
# differentiates, in a formulation that has already changed once. Revisit with a profile
# showing σ matters, and add a value-agreement test if so.
#
# Both rules differentiate the unclamped sigmoid. For large positive `x` that is not a
# choice: σ is already exactly `1.0` in Float64 from about 36.8, so both primals are flat
# there and the analytic derivative — 4.2e-18 at `x = 40` — is all there is to report.
# `sigmoid_fast`'s `x < -80` clamp does change the value, to exactly `0` where σ gives
# 6.6e-36, and there the derivative reported is σ's rather than the clamped primal's `0`.
# Deliberate: NNlib documents `sigmoid_fast` as a less accurate σ, so its clamps are an
# accuracy compromise rather than a different function, and matching them would put a
# discontinuity in the derivative at exactly -80 for the sake of a value below 1e-35. Finite
# differences cannot see either case — they give `0` at both ends.
for f in (:σ, :sigmoid_fast)
    @eval @is_primitive MinimalCtx Tuple{typeof(NNlib.$f),P} where {P<:IEEEFloat}
    @eval function Mooncake.frule!!(
        ::Dual{typeof(NNlib.$f)}, x::Dual{P}
    ) where {P<:IEEEFloat}
        t = exp(-abs(primal(x)))
        d = t / (one(P) + t)^2
        return Dual(NNlib.$f(primal(x)), tangent(x) * d)
    end
    @eval function Mooncake.rrule!!(
        ::CoDual{typeof(NNlib.$f)}, x::CoDual{P}
    ) where {P<:IEEEFloat}
        t = exp(-abs(primal(x)))
        d = t / (one(P) + t)^2
        sigmoid_pb!!(dΩ::P) = NoRData(), dΩ * d
        return zero_fcodual(NNlib.$f(primal(x))), sigmoid_pb!!
    end
    # GPU elementwise and reduction rules evaluate the whole fused broadcast on `NDual`s
    # inside one kernel, so they never reach the rules above and need the derivative too.
    @eval @inline function NNlib.$f(x::NDual{P,N}) where {P<:IEEEFloat,N}
        t = exp(-abs(x.value))
        d = t / (one(P) + t)^2
        return NDual{P,N}(NNlib.$f(x.value), ntuple(i -> x.partials[i] * d, Val(N)))
    end
end

# `tanh_fast(x::Float64)` evaluates both `y = (exp(2x) - 1) / (exp(2x) + 1)` and a Remez
# polynomial `ypoly`, and only then selects between them and `sign(x)` with
# `ifelse(x^2 > 900, sign(x), ifelse(x^2 < 0.017, ypoly, y))`. `ifelse` is a call, so the
# primal computes the branches it discards, and past `|x| ≈ 355` the `y` branch is
# `Inf/Inf`, i.e. `NaN`. The primal is unharmed — `x^2 > 900` switched to `sign(x)` from
# `|x| > 30`, well below — but reverse mode sends a zero cotangent into the discarded `y`
# and the quotient's pullback forms `0 * Inf`, so `NaN` reaches the argument. A rule keeps
# AD out of that body altogether. `gelu`/`gelu_tanh` inherit the failure at `|x| ≈ 21`, far
# inside their useful range, because they feed `λ(x + 0.044715x³)` through `tanh_fast` and
# that argument reaches 355 first. An `NDual` method is deliberately absent: `NDual` is not
# an `IEEEFloat`, so the in-kernel path reaches `Base.tanh` and never this body.
#
# Below `|x| ≈ 0.13` the primal is `ypoly` while this rule reports the analytic derivative,
# the same rule-versus-primal gap recorded for `σ`'s clamps above. The discrepancy is far
# under the polynomial's own approximation error, and matching it would mean restating the
# primal here.
#
# Writing `u = exp(-2 * abs(x))`, the derivative `1 - tanh(x)^2` is `4u / (1 + u)^2`. As for
# `σ` above, that form is used because the textbook one collapses once `tanh(x)` rounds to
# `1.0` (exactly `0` for Float64 `|x| ≳ 19.5`, Float32 `|x| ≳ 9`, against a true value near
# `1e-17`).
@is_primitive MinimalCtx Tuple{typeof(tanh_fast),P} where {P<:IEEEFloat}
function Mooncake.frule!!(::Dual{typeof(tanh_fast)}, x::Dual{P}) where {P<:IEEEFloat}
    u = exp(-2 * abs(primal(x)))
    d = 4u / (one(P) + u)^2
    return Dual(tanh_fast(primal(x)), tangent(x) * d)
end
function Mooncake.rrule!!(::CoDual{typeof(tanh_fast)}, x::CoDual{P}) where {P<:IEEEFloat}
    u = exp(-2 * abs(primal(x)))
    d = 4u / (one(P) + u)^2
    tanh_fast_pb!!(dΩ::P) = NoRData(), dΩ * d
    return zero_fcodual(tanh_fast(primal(x))), tanh_fast_pb!!
end

end
