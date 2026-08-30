module MooncakeFluxExt

using Mooncake, Flux
using Base: IEEEFloat
using LinearAlgebra: mul!
import Flux: NNlib
import Mooncake:
    DefaultCtx,
    frule!!,
    rrule!!,
    @is_primitive,
    Dual,
    CoDual,
    NoRData,
    _fields,
    arrayify,
    primal,
    tangent,
    zero_fcodual

# These layers create the value passed to bias_act!, so its in-place update is not externally
# observable. Handling the composition here avoids saving and restoring that disposable
# intermediate in the lower-level mutation-aware rule.
const FastFluxActivation = Union{typeof(identity),typeof(tanh)}
const FastScaleActivation = Union{typeof(identity),typeof(abs2)}

@inline _activation_derivative(::typeof(identity), x) = one(x)
@inline function _activation_derivative(::typeof(tanh), x)
    u = exp(-2 * abs(x))
    return 4u / (one(x) + u)^2
end

function _bias_activate!(::typeof(identity), x, b)
    x .+= b
    return x
end
function _bias_activate!(activation::typeof(tanh), x, b)
    # Preserve the preactivation: saturated outputs do not determine nonzero derivatives.
    return NNlib.fast_act(activation, x).(x .+ b)
end

function _activate_tangent!(::typeof(identity), dx, x, b, db)
    dx .+= db
    return dx
end
function _activate_tangent!(activation::typeof(tanh), dx, x, b, db)
    dx .= _activation_derivative.(activation, x .+ b) .* (dx .+ db)
    return dx
end

_apply_activation_pullback!(::typeof(identity), dx, x, b) = dx
function _apply_activation_pullback!(activation::typeof(tanh), dx, x, b)
    dx .*= _activation_derivative.(activation, x .+ b)
    return dx
end

function _accum_broadcast!(dest, src)
    dims = Tuple(
        dim for dim in 1:ndims(src) if
        dim > ndims(dest) || (size(dest, dim) == 1 && size(src, dim) != 1)
    )
    dest .+= isempty(dims) ? src : reshape(sum(src; dims), size(dest))
    return nothing
end

@is_primitive DefaultCtx Tuple{
    <:Flux.Scale{S,<:DenseArray{P},<:DenseArray{P}},<:DenseArray{P}
} where {P<:IEEEFloat,S<:FastScaleActivation}

function frule!!(
    layer::Dual{<:Flux.Scale{S,<:DenseArray{P},<:DenseArray{P}}}, x::Dual{<:DenseArray{P}}
) where {P<:IEEEFloat,S<:FastScaleActivation}
    p_layer = primal(layer)
    d_layer = _fields(tangent(layer))
    px, dx = arrayify(x)
    z = p_layer.scale .* px .+ p_layer.bias
    dz = d_layer.scale .* px .+ p_layer.scale .* dx .+ d_layer.bias
    if p_layer.σ === abs2
        dz .*= 2 .* z
    end
    return Dual(p_layer.σ.(z), dz)
end

function rrule!!(
    layer::CoDual{<:Flux.Scale{S,<:DenseArray{P},<:DenseArray{P}}},
    x::CoDual{<:DenseArray{P}},
) where {P<:IEEEFloat,S<:FastScaleActivation}
    p_layer = primal(layer)
    d_layer = _fields(tangent(layer))
    px, dx = arrayify(x)
    z = p_layer.scale .* px .+ p_layer.bias
    y = p_layer.σ.(z)
    dy = zero(y)
    function scale_pullback!!(::NoRData)
        if p_layer.σ === abs2
            dy .*= 2 .* z
        end
        _accum_broadcast!(d_layer.scale, dy .* px)
        _accum_broadcast!(d_layer.bias, dy)
        _accum_broadcast!(dx, dy .* p_layer.scale)
        return NoRData(), NoRData()
    end
    return CoDual(y, dy), scale_pullback!!
end

@is_primitive DefaultCtx Tuple{
    <:Flux.LayerNorm{
        typeof(identity),<:Flux.Scale{typeof(identity),<:DenseArray{P},<:DenseArray{P}},P
    },
    <:DenseArray{P},
} where {P<:IEEEFloat}

function frule!!(
    layer::Dual{
        <:Flux.LayerNorm{
            typeof(identity),
            <:Flux.Scale{typeof(identity),<:DenseArray{P},<:DenseArray{P}},
            P,
        },
    },
    x::Dual{<:DenseArray{P}},
) where {P<:IEEEFloat}
    p_layer = primal(layer)
    d_layer = _fields(tangent(layer))
    p_diag = p_layer.diag
    d_diag = _fields(d_layer.diag)
    px, dx = arrayify(x)
    dims = 1:length(p_layer.size)
    mean_x = NNlib.mean(px; dims)
    centered = px .- mean_x
    variance = NNlib.mean(abs2, centered; dims)
    inv_std = inv.(sqrt.(variance .+ p_layer.ϵ))
    normalized = centered .* inv_std
    dcentered = dx .- NNlib.mean(dx; dims)
    dvariance = 2 .* NNlib.mean(centered .* dcentered; dims)
    dnormalized =
        dcentered .* inv_std .-
        normalized .* (dvariance .+ d_layer.ϵ) ./ (2 .* (variance .+ p_layer.ϵ))
    y = p_diag.scale .* normalized .+ p_diag.bias
    dy = d_diag.scale .* normalized .+ p_diag.scale .* dnormalized .+ d_diag.bias
    return Dual(y, dy)
end

function rrule!!(
    layer::CoDual{
        <:Flux.LayerNorm{
            typeof(identity),
            <:Flux.Scale{typeof(identity),<:DenseArray{P},<:DenseArray{P}},
            P,
        },
    },
    x::CoDual{<:DenseArray{P}},
) where {P<:IEEEFloat}
    p_layer = primal(layer)
    d_layer = _fields(tangent(layer))
    p_diag = p_layer.diag
    d_diag = _fields(d_layer.diag)
    px, dx = arrayify(x)
    dims = 1:length(p_layer.size)
    mean_x = NNlib.mean(px; dims)
    centered = px .- mean_x
    variance = NNlib.mean(abs2, centered; dims)
    inv_std = inv.(sqrt.(variance .+ p_layer.ϵ))
    normalized = centered .* inv_std
    y = p_diag.scale .* normalized .+ p_diag.bias
    dy = zero(y)
    function layernorm_pullback!!(::NoRData)
        _accum_broadcast!(d_diag.scale, dy .* normalized)
        _accum_broadcast!(d_diag.bias, dy)
        dnormalized = dy .* p_diag.scale
        dx .+=
            inv_std .* (
                dnormalized .- NNlib.mean(dnormalized; dims) .-
                normalized .* NNlib.mean(dnormalized .* normalized; dims)
            )
        dϵ = sum(-P(0.5) .* dnormalized .* normalized ./ (variance .+ p_layer.ϵ))
        layer_rdata = Mooncake.zero_rdata(p_layer)
        layer_rdata = Mooncake.RData(merge(layer_rdata.data, (; ϵ=dϵ)))
        return layer_rdata, NoRData()
    end
    return CoDual(y, dy), layernorm_pullback!!
end

@is_primitive DefaultCtx Tuple{
    <:Flux.Dense{A,<:DenseArray{P,2},<:DenseArray{P,1}},<:DenseArray{P,N}
} where {P<:IEEEFloat,N,A<:FastFluxActivation}

function frule!!(
    layer::Dual{<:Flux.Dense{A,<:DenseArray{P,2},<:DenseArray{P,1}}},
    x::Dual{<:DenseArray{P,N}},
) where {P<:IEEEFloat,N,A<:FastFluxActivation}
    p_layer = primal(layer)
    d_layer = _fields(tangent(layer))
    px, dx = arrayify(x)
    px_matrix = reshape(px, size(px, 1), :)
    dx_matrix = reshape(dx, size(dx, 1), :)
    preactivation = p_layer.weight * px_matrix
    dy = d_layer.weight * px_matrix
    mul!(dy, p_layer.weight, dx_matrix, one(P), one(P))
    _activate_tangent!(p_layer.σ, dy, preactivation, p_layer.bias, d_layer.bias)
    y = _bias_activate!(p_layer.σ, preactivation, p_layer.bias)
    output_size = (size(y, 1), size(px)[2:end]...)
    return Dual(reshape(y, output_size), reshape(dy, output_size))
end

function rrule!!(
    layer::CoDual{<:Flux.Dense{A,<:DenseArray{P,2},<:DenseArray{P,1}}},
    x::CoDual{<:DenseArray{P,N}},
) where {P<:IEEEFloat,N,A<:FastFluxActivation}
    p_layer = primal(layer)
    d_layer = _fields(tangent(layer))
    px, dx = arrayify(x)
    px_matrix = reshape(px, size(px, 1), :)
    dx_matrix = reshape(dx, size(dx, 1), :)
    preactivation = p_layer.weight * px_matrix
    y = _bias_activate!(p_layer.σ, preactivation, p_layer.bias)
    dy = zero(y)
    function dense_pullback!!(::NoRData)
        _apply_activation_pullback!(p_layer.σ, dy, preactivation, p_layer.bias)
        d_layer.bias .+= vec(sum(dy; dims=2))
        mul!(d_layer.weight, dy, adjoint(px_matrix), one(P), one(P))
        mul!(dx_matrix, adjoint(p_layer.weight), dy, one(P), one(P))
        return NoRData(), NoRData()
    end
    output_size = (size(y, 1), size(px)[2:end]...)
    return CoDual(reshape(y, output_size), reshape(dy, output_size)), dense_pullback!!
end

function _accum_conv_bias!(db, dy, ::Val{N}) where {N}
    dims = ntuple(dim -> dim < N - 1 ? dim : dim + 1, Val(N - 1))
    db .+= vec(sum(dy; dims))
    return nothing
end

for (Layer, primal_op, input_grad_op, weight_grad_op, dims_fn) in (
    (:Conv, :conv, :∇conv_data!, :∇conv_filter!, :conv_dims),
    (:ConvTranspose, :∇conv_data, :conv!, :∇conv_filter!, :conv_transpose_dims),
)
    @eval begin
        @is_primitive DefaultCtx Tuple{
            <:Flux.$Layer{D,M,A,<:DenseArray{P,N},<:DenseArray{P,1}},<:DenseArray{P,N}
        } where {P<:IEEEFloat,D,M,N,A<:FastFluxActivation}

        function frule!!(
            layer::Dual{<:Flux.$Layer{D,M,A,<:DenseArray{P,N},<:DenseArray{P,1}}},
            x::Dual{<:DenseArray{P,N}},
        ) where {P<:IEEEFloat,D,M,N,A<:FastFluxActivation}
            p_layer = primal(layer)
            d_layer = _fields(tangent(layer))
            px, dx = arrayify(x)
            dims = Flux.$dims_fn(p_layer, px)
            preactivation = NNlib.$primal_op(px, p_layer.weight, dims)
            dy = NNlib.$primal_op(dx, p_layer.weight, dims)
            dy .+= NNlib.$primal_op(px, d_layer.weight, dims)
            bias = Flux.conv_reshape_bias(p_layer)
            dbias = Flux.conv_reshape_bias(d_layer.bias, p_layer.stride)
            _activate_tangent!(p_layer.σ, dy, preactivation, bias, dbias)
            y = _bias_activate!(p_layer.σ, preactivation, bias)
            return Dual(y, dy)
        end

        function rrule!!(
            layer::CoDual{<:Flux.$Layer{D,M,A,<:DenseArray{P,N},<:DenseArray{P,1}}},
            x::CoDual{<:DenseArray{P,N}},
        ) where {P<:IEEEFloat,D,M,N,A<:FastFluxActivation}
            p_layer = primal(layer)
            d_layer = _fields(tangent(layer))
            px, dx = arrayify(x)
            dims = Flux.$dims_fn(p_layer, px)
            preactivation = NNlib.$primal_op(px, p_layer.weight, dims)
            bias = Flux.conv_reshape_bias(p_layer)
            y = _bias_activate!(p_layer.σ, preactivation, bias)
            dy = zero(y)
            function conv_layer_pullback!!(::NoRData)
                _apply_activation_pullback!(p_layer.σ, dy, preactivation, bias)
                _accum_conv_bias!(d_layer.bias, dy, Val(N))
                updated_dx = NNlib.$input_grad_op(
                    dx, dy, p_layer.weight, dims; alpha=one(P), beta=one(P)
                )
                # cuDNN returns a depadded view for asymmetric padding.
                updated_dx === dx || copyto!(dx, updated_dx)
                NNlib.$weight_grad_op(
                    d_layer.weight,
                    $(Layer === :Conv ? :(px, dy) : :(dy, px))...,
                    dims;
                    alpha=one(P),
                    beta=one(P),
                )
                return NoRData(), NoRData()
            end
            return CoDual(y, dy), conv_layer_pullback!!
        end
    end
end

@is_primitive DefaultCtx Tuple{<:Flux.MeanPool,<:DenseArray{P,N}} where {P<:IEEEFloat,N}

function _meanpool_dims(layer::Flux.MeanPool, x)
    return NNlib.PoolDims(x, layer.k; padding=layer.pad, stride=layer.stride)
end

function frule!!(
    layer::Dual{<:Flux.MeanPool}, x::Dual{<:DenseArray{P,N}}
) where {P<:IEEEFloat,N}
    p_layer = primal(layer)
    px, dx = arrayify(x)
    dims = _meanpool_dims(p_layer, px)
    return Dual(NNlib.meanpool(px, dims), NNlib.meanpool(dx, dims))
end

function rrule!!(
    layer::CoDual{<:Flux.MeanPool}, x::CoDual{<:DenseArray{P,N}}
) where {P<:IEEEFloat,N}
    p_layer = primal(layer)
    px, dx = arrayify(x)
    dims = _meanpool_dims(p_layer, px)
    y = NNlib.meanpool(px, dims)
    dy = zero(y)
    function meanpool_pullback!!(::NoRData)
        NNlib.∇meanpool!(dx, dy, y, px, dims; beta=one(P))
        return NoRData(), NoRData()
    end
    return CoDual(y, dy), meanpool_pullback!!
end

@is_primitive DefaultCtx Tuple{
    typeof(Flux.Losses.mse),Array{P},Array{P}
} where {P<:IEEEFloat}

# This is a performance-specific rule motivated by https://github.com/chalk-lab/Mooncake.jl/issues/466
function rrule!!(
    ::CoDual{typeof(Flux.Losses.mse)}, X::CoDual{<:Array{P}}, Y::CoDual{<:Array{P}}
) where {P<:IEEEFloat}
    function flux_mse_pullback(dloss::P)
        scale = dloss * P(2) / length(X.x)
        @inbounds for index in eachindex(X.x, Y.x)
            delta = X.x[index] - Y.x[index]
            X.dx[index] += scale * delta
            Y.dx[index] -= scale * delta
        end
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(Flux.Losses.mse(X.x, Y.x)), flux_mse_pullback
end

# GPUArrays are `DenseArray`s, so this covers Flux's GPU backends without making the Flux
# extension depend on a particular GPU package. The `Array` method above remains the CPU
# reverse-mode fast path.
@is_primitive DefaultCtx Tuple{
    typeof(Flux.Losses.mse),DenseArray{P},DenseArray{P}
} where {P<:IEEEFloat}

function frule!!(
    ::Dual{typeof(Flux.Losses.mse)}, X::Dual{<:DenseArray{P}}, Y::Dual{<:DenseArray{P}}
) where {P<:IEEEFloat}
    pX, dX = primal(X), tangent(X)
    pY, dY = primal(Y), tangent(Y)
    dmse = (P(2) / length(pX)) * sum((pX .- pY) .* (dX .- dY))
    return Dual(Flux.Losses.mse(pX, pY), dmse)
end

function rrule!!(
    ::CoDual{typeof(Flux.Losses.mse)},
    X::CoDual{<:DenseArray{P}},
    Y::CoDual{<:DenseArray{P}},
) where {P<:IEEEFloat}
    pX, dX = primal(X), tangent(X)
    pY, dY = primal(Y), tangent(Y)
    function flux_dense_mse_pullback(dloss::P)
        scale = dloss * P(2) / length(pX)
        dX .+= scale .* (pX .- pY)
        dY .-= scale .* (pX .- pY)
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(Flux.Losses.mse(pX, pY)), flux_dense_mse_pullback
end

end
