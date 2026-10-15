module MooncakeLuxLibExt

using LuxLib, Random, Mooncake
using Base: IEEEFloat

import LuxLib: Impl, Utils
import LuxLib.NNlib.GPUArraysCore: AbstractGPUArray
using MLDataDevices: get_device_type
using Static: StaticBool
using Mooncake:
    @from_rrule,
    DefaultCtx,
    MinimalCtx,
    @mooncake_overlay,
    CoDual,
    zero_tangent,
    primal,
    @is_primitive,
    NoRData,
    extract,
    zero_rdata,
    @zero_adjoint,
    ReverseMode

@from_rrule(DefaultCtx, Tuple{typeof(Impl.matmul),Array{P},Array{P}} where {P<:IEEEFloat})
@from_rrule(
    DefaultCtx,
    Tuple{typeof(Impl.matmuladd),Array{P},Array{P},Vector{P}} where {P<:IEEEFloat},
)
@from_rrule(
    DefaultCtx,
    Tuple{typeof(Impl.batched_matmul_fallback),Array{P,3},Array{P,3}} where {P<:IEEEFloat},
)
@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.batched_matmul_fallback),AbstractGPUArray{P,3},AbstractGPUArray{P,3}
    } where {P<:IEEEFloat},
)
@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.batchnorm_cudnn),
        Union{Nothing,AbstractGPUArray{P,1}},
        Union{Nothing,AbstractGPUArray{P,1}},
        AbstractGPUArray{P,N},
        Union{Nothing,AbstractGPUArray{P,1}},
        Union{Nothing,AbstractGPUArray{P,1}},
        P,
        P,
        StaticBool,
    } where {P<:IEEEFloat,N},
)

## For mooncake we are missing some rules. For now use the basic versions of the kernels
@mooncake_overlay LuxLib.internal_operation_mode(xs::Tuple) = LuxLib.GenericBroadcastOp{
    get_device_type(xs)
}()

# Utils extensions
@mooncake_overlay Utils.within_autodiff(x) = Utils.True()

# zero gradient/non differentiable functions
@zero_adjoint DefaultCtx Tuple{typeof(Utils.static_training_mode_check),Vararg}
@zero_adjoint DefaultCtx Tuple{
    typeof(Impl.generate_dropout_mask),AbstractRNG,Any,Any,Any,Any
}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.get_non_heads_dim),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.make_causal_mask),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.get_non_contracting_dim),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.get_batched_matmul_repeat_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.batchnorm_reduce_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.get_batchnorm_statistics),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.groupnorm_reduce_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.flattened_bias_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.check_dropout_mask_shape_mismatch),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.dropout_shape),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.dropout_fptype),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.generate_alpha_dropout_noise),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.update_running_statistics),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.update_normalization_statistics),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.get_norm_reshape_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.instancenorm_reduce_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(Impl.compute_layernorm_dims),Vararg}

# Re-implement LuxLib.Impl.batchnorm_affine_normalize_internal and LuxLib.Impl.fused_conv to ensure that Mooncake can differentiate them.
@mooncake_overlay function LuxLib.Impl.fused_conv(
    ::LuxLib.Impl.AbstractInternalArrayOpMode,
    act::F,
    weight::AbstractArray{wT,N},
    x::AbstractArray{xT,N},
    bias::LuxLib.Optional{<:AbstractVector},
    cdims::Impl.ConvDims,
) where {F,wT,xT,N}
    return LuxLib.Impl.bias_activation(act, Impl.conv(x, weight, cdims), bias)
end

# `mean_var(x; dims, corrected)` on CPU arrays, used for batchnorm/layernorm statistics.
# Mooncake would otherwise differentiate `Statistics.varm` element by element. Same gradient
# as LuxLib's own `rrule`: dx = dμ / n + 2 / (n - corrected) * dσ² * (x - μ).
const _MeanVarKw = NamedTuple{
    (:dims, :corrected),<:Tuple{Union{Int,Tuple{Vararg{Int}}},Bool}
}
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(Core.kwcall),_MeanVarKw,typeof(Impl.mean_var),Array{P}
} where {P<:IEEEFloat}
function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:_MeanVarKw},
    ::CoDual{typeof(Impl.mean_var)},
    x::CoDual{<:Array{P}},
) where {P<:IEEEFloat}
    (; dims, corrected) = primal(kw)
    px, dx = extract(x)
    μ, σ² = Impl.mean_var(px; dims, corrected)
    dμ, dσ² = zero(μ), zero(σ²)
    function mean_var_pb!!(::NoRData)
        n = prod(d -> size(px, d), dims)
        c = P(2 / (n - corrected))
        dx .+= dμ ./ P(n) .+ c .* dσ² .* (px .- μ)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual((μ, σ²), (dμ, dσ²)), mean_var_pb!!
end

# Under the generic broadcast mode, batchnorm normalises with one broadcast through `sqrt`,
# which Mooncake differentiates element by element. On CPU arrays take LuxLib's looped path
# instead, which reaches the native rule for `_batchnorm_affine_normalize_identity` below.
@mooncake_overlay function LuxLib.Impl.batchnorm_affine_normalize(
    ::LuxLib.GenericBroadcastOp,
    act::F,
    x::Array{xT,N},
    μ::Array{μT,N},
    σ²::Array{σ²T,N},
    γ::LuxLib.Optional{<:Vector},
    β::LuxLib.Optional{<:Vector},
    ϵ,
) where {F,xT,μT,σ²T,N}
    return LuxLib.Impl.batchnorm_affine_normalize(
        LuxLib.LoopedArrayOp(), act, x, μ, σ², γ, β, ϵ
    )
end

# Helper function for the Lux affine transform.
function _batchnorm_affine_normalize_identity(
    opmode::Impl.AbstractInternalArrayOpMode,
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::LuxLib.Optional{<:AbstractVector},
    β::LuxLib.Optional{<:AbstractVector},
    ϵ::Real,
) where {xT}
    PT_γ′ = promote_type(Impl.safe_eltype(γ), Impl.safe_eltype(σ²), Impl.safe_eltype(ϵ))
    γ′ = similar(x, PT_γ′, size(x, 2))
    PT = promote_type(
        Impl.safe_eltype(x),
        Impl.safe_eltype(μ),
        Impl.safe_eltype(σ²),
        Impl.safe_eltype(γ),
        Impl.safe_eltype(β),
    )
    y = similar(x, PT)
    Impl.batchnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ, γ′)
    return y
end

# Native Mooncake rule for differentiating through batchnorm_affine_normalize_internal.
@is_primitive MinimalCtx Tuple{
    typeof(_batchnorm_affine_normalize_identity),
    Impl.AbstractInternalArrayOpMode,
    AbstractArray{<:Any,3},
    AbstractVector,
    AbstractVector,
    LuxLib.Optional{<:AbstractVector},
    LuxLib.Optional{<:AbstractVector},
    Real,
}

function Mooncake.rrule!!(
    ::CoDual{typeof(_batchnorm_affine_normalize_identity)},
    opmode::CoDual{<:Impl.AbstractInternalArrayOpMode},
    x::CoDual{<:AbstractArray{xT,3}},
    μ::CoDual{<:AbstractVector},
    σ²::CoDual{<:AbstractVector},
    γ::CoDual{<:LuxLib.Optional{<:AbstractVector}},
    β::CoDual{<:LuxLib.Optional{<:AbstractVector}},
    ϵ::CoDual{<:Real},
) where {xT}
    _opmode, _ϵ = primal(opmode), primal(ϵ)
    _x, x̄ = extract(x)
    _μ, μ̄ = extract(μ)
    _σ², σ²̄ = extract(σ²)
    _γ, γ̄ = extract(γ)
    _β, β̄ = extract(β)

    PT_γ′ = promote_type(Impl.safe_eltype(_γ), Impl.safe_eltype(_σ²), Impl.safe_eltype(_ϵ))
    γ′ = similar(_x, PT_γ′, size(_x, 2))
    PT = promote_type(
        Impl.safe_eltype(_x),
        Impl.safe_eltype(_μ),
        Impl.safe_eltype(_σ²),
        Impl.safe_eltype(_γ),
        Impl.safe_eltype(_β),
    )
    y = similar(_x, PT)
    Impl.batchnorm_affine_normalize_internal!(
        y, _opmode, identity, _x, _μ, _σ², _γ, _β, _ϵ, γ′
    )
    ȳ = zero_tangent(y)

    function pb!!(::NoRData)
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = Impl.∇batchnorm_affine_normalize(
            _opmode, ȳ, _x, _μ, _σ², _γ, _β, _ϵ, γ′
        )

        x̄ .+= ∂x
        μ̄ .+= ∂μ
        σ²̄ .+= ∂σ²
        isnothing(primal(γ)) || (γ̄ .+= ∂γ)
        isnothing(primal(β)) || (β̄ .+= ∂β)

        return NoRData(),
        NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(),
        zero_rdata(_ϵ)
    end

    return CoDual(y, ȳ), pb!!
end

# Native rule for LuxLib's fused bias + activation under the generic broadcast mode that the
# overlay on `internal_operation_mode` forces. Without it, Mooncake differentiates
# `σ.(x .+ b)` element by element on CPU; LuxLib's own ChainRules rule needs a `RuleConfig`,
# which `@from_rrule` cannot supply. Restricted to activations whose derivative LuxLib can
# compute from the output alone (`only_derivative(y, σ, NotaNumber())`).
const _OutputDerivActivation = Union{
    typeof(identity),
    typeof(LuxLib.NNlib.relu),
    typeof(tanh),
    typeof(LuxLib.NNlib.sigmoid),
    typeof(LuxLib.NNlib.tanh_fast),
    typeof(LuxLib.NNlib.sigmoid_fast),
}

@is_primitive MinimalCtx Tuple{
    typeof(Impl.bias_activation),
    LuxLib.GenericBroadcastOp,
    _OutputDerivActivation,
    Array{P,N},
    Vector{P},
} where {P<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(Impl.bias_activation)},
    opmode::CoDual{<:LuxLib.GenericBroadcastOp},
    σ::CoDual{<:_OutputDerivActivation},
    x::CoDual{Array{P,N}},
    b::CoDual{Vector{P}},
) where {P<:IEEEFloat,N}
    _σ = primal(σ)
    _x, x̄ = extract(x)
    _b, b̄ = extract(b)
    y = Impl.bias_activation(primal(opmode), _σ, _x, _b)
    ȳ = zero_tangent(y)

    function bias_activation_pb!!(::NoRData)
        ∂x = Impl.∇activation(ȳ, y, _σ, Utils.NotaNumber())
        x̄ .+= ∂x
        b̄ .+= Impl.∇bias_add(_b, ∂x)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, ȳ), bias_activation_pb!!
end

@mooncake_overlay function LuxLib.Impl.batchnorm_affine_normalize_internal(
    opmode::Impl.AbstractInternalArrayOpMode,
    act::F,
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::LuxLib.Optional{<:AbstractVector},
    β::LuxLib.Optional{<:AbstractVector},
    ϵ::Real,
) where {F,xT}
    y = _batchnorm_affine_normalize_identity(opmode, x, μ, σ², γ, β, ϵ)
    return act.(y)
end

end
