module MooncakeLuxLibExt

using LuxLib, Random, Mooncake
using NNlib: gelu
using Base: IEEEFloat
using Static: True
import ChainRulesCore as CRC
import LuxLib: Impl, Utils
import LuxLib.Utils: static_training_mode_check
using MLDataDevices: get_device_type
import Mooncake:
    @from_rrule,
    DefaultCtx,
    @mooncake_overlay,
    CoDual,
    @zero_adjoint,
    @zero_derivative,
    MooncakeRuleConfig

## Mooncake imports CRC rules for Lux kernels from Lux.LuxLib
# activation
@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.activation!!),Impl.AbstractInternalArrayOpMode,True,F,AbstractArray{T}
    } where {F,T},
    false,
    MooncakeRuleConfig()
)

@from_rrule(
    DefaultCtx,
    Tuple{typeof(Impl.activation),Impl.LoopedArrayOp,F,AbstractArray{T}} where {F,T},
    false,
    MooncakeRuleConfig()
)

@zero_derivative DefaultCtx Tuple{typeof(Impl.select_fastest_activation),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.sleefpirates_fast_act),Vararg}
@zero_derivative DefaultCtx Tuple{
    typeof(Impl.internal_operation_mode),Union{Tuple,AbstractArray}
}

# attention
@zero_derivative DefaultCtx Tuple{typeof(Impl.get_non_heads_dim),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.make_causal_mask),Vararg}

# batched mul
@zero_derivative DefaultCtx Tuple{typeof(Impl.get_non_contracting_dim),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.get_batched_matmul_repeat_dims),Vararg}

@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.batched_matmul_fallback),AbstractArray{xT,3},AbstractArray{yT,3}
    } where {xT,yT}
)

# batchnorm
@zero_derivative DefaultCtx Tuple{typeof(Impl.batchnorm_reduce_dims),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.get_batchnorm_statistics),Vararg}

# bias_activation
@zero_derivative DefaultCtx Tuple{typeof(Impl.flattened_bias_dims),Vararg}

@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.bias_activation),
        Impl.AbstractInternalArrayOpMode,
        F,
        AbstractArray{xT,N},
        AbstractVector,
    } where {F,N,xT},
    false,
    MooncakeRuleConfig()
)

@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.bias_activation!!),
        Impl.AbstractInternalArrayOpMode,
        True,
        F,
        AbstractArray{xT,N},
        AbstractVector,
    } where {F,N,xT},
    false,
    MooncakeRuleConfig()
)

# common_ops
@from_rrule(
    DefaultCtx,
    Tuple{typeof(Impl.reshape_bias),AbstractArray{xT,N},AbstractVector{bT}} where {xT,bT,N}
)

@from_rrule(DefaultCtx, Tuple{typeof(Impl.mean_var),AbstractArray}, true)

# FIX CONV, DENSE Kernel rules for Mooncake
# conv
@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.fused_conv),
        Impl.AbstractInternalArrayOpMode,
        F,
        AbstractArray{wT,N},
        AbstractArray{xT,N},
        Impl.Optional{AbstractVector},
        Impl.ConvDims,
    } where {F,wT,xT,N},
    false,
    MooncakeRuleConfig()
)

# dense
@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.fused_dense),
        Impl.AbstractInternalArrayOpMode,
        F,
        AbstractMatrix,
        AbstractMatrix,
        Impl.Optional{<:AbstractVector},
    } where {F},
    false,
    MooncakeRuleConfig()
)

# special CRC rule dispatch over gelu with GPUBroadcastOp
@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.fused_dense),
        Impl.GPUBroadcastOp{Impl.CUDADevice},
        typeof(gelu),
        AbstractMatrix,
        AbstractMatrix,
        Impl.Optional{<:AbstractVector},
    }
)

# dropout
@zero_derivative DefaultCtx Tuple{typeof(Impl.check_dropout_mask_shape_mismatch),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.dropout_shape),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.dropout_fptype),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.generate_alpha_dropout_noise),Vararg}
@zero_derivative DefaultCtx Tuple{
    typeof(Impl.generate_dropout_mask),AbstractRNG,Any,Any,Any,Any
}

@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.alpha_dropout),
        Impl.LoopedArrayOp,
        AbstractArray,
        Any,  # p
        AbstractArray,
        Any,  # α
        Any,  # A
        Any,   # B
    }
)

@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.alpha_dropout),
        Impl.AbstractInternalArrayOpMode,
        AbstractArray,
        Any,  # p
        AbstractArray,
        Any,  # α
        Any,  # A
        Any,   # B
    }
)

@from_rrule(
    DefaultCtx, Tuple{typeof(LuxLib.Impl.dropout_dot_mul),AbstractArray,AbstractArray}
)

# groupnorm
@zero_derivative DefaultCtx Tuple{typeof(Impl.groupnorm_reduce_dims),Vararg}

# Let Mooncake derive rules for Impl.groupnorm_affine_normalize_internal 

# layernorm
@zero_derivative DefaultCtx Tuple{typeof(Impl.compute_layernorm_dims),Vararg}

# matmul
@from_rrule(
    DefaultCtx, Tuple{typeof(Impl.matmuladd),AbstractMatrix,AbstractMatrix,AbstractVector}
)

@from_rrule(DefaultCtx, Tuple{typeof(Impl.matmul),AbstractMatrix,AbstractMatrix})

# normalization
@zero_derivative DefaultCtx Tuple{typeof(Impl.update_running_statistics),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.update_normalization_statistics),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.get_norm_reshape_dims),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Impl.instancenorm_reduce_dims),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(static_training_mode_check),Vararg}

# This is a really horrible hack that we need to do until Mooncake is able to support the
# call-back-into-ad interface that ChainRules exposes.

import LuxLib.Impl:
    safe_eltype,
    batchnorm_affine_normalize_internal,
    batchnorm_affine_normalize_internal!,
    ∇batchnorm_affine_normalize,
    AbstractInternalArrayOpMode

function CRC.rrule(
    ::typeof(batchnorm_affine_normalize_internal),
    opmode::AbstractInternalArrayOpMode,
    ::typeof(identity),
    x::AbstractArray{T,N},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::LuxLib.Optional{<:AbstractVector},
    β::LuxLib.Optional{<:AbstractVector},
    ϵ::Real,
) where {T,N}
    y = similar(
        x,
        promote_type(
            safe_eltype(x), safe_eltype(μ), safe_eltype(σ²), safe_eltype(γ), safe_eltype(β)
        ),
    )
    γ′ = similar(
        x, promote_type(safe_eltype(γ), safe_eltype(σ²), safe_eltype(ϵ)), size(x, N - 1)
    )

    batchnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ, γ′)

    𝒫x, 𝒫μ, 𝒫σ² = CRC.ProjectTo(x), CRC.ProjectTo(μ), CRC.ProjectTo(σ²)
    𝒫γ = γ === nothing ? identity : CRC.ProjectTo(γ)
    𝒫β = β === nothing ? identity : CRC.ProjectTo(β)

    ∇batchnorm_affine_normalize_internal = LuxLib.Impl.@closure Δ -> begin
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇batchnorm_affine_normalize(opmode, Δ, x, μ, σ², γ, β, ϵ, γ′)
        ∂∅ = CRC.NoTangent()
        return ∂∅, ∂∅, ∂∅, 𝒫x(∂x), 𝒫μ(∂μ), 𝒫σ²(∂σ²), 𝒫γ(∂γ), 𝒫β(∂β), ∂∅
    end

    return y, ∇batchnorm_affine_normalize_internal
end

@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(batchnorm_affine_normalize_internal),
        AbstractInternalArrayOpMode,
        typeof(identity),
        AbstractArray,
        AbstractVector,
        AbstractVector,
        LuxLib.Optional{<:AbstractVector},
        LuxLib.Optional{<:AbstractVector},
        Real,
    },
)

@mooncake_overlay function batchnorm_affine_normalize_internal(
    opmode::LuxLib.AbstractInternalArrayOpMode,
    act::F,
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::Union{Nothing,AbstractVector},
    β::Union{Nothing,AbstractVector},
    ϵ::Real,
) where {F,xT}
    y = batchnorm_affine_normalize_internal(opmode, identity, x, μ, σ², γ, β, ϵ)
    LuxLib.Impl.activation!(y, opmode, act, y)
    return y
end

@mooncake_overlay function batchnorm_affine_normalize_internal(
    opmode::LuxLib.AbstractInternalArrayOpMode,
    ::typeof(identity),
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::Union{Nothing,AbstractVector},
    β::Union{Nothing,AbstractVector},
    ϵ::Real,
) where {xT}
    y = similar(
        x,
        promote_type(
            safe_eltype(x), safe_eltype(μ), safe_eltype(σ²), safe_eltype(γ), safe_eltype(β)
        ),
    )
    batchnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ)
    return y
end

end
