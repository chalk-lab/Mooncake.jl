module MooncakeLuxLibExt

using LuxLib, Random, Mooncake
using Base: IEEEFloat

import LuxLib: Impl, Utils
import LuxLib.NNlib.GPUArraysCore: AbstractGPUArray
using MLDataDevices: get_device_type
import ChainRulesCore as CRC
using Mooncake:
    @from_rrule,
    DefaultCtx,
    MinimalCtx,
    @mooncake_overlay,
    CoDual,
    zero_tangent,
    primal,
    tangent,
    @is_primitive,
    NoRData,
    extract,
    zero_rdata,
    @zero_adjoint

using Static: True

import LuxLib.Utils: static_training_mode_check, True, False, unsafe_known
import LuxLib.Impl:
    get_non_heads_dim,
    make_causal_mask,
    get_non_contracting_dim,
    get_batched_matmul_repeat_dims,
    batchnorm_reduce_dims,
    get_batchnorm_statistics,
    groupnorm_reduce_dims,
    flattened_bias_dims,
    check_dropout_mask_shape_mismatch,
    dropout_shape,
    dropout_fptype,
    generate_alpha_dropout_noise,
    generate_dropout_mask,
    update_running_statistics,
    update_normalization_statistics,
    get_norm_reshape_dims,
    instancenorm_reduce_dims,
    compute_layernorm_dims

import LuxLib.Impl:
    activation!!,
    bias_activation,
    bias_activation!!,
    bias_activation_cached!!,
    bias_add!,
    fused_conv,
    conv,
    conv_bias_act,
    conv!,
    ∇conv_bias,
    groupnorm_affine_normalize_internal,
    groupnorm_affine_normalize_internal!,
    ∇groupnorm_affine_normalize,
    concrete_bias_act_output_eltype,
    ∇bias_add,
    activation_intermediate_not_needed,
    activation_has_rrule,
    activation,
    ∇activation,
    NotaNumber,
    AbstractInternalArrayOpMode,
    LoopedArrayOp

@from_rrule(
    DefaultCtx, Tuple{typeof(LuxLib.Impl.matmul),Array{P},Array{P}} where {P<:IEEEFloat}
)
@from_rrule(
    DefaultCtx,
    Tuple{typeof(LuxLib.Impl.matmuladd),Array{P},Array{P},Vector{P}} where {P<:IEEEFloat},
)
@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(LuxLib.Impl.batched_matmul_fallback),Array{P,3},Array{P,3}
    } where {P<:IEEEFloat},
)
@from_rrule(
    DefaultCtx,
    Tuple{
        typeof(Impl.batched_matmul_fallback),AbstractGPUArray{P,3},AbstractGPUArray{P,3}
    } where {P<:IEEEFloat},
)

## For mooncake we are missing some rules. For now use the basic versions of the kernels
@mooncake_overlay LuxLib.internal_operation_mode(xs::Tuple) = LuxLib.GenericBroadcastOp{
    get_device_type(xs)
}()

# Utils extensions
@mooncake_overlay Utils.within_autodiff(x) = True()

# Re-implement a bunch of methods to ensure that Mooncake can differentiate them.
@mooncake_overlay function LuxLib.Impl.fused_dense( 
    opmode,
    act::F,
    weight::AbstractMatrix,
    x::AbstractMatrix,
    b::LuxLib.Optional{<:AbstractVector},
) where {F}
    return LuxLib.Impl.bias_activation(act, LuxLib.Impl.matmul(weight, x), b)
end

@mooncake_overlay function LuxLib.Impl.fused_conv(
    ::LuxLib.Impl.AbstractInternalArrayOpMode,
    act::F,
    weight::AbstractArray{wT,N},
    x::AbstractArray{xT,N},
    bias::LuxLib.Optional{<:AbstractVector},
    cdims::LuxLib.Impl.ConvDims,
) where {F,wT,xT,N}
    return LuxLib.Impl.bias_activation(act, LuxLib.Impl.conv(x, weight, cdims), bias)
end

# zero gradient/non differentiable functions
@zero_adjoint DefaultCtx Tuple{typeof(static_training_mode_check),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(generate_dropout_mask),AbstractRNG,Any,Any,Any,Any}
@zero_adjoint DefaultCtx Tuple{typeof(get_non_heads_dim),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(make_causal_mask),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(get_non_contracting_dim),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(get_batched_matmul_repeat_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(batchnorm_reduce_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(get_batchnorm_statistics),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(groupnorm_reduce_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(flattened_bias_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(check_dropout_mask_shape_mismatch),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(dropout_shape),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(dropout_fptype),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(generate_alpha_dropout_noise),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(update_running_statistics),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(update_normalization_statistics),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(get_norm_reshape_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(instancenorm_reduce_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(compute_layernorm_dims),Vararg}

import LuxLib.Impl:
    safe_eltype,
    batchnorm_affine_normalize_internal,
    batchnorm_affine_normalize_internal!,
    ∇batchnorm_affine_normalize,
    AbstractInternalArrayOpMode

# Helper function for the Lux affine transform.
function _batchnorm_affine_normalize_identity(
    opmode::AbstractInternalArrayOpMode,
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::LuxLib.Optional{<:AbstractVector},
    β::LuxLib.Optional{<:AbstractVector},
    ϵ::Real,
) where {xT}
    PT_γ′ = promote_type(safe_eltype(γ), safe_eltype(σ²), safe_eltype(ϵ))
    γ′ = similar(x, PT_γ′, size(x, 2))
    PT = promote_type(
        safe_eltype(x), safe_eltype(μ), safe_eltype(σ²), safe_eltype(γ), safe_eltype(β)
    )
    y = similar(x, PT)
    batchnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ, γ′)
    return y
end

# Native Mooncake rule for differentiating through batchnorm_affine_normalize_internal.
@is_primitive MinimalCtx Tuple{
    typeof(_batchnorm_affine_normalize_identity),
    AbstractInternalArrayOpMode,
    AbstractArray{<:Any,3},
    AbstractVector,
    AbstractVector,
    LuxLib.Optional{<:AbstractVector},
    LuxLib.Optional{<:AbstractVector},
    Real,
}

function Mooncake.rrule!!(
    ::CoDual{typeof(_batchnorm_affine_normalize_identity)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
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

    PT_γ′ = promote_type(safe_eltype(_γ), safe_eltype(_σ²), safe_eltype(_ϵ))
    γ′ = similar(_x, PT_γ′, size(_x, 2))
    PT = promote_type(
        safe_eltype(_x), safe_eltype(_μ), safe_eltype(_σ²), safe_eltype(_γ), safe_eltype(_β)
    )
    y = similar(_x, PT)
    batchnorm_affine_normalize_internal!(y, _opmode, identity, _x, _μ, _σ², _γ, _β, _ϵ, γ′)
    ȳ = zero_tangent(y)

    function pb!!(::NoRData)
        # ∇batchnorm_affine_normalize returns CRC.NoTangent() for γ/β when they are nothing.
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇batchnorm_affine_normalize(
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

@mooncake_overlay function batchnorm_affine_normalize_internal(
    opmode::AbstractInternalArrayOpMode,
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

# Native Mooncake rrules for groupnorm_affine_normalize_internal

# Overlay for `batchnorm_affine_normalize_internal`
#  - Use Mooncake’s helper function `_batchnorm_affine_normalize_identity`  
#     and its manually written rule.
#  - Let Mooncake differentiate through the broadcasted `act` function.
function _groupnorm_affine_normalize_identity(
    opmode::AbstractInternalArrayOpMode,
    x::AbstractArray{T,4},
    μ::AbstractArray{<:Any,4},
    σ²::AbstractArray{<:Any,4},
    γ::LuxLib.Optional{<:AbstractArray{<:Any,4}},
    β::LuxLib.Optional{<:AbstractArray{<:Any,4}},
    ϵ::Real,
) where {T}
    y = similar(
        x,
        promote_type(
            safe_eltype(x),
            safe_eltype(μ),
            safe_eltype(σ²),
            safe_eltype(γ),
            safe_eltype(β),
        ),
    )
    groupnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ)
    return y
end

@is_primitive MinimalCtx Tuple{
    typeof(_groupnorm_affine_normalize_identity),
    AbstractInternalArrayOpMode,
    AbstractArray{<:Any,4},
    AbstractArray{<:Any,4},
    AbstractArray{<:Any,4},
    LuxLib.Optional{<:AbstractArray{<:Any,4}},
    LuxLib.Optional{<:AbstractArray{<:Any,4}},
    Real,
}

function Mooncake.rrule!!(
    ::CoDual{typeof(_groupnorm_affine_normalize_identity)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    x::CoDual{<:AbstractArray{T,4}},
    μ::CoDual{<:AbstractArray{<:Any,4}},
    σ²::CoDual{<:AbstractArray{<:Any,4}},
    γ::CoDual{<:LuxLib.Optional{<:AbstractArray{<:Any,4}}},
    β::CoDual{<:LuxLib.Optional{<:AbstractArray{<:Any,4}}},
    ϵ::CoDual{<:Real},
) where {T}
    _opmode, _ϵ = primal(opmode), primal(ϵ)
    _x, x̄ = primal(x), tangent(x)
    _μ, μ̄ = primal(μ), tangent(μ)
    _σ², σ²̄ = primal(σ²), tangent(σ²)
    _γ, γ̄ = primal(γ), tangent(γ)
    _β, β̄ = primal(β), tangent(β)

    y = similar(
        _x,
        promote_type(
            safe_eltype(_x),
            safe_eltype(_μ),
            safe_eltype(_σ²),
            safe_eltype(_γ),
            safe_eltype(_β),
        ),
    )
    groupnorm_affine_normalize_internal!(y, _opmode, identity, _x, _μ, _σ², _γ, _β, _ϵ)
    ȳ = zero_tangent(y)

    function pb!!(::NoRData)
        # ∇groupnorm_affine_normalize returns CRC.NoTangent() for γ/β when they are nothing.
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇groupnorm_affine_normalize(
            _opmode, ȳ, _x, _μ, _σ², _γ, _β, _ϵ
        )
        x̄ .+= ∂x
        μ̄ .+= ∂μ
        σ²̄ .+= ∂σ²
        ∂γ isa CRC.NoTangent || (γ̄ .+= ∂γ)
        ∂β isa CRC.NoTangent || (β̄ .+= ∂β)
        # ϵ is a Real scalar — must return zero_rdata so Mooncake can accumulate it
        # correctly when chained through the overlay.
        return NoRData(),
        NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), zero_rdata(_ϵ)
    end

    return CoDual(y, ȳ), pb!!
end

@mooncake_overlay function groupnorm_affine_normalize_internal(
    opmode::AbstractInternalArrayOpMode,
    f::F,
    x::AbstractArray{T,4},
    μ::AbstractArray{<:Any,4},
    σ²::AbstractArray{<:Any,4},
    γ::LuxLib.Optional{<:AbstractArray{<:Any,4}},
    β::LuxLib.Optional{<:AbstractArray{<:Any,4}},
    ϵ::Real,
) where {F,T}
    y = _groupnorm_affine_normalize_identity(opmode, x, μ, σ², γ, β, ϵ)
    return f.(y)
end

end
