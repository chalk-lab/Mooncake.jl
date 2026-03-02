module MooncakeLuxLibExt

using LuxLib, Random, Mooncake
using Base: IEEEFloat

import LuxLib: Impl, Utils
import LuxLib.Utils: static_training_mode_check
using MLDataDevices: get_device_type
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
    prepare_pullback_cache,
    value_and_pullback!!,
    zero_rdata

using Static: True

@from_rrule(DefaultCtx, Tuple{typeof(Impl.matmul),Array{P},Array{P}} where {P<:IEEEFloat})
@from_rrule(
    DefaultCtx,
    Tuple{typeof(Impl.matmuladd),Array{P},Array{P},Vector{P}} where {P<:IEEEFloat},
)
@from_rrule(
    DefaultCtx,
    Tuple{typeof(Impl.batched_matmul_fallback),Array{P,3},Array{P,3}} where {P<:IEEEFloat},
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
    return bias_activation(act, Impl.matmul(weight, x), b)
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

Mooncake.@zero_adjoint DefaultCtx Tuple{typeof(static_training_mode_check),Vararg}
Mooncake.@zero_adjoint DefaultCtx Tuple{
    typeof(LuxLib.Impl.generate_dropout_mask),AbstractRNG,Any,Any,Any,Any
}

import LuxLib.Impl:
    safe_eltype,
    batchnorm_affine_normalize_internal,
    batchnorm_affine_normalize_internal!,
    ∇batchnorm_affine_normalize,
    AbstractInternalArrayOpMode

import ChainRulesCore as CRC

# AD helper mapping function for the Lux affine transform.
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

# Native Mooncake rule for batchnorm_affine_normalize_internal.
# `MinimalCtx` to avoid upstream CRC.rrule which uses `rrule_via_ad` which is avoided by Mooncake for performance.
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

# Overlay for `batchnorm_affine_normalize_internal` to use Mooncake AD's mapping function -> `_batchnorm_affine_normalize_identity`
# the chosen `act` function's gradient is derived by Mooncake normally.
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

end
