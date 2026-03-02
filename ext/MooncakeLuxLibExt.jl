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

# Native Mooncake rule for batchnorm_affine_normalize_internal.
# Upstream CRC.rrule uses rrule_via_ad (callback into AD) which Mooncake
# cannot handle, so we provide a direct rrule!! with manual activation pullback.
import LuxLib.Impl:
    safe_eltype,
    batchnorm_affine_normalize_internal,
    batchnorm_affine_normalize_internal!,
    ∇batchnorm_affine_normalize,
    AbstractInternalArrayOpMode

import ChainRulesCore as CRC

# MinimalCtx: upstream CRC.rrule uses rrule_via_ad which Mooncake cannot trace
@is_primitive MinimalCtx Tuple{
    typeof(batchnorm_affine_normalize_internal),
    AbstractInternalArrayOpMode,
    F,
    AbstractArray{xT,3},
    AbstractVector,
    AbstractVector,
    LuxLib.Optional{<:AbstractVector},
    LuxLib.Optional{<:AbstractVector},
    Real,
} where {F,xT}

function Mooncake.rrule!!(
    ::CoDual{typeof(batchnorm_affine_normalize_internal)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    act::CoDual{F},
    x::CoDual{<:AbstractArray{xT,3}},
    μ::CoDual{<:AbstractVector},
    σ²::CoDual{<:AbstractVector},
    γ::CoDual{<:LuxLib.Optional{<:AbstractVector}},
    β::CoDual{<:LuxLib.Optional{<:AbstractVector}},
    ϵ::CoDual{<:Real},
) where {F,xT}
    _opmode, _act, _ϵ = primal(opmode), primal(act), primal(ϵ)

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
    y_pre = similar(_x, PT)

    # forward pass using only the affine transform.
    batchnorm_affine_normalize_internal!(
        y_pre, _opmode, identity, _x, _μ, _σ², _γ, _β, _ϵ, γ′
    )

    act_cache = prepare_pullback_cache(broadcast, _act, y_pre)
    # Forward pass using the activation function.
    y_out = _act.(y_pre)
    ȳ = zero_tangent(y_out)

    function pb!!(::NoRData)
        # run pullback for activation func applied on affine transform output
        # using the gradients accumulated in affine transform output's tangent.
        _, (_, _, ∂y_pre) = value_and_pullback!!(act_cache, ȳ, broadcast, _act, y_pre)

        # use activation func gradients to run pullback for affine transform func.
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇batchnorm_affine_normalize(
            _opmode, ∂y_pre, _x, _μ, _σ², _γ, _β, _ϵ, γ′
        )

        # accumulate gradients
        x̄ .+= ∂x
        μ̄ .+= ∂μ
        σ²̄ .+= ∂σ²
        # γ, β may have NoTangent gradients for primal=nothing
        isnothing(primal(γ)) || (γ̄ .+= ∂γ)
        isnothing(primal(β)) || (β̄ .+= ∂β)

        return NoRData(),
        NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(),
        zero_rdata(primal(ϵ))
    end

    return CoDual(y_out, ȳ), pb!!
end

end
