module MooncakeLuxLibExt

using LuxLib, Random, Mooncake
using Base: IEEEFloat

import LuxLib: Impl, Utils
import LuxLib.Utils: static_training_mode_check
using MLDataDevices: get_device_type
import Mooncake: @from_rrule, DefaultCtx, @mooncake_overlay, CoDual
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
@mooncake_overlay LuxLib.internal_operation_mode(xs::Tuple) =
    LuxLib.GenericBroadcastOp{get_device_type(xs)}()

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

# This is a really horrible hack that we need to do until Mooncake is able to support the
# call-back-into-ad interface that ChainRules exposes.

import LuxLib.Impl:
    safe_eltype,
    batchnorm_affine_normalize_internal,
    batchnorm_affine_normalize_internal!,
    ∇batchnorm_affine_normalize,
    AbstractInternalArrayOpMode

import ChainRulesCore as CRC

# function CRC.rrule(
#     ::typeof(batchnorm_affine_normalize_internal),
#     opmode::AbstractInternalArrayOpMode,
#     ::typeof(identity),
#     x::AbstractArray{T,N},
#     μ::AbstractVector,
#     σ²::AbstractVector,
#     γ::LuxLib.Optional{<:AbstractVector},
#     β::LuxLib.Optional{<:AbstractVector},
#     ϵ::Real,
# ) where {T,N}
#     y = similar(
#         x,
#         promote_type(
#             safe_eltype(x), safe_eltype(μ), safe_eltype(σ²), safe_eltype(γ), safe_eltype(β)
#         ),
#     )
#     γ′ = similar(
#         x, promote_type(safe_eltype(γ), safe_eltype(σ²), safe_eltype(ϵ)), size(x, N - 1)
#     )

#     batchnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ, γ′)

#     𝒫x, 𝒫μ, 𝒫σ² = CRC.ProjectTo(x), CRC.ProjectTo(μ), CRC.ProjectTo(σ²)
#     𝒫γ = γ === nothing ? identity : CRC.ProjectTo(γ)
#     𝒫β = β === nothing ? identity : CRC.ProjectTo(β)

#     ∇batchnorm_affine_normalize_internal = LuxLib.Impl.@closure Δ -> begin
#         ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇batchnorm_affine_normalize(opmode, Δ, x, μ, σ², γ, β, ϵ, γ′)
#         ∂∅ = CRC.NoTangent()
#         return ∂∅, ∂∅, ∂∅, 𝒫x(∂x), 𝒫μ(∂μ), 𝒫σ²(∂σ²), 𝒫γ(∂γ), 𝒫β(∂β), ∂∅
#     end

#     return y, ∇batchnorm_affine_normalize_internal
# end

# @from_rrule(
#     DefaultCtx,
#     Tuple{
#         typeof(batchnorm_affine_normalize_internal),
#         AbstractInternalArrayOpMode,
#         typeof(identity),
#         AbstractArray,
#         AbstractVector,
#         AbstractVector,
#         LuxLib.Optional{<:AbstractVector},
#         LuxLib.Optional{<:AbstractVector},
#         Real,
#     },
# )

# @mooncake_overlay function batchnorm_affine_normalize_internal(
#     opmode::LuxLib.AbstractInternalArrayOpMode,
#     act::F,
#     x::AbstractArray{xT,3},
#     μ::AbstractVector,
#     σ²::AbstractVector,
#     γ::Union{Nothing,AbstractVector},
#     β::Union{Nothing,AbstractVector},
#     ϵ::Real,
# ) where {F,xT}
#     y = batchnorm_affine_normalize_internal(opmode, identity, x, μ, σ², γ, β, ϵ)
#     LuxLib.Impl.activation!(y, opmode, act, y)
#     return y
# end

# @mooncake_overlay function batchnorm_affine_normalize_internal(
#     opmode::LuxLib.AbstractInternalArrayOpMode,
#     ::typeof(identity),
#     x::AbstractArray{xT,3},
#     μ::AbstractVector,
#     σ²::AbstractVector,
#     γ::Union{Nothing,AbstractVector},
#     β::Union{Nothing,AbstractVector},
#     ϵ::Real,
# ) where {xT}
#     y = similar(
#         x,
#         promote_type(
#             safe_eltype(x), safe_eltype(μ), safe_eltype(σ²), safe_eltype(γ), safe_eltype(β)
#         ),
#     )
#     batchnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ)
#     return y
# end

Mooncake.@is_primitive(
    MinimalCtx,
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

function Mooncake.rrule!!(
    ::CoDual{typeof(batchnorm_affine_normalize_internal)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    ::CoDual{typeof(identity)},
    x::CoDual{<:AbstractArray{T,N}},
    μ::CoDual{<:AbstractVector},
    σ²::CoDual{<:AbstractVector},
    γ::CoDual{<:LuxLib.Optional{<:AbstractVector}},
    β::CoDual{<:LuxLib.Optional{<:AbstractVector}},
    ϵ::CoDual{<:Real},
) where {T,N}
    _opmode = primal(opmode)
    _x, _μ, _σ², _γ, _β, _ϵ = primal(x),
    primal(μ), primal(σ²), primal(γ), primal(β),
    primal(ϵ)

    _y = similar(
        _x,
        promote_type(
            safe_eltype(_x),
            safe_eltype(_μ),
            safe_eltype(_σ²),
            safe_eltype(_γ),
            safe_eltype(_β),
        ),
    )
    _γ′ = similar(
        _x,
        promote_type(safe_eltype(_γ), safe_eltype(_σ²), safe_eltype(_ϵ)),
        size(_x, N - 1),
    )

    batchnorm_affine_normalize_internal!(
        _y, _opmode, identity, _x, _μ, _σ², _γ, _β, _ϵ, _γ′
    )

    _dy = Mooncake.zero_tangent(_y)

    pb!! = @closure Δy -> begin
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇batchnorm_affine_normalize(
            _opmode, Δy, _x, _μ, _σ², _γ, _β, _ϵ, _γ′
        )

        x.dx .+= ∂x
        μ.dx .+= ∂μ
        σ².dx .+= ∂σ²
        _γ !== nothing && (γ.dx .+= ∂γ)
        _β !== nothing && (β.dx .+= ∂β)

        ∂ϵ = tangent(ϵ)

        return NoRData(),
        NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(),
        ∂ϵ
    end

    return CoDual(_y, _dy), pb!!
end

end
