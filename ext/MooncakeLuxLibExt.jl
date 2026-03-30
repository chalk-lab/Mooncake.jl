module MooncakeLuxLibExt

using LuxLib, Random, Mooncake, NNlib
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

# zero gradient/non differentiable functions
import LuxLib.Utils: static_training_mode_check
import LuxLib.Impl:
    select_fastest_activation,
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

@zero_adjoint DefaultCtx Tuple{typeof(static_training_mode_check),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(generate_dropout_mask),AbstractRNG,Any,Any,Any,Any}
@zero_adjoint DefaultCtx Tuple{typeof(select_fastest_activation),Vararg}
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

# Native Mooncake rrules for activation, bias_activation, and fused_conv.

import LuxLib.Utils: True, False, unsafe_known
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

# Fast helper for activation: called from the overlay for activations where LuxLib has
# an optimised ∇activation kernel (activation_intermediate_not_needed or activation_has_rrule).
function _activation_fast(
    opmode::AbstractInternalArrayOpMode, σ::F, x::AbstractArray{T}
) where {F,T}
    return activation(opmode, σ, x)
end

@is_primitive MinimalCtx Tuple{
    typeof(_activation_fast),AbstractInternalArrayOpMode,F,AbstractArray
} where {F}

function Mooncake.rrule!!(
    ::CoDual{typeof(_activation_fast)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    σ::CoDual{F},
    x::CoDual{<:AbstractArray{T}},
) where {F,T}
    _opmode, _σ = primal(opmode), primal(σ)
    _x, x̄ = primal(x), tangent(x)

    if unsafe_known(activation_intermediate_not_needed(_σ, T))
        y = activation(_opmode, _σ, _x)
        ȳ = zero_tangent(y)
        function pb!!_no_intermediate(::NoRData)
            x̄ .+= ∇activation(ȳ, y, _σ, NotaNumber())
            return NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(y, ȳ), pb!!_no_intermediate
    end

    # activation_has_rrule is true (guaranteed by overlay).
    y = activation(_opmode, _σ, _x)
    ȳ = zero_tangent(y)
    function pb!!_has_rrule(::NoRData)
        x̄ .+= ∇activation(ȳ, y, _σ, _x)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), pb!!_has_rrule
end

# Overlay: fast activations go through _activation_fast; custom σ falls back to σ.(x).
@mooncake_overlay function activation(
    opmode::AbstractInternalArrayOpMode, σ::F, x::AbstractArray{T}
) where {F,T}
    if unsafe_known(activation_intermediate_not_needed(σ, T)) ||
       unsafe_known(activation_has_rrule(σ, T))
        return _activation_fast(opmode, σ, x)
    end
    return σ.(x)
end

# Under Mooncake, internal_operation_mode returns GenericBroadcastOp so activation!! is non-mutating.
@mooncake_overlay function activation!!(
    opmode::AbstractInternalArrayOpMode, ::True, σ::F, x::AbstractArray
) where {F}
    return activation(opmode, σ, x)
end

# Helper for bias add with identity activation; rrule uses LuxLib's ∇bias_add.
function _bias_add_identity(
    opmode::AbstractInternalArrayOpMode, x::AbstractArray{xT,N}, bias::AbstractVector
) where {xT,N}
    T = concrete_bias_act_output_eltype(identity, x, bias)
    out = similar(x, T)
    bias_add!(out, opmode, x, bias)
    return out
end

@is_primitive MinimalCtx Tuple{
    typeof(_bias_add_identity),AbstractInternalArrayOpMode,AbstractArray,AbstractVector
}

function Mooncake.rrule!!(
    ::CoDual{typeof(_bias_add_identity)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    x::CoDual{<:AbstractArray{xT,N}},
    bias::CoDual{<:AbstractVector},
) where {xT,N}
    _opmode = primal(opmode)
    _x, x̄ = primal(x), tangent(x)
    _bias, b̄ = primal(bias), tangent(bias)

    T = concrete_bias_act_output_eltype(identity, _x, _bias)
    out = similar(_x, T)
    bias_add!(out, _opmode, _x, _bias)
    ō = zero_tangent(out)

    function pb!!(::NoRData)
        x̄ .+= ō
        b̄ .+= ∇bias_add(_bias, ō)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(out, ō), pb!!
end

# Fast helper for bias_activation; called from the overlay for fast activations.
function _bias_activation_fast(
    opmode::AbstractInternalArrayOpMode,
    σ::F,
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {F,xT,N}
    return bias_activation(opmode, σ, x, bias)
end

@is_primitive MinimalCtx Tuple{
    typeof(_bias_activation_fast),
    AbstractInternalArrayOpMode,
    F,
    AbstractArray,
    AbstractVector,
} where {F}

function Mooncake.rrule!!(
    ::CoDual{typeof(_bias_activation_fast)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    σ::CoDual{F},
    x::CoDual{<:AbstractArray{xT,N}},
    bias::CoDual{<:AbstractVector},
) where {F,xT,N}
    _opmode, _σ = primal(opmode), primal(σ)
    _x, x̄ = primal(x), tangent(x)
    _bias, b̄ = primal(bias), tangent(bias)
    T = concrete_bias_act_output_eltype(_σ, _x, _bias)

    if unsafe_known(activation_intermediate_not_needed(_σ, T))
        y = bias_activation(_opmode, _σ, _x, _bias)
        ȳ = zero_tangent(y)
        function pb!!_no_intermediate(::NoRData)
            ∂x = ∇activation(ȳ, y, _σ, NotaNumber())
            x̄ .+= ∂x
            b̄ .+= ∇bias_add(_bias, ∂x)
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(y, ȳ), pb!!_no_intermediate
    end

    # activation_has_rrule is true (guaranteed by overlay).
    tmp = similar(_x, T)
    bias_add!(tmp, _opmode, _x, _bias)
    y = activation(_opmode, _σ, tmp)
    ȳ = zero_tangent(y)
    function pb!!_has_rrule(::NoRData)
        ∂x = ∇activation(ȳ, y, _σ, tmp)
        x̄ .+= ∂x
        b̄ .+= ∇bias_add(_bias, ∂x)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), pb!!_has_rrule
end

# Overlay: fast activations go through _bias_activation_fast; custom σ uses _bias_add_identity + σ.(y).
@mooncake_overlay function bias_activation(
    opmode::AbstractInternalArrayOpMode,
    σ::F,
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {F,xT,N}
    T = concrete_bias_act_output_eltype(σ, x, bias)
    if unsafe_known(activation_intermediate_not_needed(σ, T)) ||
       unsafe_known(activation_has_rrule(σ, T))
        return _bias_activation_fast(opmode, σ, x, bias)
    end
    y = _bias_add_identity(opmode, x, bias)
    return σ.(y)
end

# Under Mooncake, internal_operation_mode returns GenericBroadcastOp so bias_activation!! is non-mutating.
@mooncake_overlay function bias_activation!!(
    opmode::AbstractInternalArrayOpMode,
    ::True,
    σ::F,
    x::AbstractArray,
    bias::AbstractVector,
) where {F}
    return bias_activation(opmode, σ, x, bias)
end

# Fast helper for fused_conv; rrule uses LuxLib's fused ∇conv_bias kernel.
function _fused_conv_fast(
    opmode::AbstractInternalArrayOpMode,
    act::F,
    weight::AbstractArray{wT,N},
    x::AbstractArray{xT,N},
    bias::LuxLib.Optional{<:AbstractVector},
    cdims::LuxLib.Impl.ConvDims,
) where {F,wT,xT,N}
    return fused_conv(opmode, act, weight, x, bias, cdims)
end

@is_primitive MinimalCtx Tuple{
    typeof(_fused_conv_fast),
    AbstractInternalArrayOpMode,
    F,
    AbstractArray{<:Any,N},
    AbstractArray{<:Any,N},
    LuxLib.Optional{<:AbstractVector},
    LuxLib.Impl.ConvDims,
} where {F,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(_fused_conv_fast)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    act::CoDual{F},
    weight::CoDual{<:AbstractArray{wT,N}},
    x::CoDual{<:AbstractArray{xT,N}},
    bias::CoDual{<:LuxLib.Optional{<:AbstractVector}},
    cdims::CoDual{<:LuxLib.Impl.ConvDims},
) where {F,wT,xT,N}
    _opmode, _act = primal(opmode), primal(act)
    _weight, w̄ = primal(weight), tangent(weight)
    _x, x̄ = primal(x), tangent(x)
    _bias, b̄ = primal(bias), tangent(bias)
    _cdims = primal(cdims)

    T = concrete_bias_act_output_eltype(_act, _weight, _x, _bias)

    if unsafe_known(activation_intermediate_not_needed(_act, T))
        y = conv_bias_act(_x, _weight, _cdims, _bias, _act)
        ȳ = zero_tangent(y)
        function pb!!_no_intermediate(::NoRData)
            ∂y = ∇activation(ȳ, y, _act, NotaNumber())
            # ∇conv_bias follows CRC conventions: returns CRC.NoTangent() when bias=nothing.
            ∂w, ∂x, ∂b = ∇conv_bias(∂y, _weight, _x, _bias, _cdims)
            w̄ .+= ∂w
            x̄ .+= ∂x
            ∂b isa CRC.NoTangent || (b̄ .+= ∂b)
            return NoRData(),
            NoRData(), NoRData(), NoRData(), NoRData(), NoRData(),
            NoRData()
        end
        return CoDual(y, ȳ), pb!!_no_intermediate
    end

    # activation_has_rrule is true (guaranteed by overlay).
    y_pre = similar(
        _x, T, NNlib.output_size(_cdims)..., NNlib.channels_out(_cdims), size(_x, N)
    )
    conv!(y_pre, _x, _weight, _cdims)
    z, tmp = bias_activation_cached!!(_act, y_pre, _bias)
    ȳ = zero_tangent(z)
    function pb!!_has_rrule(::NoRData)
        ∂y = ∇activation(ȳ, z, _act, tmp)
        ∂w, ∂x, ∂b = ∇conv_bias(∂y, _weight, _x, _bias, _cdims)
        w̄ .+= ∂w
        x̄ .+= ∂x
        ∂b isa CRC.NoTangent || (b̄ .+= ∂b)
        return NoRData(),
        NoRData(), NoRData(), NoRData(), NoRData(), NoRData(),
        NoRData()
    end
    return CoDual(z, ȳ), pb!!_has_rrule
end

# Overlay: fast activations go through _fused_conv_fast; custom act decomposes into conv + bias_activation.
@mooncake_overlay function fused_conv(
    opmode::AbstractInternalArrayOpMode,
    act::F,
    weight::AbstractArray{wT,N},
    x::AbstractArray{xT,N},
    bias::LuxLib.Optional{<:AbstractVector},
    cdims::LuxLib.Impl.ConvDims,
) where {F,wT,xT,N}
    T = concrete_bias_act_output_eltype(act, weight, x, bias)
    if unsafe_known(activation_intermediate_not_needed(act, T)) ||
       unsafe_known(activation_has_rrule(act, T))
        return _fused_conv_fast(opmode, act, weight, x, bias, cdims)
    end
    y_pre = conv(x, weight, cdims)
    return bias_activation(act, y_pre, bias)
end

# Native Mooncake rrules for groupnorm_affine_normalize_internal

import LuxLib.Impl:
    groupnorm_affine_normalize_internal,
    groupnorm_affine_normalize_internal!,
    ∇groupnorm_affine_normalize

# Helper function for groupnorm with identity activation (analogous to
# _batchnorm_affine_normalize_identity).
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
