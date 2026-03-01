module MooncakeLuxLibExt

using LuxLib, Random, Mooncake
using Base: IEEEFloat

import LuxLib: Impl, Utils
using MLDataDevices: get_device_type
using Mooncake:
    @from_rrule,
    DefaultCtx,
    MinimalCtx,
    @mooncake_overlay,
    CoDual,
    NoTangent,
    zero_tangent,
    primal,
    @is_primitive,
    NoRData,
    extract,
    zero_fcodual,
    tangent,
    prepare_pullback_cache,
    value_and_pullback!!,
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

import LuxLib.Utils: static_training_mode_check

@zero_adjoint DefaultCtx Tuple{typeof(static_training_mode_check),Vararg}

import LuxLib.Impl:
    select_fastest_activation,
    sleefpirates_fast_act,
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

@zero_adjoint DefaultCtx Tuple{typeof(select_fastest_activation),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(sleefpirates_fast_act),Vararg}
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
@zero_adjoint DefaultCtx Tuple{typeof(generate_dropout_mask),AbstractRNG,Any,Any,Any,Any}
@zero_adjoint DefaultCtx Tuple{typeof(update_running_statistics),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(update_normalization_statistics),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(get_norm_reshape_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(instancenorm_reduce_dims),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(compute_layernorm_dims),Vararg}

# This is a really horrible hack that we need to do until Mooncake is able to support the
# call-back-into-ad interface that ChainRules exposes.

import LuxLib.Impl:
    safe_eltype,
    batchnorm_affine_normalize_internal,
    batchnorm_affine_normalize_internal!,
    ∇batchnorm_affine_normalize,
    AbstractInternalArrayOpMode

import ChainRulesCore as CRC

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

# LuxLib native Mooncake rrules
# Activation.    
import LuxLib.Utils: True, False, unsafe_known
import LuxLib.Impl:
    activation!!,
    activation!,
    LoopedArrayOp,
    bias_activation,
    bias_activation!!,
    bias_activation!,
    bias_activation_cached!!,
    bias_add!,
    broadcast_bias_activation_generic,
    fused_conv,
    conv_bias_act,
    conv!,
    ∇fused_conv,
    ∇conv_bias,
    groupnorm_affine_normalize_internal,
    groupnorm_affine_normalize_internal!,
    ∇groupnorm_affine_normalize,
    concrete_bias_act_output_eltype,
    reshape_bias,
    ∇bias_add,
    activation_intermediate_not_needed,
    activation_has_rrule,
    activation,
    ∇activation,
    AbstractInternalArrayOpMode,
    LoopedArrayOp

@is_primitive MinimalCtx Tuple{
    typeof(activation!!),AbstractInternalArrayOpMode,True,F,AbstractArray
} where {F}

function Mooncake.rrule!!(
    ::CoDual{typeof(activation!!)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    ::CoDual{True},
    σ::CoDual{F},
    x::CoDual{<:AbstractArray{T}},
) where {F,T}
    _opmode, _σ = primal(opmode), primal(σ)
    _x, x̄ = primal(x), tangent(x)

    if unsafe_known(activation_intermediate_not_needed(_σ, T))
        activation!(_x, _opmode, _σ, _x)

        function pb!!_no_intermediate(::NoRData)
            x̄ .+= ∇activation(_x, _x, _σ, NotaNumber())  # Δ=x̄ accumulated upstream
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(_x, x̄), pb!!_no_intermediate
    end

    if unsafe_known(activation_has_rrule(_σ, T))
        y = activation(_opmode, _σ, _x)
        ȳ = zero_tangent(y)

        function pb!!_has_rrule(::NoRData)
            x̄ .+= ∇activation(ȳ, y, _σ, _x)
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(y, ȳ), pb!!_has_rrule
    end

    act_cache = prepare_pullback_cache(broadcast, _σ, _x)
    y = _σ.(_x)
    ȳ = zero_tangent(y)

    function pb!!_fallback(::NoRData)
        _, (_, _, ∂x) = value_and_pullback!!(act_cache, ȳ, broadcast, _σ, _x)
        x̄ .+= ∂x
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), pb!!_fallback
end

@is_primitive MinimalCtx Tuple{typeof(activation),LoopedArrayOp,F,AbstractArray} where {F}

function Mooncake.rrule!!(
    ::CoDual{typeof(activation)},
    opmode::CoDual{<:LoopedArrayOp},
    σ::CoDual{F},
    x::CoDual{<:AbstractArray{T}},
) where {F,T}
    _opmode, _σ = primal(opmode), primal(σ)
    _x, x̄ = primal(x), tangent(x)

    if unsafe_known(activation_has_rrule(_σ, T))
        y = activation(_opmode, _σ, _x)
        ȳ = zero_tangent(y)

        function pb!!_has_rrule(::NoRData)
            x̄ .+= ∇activation(ȳ, y, _σ, _x)
            return NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(y, ȳ), pb!!_has_rrule
    end

    act_cache = prepare_pullback_cache(broadcast, _σ, _x)
    y = _σ.(_x)
    ȳ = zero_tangent(y)

    function pb!!_fallback(::NoRData)
        _, (_, _, ∂x) = value_and_pullback!!(act_cache, ȳ, broadcast, _σ, _x)
        x̄ .+= ∂x
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), pb!!_fallback
end

# bias_activation

@is_primitive MinimalCtx Tuple{
    typeof(bias_activation),AbstractInternalArrayOpMode,F,AbstractArray,AbstractVector
} where {F}

function Mooncake.rrule!!(
    ::CoDual{typeof(bias_activation)},
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

    if unsafe_known(activation_has_rrule(_σ, T))
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

    _rb = reshape_bias(_x, _bias)
    act_cache = prepare_pullback_cache(broadcast_bias_activation_generic, _σ, _x, _rb)
    y = broadcast_bias_activation_generic(_σ, _x, _rb)
    ȳ = zero_tangent(y)

    function pb!!_fallback(::NoRData)
        _, (_, ∂x, ∂rb) = value_and_pullback!!(
            act_cache, ȳ, broadcast_bias_activation_generic, _σ, _x, _rb
        )
        x̄ .+= ∂x
        b̄ .+= vec(∂rb)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), pb!!_fallback
end

# bias_activation!!

@is_primitive MinimalCtx Tuple{
    typeof(bias_activation!!),
    AbstractInternalArrayOpMode,
    True,
    F,
    AbstractArray,
    AbstractVector,
} where {F}

function Mooncake.rrule!!(
    ::CoDual{typeof(bias_activation!!)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    ::CoDual{True},
    σ::CoDual{F},
    x::CoDual{<:AbstractArray{xT,N}},
    bias::CoDual{<:AbstractVector},
) where {F,xT,N}
    _opmode, _σ = primal(opmode), primal(σ)
    _x, x̄ = primal(x), tangent(x)
    _bias, b̄ = primal(bias), tangent(bias)

    T = concrete_bias_act_output_eltype(_σ, _x, _bias)

    if unsafe_known(activation_intermediate_not_needed(_σ, T))
        bias_activation!(_x, _opmode, _σ, _x, _bias)
        ȳ = zero_tangent(_x)

        function pb!!_no_intermediate(::NoRData)
            ∂x = ∇activation(ȳ, _x, _σ, NotaNumber())
            x̄ .+= ∂x
            b̄ .+= ∇bias_add(_bias, ∂x)
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(_x, ȳ), pb!!_no_intermediate
    end

    if unsafe_known(activation_has_rrule(_σ, T))
        y, tmp = bias_activation_cached!!(_σ, _x, _bias)
        ȳ = zero_tangent(y)

        function pb!!_has_rrule(::NoRData)
            ∂x = ∇activation(ȳ, y, _σ, tmp)
            x̄ .+= ∂x
            b̄ .+= ∇bias_add(_bias, ∂x)
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(y, ȳ), pb!!_has_rrule
    end

    act_cache = prepare_pullback_cache(bias_activation, _opmode, _σ, _x, _bias)
    y = bias_activation(_opmode, _σ, _x, _bias)
    ȳ = zero_tangent(y)

    function pb!!_fallback(::NoRData)
        _, (_, _, _, ∂x, ∂b) = value_and_pullback!!(
            act_cache, ȳ, bias_activation, _opmode, _σ, _x, _bias
        )
        x̄ .+= ∂x
        b̄ .+= ∂b
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), pb!!_fallback
end

# fused_conv

@is_primitive MinimalCtx Tuple{
    typeof(fused_conv),
    AbstractInternalArrayOpMode,
    F,
    AbstractArray{<:Any,N},
    AbstractArray{<:Any,N},
    LuxLib.Optional{<:AbstractVector},
    LuxLib.Impl.ConvDims,
} where {F,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(fused_conv)},
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

    y_pre = similar(
        _x, T, NNlib.output_size(_cdims)..., NNlib.channels_out(_cdims), size(_x, N)
    )
    conv!(y_pre, _x, _weight, _cdims)

    if unsafe_known(activation_has_rrule(_act, T))
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

    act_cache = prepare_pullback_cache(bias_activation, _act, y_pre, _bias)
    z = bias_activation(_act, y_pre, _bias)
    ȳ = zero_tangent(z)

    function pb!!_fallback(::NoRData)
        _, (_, ∂y_pre, ∂b) = value_and_pullback!!(
            act_cache, ȳ, bias_activation, _act, y_pre, _bias
        )
        ∂w, ∂x, _ = ∇conv_bias(∂y_pre, ∂b, _weight, _x, _bias, _cdims)
        w̄ .+= ∂w
        x̄ .+= ∂x
        ∂b isa CRC.NoTangent || (b̄ .+= ∂b)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(z, ȳ), pb!!_fallback
end

# groupnorm_affine_normalize_internal

import LuxLib.Impl:
    groupnorm_affine_normalize_internal,
    groupnorm_affine_normalize_internal!,
    ∇groupnorm_affine_normalize

@is_primitive MinimalCtx Tuple{
    typeof(groupnorm_affine_normalize_internal),
    AbstractInternalArrayOpMode,
    F,
    AbstractArray{<:Any,4},
    AbstractArray{<:Any,4},
    AbstractArray{<:Any,4},
    LuxLib.Optional{<:AbstractArray{<:Any,4}},
    LuxLib.Optional{<:AbstractArray{<:Any,4}},
    Real,
} where {F}

function Mooncake.rrule!!(
    ::CoDual{typeof(groupnorm_affine_normalize_internal)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
    f::CoDual{F},
    x::CoDual{<:AbstractArray{T,4}},
    μ::CoDual{<:AbstractArray{μT,4}},
    σ²::CoDual{<:AbstractArray{σ²T,4}},
    γ::CoDual{<:LuxLib.Optional{<:AbstractArray{<:Any,4}}},
    β::CoDual{<:LuxLib.Optional{<:AbstractArray{<:Any,4}}},
    ϵ::CoDual{<:Real},
) where {F,T,μT,σ²T}
    _opmode, _f, _ϵ = primal(opmode), primal(f), primal(ϵ)
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

    act_cache = prepare_pullback_cache(_f, y)
    z = _f.(y)
    ȳ = zero_tangent(z)

    function pb!!(::NoRData)
        _, (_, ∂y) = value_and_pullback!!(act_cache, ȳ, _f, y)

        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇groupnorm_affine_normalize(
            _opmode, ∂y, _x, _μ, _σ², _γ, _β, _ϵ
        )

        x̄ .+= ∂x
        μ̄ .+= ∂μ
        σ²̄ .+= ∂σ²
        ∂γ isa CRC.NoTangent || (γ̄ .+= ∂γ)
        ∂β isa CRC.NoTangent || (β̄ .+= ∂β)

        return NoRData(),
        NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(),
        NoRData()
    end

    return CoDual(z, ȳ), pb!!
end

end
