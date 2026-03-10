module MooncakeLuxLibExt

using LuxLib, Random, Mooncake, NNlib
using Base: IEEEFloat

import LuxLib: Impl, Utils
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

@zero_adjoint DefaultCtx Tuple{typeof(static_training_mode_check),Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(generate_dropout_mask),AbstractRNG,Any,Any,Any,Any}
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

# Overlay for `batchnorm_affine_normalize_internal`
#  - Use Mooncake’s helper function `_batchnorm_affine_normalize_identity` and its manually written rule.
#  - Let Mooncake differentiate through the broadcasted `act` function.
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

# Native Mooncake rrules for activation

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
        _x_orig = copy(_x)
        activation!(_x, _opmode, _σ, _x)

        function pb!!_no_intermediate(::NoRData)
            ∂x = ∇activation(x̄, _x, _σ, NotaNumber())
            copyto!(x̄, ∂x)
            copyto!(_x, _x_orig)
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(_x, x̄), pb!!_no_intermediate
    end

    if unsafe_known(activation_has_rrule(_σ, T))
        _x_orig = copy(_x)
        y = activation(_opmode, _σ, _x)
        copyto!(_x, y)

        function pb!!_has_rrule(::NoRData)
            ∂x = ∇activation(x̄, y, _σ, _x_orig)
            copyto!(x̄, ∂x)
            copyto!(_x, _x_orig)
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(_x, x̄), pb!!_has_rrule
    end

    _x_orig = copy(_x)
    y = _σ.(_x)
    copyto!(_x, y)
    act_cache = prepare_pullback_cache(broadcast, _σ, _x_orig)

    function pb!!_fallback(::NoRData)
        _, (_, _, ∂x) = value_and_pullback!!(act_cache, copy(x̄), broadcast, _σ, _x_orig)
        copyto!(x̄, ∂x)
        copyto!(_x, _x_orig)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(_x, x̄), pb!!_fallback
end

@is_primitive MinimalCtx Tuple{
    typeof(activation),AbstractInternalArrayOpMode,F,AbstractArray
} where {F}

function Mooncake.rrule!!(
    ::CoDual{typeof(activation)},
    opmode::CoDual{<:AbstractInternalArrayOpMode},
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

# Native Mooncake rrules for bias_activation

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
        _x_orig = copy(_x)
        y = bias_activation!!(_opmode, True(), _σ, _x, _bias)
        mutated = y === _x
        ȳ = mutated ? x̄ : zero_tangent(y)

        function pb!!_no_intermediate(::NoRData)
            ∂x = ∇activation(ȳ, y, _σ, NotaNumber())
            b̄ .+= ∇bias_add(_bias, ∂x)
            if mutated
                copyto!(x̄, ∂x)
                copyto!(_x, _x_orig)
            else
                x̄ .+= ∂x
            end
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(y, ȳ), pb!!_no_intermediate
    end

    if unsafe_known(activation_has_rrule(_σ, T))
        _x_orig = copy(_x)
        tmp = similar(_x, T)
        bias_add!(tmp, _opmode, _x, _bias)
        y = activation(_opmode, _σ, tmp)
        mutated = !(_opmode isa LuxLib.GenericBroadcastOp)
        mutated && copyto!(_x, y)
        ȳ = mutated ? x̄ : zero_tangent(y)

        function pb!!_has_rrule(::NoRData)
            ∂x = ∇activation(ȳ, y, _σ, tmp)
            b̄ .+= ∇bias_add(_bias, ∂x)
            if mutated
                copyto!(x̄, ∂x)
                copyto!(_x, _x_orig)
            else
                x̄ .+= ∂x
            end
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(mutated ? _x : y, ȳ), pb!!_has_rrule
    end

    # Fallback
    _x_orig = copy(_x)
    _rb = reshape_bias(_x_orig, _bias)
    y = broadcast_bias_activation_generic(_σ, _x_orig, _rb)
    mutated = y === _x
    mutated || copyto!(_x, y)
    ȳ = mutated ? x̄ : zero_tangent(y)
    act_cache = prepare_pullback_cache(broadcast_bias_activation_generic, _σ, _x_orig, _rb)

    function pb!!_fallback(::NoRData)
        _, (_, ∂x, ∂rb) = value_and_pullback!!(
            act_cache, copy(ȳ), broadcast_bias_activation_generic, _σ, _x_orig, _rb
        )
        if mutated
            copyto!(x̄, ∂x)
            copyto!(_x, _x_orig)
        else
            x̄ .+= ∂x
        end
        b̄ .+= vec(∂rb)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), pb!!_fallback
end

# Native Mooncake rrules for fused_conv

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

# Native Mooncake rrules for groupnorm_affine_normalize_internal

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

    act_cache = prepare_pullback_cache(broadcast, _f, y)
    z = _f.(y)
    ȳ = zero_tangent(z)

    function pb!!(::NoRData)
        _, (_, _, ∂y) = value_and_pullback!!(act_cache, ȳ, broadcast, _f, y)

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
