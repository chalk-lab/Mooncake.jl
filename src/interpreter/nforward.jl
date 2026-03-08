# ── nforward: NDual-backed forward-mode engine ────────────────────────────────────
# `nforward` evaluates code by lifting inputs into `NDual`s and running the primal
# function directly on those lifted values. It does not reuse Mooncake's ordinary forward
# interpreter, even when `chunk_size == 1`.
#
# ── File layout ────────────────────────────────────────────────────────────────────
# This file is organized as:
# - core types
# - public rule/cache entrypoints
# - shared validation and layout helpers
# - reverse accumulation and execution
# - forward evaluation pipeline
# - cache spec checks
# - cached scalar/array fast paths
#
# ── High-level interfaces ──────────────────────────────────────────────────────────
#   nforward_build_frule(f, x...; chunk_size)
#     returns `NForwardRule`
#     consumed via `rule(f::Dual, x::Dual...)`
#     obeys the standard `frule!!` interface
#     also accepts `sig::Type{<:Tuple}` for signature-based construction
#
#   nforward_build_rrule(f, x...; chunk_size)
#     returns `NForwardRRule`
#     consumed via `rule(f::CoDual, x::CoDual...)`
#     obeys the standard `rrule!!` interface
#     also accepts `sig::Type{<:Tuple}` for signature-based construction
#
#   nforward_prepare_cache(f, x...; chunk_size)
#     returns `NForwardCache`
#     consumed via `value_and_derivative!!(cache, f::Dual, x::Dual...)`
#     consumed via `value_and_gradient!!(cache, f, x...)`
#
# ── Constraints ────────────────────────────────────────────────────────────────────
# - `chunk_size` is global across the whole call
# - supported primals are IEEE float scalars, complex IEEE float scalars, and dense arrays
#   with those element types
# - rule construction requires stateless callables (singleton callable types)
# - `friendly_tangents=true`, `debug_mode=true`, and differentiation with respect to `f`
#   are intentionally unsupported here
#
# ── Primitive Reverse-Mode Example ────────────────────────────────────────────────
# `nforward` can also be used to define reverse-mode rules, and is especially useful
# when dual-number forward differentiation is more compiler-friendly than IR-transform-
# based AD, for example when differentiating through CUDA kernels. It also often has
# significantly lower compilation latency:
#   f(x) = sum(abs2, x)
#   sig = Tuple{typeof(f),Vector{Float64}}
#   Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode sig
#   Mooncake.build_primitive_rrule(::Type{sig}) =
#       Mooncake.nforward_build_rrule(sig; chunk_size=4)
#
include(joinpath(@__DIR__, "..", "..", "ext", "MooncakeCUDAExt", "ndual.jl"))

#
# Core types
#
# High-level rule/cache objects first. The implementation details they rely on are defined
# in later sections.

"""
    NForwardRule

Callable forward-mode rule used by `nforward`.

`NForwardRule` is built from a statically-known call signature.
"""
struct NForwardRule{sig,N} end

"""
    NForwardRRule

Callable reverse-mode rule used by `nforward`.

`NForwardRRule` is built from a statically-known call signature. Both direct
[`nforward_build_rrule`](@ref) calls and primitive reverse-mode registration route through
that signature-based construction path. `buf` holds reusable typed scratch buffers for
cached scalar-output fast paths when that is available.
"""
struct NForwardRRule{sig,N,Tbuf<:Base.RefValue}
    buf::Tbuf
end

"""
    NForwardCache

Cache returned by [`nforward_prepare_cache`](@ref). It owns the forward/reverse rules,
runtime input signatures, and reusable zero tangents used by cached `nforward` calls.
"""
struct NForwardCache{RF,RR,S<:Tuple,TT<:Tuple}
    frule::RF
    rrule::RR
    sigs::S
    tangents::TT
end

"""
    NForwardPullback

Concrete pullback object for `nforward` reverse rules. It stores the primal callable,
primals, input tangents, and output fdata needed to rerun chunked NDual passes during the
reverse sweep.
"""
struct NForwardPullback{F,N,P,T,Y}
    f::F
    primals::P
    tangents::T
    y_fdata::Y
end

#
# Public construction and cached execution
#
# These are the main `nforward` entry points. A reviewer should be able to read this section
# first, then dive into the lower-level pipelines only as needed.

"""
    nforward_build_frule(f, x...; chunk_size)
    nforward_build_frule(sig::Type{<:Tuple}; chunk_size)

Build a forward-mode rule through `nforward`.

This path is independent from Mooncake's ordinary forwards-mode interpreter and obeys the
standard `frule!!` interface. It evaluates the primal function directly on NDual-lifted
scalar / dense-array inputs. Rule construction is signature-based, so `nforward` only
supports stateless callables here.
"""
function nforward_build_frule(
    sig::Type{<:Tuple}; chunk_size::Integer, debug_mode=false, silence_debug_messages=true
)
    chunk_size = _nforward_check_chunk_size(chunk_size)
    _nforward_check_callable_sig(sig)
    debug_mode &&
        throw(ArgumentError("nforward does not currently support `debug_mode=true`."))
    silence_debug_messages
    return NForwardRule{sig,chunk_size}()
end

function nforward_build_frule(
    f, x...; chunk_size::Integer, debug_mode=false, silence_debug_messages=true
)
    return nforward_build_frule(
        typeof((f, x...)); chunk_size, debug_mode, silence_debug_messages
    )
end

function (rule::NForwardRule{sig,N})(f::Dual, x::Vararg{Dual,M}) where {sig,N,M}
    __verify_sig(rule, (f, x...))
    _nforward_check_function_tangent(tangent(f))
    primals = map(primal, x)
    tangents = map(tangent, x)
    y, dy = _nforward_eval(primal(f), primals, tangents, Val(N))
    return Dual(y, dy)
end

"""
    nforward_build_rrule(f, x...; chunk_size)
    nforward_build_rrule(sig::Type{<:Tuple}; chunk_size)

Build a reverse-mode rule through `nforward`.

The reverse rule is derived from chunked NDual forward passes and obeys the standard
`rrule!!` interface. Rule construction is signature-based, so `nforward` only supports
stateless callables here.
"""
function nforward_build_rrule(
    sig::Type{<:Tuple}; chunk_size::Integer, debug_mode=false, silence_debug_messages=true
)
    chunk_size = _nforward_check_chunk_size(chunk_size)
    _nforward_check_callable_sig(sig)
    debug_mode &&
        throw(ArgumentError("nforward does not currently support `debug_mode=true`."))
    silence_debug_messages
    buf = _nforward_buf_ref(sig, Val(chunk_size))
    return NForwardRRule{sig,chunk_size,typeof(buf)}(buf)
end

function nforward_build_rrule(
    f, x...; chunk_size::Integer, debug_mode=false, silence_debug_messages=true
)
    return nforward_build_rrule(
        typeof((f, x...)); chunk_size, debug_mode, silence_debug_messages
    )
end

"""
    (rule::NForwardRRule)(f::CoDual, x::Vararg{CoDual})

Evaluate an `nforward` reverse rule and return both the output `CoDual` and pullback.
"""
function (rule::NForwardRRule{sig,N})(f::CoDual, x::Vararg{CoDual,M}) where {sig,N,M}
    __verify_sig(rule, (f, x...))
    tangent(f) isa NoFData || throw(
        ArgumentError("nforward does not support differentiating with respect to `f`.")
    )
    return _nforward_rrule_call(primal(f), x, Val(N))
end

"""
    nforward_prepare_cache(f, x...; chunk_size=nothing, config=Mooncake.Config())

Prepare a cache for [`value_and_derivative!!`](@ref) and [`value_and_gradient!!`](@ref)
when used with an [`NForwardCache`](@ref).

If `chunk_size` is omitted, nforward uses `min(total_dof, 8)` with a floor of 1.
"""
@unstable @inline function nforward_prepare_cache(
    f, x::Vararg{Any,N}; chunk_size=nothing, config=Config()
) where {N}
    _nforward_check_config(config)
    map(_nforward_check_primal, x)
    chunk_size = _nforward_resolve_chunk_size(chunk_size, x)
    sigs = map(_nforward_sig, (f, x...))
    frule = nforward_build_frule(
        f,
        x...;
        chunk_size,
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    rrule = nforward_build_rrule(
        f,
        x...;
        chunk_size,
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    tangents = map(zero_tangent, (f, x...))
    return NForwardCache(frule, rrule, sigs, tangents)
end

"""
    value_and_derivative!!(cache::NForwardCache, f::Dual, x::Dual...)

Run the cached `nforward` forward rule after checking that the runtime inputs match the cache
signature.
"""
function value_and_derivative!!(cache::NForwardCache, f::Dual, x::Vararg{Dual,N}) where {N}
    _nforward_check_sigs(cache.sigs, (primal(f), map(primal, x)...))
    return cache.frule(f, x...)
end

"""
    value_and_gradient!!(cache::NForwardCache, f, x...; args_to_zero=...)

Run the cached `nforward` reverse rule after validating cache specs and zeroing the cached
input tangents.
"""
function value_and_gradient!!(
    cache::NForwardCache,
    f::F,
    x::Vararg{Any,N};
    args_to_zero::NTuple=ntuple(Returns(true), Val(N + 1)),
) where {F,N}
    _nforward_check_args_to_zero(args_to_zero, N + 1)
    _nforward_check_sigs(cache.sigs, (f, x...))
    tangents = tuple_map(set_to_zero_maybe!!, cache.tangents, args_to_zero)
    return __value_and_gradient!!(cache.rrule, tuple_map(CoDual, (f, x...), tangents)...)
end

# Optimization note:
# This scalar specialization bypasses the general pullback-based reverse path for cached
# `value_and_gradient!!` calls. Evaluating one NDual-lifted primal directly is enough to recover
# the scalar primal and derivative, which removes the remaining steady-state allocations for
# singleton scalar inputs.
"""
    __value_and_gradient!!(rule::NForwardRRule, f::CoDual, x::CoDual)

Dispatch the scalar cached fast path for `nforward` reverse mode.
"""
function __value_and_gradient!!(
    rule::NForwardRRule{sig,N}, f::CoDual, x::CoDual{T}
) where {sig,N,T<:IEEEFloat}
    __verify_sig(rule, (f, x))
    return _nforward_scalar_value_and_gradient(primal(f), f, x, Val(N))
end

"""
    __value_and_gradient!!(rule::NForwardRRule, f::CoDual, x::CoDual{Vector})

 Cached scalar-output vector fast path for `nforward` reverse rules.
"""
function __value_and_gradient!!(
    rule::NForwardRRule{sig,chunk_size,<:Base.RefValue{Union{Nothing,Tworkspace}}},
    f::CoDual,
    x::CoDual{Vector{T}},
) where {
    sig,
    chunk_size,
    T<:IEEEFloat,
    Tworkspace<:NamedTuple{(:seed, :lifted),Tuple{Matrix{T},Vector{NDual{T,chunk_size}}}},
}
    __verify_sig(rule, (f, x))
    _nforward_check_function_tangent(tangent(f))
    y, x_grad = _nforward_vector_scalar_value_and_gradient(
        primal(f), x, rule.buf, Val(chunk_size)
    )
    return y, (_nforward_function_gradient(f), x_grad)
end

"""
    __value_and_gradient!!(rule::NForwardRRule, f::CoDual, x::CoDual{<:Array})

Scalar-output dense-array fast path used when the vector specialization does not apply.
"""
function __value_and_gradient!!(
    rule::NForwardRRule{sig,chunk_size}, f::CoDual, x::CoDual{A}
) where {sig,chunk_size,T<:IEEEFloat,N,A<:Array{T,N}}
    __verify_sig(rule, (f, x))
    _nforward_check_function_tangent(tangent(f))
    y, x_grad = _nforward_array_scalar_value_and_gradient(
        primal(f), x, rule.buf, Val(chunk_size)
    )
    return y, (_nforward_function_gradient(f), x_grad)
end

const NFORWARD_DEBUG_MODE_WARNING =
    "nforward-backed reverse-mode rules ignore `debug_mode=true`; " *
    "Mooncake's outer debug wrapper still checks CoDual inputs/outputs, but the " *
    "inner nforward rule executes without nforward-specific debug checks."

"""
    _copy(rule::NForwardRRule)

Copy an `NForwardRRule` while resetting cached workspace state.
"""
function _copy(x::P) where {P<:NForwardRRule}
    return P(typeof(x.buf)(nothing))
end

"""
    verify_fwds_inputs(rule::NForwardRRule, x)

Emit the outer debug-mode warning before delegating to generic forward-input checks.
"""
@noinline function verify_fwds_inputs(rule::NForwardRRule, @nospecialize(x::Tuple))
    @warn NFORWARD_DEBUG_MODE_WARNING
    return invoke(verify_fwds_inputs, Tuple{Any,Tuple}, rule, x)
end

#
# Validation and layout helpers
#
# Shared validation, sizing, and shape utilities used across the forward, reverse, and cached
# execution paths.

@inline function _nforward_check_chunk_size(chunk_size::Integer)
    chunk_size > 0 && return Int(chunk_size)
    throw(ArgumentError("`chunk_size` must be a positive integer, got $chunk_size."))
end

@inline function _nforward_default_chunk_size(x::Tuple)
    return max(1, min(sum(_nforward_input_dof, x), 8))
end

"""
    _nforward_resolve_chunk_size(chunk_size, x)

Return the chunk size used for an `nforward` call, defaulting to
`min(total_dof(x), 8)` when `chunk_size === nothing`.
"""
@inline function _nforward_resolve_chunk_size(chunk_size, x::Tuple)
    return if isnothing(chunk_size)
        _nforward_default_chunk_size(x)
    else
        _nforward_check_chunk_size(chunk_size)
    end
end

@inline _nforward_is_supported_scalar(::Type{<:IEEEFloat}) = true
@inline _nforward_is_supported_scalar(::Type{<:Complex{<:IEEEFloat}}) = true
@inline _nforward_is_supported_scalar(::Type) = false

@inline _nforward_is_supported_primal(::Type{<:IEEEFloat}) = true
@inline _nforward_is_supported_primal(::Type{<:Complex{<:IEEEFloat}}) = true
@inline function _nforward_is_supported_primal(::Type{<:Array{ET}}) where {ET}
    _nforward_is_supported_scalar(ET)
end
@inline _nforward_is_supported_primal(::Type) = false

@inline _nforward_input_error(x) = throw(
    ArgumentError(
        "nforward currently supports only IEEEFloat / Complex IEEEFloat scalars and " *
        "dense Array inputs with those element types. Got $(typeof(x)).",
    ),
)

@inline function _nforward_output_error(y)
    throw(
        ArgumentError(
            "nforward currently supports only scalar or dense Array outputs with IEEEFloat / " *
            "Complex IEEEFloat element types. Got $(typeof(y)).",
        ),
    )
end

@inline function _nforward_check_primal(x)
    _nforward_is_supported_primal(typeof(x)) || _nforward_input_error(x)
    return x
end

@inline function _nforward_check_function_tangent(df)
    df isa NoTangent && return nothing
    throw(ArgumentError("nforward does not support differentiating with respect to `f`."))
end

@inline function _nforward_check_callable_sig(sig::Type{<:Tuple})
    F = sig.parameters[1]
    Base.issingletontype(F) || throw(
        ArgumentError(
            "nforward only supports stateless callables for rule construction. Got $F."
        ),
    )
    f = F.instance
    argsig = Tuple{(sig.parameters[i] for i in 2:length(sig.parameters))...}
    hasmethod(f, argsig) && return sig
    throw(
        ArgumentError("nforward rule construction expected a callable signature, got $sig.")
    )
end

@inline function __verify_sig(::NForwardRule{sig}, fx::Tuple) where {sig}
    Tfx = Tuple{map(_typeof ∘ primal, fx)...}
    Tfx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tfx do not subtype rule signature, $sig"))
end

@inline function __verify_sig(::NForwardRRule{sig}, fx::Tuple) where {sig}
    Tfx = Tuple{map(_typeof ∘ primal, fx)...}
    Tfx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tfx do not subtype rule signature, $sig"))
end

@inline function _nforward_check_config(config)
    config.friendly_tangents && throw(
        ArgumentError("nforward does not currently support `friendly_tangents=true`.")
    )
    config.debug_mode &&
        throw(ArgumentError("nforward does not currently support `debug_mode=true`."))
    return nothing
end

@inline function _nforward_check_args_to_zero(args_to_zero::NTuple, expected::Int)
    length(args_to_zero) == expected ||
        throw(ArgumentError("`args_to_zero` must have length $expected for this call."))
    all(args_to_zero) && return nothing
    throw(
        ArgumentError(
            "nforward does not currently support custom `args_to_zero`; pass the default " *
            "all-true tuple instead.",
        ),
    )
end

@inline _nforward_input_dof(x::IEEEFloat) = 1
@inline _nforward_input_dof(x::Complex{<:IEEEFloat}) = 2
@inline _nforward_input_dof(x::AbstractArray{<:IEEEFloat}) = length(x)
@inline _nforward_input_dof(x::AbstractArray{<:Complex{<:IEEEFloat}}) = 2 * length(x)

#
# Reverse accumulation utilities
#
# These helpers seed input directions, contract output tangents with cotangents, and scatter
# each chunk's contributions into gradient storage.

function _nforward_seed_tangent(x::IEEEFloat, chunk_size::Int, start_slot::Int, offset::Int)
    lane = (offset + 1) - start_slot + 1
    if chunk_size == 1
        return lane == 1 ? one(x) : zero(x)
    end
    return ntuple(k -> typeof(x)(k == lane), Val(chunk_size))
end

function _nforward_seed_tangent(
    x::Complex{T}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    if chunk_size == 1
        if offset + 1 == start_slot
            return complex(one(T), zero(T))
        elseif offset + 2 == start_slot
            return complex(zero(T), one(T))
        else
            return zero(x)
        end
    end
    return ntuple(k -> begin
        slot = start_slot + k - 1
        if offset + 1 == slot
            complex(one(T), zero(T))
        elseif offset + 2 == slot
            complex(zero(T), one(T))
        else
            zero(x)
        end
    end, Val(chunk_size))
end

function _nforward_seed_tangent(
    x::AbstractArray{T}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    if chunk_size == 1
        dx = zero_tangent(x)
        global_slot = start_slot
        if offset < global_slot <= offset + length(x)
            dx[global_slot - offset] = one(T)
        end
        return dx
    end
    dx = zeros(T, size(x)..., chunk_size)
    cart_inds = CartesianIndices(x)
    for lane in 1:chunk_size
        global_slot = start_slot + lane - 1
        if offset < global_slot <= offset + length(x)
            idx = Tuple(cart_inds[global_slot - offset])
            dx[idx..., lane] = one(T)
        end
    end
    return dx
end

function _nforward_seed_tangent(
    x::AbstractArray{Complex{T}}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    if chunk_size == 1
        dx = zero_tangent(x)
        global_slot = start_slot
        if offset < global_slot <= offset + 2 * length(x)
            elem = cld(global_slot - offset, 2)
            dx[elem] = if isodd(global_slot - offset)
                complex(one(T), zero(T))
            else
                complex(zero(T), one(T))
            end
        end
        return dx
    end
    dx = zeros(Complex{T}, size(x)..., chunk_size)
    cart_inds = CartesianIndices(x)
    for lane in 1:chunk_size
        global_slot = start_slot + lane - 1
        if offset < global_slot <= offset + 2 * length(x)
            local_slot = global_slot - offset
            elem = cld(local_slot, 2)
            idx = Tuple(cart_inds[elem])
            dx[idx..., lane] =
                isodd(local_slot) ? complex(one(T), zero(T)) : complex(zero(T), one(T))
        end
    end
    return dx
end

@inline function _nforward_add_slot!(
    g::Base.RefValue{T}, local_slot::Int, v
) where {T<:IEEEFloat}
    local_slot == 1 && (g[] += v)
    return nothing
end

@inline function _nforward_add_slot!(
    g::Base.RefValue{Complex{T}}, local_slot::Int, v
) where {T<:IEEEFloat}
    if local_slot == 1
        g[] += complex(v, zero(T))
    elseif local_slot == 2
        g[] += complex(zero(T), v)
    end
    return nothing
end

@inline function _nforward_add_slot!(
    g::AbstractArray{T}, local_slot::Int, v
) where {T<:IEEEFloat}
    g[local_slot] += v
    return nothing
end

@inline function _nforward_add_slot!(
    g::AbstractArray{Complex{T}}, local_slot::Int, v
) where {T<:IEEEFloat}
    elem = cld(local_slot, 2)
    g[elem] += isodd(local_slot) ? complex(v, zero(T)) : complex(zero(T), v)
    return nothing
end

function _nforward_scatter_chunk!(grads::Tuple, inputs::Tuple, dy, start_slot::Int)
    lanes = dy isa Tuple ? dy : (dy,)
    global_slot = start_slot
    for lane_val in lanes
        offset = 0
        for (i, x) in enumerate(inputs)
            dof = _nforward_input_dof(x)
            if offset < global_slot <= offset + dof
                local_slot = global_slot - offset
                _nforward_add_slot!(grads[i], local_slot, lane_val)
                break
            end
            offset += dof
        end
        global_slot += 1
    end
    return nothing
end

function _nforward_gradient_refs(primals::Tuple, tangents::Tuple)
    return map(primals, tangents) do x, dx
        g = zero_tangent(x, dx)
        x isa Number ? Ref(g) : set_to_zero!!(g)
    end
end

_nforward_unwrap_gradient(g::Base.RefValue) = g[]
_nforward_unwrap_gradient(g) = g

@inline function _nforward_accumulate_scalar_gradient(
    g::T, local_slot::Int, v
) where {T<:IEEEFloat}
    local_slot == 1 ? g + v : g
end

@inline function _nforward_accumulate_scalar_gradient(
    g::Complex{T}, local_slot::Int, v
) where {T<:IEEEFloat}
    if local_slot == 1
        return g + complex(v, zero(T))
    elseif local_slot == 2
        return g + complex(zero(T), v)
    end
    return g
end

@inline function _nforward_real_dot(a::T, b::T) where {T<:IEEEFloat}
    return a * b
end

@inline function _nforward_real_dot(a::Complex{T}, b::Complex{T}) where {T<:IEEEFloat}
    return real(conj(a) * b)
end

@inline _nforward_zero_output_tangent(y::T, ::Val{1}) where {T<:IEEEFloat} = zero(T)
@inline _nforward_zero_output_tangent(y::Complex{T}, ::Val{1}) where {T<:IEEEFloat} = zero(
    y
)
@inline function _nforward_zero_output_tangent(y::T, ::Val{N}) where {T<:IEEEFloat,N}
    return ntuple(_ -> zero(T), Val(N))
end
@inline function _nforward_zero_output_tangent(
    y::Complex{T}, ::Val{N}
) where {T<:IEEEFloat,N}
    return ntuple(_ -> zero(y), Val(N))
end

@inline function _nforward_zero_output_tangent(
    y::AbstractArray{T}, ::Val{1}
) where {T<:IEEEFloat}
    return zero_tangent(y)
end
@inline function _nforward_zero_output_tangent(
    y::AbstractArray{Complex{T}}, ::Val{1}
) where {T<:IEEEFloat}
    return zero_tangent(y)
end
@inline function _nforward_zero_output_tangent(
    y::AbstractArray{T}, ::Val{N}
) where {T<:IEEEFloat,N}
    return zeros(T, size(y)..., N)
end
@inline function _nforward_zero_output_tangent(
    y::AbstractArray{Complex{T}}, ::Val{N}
) where {T<:IEEEFloat,N}
    return zeros(Complex{T}, size(y)..., N)
end

@inline _nforward_scalar_lane(dy::T, ::Val{1}, ::Val{1}) where {T<:IEEEFloat} = dy
@inline _nforward_scalar_lane(dy::Complex{T}, ::Val{1}, ::Val{1}) where {T<:IEEEFloat} = dy
@inline function _nforward_scalar_lane(dy::NTuple{N,T}, ::Val{N}, ::Val{K}) where {N,T,K}
    return dy[K]
end
@inline _nforward_scalar_lane(dy::T, ::Val{1}, ::Int) where {T<:IEEEFloat} = dy
@inline _nforward_scalar_lane(dy::Complex{T}, ::Val{1}, ::Int) where {T<:IEEEFloat} = dy
@inline function _nforward_scalar_lane(dy::NTuple{N,T}, ::Val{N}, lane::Int) where {N,T}
    return dy[lane]
end

function _nforward_contract_output(ȳ::T, dy::T) where {T<:IEEEFloat}
    return (_nforward_real_dot(ȳ, dy),)
end

function _nforward_contract_output(ȳ::Complex{T}, dy::Complex{T}) where {T<:IEEEFloat}
    return (_nforward_real_dot(ȳ, dy),)
end

function _nforward_contract_output(ȳ::T, dy::NTuple{N,T}) where {T<:IEEEFloat,N}
    return ntuple(k -> _nforward_real_dot(ȳ, dy[k]), Val(N))
end

function _nforward_contract_output(
    ȳ::Complex{T}, dy::NTuple{N,Complex{T}}
) where {T<:IEEEFloat,N}
    return ntuple(k -> _nforward_real_dot(ȳ, dy[k]), Val(N))
end

function _nforward_contract_output(ȳ::A, dy::A) where {T<:IEEEFloat,A<:AbstractArray{T}}
    acc = zero(T)
    @inbounds for I in CartesianIndices(ȳ)
        acc += _nforward_real_dot(ȳ[I], dy[I])
    end
    return (acc,)
end

function _nforward_contract_output(
    ȳ::A, dy::A
) where {T<:Complex{<:IEEEFloat},A<:AbstractArray{T}}
    acc = zero(real(eltype(ȳ)))
    @inbounds for I in CartesianIndices(ȳ)
        acc += _nforward_real_dot(ȳ[I], dy[I])
    end
    return (acc,)
end

function _nforward_contract_output(
    ȳ::A, dy::B
) where {T<:IEEEFloat,A<:AbstractArray{T},B<:AbstractArray{T}}
    ndims(dy) == ndims(ȳ) + 1 || _nforward_output_error(dy)
    size(dy)[1:(end - 1)] == size(ȳ) || _nforward_output_error(dy)
    N = size(dy, ndims(dy))
    return ntuple(Val(N)) do k
        acc = zero(T)
        @inbounds for I in CartesianIndices(ȳ)
            idx = Tuple(I)
            acc += _nforward_real_dot(ȳ[I], dy[idx..., k])
        end
        acc
    end
end

function _nforward_contract_output(
    ȳ::A, dy::B
) where {T<:Complex{<:IEEEFloat},A<:AbstractArray{T},B<:AbstractArray{T}}
    ndims(dy) == ndims(ȳ) + 1 || _nforward_output_error(dy)
    size(dy)[1:(end - 1)] == size(ȳ) || _nforward_output_error(dy)
    N = size(dy, ndims(dy))
    Treal = real(eltype(ȳ))
    return ntuple(Val(N)) do k
        acc = zero(Treal)
        @inbounds for I in CartesianIndices(ȳ)
            idx = Tuple(I)
            acc += _nforward_real_dot(ȳ[I], dy[idx..., k])
        end
        acc
    end
end

function _nforward_contract_output(ȳ, dy)
    _nforward_output_error(dy)
end

#
# Reverse execution
#
# `NForwardPullback` is a concrete callable struct rather than a closure so direct
# `nforward_build_rrule(...)(...)` calls can stay allocation-free on the scalar path.
# The pullback still carries the cached primals / tangents / output fdata needed to rerun
# chunked NDual passes during the reverse sweep.

"""
    _nforward_rrule_call(f, x, chunk_size_or_val)

Run the shared reverse-mode `nforward` path: evaluate the primal on the runtime primals,
wrap the result in the `CoDual` shape expected by `rrule!!`, and build the pullback that
reruns chunked NDual passes during the reverse sweep.
"""
@inline function _nforward_rrule_call(f, x::Tuple{Vararg{CoDual,M}}, ::Val{N}) where {M,N}
    primals = map(primal, x)
    tangents = map(tangent, x)
    y_primal = f(primals...)
    _nforward_is_supported_primal(typeof(y_primal)) || _nforward_output_error(y_primal)
    y = CoDual(y_primal, fdata(zero_tangent(y_primal)))
    return y, _nforward_pullback(f, primals, tangents, tangent(y), Val(N))
end

@inline function _nforward_rrule_call(f, x::Tuple, chunk_size::Integer)
    return _nforward_rrule_call(f, x, Val(_nforward_check_chunk_size(chunk_size)))
end

"""
    _nforward_pullback(rule, primals, tangents, y_fdata)

Package the state needed for a later reverse sweep into an `NForwardPullback`.
"""
function _nforward_pullback(f, primals::Tuple, tangents::Tuple, y_fdata, ::Val{N}) where {N}
    return NForwardPullback{typeof(f),N,typeof(primals),typeof(tangents),typeof(y_fdata)}(
        f, primals, tangents, y_fdata
    )
end

"""
    _nforward_scalar_gradient_rdata(pb, y_rdata)

Compute scalar-input reverse data for the specialized scalar pullback path.
"""
function _nforward_scalar_gradient_rdata(
    pb::NForwardPullback{F,N,Tuple{T},Tuple{NoFData},Y}, y_rdata
) where {F,N,T<:Number,Y}
    ȳ = tangent(pb.y_fdata, y_rdata)
    x = pb.primals[1]
    g = zero_tangent(x, pb.tangents[1])
    total_dof = _nforward_input_dof(x)
    for start_slot in 1:N:total_dof
        tangents = (_nforward_seed_tangent(x, N, start_slot, 0),)
        _, dy = _nforward_eval(pb.f, pb.primals, tangents, Val(N))
        lane_vals = _nforward_contract_output(ȳ, dy)
        global_slot = start_slot
        for lane_val in lane_vals
            g = _nforward_accumulate_scalar_gradient(g, global_slot, lane_val)
            global_slot += 1
        end
    end
    return rdata(g)
end

"""
    (pb::NForwardPullback)(y_rdata)

Scalar-input pullback specialization returning reverse data without the generic scatter path.
"""
function (pb::NForwardPullback{F,N,Tuple{T},Tuple{NoFData},Y})(
    y_rdata
) where {F,N,T<:Number,Y}
    return (rdata(zero_tangent(pb.f)), _nforward_scalar_gradient_rdata(pb, y_rdata))
end

"""
    (pb::NForwardPullback)(y_rdata)

Generic `nforward` pullback that reruns chunked NDual passes and scatters VJP contributions
into the cached gradient containers.
"""
function (pb::NForwardPullback{F,N})(y_rdata) where {F,N}
    ȳ = tangent(pb.y_fdata, y_rdata)
    grads = _nforward_gradient_refs(pb.primals, pb.tangents)
    total_dof = sum(_nforward_input_dof, pb.primals)
    for start_slot in 1:N:total_dof
        offset = 0
        tangents = map(pb.primals) do x
            t = _nforward_seed_tangent(x, N, start_slot, offset)
            offset += _nforward_input_dof(x)
            t
        end
        _, dy = _nforward_eval(pb.f, pb.primals, tangents, Val(N))
        lane_vals = _nforward_contract_output(ȳ, dy)
        _nforward_scatter_chunk!(grads, pb.primals, lane_vals, start_slot)
    end
    return tuple(
        rdata(zero_tangent(pb.f)), map(g -> rdata(_nforward_unwrap_gradient(g)), grads)...
    )
end

#
# Forward evaluation pipeline
#
# `_nforward_eval` is the high-level lifted evaluation step used by both the forward rule and
# the reverse pullback. The lift/extract helpers below are the data-conversion pieces it uses.

"""
    _nforward_eval(f, primals, tangents, ::Val{N})

Evaluate `f` on NDual-lifted primals and extract both primal output and chunked tangent data.
"""
function _nforward_eval(f, primals::Tuple, tangents::Tuple, ::Val{N}) where {N}
    lifted = map(
        (x, dx) -> _nforward_lift(_nforward_check_primal(x), dx, Val(N)), primals, tangents
    )
    return _nforward_extract(f(lifted...), Val(N))
end

"""
    _nforward_eval(f, primals::Tuple{<:Number}, tangents, ::Val{N})

Scalar-input specialization of `_nforward_eval` that avoids tuple-based lifting overhead.
"""
function _nforward_eval(
    f, primals::Tuple{T}, tangents::Tuple{D}, ::Val{N}
) where {T<:Number,D,N}
    lifted = _nforward_lift(_nforward_check_primal(primals[1]), tangents[1], Val(N))
    return _nforward_extract(f(lifted), Val(N))
end

#
# Forward lift/extract helpers
#
# These utilities translate between Mooncake tangent layouts and NDual-based lifted values.

@inline function _nforward_scalar_partials(x::T, dx, ::Val{N}) where {T<:IEEEFloat,N}
    if N == 1 && dx isa Real
        return (T(dx),)
    elseif dx isa Tuple && length(dx) == N
        return ntuple(i -> T(dx[i]), Val(N))
    elseif dx isa AbstractVector && length(dx) == N
        return ntuple(i -> T(dx[i]), Val(N))
    end
    throw(
        ArgumentError(
            "Expected scalar tangent for $(T) to be a Real when chunk_size == 1, or " *
            "a length-$N tuple/vector of reals. Got $(typeof(dx)).",
        ),
    )
end

@inline function _nforward_complex_partials(
    x::Complex{T}, dx, ::Val{N}
) where {T<:IEEEFloat,N}
    if N == 1 && dx isa Complex
        return (T(real(dx)),), (T(imag(dx)),)
    elseif dx isa Tuple && length(dx) == N
        return ntuple(i -> T(real(dx[i])), Val(N)), ntuple(i -> T(imag(dx[i])), Val(N))
    elseif dx isa AbstractVector && length(dx) == N
        return ntuple(i -> T(real(dx[i])), Val(N)), ntuple(i -> T(imag(dx[i])), Val(N))
    end
    throw(
        ArgumentError(
            "Expected complex scalar tangent for $(typeof(x)) to be a Complex when " *
            "chunk_size == 1, or a length-$N tuple/vector of complex values. Got $(typeof(dx)).",
        ),
    )
end

@inline function _nforward_array_tangent_dims(x::AbstractArray, ::Val{N}) where {N}
    return (size(x)..., N)
end

@inline function _nforward_check_array_tangent(
    x::AbstractArray, dx::AbstractArray, ::Val{N}
) where {N}
    if N == 1 && size(dx) == size(x)
        return :plain
    elseif size(dx) == _nforward_array_tangent_dims(x, Val(N))
        return :chunked
    end
    throw(
        ArgumentError(
            "Expected array tangent for input of size $(size(x)) to have size $(size(x)) " *
            "when chunk_size == 1, or size $(_nforward_array_tangent_dims(x, Val(N))) " *
            "otherwise. Got size $(size(dx)).",
        ),
    )
end

function _nforward_lift(x::T, dx, ::Val{N}) where {T<:IEEEFloat,N}
    return NDual{T,N}(x, _nforward_scalar_partials(x, dx, Val(N)))
end

function _nforward_lift(x::Complex{T}, dx, ::Val{N}) where {T<:IEEEFloat,N}
    re, im = _nforward_complex_partials(x, dx, Val(N))
    return Complex(NDual{T,N}(real(x), re), NDual{T,N}(imag(x), im))
end

function _nforward_lift(x::A, dx::AbstractArray, ::Val{N}) where {ET,A<:AbstractArray{ET},N}
    _nforward_is_supported_scalar(ET) || _nforward_input_error(x)
    tangent_layout = _nforward_check_array_tangent(x, dx, Val(N))
    out = similar(x, ET <: IEEEFloat ? NDual{ET,N} : Complex{NDual{ET.parameters[1],N}})
    @inbounds for I in CartesianIndices(x)
        idx = Tuple(I)
        if tangent_layout === :plain
            out[I] = _nforward_lift(x[I], dx[I], Val(N))
        else
            if ET <: IEEEFloat
                out[I] = NDual{ET,N}(x[I], ntuple(k -> ET(dx[idx..., k]), Val(N)))
            else
                T = ET.parameters[1]
                out[I] = Complex(
                    NDual{T,N}(real(x[I]), ntuple(k -> T(real(dx[idx..., k])), Val(N))),
                    NDual{T,N}(imag(x[I]), ntuple(k -> T(imag(dx[idx..., k])), Val(N))),
                )
            end
        end
    end
    return out
end

@inline function _nforward_extract_scalar(d::NDual{T,N}, ::Val{N}) where {T,N}
    return if N == 1
        ndual_value(d), ndual_partial(d, 1)
    else
        ndual_value(d), ntuple(k -> ndual_partial(d, k), Val(N))
    end
end

@inline function _nforward_extract_scalar(z::Complex{NDual{T,N}}, ::Val{N}) where {T,N}
    primal = complex(ndual_value(real(z)), ndual_value(imag(z)))
    tangent = if N == 1
        complex(ndual_partial(real(z), 1), ndual_partial(imag(z), 1))
    else
        ntuple(k -> complex(ndual_partial(real(z), k), ndual_partial(imag(z), k)), Val(N))
    end
    return primal, tangent
end

function _nforward_extract(y::NDual{T,N}, ::Val{N}) where {T,N}
    return _nforward_extract_scalar(y, Val(N))
end

function _nforward_extract(y::Complex{NDual{T,N}}, ::Val{N}) where {T,N}
    return _nforward_extract_scalar(y, Val(N))
end

function _nforward_extract(y::AbstractArray{<:NDual{T,N}}, ::Val{N}) where {T,N}
    primal = similar(y, T)
    tangent = N == 1 ? similar(y, T) : similar(y, T, size(y)..., N)
    @inbounds for I in CartesianIndices(y)
        primal[I] = ndual_value(y[I])
        idx = Tuple(I)
        if N == 1
            tangent[I] = ndual_partial(y[I], 1)
        else
            for k in 1:N
                tangent[idx..., k] = ndual_partial(y[I], k)
            end
        end
    end
    return primal, tangent
end

function _nforward_extract(
    y::AbstractArray{<:Complex{NDual{Treal,N}}}, ::Val{N}
) where {Treal,N}
    T = Complex{Treal}
    primal = similar(y, T)
    tangent = N == 1 ? similar(y, T) : similar(y, T, size(y)..., N)
    @inbounds for I in CartesianIndices(y)
        primal[I] = complex(ndual_value(real(y[I])), ndual_value(imag(y[I])))
        idx = Tuple(I)
        if N == 1
            tangent[I] = complex(ndual_partial(real(y[I]), 1), ndual_partial(imag(y[I]), 1))
        else
            for k in 1:N
                tangent[idx..., k] = complex(
                    ndual_partial(real(y[I]), k), ndual_partial(imag(y[I]), k)
                )
            end
        end
    end
    return primal, tangent
end

function _nforward_extract(y::T, ::Val{N}) where {T<:IEEEFloat,N}
    return y, _nforward_zero_output_tangent(y, Val(N))
end

function _nforward_extract(y::Complex{T}, ::Val{N}) where {T<:IEEEFloat,N}
    return y, _nforward_zero_output_tangent(y, Val(N))
end

function _nforward_extract(y::AbstractArray{T}, ::Val{N}) where {T<:IEEEFloat,N}
    return y, _nforward_zero_output_tangent(y, Val(N))
end

function _nforward_extract(y::AbstractArray{Complex{T}}, ::Val{N}) where {T<:IEEEFloat,N}
    return y, _nforward_zero_output_tangent(y, Val(N))
end

function _nforward_extract(y, ::Val)
    _nforward_output_error(y)
end

#
# Cache specs
#
# Cached entrypoints use these shape/type descriptors before dispatching into the shared
# forward or reverse pipelines.

@inline function _nforward_sig(x)
    if x isa Function
        return (kind=:function, type=typeof(x), size=())
    elseif x isa AbstractArray
        return (kind=:array, type=typeof(x), size=size(x))
    else
        return (kind=:scalar, type=typeof(x), size=())
    end
end

function _nforward_check_sigs(sigs::Tuple, fx::Tuple)
    length(sigs) == length(fx) || throw(ArgumentError("nforward cache arity mismatch."))
    for (sig, x) in zip(sigs, fx)
        typeof(x) == sig.type || throw(
            ArgumentError(
                "nforward cache type mismatch: expected $(sig.type), got $(typeof(x))."
            ),
        )
        x isa AbstractArray || continue
        size(x) == sig.size || throw(
            ArgumentError(
                "nforward cache size mismatch: expected $(sig.size), got $(size(x))."
            ),
        )
    end
    return nothing
end

@inline _nforward_function_gradient(f::CoDual) = tangent(fdata(tangent(f)), NoRData())

@inline function _nforward_scalar_value_and_gradient(
    f_runtime, f::CoDual, x::CoDual{T}, ::Val{N}
) where {T<:IEEEFloat,N}
    _nforward_check_function_tangent(tangent(f))
    x_primal = _nforward_check_primal(primal(x))
    seed = _nforward_seed_tangent(x_primal, N, 1, 0)
    y, dy = _nforward_extract(f_runtime(_nforward_lift(x_primal, seed, Val(N))), Val(N))
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    x_grad = tangent(fdata(tangent(x)), _nforward_scalar_lane(dy, Val(N), Val(1)))
    return y, (_nforward_function_gradient(f), x_grad)
end

@inline function _nforward_scalar_value_and_gradient(
    f_runtime, f::CoDual, x::CoDual{T}, chunk_size::Integer
) where {T<:IEEEFloat}
    return _nforward_scalar_value_and_gradient(
        f_runtime, f, x, Val(_nforward_check_chunk_size(chunk_size))
    )
end

#
# Cached array/vector scalar fast paths
#
# The array and vector entrypoints stay separate so the vector path can keep its typed
# workspace specialization without obscuring the generic dense-array path.

function _nforward_array_scalar_value_and_gradient(
    f_runtime, x::CoDual{A}, buf::Base.RefValue, ::Val{C}
) where {C,T<:IEEEFloat,N,A<:Array{T,N}}
    x_primal = _nforward_check_primal(primal(x))
    grad = set_to_zero!!(tangent(x))
    seed, lifted = _nforward_array_scalar_bufs!(buf, x_primal, Val(C))

    y = zero(T)
    for start_slot in 1:C:length(x_primal)
        _nforward_seed_chunk!(seed, x_primal, start_slot)
        _nforward_lift!(lifted, x_primal, seed, Val(C))
        y, lane_vals = _nforward_scalar_lanes(f_runtime(lifted), T, Val(C))
        global_slot = start_slot
        for lane_val in lane_vals
            global_slot > length(x_primal) && break
            grad[global_slot] += lane_val
            global_slot += 1
        end
    end

    return y, grad
end

function _nforward_vector_scalar_value_and_gradient(
    f_runtime, x::CoDual{Vector{T}}, buf::Base.RefValue{Union{Nothing,Tworkspace}}, ::Val{C}
) where {
    T<:IEEEFloat,
    C,
    Tworkspace<:NamedTuple{(:seed, :lifted),Tuple{Matrix{T},Vector{NDual{T,C}}}},
}
    x_primal = _nforward_check_primal(primal(x))
    grad = set_to_zero!!(tangent(x))
    seed, lifted = _nforward_vector_scalar_bufs!(buf, x_primal, Val(C))

    y = zero(T)
    for start_slot in 1:C:length(x_primal)
        _nforward_seed_chunk!(seed, x_primal, start_slot)
        _nforward_lift!(lifted, x_primal, seed, Val(C))
        y, lane_vals = _nforward_scalar_lanes(f_runtime(lifted), T, Val(C))
        global_slot = start_slot
        @inbounds for lane_val in lane_vals
            global_slot > length(x_primal) && break
            grad[global_slot] += lane_val
            global_slot += 1
        end
    end

    return y, grad
end

#
# Cached array/vector scalar fast-path helpers
#
# Workspace caching and chunk lifting stay split between generic arrays and typed vectors to
# preserve inference and steady-state allocation behavior.

_nforward_buf_ref(sig, ::Val{nothing}) = Ref{Any}(nothing)
_nforward_buf_ref(sig, ::Val) = Ref{Any}(nothing)

function _nforward_buf_ref(::Type{Tuple{F,Vector{T}}}, ::Val{C}) where {F,T<:IEEEFloat,C}
    Tworkspace = NamedTuple{(:seed, :lifted),Tuple{Matrix{T},Vector{NDual{T,C}}}}
    return Ref{Union{Nothing,Tworkspace}}(nothing)
end

function _nforward_array_scalar_bufs!(
    buf::Base.RefValue, x::Array{T,N}, ::Val{C}
) where {T<:IEEEFloat,N,C}
    Tlift = NDual{T,C}
    Tworkspace = NamedTuple{(:seed, :lifted),Tuple{Array{T,N + 1},Array{Tlift,N}}}
    ws = buf[]
    expected_seed_size = (size(x)..., C)
    if !(ws isa Tworkspace) ||
        size(ws.seed) ≠ expected_seed_size ||
        size(ws.lifted) ≠ size(x)
        ws = Tworkspace((zeros(T, expected_seed_size), Array{Tlift}(undef, size(x))))
        buf[] = ws
    end
    typed_ws = ws::Tworkspace
    return typed_ws.seed, typed_ws.lifted
end

function _nforward_vector_scalar_bufs!(
    buf::Base.RefValue{Union{Nothing,Tworkspace}}, x::Vector{T}, ::Val{C}
) where {
    T<:IEEEFloat,
    C,
    Tworkspace<:NamedTuple{(:seed, :lifted),Tuple{Matrix{T},Vector{NDual{T,C}}}},
}
    ws = buf[]
    if isnothing(ws) || size(ws.seed) != (length(x), C)
        ws = Tworkspace((zeros(T, length(x), C), Vector{NDual{T,C}}(undef, length(x))))
        buf[] = ws
    end
    typed_ws = ws::Tworkspace
    return typed_ws.seed, typed_ws.lifted
end

function _nforward_seed_chunk!(
    seed::Array{T}, x::Array{T}, start_slot::Int
) where {T<:IEEEFloat}
    fill!(seed, zero(T))
    cart_inds = CartesianIndices(x)
    chunk_size = size(seed, ndims(seed))
    for lane in 1:chunk_size
        global_slot = start_slot + lane - 1
        global_slot > length(x) && break
        idx = Tuple(cart_inds[global_slot])
        seed[idx..., lane] = one(T)
    end
    return seed
end

function _nforward_lift!(
    out::Array{NDual{T,C}}, x::Array{T}, dx::Array{T}, ::Val{C}
) where {T<:IEEEFloat,C}
    @inbounds for I in CartesianIndices(x)
        idx = Tuple(I)
        out[I] = NDual{T,C}(x[I], ntuple(k -> dx[idx..., k], Val(C)))
    end
    return out
end

"""
    _nforward_scalar_lanes(y_raw, ::Type{T}, ::Val{C})

Decode a scalar-output `nforward` chunk evaluation into its primal value plus the `C`
directional derivatives carried by that chunk. Constant outputs are treated as zero-tangent
outputs, so this helper works for both NDual-carrying and plain scalar results.
"""
@inline function _nforward_scalar_lanes(
    y_raw::NDual{T,C}, ::Type{T}, ::Val{C}
) where {T<:IEEEFloat,C}
    return ndual_value(y_raw), ntuple(k -> ndual_partial(y_raw, k), Val(C))
end

@inline function _nforward_scalar_lanes(y_raw, ::Type{T}, ::Val{C}) where {T<:IEEEFloat,C}
    y, dy = _nforward_extract(y_raw, Val(C))
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    return y, ntuple(k -> _nforward_scalar_lane(dy, Val(C), Val(k)), Val(C))
end
