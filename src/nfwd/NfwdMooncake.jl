module NfwdMooncake

import ..Mooncake
using Base: IEEEFloat
using ..Nfwd
import ..Mooncake:
    @unstable,
    CoDual,
    Dual,
    ForwardCache,
    NfwdCache,
    NoFData,
    NoRData,
    NoTangent,
    NTangent,
    __value_and_gradient!!,
    __verify_sig,
    _chunk_pack_tangent,
    _fcache_derivative_chunked!!,
    _typeof,
    _fcache_derivative_chunked_loop!!,
    fdata,
    primal,
    rdata,
    tangent,
    throw_val_and_grad_ret_type_error,
    tuple_map,
    value_and_derivative!!,
    value_and_gradient!!,
    verify_fwds_inputs,
    zero_tangent

# ── nfwd: NDual-backed forward-mode engine ────────────────────────────────────────
# `nfwd` evaluates code by lifting inputs into `NDual`s and running the primal
# function directly on those lifted values. It does not reuse Mooncake's `frule!!`
# (aka ir-based forward) path, even when `chunk_size == 1`.
#
# ── File layout ────────────────────────────────────────────────────────────────────
# This file is organized as:
# - core types
# - public rule entrypoints
# - shared validation and layout helpers
# - reverse accumulation and execution
# - forward evaluation pipeline
# - cache spec checks
# - cached scalar/array fast paths
#
# ── High-level interfaces ──────────────────────────────────────────────────────────
#   build_frule(f, x...; chunk_size)
#     returns `Rule`
#     consumed via `rule(f::Dual, x::Dual...)`
#     obeys the standard `frule!!` interface
#     also accepts `sig::Type{<:Tuple}` for signature-based construction
#
#   build_rrule(f, x...; chunk_size)
#     returns `RRule`
#     consumed via `rule(f::CoDual, x::CoDual...)`
#     obeys the standard `rrule!!` interface
#     also accepts `sig::Type{<:Tuple}` for signature-based construction
#
# ── Constraints ────────────────────────────────────────────────────────────────────
# - `chunk_size` is global across the whole call
# - supported primals are IEEE float scalars, complex IEEE float scalars, dense arrays
#   with those element types, and tuples thereof
# - rule construction requires stateless callables (singleton callable types)
# - `friendly_tangents=true`, `debug_mode=true`, and differentiation with respect to `f`
#   are intentionally unsupported here
#
# ── Primitive Reverse-Mode Example ────────────────────────────────────────────────
# `nfwd` can also be used to define reverse-mode rules, and is especially useful
# when dual-number forward differentiation is more compiler-friendly than IR-transform-
# based AD, for example when differentiating through CUDA kernels. It also often has
# significantly lower compilation latency:
#   f(x) = sum(abs2, x)
#   sig = Tuple{typeof(f),Vector{Float64}}
#   Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode sig
#   Mooncake.build_primitive_rrule(::Type{sig}) =
#       Mooncake.NfwdMooncake.build_rrule(sig; chunk_size=4)
#
#
# Core types
#
# High-level rule/cache objects first. The implementation details they rely on are defined
# in later sections.

# ── Experimental chunked IR path ──────────────────────────────────────────────────
# This path does not generate `f(::Vector{NDual{...}})`-style overloads and it does not
# introduce lifted primal structs. Instead it reuses Mooncake's IR-based forward transform
# and changes only the tangent representation used inside that transform.
#
# The key idea is:
# - keep the primal in `Dual{P,...}` unchanged, so dispatch still sees the original
#   argument types (for example `Vector{Float64}` or `Box64`);
# - reuse Mooncake's existing tangent tree shape for arrays, tuples, and structs; and
# - replace only differentiable leaf tangents with packed `NDual` lanes.
#
# Concretely, the transform builds `Dual{P, packed_tangent_type(P)}` rather than the
# usual `Dual{P, tangent_type(P)}`. For a tangent like
#   `Tangent((; a=1.0, b=[2.0, 3.0]))`
# the packed form is still a `Tangent` with the same field layout, but `a` becomes one
# `NDual` and `b` becomes an array of `NDual`s.
#
# Importantly, this does *not* semantically store a second copy of the primal inside the
# tangent. The real primal remains `primal(x)`. Packed scalar tangent leaves are represented
# as `NDual(zero(T), partials)` and only the partial lanes are used. There is still some
# storage overhead because `NDual` has a `value` slot and because tangents already mirror
# the primal's container structure, but the tangent does not carry another logical primal.
#
# This is why the path fixes the two motivating failures:
# - concrete dispatch is preserved because methods still receive the original primal types;
# - concrete struct construction works because the primal program still constructs the
#   original struct, while Mooncake's `_new_` rule constructs a matching packed tangent
#   separately, rather than requiring the primal struct itself to store `NDual`s.
#
# The builder below:
# 1. packs ordinary Mooncake tangents / `NTangent` bundles into the packed tangent tree,
# 2. runs the ordinary derived `frule`, and
# 3. unpacks the result back to Mooncake's public tangent representation.
#
# Scope note: this is currently an experimental forward-mode `frule` path only, not a full
# replacement for the existing `nfwd` reverse/cache machinery.
# `ChunkedIRMode{N}` selects the chunked IR path: derive a forward rule from IR, keep
# primal values unchanged, and store the `N` derivative lanes only in the tangent.
struct ChunkedIRMode{N} end

# `ChunkedIRRule` is the public rule object returned by `build_chunked_frule` for the
# chunked IR path.
# It lives here, rather than in `forward_mode.jl`, because the raw derived IR rule does not
# know about Mooncake's chunked public tangent interface (`NTangent`). This wrapper adds
# the chunk pack/unpack boundary that converts between public chunked tangents and the
# internal `NDual`-leaf tangent tree used by the chunked IR path.
struct ChunkedIRRule{sig,N,R}
    rule::R
end

# `ChunkedNDualRule` adapts a primitive nfwd rule to the packed tangent interface used by
# the chunked frontend. It is used only at primitive boundaries whose primal argument
# types are all nfwd-supported.
struct ChunkedNDualRule{sig,N,R}
    rule::R
end

@inline rule_chunk_size(::Type{<:ChunkedIRRule{sig,N}}) where {sig,N} = N
@inline rule_chunk_size(::Type{<:ChunkedNDualRule{sig,N}}) where {sig,N} = N
@inline (rule::ChunkedIRRule)(args...) = rule.rule(args...)

function build_chunked_frule(
    sig::Type{<:Tuple}; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    resolved = _chunked_resolve_rule_chunk_size(sig, chunk_size)
    inner = Mooncake.build_frule(
        Mooncake.get_interpreter(Mooncake.ForwardMode),
        sig;
        debug_mode,
        silence_debug_messages,
        tangent_mode=ChunkedIRMode{resolved}(),
    )
    return ChunkedIRRule{sig,resolved,typeof(inner)}(inner)
end

function build_chunked_frule(
    f, x...; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    return build_chunked_frule(
        typeof((f, x...)); chunk_size, debug_mode, silence_debug_messages
    )
end

# Design note:
# This file exposes two forward-mode interfaces, but they are intentionally separate:
#
# 1. Chunked IR path.
#    Pathway:
#      `build_chunked_frule`
#      -> `Mooncake.build_frule(...; tangent_mode=ChunkedIRMode)`
#      -> derived IR executes on ordinary primal values, while primitive calls may select
#         `ChunkedNDualRule` when their primal signature is nfwd-supported.
#    In this path, the primal values keep their original Julia types and only the tangent
#    is changed. The tangent still uses Mooncake's normal structural tangent layout for
#    tuples, arrays, and structs, but differentiable leaf tangents are packed into
#    `NDual` lanes. This is the semantics-preserving path: Julia dispatch still sees the
#    same primal argument types as the original program.
#
# 2. Nfwd path.
#    Pathway:
#      `NfwdMooncake.build_frule` / `NfwdMooncake.build_rrule`
#      -> execute Julia code directly on NDual-lifted primals.
#    This path is explicit. It is still useful when callers intentionally want `nfwd`
#    semantics and know their code is compatible with NDual replay.
#
# The key correctness rule is that the chunked frontend never lowers derived user code to
# NDual replay automatically. Even if the top-level call appears safe, dispatch can still
# change at nested calls or for different chunk widths. Primitive boundaries are the one
# safe exception: there is no inner Julia call graph to replay, so for primitive signatures
# whose primal argument types are nfwd-supported we default to the existing nfwd rule.
@inline _nfwd_supported_primal_type(::Type{<:IEEEFloat}) = true
@inline _nfwd_supported_primal_type(::Type{<:Complex{<:IEEEFloat}}) = true
@inline _nfwd_supported_primal_type(::Type{<:Array{T}}) where {T} = Nfwd._nfwd_is_supported_scalar(
    T
)
@generated function _nfwd_supported_primal_type(::Type{T}) where {T<:Tuple}
    checks = map(p -> _nfwd_supported_primal_type(p), T.parameters)
    return all(checks) ? :(true) : :(false)
end
@inline _nfwd_supported_primal_type(::Type) = false

function _nfwd_supports_primitive_sig(sig::Type{<:Tuple})
    try
        Nfwd._nfwd_check_callable_sig(sig)
    catch
        return false
    end
    return all(_nfwd_supported_primal_type, Base.tail(sig.parameters))
end

function Mooncake._fwd_primitive_rule(
    ::ChunkedIRMode{N},
    interp,
    sig::Type{<:Tuple};
    debug_mode=false,
    silence_debug_messages=true,
) where {N}
    _nfwd_supports_primitive_sig(sig) || return nothing
    inner = build_frule(sig; chunk_size=N, silence_debug_messages)
    return ChunkedNDualRule{sig,N,typeof(inner)}(inner)
end

@inline _nfwd_packed_tangent_type(::Val{N}, ::Type{P}) where {N,P} = _nfwd_pack_tangent_storage_type(
    Val(N), Mooncake.tangent_type(P)
)
@inline _nfwd_packed_tangent_type(::Val, ::Type{<:Function}) = NoTangent
@inline _nfwd_pack_tangent_storage_type(::Val, ::Type{Any}) = Any
@inline _nfwd_pack_tangent_storage_type(::Val, ::Type{NoTangent}) = NoTangent
@inline _nfwd_pack_tangent_storage_type(::Val{N}, ::Type{T}) where {N,T<:IEEEFloat} = NDual{
    T,N
}
@inline _nfwd_pack_tangent_storage_type(::Val{N}, ::Type{Complex{T}}) where {N,T<:IEEEFloat} = Complex{
    NDual{T,N}
}
@inline _nfwd_pack_tangent_storage_type(::Val{N}, ::Type{Array{T,Nd}}) where {N,T,Nd} = Array{
    _nfwd_pack_tangent_storage_type(Val(N), T),Nd
}
@inline _nfwd_pack_tangent_storage_type(::Val{N}, ::Type{Mooncake.Tangent{F}}) where {N,F} = Mooncake.Tangent{
    _nfwd_pack_tangent_storage_type(Val(N), F)
}
@inline _nfwd_pack_tangent_storage_type(::Val{N}, ::Type{Mooncake.MutableTangent{F}}) where {N,F} = Mooncake.MutableTangent{
    _nfwd_pack_tangent_storage_type(Val(N), F)
}
@inline _nfwd_pack_tangent_storage_type(::Val{N}, ::Type{Mooncake.PossiblyUninitTangent{F}}) where {N,F} = Mooncake.PossiblyUninitTangent{
    _nfwd_pack_tangent_storage_type(Val(N), F)
}
@inline _nfwd_pack_tangent_storage_type(::Val, ::Type{T}) where {T} = T

@generated function _nfwd_pack_tangent_storage_type(::Val{N}, ::Type{T}) where {N,T<:Tuple}
    packed = map(p -> _nfwd_pack_tangent_storage_type(Val(N), p), T.parameters)
    return Expr(:curly, :Tuple, packed...)
end

@generated function _nfwd_pack_tangent_storage_type(
    ::Val{N}, ::Type{NT}
) where {N,names,Ts,NT<:NamedTuple{names,Ts}}
    packed_ts = _nfwd_pack_tangent_storage_type(Val(N), Ts)
    return :(NamedTuple{$names,$packed_ts})
end

function _nfwd_packed_dual_type(::Val{N}, ::Type{P}) where {N,P}
    P == Union{} && return Union{}
    P == DataType && return Dual
    P isa Union && return Union{
        _nfwd_packed_dual_type(Val(N), P.a),_nfwd_packed_dual_type(Val(N), P.b)
    }
    (P isa UnionAll || P == UnionAll) && return Dual
    return isconcretetype(P) ? Dual{P,_nfwd_packed_tangent_type(Val(N), P)} : Dual
end

function Mooncake._fwd_dual_type(::ChunkedIRMode{N}, ::Type{P}) where {N,P}
    _nfwd_packed_dual_type(Val(N), P)
end
function Mooncake._fwd_zero_dual(::ChunkedIRMode{N}, x) where {N}
    Dual(x, _nfwd_pack_public_tangent(x, zero_tangent(x), Val(N)))
end
function Mooncake._fwd_uninit_dual(::ChunkedIRMode{N}, x) where {N}
    Mooncake._fwd_zero_dual(ChunkedIRMode{N}(), x)
end

@inline _nfwd_pack_struct_lane_field(t::Mooncake.Tangent, i::Int) = getfield(t.fields, i)
@inline _nfwd_pack_struct_lane_field(t::Mooncake.MutableTangent, i::Int) = getfield(
    t.fields, i
)

@inline _nfwd_chunk_cache(::Val{true}) = IdDict{Any,Any}()
@inline _nfwd_chunk_cache(::Val{false}) = Mooncake.NoCache()
@inline _nfwd_chunk_cache(::Type{T}) where {T} = _nfwd_chunk_cache(
    Mooncake.require_tangent_cache(T)
)
@inline _nfwd_cache_lookup(::Mooncake.NoCache, _) = nothing
@inline _nfwd_cache_lookup(c::IdDict{Any,Any}, key) = get(c, key, nothing)
@inline _nfwd_cache_store!(::Mooncake.NoCache, _, value) = value
@inline function _nfwd_cache_store!(c::IdDict{Any,Any}, key, value)
    c[key] = value
    return value
end

@inline function _nfwd_pack_public_tangent(x, dx, ::Val{N}) where {N}
    lanes = if dx isa NTangent
        length(dx) == N || throw(
            ArgumentError(
                "Packed chunk rule expected $N tangent lanes, got $(length(dx))."
            ),
        )
        dx.lanes
    else
        ntuple(_ -> dx, Val(N))
    end
    return _nfwd_pack_lanes(_nfwd_chunk_cache(typeof(x)), x, lanes, Val(N))
end

@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x, ::NTuple{N,NoTangent}, ::Val{N}) where {N} = NoTangent()
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::T, ::NTuple{N,NoTangent}, ::Val{N}) where {T<:IEEEFloat,N} = NoTangent()
@inline function _nfwd_pack_lanes(
    ::Mooncake.MaybeCache, x::Complex{T}, ::NTuple{N,NoTangent}, ::Val{N}
) where {T<:IEEEFloat,N}
    return NoTangent()
end
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::Array, ::NTuple{N,NoTangent}, ::Val{N}) where {N} = NoTangent()
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::Tuple, ::NTuple{N,NoTangent}, ::Val{N}) where {N} = NoTangent()
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::NamedTuple, ::NTuple{N,NoTangent}, ::Val{N}) where {N} = NoTangent()

@inline function _nfwd_pack_lanes(x::T, lanes::NTuple{N}, ::Val{N}) where {T<:IEEEFloat,N}
    return NDual{T,N}(zero(T), ntuple(k -> T(lanes[k]), Val(N)))
end
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::T, lanes::NTuple{N}, v::Val{N}) where {T<:IEEEFloat,N} = _nfwd_pack_lanes(
    x, lanes, v
)

@inline function _nfwd_pack_lanes(
    x::Complex{T}, lanes::NTuple{N}, ::Val{N}
) where {T<:IEEEFloat,N}
    return Complex(
        NDual{T,N}(zero(T), ntuple(k -> T(real(lanes[k])), Val(N))),
        NDual{T,N}(zero(T), ntuple(k -> T(imag(lanes[k])), Val(N))),
    )
end
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::Complex{T}, lanes::NTuple{N}, v::Val{N}) where {T<:IEEEFloat,N} = _nfwd_pack_lanes(
    x, lanes, v
)

function _nfwd_pack_lanes(
    c::Mooncake.MaybeCache, x::Array, lanes::NTuple{N}, ::Val{N}
) where {N}
    ET = _nfwd_packed_tangent_type(Val(N), eltype(x))
    cached = _nfwd_cache_lookup(c, x)
    !isnothing(cached) && return cached::Array{ET,ndims(x)}
    out = Array{ET,ndims(x)}(undef, size(x))
    _nfwd_cache_store!(c, x, out)
    @inbounds for I in CartesianIndices(x)
        out[I] = _nfwd_pack_lanes(c, x[I], ntuple(k -> lanes[k][I], Val(N)), Val(N))
    end
    return out
end

@inline function _nfwd_pack_lanes(
    c::Mooncake.MaybeCache, x::Tuple, lanes::NTuple{N}, ::Val{N}
) where {N}
    Mooncake.tangent_type(typeof(x)) === NoTangent && return NoTangent()
    return ntuple(
        i -> _nfwd_pack_lanes(c, x[i], ntuple(k -> lanes[k][i], Val(N)), Val(N)),
        Val(length(x)),
    )
end

@inline function _nfwd_pack_lanes(
    c::Mooncake.MaybeCache, x::NamedTuple, lanes::NTuple{N}, ::Val{N}
) where {N}
    names = fieldnames(typeof(x))
    values = ntuple(
        i -> _nfwd_pack_lanes(
            c, getfield(x, i), ntuple(k -> getfield(lanes[k], i), Val(N)), Val(N)
        ),
        Val(fieldcount(typeof(x))),
    )
    return NamedTuple{names}(values)
end

@inline function _nfwd_pack_possible_uninit_lanes(
    c::Mooncake.MaybeCache, x, lanes::NTuple{N,<:Mooncake.PossiblyUninitTangent}, ::Val{N}
) where {N}
    all(lane -> !Mooncake.is_init(lane), lanes) &&
        return Mooncake.PossiblyUninitTangent(_nfwd_packed_tangent_type(Val(N), typeof(x)))
    inner = _nfwd_pack_lanes(
        c,
        x,
        ntuple(
            k -> Mooncake.is_init(lanes[k]) ? Mooncake.val(lanes[k]) : zero_tangent(x),
            Val(N),
        ),
        Val(N),
    )
    return Mooncake.PossiblyUninitTangent(inner)
end
@inline _nfwd_pack_lanes(c::Mooncake.MaybeCache, x::T, lanes::NTuple{N,<:Mooncake.PossiblyUninitTangent}, v::Val{N}) where {T<:IEEEFloat,N} = _nfwd_pack_possible_uninit_lanes(
    c, x, lanes, v
)
@inline _nfwd_pack_lanes(c::Mooncake.MaybeCache, x::Complex{T}, lanes::NTuple{N,<:Mooncake.PossiblyUninitTangent}, v::Val{N}) where {T<:IEEEFloat,N} = _nfwd_pack_possible_uninit_lanes(
    c, x, lanes, v
)
@inline _nfwd_pack_lanes(c::Mooncake.MaybeCache, x::Array, lanes::NTuple{N,<:Mooncake.PossiblyUninitTangent}, v::Val{N}) where {N} = _nfwd_pack_possible_uninit_lanes(
    c, x, lanes, v
)
@inline _nfwd_pack_lanes(c::Mooncake.MaybeCache, x::Tuple, lanes::NTuple{N,<:Mooncake.PossiblyUninitTangent}, v::Val{N}) where {N} = _nfwd_pack_possible_uninit_lanes(
    c, x, lanes, v
)
@inline _nfwd_pack_lanes(c::Mooncake.MaybeCache, x::NamedTuple, lanes::NTuple{N,<:Mooncake.PossiblyUninitTangent}, v::Val{N}) where {N} = _nfwd_pack_possible_uninit_lanes(
    c, x, lanes, v
)
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::T, ::Tuple{}, ::Val{0}) where {T<:IEEEFloat} = NoTangent()
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::Complex{T}, ::Tuple{}, ::Val{0}) where {T<:IEEEFloat} = NoTangent()
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::Array, ::Tuple{}, ::Val{0}) = NoTangent()
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::Tuple, ::Tuple{}, ::Val{0}) = NoTangent()
@inline _nfwd_pack_lanes(::Mooncake.MaybeCache, x::NamedTuple, ::Tuple{}, ::Val{0}) = NoTangent()

@inline _nfwd_pack_struct_field_value(::Type{T}, x) where {T} = convert(T, x)
@inline _nfwd_pack_struct_field_value(::Type{Mooncake.PossiblyUninitTangent{T}}, x) where {T} = Mooncake.PossiblyUninitTangent{
    T
}(
    x
)
@inline _nfwd_pack_struct_field_value(
    ::Type{Mooncake.PossiblyUninitTangent{T}}, x::Mooncake.PossiblyUninitTangent{T}
) where {T} = x
@inline function _nfwd_pack_struct_field_value(
    ::Type{Mooncake.PossiblyUninitTangent{T}}, x::Mooncake.PossiblyUninitTangent
) where {T}
    return if Mooncake.is_init(x)
        Mooncake.PossiblyUninitTangent{T}(_nfwd_pack_struct_field_value(T, Mooncake.val(x)))
    else
        Mooncake.PossiblyUninitTangent(T)
    end
end

function _nfwd_pack_lanes(c::Mooncake.MaybeCache, x, lanes::NTuple{N}, ::Val{N}) where {N}
    Mooncake.tangent_type(typeof(x)) === NoTangent && return NoTangent()
    packed_type = _nfwd_packed_tangent_type(Val(N), typeof(x))
    packed_fields_type = Mooncake.fields_type(packed_type)
    if packed_type <: Mooncake.MutableTangent
        cached = _nfwd_cache_lookup(c, x)
        !isnothing(cached) && return cached::packed_type
        packed = packed_type()
        _nfwd_cache_store!(c, x, packed)
        values = ntuple(
            i -> _nfwd_pack_struct_field_value(
                fieldtype(packed_fields_type, i),
                _nfwd_pack_lanes(
                    c,
                    getfield(x, i),
                    ntuple(k -> _nfwd_pack_struct_lane_field(lanes[k], i), Val(N)),
                    Val(N),
                ),
            ),
            Val(fieldcount(typeof(x))),
        )
        packed.fields = packed_fields_type(values)
        return packed
    end
    values = ntuple(
        i -> _nfwd_pack_struct_field_value(
            fieldtype(packed_fields_type, i),
            _nfwd_pack_lanes(
                c,
                getfield(x, i),
                ntuple(k -> _nfwd_pack_struct_lane_field(lanes[k], i), Val(N)),
                Val(N),
            ),
        ),
        Val(fieldcount(typeof(x))),
    )
    return packed_type(NamedTuple{fieldnames(typeof(x))}(values))
end

@inline _nfwd_unpack_packed_tangent(y, dy, ::Val{1}) = _nfwd_unpack_packed_lane(
    _nfwd_chunk_cache(typeof(dy)), y, dy, Val(1)
)
@inline function _nfwd_unpack_packed_tangent(
    y::T, dy::NDual{T,1}, ::Val{1}
) where {T<:IEEEFloat}
    return Nfwd.ndual_partial(dy, 1)
end
@inline function _nfwd_unpack_packed_tangent(
    y::Complex{T}, dy::Complex{NDual{T,1}}, ::Val{1}
) where {T<:IEEEFloat}
    return Nfwd.ndual_partial(dy, 1)
end
@inline function _nfwd_unpack_packed_tangent(
    y::T, dy::NDual{T,N}, ::Val{N}
) where {T<:IEEEFloat,N}
    return NTangent(ntuple(k -> Nfwd.ndual_partial(dy, k), Val(N)))
end
@inline function _nfwd_unpack_packed_tangent(
    y::Complex{T}, dy::Complex{NDual{T,N}}, ::Val{N}
) where {T<:IEEEFloat,N}
    return NTangent(ntuple(k -> Nfwd.ndual_partial(dy, k), Val(N)))
end
@inline function _nfwd_unpack_packed_tangent(y, dy, ::Val{N}) where {N}
    return NTangent(
        ntuple(
            k -> _nfwd_unpack_packed_lane(_nfwd_chunk_cache(typeof(dy)), y, dy, Val(k)),
            Val(N),
        ),
    )
end

@inline _nfwd_unpack_packed_lane(::Mooncake.MaybeCache, y, ::NoTangent, ::Val) = NoTangent()
@inline _nfwd_unpack_packed_lane(::Mooncake.MaybeCache, y::T, dy::NDual{T,N}, ::Val{k}) where {T<:IEEEFloat,N,k} = Nfwd.ndual_partial(
    dy, k
)
@inline function _nfwd_unpack_packed_lane(
    ::Mooncake.MaybeCache, y::Complex{T}, dy::Complex{NDual{T,N}}, ::Val{k}
) where {T<:IEEEFloat,N,k}
    return Nfwd.ndual_partial(dy, k)
end

function _nfwd_unpack_packed_lane(
    c::Mooncake.MaybeCache, y::Array, dy::Array, ::Val{k}
) where {k}
    ET = Mooncake.tangent_type(eltype(y))
    cached = _nfwd_cache_lookup(c, dy)
    !isnothing(cached) && return cached::Array{ET,ndims(y)}
    out = Array{ET,ndims(y)}(undef, size(y))
    _nfwd_cache_store!(c, dy, out)
    @inbounds for I in CartesianIndices(y)
        out[I] = _nfwd_unpack_packed_lane(c, y[I], dy[I], Val(k))
    end
    return out
end

@inline function _nfwd_unpack_packed_lane(
    c::Mooncake.MaybeCache, y::Tuple, dy::Tuple, ::Val{k}
) where {k}
    return ntuple(i -> _nfwd_unpack_packed_lane(c, y[i], dy[i], Val(k)), Val(length(y)))
end

@inline function _nfwd_unpack_packed_lane(
    c::Mooncake.MaybeCache, y::NamedTuple, dy::NamedTuple, ::Val{k}
) where {k}
    names = fieldnames(typeof(y))
    values = ntuple(
        i -> _nfwd_unpack_packed_lane(c, getfield(y, i), getfield(dy, i), Val(k)),
        Val(fieldcount(typeof(y))),
    )
    return NamedTuple{names}(values)
end

function _nfwd_unpack_packed_lane(
    c::Mooncake.MaybeCache, y, dy::Mooncake.Tangent, ::Val{k}
) where {k}
    T = Mooncake.tangent_type(typeof(y))
    values = ntuple(
        i -> _nfwd_unpack_packed_lane(
            c, getfield(y, i), Mooncake.val(getfield(dy.fields, i)), Val(k)
        ),
        Val(fieldcount(typeof(y))),
    )
    return T(NamedTuple{fieldnames(typeof(y))}(values))
end

function _nfwd_unpack_packed_lane(
    c::Mooncake.MaybeCache, y, dy::Mooncake.MutableTangent, ::Val{k}
) where {k}
    T = Mooncake.tangent_type(typeof(y))
    cached = _nfwd_cache_lookup(c, dy)
    !isnothing(cached) && return cached::T
    out = T()
    _nfwd_cache_store!(c, dy, out)
    values = ntuple(
        i -> _nfwd_unpack_packed_lane(
            c, getfield(y, i), Mooncake.val(getfield(dy.fields, i)), Val(k)
        ),
        Val(fieldcount(typeof(y))),
    )
    out.fields = NamedTuple{fieldnames(typeof(y))}(values)
    return out
end

@inline function _nfwd_packed_lane_count_type(::Type{<:NDual{T,N}}) where {T,N}
    return N
end
@inline function _nfwd_packed_lane_count_type(::Type{<:Complex{NDual{T,N}}}) where {T,N}
    return N
end
@inline function _nfwd_packed_lane_count_type(::Type{<:Array{T}}) where {T}
    return _nfwd_packed_lane_count_type(T)
end
@inline function _nfwd_packed_lane_count_type(::Type{<:Mooncake.Tangent{F}}) where {F}
    return _nfwd_packed_lane_count_type(F)
end
@inline function _nfwd_packed_lane_count_type(
    ::Type{<:Mooncake.MutableTangent{F}}
) where {F}
    return _nfwd_packed_lane_count_type(F)
end
@inline function _nfwd_packed_lane_count_type(
    ::Type{<:Mooncake.PossiblyUninitTangent{F}}
) where {F}
    return _nfwd_packed_lane_count_type(F)
end

function _nfwd_merge_lane_counts(a, b)
    isnothing(a) && return b
    isnothing(b) && return a
    a == b || throw(ArgumentError("Packed tangent lanes disagree: $a vs $b."))
    return a
end

@generated function _nfwd_packed_lane_count_type(::Type{T}) where {T<:Tuple}
    counts = map(_nfwd_packed_lane_count_type, T.parameters)
    merged = nothing
    for c in counts
        merged = _nfwd_merge_lane_counts(merged, c)
    end
    return isnothing(merged) ? :(nothing) : merged
end

@generated function _nfwd_packed_lane_count_type(
    ::Type{NT}
) where {names,Ts,NT<:NamedTuple{names,Ts}}
    merged = _nfwd_packed_lane_count_type(Ts)
    return isnothing(merged) ? :(nothing) : merged
end

@inline _nfwd_packed_lane_count_type(::Type) = nothing

function value_and_derivative!!(
    rule::ChunkedIRRule{sig,N}, fx::Vararg{Tuple{Any,Any},M}
) where {sig,N,M}
    packed_args = tuple_map(
        (x, dx) -> Dual(x, _nfwd_pack_public_tangent(x, dx, Val(N))),
        tuple_map(first, fx),
        tuple_map(last, fx),
    )
    output = rule.rule(packed_args...)
    return primal(output),
    _nfwd_unpack_packed_tangent(primal(output), tangent(output), Val(N))
end

@inline _collapse_chunked_function_tangent(dx::NoTangent) = NoTangent()
@inline function _collapse_chunked_function_tangent(dx::NTangent)
    all(t -> t isa NoTangent, dx) || throw(
        ArgumentError(
            "Chunked nddual primitive rules do not support differentiating with respect to `f`.",
        ),
    )
    return NoTangent()
end
@inline _collapse_chunked_function_tangent(dx) = dx

@inline function (rule::ChunkedNDualRule{sig,N})(f::Dual, x::Vararg{Dual,M}) where {sig,N,M}
    fx = (
        (primal(f), tangent(f)),
        ntuple(
            i -> (
                primal(x[i]),
                _nfwd_unpack_packed_tangent(primal(x[i]), tangent(x[i]), Val(N)),
            ),
            Val(M),
        )...,
    )
    y, dy = value_and_derivative!!(rule.rule, fx...)
    return Dual(y, _nfwd_pack_public_tangent(y, dy, Val(N)))
end

@inline function value_and_derivative!!(
    rule::ChunkedNDualRule{sig,N}, fx::Vararg{Tuple{Any,Any},M}
) where {sig,N,M}
    normalized_fx = (
        (first(fx[1]), _collapse_chunked_function_tangent(last(fx[1]))), Base.tail(fx)...
    )
    return value_and_derivative!!(rule.rule, normalized_fx...)
end

@inline _nfwd_unpack_output_lane(yi::IEEEFloat, dyi::Tuple, ::Val{lane}) where {lane} = dyi[lane]
@inline _nfwd_unpack_output_lane(yi::IEEEFloat, dyi::IEEEFloat, ::Val{1}) = dyi
@inline _nfwd_unpack_output_lane(yi::Complex{<:IEEEFloat}, dyi::Tuple, ::Val{lane}) where {lane} = dyi[lane]
@inline _nfwd_unpack_output_lane(
    yi::Complex{<:IEEEFloat}, dyi::Complex{<:IEEEFloat}, ::Val{1}
) = dyi
@inline _nfwd_unpack_output_lane(yi::Array, dyi::Array, ::Val{lane}) where {lane} = selectdim(
    dyi, ndims(dyi), lane
)
@inline function _nfwd_unpack_output_lane(yi::Tuple, dyi::Tuple, ::Val{lane}) where {lane}
    return tuple_map((yij, dyij) -> _nfwd_unpack_output_lane(yij, dyij, Val(lane)), yi, dyi)
end

@inline function _nfwd_pack_output_tangent(y, dy, ::Val{N}) where {N}
    lanes = ntuple(k -> _nfwd_unpack_output_lane(y, dy, Val(k)), Val(N))
    return _nfwd_pack_lanes(y, lanes, Val(N))
end

@inline function _chunked_ir_small_vector_pack_buffer(
    x::Vector{T}, ::Val{N}
) where {T<:IEEEFloat,N}
    return Ref(Vector{NDual{T,N}}(undef, length(x)))
end

@inline function _chunked_ir_small_vector_value_and_derivative!(
    rule::ChunkedIRRule{sig,N},
    f,
    x::Vector{T},
    seed_buffer::Matrix{T},
    packed_ref::Base.RefValue{Vector{NDual{T,N}}},
) where {sig,N,T<:IEEEFloat}
    packed = packed_ref[]
    @inbounds for i in eachindex(x)
        packed[i] = NDual{T,N}(zero(T), ntuple(lane -> seed_buffer[i, lane], Val(N)))
    end
    output = rule.rule(Dual(f, NoTangent()), Dual(x, packed))
    return primal(output),
    _nfwd_unpack_packed_tangent(primal(output), tangent(output), Val(N))
end

# Cache-side chunked forward execution: attempt one call through a prebuilt chunked rule
# from `prepare_derivative_cache`, then fall back to the generic lane loop only when no
# rule for that chunk width is available.
@inline function _maybe_chunk_frule_chunked_ir(
    cache::ForwardCache, input_primals::Tuple, input_tangents::Tuple, ::Val{N}
) where {N}
    fastpath = cache.chunkcache
    isnothing(fastpath) && return nothing
    rule = if N == 2
        fastpath.frule_2
    elseif N == 1
        fastpath.frule_1
    elseif N == 3
        fastpath.frule_3
    elseif N == 4
        fastpath.frule_4
    elseif N == 5
        fastpath.frule_5
    elseif N == 6
        fastpath.frule_6
    elseif N == 7
        fastpath.frule_7
    elseif N == 8
        fastpath.frule_8
    else
        nothing
    end
    isnothing(rule) && return nothing
    return value_and_derivative!!(rule, map(tuple, input_primals, input_tangents)...)
end

@noinline function _fcache_derivative_chunked!!(
    cache::ForwardCache{R,IT,OP,FG,GW,CF},
    ::Val{N},
    x_dx::Vararg{Tuple,M};
    friendly_tangents::Bool=false,
) where {R,IT<:Union{Nothing,Tuple},OP,FG,GW,CF<:NfwdCache,N,M}
    N < 1 && throw(ArgumentError("NTangent inputs must contain at least one lane."))
    input_primals = map(first, x_dx)
    input_tangents = map(last, x_dx)
    friendly_tangents &&
        return _fcache_derivative_chunked_loop!!(cache, Val(N), x_dx...; friendly_tangents)
    chunked_output = _maybe_chunk_frule_chunked_ir(
        cache, input_primals, input_tangents, Val(N)
    )
    !isnothing(chunked_output) && return chunked_output
    return _fcache_derivative_chunked_loop!!(cache, Val(N), x_dx...; friendly_tangents)
end

"""
    Pullback

Concrete pullback object for `nfwd` reverse rules. It stores the primal callable,
primals, input tangents, and output fdata needed to rerun chunked NDual passes during the
reverse sweep.

!!! note
    The scalar specialization `Pullback{F,N,Tuple{T},Tuple{NoFData},Y}` with
    `T<:Number` must remain an `isbits` type for that path to stay allocation-free. The
    generic path (array or multi-input primals) is not isbits and allocates as usual.
    Do not add heap-allocated fields without auditing both paths.
"""
struct Pullback{F,N,P,T,Y}
    f::F
    primals::P
    tangents::T
    y_fdata::Y
end

"""
    ArrayScalarPullback

Lightweight pullback returned by the optimised single-array-input / scalar-output rrule fast
path.  The full gradient (∂f/∂x_i for all i) is computed eagerly during the rrule call and
stored in `grad` (a separate copy, not aliased with `tangent(x_codual)`).  `fdata` is a
reference to `tangent(x_codual)`.  The pullback accumulates `ȳ * grad` into `fdata`,
satisfying Mooncake's standard increment semantics for mutable array tangents.
"""
struct ArrayScalarPullback{G<:AbstractArray}
    grad::G   # precomputed ∂f/∂x; does NOT alias fdata
    fdata::G  # tangent(x_cd); the accumulation target
end

function (pb::ArrayScalarPullback)(y_rdata)
    if isone(y_rdata)
        pb.fdata .+= pb.grad
    else
        pb.fdata .+= y_rdata .* pb.grad
    end
    return (NoRData(), NoRData())
end

#
# Public construction and execution
#
# These are the main `nfwd` entry points. A reviewer should be able to read this section
# first, then dive into the lower-level pipelines only as needed.

"""
    build_frule(f, x...; chunk_size=nothing)
    build_frule(sig::Type{<:Tuple}; chunk_size=nothing)

Build a forward-mode rule through `nfwd`.

This path is independent from Mooncake's `frule!!` (aka ir-based forward) path and obeys
the standard `frule!!` interface. It evaluates the primal function directly on
NDual-lifted scalar / dense-array inputs. Rule construction is signature-based, so `nfwd`
only supports stateless callables here.

If `chunk_size` is omitted, nfwd automatically selects `min(DOF, hardware_preferred_width)`
from the signature, where `hardware_preferred_width` is 8 (one AVX-512 / two AVX2 Float64
registers). For scalar-only signatures the DOF is known exactly at type level; for
signatures containing arrays the preferred width is used directly.

!!! warning "Not thread-safe"
    The returned `Rule` holds a mutable workspace buffer that is updated in-place
    on every call. Do not share a single rule across threads; build one rule per thread.

!!! note "debug_mode"
    The `debug_mode` keyword is accepted for API consistency with Mooncake's other
    rule/cache builders but always throws when `true`; nfwd-specific debug checks are
    not yet implemented. Mooncake's outer debug wrapper still validates CoDual
    inputs/outputs when the rule is invoked inside a debug-mode rrule.

## Example

```julia
julia> using Mooncake

julia> frule = Mooncake.NfwdMooncake.build_frule(
           Tuple{typeof(sum), Vector{Float64}}; chunk_size=1
       );

julia> x = [1.0, 2.0, 3.0];

julia> frule(Mooncake.Dual(sum, Mooncake.NoTangent()), Mooncake.Dual(x, ones(3)))
Mooncake.Dual(6.0, 3.0)
```
"""
function build_frule(
    sig::Type{<:Tuple}; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    resolved = _nfwd_resolve_rule_chunk_size(sig, chunk_size; debug_mode)
    buf = _nfwd_frule_buf_ref(sig, Val(resolved))
    return Rule{sig,resolved,typeof(buf)}(buf)
end

function build_frule(
    f, x...; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    return build_frule(typeof((f, x...)); chunk_size, debug_mode, silence_debug_messages)
end

# Primitive scalar wrappers in rules_via_nfwd.jl only need these nfwd execution helpers.
# Calling these helpers avoids constructing a fresh Rule/RRule wrapper at every primitive
# callsite, which would otherwise add avoidable allocations and dispatch overhead to hot
# scalar rules. Rule and RRule still back the public build_frule/build_rrule APIs, where
# the caller builds one wrapper, reuses it, and keeps its mutable workspace private to
# that instance. Primitive rules do not have that caller-owned lifecycle: they are
# entered through ordinary Mooncake dispatch, so using Rule/RRule there would either
# build a new mutable wrapper per call or hide shared mutable workspace behind a plain
# rule method. That shared-state hazard is not unique to nfwd, but it matters most here
# because primitive rules are expected to behave like ordinary stateless methods.
@inline function _nfwd_primitive_frule_call(
    ::Val{N}, f::Dual, x::Vararg{Dual,M}
) where {M,N}
    _nfwd_check_function_tangent(tangent(f))
    primals = map(primal, x)
    tangents = map(tangent, x)
    packed = _nfwd_primitive_packed_lane_count(tangents)
    y, dy = _nfwd_eval(
        primal(f), primals, tangents, _nfwd_primitive_chunk_size(Val(N), tangents)
    )
    return Dual(y, isnothing(packed) ? dy : _nfwd_pack_output_tangent(y, dy, Val(packed)))
end

# The generic vararg path can allocate for small scalar primitive wrappers, so keep
# fixed-arity entry points here for common binary/ternary rules such as `atan`, `log`,
# and `clamp`.
@inline function _nfwd_primitive_frule_call(::Val{N}, f::Dual, x1::Dual, x2::Dual) where {N}
    _nfwd_check_function_tangent(tangent(f))
    tangents = (tangent(x1), tangent(x2))
    packed = _nfwd_primitive_packed_lane_count(tangents)
    y, dy = _nfwd_eval(
        primal(f),
        (primal(x1), primal(x2)),
        tangents,
        _nfwd_primitive_chunk_size(Val(N), tangents),
    )
    return Dual(y, isnothing(packed) ? dy : _nfwd_pack_output_tangent(y, dy, Val(packed)))
end

@inline function _nfwd_primitive_frule_call(
    ::Val{N}, f::Dual, x1::Dual, x2::Dual, x3::Dual
) where {N}
    _nfwd_check_function_tangent(tangent(f))
    tangents = (tangent(x1), tangent(x2), tangent(x3))
    packed = _nfwd_primitive_packed_lane_count(tangents)
    y, dy = _nfwd_eval(
        primal(f),
        (primal(x1), primal(x2), primal(x3)),
        tangents,
        _nfwd_primitive_chunk_size(Val(N), tangents),
    )
    return Dual(y, isnothing(packed) ? dy : _nfwd_pack_output_tangent(y, dy, Val(packed)))
end

function (rule::Rule{sig,N})(f::Dual, x::Vararg{Dual,M}) where {sig,N,M}
    _nfwd_verify_sig(rule, (f, x...))
    _nfwd_check_function_tangent(tangent(f))
    primals = map(primal, x)
    tangents = map(tangent, x)
    y, dy = _nfwd_eval(primal(f), primals, tangents, Val(N))
    return Dual(y, dy)
end

# Scalar-input specializations avoid the generic vararg/map path, which otherwise leaves
# small cached nfwd rules on an allocating hot path.
@inline function (rule::Rule{sig,N})(f::Dual, x::Dual{T,D}) where {sig,N,T<:Number,D}
    _nfwd_verify_sig(rule, (f, x))
    _nfwd_check_function_tangent(tangent(f))
    y, dy = _nfwd_eval(primal(f), (primal(x),), (tangent(x),), Val(N))
    return Dual(y, dy)
end

@inline function (rule::Rule{sig,N})(
    f::Dual, x1::Dual{T1,D1}, x2::Dual{T2,D2}
) where {sig,N,T1<:Number,T2<:Number,D1,D2}
    _nfwd_verify_sig(rule, (f, x1, x2))
    _nfwd_check_function_tangent(tangent(f))
    y, dy = _nfwd_eval(
        primal(f), (primal(x1), primal(x2)), (tangent(x1), tangent(x2)), Val(N)
    )
    return Dual(y, dy)
end

@inline function (rule::Rule{sig,N})(
    f::Dual, x1::Dual{T1,D1}, x2::Dual{T2,D2}, x3::Dual{T3,D3}
) where {sig,N,T1<:Number,T2<:Number,T3<:Number,D1,D2,D3}
    _nfwd_verify_sig(rule, (f, x1, x2, x3))
    _nfwd_check_function_tangent(tangent(f))
    y, dy = _nfwd_eval(
        primal(f),
        (primal(x1), primal(x2), primal(x3)),
        (tangent(x1), tangent(x2), tangent(x3)),
        Val(N),
    )
    return Dual(y, dy)
end

# Optimised single-array-input frule: reuses a pre-allocated lifted buffer when the tangent
# is in chunk layout (ndims(dx) == ndims(x) + 1). Falls through to the generic allocating
# path for the plain layout and for malformed tangent dimensions, where `_nfwd_eval`
# produces the user-facing validation error.
function (rule::Rule{sig,N})(
    f::Dual, x::Dual{Array{T,Nd},Array{T,Nd1}}
) where {sig,N,T<:IEEEFloat,Nd,Nd1}
    _nfwd_verify_sig(rule, (f, x))
    _nfwd_check_function_tangent(tangent(f))
    px = _nfwd_check_primal(primal(x))
    dx = tangent(x)
    if Nd1 == Nd + 1  # chunk layout — use in-place lift with pre-allocated buffer
        lifted = _nfwd_frule_lifted!(rule.buf, px, dx, Val(N))
        y, dy = _nfwd_extract(primal(f)(lifted), Val(N))
    else  # non-chunk layout — fall back to the allocating path
        y, dy = _nfwd_eval(primal(f), (px,), (dx,), Val(N))
    end
    return Dual(y, dy)
end

@inline _nfwd_rule_pack_buffer(::IEEEFloat) = nothing
@inline _nfwd_rule_pack_buffer(::Complex{<:IEEEFloat}) = nothing
@inline function _nfwd_rule_pack_buffer(
    x::Array{T}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    return Ref{Union{Nothing,Array{T}}}(nothing)
end
@inline _nfwd_rule_pack_buffer(x::Tuple) = tuple_map(_nfwd_rule_pack_buffer, x)

@inline function Mooncake.value_and_derivative!!(
    rule::Rule{sig,N}, fx::Vararg{Tuple{Any,Any},M}
) where {sig,N,M}
    # The generic `value_and_derivative!!(rule, ...)` entrypoint in `interface.jl` dispatches
    # here for `NfwdMooncake.Rule`. This path detects `NTangent`, packs it once into nfwd's
    # native width-N tangent layout, calls the rule once, and unpacks the result back to
    # `NTangent`; it is not the lane-loop fallback used by the generic cached chunk path.
    input_primals = tuple_map(first, fx)
    input_tangents = tuple_map(last, fx)
    lane_count = Mooncake._fcache_derivative_ntangent_lane_count(input_tangents)
    isnothing(lane_count) && return invoke(
        Mooncake.value_and_derivative!!,
        Tuple{Any,Vararg{Tuple{Any,Any},M}},
        rule,
        fx...,
    )
    lane_count isa Val{N} || throw(
        ArgumentError(
            "NTangent inputs have $(typeof(lane_count).parameters[1]) lanes, but this nfwd rule " *
            "was built with chunk_size=$N.",
        ),
    )
    pack_buffers = tuple_map(_nfwd_rule_pack_buffer, Base.tail(input_primals))
    packed_tangents = ntuple(
        i -> _chunk_pack_tangent(
            Base.tail(input_primals)[i],
            Base.tail(input_tangents)[i],
            pack_buffers[i],
            Val(N),
        ),
        Val(fieldcount(typeof(pack_buffers))),
    )
    # Keep this at the Rule/Dual boundary: `f` stays on its ordinary width-1 tangent,
    # while the argument tangents are packed to the rule's native width-N layout and the
    # rule itself performs the NDual lift internally.
    output = rule(
        Dual(first(input_primals), first(input_tangents)),
        tuple_map(Dual, Base.tail(input_primals), packed_tangents)...,
    )
    y = primal(output)
    dy = tangent(output)
    return y, NTangent(ntuple(lane -> _nfwd_unpack_output_lane(y, dy, Val(lane)), Val(N)))
end

"""
    build_rrule(f, x...; chunk_size=nothing)
    build_rrule(sig::Type{<:Tuple}; chunk_size=nothing)

Build a reverse-mode rule through `nfwd`.

The reverse rule is derived from chunked NDual forward passes and obeys the standard
`rrule!!` interface. Rule construction is signature-based, so `nfwd` only supports
stateless callables here.

If `chunk_size` is omitted, nfwd automatically selects `min(DOF, hardware_preferred_width)`
from the signature, where `hardware_preferred_width` is 8 (one AVX-512 / two AVX2 Float64
registers). For scalar-only signatures the DOF is known exactly at type level; for
signatures containing arrays the preferred width is used directly.

!!! warning "Not thread-safe"
    The returned `RRule` holds mutable workspace buffers (`buf`, `grad_buf`) that
    are updated in-place on every call. Do not share a single rule across threads; build
    one rule per thread, or use `Mooncake.prepare_derivative_cache` and create one cache
    per thread.

!!! note "debug_mode"
    The `debug_mode` keyword is accepted for API consistency with Mooncake's other
    rule/cache builders but always throws when `true`; nfwd-specific debug checks are
    not yet implemented. Mooncake's outer debug wrapper still validates CoDual
    inputs/outputs when the rule is invoked inside a debug-mode rrule.

## Example

```julia
julia> using Mooncake

julia> f(x) = sum(abs2, x)
f (generic function with 1 method)

julia> rrule = Mooncake.NfwdMooncake.build_rrule(
           Tuple{typeof(f), Vector{Float64}}; chunk_size=1
       );

julia> x = [1.0, 2.0, 3.0];

julia> y, pb!! = rrule(
           Mooncake.CoDual(f, Mooncake.NoFData()),
           Mooncake.CoDual(x, zeros(3)),
       );

julia> Mooncake.primal(y)
14.0

julia> pb!!(1.0)
(Mooncake.NoRData(), [2.0, 4.0, 6.0])
```
"""
function build_rrule(
    sig::Type{<:Tuple}; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    resolved = _nfwd_resolve_rule_chunk_size(sig, chunk_size; debug_mode)
    buf = _nfwd_buf_ref(sig, Val(resolved))
    grad_buf = _nfwd_grad_buf_ref(sig)
    scalar_out = _nfwd_infer_scalar_output(sig)
    return RRule{sig,resolved,typeof(buf),scalar_out,typeof(grad_buf)}(buf, grad_buf)
end

function build_rrule(
    f, x...; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    return build_rrule(typeof((f, x...)); chunk_size, debug_mode, silence_debug_messages)
end

@inline function _nfwd_primitive_rrule_call(
    ::Val{N}, f::CoDual, x::Vararg{CoDual,M}
) where {M,N}
    _nfwd_check_function_tangent(tangent(f))
    return _nfwd_rrule_call(primal(f), x, Val(N))
end

"""
    (rule::RRule)(f::CoDual, x::Vararg{CoDual})

Evaluate an `nfwd` reverse rule and return both the output `CoDual` and pullback.
`f` must be a stateless callable: `tangent(f)` must be `NoFData`, otherwise an
`ArgumentError` is thrown. Differentiating with respect to `f` is not supported.
"""
function (rule::RRule{sig,N})(f::CoDual, x::Vararg{CoDual,M}) where {sig,N,M}
    _nfwd_verify_sig(rule, (f, x...))
    _nfwd_check_function_tangent(tangent(f))
    return _nfwd_rrule_call(primal(f), x, Val(N))
end

# Optimised single-real-array-input rrule, scalar-output fast path.
#
# When `scalar_out=true` (inferred at rule-build time), the output is known to be an
# IEEEFloat scalar so we skip the redundant primal type-check call.  For small DOF / single-
# chunk problems that extra call would cost one full function evaluation — e.g. for
# `large_single_block` (DOF=2, 400 scalar ops) it was ~23% of total rrule time.
#
# The pullback only needs to scale the pre-computed gradient by the output cotangent —
# zero per-call allocations at steady state.
function (rule::RRule{sig,N,Tbuf,true})(
    f::CoDual, x::CoDual{A}
) where {sig,N,Tbuf,T<:IEEEFloat,Nd,A<:Array{T,Nd}}
    f_runtime, x_primal = _nfwd_prepare_array_rrule_call(rule, f, x)
    # Output type is known scalar — skip primal call and go straight to gradient sweep.
    # Gradient is written to the pre-allocated grad_buf (not into tangent(x)), so the
    # pullback can accumulate into the existing fdata without a copy.
    return _nfwd_array_scalar_rrule_result(rule, f_runtime, x, x_primal, Val(N))
end

# Fallback: output type not known to be scalar at build time.  Run a primal call to
# dispatch between the scalar fast path and the generic chunked path.
function (rule::RRule{sig,N,Tbuf,false})(
    f::CoDual, x::CoDual{A}
) where {sig,N,Tbuf,T<:IEEEFloat,Nd,A<:Array{T,Nd}}
    f_runtime, x_primal = _nfwd_prepare_array_rrule_call(rule, f, x)
    y_primal = f_runtime(x_primal)
    if y_primal isa IEEEFloat
        return _nfwd_array_scalar_rrule_result(rule, f_runtime, x, x_primal, Val(N))
    else
        _nfwd_is_supported_primal(y_primal) || _nfwd_output_error((x_primal,), y_primal)
        y_cd = CoDual(y_primal, fdata(zero_tangent(y_primal)))
        return y_cd,
        _nfwd_pullback(f_runtime, (x_primal,), (tangent(x),), tangent(y_cd), Val(N))
    end
end

# Optimization note:
# This scalar specialization bypasses the general pullback-based reverse path for cached
# `value_and_gradient!!` calls. Evaluating one NDual-lifted primal directly is enough to recover
# the scalar primal and derivative, which removes the remaining steady-state allocations for
# singleton scalar inputs.
#
# Complex scalars (CoDual{<:Complex}) have no matching specialization here and fall through
# to the generic `__value_and_gradient!!` in src/interface.jl, which runs the full pullback.
# This is correct and tested; it is simply not on the allocation-free fast path.
"""
    __value_and_gradient!!(rule::RRule, f::CoDual, x::CoDual)

Dispatch the scalar cached fast path for `nfwd` reverse mode.
"""
function __value_and_gradient!!(
    rule::RRule{sig,N}, f::CoDual, x::CoDual{T}
) where {sig,N,T<:IEEEFloat}
    _nfwd_verify_sig(rule, (f, x))
    return _nfwd_scalar_value_and_gradient(primal(f), f, x, Val(N))
end

"""
    __value_and_gradient!!(rule::RRule, f::CoDual, x::CoDual{<:Array})

Scalar-output dense-array fast path for `nfwd` reverse rules. Dispatches to a
typed-workspace path for `Vector{T}` inputs (via the buf type parameter) and a generic
workspace path for higher-dimensional arrays.
"""
function __value_and_gradient!!(
    rule::RRule{sig,chunk_size}, f::CoDual, x::CoDual{A}
) where {sig,chunk_size,T<:IEEEFloat,N,A<:Array{T,N}}
    _nfwd_verify_sig(rule, (f, x))
    _nfwd_check_function_tangent(tangent(f))
    y = _nfwd_array_scalar_value_and_gradient(primal(f), x, rule.buf, Val(chunk_size))
    return y, (_nfwd_function_gradient(f), tangent(x))
end

const NFWD_DEBUG_MODE_WARNING =
    "nfwd-backed reverse-mode rules ignore `debug_mode=true`; " *
    "Mooncake's outer debug wrapper still checks CoDual inputs/outputs, but the " *
    "inner nfwd rule executes without nfwd-specific debug checks."

"""
    _copy(rule::Rule)

Copy a `Rule` while resetting cached workspace state.
"""
function _copy(x::Rule{sig,N,Tbuf}) where {sig,N,Tbuf}
    return Rule{sig,N,Tbuf}(Tbuf(nothing))
end

"""
    _copy(rule::RRule)

Copy an `RRule` while resetting cached workspace state.
"""
function _copy(x::RRule{sig,N,Tbuf,scalar_out,Tgbuf}) where {sig,N,Tbuf,scalar_out,Tgbuf}
    return RRule{sig,N,Tbuf,scalar_out,Tgbuf}(Tbuf(nothing), Tgbuf(nothing))
end

# RRule bakes sig into its type parameters and validates internally via
# _nfwd_verify_sig on every call; no redundant check is needed here.
__verify_sig(::RRule, ::Tuple) = nothing

"""
    verify_fwds_inputs(rule::RRule, x)

Emit the outer debug-mode warning before delegating to generic forward-input checks.
"""
@noinline function verify_fwds_inputs(rule::RRule, @nospecialize(x::Tuple))
    @warn NFWD_DEBUG_MODE_WARNING
    return invoke(verify_fwds_inputs, Tuple{Any,Tuple}, rule, x)
end

#
# Validation and layout helpers
#
# Shared validation, sizing, and shape utilities used across the forward, reverse, and cached
# execution paths.

@inline function _nfwd_check_function_tangent(df)
    df isa Union{NoTangent,NoFData} && return nothing
    throw(ArgumentError("nfwd does not support differentiating with respect to `f`."))
end

@inline function _chunked_resolve_rule_chunk_size(sig::Type{<:Tuple}, chunk_size)
    resolved = isnothing(chunk_size) ? _nfwd_sig_default_chunk_size(sig) : chunk_size
    return _nfwd_check_chunk_size(resolved)
end

@inline function _nfwd_resolve_rule_chunk_size(
    sig::Type{<:Tuple}, chunk_size; debug_mode::Bool
)
    resolved = isnothing(chunk_size) ? _nfwd_sig_default_chunk_size(sig) : chunk_size
    return _nfwd_validate(sig, resolved; debug_mode)
end

@inline function _nfwd_verify_sig(rule::Union{Rule,RRule}, fx::Tuple)
    sig = _nfwd_rule_sig(rule)
    Tfx = Tuple{map(_typeof ∘ primal, fx)...}
    # Use <: (subtype) rather than == so that a rule built for an abstract signature
    # (e.g. Tuple{typeof(f), AbstractVector{Float64}}) also accepts concrete subtypes
    # at call time. This mirrors the convention used elsewhere in Mooncake's dispatch.
    Tfx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tfx do not subtype rule signature, $sig"))
end

@inline function _nfwd_check_config(config)
    config.friendly_tangents &&
        throw(ArgumentError("nfwd does not currently support `friendly_tangents=true`."))
    config.debug_mode &&
        throw(ArgumentError("nfwd does not currently support `debug_mode=true`."))
    return nothing
end

#
# Reverse accumulation utilities
#
# These helpers seed input directions, contract output tangents with cotangents, and scatter
# each chunk's contributions into gradient storage.

@inline function _nfwd_seed_tangent(
    x::IEEEFloat, chunk_size::Int, start_slot::Int, offset::Int
)
    # offset+1 is this scalar's global slot; lane is its 1-indexed position in the chunk.
    global_slot = offset + 1
    lane = global_slot - start_slot + 1
    if chunk_size == 1
        return lane == 1 ? one(x) : zero(x)
    end
    return ntuple(k -> typeof(x)(k == lane), Val(chunk_size))
end

function _nfwd_seed_tangent(
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

function _nfwd_seed_tangent(
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

function _nfwd_seed_tangent(
    x::AbstractArray{Complex{T}}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    # Each complex element contributes 2 DOFs in consecutive global slots:
    #   odd  local_slot → seed the real part  (complex(1, 0))
    #   even local_slot → seed the imaginary part (complex(0, 1))
    # So element index = cld(local_slot, 2) and part = isodd(local_slot).
    if chunk_size == 1
        dx = zero_tangent(x)
        global_slot = start_slot
        if offset < global_slot <= offset + 2 * length(x)
            local_slot = global_slot - offset
            elem = cld(local_slot, 2)
            dx[elem] = if isodd(local_slot)
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

@inline function _nfwd_add_slot!(
    g::Base.RefValue{T}, local_slot::Int, v
) where {T<:IEEEFloat}
    local_slot == 1 && (g[] += v)
    return nothing
end

@inline function _nfwd_add_slot!(
    g::Base.RefValue{Complex{T}}, local_slot::Int, v
) where {T<:IEEEFloat}
    if local_slot == 1
        g[] += complex(v, zero(T))
    elseif local_slot == 2
        g[] += complex(zero(T), v)
    end
    return nothing
end

@inline function _nfwd_add_slot!(
    g::AbstractArray{T}, local_slot::Int, v
) where {T<:IEEEFloat}
    g[local_slot] += v
    return nothing
end

@inline function _nfwd_add_slot!(
    g::AbstractArray{Complex{T}}, local_slot::Int, v
) where {T<:IEEEFloat}
    elem = cld(local_slot, 2)
    g[elem] += isodd(local_slot) ? complex(v, zero(T)) : complex(zero(T), v)
    return nothing
end

function _nfwd_scatter_chunk!(grads::Tuple, inputs::Tuple, dy::Tuple, start_slot::Int)
    global_slot = start_slot
    for lane_val in dy
        offset = 0
        for (i, x) in enumerate(inputs)
            dof = _nfwd_input_dof(x)
            if offset < global_slot <= offset + dof
                local_slot = global_slot - offset
                _nfwd_add_slot!(grads[i], local_slot, lane_val)
                break
            end
            offset += dof
        end
        global_slot += 1
    end
    return nothing
end

@inline _nfwd_gradient_refs(::Tuple{}, ::Tuple{}) = ()
@inline function _nfwd_gradient_refs(primals::Tuple, tangents::Tuple)
    x = first(primals)
    dx = first(tangents)
    g = if x isa Number
        Ref(zero_tangent(x, dx))
    else
        # Use a fresh zeros array (not the fdata) for VJP accumulation. The generic
        # pullback adds this into the fdata at the end so that existing fdata content
        # (e.g. contributions from other uses of the same array) is preserved.
        zero_tangent(x)
    end
    return (g, _nfwd_gradient_refs(Base.tail(primals), Base.tail(tangents))...)
end

_nfwd_unwrap_gradient(g::Base.RefValue) = g[]
_nfwd_unwrap_gradient(g) = g

@inline _nfwd_accumulate_array_gradients!(::Tuple{}, ::Tuple{}) = nothing
@inline function _nfwd_accumulate_array_gradients!(tangents::Tuple, grads::Tuple)
    fdata = first(tangents)
    grad = first(grads)
    fdata isa AbstractArray && (fdata .+= _nfwd_unwrap_gradient(grad))
    _nfwd_accumulate_array_gradients!(Base.tail(tangents), Base.tail(grads))
    return nothing
end

@inline _nfwd_gradient_rdatas(::Tuple{}) = ()
@inline function _nfwd_gradient_rdatas(grads::Tuple)
    return (
        rdata(_nfwd_unwrap_gradient(first(grads))),
        _nfwd_gradient_rdatas(Base.tail(grads))...,
    )
end

@inline _nfwd_zero_scalar_grads(::Tuple{}, ::Tuple{}) = ()
@inline function _nfwd_zero_scalar_grads(primals::Tuple, tangents::Tuple)
    return (
        zero_tangent(first(primals), first(tangents)),
        _nfwd_zero_scalar_grads(Base.tail(primals), Base.tail(tangents))...,
    )
end

@inline function _nfwd_update_scalar_grad(
    grads::Tuple, primals::Tuple, global_slot::Int, lane_val, offset::Int=0
)
    x = first(primals)
    dof = _nfwd_input_dof(x)
    if offset < global_slot <= offset + dof
        local_slot = global_slot - offset
        return (
            _nfwd_accumulate_scalar_gradient(first(grads), local_slot, lane_val),
            Base.tail(grads)...,
        )
    end
    return (
        first(grads),
        _nfwd_update_scalar_grad(
            Base.tail(grads), Base.tail(primals), global_slot, lane_val, offset + dof
        )...,
    )
end

@inline function _nfwd_scatter_scalar_chunk(
    grads::Tuple, primals::Tuple, dy::Tuple, start_slot::Int
)
    global_slot = start_slot
    for lane_val in dy
        grads = _nfwd_update_scalar_grad(grads, primals, global_slot, lane_val)
        global_slot += 1
    end
    return grads
end

# `slot` is the 1-based DOF index within the scalar/complex input: 1 for the real
# component (or the sole IEEEFloat slot), 2 for the imaginary component of a complex.
# Called from `_nfwd_scalar_gradient_rdata` with the loop's global_slot, which
# equals the local slot because that path is specialised to a single input at offset 0.
@inline function _nfwd_accumulate_scalar_gradient(g::T, slot::Int, v) where {T<:IEEEFloat}
    slot == 1 ? g + v : g
end

@inline function _nfwd_accumulate_scalar_gradient(
    g::Complex{T}, slot::Int, v
) where {T<:IEEEFloat}
    if slot == 1
        return g + complex(v, zero(T))
    elseif slot == 2
        return g + complex(zero(T), v)
    end
    return g
end

@inline function _nfwd_real_dot(a::T, b::T) where {T<:IEEEFloat}
    return a * Nfwd._nfwd_zero_mask(a, b)
end

@inline function _nfwd_real_dot(a::Complex{T}, b::Complex{T}) where {T<:IEEEFloat}
    return real(conj(a) * Nfwd._nfwd_zero_mask(a, b))
end

# Scalar (real or complex): chunk_size=1 → plain scalar zero; chunk_size=N → N-tuple of zeros.
@inline _nfwd_zero_output_tangent(y::Union{IEEEFloat,Complex{<:IEEEFloat}}, ::Val{1}) = zero(
    y
)
@inline _nfwd_zero_output_tangent(y::Union{IEEEFloat,Complex{<:IEEEFloat}}, ::Val{N}) where {N} = ntuple(
    _ -> zero(y), Val(N)
)

# Array (real or complex elements): chunk_size=1 → same-shape zero array; chunk_size=N → N extra lanes.
@inline _nfwd_zero_output_tangent(y::AbstractArray{<:Union{IEEEFloat,Complex{<:IEEEFloat}}}, ::Val{1}) = zero_tangent(
    y
)
@inline function _nfwd_zero_output_tangent(
    y::AbstractArray{<:Union{IEEEFloat,Complex{<:IEEEFloat}}}, ::Val{N}
) where {N}
    return zeros(eltype(y), size(y)..., N)
end

# Tuple outputs: recurse element-wise.
@inline function _nfwd_zero_output_tangent(y::Tuple, ::Val{N}) where {N}
    return map(yi -> _nfwd_zero_output_tangent(yi, Val(N)), y)
end

# chunk_size=1: tangent is a plain scalar — return it regardless of which lane is requested.
@inline _nfwd_scalar_lane(dy, ::Val{1}, _) = dy
# chunk_size=N: tangent is an NTuple — index with a static Val{K} or a runtime Int.
@inline _nfwd_scalar_lane(dy::Tuple{Any}, ::Val{1}, ::Val{K}) where {K} = dy[1]
@inline _nfwd_scalar_lane(dy::Tuple{Any}, ::Val{1}, _lane::Int) = dy[1]
@inline _nfwd_scalar_lane(dy::NTuple{N}, ::Val{N}, ::Val{K}) where {N,K} = dy[K]
@inline _nfwd_scalar_lane(dy::NTuple{N}, ::Val{N}, lane::Int) where {N} = dy[lane]

function _nfwd_contract_output(ȳ::T, dy::T) where {T<:IEEEFloat}
    return (_nfwd_real_dot(ȳ, dy),)
end

function _nfwd_contract_output(ȳ::Complex{T}, dy::Complex{T}) where {T<:IEEEFloat}
    return (_nfwd_real_dot(ȳ, dy),)
end

function _nfwd_contract_output(ȳ::T, dy::NTuple{N,T}) where {T<:IEEEFloat,N}
    return ntuple(k -> _nfwd_real_dot(ȳ, dy[k]), Val(N))
end

function _nfwd_contract_output(
    ȳ::Complex{T}, dy::NTuple{N,Complex{T}}
) where {T<:IEEEFloat,N}
    return ntuple(k -> _nfwd_real_dot(ȳ, dy[k]), Val(N))
end

# Single-chunk array case (ȳ and dy have the same shape — real or complex elements).
function _nfwd_contract_output(
    ȳ::A, dy::A
) where {A<:AbstractArray{<:Union{IEEEFloat,Complex{<:IEEEFloat}}}}
    acc = zero(real(eltype(ȳ)))
    @inbounds for I in CartesianIndices(ȳ)
        acc += _nfwd_real_dot(ȳ[I], dy[I])
    end
    return (acc,)
end

# Multi-chunk array case (dy has one extra trailing dimension of size N — real or complex).
# Both arrays must share the same element type T.  Mixed-precision cases (e.g.
# ȳ::Vector{Float32} with dy::Matrix{Float64}) fall through to the generic error overload
# below.  In practice nfwd keeps element types consistent across primal/tangent, so
# this situation only arises from incorrect external use.
function _nfwd_contract_output(
    ȳ::A, dy::B
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},A<:AbstractArray{T},B<:AbstractArray{T}}
    ndims(dy) == ndims(ȳ) + 1 || _nfwd_output_error(dy)
    size(dy)[1:(end - 1)] == size(ȳ) || _nfwd_output_error(dy)
    N = size(dy, ndims(dy))
    return ntuple(Val(N)) do k
        acc = zero(real(T))
        @inbounds for I in CartesianIndices(ȳ)
            idx = Tuple(I)
            acc += _nfwd_real_dot(ȳ[I], dy[idx..., k])
        end
        acc
    end
end

# Tuple outputs: contract each element independently and sum lane contributions.
function _nfwd_contract_output(ȳ::Tuple, dy::Tuple)
    length(ȳ) == length(dy) || _nfwd_output_error(dy)
    contributions = map(_nfwd_contract_output, ȳ, dy)
    return foldl((a, b) -> map(+, a, b), contributions)
end

function _nfwd_contract_output(ȳ, dy)
    _nfwd_output_error(dy)
end

#
# Reverse execution
#
# `Pullback` is a concrete callable struct rather than a closure so direct
# `build_rrule(...)(...)` calls can stay allocation-free on the scalar path.
# The pullback still carries the cached primals / tangents / output fdata needed to rerun
# chunked NDual passes during the reverse sweep.

"""
    _nfwd_rrule_call(f, x, chunk_size_or_val)

Run the shared reverse-mode `nfwd` path: evaluate the primal on the runtime primals,
wrap the result in the `CoDual` shape expected by `rrule!!`, and build the pullback that
reruns chunked NDual passes during the reverse sweep.
"""
@inline function _nfwd_rrule_call(f, x::Tuple{Vararg{CoDual,M}}, ::Val{N}) where {M,N}
    primals = map(primal, x)
    tangents = map(tangent, x)
    y_primal = f(primals...)
    _nfwd_is_supported_primal(y_primal) || _nfwd_output_error(primals, y_primal)
    y = CoDual(y_primal, fdata(zero_tangent(y_primal)))
    return y, _nfwd_pullback(f, primals, tangents, tangent(y), Val(N))
end

# Match the fixed-arity forward fast paths above: the generic tuple path can allocate for
# small scalar primitive pullbacks as well.
@inline function _nfwd_rrule_call(f, x::Tuple{CoDual,CoDual}, ::Val{N}) where {N}
    primals = (primal(x[1]), primal(x[2]))
    tangents = (tangent(x[1]), tangent(x[2]))
    y_primal = f(primals...)
    _nfwd_is_supported_primal(y_primal) || _nfwd_output_error(primals, y_primal)
    y = CoDual(y_primal, fdata(zero_tangent(y_primal)))
    return y, _nfwd_pullback(f, primals, tangents, tangent(y), Val(N))
end

@inline function _nfwd_rrule_call(f, x::Tuple{CoDual,CoDual,CoDual}, ::Val{N}) where {N}
    primals = (primal(x[1]), primal(x[2]), primal(x[3]))
    tangents = (tangent(x[1]), tangent(x[2]), tangent(x[3]))
    y_primal = f(primals...)
    _nfwd_is_supported_primal(y_primal) || _nfwd_output_error(primals, y_primal)
    y = CoDual(y_primal, fdata(zero_tangent(y_primal)))
    return y, _nfwd_pullback(f, primals, tangents, tangent(y), Val(N))
end

@inline function _nfwd_rrule_call(f, x::Tuple, chunk_size::Integer)
    return _nfwd_rrule_call(f, x, Val(_nfwd_check_chunk_size(chunk_size)))
end

"""
    _nfwd_pullback(rule, primals, tangents, y_fdata)

Package the state needed for a later reverse sweep into an `Pullback`.
"""
function _nfwd_pullback(f, primals::Tuple, tangents::Tuple, y_fdata, ::Val{N}) where {N}
    return Pullback{typeof(f),N,typeof(primals),typeof(tangents),typeof(y_fdata)}(
        f, primals, tangents, y_fdata
    )
end

@inline _nfwd_seed_tangents(::Tuple{}, ::Val{N}, start_slot::Int, offset::Int=0) where {N} = ()
@inline function _nfwd_seed_tangents(
    primals::Tuple, ::Val{N}, start_slot::Int, offset::Int=0
) where {N}
    x = first(primals)
    return (
        _nfwd_seed_tangent(x, N, start_slot, offset),
        _nfwd_seed_tangents(
            Base.tail(primals), Val(N), start_slot, offset + _nfwd_input_dof(x)
        )...,
    )
end
"""
    _nfwd_scalar_gradient_rdata(pb, y_rdata)

Compute scalar-input reverse data for the specialized scalar pullback path.
"""
function _nfwd_scalar_gradient_rdata(
    pb::Pullback{F,N,Tuple{T},Tuple{NoFData},Y}, y_rdata
) where {F,N,T<:Number,Y}
    ȳ = tangent(pb.y_fdata, y_rdata)
    x = pb.primals[1]
    g = zero_tangent(x, pb.tangents[1])
    total_dof = _nfwd_input_dof(x)
    for start_slot in 1:N:total_dof
        tangents = (_nfwd_seed_tangent(x, N, start_slot, 0),)
        _, dy = _nfwd_eval(pb.f, pb.primals, tangents, Val(N))
        lane_vals = _nfwd_contract_output(ȳ, dy)
        global_slot = start_slot
        for lane_val in lane_vals
            g = _nfwd_accumulate_scalar_gradient(g, global_slot, lane_val)
            global_slot += 1
        end
    end
    return rdata(g)
end

"""
    (pb::Pullback)(y_rdata)

Scalar-input pullback specialization returning reverse data without the generic scatter path.
"""
function (pb::Pullback{F,N,Tuple{T},Tuple{NoFData},Y})(y_rdata) where {F,N,T<:Number,Y}
    return (rdata(zero_tangent(pb.f)), _nfwd_scalar_gradient_rdata(pb, y_rdata))
end

function (pb::Pullback{F,N,P,T,Y})(
    y_rdata
) where {F,N,P<:Tuple{Vararg{Number}},T<:Tuple{Vararg{NoFData}},Y}
    ȳ = tangent(pb.y_fdata, y_rdata)
    # Accumulate gradients in tuple form so multi-scalar pullbacks stay allocation-free.
    grads = _nfwd_zero_scalar_grads(pb.primals, pb.tangents)
    total_dof = _nfwd_input_dof(pb.primals)
    for start_slot in 1:N:total_dof
        seeded_tangents = _nfwd_seed_tangents(pb.primals, Val(N), start_slot)
        _, dy = _nfwd_eval(pb.f, pb.primals, seeded_tangents, Val(N))
        lane_vals = _nfwd_contract_output(ȳ, dy)
        grads = _nfwd_scatter_scalar_chunk(grads, pb.primals, lane_vals, start_slot)
    end
    return tuple(rdata(zero_tangent(pb.f)), _nfwd_gradient_rdatas(grads)...)
end

"""
    (pb::Pullback)(y_rdata)

Generic `nfwd` pullback that reruns chunked NDual passes and scatters VJP contributions
into the cached gradient containers.
"""
function (pb::Pullback{F,N})(y_rdata) where {F,N}
    ȳ = tangent(pb.y_fdata, y_rdata)
    grads = _nfwd_gradient_refs(pb.primals, pb.tangents)
    total_dof = _nfwd_input_dof(pb.primals)
    for start_slot in 1:N:total_dof
        seeded_tangents = _nfwd_seed_tangents(pb.primals, Val(N), start_slot)
        _, dy = _nfwd_eval(pb.f, pb.primals, seeded_tangents, Val(N))
        lane_vals = _nfwd_contract_output(ȳ, dy)
        _nfwd_scatter_chunk!(grads, pb.primals, lane_vals, start_slot)
    end
    # For array inputs the gradient lives in grads[i] (a fresh zeros array). Accumulate it
    # into the fdata (pb.tangents[i]) so that existing fdata contributions are preserved.
    _nfwd_accumulate_array_gradients!(pb.tangents, grads)
    return tuple(rdata(zero_tangent(pb.f)), _nfwd_gradient_rdatas(grads)...)
end

#
# Forward evaluation pipeline
#
# `_nfwd_eval` is the high-level lifted evaluation step used by both the forward rule and
# the reverse pullback. The lift/extract helpers below are the data-conversion pieces it uses.

"""
    _nfwd_eval(f, primals, tangents, ::Val{N})

Evaluate `f` on NDual-lifted primals and extract both primal output and chunked tangent data.
"""
function _nfwd_eval(f, primals::Tuple, tangents::Tuple, ::Val{N}) where {N}
    lifted = map(
        (x, dx) -> _nfwd_lift(_nfwd_check_primal(x), dx, Val(N)), primals, tangents
    )
    return _nfwd_extract(f(lifted...), primals, Val(N))
end

"""
    _nfwd_eval(f, primals::Tuple{<:Number}, tangents, ::Val{N})

Scalar-input specialization of `_nfwd_eval` that avoids tuple-based lifting overhead.
"""
function _nfwd_eval(
    f, primals::Tuple{T}, tangents::Tuple{D}, ::Val{N}
) where {T<:Number,D,N}
    lifted = _nfwd_lift(_nfwd_check_primal(primals[1]), tangents[1], Val(N))
    return _nfwd_extract(f(lifted), primals, Val(N))
end

# Small scalar tuples can allocate when lifted through the generic `map` path above, so
# keep fixed-arity scalar specializations for the common binary/ternary primitive
# wrappers that are expected to stay allocation-free.
function _nfwd_eval(
    f, primals::Tuple{T1,T2}, tangents::Tuple{D1,D2}, ::Val{N}
) where {T1<:Number,T2<:Number,D1,D2,N}
    lifted1 = _nfwd_lift(_nfwd_check_primal(primals[1]), tangents[1], Val(N))
    lifted2 = _nfwd_lift(_nfwd_check_primal(primals[2]), tangents[2], Val(N))
    return _nfwd_extract(f(lifted1, lifted2), primals, Val(N))
end

function _nfwd_eval(
    f, primals::Tuple{T1,T2,T3}, tangents::Tuple{D1,D2,D3}, ::Val{N}
) where {T1<:Number,T2<:Number,T3<:Number,D1,D2,D3,N}
    lifted1 = _nfwd_lift(_nfwd_check_primal(primals[1]), tangents[1], Val(N))
    lifted2 = _nfwd_lift(_nfwd_check_primal(primals[2]), tangents[2], Val(N))
    lifted3 = _nfwd_lift(_nfwd_check_primal(primals[3]), tangents[3], Val(N))
    return _nfwd_extract(f(lifted1, lifted2, lifted3), primals, Val(N))
end

#
# Forward lift/extract helpers
#
# These utilities translate between Mooncake tangent layouts and NDual-based lifted values.

@inline _nfwd_primitive_packed_lane_count(::NoTangent) = nothing
@inline _nfwd_primitive_packed_lane_count(::IEEEFloat) = nothing
@inline _nfwd_primitive_packed_lane_count(::Complex{<:IEEEFloat}) = nothing
@inline _nfwd_primitive_packed_lane_count(::NDual{T,N}) where {T,N} = N
@inline _nfwd_primitive_packed_lane_count(::Complex{NDual{T,N}}) where {T,N} = N
@inline function _nfwd_primitive_packed_lane_count(x::AbstractArray)
    ET = eltype(x)
    return if ET <: NDual
        ET.parameters[2]
    elseif ET <: Complex{<:NDual}
        ET.parameters[1].parameters[2]
    else
        nothing
    end
end
@inline function _nfwd_primitive_packed_lane_count(x::Tuple)
    packed = nothing
    for xi in x
        packed = _nfwd_merge_lane_counts(packed, _nfwd_primitive_packed_lane_count(xi))
    end
    return packed
end
@inline _nfwd_primitive_packed_lane_count(::Any) = nothing

@inline function _nfwd_primitive_chunk_size(::Val{N}, tangents::Tuple) where {N}
    packed = _nfwd_primitive_packed_lane_count(tangents)
    return isnothing(packed) ? Val(N) : Val(packed)
end

@inline function _nfwd_scalar_partials(x::T, dx, ::Val{N}) where {T<:IEEEFloat,N}
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
            "a length-$N tuple/vector of reals. Got $(typeof(dx)): $dx.",
        ),
    )
end

@inline function _nfwd_scalar_partials(
    x::T, dx::NDual{S,N}, ::Val{N}
) where {T<:IEEEFloat,S<:IEEEFloat,N}
    return ntuple(i -> T(Nfwd.ndual_partial(dx, i)), Val(N))
end

@inline function _nfwd_complex_partials(x::Complex{T}, dx, ::Val{N}) where {T<:IEEEFloat,N}
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
            "chunk_size == 1, or a length-$N tuple/vector of complex values. " *
            "Got $(typeof(dx)): $dx.",
        ),
    )
end

@inline function _nfwd_complex_partials(
    x::Complex{T}, dx::Complex{NDual{S,N}}, ::Val{N}
) where {T<:IEEEFloat,S<:IEEEFloat,N}
    return ntuple(i -> T(Nfwd.ndual_partial(real(dx), i)), Val(N)),
    ntuple(i -> T(Nfwd.ndual_partial(imag(dx), i)), Val(N))
end

@inline function _nfwd_array_tangent_dims(x::AbstractArray, ::Val{N}) where {N}
    return (size(x)..., N)
end

@inline function _nfwd_check_array_tangent(
    x::AbstractArray, dx::AbstractArray, ::Val{N}
) where {N}
    if N == 1 && size(dx) == size(x)
        return :plain
    elseif size(dx) == _nfwd_array_tangent_dims(x, Val(N))
        return :chunked
    end
    throw(
        ArgumentError(
            "Expected array tangent for input of size $(size(x)) to have size $(size(x)) " *
            "when chunk_size == 1, or size $(_nfwd_array_tangent_dims(x, Val(N))) " *
            "otherwise. Got size $(size(dx)).",
        ),
    )
end

@inline function _nfwd_lift(x::T, dx, ::Val{N}) where {T<:IEEEFloat,N}
    return NDual{T,N}(x, _nfwd_scalar_partials(x, dx, Val(N)))
end

function _nfwd_lift(x::Complex{T}, dx, ::Val{N}) where {T<:IEEEFloat,N}
    re, im = _nfwd_complex_partials(x, dx, Val(N))
    return Complex(NDual{T,N}(real(x), re), NDual{T,N}(imag(x), im))
end

function _nfwd_lift(x::A, dx::AbstractArray, ::Val{N}) where {ET,A<:AbstractArray{ET},N}
    _nfwd_is_supported_scalar(ET) || _nfwd_input_error(x)
    tangent_layout = _nfwd_check_array_tangent(x, dx, Val(N))
    out = similar(x, ET <: IEEEFloat ? NDual{ET,N} : Complex{NDual{ET.parameters[1],N}})
    @inbounds for I in CartesianIndices(x)
        idx = Tuple(I)
        if tangent_layout === :plain
            out[I] = _nfwd_lift(x[I], dx[I], Val(N))
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

@inline function _nfwd_extract_scalar(d::NDual{T,N}, ::Val{N}) where {T,N}
    return if N == 1
        Nfwd._nfwd_dual_value(d), Nfwd._nfwd_dual_partial(d, 1)
    else
        Nfwd._nfwd_dual_value(d), ntuple(k -> Nfwd._nfwd_dual_partial(d, k), Val(N))
    end
end

@inline function _nfwd_extract_scalar(z::Complex{NDual{T,N}}, ::Val{N}) where {T,N}
    primal = Nfwd._nfwd_dual_value(z)
    tangent = if N == 1
        Nfwd._nfwd_dual_partial(z, 1)
    else
        ntuple(k -> Nfwd._nfwd_dual_partial(z, k), Val(N))
    end
    return primal, tangent
end

@inline function _nfwd_extract(y::NDual{T,N}, ::Val{N}) where {T,N}
    return _nfwd_extract_scalar(y, Val(N))
end

@inline function _nfwd_extract(y::NDual{T,N}, primals::Tuple, ::Val{N}) where {T,N}
    return _nfwd_extract(y, Val(N))
end

@inline function _nfwd_extract(y::Complex{NDual{T,N}}, ::Val{N}) where {T,N}
    return _nfwd_extract_scalar(y, Val(N))
end

@inline function _nfwd_extract(y::Complex{NDual{T,N}}, primals::Tuple, ::Val{N}) where {T,N}
    return _nfwd_extract(y, Val(N))
end

function _nfwd_extract(y::AbstractArray{<:NDual{T,N}}, ::Val{N}) where {T,N}
    primal = similar(y, T)
    tangent = N == 1 ? similar(y, T) : similar(y, T, size(y)..., N)
    @inbounds for I in CartesianIndices(y)
        primal[I] = Nfwd._nfwd_dual_value(y[I])
        idx = Tuple(I)
        if N == 1
            tangent[I] = Nfwd._nfwd_dual_partial(y[I], 1)
        else
            for k in 1:N
                tangent[idx..., k] = Nfwd._nfwd_dual_partial(y[I], k)
            end
        end
    end
    return primal, tangent
end

@inline function _nfwd_extract(
    y::AbstractArray{<:NDual{T,N}}, primals::Tuple, ::Val{N}
) where {T,N}
    return _nfwd_extract(y, Val(N))
end

function _nfwd_extract(
    y::AbstractArray{<:Complex{NDual{Treal,N}}}, ::Val{N}
) where {Treal,N}
    T = Complex{Treal}
    primal = similar(y, T)
    tangent = N == 1 ? similar(y, T) : similar(y, T, size(y)..., N)
    @inbounds for I in CartesianIndices(y)
        primal[I] = Nfwd._nfwd_dual_value(y[I])
        idx = Tuple(I)
        if N == 1
            tangent[I] = Nfwd._nfwd_dual_partial(y[I], 1)
        else
            for k in 1:N
                tangent[idx..., k] = Nfwd._nfwd_dual_partial(y[I], k)
            end
        end
    end
    return primal, tangent
end

@inline function _nfwd_extract(
    y::AbstractArray{<:Complex{NDual{Treal,N}}}, primals::Tuple, ::Val{N}
) where {Treal,N}
    return _nfwd_extract(y, Val(N))
end

# Tuple outputs: recurse into each element; primal and tangent are both tuples.
function _nfwd_extract(y::Tuple, ::Val{N}) where {N}
    pairs = map(yi -> _nfwd_extract(yi, Val(N)), y)
    return map(first, pairs), map(last, pairs)
end

function _nfwd_extract(y::Tuple, primals::Tuple, ::Val{N}) where {N}
    pairs = map(yi -> _nfwd_extract(yi, primals, Val(N)), y)
    return map(first, pairs), map(last, pairs)
end

# Non-NDual outputs: the primal carries no tangent information; synthesize a zero tangent.
# Unsupported types fall through to _nfwd_output_error via the is_supported_primal guard.
function _nfwd_extract(y, ::Val{N}) where {N}
    _nfwd_is_supported_primal(y) || _nfwd_output_error(y)
    return y, _nfwd_zero_output_tangent(y, Val(N))
end

function _nfwd_extract(y, primals::Tuple, ::Val{N}) where {N}
    _nfwd_is_supported_primal(y) || _nfwd_output_error(primals, y)
    return y, _nfwd_zero_output_tangent(y, Val(N))
end

@inline _nfwd_function_gradient(f::CoDual) = tangent(fdata(tangent(f)), NoRData())

@inline function _nfwd_prepare_array_rrule_call(
    rule::RRule, f::CoDual, x::CoDual{A}
) where {A<:Array}
    _nfwd_verify_sig(rule, (f, x))
    _nfwd_check_function_tangent(tangent(f))
    return primal(f), _nfwd_check_primal(primal(x))
end

@inline function _nfwd_array_scalar_rrule_result(
    rule::RRule{sig,N}, f_runtime, x::CoDual{A}, x_primal::A, ::Val{N}
) where {sig,N,T<:IEEEFloat,Nd,A<:Array{T,Nd}}
    grad_arr = _nfwd_lazy_grad_buf!(rule.grad_buf, x_primal)
    y = _nfwd_array_scalar_value_and_gradient(f_runtime, x, rule.buf, grad_arr, Val(N))
    y_cd = CoDual(y, fdata(zero_tangent(y)))
    return y_cd, ArrayScalarPullback(grad_arr, tangent(x))
end

@inline function _nfwd_scalar_value_and_gradient(
    f_runtime, f::CoDual, x::CoDual{T}, ::Val{N}
) where {T<:IEEEFloat,N}
    _nfwd_check_function_tangent(tangent(f))
    x_primal = _nfwd_check_primal(primal(x))
    seed = _nfwd_seed_tangent(x_primal, N, 1, 0)
    y, dy = _nfwd_extract(f_runtime(_nfwd_lift(x_primal, seed, Val(N))), Val(N))
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    x_grad = tangent(fdata(tangent(x)), _nfwd_scalar_lane(dy, Val(N), Val(1)))
    return y, (_nfwd_function_gradient(f), x_grad)
end

@inline function _nfwd_scalar_value_and_gradient(
    f_runtime, f::CoDual, x::CoDual{T}, chunk_size::Integer
) where {T<:IEEEFloat}
    return _nfwd_scalar_value_and_gradient(
        f_runtime, f, x, Val(_nfwd_check_chunk_size(chunk_size))
    )
end

#
# Cached array scalar fast path
#
# A single helper covers both Vector{T} and higher-dimensional Array{T,N} inputs.
# The lifted array is fetched (and lazily allocated) via _nfwd_rrule_lifted!, which
# dispatches on the buf ref type: typed-ref bufs (Vector{T} inputs) stay fully inferred;
# untyped Ref{Any} bufs (N-D arrays) fall back to a runtime type check.
#
# The inner loop uses an incremental seeding strategy:
#   1. The lifted array is initialised once with zero partials — O(n).
#   2. Per chunk: only the C active elements are set to unit-partial form — O(C).
#   3. After f is evaluated, those C elements are reset to zero — O(C).
#
# This replaces the previous O(n×C) approach (fill! the full seed + lift! every chunk)
# with O(n) + O(C)×chunks, matching how ForwardDiff manages its GradientConfig.

# Primary overload: gradient is written to an explicitly provided buffer `grad`.
# `grad` must have the same shape and element type as `primal(x)`.
# Each element of `grad` is written exactly once, so no fill! is needed — callers must
# ensure `grad` is zeroed before use (e.g. via set_to_zero_maybe!! in value_and_gradient!!).
# Does NOT touch `tangent(x)`.
@inline function _nfwd_array_scalar_value_and_gradient(
    f_runtime, x::CoDual{A}, buf::Base.RefValue, grad::A, ::Val{C}
) where {C,T<:IEEEFloat,N,A<:Array{T,N}}
    x_primal = _nfwd_check_primal(primal(x))
    n = length(x_primal)
    lifted = _nfwd_rrule_lifted!(buf, x_primal, Val(C))

    # For multi-chunk cases (DOF > C), init the full lifted array to zero-partial form so
    # that non-seeded slots stay zero across chunks.  For DOF ≤ C (single chunk) every
    # element is seeded immediately, so init is dead and skipped.
    n > C && _nfwd_init_lifted!(lifted, x_primal, Val(C))
    cart = CartesianIndices(x_primal)

    y = zero(T)
    for start_slot in 1:C:n
        _nfwd_seed_lifted_chunk!(lifted, x_primal, cart, start_slot, Val(C))
        y, lane_vals = _nfwd_scalar_lanes(f_runtime(lifted), T, Val(C))
        # Skip unseed on the last chunk: the buffer is always re-seeded (or re-inited) at
        # the start of the next call, so leaving the last chunk seeded is safe.
        start_slot + C <= n &&
            _nfwd_unseed_lifted_chunk!(lifted, x_primal, cart, start_slot, Val(C))
        global_slot = start_slot
        @inbounds for lane_val in lane_vals
            global_slot > n && break
            grad[global_slot] = lane_val  # write (not accumulate); each slot written once
            global_slot += 1
        end
    end

    return y
end

# Backward-compatible overload: writes gradient into tangent(x) directly.
# The caller (value_and_gradient!!) must zero tangent(x) before each call via
# set_to_zero_maybe!!, since the primary overload no longer calls fill!.
# Used by __value_and_gradient!! where the caller expects the returned gradient to be
# the fdata array (i.e. tangent(x) == x_grad after the call).
@inline function _nfwd_array_scalar_value_and_gradient(
    f_runtime, x::CoDual{A}, buf::Base.RefValue, ::Val{C}
) where {C,T<:IEEEFloat,N,A<:Array{T,N}}
    return _nfwd_array_scalar_value_and_gradient(f_runtime, x, buf, tangent(x), Val(C))
end

#
# Workspace helpers (_nfwd_buf_ref / _nfwd_rrule_lifted! /
#                    _nfwd_grad_buf_ref / _nfwd_lazy_grad_buf!)
#
# The rrule buf stores only the lifted Array{NDual{T,C}} — no seed array needed.
# The grad_buf stores a pre-allocated gradient array matching the input shape.
# Two ref types are used for each, matching the frule buf pattern:
#   - Typed-ref (Array{T,Nd} input): fully inferred, no runtime isa check.
#   - Generic Ref{Any} (non-array / unsupported input): workspace type recovered at runtime.

_nfwd_buf_ref(sig, ::Val) = Ref{Any}(nothing)

function _nfwd_buf_ref(::Type{Tuple{F,Array{T,Nd}}}, ::Val{C}) where {F,T<:IEEEFloat,Nd,C}
    return Ref{Union{Nothing,Array{NDual{T,C},Nd}}}(nothing)
end

_nfwd_grad_buf_ref(sig) = Ref{Any}(nothing)

function _nfwd_grad_buf_ref(::Type{Tuple{F,Array{T,Nd}}}) where {F,T<:IEEEFloat,Nd}
    return Ref{Union{Nothing,Array{T,Nd}}}(nothing)
end

@inline function _nfwd_alloc_workspace(::Type{Vector{T}}, dims::Tuple{Int}) where {T}
    return Vector{T}(undef, dims[1])
end

@inline function _nfwd_alloc_workspace(::Type{Array{T,N}}, dims::NTuple{N,Int}) where {T,N}
    return Array{T,N}(undef, dims)
end

@inline function _nfwd_array_workspace!(
    buf::Base.RefValue{Union{Nothing,A}}, ::Type{A}, dims
) where {A<:Array}
    ws = buf[]
    if ws === nothing || size(ws::A) != dims
        ws = _nfwd_alloc_workspace(A, dims)
        buf[] = ws
    end
    return ws::A
end

@inline function _nfwd_array_workspace!(
    buf::Base.RefValue, ::Type{A}, dims
) where {A<:Array}
    ws = buf[]
    if !(ws isa A && size(ws) == dims)
        ws = _nfwd_alloc_workspace(A, dims)
        buf[] = ws
    end
    return ws::A
end

# Typed-ref path: fully inferred (covers all array ranks, including Vector).
@inline function _nfwd_rrule_lifted!(
    buf::Base.RefValue{Union{Nothing,Array{NDual{T,C},N}}}, x::Array{T,N}, ::Val{C}
) where {T<:IEEEFloat,N,C}
    return _nfwd_array_workspace!(buf, Array{NDual{T,C},N}, size(x))
end

# Generic path: buf is Ref{Any}; workspace type recovered at runtime.
@inline function _nfwd_rrule_lifted!(
    buf::Base.RefValue, x::Array{T,N}, ::Val{C}
) where {T<:IEEEFloat,N,C}
    return _nfwd_array_workspace!(buf, Array{NDual{T,C},N}, size(x))
end

# Typed-ref path: fully inferred (covers all array ranks, including Vector).
@inline function _nfwd_lazy_grad_buf!(
    grad_buf::Base.RefValue{Union{Nothing,Array{T,N}}}, x_primal::Array{T,N}
) where {T<:IEEEFloat,N}
    return _nfwd_array_workspace!(grad_buf, Array{T,N}, size(x_primal))
end

# Generic path: grad_buf is Ref{Any}; gradient array type recovered at runtime.
@inline function _nfwd_lazy_grad_buf!(
    grad_buf::Base.RefValue, x_primal::Array{T,N}
) where {T<:IEEEFloat,N}
    return _nfwd_array_workspace!(grad_buf, Array{T,N}, size(x_primal))
end

# Initialise every element of `lifted` to NDual(x[i], 0̄) — O(n), called once per
# value_and_gradient!! invocation.  Chunks then update only C elements each.
@inline function _nfwd_init_lifted!(
    lifted::Array{NDual{T,C},N}, x::Array{T,N}, ::Val{C}
) where {T<:IEEEFloat,C,N}
    z = ntuple(_ -> zero(T), Val(C))
    @inbounds for I in CartesianIndices(x)
        lifted[I] = NDual{T,C}(x[I], z)
    end
    return lifted
end

# Set C elements starting at `start_slot` to unit-partial form — O(C).
@inline function _nfwd_seed_lifted_chunk!(
    lifted::Array{NDual{T,C},N},
    x::Array{T,N},
    cart::CartesianIndices,
    start_slot::Int,
    ::Val{C},
) where {T<:IEEEFloat,C,N}
    @inbounds for lane in 1:C
        gs = start_slot + lane - 1
        gs > length(x) && break
        I = cart[gs]
        lifted[I] = NDual{T,C}(x[I], ntuple(k -> T(k == lane), Val(C)))
    end
    return lifted
end

# Reset those same C elements back to zero-partial form — O(C).
@inline function _nfwd_unseed_lifted_chunk!(
    lifted::Array{NDual{T,C},N},
    x::Array{T,N},
    cart::CartesianIndices,
    start_slot::Int,
    ::Val{C},
) where {T<:IEEEFloat,C,N}
    z = ntuple(_ -> zero(T), Val(C))
    @inbounds for lane in 1:C
        gs = start_slot + lane - 1
        gs > length(x) && break
        I = cart[gs]
        lifted[I] = NDual{T,C}(x[I], z)
    end
    return lifted
end

function _nfwd_lift!(
    out::Array{NDual{T,C}}, x::Array{T}, dx::Array{T}, ::Val{C}
) where {T<:IEEEFloat,C}
    @inbounds for I in CartesianIndices(x)
        idx = Tuple(I)
        out[I] = NDual{T,C}(x[I], ntuple(k -> dx[idx..., k], Val(C)))
    end
    return out
end

#
# Frule lifted-array buffer helpers (_nfwd_frule_buf_ref / _nfwd_frule_lifted!)
#
# The frule receives the seed tangent directly from the caller (as the tangent part of the
# input Dual), so no seed buffer is needed — only a pre-allocated Array{NDual{T,C}} of the
# same shape as the primal input.  Two buf types are used:
#   - Typed-ref (Array{T,Nd} input): fully inferred, no runtime isa check.
#   - Generic Ref{Any} (non-array / unsupported input): workspace type recovered at runtime.

_nfwd_frule_buf_ref(sig, ::Val) = Ref{Any}(nothing)

function _nfwd_frule_buf_ref(
    ::Type{Tuple{F,Array{T,Nd}}}, ::Val{C}
) where {F,T<:IEEEFloat,Nd,C}
    return Ref{Union{Nothing,Array{NDual{T,C},Nd}}}(nothing)
end

# Typed-ref path for Array{T,Nd} inputs (covers all ranks, including Vector).
function _nfwd_frule_lifted!(
    buf::Base.RefValue{Union{Nothing,Array{NDual{T,C},Nd}}},
    x::Array{T,Nd},
    dx::Array{T},
    ::Val{C},
) where {T<:IEEEFloat,Nd,C}
    ws = _nfwd_array_workspace!(buf, Array{NDual{T,C},Nd}, size(x))
    return _nfwd_lift!(ws, x, dx, Val(C))
end

# Generic path for Array{T,Nd} inputs (Ref{Any} buf).
function _nfwd_frule_lifted!(
    buf::Base.RefValue, x::Array{T,Nd}, dx::Array{T}, ::Val{C}
) where {T<:IEEEFloat,Nd,C}
    ws = _nfwd_array_workspace!(buf, Array{NDual{T,C},Nd}, size(x))
    return _nfwd_lift!(ws, x, dx, Val(C))
end

"""
    _nfwd_scalar_lanes(y_raw, ::Type{T}, ::Val{C})

Decode a scalar-output `nfwd` chunk evaluation into its primal value plus the `C`
directional derivatives carried by that chunk. Constant outputs are treated as zero-tangent
outputs, so this helper works for both NDual-carrying and plain scalar results.
"""
@inline function _nfwd_scalar_lanes(
    y_raw::NDual{T,C}, ::Type{T}, ::Val{C}
) where {T<:IEEEFloat,C}
    return Nfwd.ndual_value(y_raw), ntuple(k -> Nfwd.ndual_partial(y_raw, k), Val(C))
end

@inline function _nfwd_scalar_lanes(y_raw, ::Type{T}, ::Val{C}) where {T<:IEEEFloat,C}
    y, dy = _nfwd_extract(y_raw, Val(C))
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    return y, ntuple(k -> _nfwd_scalar_lane(dy, Val(C), Val(k)), Val(C))
end

end
