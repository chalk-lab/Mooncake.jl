module NfwdMooncake

import ..Mooncake
using Base: IEEEFloat
using LinearAlgebra: LinearAlgebra
using Random: AbstractRNG
using ..Nfwd
import ..Mooncake:
    CoDual,
    Dual,
    Lifted,
    NTangent,
    NoFData,
    NoRData,
    NoTangent,
    _count_slots,
    dual_type,
    fdata,
    primal,
    randn_dual,
    rdata,
    tangent,
    tuple_map,
    uninit_dual,
    zero_dual,
    zero_tangent

# ── nfwd: NDual-backed scalar primitive engine ─────────────────────────────────────
# Provides `_nfwd_primitive_frule_call` and `_nfwd_primitive_rrule_call` used by
# `rules_via_nfwd.jl` to implement frule!!/rrule!! for scalar math primitives
# (sin, cos, exp, log, etc.) via NDual lifting.

"""
    Pullback

Concrete pullback object for `nfwd` reverse rules. It stores the primal callable,
primals, input tangents, and output fdata needed to rerun chunked NDual passes during the
reverse sweep.

!!! note
    The scalar specialization `Pullback{F,N,Tuple{T},Tuple{NoFData},Y}` with
    `T<:Number` must remain an `isbits` type for that path to stay allocation-free. The
    generic path (array or multi-input primals) is not isbits and allocates as usual.
    Do not add heap-allocated fields without checking both paths.
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

@inline function _nfwd_primitive_rrule_call(
    ::Val{N}, f::CoDual, x::Vararg{CoDual,M}
) where {M,N}
    _nfwd_check_function_tangent(tangent(f))
    return _nfwd_rrule_call(primal(f), x, Val(N))
end

function build_rrule(
    sig::Type{<:Tuple}; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    N = _nfwd_validate(
        sig,
        isnothing(chunk_size) ? _nfwd_sig_default_chunk_size(sig) : chunk_size;
        debug_mode,
    )
    buf = Ref{Any}(nothing)
    grad_buf = Ref{Any}(nothing)
    return RRule{sig,N,typeof(buf),_nfwd_infer_scalar_output(sig),typeof(grad_buf)}(
        buf, grad_buf
    )
end

function build_rrule(
    f, x...; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    return build_rrule(
        Mooncake._typeof((f, x...)); chunk_size, debug_mode, silence_debug_messages
    )
end

function (rule::RRule{sig,N})(f::CoDual, x::Vararg{CoDual,M}) where {sig,N,M}
    _nfwd_check_function_tangent(tangent(f))
    return _nfwd_rrule_call(primal(f), x, Val(N))
end

@inline function _nfwd_check_function_tangent(df)
    df isa Union{NoTangent,NoFData} && return nothing
    throw(ArgumentError("nfwd does not support differentiating with respect to `f`."))
end

#
# Reverse accumulation utilities
#
# These helpers seed input directions, contract output tangents with cotangents, and scatter
# each chunk's contributions into gradient storage.

@inline function _nfwd_seed_tangent(
    x::IEEEFloat, chunk_size::Int, start_slot::Int, offset::Int
)
    # offset+1 is this scalar's global slot; dir is its 1-indexed position in the chunk.
    global_slot = offset + 1
    dir = global_slot - start_slot + 1
    if chunk_size == 1
        return dir == 1 ? one(x) : zero(x)
    end
    return ntuple(k -> typeof(x)(k == dir), Val(chunk_size))
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
    for dir in 1:chunk_size
        global_slot = start_slot + dir - 1
        if offset < global_slot <= offset + length(x)
            idx = Tuple(cart_inds[global_slot - offset])
            dx[idx..., dir] = one(T)
        end
    end
    return dx
end

function _nfwd_seed_tangent(
    x::AbstractArray{Complex{T}}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    # Each complex element contributes 2 slots in consecutive global positions:
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
    for dir in 1:chunk_size
        global_slot = start_slot + dir - 1
        if offset < global_slot <= offset + 2 * length(x)
            local_slot = global_slot - offset
            elem = cld(local_slot, 2)
            idx = Tuple(cart_inds[elem])
            dx[idx..., dir] =
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
    function scatter_leaf!(x, (offset, remaining_grads))
        g = first(remaining_grads)
        n_slots = _count_slots(x)
        for k in 1:n_slots
            dir = offset + k - start_slot + 1
            if 1 <= dir <= length(dy)
                _nfwd_add_slot!(g, k, dy[dir])
            end
        end
        return nothing, (offset + n_slots, Base.tail(remaining_grads))
    end
    _nfwd_unfold_slots(scatter_leaf!, inputs, (0, grads))
    return nothing
end

@inline _nfwd_gradient_ref(::Tuple{}, ::Tuple{}) = ()
@inline function _nfwd_gradient_ref(primals::Tuple, tangents::Tuple)
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
    return (g, _nfwd_gradient_ref(Base.tail(primals), Base.tail(tangents))...)
end

_nfwd_unwrap_gradient(g::Base.RefValue) = g[]
_nfwd_unwrap_gradient(g) = g

@inline _nfwd_accumulate_gradient!(::Tuple{}, ::Tuple{}) = nothing
@inline function _nfwd_accumulate_gradient!(tangents::Tuple, grads::Tuple)
    fdata = first(tangents)
    grad = first(grads)
    fdata isa AbstractArray && (fdata .+= _nfwd_unwrap_gradient(grad))
    _nfwd_accumulate_gradient!(Base.tail(tangents), Base.tail(grads))
    return nothing
end

@inline _nfwd_zero_gradient(::Tuple{}, ::Tuple{}) = ()
@inline function _nfwd_zero_gradient(primals::Tuple, tangents::Tuple)
    return (
        zero_tangent(first(primals), first(tangents)),
        _nfwd_zero_gradient(Base.tail(primals), Base.tail(tangents))...,
    )
end

@inline function _nfwd_scatter_chunk(
    grads::Tuple, primals::Tuple, dy::Tuple, start_slot::Int
)
    function scatter_leaf(x, (offset, remaining_grads))
        g = first(remaining_grads)
        n_slots = _count_slots(x)
        for k in 1:n_slots
            dir = offset + k - start_slot + 1
            if 1 <= dir <= length(dy)
                g = _nfwd_accumulate_gradient(g, k, dy[dir])
            end
        end
        return g, (offset + n_slots, Base.tail(remaining_grads))
    end
    new_grads, _ = _nfwd_unfold_slots(scatter_leaf, primals, (0, grads))
    return new_grads
end

# `slot` is the 1-based slot index within the scalar/complex input: 1 for the real
# component (or the sole IEEEFloat slot), 2 for the imaginary component of a complex.
# Called from `_nfwd_rdata` with the loop's global_slot, which
# equals the local slot because that path is specialised to a single input at offset 0.
@inline function _nfwd_accumulate_gradient(g::T, slot::Int, v) where {T<:IEEEFloat}
    return slot == 1 ? g + v : g
end

@inline function _nfwd_accumulate_gradient(g::Complex{T}, slot::Int, v) where {T<:IEEEFloat}
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

# Array (real or complex elements): chunk_size=1 → same-shape zero array; chunk_size=N → N extra dirs.
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

# Tuple outputs: contract each element independently and sum dir contributions.
function _nfwd_contract_output(ȳ::Tuple, dy::Tuple)
    length(ȳ) == length(dy) || _nfwd_output_error(dy)
    isempty(ȳ) && return ()
    contributions = map(_nfwd_contract_output, ȳ, dy)
    return foldl((a, b) -> map(+, a, b), contributions)
end

function _nfwd_contract_output(ȳ, dy)
    return _nfwd_output_error(dy)
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

@inline function _nfwd_seed_tangents(
    primals::Tuple, ::Val{N}, start_slot::Int, offset::Int=0
) where {N}
    function seed_leaf(x, off)
        return _nfwd_seed_tangent(x, N, start_slot, off), off + _count_slots(x)
    end
    tangents, _ = _nfwd_unfold_slots(seed_leaf, primals, offset)
    return tangents
end

"""
    _nfwd_rdata(grads::Tuple) -> Tuple

Extract `rdata` from each element of a pre-computed gradient tuple.
"""
@inline _nfwd_rdata(::Tuple{}) = ()
@inline function _nfwd_rdata(grads::Tuple)
    return (rdata(_nfwd_unwrap_gradient(first(grads))), _nfwd_rdata(Base.tail(grads))...)
end

"""
    _nfwd_rdata(pb::Pullback, y_rdata)

Compute reverse data for a scalar-input pullback via chunked forward passes.
"""
function _nfwd_rdata(
    pb::Pullback{F,N,Tuple{T},Tuple{NoFData},Y}, y_rdata
) where {F,N,T<:Number,Y}
    ȳ = tangent(pb.y_fdata, y_rdata)
    x = pb.primals[1]
    g = zero_tangent(x, pb.tangents[1])
    total_slots = _count_slots(x)
    for start_slot in 1:N:total_slots
        tangents = (_nfwd_seed_tangent(x, N, start_slot, 0),)
        _, dy = _nfwd_eval(pb.f, pb.primals, tangents, Val(N))
        dir_vals = _nfwd_contract_output(ȳ, dy)
        global_slot = start_slot
        for dir_val in dir_vals
            g = _nfwd_accumulate_gradient(g, global_slot, dir_val)
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
    return (rdata(zero_tangent(pb.f)), _nfwd_rdata(pb, y_rdata))
end

function (pb::Pullback{F,N,P,T,Y})(
    y_rdata
) where {F,N,P<:Tuple{Vararg{Number}},T<:Tuple{Vararg{NoFData}},Y}
    ȳ = tangent(pb.y_fdata, y_rdata)
    # Accumulate gradients in tuple form so multi-scalar pullbacks stay allocation-free.
    grads = _nfwd_zero_gradient(pb.primals, pb.tangents)
    total_slots = _count_slots(pb.primals)
    for start_slot in 1:N:total_slots
        seeded_tangents = _nfwd_seed_tangents(pb.primals, Val(N), start_slot)
        _, dy = _nfwd_eval(pb.f, pb.primals, seeded_tangents, Val(N))
        dir_vals = _nfwd_contract_output(ȳ, dy)
        grads = _nfwd_scatter_chunk(grads, pb.primals, dir_vals, start_slot)
    end
    return tuple(rdata(zero_tangent(pb.f)), _nfwd_rdata(grads)...)
end

"""
    (pb::Pullback)(y_rdata)

Generic `nfwd` pullback that reruns chunked NDual passes and scatters VJP contributions
into the cached gradient containers.
"""
function (pb::Pullback{F,N})(y_rdata) where {F,N}
    ȳ = tangent(pb.y_fdata, y_rdata)
    grads = _nfwd_gradient_ref(pb.primals, pb.tangents)
    total_slots = _count_slots(pb.primals)
    for start_slot in 1:N:total_slots
        seeded_tangents = _nfwd_seed_tangents(pb.primals, Val(N), start_slot)
        _, dy = _nfwd_eval(pb.f, pb.primals, seeded_tangents, Val(N))
        dir_vals = _nfwd_contract_output(ȳ, dy)
        _nfwd_scatter_chunk!(grads, pb.primals, dir_vals, start_slot)
    end
    # For array inputs the gradient lives in grads[i] (a fresh zeros array). Accumulate it
    # into the fdata (pb.tangents[i]) so that existing fdata contributions are preserved.
    _nfwd_accumulate_gradient!(pb.tangents, grads)
    return tuple(rdata(zero_tangent(pb.f)), _nfwd_rdata(grads)...)
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

@inline function _nfwd_partials(x::T, dx, ::Val{N}) where {T<:IEEEFloat,N}
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

@inline function _nfwd_partials(x::Complex{T}, dx, ::Val{N}) where {T<:IEEEFloat,N}
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

@inline function _nfwd_check_tangent(
    x::AbstractArray, dx::AbstractArray, ::Val{N}
) where {N}
    expected = (size(x)..., N)
    if N == 1 && size(dx) == size(x)
        return :plain
    elseif size(dx) == expected
        return :chunked
    end
    throw(
        ArgumentError(
            "Expected array tangent for input of size $(size(x)) to have size $(size(x)) " *
            "when chunk_size == 1, or size $(expected) " *
            "otherwise. Got size $(size(dx)).",
        ),
    )
end

@inline function _nfwd_lift(x::T, dx, ::Val{N}) where {T<:IEEEFloat,N}
    return NDual{T,N}(x, _nfwd_partials(x, dx, Val(N)))
end

function _nfwd_lift(x::Complex{T}, dx, ::Val{N}) where {T<:IEEEFloat,N}
    re, im = _nfwd_partials(x, dx, Val(N))
    return Complex(NDual{T,N}(real(x), re), NDual{T,N}(imag(x), im))
end

function _nfwd_lift(x::A, dx::AbstractArray, ::Val{N}) where {ET,A<:AbstractArray{ET},N}
    _nfwd_is_supported_scalar(ET) || _nfwd_input_error(x)
    tangent_layout = _nfwd_check_tangent(x, dx, Val(N))
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

function _nfwd_lift(x::Tuple, dx::Tuple, ::Val{N}) where {N}
    return map((xi, dxi) -> _nfwd_lift(xi, dxi, Val(N)), x, dx)
end

@inline function _nfwd_extract(d::NDual{T,N}, ::Val{N}) where {T,N}
    return if N == 1
        Nfwd._nfwd_dual_value(d), Nfwd._nfwd_dual_partial(d, 1)
    else
        Nfwd._nfwd_dual_value(d), ntuple(k -> Nfwd._nfwd_dual_partial(d, k), Val(N))
    end
end

@inline function _nfwd_extract(z::Complex{NDual{T,N}}, ::Val{N}) where {T,N}
    primal = Nfwd._nfwd_dual_value(z)
    tangent = if N == 1
        Nfwd._nfwd_dual_partial(z, 1)
    else
        ntuple(k -> Nfwd._nfwd_dual_partial(z, k), Val(N))
    end
    return primal, tangent
end

@inline function _nfwd_extract(y::NDual{T,N}, primals::Tuple, ::Val{N}) where {T,N}
    return _nfwd_extract(y, Val(N))
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

# Primitive scalar frule/rrule helpers used by rules_via_nfwd.jl.
@inline function _nfwd_primitive_frule_call(
    ::Val{N}, f::Dual, x::Vararg{Dual,M}
) where {M,N}
    _nfwd_check_function_tangent(tangent(f))
    primals = map(primal, x)
    tangents = map(tangent, x)
    y, dy = _nfwd_eval(primal(f), primals, tangents, Val(N))
    return Dual(y, dy)
end

# The generic vararg path can allocate for small scalar primitive wrappers, so keep
# fixed-arity entry points here for common binary/ternary rules such as `atan`, `log`,
# and `clamp`.
@inline function _nfwd_primitive_frule_call(::Val{N}, f::Dual, x1::Dual, x2::Dual) where {N}
    _nfwd_check_function_tangent(tangent(f))
    y, dy = _nfwd_eval(
        primal(f), (primal(x1), primal(x2)), (tangent(x1), tangent(x2)), Val(N)
    )
    return Dual(y, dy)
end

@inline function _nfwd_primitive_frule_call(
    ::Val{N}, f::Dual, x1::Dual, x2::Dual, x3::Dual
) where {N}
    _nfwd_check_function_tangent(tangent(f))
    y, dy = _nfwd_eval(
        primal(f),
        (primal(x1), primal(x2), primal(x3)),
        (tangent(x1), tangent(x2), tangent(x3)),
        Val(N),
    )
    return Dual(y, dy)
end

# ── NDual as a first-class dual type ─────────────────────────────────────────

# dual_type specialisations: IEEEFloat, Complex, and Array variants.
dual_type(::Val{N}, ::Type{T}) where {N,T<:IEEEFloat} = NDual{T,N}
dual_type(::Val{N}, ::Type{Complex{T}}) where {N,T<:IEEEFloat} = Complex{NDual{T,N}}
function dual_type(::Val{N}, ::Type{Array{T,D}}) where {N,T<:IEEEFloat,D}
    return Array{NDual{T,N},D}
end
function dual_type(::Val{N}, ::Type{Array{Complex{T},D}}) where {N,T<:IEEEFloat,D}
    return Array{Complex{NDual{T,N}},D}
end

# MemoryRef / Memory overloads (1.11+): element-wise lifting to NDual containers.
@static if VERSION >= v"1.11-"
    dual_type(::Val{N}, ::Type{MemoryRef{T}}) where {N,T<:IEEEFloat} = MemoryRef{NDual{T,N}}
    dual_type(::Val{N}, ::Type{Memory{T}}) where {N,T<:IEEEFloat} = Memory{NDual{T,N}}
    function dual_type(::Val{N}, ::Type{MemoryRef{Complex{T}}}) where {N,T<:IEEEFloat}
        return MemoryRef{Complex{NDual{T,N}}}
    end
    function dual_type(::Val{N}, ::Type{Memory{Complex{T}}}) where {N,T<:IEEEFloat}
        return Memory{Complex{NDual{T,N}}}
    end
end

# Type-literal overloads: `_uninit_dual(Val(N), ::Type{Array{T,D}})` substitutes
# the inner element type so downstream `Array{T,D}(undef, n)` calls allocate
# NDual containers. The OC slot inferred from `lifted_type` must match that
# substituted V — override `dual_type(Val(N), Type{Array{T,D}})` to mirror the
# runtime substitution.
function dual_type(
    ::Val{N}, ::Type{Type{Array{T,D}}}
) where {N,T<:Union{IEEEFloat,Complex{<:IEEEFloat}},D}
    return Dual{Type{Array{dual_type(Val(N), T),D}},NoTangent}
end
@static if VERSION >= v"1.11-"
    function dual_type(::Val{N}, ::Type{Type{Memory{T}}}) where {N,T<:IEEEFloat}
        return Dual{Type{Memory{dual_type(Val(N), T)}},NoTangent}
    end
    function dual_type(::Val{N}, ::Type{Type{Memory{Complex{T}}}}) where {N,T<:IEEEFloat}
        return Dual{Type{Memory{Complex{dual_type(Val(N), T)}}},NoTangent}
    end
end

# Val{0} ambiguity resolvers: dual_type(Val(0), P) = P for all P.
dual_type(::Val{0}, ::Type{T}) where {T<:IEEEFloat} = T
dual_type(::Val{0}, ::Type{Complex{T}}) where {T<:IEEEFloat} = Complex{T}
dual_type(::Val{0}, ::Type{Array{T,D}}) where {T<:IEEEFloat,D} = Array{T,D}
function dual_type(::Val{0}, ::Type{Array{Complex{T},D}}) where {T<:IEEEFloat,D}
    return Array{Complex{T},D}
end
@static if VERSION >= v"1.11-"
    dual_type(::Val{0}, ::Type{MemoryRef{T}}) where {T<:IEEEFloat} = MemoryRef{T}
    dual_type(::Val{0}, ::Type{Memory{T}}) where {T<:IEEEFloat} = Memory{T}
    function dual_type(::Val{0}, ::Type{MemoryRef{Complex{T}}}) where {T<:IEEEFloat}
        return MemoryRef{Complex{T}}
    end
    function dual_type(::Val{0}, ::Type{Memory{Complex{T}}}) where {T<:IEEEFloat}
        return Memory{Complex{T}}
    end
end

# ── Wrapper-type structural lifts (specialisations) ──────────────────────────
# `Diagonal`, `Adjoint`, `SubArray` are immutable `AbstractArray` wrappers
# holding an inner `Array{T<:IEEEFloat}` field. The generic recursive lift
# in `tangents/dual.jl` (§13 of `notes/mooncake/dual-types.md`) handles
# arbitrary immutable structs via a `NamedTuple{names, Tuple{Vᵢ…}}` inner V.
# The per-wrapper specialisations below provide a wrapper-shaped inner V
# (e.g. `Diagonal{NDual{T,N}, Vector{NDual{T,N}}}`) that rules can pattern-
# match for dispatch on the original wrapper type. Both forms close the
# silent-corruption gap; the per-wrapper form is preferred where the
# wrapper-shape inner V is structurally useful for rule bodies.
#
# Each specialisation maps `W{T, Inner{T}}` →
# `W{NDual{T,N}, Inner{NDual{T,N}}}` at the type level, mirrors that on
# values via `_uninit_dual`, and provides the matching `Lifted` constructor.
# Field extraction through
# `_lgetfield_impl(::AbstractArray{<:NDual}, ::Val{f}) = getfield(x, f)`
# (in `rules/misc.jl`) returns canonical V for any leaf field.

# Diagonal{T, Vector{T}} — single :diag::Vector{T} field.
function dual_type(
    ::Val{N}, ::Type{LinearAlgebra.Diagonal{T,Vector{T}}}
) where {N,T<:IEEEFloat}
    return LinearAlgebra.Diagonal{NDual{T,N},Vector{NDual{T,N}}}
end
function dual_type(
    ::Val{0}, ::Type{LinearAlgebra.Diagonal{T,Vector{T}}}
) where {T<:IEEEFloat}
    return LinearAlgebra.Diagonal{T,Vector{T}}
end

# Adjoint{T, Matrix{T}} — single :parent::Matrix{T} field.
function dual_type(
    ::Val{N}, ::Type{LinearAlgebra.Adjoint{T,Matrix{T}}}
) where {N,T<:IEEEFloat}
    return LinearAlgebra.Adjoint{NDual{T,N},Matrix{NDual{T,N}}}
end
function dual_type(
    ::Val{0}, ::Type{LinearAlgebra.Adjoint{T,Matrix{T}}}
) where {T<:IEEEFloat}
    return LinearAlgebra.Adjoint{T,Matrix{T}}
end

# SubArray{T, D, Array{T,Dp}, I, L} — :parent::Array{T,Dp} is the only
# differentiable field; :indices, :offset1, :stride1 are NoTangent.
function dual_type(
    ::Val{N}, ::Type{SubArray{T,D,Array{T,Dp},I,L}}
) where {N,T<:IEEEFloat,D,Dp,I,L}
    return SubArray{NDual{T,N},D,Array{NDual{T,N},Dp},I,L}
end
function dual_type(
    ::Val{0}, ::Type{SubArray{T,D,Array{T,Dp},I,L}}
) where {T<:IEEEFloat,D,Dp,I,L}
    return SubArray{T,D,Array{T,Dp},I,L}
end

@inline _type_has_ndual(::Type) = false
@inline _type_has_ndual(::Type{<:NDual}) = true
@inline _type_has_ndual(::Type{<:Complex{<:NDual}}) = true
@inline _type_has_ndual(::Type{<:AbstractArray{<:NDual}}) = true
@inline _type_has_ndual(::Type{<:AbstractArray{<:Complex{<:NDual}}}) = true
function _type_has_ndual(::Type{T}) where {T<:Tuple}
    for i in 1:fieldcount(T)
        _type_has_ndual(fieldtype(T, i)) && return true
    end
    return false
end
@inline function _type_has_ndual(::Type{<:Base.Broadcast.Extruded{X}}) where {X}
    return _type_has_ndual(X)
end
@inline function _type_has_ndual(
    ::Type{<:Base.Broadcast.Broadcasted{Style,Axes,F,Args}}
) where {Style,Axes,F,Args}
    return _type_has_ndual(Args)
end

# Base.Broadcast helper structs used by `x .= f.(x)` need wrapper-shaped V
# around lifted Array leaves.
function dual_type(
    ::Val{N}, ::Type{Base.Broadcast.Extruded{Array{T,Dims},K,D}}
) where {N,T<:IEEEFloat,Dims,K,D}
    return Base.Broadcast.Extruded{Array{NDual{T,N},Dims},K,D}
end
function dual_type(
    ::Val{N}, ::Type{Base.Broadcast.Extruded{Array{Complex{T},Dims},K,D}}
) where {N,T<:IEEEFloat,Dims,K,D}
    return Base.Broadcast.Extruded{Array{Complex{NDual{T,N}},Dims},K,D}
end
function dual_type(
    ::Val{0}, ::Type{Base.Broadcast.Extruded{Array{T,Dims},K,D}}
) where {T<:IEEEFloat,Dims,K,D}
    return Base.Broadcast.Extruded{Array{T,Dims},K,D}
end
function dual_type(
    ::Val{0}, ::Type{Base.Broadcast.Extruded{Array{Complex{T},Dims},K,D}}
) where {T<:IEEEFloat,Dims,K,D}
    return Base.Broadcast.Extruded{Array{Complex{T},Dims},K,D}
end
function dual_type(
    ::Val{N}, ::Type{Base.Broadcast.Broadcasted{Style,Axes,F,Args}}
) where {N,Style,Axes,F,Args<:Tuple}
    P = Base.Broadcast.Broadcasted{Style,Axes,F,Args}
    ArgsV = dual_type(Val(N), Args)
    return if ArgsV === Args
        Dual{P,NoTangent}
    elseif _type_has_ndual(ArgsV)
        Base.Broadcast.Broadcasted{Style,Axes,F,ArgsV}
    else
        Dual{P,Mooncake.tangent_type(P)}
    end
end
function dual_type(
    ::Val{0}, ::Type{Base.Broadcast.Broadcasted{Style,Axes,F,Args}}
) where {Style,Axes,F,Args<:Tuple}
    return Base.Broadcast.Broadcasted{Style,Axes,F,Args}
end

# tangent_type(NDual) uses the default struct-based tangent_type. HVP runs
# reverse mode on f(NDual(x,v)), so build_rrule needs tangent_type(NDual) to
# construct CoDuals for NDual-typed arguments.

# primal / tangent accessors
@inline primal(d::NDual) = d.value
@inline tangent(d::NDual{T,N}) where {T,N} = NTangent(d.partials)
@inline Mooncake._field_primal(d::NDual) = primal(d)
@inline Mooncake._field_tangent(d::NDual) = tangent(d)

# __get_primal for NDual-bearing shapes — primal_mode.jl defines the
# `Dual` overload, reverse_mode.jl defines `CoDual` and the generic
# fallback. The bare frule result paths (especially through the generic
# Lifted-aware adapter) need these so `_wrap_rule_result` recovers the
# correct primal type for `Lifted{P_out, N, V}`.
Mooncake.__get_primal(x::NDual) = primal(x)
Mooncake.__get_primal(x::Complex{<:NDual}) = primal(x)
Mooncake.__get_primal(x::AbstractArray{<:NDual}) = map(d -> d.value, x)
function Mooncake.__get_primal(x::AbstractArray{<:Complex{<:NDual}})
    map(z -> complex(z.re.value, z.im.value), x)
end

# `__primal_type` — type-level analog (no value materialization). Mirrors the
# `__get_primal` overloads above so rule bodies that only need
# `__primal_type(_typeof(bare_result))` for `_wrap_rule_result` can use
# `__primal_type(_typeof(bare_result))` instead, skipping the deinterleave.
@inline Mooncake.__primal_type(::Type{NDual{T,N}}) where {T,N} = T
@inline Mooncake.__primal_type(::Type{Complex{NDual{T,N}}}) where {T,N} = Complex{T}
@inline Mooncake.__primal_type(::Type{Array{NDual{T,N},D}}) where {T,N,D} = Array{T,D}
@inline function Mooncake.__primal_type(::Type{Array{Complex{NDual{T,N}},D}}) where {T,N,D}
    return Array{Complex{T},D}
end
@inline function Mooncake.__primal_type(
    ::Type{Base.Broadcast.Extruded{X,K,D}}
) where {X<:Array{<:Union{NDual,Complex{<:NDual}}},K,D}
    return Base.Broadcast.Extruded{Mooncake.__primal_type(X),K,D}
end
@inline function Mooncake.__primal_type(
    ::Type{Base.Broadcast.Broadcasted{Style,Axes,F,Args}}
) where {Style,Axes,F,Args<:Tuple}
    return Base.Broadcast.Broadcasted{Style,Axes,F,Mooncake.__primal_type(Args)}
end
@static if VERSION >= v"1.11-"
    Mooncake.__get_primal(x::Memory{<:NDual{T}}) where {T} = map(d -> d.value, x)
    Mooncake.__get_primal(x::Memory{<:Complex{<:NDual}}) = map(
        z -> complex(z.re.value, z.im.value), x
    )
    function Mooncake.__get_primal(x::MemoryRef{<:NDual{T}}) where {T}
        new_mem = Mooncake.__get_primal(x.mem)
        # `Core.memoryrefoffset(x)` returns ≥1 even for refs into empty memories
        # (the default offset is 1); calling `memoryref(empty_mem, 1)` then
        # BoundsErrors. Use the no-offset form when the source memory is empty.
        return if isempty(new_mem)
            memoryref(new_mem)
        else
            memoryref(new_mem, Core.memoryrefoffset(x))
        end
    end
    function Mooncake.__get_primal(x::MemoryRef{<:Complex{<:NDual}})
        new_mem = Mooncake.__get_primal(x.mem)
        return if isempty(new_mem)
            memoryref(new_mem)
        else
            memoryref(new_mem, Core.memoryrefoffset(x))
        end
    end
    @inline Mooncake.__primal_type(::Type{Memory{NDual{T,N}}}) where {T,N} = Memory{T}
    @inline function Mooncake.__primal_type(::Type{Memory{Complex{NDual{T,N}}}}) where {T,N}
        return Memory{Complex{T}}
    end
    @inline Mooncake.__primal_type(::Type{MemoryRef{NDual{T,N}}}) where {T,N} = MemoryRef{T}
    @inline function Mooncake.__primal_type(
        ::Type{MemoryRef{Complex{NDual{T,N}}}}
    ) where {T,N}
        return MemoryRef{Complex{T}}
    end
end

# _partial_i for NDual vararg transpose — primal_mode.jl defines the Dual overload.
Mooncake._partial_i(x::NDual, i::Int) = x.partials[i]

primal(z::Complex{<:NDual}) = complex(z.re.value, z.im.value)
@inline Mooncake._field_primal(z::Complex{<:NDual}) = primal(z)
@inline Mooncake._field_tangent(z::Complex{<:NDual}) = tangent(z)

# `verify_dual_type` overloads for canonical-V leaf scalars. These complement
# the bare-Tuple/NamedTuple/Lifted overloads in `tangents/dual.jl`. The
# `NDual` and `Complex{<:NDual}` cases satisfy the "verify value is a valid
# inner dual" contract trivially (both are isbits canonical V).
Mooncake.verify_dual_type(::NDual) = true
Mooncake.verify_dual_type(::Complex{<:NDual}) = true
Mooncake.verify_dual_type(::AbstractArray{<:NDual}) = true
Mooncake.verify_dual_type(::AbstractArray{<:Complex{<:NDual}}) = true
function tangent(z::Complex{NDual{T,N}}) where {T,N}
    return NTangent(ntuple(i -> complex(z.re.partials[i], z.im.partials[i]), Val(N)))
end

# NDual constructor from NTangent
function Nfwd.NDual(p::T, t::NTangent{NTuple{N,T}}) where {T<:IEEEFloat,N}
    return NDual{T,N}(p, t.lanes)
end

# ── Lifted inner-type constructors for NDual containers ──────────────────────
# These let `Lifted{P, N}(primal, tangent)` delegate to the inner type's
# constructor without rule bodies choosing the inner shape. Each method covers
# one canonical (primal-shape, tangent-shape) pair the wrapper may hand it.

# Bare NDual: scalar broadcast and NTangent extract (parametric forms).
function NDual{T,N}(value::T, tangent::T) where {T<:IEEEFloat,N}
    return NDual{T,N}(value, ntuple(_ -> tangent, Val(N)))
end
function NDual{T,N}(value::T, tangent::NTangent{NTuple{N,T}}) where {T<:IEEEFloat,N}
    return NDual{T,N}(value, tangent.lanes)
end

# Complex{NDual}: element-wise from Complex primal + Complex tangent.
function (::Type{Complex{NDual{T,N}}})(
    primal::Complex{T}, tangent::Complex{T}
) where {T<:IEEEFloat,N}
    return Complex(
        NDual{T,N}(real(primal), real(tangent)), NDual{T,N}(imag(primal), imag(tangent))
    )
end

# Array{NDual}: element-wise from Array primal + Array tangent.
function (::Type{Array{NDual{T,N},D}})(
    primal::Array{T,D}, tangent::Array{T,D}
) where {T<:IEEEFloat,N,D}
    return map((p, t) -> NDual{T,N}(p, t), primal, tangent)
end
# Width-N counterpart: tangent is `Array{NTangent{NTuple{N,T}}, D}`. Element
# `Array{NDual{T,1}}(p, t)` from `t::Vector{NTangent{Tuple{T}}}` arises in
# vararg-grouping paths that retain `NTangent` even at width=1.
function (::Type{Array{NDual{T,N},D}})(
    primal::Array{T,D}, tangent::Array{NTangent{NTuple{N,T}},D}
) where {T<:IEEEFloat,N,D}
    return map((p, t) -> NDual{T,N}(p, t.lanes), primal, tangent)
end
function (::Type{Array{Complex{NDual{T,N}},D}})(
    primal::Array{Complex{T},D}, tangent::Array{Complex{T},D}
) where {T<:IEEEFloat,N,D}
    return map((p, t) -> Complex{NDual{T,N}}(p, t), primal, tangent)
end

# ── Wrapper-type Lifted constructors ─────────────────────────────────────────
# These mirror the `Array{NDual{T,N},D}(primal, tangent)` form for the array-
# wrapper structural lifts above. Each extracts the wrapper's inner array from
# the primal, pairs it with the matching field of `tangent::Mooncake.Tangent`,
# zips into NDual elements via the inner array's own constructor, and rebuilds
# the wrapper structurally.

function (::Type{LinearAlgebra.Diagonal{NDual{T,N},Vector{NDual{T,N}}}})(
    primal::LinearAlgebra.Diagonal{T,Vector{T}}, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,N}
    diag_t = Mooncake._get_tangent_field(tangent, :diag)
    return LinearAlgebra.Diagonal(Vector{NDual{T,N}}(primal.diag, diag_t))
end

function (::Type{LinearAlgebra.Adjoint{NDual{T,N},Matrix{NDual{T,N}}}})(
    primal::LinearAlgebra.Adjoint{T,Matrix{T}}, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,N}
    parent_t = Mooncake._get_tangent_field(tangent, :parent)
    return LinearAlgebra.Adjoint(Matrix{NDual{T,N}}(parent(primal), parent_t))
end

function (::Type{SubArray{NDual{T,N},D,Array{NDual{T,N},Dp},I,L}})(
    primal::SubArray{T,D,Array{T,Dp},I,L}, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,N,D,Dp,I,L}
    parent_t = Mooncake._get_tangent_field(tangent, :parent)
    parent_lifted = Array{NDual{T,N},Dp}(parent(primal), parent_t)
    return view(
        parent_lifted, primal.indices...
    )::SubArray{NDual{T,N},D,Array{NDual{T,N},Dp},I,L}
end

# Memory / MemoryRef: element-wise (1.11+). MemoryRef constructors lift the
# underlying Memory and re-establish the primal's offset; the (primal, tangent)
# pair is assumed aligned (same length, same offset).
@static if VERSION >= v"1.11-"
    function (::Type{Memory{NDual{T,N}}})(
        primal::Memory{T}, tangent::Memory{T}
    ) where {T<:IEEEFloat,N}
        out = Memory{NDual{T,N}}(undef, length(primal))
        @inbounds for i in eachindex(primal)
            out[i] = NDual{T,N}(primal[i], tangent[i])
        end
        return out
    end
    function (::Type{Memory{Complex{NDual{T,N}}}})(
        primal::Memory{Complex{T}}, tangent::Memory{Complex{T}}
    ) where {T<:IEEEFloat,N}
        out = Memory{Complex{NDual{T,N}}}(undef, length(primal))
        @inbounds for i in eachindex(primal)
            out[i] = Complex{NDual{T,N}}(primal[i], tangent[i])
        end
        return out
    end
    function (::Type{MemoryRef{NDual{T,N}}})(
        primal::MemoryRef{T}, tangent::MemoryRef{T}
    ) where {T<:IEEEFloat,N}
        return memoryref(
            Memory{NDual{T,N}}(primal.mem, tangent.mem), Core.memoryrefoffset(primal)
        )
    end
    function (::Type{MemoryRef{Complex{NDual{T,N}}}})(
        primal::MemoryRef{Complex{T}}, tangent::MemoryRef{Complex{T}}
    ) where {T<:IEEEFloat,N}
        return memoryref(
            Memory{Complex{NDual{T,N}}}(primal.mem, tangent.mem),
            Core.memoryrefoffset(primal),
        )
    end
end

# ── Width helpers for NDual ───────────────────────────────────────────────────

# Width-aware `zero_dual(::Val{N}, x)`, `randn_dual(::Val{N}, rng, x)`,
# `uninit_dual(::Val{N}, x)`: produce a value whose static type matches
# `dual_type(Val(N), typeof(x))`. Partials are zero / random / zero
# respectively (uninit currently uses zero, matching the width-1 fallback).
# Lets external callers construct width-N seed inputs without reaching into
# NDual constructors directly. `Val(0)` is the primal passthrough.

# `_make_partials(gen, ::Type{T}, ::Val{N})` builds an N-tuple of `T` from the
# generator function `gen`, which takes a "lane index" Int. The three entry
# points pick the right `gen`:
#   zero_dual → `_ -> zero(T)`
#   uninit_dual → `_ -> zero(T)` (uninit semantics are zero on the bare path)
#   randn_dual → `_ -> randn(rng, T)`
@inline _make_partials(gen, ::Type{T}, ::Val{N}) where {T,N} = ntuple(gen, Val(N))

# Internal builder for one width-N container construction. `gen` produces one
# scalar partial per call; called once per (element-index, lane-index) for arrays.
@inline _ndual_zero(x::T, ::Val{N}, gen) where {T<:IEEEFloat,N} = NDual{T,N}(
    x, _make_partials(gen, T, Val(N))
)
@inline function _ndual_zero(z::Complex{T}, ::Val{N}, gen) where {T<:IEEEFloat,N}
    re = NDual{T,N}(real(z), _make_partials(gen, T, Val(N)))
    im = NDual{T,N}(imag(z), _make_partials(gen, T, Val(N)))
    return Complex(re, im)
end

# Containers: lift each element via `_ndual_zero`. Memory/MemoryRef are
# 1.11+ only; Array/Vector cover the legacy path.
@inline function _ndual_array(x::Array{T,D}, w::Val, gen) where {T<:IEEEFloat,D}
    out = Array{dual_type(w, T),D}(undef, size(x))
    @inbounds for i in eachindex(x)
        out[i] = _ndual_zero(x[i], w, gen)
    end
    return out
end
@inline function _ndual_array(x::Array{Complex{T},D}, w::Val, gen) where {T<:IEEEFloat,D}
    out = Array{dual_type(w, Complex{T}),D}(undef, size(x))
    @inbounds for i in eachindex(x)
        out[i] = _ndual_zero(x[i], w, gen)
    end
    return out
end
@static if VERSION >= v"1.11-"
    @inline function _ndual_memory(x::Memory{T}, w::Val, gen) where {T<:IEEEFloat}
        out = Memory{dual_type(w, T)}(undef, length(x))
        @inbounds for i in eachindex(x)
            out[i] = _ndual_zero(x[i], w, gen)
        end
        return out
    end
    @inline function _ndual_memory(x::Memory{Complex{T}}, w::Val, gen) where {T<:IEEEFloat}
        out = Memory{dual_type(w, Complex{T})}(undef, length(x))
        @inbounds for i in eachindex(x)
            out[i] = _ndual_zero(x[i], w, gen)
        end
        return out
    end
end

# Val{0} primal passthrough: matches `dual_type(Val(0), P) == P`.
for f in (:zero_dual, :uninit_dual)
    @eval begin
        @inline Mooncake.$f(::Val{0}, x) = x
        @inline Mooncake.$f(::Val{0}, x::IEEEFloat) = x
        @inline Mooncake.$f(::Val{0}, z::Complex{<:IEEEFloat}) = z
        @inline Mooncake.$f(::Val{0}, x::Array{<:IEEEFloat}) = x
        @inline Mooncake.$f(::Val{0}, x::Array{<:Complex{<:IEEEFloat}}) = x
    end
end
@inline Mooncake.randn_dual(::Val{0}, ::AbstractRNG, x) = x
@inline Mooncake.randn_dual(::Val{0}, ::AbstractRNG, x::IEEEFloat) = x
@inline Mooncake.randn_dual(::Val{0}, ::AbstractRNG, z::Complex{<:IEEEFloat}) = z
@inline Mooncake.randn_dual(::Val{0}, ::AbstractRNG, x::Array{<:IEEEFloat}) = x
@inline Mooncake.randn_dual(::Val{0}, ::AbstractRNG, x::Array{<:Complex{<:IEEEFloat}}) = x
@static if VERSION >= v"1.11-"
    for f in (:zero_dual, :uninit_dual)
        @eval begin
            @inline Mooncake.$f(::Val{0}, x::Memory{<:IEEEFloat}) = x
            @inline Mooncake.$f(::Val{0}, x::Memory{<:Complex{<:IEEEFloat}}) = x
            @inline Mooncake.$f(::Val{0}, x::MemoryRef{<:IEEEFloat}) = x
            @inline Mooncake.$f(::Val{0}, x::MemoryRef{<:Complex{<:IEEEFloat}}) = x
        end
    end
    @inline Mooncake.randn_dual(::Val{0}, ::AbstractRNG, x::Memory{<:IEEEFloat}) = x
    @inline Mooncake.randn_dual(
        ::Val{0}, ::AbstractRNG, x::Memory{<:Complex{<:IEEEFloat}}
    ) = x
    @inline Mooncake.randn_dual(::Val{0}, ::AbstractRNG, x::MemoryRef{<:IEEEFloat}) = x
    @inline Mooncake.randn_dual(
        ::Val{0}, ::AbstractRNG, x::MemoryRef{<:Complex{<:IEEEFloat}}
    ) = x
end

# Width-N entry points share the per-element builder. `zero_dual` and `uninit_dual`
# both pick a zero generator (matching the bare `zero_dual(x::NDual)` /
# `uninit_dual(x::NDual)` semantics).

@inline Mooncake.zero_dual(w::Val, x::IEEEFloat) = _ndual_zero(x, w, _ -> zero(typeof(x)))
@inline Mooncake.zero_dual(w::Val, z::Complex{<:IEEEFloat}) = _ndual_zero(
    z, w, _ -> zero(typeof(real(z)))
)
@inline Mooncake.zero_dual(w::Val, x::Array{<:IEEEFloat}) = _ndual_array(
    x, w, _ -> zero(eltype(x))
)
@inline Mooncake.zero_dual(w::Val, x::Array{<:Complex{<:IEEEFloat}}) = _ndual_array(
    x, w, _ -> zero(real(eltype(x)))
)

# Ptr — `tangent_type(Ptr{P}) = Ptr{tangent_type(P)}`, a single ptr, not a
# width-N expansion. The single-arg `zero_tangent(::Ptr)` deliberately
# errors (callers must use the 2-arg form with a known fdata pointer); but
# at lifted-IR boundaries (e.g. `lgetfield(::Memory, :ptr)`) we only need
# a `Dual{Ptr,Ptr{NoTangent}}` slot whose tangent isn't dereferenced.
# Default to a null tangent pointer of the correct type.
@inline function Mooncake.zero_dual(::Val, x::Ptr)
    T = Mooncake.tangent_type(typeof(x))
    return T === NoTangent ? Dual(x, NoTangent()) : Dual(x, T(0))
end

@inline Mooncake.uninit_dual(w::Val, x::IEEEFloat) = _ndual_zero(x, w, _ -> zero(typeof(x)))
@inline Mooncake.uninit_dual(w::Val, z::Complex{<:IEEEFloat}) = _ndual_zero(
    z, w, _ -> zero(typeof(real(z)))
)
@inline Mooncake.uninit_dual(w::Val, x::Array{<:IEEEFloat}) = _ndual_array(
    x, w, _ -> zero(eltype(x))
)
@inline Mooncake.uninit_dual(w::Val, x::Array{<:Complex{<:IEEEFloat}}) = _ndual_array(
    x, w, _ -> zero(real(eltype(x)))
)

@inline Mooncake.randn_dual(w::Val, rng::AbstractRNG, x::IEEEFloat) = _ndual_zero(
    x, w, _ -> randn(rng, typeof(x))
)
@inline Mooncake.randn_dual(w::Val, rng::AbstractRNG, z::Complex{<:IEEEFloat}) = _ndual_zero(
    z, w, _ -> randn(rng, typeof(real(z)))
)
@inline Mooncake.randn_dual(w::Val, rng::AbstractRNG, x::Array{<:IEEEFloat}) = _ndual_array(
    x, w, _ -> randn(rng, eltype(x))
)
@inline Mooncake.randn_dual(w::Val, rng::AbstractRNG, x::Array{<:Complex{<:IEEEFloat}}) = _ndual_array(
    x, w, _ -> randn(rng, real(eltype(x)))
)

@static if VERSION >= v"1.11-"
    @inline Mooncake.zero_dual(w::Val, x::Memory{<:IEEEFloat}) = _ndual_memory(
        x, w, _ -> zero(eltype(x))
    )
    @inline Mooncake.zero_dual(w::Val, x::Memory{<:Complex{<:IEEEFloat}}) = _ndual_memory(
        x, w, _ -> zero(real(eltype(x)))
    )
    @inline Mooncake.uninit_dual(w::Val, x::Memory{<:IEEEFloat}) = _ndual_memory(
        x, w, _ -> zero(eltype(x))
    )
    @inline Mooncake.uninit_dual(w::Val, x::Memory{<:Complex{<:IEEEFloat}}) = _ndual_memory(
        x, w, _ -> zero(real(eltype(x)))
    )
    @inline Mooncake.randn_dual(w::Val, rng::AbstractRNG, x::Memory{<:IEEEFloat}) = _ndual_memory(
        x, w, _ -> randn(rng, eltype(x))
    )
    @inline Mooncake.randn_dual(w::Val, rng::AbstractRNG, x::Memory{<:Complex{<:IEEEFloat}}) = _ndual_memory(
        x, w, _ -> randn(rng, real(eltype(x)))
    )

    # MemoryRef: lift the underlying Memory and rebuild the offset.
    @inline Mooncake.zero_dual(w::Val, x::MemoryRef{<:IEEEFloat}) = memoryref(
        Mooncake.zero_dual(w, x.mem), Core.memoryrefoffset(x)
    )
    @inline Mooncake.zero_dual(w::Val, x::MemoryRef{<:Complex{<:IEEEFloat}}) = memoryref(
        Mooncake.zero_dual(w, x.mem), Core.memoryrefoffset(x)
    )
    @inline Mooncake.uninit_dual(w::Val, x::MemoryRef{<:IEEEFloat}) = memoryref(
        Mooncake.uninit_dual(w, x.mem), Core.memoryrefoffset(x)
    )
    @inline Mooncake.uninit_dual(w::Val, x::MemoryRef{<:Complex{<:IEEEFloat}}) = memoryref(
        Mooncake.uninit_dual(w, x.mem), Core.memoryrefoffset(x)
    )
    @inline Mooncake.randn_dual(w::Val, rng::AbstractRNG, x::MemoryRef{<:IEEEFloat}) = memoryref(
        Mooncake.randn_dual(w, rng, x.mem), Core.memoryrefoffset(x)
    )
    @inline Mooncake.randn_dual(w::Val, rng::AbstractRNG, x::MemoryRef{<:Complex{<:IEEEFloat}}) = memoryref(
        Mooncake.randn_dual(w, rng, x.mem), Core.memoryrefoffset(x)
    )
end

# Identity passes for values already in `dual_type(Val(N), P)` shape, used by
# rules that canonicalise their output via `zero_dual(Val(N), v)`. Both
# `Val{0}` and `Val` (N≥1) overloads are needed: the `Val{0}` form
# disambiguates against the `zero_dual(::Val{0}, x) = x` catch-all above.
for valT in (:(Val{0}), :Val)
    @eval begin
        @inline Mooncake.zero_dual(::$valT, x::NDual) = x
        @inline Mooncake.zero_dual(::$valT, x::Complex{<:NDual}) = x
        @inline Mooncake.zero_dual(::$valT, x::AbstractArray{<:NDual}) = x
        @inline Mooncake.zero_dual(::$valT, x::AbstractArray{<:Complex{<:NDual}}) = x
    end
    @static if VERSION >= v"1.11-"
        @eval begin
            @inline Mooncake.zero_dual(::$valT, x::Memory{<:NDual}) = x
            @inline Mooncake.zero_dual(::$valT, x::Memory{<:Complex{<:NDual}}) = x
            @inline Mooncake.zero_dual(::$valT, x::MemoryRef{<:NDual}) = x
            @inline Mooncake.zero_dual(::$valT, x::MemoryRef{<:Complex{<:NDual}}) = x
        end
    end
end

# Bare-arg `zero_dual` for already-lifted NDual containers — zero each element
# via scalar `zero_dual(::NDual)` (the catch-all would wrap in width-1 Dual).
@inline Mooncake.zero_dual(x::AbstractArray{<:NDual}) = map(zero_dual, x)
@inline Mooncake.zero_dual(x::AbstractArray{<:Complex{<:NDual}}) = map(zero_dual, x)
@static if VERSION >= v"1.11-"
    @inline Mooncake.zero_dual(x::Memory{<:NDual}) = map(zero_dual, x)
    @inline Mooncake.zero_dual(x::Memory{<:Complex{<:NDual}}) = map(zero_dual, x)
    @inline Mooncake.zero_dual(x::MemoryRef{<:NDual}) = memoryref(
        zero_dual(x.mem), Core.memoryrefoffset(x)
    )
    @inline Mooncake.zero_dual(x::MemoryRef{<:Complex{<:NDual}}) = memoryref(
        zero_dual(x.mem), Core.memoryrefoffset(x)
    )
end

zero_dual(x::NDual{T,N}) where {T,N} = NDual{T,N}(x.value, ntuple(_ -> zero(T), Val(N)))

function randn_dual(rng::AbstractRNG, x::NDual{T,N}) where {T,N}
    return NDual{T,N}(x.value, ntuple(_ -> randn(rng, T), Val(N)))
end

function uninit_dual(x::NDual{T,N}) where {T,N}
    return NDual{T,N}(x.value, ntuple(_ -> zero(T), Val(N)))
end

# Complex{NDual} helpers
function zero_dual(z::Complex{NDual{T,N}}) where {T,N}
    return complex(zero_dual(z.re), zero_dual(z.im))
end

function randn_dual(rng::AbstractRNG, z::Complex{NDual{T,N}}) where {T,N}
    return complex(randn_dual(rng, z.re), randn_dual(rng, z.im))
end

function uninit_dual(z::Complex{NDual{T,N}}) where {T,N}
    return complex(uninit_dual(z.re), uninit_dual(z.im))
end

# ── Array{NDual} accessors ───────────────────────────────────────────────────

function primal(a::Array{NDual{T,N},D}) where {T,N,D}
    return map(d -> d.value, a)
end

function tangent(a::Array{NDual{T,1},D}) where {T,D}
    return map(d -> d.partials[1], a)
end

function tangent(a::Array{NDual{T,N},D}) where {T,N,D}
    return NTangent(ntuple(i -> map(d -> d.partials[i], a), Val(N)))
end
@inline Mooncake._field_primal(a::Array{<:NDual}) = primal(a)
@inline Mooncake._field_tangent(a::Array{<:NDual}) = tangent(a)

function primal(a::Array{Complex{NDual{T,N}},D}) where {T,N,D}
    return map(z -> complex(z.re.value, z.im.value), a)
end

function tangent(a::Array{Complex{NDual{T,1}},D}) where {T,D}
    return map(z -> complex(z.re.partials[1], z.im.partials[1]), a)
end

function tangent(a::Array{Complex{NDual{T,N}},D}) where {T,N,D}
    return NTangent(
        ntuple(i -> map(z -> complex(z.re.partials[i], z.im.partials[i]), a), Val(N))
    )
end
@inline Mooncake._field_primal(a::Array{<:Complex{<:NDual}}) = primal(a)
@inline Mooncake._field_tangent(a::Array{<:Complex{<:NDual}}) = tangent(a)

# ── Wrapper-type accessors ───────────────────────────────────────────────────
# `primal` rebuilds the primal wrapper around `primal` of the inner array.
# `tangent` returns a `Tangent` whose differentiable field carries the inner
# array's element-wise `NTangent` representation; non-differentiable fields
# (e.g. `SubArray`'s `:indices`/`:offset1`/`:stride1`) get `NoTangent`.

function primal(
    d::LinearAlgebra.Diagonal{NDual{T,N},Vector{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    return LinearAlgebra.Diagonal(primal(d.diag))
end
function tangent(
    d::LinearAlgebra.Diagonal{NDual{T,N},Vector{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; diag=tangent(d.diag)))
end

function primal(
    a::LinearAlgebra.Adjoint{NDual{T,N},Matrix{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    return LinearAlgebra.Adjoint(primal(parent(a)))
end
function tangent(
    a::LinearAlgebra.Adjoint{NDual{T,N},Matrix{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; parent=tangent(parent(a))))
end

function primal(
    s::SubArray{NDual{T,N},D,Array{NDual{T,N},Dp},I,L}
) where {T<:IEEEFloat,N,D,Dp,I,L}
    return view(primal(parent(s)), s.indices...)
end
function tangent(
    s::SubArray{NDual{T,N},D,Array{NDual{T,N},Dp},I,L}
) where {T<:IEEEFloat,N,D,Dp,I,L}
    return Mooncake.Tangent((;
        parent=tangent(parent(s)),
        indices=NoTangent(),
        offset1=NoTangent(),
        stride1=NoTangent(),
    ))
end

function primal(x::Base.Broadcast.Extruded{<:Array{<:Union{NDual,Complex{<:NDual}}}})
    return Base.Broadcast.Extruded(primal(x.x), x.keeps, x.defaults)
end
function tangent(x::Base.Broadcast.Extruded{<:Array{<:Union{NDual,Complex{<:NDual}}}})
    return Mooncake.Tangent((; x=tangent(x.x), keeps=NoTangent(), defaults=NoTangent()))
end
function primal(x::Base.Broadcast.Broadcasted{Style,Axes,F}) where {Style,Axes,F}
    _has_ndual(x.args) || return x
    args = primal(x.args)
    return Base.Broadcast.Broadcasted{Style}(x.f, args, x.axes)
end
function tangent(x::Base.Broadcast.Broadcasted)
    _has_ndual(x.args) || return NoTangent()
    return Mooncake.Tangent((;
        style=NoTangent(), f=NoTangent(), args=tangent(x.args), axes=NoTangent()
    ))
end

# `_tangent_dir` for wrapper-NDual containers — return the per-direction
# tangent in the wrapper's `tangent_type` shape (a `Tangent`), matching the
# `randn_tangent(::W)` shape used by FD comparison in `test_frule_correctness`.
@inline function _tangent_dir(
    x::LinearAlgebra.Diagonal{NDual{T,N},Vector{NDual{T,N}}}, i
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; diag=_tangent_dir(x.diag, i)))
end
@inline function _tangent_dir(
    x::LinearAlgebra.Adjoint{NDual{T,N},Matrix{NDual{T,N}}}, i
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; parent=_tangent_dir(parent(x), i)))
end
@inline function _tangent_dir(
    x::SubArray{NDual{T,N},D,Array{NDual{T,N},Dp},I,L}, i
) where {T<:IEEEFloat,N,D,Dp,I,L}
    return Mooncake.Tangent((;
        parent=_tangent_dir(parent(x), i),
        indices=NoTangent(),
        offset1=NoTangent(),
        stride1=NoTangent(),
    ))
end

@static if VERSION >= v"1.11-"
    # ── Memory / MemoryRef {NDual} accessors ────────────────────────────────
    # Mirror `__get_primal` shapes: rule bodies that call `primal` / `tangent`
    # on a bare NDual container (e.g. `zero_derivative`'s `map(primal, args)`)
    # need these to avoid `MethodError`s when an unmigrated rule is invoked
    # via the generic Lifted-aware adapter with NDual-bearing arguments.
    primal(m::Memory{<:NDual{T}}) where {T} = map(d -> d.value, m)
    primal(m::Memory{<:Complex{<:NDual}}) = map(z -> complex(z.re.value, z.im.value), m)
    tangent(m::Memory{NDual{T,N}}) where {T,N} = map(d -> NTangent(d.partials), m)
    function tangent(m::Memory{Complex{NDual{T,N}}}) where {T,N}
        return map(
            z -> NTangent(ntuple(i -> complex(z.re.partials[i], z.im.partials[i]), Val(N))),
            m,
        )
    end
    primal(x::MemoryRef{<:NDual}) = memoryref(primal(x.mem), Core.memoryrefoffset(x))
    function primal(x::MemoryRef{<:Complex{<:NDual}})
        return memoryref(primal(x.mem), Core.memoryrefoffset(x))
    end
    tangent(x::MemoryRef{<:NDual}) = memoryref(tangent(x.mem), Core.memoryrefoffset(x))
    function tangent(x::MemoryRef{<:Complex{<:NDual}})
        return memoryref(tangent(x.mem), Core.memoryrefoffset(x))
    end
end

# Val(0) passthrough disambiguations: each width-N specialisation below needs
# a matching `_uninit_dual(::Val{0}, ::SameShape) = v` overload so Aqua's
# ambiguity check sees a strictly-more-specific resolution for Val(0).
# Without these, `_uninit_dual(::Val{0}, ::Array{Float64,1})` etc. are
# ambiguous between the generic `_uninit_dual(::Val{0}, v) = v` in
# `primal_mode.jl` and these width-N specialised overloads.
@inline Mooncake._uninit_dual(::Val{0}, v::Array{T,D}) where {T<:IEEEFloat,D} = v
@inline Mooncake._uninit_dual(
    ::Val{0}, v::LinearAlgebra.Diagonal{T,Vector{T}}
) where {T<:IEEEFloat} = v
@inline Mooncake._uninit_dual(
    ::Val{0}, v::LinearAlgebra.Adjoint{T,Matrix{T}}
) where {T<:IEEEFloat} = v
@inline Mooncake._uninit_dual(
    ::Val{0}, v::SubArray{T,D,Array{T,Dp},I,L}
) where {T<:IEEEFloat,D,Dp,I,L} = v
@inline Mooncake._uninit_dual(
    ::Val{0}, T::Type{Array{S,D}}
) where {S<:Union{IEEEFloat,Complex{<:IEEEFloat}},D} = T
@static if VERSION >= v"1.11-"
    @inline Mooncake._uninit_dual(::Val{0}, T::Type{Memory{S}}) where {S<:IEEEFloat} = T
    @inline Mooncake._uninit_dual(
        ::Val{0}, T::Type{Memory{Complex{S}}}
    ) where {S<:IEEEFloat} = T
end

# _uninit_dual for Array at width N: returns `Lifted{Array{T,D}, N, Array{NDual{T,N},D}}`.
# The outer Lifted wrap matches `lifted_type(Val(N), Array{T,D})`.
function Mooncake._uninit_dual(::Val{N}, v::Array{T,D}) where {N,T<:IEEEFloat,D}
    ndual_arr = Array{NDual{T,N},D}(undef, size(v)...)
    @inbounds for i in eachindex(v)
        ndual_arr[i] = NDual{T,N}(v[i], ntuple(_ -> zero(T), Val(N)))
    end
    return Lifted{Array{T,D},N}(ndual_arr)
end

# Wrapper-type `_uninit_dual`: mirror the structural lifts above by lifting the
# inner array via the existing Array overload, then re-wrapping in the same
# wrapper type. The outer `Lifted{W{T,...}, N}` matches `lifted_type(Val(N), W{T,...})`.
function Mooncake._uninit_dual(
    ::Val{N}, v::LinearAlgebra.Diagonal{T,Vector{T}}
) where {N,T<:IEEEFloat}
    ndual_diag = Mooncake._uninit_dual(Val(N), v.diag).value
    return Lifted{LinearAlgebra.Diagonal{T,Vector{T}},N}(LinearAlgebra.Diagonal(ndual_diag))
end

function Mooncake._uninit_dual(
    ::Val{N}, v::LinearAlgebra.Adjoint{T,Matrix{T}}
) where {N,T<:IEEEFloat}
    ndual_parent = Mooncake._uninit_dual(Val(N), parent(v)).value
    return Lifted{LinearAlgebra.Adjoint{T,Matrix{T}},N}(LinearAlgebra.Adjoint(ndual_parent))
end

function Mooncake._uninit_dual(
    ::Val{N}, v::SubArray{T,D,Array{T,Dp},I,L}
) where {N,T<:IEEEFloat,D,Dp,I,L}
    ndual_parent = Mooncake._uninit_dual(Val(N), parent(v)).value
    return Lifted{SubArray{T,D,Array{T,Dp},I,L},N}(view(ndual_parent, v.indices...))
end

# Lift `Array{T,D}` type literals (mirrors the Memory variant below): substitute
# the inner element type so subsequent `Array{T,D}(undef, n)` calls carry the
# lifted element type inside a `Dual{Type{...}, NoTangent}` envelope. The wrapper
# `Lifted{Type{Array{T,D}}, N}` records the original primal type literal; the
# inner `V` carries the substituted form for downstream constructor lifting.
function Mooncake._uninit_dual(
    w::Val{N}, ::Type{Array{T,D}}
) where {N,T<:Union{IEEEFloat,Complex{<:IEEEFloat}},D}
    inner = Dual(Array{dual_type(w, T),D}, NoTangent())
    return Lifted{Type{Array{T,D}},N}(inner)
end

# Memory container overloads (1.11+): mirror the Array{T} dual_type lift.
@static if VERSION >= v"1.11-"
    function Mooncake._uninit_dual(w::Val{N}, ::Type{Memory{T}}) where {N,T<:IEEEFloat}
        inner = Dual(Memory{dual_type(w, T)}, NoTangent())
        return Lifted{Type{Memory{T}},N}(inner)
    end
    function Mooncake._uninit_dual(
        w::Val{N}, ::Type{Memory{Complex{T}}}
    ) where {N,T<:IEEEFloat}
        inner = Dual(Memory{Complex{dual_type(w, T)}}, NoTangent())
        return Lifted{Type{Memory{Complex{T}}},N}(inner)
    end

    @inline function Mooncake.zero_derivative(
        f::Dual, x1::T, x_rest::Vararg{T}
    ) where {T<:Union{Memory{<:Dual},Memory{<:Complex{<:Dual}}}}
        return zero_dual(
            primal(f)(map(x -> x isa Dual ? primal(x) : x, (x1, x_rest...))...)
        )
    end
end

# ── NDual container dispatch helpers ──────────────────────────────────────────
#
# Centralised here (rather than scattered across rule files) so that adding a new
# container type only requires touching a single file. A missing overload causes
# the FCache forward path to silently fall back to width-1 — see
# test/tangents/dual.jl "NDual dispatch helpers" for the regression guard.

# `_has_ndual` — true if any argument carries NDual data.
@inline _has_ndual() = false
@inline _has_ndual(::NDual, rest...) = true
@inline _has_ndual(::Complex{<:NDual}, rest...) = true
@inline _has_ndual(::AbstractArray{<:NDual}, rest...) = true
@inline _has_ndual(::AbstractArray{<:Complex{<:NDual}}, rest...) = true
@inline _has_ndual(::Dual{<:Any,<:NTangent}, rest...) = true
@inline _has_ndual(x::Dual, rest...) = _has_ndual(tangent(x), rest...)
@inline _has_ndual(x::Tuple, rest...) = _has_ndual(x..., rest...)
@inline _has_ndual(x::NamedTuple, rest...) = _has_ndual(values(x)..., rest...)
@inline _has_ndual(x::Mooncake.Lifted, rest...) = _has_ndual(x.value, rest...)
@inline _has_ndual(x::Base.Broadcast.Extruded, rest...) = _has_ndual(x.x, rest...)
@inline _has_ndual(x::Base.Broadcast.Broadcasted, rest...) = _has_ndual(x.args, rest...)
@inline _has_ndual(_, rest...) = _has_ndual(rest...)
@static if VERSION >= v"1.11-"
    @inline _has_ndual(::MemoryRef{<:NDual}, rest...) = true
    @inline _has_ndual(::MemoryRef{<:Complex{<:NDual}}, rest...) = true
end

# `_HasNDual` is the Union alias used by rule signatures (e.g. memory.jl /
# array_legacy.jl) to dispatch on bare NDual containers without rewrapping.
const _HasNDual = Union{NDual,Complex{<:NDual}}

# `_dual_or_ndual(val, tangent)` — combine a primal field with its tangent into the
# canonical width-aware dual representation: Dual for non-IEEEFloat, NDual for
# IEEEFloat-bearing primals when the tangent is an NTangent.
@inline _dual_or_ndual(val, tangent) = Dual(val, tangent)
@inline _dual_or_ndual(val::IEEEFloat, t::NTangent) = NDual(val, t.lanes)
@inline function _dual_or_ndual(
    val::Complex{T}, t::NTangent{<:NTuple{W}}
) where {T<:IEEEFloat,W}
    lanes = t.lanes
    re = NDual(real(val), ntuple(j -> real(lanes[j]), Val(W)))
    im = NDual(imag(val), ntuple(j -> imag(lanes[j]), Val(W)))
    return Complex(re, im)
end
@static if VERSION >= v"1.11-"
    @inline function _dual_or_ndual(
        val::Memory{T}, t::NTangent{<:NTuple{W}}
    ) where {T<:IEEEFloat,W}
        lanes = t.lanes
        result = Memory{NDual{T,W}}(undef, length(val))
        @inbounds for i in eachindex(val)
            result[i] = NDual(val[i], ntuple(j -> lanes[j][i], Val(W)))
        end
        return result
    end
end

# `_ndual_width` — extract Val(W) from any NDual-bearing argument; used by `_new_`
# to size NTangent output. Errors loudly when called with no NDual arguments so
# missing overloads fail fast rather than producing wrong-width tangents.
@inline _ndual_width(x::Tuple, rest...) = _ndual_width(x..., rest...)
@inline _ndual_width(::NDual{T,W}, rest...) where {T,W} = Val(W)
@inline _ndual_width(::Complex{NDual{T,W}}, rest...) where {T,W} = Val(W)
@inline _ndual_width(::AbstractArray{NDual{T,W}}, rest...) where {T,W} = Val(W)
@inline _ndual_width(::AbstractArray{Complex{NDual{T,W}}}, rest...) where {T,W} = Val(W)
@inline _ndual_width(::Dual{<:Any,NTangent{L}}, rest...) where {L<:Tuple} = Val(
    fieldcount(L)
)
@inline _ndual_width(x::Dual, rest...) = _ndual_width(tangent(x), rest...)
@inline _ndual_width(x::Base.Broadcast.Extruded, rest...) = _ndual_width(x.x, rest...)
@inline _ndual_width(x::Base.Broadcast.Broadcasted, rest...) = _ndual_width(x.args, rest...)
@inline _ndual_width(_, rest...) = _ndual_width(rest...)
@inline _ndual_width() = error("_ndual_width called with no NDual arguments")
@static if VERSION >= v"1.11-"
    @inline _ndual_width(::MemoryRef{NDual{T,W}}, rest...) where {T,W} = Val(W)
    @inline _ndual_width(::MemoryRef{Complex{NDual{T,W}}}, rest...) where {T,W} = Val(W)
end

# `_ndual_primal` — extract the primal value from any NDual-bearing representation.
# Used by `_new_` to construct the primal struct without partials.
@inline _ndual_primal(x::Dual) = primal(x)
@inline _ndual_primal(x::NDual) = primal(x)
@inline _ndual_primal(x::Complex{<:NDual}) = primal(x)
# Wrapper-shape AbstractArrays: dispatch to the per-wrapper `primal` overload
# (Diagonal, Adjoint, SubArray) which preserves the wrapper structure.
# `map(d -> d.value, ::Adjoint)` would return a Matrix, breaking the shape
# pairing with `_tangent_dir(::Adjoint{<:NDual})` which returns a Tangent.
@inline _ndual_primal(d::LinearAlgebra.Diagonal{<:NDual}) = primal(d)
@inline _ndual_primal(a::LinearAlgebra.Adjoint{<:NDual}) = primal(a)
@inline _ndual_primal(s::SubArray{<:NDual}) = primal(s)
@inline _ndual_primal(x::AbstractArray{<:NDual}) = map(d -> d.value, x)
@inline _ndual_primal(x::AbstractArray{<:Complex{<:NDual}}) = map(
    z -> complex(z.re.value, z.im.value), x
)
@inline _ndual_primal(x::Tuple) = map(_ndual_primal, x)
@inline _ndual_primal(x::NamedTuple{names}) where {names} = NamedTuple{names}(
    map(_ndual_primal, values(x))
)
@inline function _ndual_primal(x::Base.Broadcast.Extruded)
    return Base.Broadcast.Extruded(_ndual_primal(x.x), x.keeps, x.defaults)
end
@inline function _ndual_primal(
    x::Base.Broadcast.Broadcasted{Style,Axes,F}
) where {Style,Axes,F}
    args = _ndual_primal(x.args)
    return Base.Broadcast.Broadcasted{Style}(x.f, args, x.axes)
end
@inline _ndual_primal(x::Mooncake.Lifted) = _ndual_primal(x.value)
@static if VERSION >= v"1.11-"
    @inline _ndual_primal(x::MemoryRef{<:NDual}) = primal(x)
    @inline _ndual_primal(x::MemoryRef{<:Complex{<:NDual}}) = primal(x)
end
@inline _ndual_primal(x) = x

# `_tangent_dir(x, i)` — extract the i-th direction tangent from any NDual-bearing
# representation. Used by `_new_` to assemble per-direction NTangent lanes.
@inline _tangent_dir(x::NDual, i) = x.partials[i]
@inline _tangent_dir(x::Complex{<:NDual}, i) = complex(x.re.partials[i], x.im.partials[i])
@inline _tangent_dir(x::Dual{<:Any,<:NTangent}, i) = tangent(x).lanes[i]
@inline _tangent_dir(x::Dual{<:Any,<:Tuple}, i) = map(
    t -> _tangent_dir_elem(t, i), tangent(x)
)
@inline _tangent_dir(x::Dual, _) = tangent(x)
@inline _tangent_dir(x::AbstractArray{NDual{T,N}}, i) where {T,N} = map(
    d -> d.partials[i], x
)
@inline _tangent_dir(x::AbstractArray{Complex{NDual{T,N}}}, i) where {T,N} = map(
    z -> complex(z.re.partials[i], z.im.partials[i]), x
)
# Mirror `tangent_type(P<:Tuple)`'s all-NoTangent fold: if every element's
# direction tangent is `NoTangent`, return a single `NoTangent` so that
# downstream `build_output_tangent` can place it in a `NoTangent` field
# without a `Tuple{NoTangent...}` → `NoTangent` convert error.
@inline function _tangent_dir(x::Tuple, i)
    inner = map(xi -> _tangent_dir(xi, i), x)
    return inner isa Tuple{Vararg{NoTangent}} ? NoTangent() : inner
end
@inline function _tangent_dir(x::NamedTuple{names}, i) where {names}
    inner = _tangent_dir(values(x), i)
    return inner isa NoTangent ? inner : NamedTuple{names}(inner)
end
@inline function _tangent_dir(x::Base.Broadcast.Extruded, i)
    return Mooncake.Tangent((;
        x=_tangent_dir(x.x, i), keeps=NoTangent(), defaults=NoTangent()
    ))
end
@inline function _tangent_dir(x::Base.Broadcast.Broadcasted, i)
    return Mooncake.Tangent((;
        style=NoTangent(), f=NoTangent(), args=_tangent_dir(x.args, i), axes=NoTangent()
    ))
end
@static if VERSION >= v"1.11-"
    @inline function _tangent_dir(x::MemoryRef{NDual{T,N}}, i) where {T,N}
        return memoryref(map(d -> d.partials[i], x.mem), Core.memoryrefoffset(x))
    end
    @inline function _tangent_dir(x::MemoryRef{Complex{NDual{T,N}}}, i) where {T,N}
        return memoryref(
            map(z -> complex(z.re.partials[i], z.im.partials[i]), x.mem),
            Core.memoryrefoffset(x),
        )
    end
end
@inline _tangent_dir(x, _) = zero_tangent(x)

# Lifted slot-shape passthrough: extract the i-th direction tangent from the
# Lifted's inner V. Mirrors how `tangent(::Lifted)` passes through to the
# inner V's tangent.
@inline _tangent_dir(x::Mooncake.Lifted, i) = _tangent_dir(x.value, i)

# Immutable struct-primal slot with NamedTuple inner V (recursive lift,
# see dual.jl): build the per-direction tangent via `build_output_tangent`
# so the result matches `tangent_type(P)` (Tangent shape with
# `PossiblyUninitTangent` wrapping where present). For NamedTuple primals
# the bare NamedTuple result is canonical.
@inline function _tangent_dir(d::Mooncake.Lifted{P,N,V}, i) where {P,N,V<:NamedTuple}
    P <: NamedTuple && return _tangent_dir(d.value, i)
    return _build_struct_tangent_dir(P, d.value, i)
end
@generated function _build_struct_tangent_dir(
    ::Type{P}, value::V, i
) where {P,V<:NamedTuple{names}} where {names}
    primal_exprs = [:(_ndual_primal(value.$n)) for n in names]
    tangent_exprs = [:(_tangent_dir(value.$n, i)) for n in names]
    return :(Mooncake.build_output_tangent(
        $P, ($(primal_exprs...),), ($(tangent_exprs...),)
    ))
end

@inline _tangent_dir_elem(t::NTangent, i) = t.lanes[i]
@inline _tangent_dir_elem(t, _) = t

# `_find_ndual_memref` — locate the NDual-typed `MemoryRef` in an argument list,
# used by the `Array` branch of `_new_` to determine the NDual element type for
# the result container.
@inline _find_ndual_memref(_, rest...) = _find_ndual_memref(rest...)
@inline _find_ndual_memref() = nothing
@static if VERSION >= v"1.11-"
    @inline _find_ndual_memref(x::MemoryRef{<:Union{NDual,Complex{<:NDual}}}, rest...) = x
end

# Width-N counterpart to the `Array{<:Dual}` zero_derivative overload in
# `tools_for_rules.jl`. NDual arrays carry tangents in their elements; the
# `primal(::Array{NDual})` overload (above) extracts the underlying primal
# array, and the result is wrapped at the input width via `_ndual_width`.
@inline function Mooncake.zero_derivative(
    f::Dual, x1::T, x_rest::Vararg{T}
) where {T<:Union{Array{<:NDual},Array{<:Complex{<:NDual}}}}
    w = _ndual_width(x1, x_rest...)
    return Mooncake.zero_dual(w, primal(f)(map(primal, (x1, x_rest...))...))
end

# `zero_derivative(f::Dual, ::Tuple)` for chunked-path callers that pass a bare
# tuple of lifted args (e.g. `Broadcast.eltypes((arr, scalar))`); concrete tuple
# types lift element-wise so the tuple itself stays unwrapped. `_ndual_width`
# errors loudly if `x` carries no NDual content, matching the
# `tools_for_rules.jl:290` MethodError contract for non-lifted args.
@inline function Mooncake.zero_derivative(f::Dual, x::Tuple)
    return Mooncake.zero_dual(_ndual_width(x), primal(f)(_ndual_primal(x)))
end

end
