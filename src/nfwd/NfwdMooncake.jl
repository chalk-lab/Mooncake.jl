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

# Val{0} ambiguity resolvers: dual_type(Val(0), P) = P for all P. Also
# resolves the `Type{Type{...}}`-slot specialisations against the generic
# `dual_type(::Val{0}, ::Type{Type{P}})` catch-all in `src/tangents/dual.jl`.
dual_type(::Val{0}, ::Type{T}) where {T<:IEEEFloat} = T
dual_type(::Val{0}, ::Type{Complex{T}}) where {T<:IEEEFloat} = Complex{T}
dual_type(::Val{0}, ::Type{Array{T,D}}) where {T<:IEEEFloat,D} = Array{T,D}
function dual_type(::Val{0}, ::Type{Array{Complex{T},D}}) where {T<:IEEEFloat,D}
    return Array{Complex{T},D}
end
function dual_type(
    ::Val{0}, ::Type{Type{Array{T,D}}}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},D}
    return Type{Array{T,D}}
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
    function dual_type(::Val{0}, ::Type{Type{Memory{T}}}) where {T<:IEEEFloat}
        return Type{Memory{T}}
    end
    function dual_type(::Val{0}, ::Type{Type{Memory{Complex{T}}}}) where {T<:IEEEFloat}
        return Type{Memory{Complex{T}}}
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

# Adjoint{T, <:AbstractArray{T}} — single :parent field. Canonical NDual
# V is `Adjoint{NDual{T,N}, dual_type(Val(N), P)}`; the parent's own
# canonical V (Array, SubArray, …) is computed recursively, mirroring
# the Transpose broadening. Seed factories route through the generic
# `Adjoint{NDual{T,N}, V_parent}(primal, tangent::Tangent)` ctor below.
#
# Note: the broadening from `<:Array{T}` to `<:AbstractArray{T}` keeps
# parity with the Transpose template but is not strictly necessary — the
# structural NamedTuple lift already produced a recursively coherent V
# for non-Array parents. Tighten back to `<:Array{T}` if the wrapper-
# shaped form turns out to cause issues; rule code that relies on
# matching `Adjoint{<:NDual}` patterns is the only known beneficiary.
function dual_type(
    ::Val{N}, ::Type{LinearAlgebra.Adjoint{T,P}}
) where {N,T<:IEEEFloat,P<:AbstractArray{T}}
    return LinearAlgebra.Adjoint{NDual{T,N},dual_type(Val(N), P)}
end
function dual_type(
    ::Val{0}, ::Type{LinearAlgebra.Adjoint{T,P}}
) where {T<:IEEEFloat,P<:AbstractArray{T}}
    return LinearAlgebra.Adjoint{T,P}
end
@inline function Mooncake.zero_dual(
    w::Val{N}, x::LinearAlgebra.Adjoint{T,P}
) where {N,T<:IEEEFloat,P<:AbstractArray{T}}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.zero_tangent(x))::V
end
@inline function Mooncake.uninit_dual(
    w::Val{N}, x::LinearAlgebra.Adjoint{T,P}
) where {N,T<:IEEEFloat,P<:AbstractArray{T}}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.uninit_tangent(x))::V
end
@inline function Mooncake.randn_dual(
    w::Val{N}, rng::AbstractRNG, x::LinearAlgebra.Adjoint{T,P}
) where {N,T<:IEEEFloat,P<:AbstractArray{T}}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.randn_tangent(rng, x))::V
end
@inline Mooncake.zero_dual(
    ::Val{0}, x::LinearAlgebra.Adjoint{T,<:AbstractArray{T}}
) where {T<:IEEEFloat} = x
@inline Mooncake.uninit_dual(
    ::Val{0}, x::LinearAlgebra.Adjoint{T,<:AbstractArray{T}}
) where {T<:IEEEFloat} = x
@inline Mooncake.randn_dual(
    ::Val{0}, ::AbstractRNG, x::LinearAlgebra.Adjoint{T,<:AbstractArray{T}}
) where {T<:IEEEFloat} = x

# `data::Matrix{T}`-shaped LinearAlgebra wrappers: canonical NDual-element
# form mirrors the Adjoint/Transpose template. All five wrappers below
# have an identical single `data::Matrix{T}` layout, so the
# dual_type/constructor/accessor blocks generate uniformly.
for W in (
    :(LinearAlgebra.UpperTriangular),
    :(LinearAlgebra.LowerTriangular),
    :(LinearAlgebra.UnitUpperTriangular),
    :(LinearAlgebra.UnitLowerTriangular),
    :(LinearAlgebra.UpperHessenberg),
)
    @eval begin
        function dual_type(::Val{N}, ::Type{$(W){T,P}}) where {N,T<:IEEEFloat,P<:Matrix{T}}
            return $(W){NDual{T,N},Matrix{NDual{T,N}}}
        end
        function dual_type(::Val{0}, ::Type{$(W){T,P}}) where {T<:IEEEFloat,P<:Matrix{T}}
            return $(W){T,P}
        end
    end
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
# Complex-element variant.
function dual_type(
    ::Val{N}, ::Type{SubArray{Complex{T},D,Array{Complex{T},Dp},I,L}}
) where {N,T<:IEEEFloat,D,Dp,I,L}
    return SubArray{Complex{NDual{T,N}},D,Array{Complex{NDual{T,N}},Dp},I,L}
end
function dual_type(
    ::Val{0}, ::Type{SubArray{Complex{T},D,Array{Complex{T},Dp},I,L}}
) where {T<:IEEEFloat,D,Dp,I,L}
    return SubArray{Complex{T},D,Array{Complex{T},Dp},I,L}
end

# ReshapedArray with any `AbstractArray{T}` parent: canonical NDual-element
# form `ReshapedArray{NDual{T,N}, D, dual_type(Val(N), P), MI}`. The parent's
# own canonical V (Array, SubArray, …) is computed recursively; the
# `ReshapedArray{NDual{T,N}, …}(primal, tangent)` constructor below delegates
# the parent build to `V_parent`'s own (primal, tangent) constructor, so
# this works uniformly over any AbstractArray parent. Mirrors the Transpose
# template at this file's `Transpose` block.
#
# Previous attempt (`111512346`) restricted to `P<:Array{T}` parents only and
# was reverted (`e98728622`) because lapack tests use `ReshapedArray{T,D,
# <:SubArray{T,…},MI}`. The parameterisation over `P<:AbstractArray{T}` here
# admits any parent that itself has a canonical NDual lift.
function dual_type(
    ::Val{N}, ::Type{Base.ReshapedArray{T,D,P,MI}}
) where {N,T<:IEEEFloat,D,P<:AbstractArray{T},MI}
    return Base.ReshapedArray{NDual{T,N},D,dual_type(Val(N), P),MI}
end
function dual_type(
    ::Val{0}, ::Type{Base.ReshapedArray{T,D,P,MI}}
) where {T<:IEEEFloat,D,P<:AbstractArray{T},MI}
    return Base.ReshapedArray{T,D,P,MI}
end
# Complex-element variant.
function dual_type(
    ::Val{N}, ::Type{Base.ReshapedArray{Complex{T},D,P,MI}}
) where {N,T<:IEEEFloat,D,P<:AbstractArray{Complex{T}},MI}
    return Base.ReshapedArray{Complex{NDual{T,N}},D,dual_type(Val(N), P),MI}
end
function dual_type(
    ::Val{0}, ::Type{Base.ReshapedArray{Complex{T},D,P,MI}}
) where {T<:IEEEFloat,D,P<:AbstractArray{Complex{T}},MI}
    return Base.ReshapedArray{Complex{T},D,P,MI}
end

# Durable bare-`Dual` width-1 exceptions.
#
#   - `ReinterpretArray`: the non-trivial use is `T !== S` (e.g.,
#     `reinterpret(Complex{Float64}, ::Vector{Float64})`). [Phase 2 TODO:
#     The byte layout actually aligns — `Vector{NDual{T,N}}` reinterpreted
#     as `Vector{Complex{NDual{T,N}}}` matches the user's
#     `Vector{T} → Vector{Complex{T}}` cast. Migrate to canonical NDual
#     form using the Transpose/ReshapedArray template once a test exposes
#     the actual non-trivial use case.]
#
# Pinned by the `width-1 wrapper bare-Dual durable exceptions` testset
# in `test/tangents/dual.jl`.
for Wrapper in (
    :(Base.ReinterpretArray{T,D,S,P,W} where {T<:IEEEFloat,D,S,P,W}),
    :(LinearAlgebra.Symmetric{T,P} where {T<:IEEEFloat,P<:StridedMatrix{T}}),
    :(LinearAlgebra.Hermitian{T,P} where {T<:IEEEFloat,P<:StridedMatrix{T}}),
)
    @eval begin
        function dual_type(
            ::Val{1}, ::Type{$(Wrapper.args[1])}
        ) where {$(Wrapper.args[2:end]...)}
            return Dual{$(Wrapper.args[1]),Mooncake.tangent_type($(Wrapper.args[1]))}
        end
        function dual_type(
            ::Val{N}, ::Type{$(Wrapper.args[1])}
        ) where {N,$(Wrapper.args[2:end]...)}
            return Dual{
                $(Wrapper.args[1]),Mooncake.tangent_type(Val(N), $(Wrapper.args[1]))
            }
        end
        dual_type(::Val{0}, ::Type{$(Wrapper.args[1])}) where {$(Wrapper.args[2:end]...)} =
            $(Wrapper.args[1])
    end
end
# Transpose with any `AbstractArray{T}` parent: canonical NDual-element
# form `Transpose{NDual{T,N}, dual_type(Val(N), P)}`. The parent's own
# canonical V (Array, SubArray, …) is computed recursively; the
# `Transpose{NDual{T,N}, V_parent}(primal, tangent)` constructor below
# delegates the parent build to `V_parent`'s own (primal, tangent)
# constructor, so this seed/conversion path works for any parent type
# that itself has a canonical NDual lift.
function dual_type(
    ::Val{N}, ::Type{LinearAlgebra.Transpose{T,P}}
) where {N,T<:IEEEFloat,P<:AbstractArray{T}}
    return LinearAlgebra.Transpose{NDual{T,N},dual_type(Val(N), P)}
end
function dual_type(
    ::Val{0}, ::Type{LinearAlgebra.Transpose{T,P}}
) where {T<:IEEEFloat,P<:AbstractArray{T}}
    return LinearAlgebra.Transpose{T,P}
end
@inline function Mooncake.zero_dual(
    w::Val{N}, x::LinearAlgebra.Transpose{T,P}
) where {N,T<:IEEEFloat,P<:AbstractArray{T}}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.zero_tangent(x))::V
end
@inline function Mooncake.uninit_dual(
    w::Val{N}, x::LinearAlgebra.Transpose{T,P}
) where {N,T<:IEEEFloat,P<:AbstractArray{T}}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.uninit_tangent(x))::V
end
@inline function Mooncake.randn_dual(
    w::Val{N}, rng::AbstractRNG, x::LinearAlgebra.Transpose{T,P}
) where {N,T<:IEEEFloat,P<:AbstractArray{T}}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.randn_tangent(rng, x))::V
end
# Val(0) passthroughs disambiguate against the generic `Val{0}` primal
# passthrough below.
@inline Mooncake.zero_dual(
    ::Val{0}, x::LinearAlgebra.Transpose{T,<:AbstractArray{T}}
) where {T<:IEEEFloat} = x
@inline Mooncake.uninit_dual(
    ::Val{0}, x::LinearAlgebra.Transpose{T,<:AbstractArray{T}}
) where {T<:IEEEFloat} = x
@inline Mooncake.randn_dual(
    ::Val{0}, ::AbstractRNG, x::LinearAlgebra.Transpose{T,<:AbstractArray{T}}
) where {T<:IEEEFloat} = x

# `Transpose{NDual{T,N}, V_parent}(primal, tangent::Tangent)` ctor — the
# Lifted bridge and seed factories route here. Delegates the parent lift
# to `V_parent`'s own (primal, tangent) constructor (defined for `Array`,
# `SubArray`, and other shapes that have a canonical NDual form), so
# Transpose works uniformly over any AbstractArray parent.
function (::Type{LinearAlgebra.Transpose{NDual{T,N},V_parent}})(
    primal::LinearAlgebra.Transpose{T,P}, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,N,P<:AbstractArray{T},V_parent<:AbstractArray{NDual{T,N}}}
    parent_t = Mooncake._get_tangent_field(tangent, :parent)
    parent_lifted = V_parent(parent(primal), parent_t)
    return transpose(parent_lifted)::LinearAlgebra.Transpose{NDual{T,N},V_parent}
end

# ReshapedArray canonical V ctor + seed factories — same template as Transpose.
# Delegates the parent lift to V_parent's own (primal, tangent) ctor.
function (::Type{Base.ReshapedArray{NDual{T,N},D,V_parent,MI}})(
    primal::Base.ReshapedArray{T,D,P,MI}, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,N,D,P<:AbstractArray{T},V_parent<:AbstractArray{NDual{T,N}},MI}
    parent_t = Mooncake._get_tangent_field(tangent, :parent)
    parent_lifted = V_parent(parent(primal), parent_t)
    return Base.ReshapedArray{NDual{T,N},D,V_parent,MI}(
        parent_lifted, primal.dims, primal.mi
    )
end
function (::Type{Base.ReshapedArray{Complex{NDual{T,N}},D,V_parent,MI}})(
    primal::Base.ReshapedArray{Complex{T},D,P,MI}, tangent::Mooncake.Tangent
) where {
    T<:IEEEFloat,
    N,
    D,
    P<:AbstractArray{Complex{T}},
    V_parent<:AbstractArray{Complex{NDual{T,N}}},
    MI,
}
    parent_t = Mooncake._get_tangent_field(tangent, :parent)
    parent_lifted = V_parent(parent(primal), parent_t)
    return Base.ReshapedArray{Complex{NDual{T,N}},D,V_parent,MI}(
        parent_lifted, primal.dims, primal.mi
    )
end
# Lifted bridge — mirrors Transpose's `Lifted{P,1}(primal, ::Tangent)` shim.
@inline function Mooncake.Lifted{P,1}(
    primal::P, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,P<:Base.ReshapedArray{T,<:Any,<:AbstractArray{T}}}
    InnerT = Mooncake.dual_type(Val(1), P)
    return Mooncake.Lifted{P,1,InnerT}(InnerT(primal, tangent))
end
@inline function Mooncake.Lifted{P,1}(
    primal::P, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,P<:Base.ReshapedArray{Complex{T},<:Any,<:AbstractArray{Complex{T}}}}
    InnerT = Mooncake.dual_type(Val(1), P)
    return Mooncake.Lifted{P,1,InnerT}(InnerT(primal, tangent))
end
# Seed factories — route through the canonical V ctor.
@inline function Mooncake.zero_dual(
    w::Val{N}, x::Base.ReshapedArray{T,D,P,MI}
) where {N,T<:IEEEFloat,D,P<:AbstractArray{T},MI}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.zero_tangent(x))::V
end
@inline function Mooncake.uninit_dual(
    w::Val{N}, x::Base.ReshapedArray{T,D,P,MI}
) where {N,T<:IEEEFloat,D,P<:AbstractArray{T},MI}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.uninit_tangent(x))::V
end
@inline function Mooncake.randn_dual(
    w::Val{N}, rng::AbstractRNG, x::Base.ReshapedArray{T,D,P,MI}
) where {N,T<:IEEEFloat,D,P<:AbstractArray{T},MI}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.randn_tangent(rng, x))::V
end
@inline Mooncake.zero_dual(
    ::Val{0}, x::Base.ReshapedArray{T,D,P,MI}
) where {T<:IEEEFloat,D,P<:AbstractArray{T},MI} = x
@inline Mooncake.uninit_dual(
    ::Val{0}, x::Base.ReshapedArray{T,D,P,MI}
) where {T<:IEEEFloat,D,P<:AbstractArray{T},MI} = x
@inline Mooncake.randn_dual(
    ::Val{0}, ::AbstractRNG, x::Base.ReshapedArray{T,D,P,MI}
) where {T<:IEEEFloat,D,P<:AbstractArray{T},MI} = x
# Complex-element seed factories.
@inline function Mooncake.zero_dual(
    w::Val{N}, x::Base.ReshapedArray{Complex{T},D,P,MI}
) where {N,T<:IEEEFloat,D,P<:AbstractArray{Complex{T}},MI}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.zero_tangent(x))::V
end
@inline function Mooncake.uninit_dual(
    w::Val{N}, x::Base.ReshapedArray{Complex{T},D,P,MI}
) where {N,T<:IEEEFloat,D,P<:AbstractArray{Complex{T}},MI}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.uninit_tangent(x))::V
end
@inline function Mooncake.randn_dual(
    w::Val{N}, rng::AbstractRNG, x::Base.ReshapedArray{Complex{T},D,P,MI}
) where {N,T<:IEEEFloat,D,P<:AbstractArray{Complex{T}},MI}
    V = Mooncake.dual_type(w, typeof(x))
    return V(x, Mooncake.randn_tangent(rng, x))::V
end
@inline Mooncake.zero_dual(
    ::Val{0}, x::Base.ReshapedArray{Complex{T},D,P,MI}
) where {T<:IEEEFloat,D,P<:AbstractArray{Complex{T}},MI} = x
@inline Mooncake.uninit_dual(
    ::Val{0}, x::Base.ReshapedArray{Complex{T},D,P,MI}
) where {T<:IEEEFloat,D,P<:AbstractArray{Complex{T}},MI} = x
@inline Mooncake.randn_dual(
    ::Val{0}, ::AbstractRNG, x::Base.ReshapedArray{Complex{T},D,P,MI}
) where {T<:IEEEFloat,D,P<:AbstractArray{Complex{T}},MI} = x
# primal / tangent accessors for canonical ReshapedArray V.
function primal(
    r::Base.ReshapedArray{NDual{T,N},D,V_parent,MI}
) where {T<:IEEEFloat,N,D,V_parent<:AbstractArray{NDual{T,N}},MI}
    return Base.ReshapedArray(primal(parent(r)), r.dims, r.mi)
end
function tangent(
    r::Base.ReshapedArray{NDual{T,N},D,V_parent,MI}
) where {T<:IEEEFloat,N,D,V_parent<:AbstractArray{NDual{T,N}},MI}
    return Mooncake.Tangent((; parent=tangent(parent(r)), dims=NoTangent(), mi=NoTangent()))
end
function primal(
    r::Base.ReshapedArray{Complex{NDual{T,N}},D,V_parent,MI}
) where {T<:IEEEFloat,N,D,V_parent<:AbstractArray{Complex{NDual{T,N}}},MI}
    return Base.ReshapedArray(primal(parent(r)), r.dims, r.mi)
end
function tangent(
    r::Base.ReshapedArray{Complex{NDual{T,N}},D,V_parent,MI}
) where {T<:IEEEFloat,N,D,V_parent<:AbstractArray{Complex{NDual{T,N}}},MI}
    return Mooncake.Tangent((; parent=tangent(parent(r)), dims=NoTangent(), mi=NoTangent()))
end

# Bridge for `Lifted{P,1}(primal, tangent::Tangent)` (CoDual→Lifted in the
# test framework): delegate to the canonical V ctor above so any
# AbstractArray-parent Transpose flows through the same constructor.
@inline function Mooncake.Lifted{P,1}(
    primal::P, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,P<:LinearAlgebra.Transpose{T,<:AbstractArray{T}}}
    InnerT = Mooncake.dual_type(Val(1), P)
    return Mooncake.Lifted{P,1,InnerT}(InnerT(primal, tangent))
end

# StepRangeLen with TwicePrecision ref/step fields: structural NamedTuple
# lift breaks the bare-Dual `_getindex_hiprec` / `unsafe_getindex` rule
# bodies in `src/rules/twice_precision.jl` that expect a bare StepRangeLen
# primal. Use the parallel-Dual form so `tangent(d::Dual{StepRangeLen,
# Tangent{NamedTuple{ref,step,len,offset}}})` returns the legacy Tangent
# whose fields are direct TwicePrecision / Int values.
function dual_type(
    ::Val{1}, ::Type{Base.StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},S}}
) where {T<:IEEEFloat,S}
    P = Base.StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},S}
    return Dual{P,Mooncake.tangent_type(P)}
end
function dual_type(
    ::Val{N}, ::Type{Base.StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},S}}
) where {N,T<:IEEEFloat,S}
    P = Base.StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},S}
    return Dual{P,Mooncake.tangent_type(Val(N), P)}
end
function dual_type(
    ::Val{0}, ::Type{Base.StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},S}}
) where {T<:IEEEFloat,S}
    return Base.StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},S}
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
# Wrapper types with canonical NDual-element forms: peel NDual back to the
# primal element. Without these, `_wrap_rule_result(P_out, Val(N), tuple)` in
# `interpreter/primal_mode.jl` carries the canonical NDual form through as
# the "primal" tuple element, then `dual_type(Val(N), P_out)` falls through
# to the generic structural NamedTuple lift (because no dual_type overload
# accepts a wrapper with NDual elements), producing a NamedTuple V that the
# canonical-NDual bare_result can't convert into.
@inline function Mooncake.__primal_type(
    ::Type{SubArray{NDual{T,N},D,Array{NDual{T,N},Dp},I,L}}
) where {T<:IEEEFloat,N,D,Dp,I,L}
    return SubArray{T,D,Array{T,Dp},I,L}
end
@inline function Mooncake.__primal_type(
    ::Type{SubArray{Complex{NDual{T,N}},D,Array{Complex{NDual{T,N}},Dp},I,L}}
) where {T<:IEEEFloat,N,D,Dp,I,L}
    return SubArray{Complex{T},D,Array{Complex{T},Dp},I,L}
end
@inline function Mooncake.__primal_type(
    ::Type{LinearAlgebra.Diagonal{NDual{T,N},Vector{NDual{T,N}}}}
) where {T<:IEEEFloat,N}
    return LinearAlgebra.Diagonal{T,Vector{T}}
end
@inline function Mooncake.__primal_type(
    ::Type{LinearAlgebra.Adjoint{NDual{T,N},P}}
) where {T<:IEEEFloat,N,P<:AbstractArray{NDual{T,N}}}
    return LinearAlgebra.Adjoint{T,Mooncake.__primal_type(P)}
end
@inline function Mooncake.__primal_type(
    ::Type{LinearAlgebra.Transpose{NDual{T,N},P}}
) where {T<:IEEEFloat,N,P<:AbstractArray{NDual{T,N}}}
    return LinearAlgebra.Transpose{T,Mooncake.__primal_type(P)}
end
for W in (
    :(LinearAlgebra.UpperTriangular),
    :(LinearAlgebra.LowerTriangular),
    :(LinearAlgebra.UnitUpperTriangular),
    :(LinearAlgebra.UnitLowerTriangular),
    :(LinearAlgebra.UpperHessenberg),
)
    @eval @inline function Mooncake.__primal_type(
        ::Type{$(W){NDual{T,N},Matrix{NDual{T,N}}}}
    ) where {T<:IEEEFloat,N}
        return $(W){T,Matrix{T}}
    end
end
@inline function Mooncake.__primal_type(
    ::Type{Base.ReshapedArray{NDual{T,N},D,V_parent,MI}}
) where {T<:IEEEFloat,N,D,V_parent<:AbstractArray{NDual{T,N}},MI}
    return Base.ReshapedArray{T,D,Mooncake.__primal_type(V_parent),MI}
end
@inline function Mooncake.__primal_type(
    ::Type{Base.ReshapedArray{Complex{NDual{T,N}},D,V_parent,MI}}
) where {T<:IEEEFloat,N,D,V_parent<:AbstractArray{Complex{NDual{T,N}}},MI}
    return Base.ReshapedArray{Complex{T},D,Mooncake.__primal_type(V_parent),MI}
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
# `MemoryRef{<:NDual}` and `MemoryRef{<:Complex{<:NDual}}` are valid
# canonical-V inner-dual shapes alongside their Memory/Array equivalents.
# Without these overloads, `verify_dual_type` (and downstream `debug_mode`
# / `verify_lifted_type`) falls through to the catch-all `Dual`-shape
# branch and errors with a `MethodError` on the bare MemoryRef leaf.
# `Memory{<:NDual}` already validates via the `AbstractArray{<:NDual}`
# overload above; `MemoryRef` needs its own because it is NOT
# `<: AbstractArray`.
@static if VERSION >= v"1.11-"
    Mooncake.verify_dual_type(::MemoryRef{<:NDual}) = true
    Mooncake.verify_dual_type(::MemoryRef{<:Complex{<:NDual}}) = true
end
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
# Top-level `NTangent{NTuple{N, Array{T,D}}}`: zip the per-lane arrays into
# the per-element NDual partials. Mirrors the canonical width-N return of
# `tangent(::Array{NDual{T,N},D})` so the inner-V ctor accepts the same
# shape that the matching accessor emits.
function (::Type{Array{NDual{T,N},D}})(
    primal::Array{T,D}, tangent::NTangent{NTuple{N,Array{T,D}}}
) where {T<:IEEEFloat,N,D}
    # `map(primal, tangent.lanes...)` preserves the shape of `primal` (so
    # `Matrix` stays a Matrix). A `map(eachindex(primal))` form would
    # return a flat `Vector{NDual}` because `eachindex(::Matrix)` is a
    # 1-D `LinearIndices`/`OneTo` iterator — that would surface as a
    # `Lifted{Matrix{Float64}, 1, Matrix{NDual}}(::Vector{NDual})` failure
    # in `Base._collect`-on-Matrix paths.
    return map(primal, tangent.lanes...) do p, ts::Vararg{T,N}
        NDual{T,N}(p, ts)
    end
end
function (::Type{Array{Complex{NDual{T,N}},D}})(
    primal::Array{Complex{T},D}, tangent::Array{Complex{T},D}
) where {T<:IEEEFloat,N,D}
    return map((p, t) -> Complex{NDual{T,N}}(p, t), primal, tangent)
end
# Top-level `NTangent{NTuple{N, Array{Complex{T},D}}}` form, mirroring the
# canonical width-N return of `tangent(::Array{Complex{NDual{T,N}},D})`.
function (::Type{Array{Complex{NDual{T,N}},D}})(
    primal::Array{Complex{T},D}, tangent::NTangent{NTuple{N,Array{Complex{T},D}}}
) where {T<:IEEEFloat,N,D}
    return map(eachindex(primal)) do i
        z = primal[i]
        re_lanes = ntuple(n -> real(tangent.lanes[n][i]), Val(N))
        im_lanes = ntuple(n -> imag(tangent.lanes[n][i]), Val(N))
        Complex(NDual{T,N}(real(z), re_lanes), NDual{T,N}(imag(z), im_lanes))
    end
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

# `Adjoint{NDual{T,N}, V_parent}(primal, tangent::Tangent)` ctor —
# delegates the parent build to `V_parent`'s own (primal, tangent) ctor,
# so this works uniformly across any AbstractArray-parent shape (Array,
# SubArray, …) that itself has a canonical NDual lift. Mirrors the
# Transpose ctor template.
function (::Type{LinearAlgebra.Adjoint{NDual{T,N},V_parent}})(
    primal::LinearAlgebra.Adjoint{T,P}, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,N,P<:AbstractArray{T},V_parent<:AbstractArray{NDual{T,N}}}
    parent_t = Mooncake._get_tangent_field(tangent, :parent)
    parent_lifted = V_parent(parent(primal), parent_t)
    return LinearAlgebra.Adjoint(parent_lifted)::LinearAlgebra.Adjoint{NDual{T,N},V_parent}
end

for W in (
    :(LinearAlgebra.UpperTriangular),
    :(LinearAlgebra.LowerTriangular),
    :(LinearAlgebra.UnitUpperTriangular),
    :(LinearAlgebra.UnitLowerTriangular),
    :(LinearAlgebra.UpperHessenberg),
)
    @eval function (::Type{$(W){NDual{T,N},Matrix{NDual{T,N}}}})(
        primal::$(W){T,<:Matrix{T}}, tangent::Mooncake.Tangent
    ) where {T<:IEEEFloat,N}
        data_t = Mooncake._get_tangent_field(tangent, :data)
        return $(W)(Matrix{NDual{T,N}}(primal.data, data_t))
    end
    # Seed factories route through the canonical V's (primal, tangent)
    # ctor so `zero_dual`/`uninit_dual`/`randn_dual` produce the wrapper-
    # shaped canonical NDual form that `dual_type(Val(N), $(W){T,P})`
    # declares — without these, the generic fallback returns the legacy
    # `Dual{$(W), Tangent{...}}` parallel-Dual form, breaking the
    # `_typeof(zero_dual(...)) === dual_type(...)` invariant.
    @eval @inline function Mooncake.zero_dual(
        w::Val{N}, x::$(W){T,<:Matrix{T}}
    ) where {N,T<:IEEEFloat}
        V = Mooncake.dual_type(w, typeof(x))
        return V(x, Mooncake.zero_tangent(x))::V
    end
    @eval @inline function Mooncake.uninit_dual(
        w::Val{N}, x::$(W){T,<:Matrix{T}}
    ) where {N,T<:IEEEFloat}
        V = Mooncake.dual_type(w, typeof(x))
        return V(x, Mooncake.uninit_tangent(x))::V
    end
    @eval @inline function Mooncake.randn_dual(
        w::Val{N}, rng::AbstractRNG, x::$(W){T,<:Matrix{T}}
    ) where {N,T<:IEEEFloat}
        V = Mooncake.dual_type(w, typeof(x))
        return V(x, Mooncake.randn_tangent(rng, x))::V
    end
    @eval @inline Mooncake.zero_dual(
        ::Val{0}, x::$(W){T,<:Matrix{T}}
    ) where {T<:IEEEFloat} = x
    @eval @inline Mooncake.uninit_dual(
        ::Val{0}, x::$(W){T,<:Matrix{T}}
    ) where {T<:IEEEFloat} = x
    @eval @inline Mooncake.randn_dual(
        ::Val{0}, ::AbstractRNG, x::$(W){T,<:Matrix{T}}
    ) where {T<:IEEEFloat} = x
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
function (::Type{SubArray{Complex{NDual{T,N}},D,Array{Complex{NDual{T,N}},Dp},I,L}})(
    primal::SubArray{Complex{T},D,Array{Complex{T},Dp},I,L}, tangent::Mooncake.Tangent
) where {T<:IEEEFloat,N,D,Dp,I,L}
    parent_t = Mooncake._get_tangent_field(tangent, :parent)
    parent_lifted = Array{Complex{NDual{T,N}},Dp}(parent(primal), parent_t)
    return view(
        parent_lifted, primal.indices...
    )::SubArray{Complex{NDual{T,N}},D,Array{Complex{NDual{T,N}},Dp},I,L}
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
    # Width-N counterpart: tangent is `NTangent{NTuple{N, Memory{T}}}`.
    # Mirrors the Array variant at line 1459 so SplitDual-wrapped mutable
    # struct fields whose canonical V is `Memory{NDual{T,N}}` (e.g.
    # `Dict.vals`) accept the NTangent shape that the SplitDual NTangent
    # ctor passes through.
    function (::Type{Memory{NDual{T,N}}})(
        primal::Memory{T}, tangent::Mooncake.NTangent{NTuple{N,Memory{T}}}
    ) where {T<:IEEEFloat,N}
        out = Memory{NDual{T,N}}(undef, length(primal))
        lanes = tangent.lanes
        @inbounds for i in eachindex(primal)
            out[i] = NDual{T,N}(primal[i], ntuple(d -> lanes[d][i], Val(N)))
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
# Disambiguate Val{0} + Ptr against the Val{0} primal-passthrough above.
# Val{0} is the primal passthrough — return `x` unchanged.
@inline Mooncake.zero_dual(::Val{0}, x::Ptr) = x

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

# Width-1 returns a top-level `NTangent{Tuple{Array{T,D}}}`, matching the
# chunked N>=2 shape. A bare-Array return would break `verify_lifted_type`
# for `Lifted{Array{T,D},1}` slots because the slot's canonical `V` is
# `Array{NDual{T,1},D}` and its outer tangent should be top-level
# `NTangent{Tuple{Array{T,D}}}` at every positive width.
function tangent(a::Array{NDual{T,N},D}) where {T,N,D}
    return NTangent(ntuple(i -> map(d -> d.partials[i], a), Val(N)))
end
@inline Mooncake._field_primal(a::Array{<:NDual}) = primal(a)
@inline Mooncake._field_tangent(a::Array{<:NDual}) = tangent(a)

function primal(a::Array{Complex{NDual{T,N}},D}) where {T,N,D}
    return map(z -> complex(z.re.value, z.im.value), a)
end

# Complex unified width-1 and width-N: top-level
# `NTangent{NTuple{N, Array{Complex{T},D}}}` at every positive width.
function tangent(a::Array{Complex{NDual{T,N}},D}) where {T,N,D}
    return NTangent(
        ntuple(i -> map(z -> complex(z.re.partials[i], z.im.partials[i]), a), Val(N))
    )
end
@inline Mooncake._field_primal(a::Array{<:Complex{<:NDual}}) = primal(a)
@inline Mooncake._field_tangent(a::Array{<:Complex{<:NDual}}) = tangent(a)

# ReshapedArray canonical NDual V — Phase 2 of the wrapper-exception removal.
# Without these, `primal(::Lifted{<:Tuple})` walks Tuple elements via
# `_field_primal` which falls through to the catch-all (returns x unchanged),
# leaving the ReshapedArray element type as NDual instead of reconstructing
# the primal. Shows up as a Matrix-vs-ReshapedArray primal-type mismatch in
# lapack rules that wrap a ReshapedArray output in a Tuple.
@inline Mooncake._field_primal(a::Base.ReshapedArray{<:NDual}) = primal(a)
@inline Mooncake._field_tangent(a::Base.ReshapedArray{<:NDual}) = tangent(a)
@inline Mooncake._field_primal(a::Base.ReshapedArray{<:Complex{<:NDual}}) = primal(a)
@inline Mooncake._field_tangent(a::Base.ReshapedArray{<:Complex{<:NDual}}) = tangent(a)

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
    a::LinearAlgebra.Adjoint{NDual{T,N},<:AbstractArray{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    return LinearAlgebra.Adjoint(primal(parent(a)))
end
function tangent(
    a::LinearAlgebra.Adjoint{NDual{T,N},<:AbstractArray{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; parent=tangent(parent(a))))
end

for W in (
    :(LinearAlgebra.UpperTriangular),
    :(LinearAlgebra.LowerTriangular),
    :(LinearAlgebra.UnitUpperTriangular),
    :(LinearAlgebra.UnitLowerTriangular),
    :(LinearAlgebra.UpperHessenberg),
)
    @eval begin
        function primal(x::$(W){NDual{T,N},<:Matrix{NDual{T,N}}}) where {T<:IEEEFloat,N}
            return $(W)(primal(x.data))
        end
        function tangent(x::$(W){NDual{T,N},<:Matrix{NDual{T,N}}}) where {T<:IEEEFloat,N}
            return Mooncake.Tangent((; data=tangent(x.data)))
        end
    end
end

function primal(
    t::LinearAlgebra.Transpose{NDual{T,N},<:AbstractArray{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    return transpose(primal(parent(t)))
end
function tangent(
    t::LinearAlgebra.Transpose{NDual{T,N},<:AbstractArray{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; parent=tangent(parent(t))))
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
function primal(
    s::SubArray{Complex{NDual{T,N}},D,Array{Complex{NDual{T,N}},Dp},I,L}
) where {T<:IEEEFloat,N,D,Dp,I,L}
    return view(primal(parent(s)), s.indices...)
end
function tangent(
    s::SubArray{Complex{NDual{T,N}},D,Array{Complex{NDual{T,N}},Dp},I,L}
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

# `tangent(x, i)` for wrapper-NDual containers — return the per-direction
# tangent in the wrapper's `tangent_type` shape (a `Tangent`), matching the
# `randn_tangent(::W)` shape used by FD comparison in `test_frule_correctness`.
@inline function Mooncake.tangent(
    x::LinearAlgebra.Diagonal{NDual{T,N},Vector{NDual{T,N}}}, i::Integer
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; diag=tangent(x.diag, i)))
end
@inline function Mooncake.tangent(
    x::LinearAlgebra.Adjoint{NDual{T,N},Matrix{NDual{T,N}}}, i::Integer
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; parent=tangent(parent(x), i)))
end
@inline function Mooncake.tangent(
    x::SubArray{NDual{T,N},D,Array{NDual{T,N},Dp},I,L}, i::Integer
) where {T<:IEEEFloat,N,D,Dp,I,L}
    return Mooncake.Tangent((;
        parent=tangent(parent(x), i),
        indices=NoTangent(),
        offset1=NoTangent(),
        stride1=NoTangent(),
    ))
end
@inline function Mooncake.tangent(
    x::SubArray{Complex{NDual{T,N}},D,Array{Complex{NDual{T,N}},Dp},I,L}, i::Integer
) where {T<:IEEEFloat,N,D,Dp,I,L}
    return Mooncake.Tangent((;
        parent=tangent(parent(x), i),
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
    # Top-level `NTangent{NTuple{N, Memory{T}}}` mirrors the
    # `Array{NDual{T,N},D}` accessor's top-level shape and matches the
    # canonical `tangent_type(Val(N), Memory{T}) === NTangent{NTuple{N,
    # Memory{T}}}` query. A per-element `Memory{NTangent}` shape would be
    # inconsistent with the chunked Array representation.
    function tangent(m::Memory{NDual{T,N}}) where {T,N}
        return NTangent(ntuple(i -> map(d -> d.partials[i], m), Val(N)))
    end
    function tangent(m::Memory{Complex{NDual{T,N}}}) where {T,N}
        return NTangent(
            ntuple(i -> map(z -> complex(z.re.partials[i], z.im.partials[i]), m), Val(N))
        )
    end
    primal(x::MemoryRef{<:NDual}) = memoryref(primal(x.mem), Core.memoryrefoffset(x))
    function primal(x::MemoryRef{<:Complex{<:NDual}})
        return memoryref(primal(x.mem), Core.memoryrefoffset(x))
    end
    # Top-level `NTangent{NTuple{N, MemoryRef{T}}}` with one MemoryRef per
    # lane (offset preserved on each lane), mirroring the Array/Memory
    # pattern above.
    function tangent(x::MemoryRef{NDual{T,N}}) where {T,N}
        offset = Core.memoryrefoffset(x)
        return NTangent(
            ntuple(i -> memoryref(map(d -> d.partials[i], x.mem), offset), Val(N))
        )
    end
    function tangent(x::MemoryRef{Complex{NDual{T,N}}}) where {T,N}
        offset = Core.memoryrefoffset(x)
        return NTangent(
            ntuple(
                i -> memoryref(
                    map(z -> complex(z.re.partials[i], z.im.partials[i]), x.mem), offset
                ),
                Val(N),
            ),
        )
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
# pairing with `tangent(::Adjoint{<:NDual}, ::Integer)` which returns a Tangent.
@inline _ndual_primal(d::LinearAlgebra.Diagonal{<:NDual}) = primal(d)
@inline _ndual_primal(a::LinearAlgebra.Adjoint{<:NDual}) = primal(a)
@inline _ndual_primal(s::SubArray{<:NDual}) = primal(s)
@inline _ndual_primal(r::Base.ReshapedArray{<:NDual}) = primal(r)
# Complex variants — same wrapper-preserving requirement: `map` over a SubArray
# / Adjoint / Diagonal returns the underlying material type (e.g. Matrix), so
# the subsequent `tangent(...)` call would see a wrapper-shape tangent paired
# with a materialized primal, tripping `populate_address_map_internal`'s
# `MutableTangent`-for-mutable-primal assertion.
@inline _ndual_primal(d::LinearAlgebra.Diagonal{<:Complex{<:NDual}}) = primal(d)
@inline _ndual_primal(a::LinearAlgebra.Adjoint{<:Complex{<:NDual}}) = primal(a)
@inline _ndual_primal(s::SubArray{<:Complex{<:NDual}}) = primal(s)
@inline _ndual_primal(r::Base.ReshapedArray{<:Complex{<:NDual}}) = primal(r)
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

# `tangent(x, i)` — extract the i-th direction tangent from any NDual-bearing
# representation. Used by `_new_` to assemble per-direction NTangent lanes.
# Per-type lane-extraction methods live directly on
# `tangent(x, ::Integer)` rather than a private `_tangent_dir` helper.
@inline Mooncake.tangent(x::NDual, i::Integer) = x.partials[i]
@inline Mooncake.tangent(x::Complex{<:NDual}, i::Integer) = complex(
    x.re.partials[i], x.im.partials[i]
)
@inline Mooncake.tangent(x::Dual{<:Any,<:NTangent}, i::Integer) = tangent(x).lanes[i]
@inline Mooncake.tangent(x::Dual{<:Any,<:Tuple}, i::Integer) = map(
    t -> _tangent_dir_elem(t, i), tangent(x)
)
@inline Mooncake.tangent(x::Dual, _::Integer) = tangent(x)
@inline Mooncake.tangent(x::AbstractArray{NDual{T,N}}, i::Integer) where {T,N} = map(
    d -> d.partials[i], x
)
@inline Mooncake.tangent(x::AbstractArray{Complex{NDual{T,N}}}, i::Integer) where {T,N} = map(
    z -> complex(z.re.partials[i], z.im.partials[i]), x
)
# Wrap the lane-i parent tangent in a `Tangent{(parent=...)}` so it round-
# trips against `randn_tangent(::Transpose)`/`randn_tangent(::Adjoint)` in
# tests. Without this, the bare-`map`-into-wrapper path above fires and
# yields a tangent shape incompatible with the `_dot` comparison.
@inline function Mooncake.tangent(
    x::LinearAlgebra.Transpose{NDual{T,N},<:AbstractArray{NDual{T,N}}}, i::Integer
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; parent=Mooncake.tangent(parent(x), i)))
end
@inline function Mooncake.tangent(
    x::LinearAlgebra.Adjoint{NDual{T,N},<:AbstractArray{NDual{T,N}}}, i::Integer
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((; parent=Mooncake.tangent(parent(x), i)))
end
@inline function Mooncake.tangent(
    x::Base.ReshapedArray{NDual{T,N},<:Any,<:AbstractArray{NDual{T,N}}}, i::Integer
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((;
        parent=Mooncake.tangent(parent(x), i), dims=NoTangent(), mi=NoTangent()
    ))
end
@inline function Mooncake.tangent(
    x::Base.ReshapedArray{Complex{NDual{T,N}},<:Any,<:AbstractArray{Complex{NDual{T,N}}}},
    i::Integer,
) where {T<:IEEEFloat,N}
    return Mooncake.Tangent((;
        parent=Mooncake.tangent(parent(x), i), dims=NoTangent(), mi=NoTangent()
    ))
end
for W in (
    :(LinearAlgebra.UpperTriangular),
    :(LinearAlgebra.LowerTriangular),
    :(LinearAlgebra.UnitUpperTriangular),
    :(LinearAlgebra.UnitLowerTriangular),
    :(LinearAlgebra.UpperHessenberg),
)
    @eval @inline function Mooncake.tangent(
        x::$(W){NDual{T,N},<:AbstractMatrix{NDual{T,N}}}, i::Integer
    ) where {T<:IEEEFloat,N}
        return Mooncake.Tangent((; data=Mooncake.tangent(x.data, i)))
    end
end
# Mirror `tangent_type(P<:Tuple)`'s all-NoTangent fold: if every element's
# direction tangent is `NoTangent`, return a single `NoTangent` so that
# downstream `build_output_tangent` can place it in a `NoTangent` field
# without a `Tuple{NoTangent...}` → `NoTangent` convert error.
@inline function Mooncake.tangent(x::Tuple, i::Integer)
    inner = map(xi -> tangent(xi, i), x)
    return inner isa Tuple{Vararg{NoTangent}} ? NoTangent() : inner
end
@inline function Mooncake.tangent(x::NamedTuple{names}, i::Integer) where {names}
    inner = tangent(values(x), i)
    return inner isa NoTangent ? inner : NamedTuple{names}(inner)
end
@inline function Mooncake.tangent(x::Base.Broadcast.Extruded, i::Integer)
    return Mooncake.Tangent((; x=tangent(x.x, i), keeps=NoTangent(), defaults=NoTangent()))
end
@inline function Mooncake.tangent(x::Base.Broadcast.Broadcasted, i::Integer)
    return Mooncake.Tangent((;
        style=NoTangent(), f=NoTangent(), args=tangent(x.args, i), axes=NoTangent()
    ))
end
@static if VERSION >= v"1.11-"
    @inline function Mooncake.tangent(x::MemoryRef{NDual{T,N}}, i::Integer) where {T,N}
        return memoryref(map(d -> d.partials[i], x.mem), Core.memoryrefoffset(x))
    end
    @inline function Mooncake.tangent(
        x::MemoryRef{Complex{NDual{T,N}}}, i::Integer
    ) where {T,N}
        return memoryref(
            map(z -> complex(z.re.partials[i], z.im.partials[i]), x.mem),
            Core.memoryrefoffset(x),
        )
    end
end

# Lifted slot-shape passthrough: extract the i-th direction tangent from the
# Lifted's inner V. Mirrors how `tangent(::Lifted)` passes through to the
# inner V's tangent.
@inline Mooncake.tangent(x::Mooncake.Lifted, i::Integer) = tangent(x.value, i)

# Immutable struct-primal slot with NamedTuple inner V (recursive lift,
# see dual.jl): build the per-direction tangent via `build_output_tangent`
# so the result matches `tangent_type(P)` (Tangent shape with
# `PossiblyUninitTangent` wrapping where present). For NamedTuple primals
# the bare NamedTuple result is canonical, but inner struct fields still
# need `Tangent{...}` rebuild — handled by `_lane_named_tuple`.
@inline function Mooncake.tangent(
    d::Mooncake.Lifted{P,N,V}, i::Integer
) where {P,N,V<:NamedTuple}
    P <: NamedTuple && return _lane_named_tuple(P, d.value, i)
    return _build_struct_tangent_dir(P, d.value, i)
end

# Tuple-primal slot with Tuple inner V: recursively rebuild per-element
# tangents, wrapping inner struct elements as `Tangent{...}`. Only fires
# for concrete P (matching the no-`i` overload's `isconcretetype` guard);
# the abstract case falls back through the generic `tangent(::Lifted, i)`.
@inline function Mooncake.tangent(
    d::Mooncake.Lifted{P,N,V}, i::Integer
) where {P<:Tuple,N,V<:Tuple}
    isconcretetype(P) || return tangent(d.value, i)
    return _lane_tuple(P, d.value, i)
end
@generated function _build_struct_tangent_dir(
    ::Type{P}, value::V, i
) where {P,V<:NamedTuple{names}} where {names}
    primal_exprs = Expr[]
    tangent_exprs = Expr[]
    for n in names
        F = fieldtype(P, n)
        push!(primal_exprs, :(_ndual_primal(value.$n)))
        push!(tangent_exprs, :(_lane_tangent_for_field($F, value.$n, i)))
    end
    return :(Mooncake.build_output_tangent(
        $P, ($(primal_exprs...),), ($(tangent_exprs...),)
    ))
end

# Per-lane tangent for a struct/Tuple/NamedTuple field, given the primal
# field type `F` and the inner-V slot value `v`. The recursive lift inside
# `dual_type` collapses inner structs to `NamedTuple{...}` shapes, but the
# per-lane tangent must rebuild `Tangent{...}` wrappers where the primal
# field type is a struct. Without this, nested struct fields leak raw
# `NamedTuple{...}` to `build_output_tangent`, which then fails to convert
# to the expected `Tangent{...}` shape.
@inline _lane_tangent_for_field(::Type{F}, v, i) where {F} = tangent(v, i)

@inline function _lane_tangent_for_field(::Type{F}, v::NamedTuple, i) where {F}
    # NamedTuple primal: bare NamedTuple result is canonical.
    F <: NamedTuple && return _lane_named_tuple(F, v, i)
    # Struct primal whose inner V is a recursive NamedTuple: rebuild Tangent.
    return _build_struct_tangent_dir(F, v, i)
end

@inline function _lane_tangent_for_field(::Type{F}, v::Tuple, i) where {F<:Tuple}
    return _lane_tuple(F, v, i)
end

@generated function _lane_tuple(::Type{F}, v::Tuple, i) where {F<:Tuple}
    n = fieldcount(F)
    elems = [:(_lane_tangent_for_field($(fieldtype(F, k)), v[$k], i)) for k in 1:n]
    return quote
        inner = ($(elems...),)
        inner isa Tuple{Vararg{NoTangent}} ? NoTangent() : inner
    end
end

@generated function _lane_named_tuple(::Type{F}, v::NamedTuple, i) where {F<:NamedTuple}
    names = fieldnames(F)
    elems = [
        :(_lane_tangent_for_field($(fieldtype(F, names[k])), v[$k], i)) for
        k in 1:length(names)
    ]
    return quote
        inner = ($(elems...),)
        inner isa Tuple{Vararg{NoTangent}} ? NoTangent() : NamedTuple{$names}(inner)
    end
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

# When the IR routes through `Dual{<:Integer, NoTangent}` for a NoTangent
# integer arg of `^` (e.g. `x^n` where `n` is a runtime-known integer
# captured into a Dual flow), unwrap to the bare Integer and reuse the
# existing `^(::NDual, ::Integer)` rule in `Nfwd.jl`.
@inline Base.:^(a::NDual, b::Dual{<:Integer,NoTangent}) = a ^ Mooncake.primal(b)

end
