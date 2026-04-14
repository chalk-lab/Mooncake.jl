module NfwdMooncake

import ..Mooncake
using Base: IEEEFloat
using Random: AbstractRNG
using ..Nfwd
import ..Mooncake:
    CoDual,
    Dual,
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
    throw_val_and_grad_ret_type_error,
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
    slot == 1 ? g + v : g
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

# Val{0} ambiguity resolvers: dual_type(Val(0), P) = P for all P.
dual_type(::Val{0}, ::Type{T}) where {T<:IEEEFloat} = T
dual_type(::Val{0}, ::Type{Complex{T}}) where {T<:IEEEFloat} = Complex{T}
dual_type(::Val{0}, ::Type{Array{T,D}}) where {T<:IEEEFloat,D} = Array{T,D}
function dual_type(::Val{0}, ::Type{Array{Complex{T},D}}) where {T<:IEEEFloat,D}
    Array{Complex{T},D}
end
@static if VERSION >= v"1.11-"
    dual_type(::Val{0}, ::Type{MemoryRef{T}}) where {T<:IEEEFloat} = MemoryRef{T}
    dual_type(::Val{0}, ::Type{Memory{T}}) where {T<:IEEEFloat} = Memory{T}
    dual_type(::Val{0}, ::Type{MemoryRef{Complex{T}}}) where {T<:IEEEFloat} = MemoryRef{Complex{T}}
    dual_type(::Val{0}, ::Type{Memory{Complex{T}}}) where {T<:IEEEFloat} = Memory{Complex{T}}
end

# tangent_type(NDual) uses the default struct-based tangent_type. HVP runs
# reverse mode on f(NDual(x,v)), so build_rrule needs tangent_type(NDual) to
# construct CoDuals for NDual-typed arguments.

# primal / tangent accessors
primal(d::NDual) = d.value
tangent(d::NDual{T,N}) where {T,N} = NTangent(d.partials)

# __get_primal for NDual — primal_mode.jl defines the Dual overload,
# reverse_mode.jl defines CoDual and generic fallback.
Mooncake.__get_primal(x::NDual) = primal(x)

# _partial_i for NDual vararg transpose — primal_mode.jl defines the Dual overload.
Mooncake._partial_i(x::NDual, i::Int) = x.partials[i]

primal(z::Complex{<:NDual}) = complex(z.re.value, z.im.value)
function tangent(z::Complex{NDual{T,N}}) where {T,N}
    return NTangent(ntuple(i -> complex(z.re.partials[i], z.im.partials[i]), Val(N)))
end

# NDual constructor from NTangent
function Nfwd.NDual(p::T, t::NTangent{NTuple{N,T}}) where {T<:IEEEFloat,N}
    return NDual{T,N}(p, t.lanes)
end

# ── Width helpers for NDual ───────────────────────────────────────────────────

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

function tangent(a::Array{NDual{T,N},D}) where {T,N,D}
    return map(d -> NTangent(d.partials), a)
end

function primal(a::Array{Complex{NDual{T,N}},D}) where {T,N,D}
    return map(z -> complex(z.re.value, z.im.value), a)
end

function tangent(a::Array{Complex{NDual{T,N}},D}) where {T,N,D}
    return map(z -> NTangent(ntuple(i -> complex(z.re.partials[i], z.im.partials[i]), Val(N))), a)
end

# ── Width-mismatch checking ──────────────────────────────────────────────────

"""
    ndual_width(x)

Return the NDual width N for a dual-typed argument, or `nothing` for
`Dual{P, NoTangent}` arguments (which are compatible with any width).
Throws for unsupported types.
"""
ndual_width(::NDual{T,N}) where {T,N} = N
ndual_width(::Complex{NDual{T,N}}) where {T,N} = N
ndual_width(::Array{NDual{T,N}}) where {T,N} = N
ndual_width(::Array{Complex{NDual{T,N}}}) where {T,N} = N
ndual_width(::Dual{P,NoTangent}) where {P} = nothing  # compatible with any width

function check_ndual_width_consistency(args...)
    widths = filter(!isnothing, map(ndual_width, args))
    isempty(widths) && return nothing
    w = first(widths)
    if !all(==(w), widths)
        throw(ArgumentError(
            "NDual width mismatch: got widths $(collect(widths)) across arguments. " *
            "All NDual arguments in a single rule call must have the same width N."
        ))
    end
    return w
end

# _uninit_dual for Array at width N: return bare Array{NDual} (matches dual_type).
function Mooncake._uninit_dual(::Val{N}, v::Array{T,D}) where {N,T<:IEEEFloat,D}
    ndual_arr = Array{NDual{T,N},D}(undef, size(v)...)
    @inbounds for i in eachindex(v)
        ndual_arr[i] = NDual{T,N}(v[i], ntuple(_ -> zero(T), Val(N)))
    end
    return ndual_arr
end

end
