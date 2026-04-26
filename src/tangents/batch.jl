# ── BatchContainer: SoA container for batched values ──────────────────────────

# Scalar primal types that get SoA BatchContainer treatment.
# Covers all IEEE floats and their complex counterparts - both are BLAS-compatible.
const _VmapScalar = Union{IEEEFloat, Complex{<:IEEEFloat}}

"""
    BatchContainer{T, D<:AbstractArray} <: AbstractVector{T}

SoA-style container for a batch of `N` values of type `T`.

Fields:
- `data::D`: contiguous backing store; batch is always the LAST dimension of `data`
- `index::Int`: which dimension of `data` is the batch axis (= ndims(data))

Layout conventions (batch always last):
- Scalar  `T <: _VmapScalar`:  `data isa Vector{T}`       (N,)
- Vector  `T = Vector{E}`:     `data isa Matrix{E}`       (L, N)
- Matrix  `T = Matrix{E}`:     `data isa Array{E,3}`      (L, M, N)
- General `T = Array{E,K}`:    `data isa Array{E,K+1}`    (d₁,…,dₖ, N)
- Fallback (non-_VmapScalar):  `data isa Vector{T}` (AoS)

WHY BATCH IS ALWAYS LAST:
Julia is column-major. When batch is the last dimension, each batch element's data
occupies contiguous memory (one "slab" of columns). This is the invariant that makes
the reshape trick valid: reshape(data, prod(inner_dims), N_batch) gives a 2-D view
where each column is exactly one batch element. BLAS gemm then processes all batch
elements in a single call. If batch were not last, the batch elements would not be
contiguous after reshape and the multiply would produce wrong results.
"""
struct BatchContainer{T, D<:AbstractArray} <: AbstractVector{T}
    data::D
    index::Int
end

Base.size(bc::BatchContainer) = (size(bc.data, bc.index),)

# Scalar batch: 1-D storage, direct index
Base.getindex(bc::BatchContainer{T, <:AbstractVector}, i::Int) where {T} = bc.data[i]

# Array-element batch: general N-D storage, batch is last dim - selectdim is zero-copy
function Base.getindex(bc::BatchContainer{T, D}, i::Int) where {T, D<:AbstractArray}
    selectdim(bc.data, bc.index, i)
end

# WHY THE broadcastable SPLIT:
#
# Scalar batch (data::Vector{T}): returning bc.data lets Julia broadcast directly on the
# 1-D array, giving SIMD-fused element-wise ops (e.g., a.data .+ b.data) with no
# getindex overhead.
#
# Array-element batch (data::Matrix, Array{T,3}, …): returning bc itself is necessary
# because broadcast(f, bc) must iterate over batch elements (columns/slices), not over
# individual scalars inside each element. If we returned bc.data, broadcast would
# iterate over scalars within each batch element instead of over batch elements.
# Keeping bc means AbstractVector{T} iteration via getindex/selectdim - correct
# column-wise semantics.
Base.Broadcast.broadcastable(bc::BatchContainer{T, <:AbstractVector}) where {T} = bc.data
Base.Broadcast.broadcastable(bc::BatchContainer{T, <:AbstractArray})  where {T} = bc

# ── Helpers: batch size from array rank ──────────────────────────────────────

# Number of non-batch dims for an array element type stored in an N+1-dim array.
_batch_ndims(::Type{T}) where {T<:_VmapScalar}          = 0   # scalar → 1D storage
_batch_ndims(::Type{Vector{T}}) where {T<:_VmapScalar}  = 1   # vector → 2D storage
_batch_ndims(::Type{Matrix{T}}) where {T<:_VmapScalar}  = 2   # matrix → 3D storage
_batch_ndims(::Type{Array{T,N}}) where {T<:_VmapScalar, N} = N

# ── Type-stable field extraction helper ──────────────────────────────────────
# Used by @generated _pack_batch and _wrap_input for Tuple/NamedTuple elements.
# Val{I} encodes the field index in the type, so fieldtype(P, I) is known at
# compile time and the return type of the loop is concrete (Vector{fieldtype(P,I)}).
# Neither getindex.(v,$i) nor getfield.(v,$i) achieve this: broadcasting only
# tracks the type Int for the index, not the specific value, so inference widens
# to Union{fieldtypes...}. This helper avoids broadcast entirely.
@inline function _vmap_getfield(v::AbstractVector{P}, ::Val{I}) where {P, I}
    T = fieldtype(P, I)
    out = Vector{T}(undef, length(v))
    @inbounds for j in eachindex(v)
        out[j] = getfield(v[j], I)
    end
    return out
end

# ── Packing helpers ───────────────────────────────────────────────────────────

"""
    _pack_batch(results) → BatchContainer or AbstractVector

Wrap a collected broadcast result into the canonical `BatchContainer`.
Called by `LazyVmap`/`DynamicVmap` to re-package output after a broadcast fallback.
"""
_pack_batch(v::Vector{T}) where {T<:_VmapScalar} = BatchContainer{T, Vector{T}}(v, 1)

_pack_batch(v::Vector{<:AbstractVector{T}}) where {T<:_VmapScalar} =
    BatchContainer{Vector{T}, Matrix{T}}(reduce(hcat, v), 2)

_pack_batch(v::Vector{<:AbstractMatrix{T}}) where {T<:_VmapScalar} =
    BatchContainer{Matrix{T}, Array{T,3}}(cat(v...; dims=3), 3)

# General N-D array batch (N ≥ 3): stack along dim N+1.
# Specific Vector/Matrix methods above are more specific and take precedence for N=1,2.
_pack_batch(v::Vector{Array{T,N}}) where {T<:_VmapScalar, N} =
    BatchContainer{Array{T,N}, Array{T,N+1}}(cat(v...; dims=N+1), N+1)

# Non-_VmapScalar dense arrays (Array{Int}, Array{String}, …): same SoA layout.
# No BLAS rules apply; the LazyVmap broadcast fallback handles element-wise operations.
# The T<:_VmapScalar method above is more specific and takes precedence for float arrays.
_pack_batch(v::Vector{Array{T,N}}) where {T, N} =
    BatchContainer{Array{T,N}, Array{T,N+1}}(cat(v...; dims=N+1), N+1)

# Tuples: transpose N tuples into a Tangent of per-element batches.
# @generated so field count and positional names are resolved at code-generation time,
# making each recursive _pack_batch call type-stable.
@generated function _pack_batch(v::AbstractVector{P}) where {P <: Tuple}
    !isconcretetype(P) && return :(v)
    nfields = fieldcount(P)
    nfields == 0 && return :(Tangent(NamedTuple{()}(())))
    names = ntuple(i -> Symbol(:_, i), nfields)
    # ::Vector{FT} assertion: fieldtype(P,i) is known at code-gen time, so $FT
    # interpolates the concrete type into the generated AST. Julia's inference can
    # then resolve _pack_batch to the precise method without widening to AbstractFloat.
    field_exprs = ntuple(nfields) do i
        FT = fieldtype(P, i)
        :(_pack_batch(_vmap_getfield(v, Val($i))::Vector{$FT}))
    end
    return :(Tangent(NamedTuple{$names}(tuple($(field_exprs...)))))
end

# NamedTuples: same as Tuple but use actual field names (:x,:y,…) instead of :_1,:_2,…
@generated function _pack_batch(v::AbstractVector{P}) where {names, T<:Tuple, P<:NamedTuple{names,T}}
    nfields = length(names)
    nfields == 0 && return :(Tangent(NamedTuple{$names}(())))
    field_exprs = ntuple(nfields) do i
        FT = fieldtype(P, i)
        :(_pack_batch(_vmap_getfield(v, Val($i))::Vector{$FT}))
    end
    return :(Tangent(NamedTuple{$names}(tuple($(field_exprs...)))))
end

# Struct SoA: transpose N structs into Tangent/MutableTangent of per-field batches.
# Must be a regular function (not @generated): struct_batchable is user-extensible via
# @struct_batch and can be added after Mooncake loads. @generated bodies run at the world
# of the @generated definition and would never see user-registered structs.
function _pack_batch(v::AbstractVector{P}) where P
    if struct_batchable(P)
        names   = fieldnames(P)
        batched = ntuple(i -> _pack_batch([getfield(x, i) for x in v]), fieldcount(P))
        nt = NamedTuple{names}(batched)
        return ismutabletype(P) ? MutableTangent(nt) : Tangent(nt)
    end
    return v
end

"""
    _make_batch(v, N) → BatchContainer or fill

Create N copies of constant `v` in the canonical contiguous layout.
Used at IR constant-lifting sites: when a primal constant is encountered during
vmap IR generation, it is replicated to produce a batched constant for the lifted IR.
"""
_make_batch(v::T, N::Int) where {T<:_VmapScalar} =
    BatchContainer{T, Vector{T}}(fill(v, N), 1)

_make_batch(v::Vector{T}, N::Int) where {T<:_VmapScalar} =
    BatchContainer{Vector{T}, Matrix{T}}(repeat(v, 1, N), 2)

_make_batch(v::Matrix{T}, N::Int) where {T<:_VmapScalar} = begin
    data = Array{T,3}(undef, size(v, 1), size(v, 2), N)
    for i in 1:N; data[:, :, i] = v; end
    BatchContainer{Matrix{T}, Array{T,3}}(data, 3)
end

# General N-D array: allocate (size(v)..., N) and fill each slice.
function _make_batch(v::Array{T,N}, N_batch::Int) where {T<:_VmapScalar, N}
    data = Array{T,N+1}(undef, size(v)..., N_batch)
    for i in 1:N_batch
        selectdim(data, N+1, i) .= v
    end
    return BatchContainer{Array{T,N}, Array{T,N+1}}(data, N+1)
end

# General AbstractArray{E<:_VmapScalar} (sparse, static, GPU arrays, …): copy into a
# same-device backing array via `similar`. Dense Array methods above are more specific
# and take precedence when v isa Array. `similar` preserves the array type (e.g.
# CuArray stays on GPU; sparse arrays get a dense copy on the same device).
# Full GPU support (batch_type + vmap_rule!!) requires a package extension.
function _make_batch(v::P, N_batch::Int) where {E<:_VmapScalar, P<:AbstractArray{E}}
    K = ndims(v)
    data = similar(v, E, size(v)..., N_batch)
    for i in 1:N_batch
        selectdim(data, K+1, i) .= v
    end
    return BatchContainer{P, typeof(data)}(data, K+1)
end

function _pack_batch(v::AbstractVector{P}) where {E<:_VmapScalar, P<:AbstractArray{E}}
    isempty(v) && return BatchContainer{P, Array{E,ndims(P)+1}}(
        Array{E, ndims(P)+1}(undef, ntuple(_ -> 0, ndims(P))..., 0), ndims(P)+1
    )
    K = ndims(first(v))
    data = similar(first(v), E, size(first(v))..., length(v))
    for (i, x) in enumerate(v)
        selectdim(data, K+1, i) .= x
    end
    return BatchContainer{P, typeof(data)}(data, K+1)
end

# Non-_VmapScalar dense arrays (Array{Int}, Array{String}, …): SoA layout.
# The T<:_VmapScalar method above is more specific and takes precedence for float arrays.
function _make_batch(v::Array{T,N}, N_batch::Int) where {T, N}
    data = Array{T,N+1}(undef, size(v)..., N_batch)
    for i in 1:N_batch
        selectdim(data, N+1, i) .= v
    end
    return BatchContainer{Array{T,N}, Array{T,N+1}}(data, N+1)
end

# Tuples: replicate each element separately, wrap in Tangent.
# @generated so field count and positional names are resolved at code-generation time.
@generated function _make_batch(v::P, N::Int) where {P <: Tuple}
    !isconcretetype(P) && return :(fill(v, N))
    nfields = fieldcount(P)
    nfields == 0 && return :(Tangent(NamedTuple{()}(())))
    names = ntuple(i -> Symbol(:_, i), nfields)
    field_exprs = ntuple(i -> :(_make_batch(v[$i], N)), nfields)
    return :(Tangent(NamedTuple{$names}(tuple($(field_exprs...)))))
end

# NamedTuples: same as Tuple but preserves actual field names.
@generated function _make_batch(v::NamedTuple{names}, N::Int) where {names}
    nfields = length(names)
    nfields == 0 && return :(Tangent(NamedTuple{$names}(())))
    field_exprs = ntuple(i -> :(_make_batch(v[$i], N)), nfields)
    return :(Tangent(NamedTuple{$names}(tuple($(field_exprs...)))))
end

# Struct SoA: replicate each field separately into its own batch.
# Must be a regular function - struct_batchable is user-extensible; see _pack_batch note.
function _make_batch(v::P, N::Int) where P
    if struct_batchable(P)
        names   = fieldnames(P)
        batched = ntuple(i -> _make_batch(getfield(v, i), N), fieldcount(P))
        nt = NamedTuple{names}(batched)
        return ismutabletype(P) ? MutableTangent(nt) : Tangent(nt)
    end
    return fill(v, N)
end

# ── batch_type: type functor for vmap ─────────────────────────────────────────

"""
    batch_type(::Type{P})

Returns the batched representation of primal type `P` under vmap.

This is the vmap analogue of `dual_type(Val(N), P)` for forward-mode AD.
_VmapScalar arrays of rank K map to a BatchContainer backed by a rank-(K+1) array,
with the batch axis as the last dimension. BLAS and Julia's broadcast work on the
raw backing array: gemm for rank-2, reshape trick for rank-3+, SIMD broadcast for all.
Non-differentiable scalars are left unchanged (control flow stays cheap).

`batch_type(T) == T` is used as the sentinel: "this type is not batched" - used in
`const_batch!` and `vmap` to decide whether to wrap a value or leave it as-is.
"""
@unstable function batch_type(::Type{P}) where {P}
    P == Union{} && return Union{}
    P isa Union && return Union{batch_type(P.a), batch_type(P.b)}
    (P isa UnionAll || P == UnionAll) && return Vector
    # Struct SoA: opted-in concrete struct → Tangent (immutable) or MutableTangent (mutable).
    # Both wrap a NamedTuple of per-field batch types. MutableTangent is a mutable struct,
    # so setfield! in the lifted IR can replace a field's BatchContainer pointer.
    # The BatchContainer.data arrays inside are always mutable regardless of wrapper type.
    if struct_batchable(P) && isstructtype(P) && isconcretetype(P) && fieldcount(P) > 0
        names   = fieldnames(P)
        Tfields = ntuple(i -> batch_type(fieldtype(P, i)), fieldcount(P))
        NT = NamedTuple{names, Tuple{Tfields...}}
        return ismutabletype(P) ? MutableTangent{NT} : Tangent{NT}
    end
    return isconcretetype(P) ? Vector{P} : Vector
end

# ── Tuples: immutable, integer-indexed → Tangent{NamedTuple{(:_1,:_2,…), …}} ──
#
# Tuples can't use fieldnames() (it throws) so we generate positional names :_1,:_2,…
# Integer-index access in the lifted IR (getfield(t, 1), getindex(t, 1)) resolves via
# get_tangent_field(t, i::Int) = getfield(t.fields, i) - position-based, name-agnostic.
# Core.tuple(a, b) construction in the lifted IR is intercepted by vmap_rule!! below.
@unstable function batch_type(::Type{P}) where {P <: Tuple}
    !isconcretetype(P) && return Vector
    N = fieldcount(P)
    N == 0 && return Tangent{NamedTuple{(), Tuple{}}}
    names   = ntuple(i -> Symbol(:_, i), N)
    Tfields = ntuple(i -> batch_type(fieldtype(P, i)), N)
    return Tangent{NamedTuple{names, Tuple{Tfields...}}}
end

# ── NamedTuples: like Tuples but with real field names ────────────────────────
#
# fieldnames(NamedTuple{names,T}) returns names directly - no positional renaming needed.
# Field access in the lifted IR via getfield(t, :x) routes to get_tangent_field on the
# Tangent wrapper - already covered by the PossiblyMutableTangent getfield rules.
# NamedTuple construction (%new or constructor call) is handled in vmap_mode.jl.
@unstable function batch_type(::Type{P}) where {names, T<:Tuple, P<:NamedTuple{names,T}}
    !isconcretetype(P) && return Vector
    N = length(names)
    N == 0 && return Tangent{NamedTuple{names, Tuple{}}}
    Tfields = ntuple(i -> batch_type(fieldtype(P, i)), N)
    return Tangent{NamedTuple{names, Tuple{Tfields...}}}
end

# _VmapScalar scalars → 1-D vector
batch_type(::Type{T}) where {T<:_VmapScalar} = BatchContainer{T, Vector{T}}

# _VmapScalar vectors → 2-D matrix (enables BLAS gemm for matrix-times-batch-of-vectors)
batch_type(::Type{Vector{T}}) where {T<:_VmapScalar} = BatchContainer{Vector{T}, Matrix{T}}

# _VmapScalar matrices → 3-D array (enables reshape-trick BLAS gemm for matrix-times-batch-of-matrices)
batch_type(::Type{Matrix{T}}) where {T<:_VmapScalar} = BatchContainer{Matrix{T}, Array{T,3}}

# Dense Array{T,N} elements (any N) → (N+1)-D dense backing.
# Julia method specificity: Vector{T}=Array{T,1} and Matrix{T}=Array{T,2} methods
# above are more specific and take precedence for N=1,2.
batch_type(::Type{Array{T,N}}) where {T<:_VmapScalar, N} =
    BatchContainer{Array{T,N}, Array{T,N+1}}

# Non-_VmapScalar dense arrays (Array{Int}, Array{String}, …): SoA storage so that
# element access is O(1) via selectdim rather than N separate heap objects.
# No BLAS rules apply, but the element-wise broadcast catch-all fires for f.(bc.data).
# The T<:_VmapScalar methods above are more specific and take precedence for those types.
batch_type(::Type{Array{T,N}}) where {T, N} = BatchContainer{Array{T,N}, Array{T,N+1}}

# AbstractArray subtypes (sparse, static, views, …) with _VmapScalar elements.
# We copy into a dense Array{E, ndims(P)+1} backing so that BLAS and element-wise
# catch-all rules apply. The four concrete Array methods above are more specific and
# take precedence when P is a plain Array.
batch_type(::Type{P}) where {E<:_VmapScalar, P<:AbstractArray{E}} =
    BatchContainer{P, Array{E, ndims(P)+1}}

# ── Non-differentiable scalars: not batched, act as constants ─────────────────
# batch_type(T) == T is checked as the "not batched" sentinel in const_batch! and vmap.
batch_type(::Type{Bool})                                                     = Bool
batch_type(::Type{Nothing})                                                  = Nothing
batch_type(::Type{<:Type})                                                   = Type
batch_type(::Type{Symbol})                                                   = Symbol
batch_type(::Type{<:Union{UInt8,UInt16,UInt32,UInt64,UInt128}})              = UInt64
batch_type(::Type{<:Union{Int8,Int16,Int32,Int64,Int128}})                   = Int64
batch_type(::Type{Core.TypeName})                                            = Core.TypeName
batch_type(::Type{Module})                                                   = Module

# ── Struct SoA (tree storage) ─────────────────────────────────────────────────
#
# For a struct P with fields f1::T1, f2::T2, …, fₙ::Tₙ, the SoA batch type is:
#
#   NamedTuple{(:f1,:f2,…,:fₙ), Tuple{batch_type(T1), batch_type(T2), …}}
#
# so a batch of N Point(x::Float64, y::Float64) values becomes:
#
#   (x = BatchContainer{Float64,Vector{Float64}}, y = BatchContainer{Float64,Vector{Float64}})
#
# Field access in the lifted IR: getfield(nt, :x) returns the BatchContainer directly -
# zero-copy, O(1). All existing BLAS/broadcast rules then apply to each field.
#
# Struct construction in the lifted IR: %new(P, bx, by) → _construct_struct_batch(P, bx, by)
# → NamedTuple{fieldnames(P)}((bx, by)).
#
# Only immutable concrete structs are eligible. Mutable structs require setfield!,
# which is incompatible with immutable NamedTuple storage.
# Opt-in via @struct_batch - prevents accidentally SoA-ing function structs or
# non-mathematical types that happen to be immutable structs.

"""
    struct_batchable(::Type{P}) → Bool

Returns true if `P` should be stored as a tree-of-arrays NamedTuple batch in vmap.
Default is false. Use `@struct_batch P` to opt in.
"""
struct_batchable(::Type) = false

"""
    @struct_batch P

Register the concrete struct `P` for SoA (tree-of-arrays) batching under vmap.

- Immutable `P`: batch stored as `Tangent{NT}` where `NT` is a NamedTuple of per-field
  `BatchContainer`s. Immutable wrapper - no `setfield!` in lifted IR.
- Mutable `P`: batch stored as `MutableTangent{NT}`. `setfield!` in the lifted IR routes
  to `set_tangent_field!`, replacing a field's `BatchContainer` pointer in-place.

Field access (`getfield`) is O(1) and returns the field's `BatchContainer` directly.
All existing BLAS and element-wise rules apply per field. Nested `@struct_batch` structs
compose naturally: each field's batch type is itself a `Tangent`/`MutableTangent`.

```julia
struct Point; x::Float64; y::Float64; end
@struct_batch Point
# batch of [Point(1,2), Point(3,4)] →
#   Tangent{@NamedTuple{x::BatchContainer{...}, y::BatchContainer{...}}}

mutable struct Particle; pos::Point; mass::Float64; end
@struct_batch Particle
# batch →  MutableTangent{@NamedTuple{pos::Tangent{...}, mass::BatchContainer{...}}}
```
"""
macro struct_batch(T)
    quote
        Mooncake.struct_batchable(::Type{$(esc(T))}) = true
    end
end
