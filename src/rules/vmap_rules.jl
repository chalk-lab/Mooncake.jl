# ── Struct SoA field access and mutation ─────────────────────────────────────
#
# Immutable struct batch → Tangent{NT}:    getfield dispatches here, returns BatchContainer.
# Mutable struct batch  → MutableTangent{NT}: getfield + setfield! both dispatch here.
#
# get_tangent_field(t, :f) reads t.fields.f - O(1), zero-copy.
# set_tangent_field!(t, :f, v) rebuilds t.fields with the new BatchContainer and stores
# it back into t.fields (MutableTangent is a mutable struct, so this is legal).
#
# setfield! in the primal IR (for mutable structs) becomes setfield! in the lifted IR
# with t::MutableTangent and val::BatchContainer. The rule routes it to set_tangent_field!
# which replaces the field's BatchContainer pointer - the data array inside remains alive.
# Return value matches Julia's setfield! semantics: returns val.

# ── Tuple construction and access ────────────────────────────────────────────
#
# In Julia IR, (a, b) lowers to Core.tuple(a, b) - a :call, not :new.
# The @generated form fixes the names tuple at compile time (N from type params).
# getindex(t, i) for a tuple batch routes to get_tangent_field by integer index,
# same as getfield. Both are registered so either IR form is covered.

@generated function vmap_rule!!(::typeof(Core.tuple), args...)
    N     = length(args)
    names = ntuple(i -> Symbol(:_, i), N)
    return :(Tangent(NamedTuple{$names}(args)))
end

vmap_rule!!(::typeof(Base.getindex), t::Tangent, idx::Int) =
    get_tangent_field(t, idx)

# NamedTuple field access via nt[:x] syntax - routes to get_tangent_field by name.
vmap_rule!!(::typeof(Base.getindex), t::Tangent, name::Symbol) =
    get_tangent_field(t, name)

# NamedTuple constructor call: NamedTuple{names}(batched_tuple) in the lifted IR.
# The argument t is a Tangent (from tuple-batching of the values), with positional
# :_1,:_2,… names. We repackage it under the actual NamedTuple field names.
@generated function vmap_rule!!(::Type{NamedTuple{names}}, t::Tangent) where {names}
    N = length(names)
    return quote
        Tangent(NamedTuple{$names}(ntuple(i -> get_tangent_field(t, i), $N)))
    end
end

# ── Struct SoA field access and mutation ─────────────────────────────────────
vmap_rule!!(::typeof(Base.getfield), t::PossiblyMutableTangent, name::Symbol) =
    get_tangent_field(t, name)

vmap_rule!!(::typeof(Base.getfield), t::PossiblyMutableTangent, idx::Int) =
    get_tangent_field(t, idx)

# lgetfield is Mooncake's Val-based getfield used in its own lifted IR.
# Route it to get_tangent_field the same way as getfield.
vmap_rule!!(::typeof(lgetfield), t::PossiblyMutableTangent, ::Val{name}) where {name} =
    get_tangent_field(t, name)

vmap_rule!!(::typeof(Base.setfield!), t::MutableTangent, name::Symbol, val) =
    (set_tangent_field!(t, name, val); val)

vmap_rule!!(::typeof(Base.setfield!), t::MutableTangent, idx::Int, val) =
    (set_tangent_field!(t, idx, val); val)

# ── Batch rules for vmap ──────────────────────────────────────────────────────
#
# Dispatch: vmap_rule!!(f, batched_args...) → BatchContainer
# Checked at IR-generation time via hasmethod - zero runtime overhead.
# Fallback (LazyVmap/DynamicVmap broadcast) used when no rule is registered.

# ── Scalar binary ops ─────────────────────────────────────────────────────────
# These operate on the raw Vector{T} backing store - SIMD-fused, no getindex.

for op in (:+, :-, :*, :/)
    @eval function vmap_rule!!(
        ::typeof($op),
        a::BatchContainer{T, Vector{T}},
        b::BatchContainer{T, Vector{T}},
    ) where {T<:_VmapScalar}
        return BatchContainer{T, Vector{T}}(broadcast($op, a.data, b.data), 1)
    end
end

function vmap_rule!!(::typeof(*), α::T, b::BatchContainer{T, Vector{T}}) where {T<:_VmapScalar}
    return BatchContainer{T, Vector{T}}(α .* b.data, 1)
end
function vmap_rule!!(::typeof(*), b::BatchContainer{T, Vector{T}}, α::T) where {T<:_VmapScalar}
    return BatchContainer{T, Vector{T}}(b.data .* α, 1)
end

# ── Matrix × batch-of-vectors: BLAS gemm ──────────────────────────────────────
# A (m×n) × data (n×N) → (m×N) - one gemm replaces N gemv calls.
# x.data is (n, N); A*x.data is a standard (m×n)×(n×N) matrix multiply, which is
# exactly the same as [A*x.data[:,k] for k in 1:N] but done in one BLAS call.

function vmap_rule!!(
    ::typeof(*),
    A::Matrix{T},
    x::BatchContainer{Vector{T}, Matrix{T}},
) where {T<:LinearAlgebra.BlasFloat}
    return BatchContainer{Vector{T}, Matrix{T}}(A * x.data, 2)
end

# ── Matrix × batch-of-matrices: reshape trick → BLAS gemm ────────────────────
#
# WHY THE RESHAPE TRICK IS CORRECT (Julia column-major, batch-always-last):
#
# data::Array{T,3} of size (L, M, N_batch):
#   Julia stores in memory as column-major, so each matrix Xₖ = data[:,:,k]
#   occupies M contiguous columns: [col1|col2|…|colM] for batch element k.
#   The full memory layout is:
#     [col1_X1 | col2_X1 | … | colM_X1 | col1_X2 | … | colM_X{N_batch}]
#
# After reshape(data, L, M*N_batch) - zero-copy, same memory pointer:
#   columns 1..M          → all M columns of batch element 1
#   columns M+1..2M       → all M columns of batch element 2
#   …
#   columns (k-1)M+1..kM  → all M columns of batch element k
#
# A * reshape(data, L, M*N_batch), where A is (m×L):
#   result columns 1..M    = A * X₁  ✓
#   result columns M+1..2M = A * X₂  ✓   (matrix multiply distributes over column blocks)
#   …
#
# reshape(result, m, M, N_batch)[:,:,k] = A * Xₖ  ✓  for all k.
#
# This is correct ONLY because batch is the LAST dimension.
# If batch were NOT last (e.g. batch along dim 1 of a 3-D array), the batch elements
# would not form contiguous column blocks after reshape, and the multiply would
# silently mix elements across different batch items.
# This is why BatchContainer enforces index = ndims(data) (batch always last).

function vmap_rule!!(
    ::typeof(*),
    A::Matrix{T},
    x::BatchContainer{Matrix{T}, Array{T,3}},
) where {T<:LinearAlgebra.BlasFloat}
    L, M, N = size(x.data)
    result = A * reshape(x.data, L, M * N)          # zero-copy reshape → single BLAS gemm
    return BatchContainer{Matrix{T}, Array{T,3}}(reshape(result, size(A, 1), M, N), 3)
end

# ── Matrix × batch of K-D tensors (K ≥ 3): generalized reshape trick ─────────
# Specific Vector/Matrix rules above take precedence for K = 1, 2.
#
# data shape: (d₁, d₂, …, dₖ, N_batch)
# The reshape trick contracts all non-first dims (d₂,…,dₖ, N_batch) into one axis:
#   reshape(data, d₁, d₂*…*dₖ*N_batch)  →  A * reshaped  →  reshape back
#
# Correctness argument is identical to the K=2 case: Julia is column-major, batch
# is last, so each "fiber" data[:,i₂,…,iₖ,k] is contiguous in memory regardless of K.
# reshape treats the (K+1)-D array as a 2-D view without touching memory - one gemm
# replaces N_batch*prod(inner_dims) separate matrix-vector calls.
#
# Semantics: A * x[i] contracts along the FIRST axis of each batch element, leaving
# all other axes unchanged. This is the mode-1 tensor product.

function vmap_rule!!(
    ::typeof(*),
    A::Matrix{T},
    x::BatchContainer{Array{T,K}, Array{T,K1}},
) where {T<:LinearAlgebra.BlasFloat, K, K1}
    d = size(x.data)                                        # (d₁, d₂, …, dₖ, N_batch)
    dtail = Base.tail(d)                                    # (d₂, …, dₖ, N_batch)
    result = A * reshape(x.data, d[1], prod(dtail))        # (m, d₂*…*dₖ*N_batch) - one gemm
    return BatchContainer{Array{T,K}, Array{T,K1}}(reshape(result, size(A, 1), dtail...), K1)
end

# ── Matrix × any AbstractArray-element batch with dense backing ───────────────
# Covers sparse vectors, static arrays, etc. that _wrap_input/make_batch materialised
# into a dense Array backing. The reshape trick is valid as long as the backing is a
# dense Array with batch-last layout - which _make_batch/_wrap_input always produce.
# Specific Vector/Matrix/Array{T,K} rules above are more specific and take precedence.
function vmap_rule!!(
    ::typeof(*),
    A::Matrix{T},
    x::BatchContainer{P, Array{T, K1}},
) where {T<:LinearAlgebra.BlasFloat, P, K1}
    d = size(x.data)
    dtail = Base.tail(d)
    result = A * reshape(x.data, d[1], prod(dtail))
    return BatchContainer{P, Array{T, K1}}(reshape(result, size(A, 1), dtail...), K1)
end

# ── dot: batch of vector pairs → batch of scalars ─────────────────────────────
# diag(AᵀB) = sum(A .* B, dims=1) - one fused pass over (n×N) matrices.
# Both a.data and b.data are (n, N); element-wise product then column-sum gives
# [dot(a.data[:,k], b.data[:,k]) for k in 1:N] without any loop.

function vmap_rule!!(
    ::typeof(LinearAlgebra.dot),
    a::BatchContainer{Vector{T}, Matrix{T}},
    b::BatchContainer{Vector{T}, Matrix{T}},
) where {T<:LinearAlgebra.BlasFloat}
    return BatchContainer{T, Vector{T}}(vec(sum(a.data .* b.data; dims=1)), 1)
end

# ── Reductions: collapse non-batch dims, keep batch axis ─────────────────────
# For an (L, N) matrix: sum/norm/mean over dim 1 → N results.
# For an (L, M, N) array: sum/norm/mean over dims (1,2) → N results.

function vmap_rule!!(::typeof(sum), bc::BatchContainer{Vector{T}, Matrix{T}}) where {T<:_VmapScalar}
    return BatchContainer{T, Vector{T}}(vec(sum(bc.data; dims=1)), 1)
end

function vmap_rule!!(::typeof(sum), bc::BatchContainer{Matrix{T}, Array{T,3}}) where {T<:_VmapScalar}
    return BatchContainer{T, Vector{T}}(vec(sum(bc.data; dims=(1,2))), 1)
end

function vmap_rule!!(::typeof(LinearAlgebra.norm), bc::BatchContainer{Vector{T}, Matrix{T}}) where {T<:_VmapScalar}
    return BatchContainer{T, Vector{T}}(vec(sqrt.(sum(abs2.(bc.data); dims=1))), 1)
end

function vmap_rule!!(::typeof(LinearAlgebra.norm), bc::BatchContainer{Matrix{T}, Array{T,3}}) where {T<:_VmapScalar}
    return BatchContainer{T, Vector{T}}(vec(sqrt.(sum(abs2.(bc.data); dims=(1,2)))), 1)
end

function vmap_rule!!(::typeof(mean), bc::BatchContainer{Vector{T}, Matrix{T}}) where {T<:_VmapScalar}
    return BatchContainer{T, Vector{T}}(vec(mean(bc.data; dims=1)), 1)
end

function vmap_rule!!(::typeof(mean), bc::BatchContainer{Matrix{T}, Array{T,3}}) where {T<:_VmapScalar}
    return BatchContainer{T, Vector{T}}(vec(mean(bc.data; dims=(1,2))), 1)
end

# ── General N-D array batch reductions (N ≥ 3) ───────────────────────────────
# Specific Vector/Matrix methods above take precedence for N = 1, 2.
# For N ≥ 3: data is (d₁, …, dₙ, batch); reduce over dims 1..N, keeping the batch axis.
# ntuple(identity, N) = (1, 2, …, N) - produced at compile time from the type parameter N.

function vmap_rule!!(
    ::typeof(sum),
    bc::BatchContainer{Array{T,N}, Array{T,N1}},
) where {T<:_VmapScalar, N, N1}
    return BatchContainer{T, Vector{T}}(vec(sum(bc.data; dims=ntuple(identity, N))), 1)
end

function vmap_rule!!(
    ::typeof(LinearAlgebra.norm),
    bc::BatchContainer{Array{T,N}, Array{T,N1}},
) where {T<:_VmapScalar, N, N1}
    return BatchContainer{T, Vector{T}}(
        vec(sqrt.(sum(abs2.(bc.data); dims=ntuple(identity, N)))), 1
    )
end

function vmap_rule!!(
    ::typeof(mean),
    bc::BatchContainer{Array{T,N}, Array{T,N1}},
) where {T<:_VmapScalar, N, N1}
    return BatchContainer{T, Vector{T}}(vec(mean(bc.data; dims=ntuple(identity, N))), 1)
end

# ── General element-wise catch-alls ───────────────────────────────────────────
#
# WHY THESE ARE SAFE:
# The constraint D<:AbstractArray{E} where E<:_VmapScalar ensures we only match
# SoA BatchContainers whose backing store holds _VmapScalar scalars.
# For any such container, f.(bc.data) applies f to every scalar in the backing
# array - equivalent to applying f element-wise to each batch item separately,
# because the SoA layout is a flat view of those scalars (batch-last, contiguous).
#
# Dispatch priority: all specific rules above use concrete function types
# (::typeof(sum), ::typeof(*) with Matrix, etc.), which are always more specific
# than ::F (free type parameter). Julia picks the specific rule first; these
# catch-alls only fire when no specific rule was registered.
#
# Failure mode: if f is NOT element-wise (e.g., sort on a scalar), f.(bc.data)
# will throw a MethodError at runtime - loud failure, not silent wrong answer.

# Unary: sin, cos, exp, log, sqrt, abs, relu, tanh, … - one broadcast over backing array.
function vmap_rule!!(
    f::F,
    bc::BatchContainer{T, D},
) where {F, T, E<:_VmapScalar, D<:AbstractArray{E}}
    return BatchContainer{T, D}(f.(bc.data), bc.index)
end

# Binary: +, -, *, / between two same-type array batches - one fused broadcast.
# Scalar-batch rules above (D=Vector{T}) are more specific and take precedence.
function vmap_rule!!(
    f::F,
    a::BatchContainer{T, D},
    b::BatchContainer{T, D},
) where {F, T, E<:_VmapScalar, D<:AbstractArray{E}}
    return BatchContainer{T, D}(f.(a.data, b.data), a.index)
end

# Scalar × batch (both orderings) for any rank - one broadcast, no loop.
# Scalar-batch rules above (D=Vector{T}) are more specific and take precedence.
function vmap_rule!!(
    ::typeof(*), α::E, bc::BatchContainer{T, D}
) where {E<:_VmapScalar, T, D<:AbstractArray{E}}
    return BatchContainer{T, D}(α .* bc.data, bc.index)
end
function vmap_rule!!(
    ::typeof(*), bc::BatchContainer{T, D}, α::E
) where {E<:_VmapScalar, T, D<:AbstractArray{E}}
    return BatchContainer{T, D}(bc.data .* α, bc.index)
end
