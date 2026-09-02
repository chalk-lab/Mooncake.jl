# All of the code here purely exists to work around current performance limitations of
# Mooncake.jl. In order to prevent this from getting out of hand, there are several
# conventions to which we adhere when writing these rules:
# 1. for each rule, a comment is added containing a link to the issue or issues that are
#   believed to describe the deficiencies of Mooncake.jl which cause the rule to be needed.
# 2. the number of concrete types for which the signature is valid is finite, and all are
#   tested. For example, `Array{<:IEEEFloat}` is a permissible type. The only exception to
#   this is the dimension of an `Array` argument. For example, it is fine to write rules for
#   `Array{Float64}`, despite the fact that this technically includes `Array{Float64,1}`,
#   `Array{Float64,2}`, `Array{Float64,3}`, etc.
#   `Diagonal{<:IEEEFloat}` is not, on the other hand, permissible. This is because we do
#   not know what the type of its `diag` field is, and it _could_ be any `AbstractVector`.
#   Something more precise like `Diagonal{P, Vector{P}} where {P<:IEEEFloat}` is fine.
#   This convention ensures that we are confident the rules here provide a strict
#   improvement over what we currently have, and helps to prevent the addition of flakey
#   rules which cause robustness or correctness problems.

# Fold the `L` lane-tangents of a chunked `NDualArray` from its contiguous element-major partials
# block (1.11+): each element's `L` lanes are one contiguous `NTuple{L,P}` column, so
# `reinterpret`+tuple-add vectorises across lanes (packed `<L x double>`), 5–6× over the stride-`L`
# per-lane `tangent_view` reductions. `s` stays a plain scalar — a 3-arg tuple `muladd` boxes, so
# scale-then-add is kept split.
# Bind only `L` (not the element type): `NTuple{L,P}` degenerates to `Tuple{}` at `L=0`, which
# mentions no `P`, so a `where {L,P}` there trips Aqua's unbound-args check.
@inline _tadd(a::NTuple{L}, b::NTuple{L}) where {L} = ntuple(i -> a[i] + b[i], Val(L))
@inline _tscale(a::NTuple{L}, s) where {L} = ntuple(i -> a[i] * s, Val(L))

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
@is_primitive(DefaultCtx, Tuple{typeof(sum),Array{<:IEEEFloat}})
function frule!!(
    ::Lifted{typeof(sum),N},
    x::Lifted{Array{P,D},N,<:NDualArray{P,N,D,Array{P,D},NDual{P,N}}},
) where {N,P<:IEEEFloat,D}
    # Lane-`k` derivative is `Σᵢ blockₖᵢ` (∂(Σx)/∂partialₖ): fold the contiguous element-major
    # block across lanes. Reinterpreting the flat parent to `NTuple{N,P}` loads each element's
    # lane column in one wide load; `N` separate scalar loads costs ~2x at width 8. Stack-only /
    # 0-alloc, as the `:allocs` test requires.
    nda = tangent(x)
    pv = sum(getfield(nda, :primal))
    blk = getfield(getfield(nda, :partials_block), :parent)
    lanes = if N == 1
        # One lane makes the tuple fold a single serial accumulator chain, which cannot vectorise:
        # ~0.45ns per element against the primal's ~0.05ns. At width 1 the block's flat parent IS
        # that lane, so `sum` applies directly and brings its own pairwise SIMD reduction. Wider
        # chunks already get instruction-level parallelism from their `N` independent lane chains.
        (sum(blk),)
    else
        # Four accumulators, not one: a single chain gives the CPU only `N` independent adds to
        # overlap, which is latency-bound at small widths (width 2 sat at ~5.5x the primal per
        # lane). Unrolling by four gives it `4N`.
        cols = reinterpret(NTuple{N,P}, blk)
        z = ntuple(_ -> zero(P), Val(N))
        a1, a2, a3, a4 = z, z, z, z
        n = length(cols)
        j = 1
        @inbounds while j + 3 <= n
            a1 = _tadd(a1, cols[j])
            a2 = _tadd(a2, cols[j + 1])
            a3 = _tadd(a3, cols[j + 2])
            a4 = _tadd(a4, cols[j + 3])
            j += 4
        end
        @inbounds while j <= n
            a1 = _tadd(a1, cols[j])
            j += 1
        end
        _tadd(_tadd(a1, a2), _tadd(a3, a4))
    end
    return Lifted{P,N}(pv, _scalar_ndual(pv, lanes))
end
function rrule!!(::CoDual{typeof(sum)}, x::CoDual{<:Array{P}}) where {P<:IEEEFloat}
    dx = x.dx
    function sum_pb!!(dz::P)
        dx .+= dz
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(identity, x.x)), sum_pb!!
end

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
@is_primitive(DefaultCtx, Tuple{typeof(sum),ContiguousSubVector{<:IEEEFloat}})
# Summed straight off the parent's lane over the view's own index range: `arrayify` is
# `BlasFloat`-only, and this rule is claimed for every `IEEEFloat`, `Float16` included.
function frule!!(
    ::Lifted{typeof(sum),N}, x::Lifted{ContiguousSubVector{P},N}
) where {N,P<:IEEEFloat}
    px = primal(x)
    v = sum(px)
    par = tangent(x).value.parent
    idx = parentindices(px)[1]
    lanes = ntuple(Val(N)) do k
        pl = Nfwd.tangent_view(par, k)
        acc = zero(P)
        @inbounds @simd for i in idx
            acc += pl[i]
        end
        acc
    end
    return Lifted{P,N}(v, _scalar_ndual(v, lanes))
end
function rrule!!(
    ::CoDual{typeof(sum)}, x::CoDual{ContiguousSubVector{P}}
) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    function sum_view_pb!!(dz::P)
        dx .+= dz
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(px)), sum_view_pb!!
end

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
@is_primitive(DefaultCtx, Tuple{typeof(sum),typeof(abs2),Array{<:IEEEFloat}})
function frule!!(
    ::Lifted{typeof(sum),N},
    ::Lifted{typeof(abs2),N},
    x::Lifted{Array{P,D},N,<:NDualArray{P,N,D,Array{P,D},NDual{P,N}}},
) where {N,P<:IEEEFloat,D}
    # Chain rule: lane-`k` derivative of `Σᵢ pᵢ²` is `Σᵢ 2pᵢ·blockₖᵢ`: fold the contiguous block,
    # scaling each element's lane column by `2pᵢ`. Reinterpreting the flat parent to `NTuple{N,P}`
    # loads each lane column in one wide load; `N` separate scalar loads costs ~2x at width 8.
    # Stack-only / 0-alloc (the `:allocs` test).
    nda = tangent(x)
    p = getfield(nda, :primal)
    v = sum(abs2, p)
    cols = reinterpret(NTuple{N,P}, getfield(getfield(nda, :partials_block), :parent))
    acc = ntuple(_ -> zero(P), Val(N))
    @inbounds for j in eachindex(p)
        acc = _tadd(acc, _tscale(cols[j], 2 * p[j]))
    end
    lanes = acc
    return Lifted{P,N}(v, _scalar_ndual(v, lanes))
end
function rrule!!(
    ::CoDual{typeof(sum)}, ::CoDual{typeof(abs2)}, x::CoDual{<:Array{P}}
) where {P<:IEEEFloat}
    function sum_abs2_pb!!(dz::P)
        x.dx .+= 2 .* x.x .* dz
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(sum(abs2, x.x)), sum_abs2_pb!!
end

# Without this, `A * B` differentiates through the in-place `gemm!` rule, which copies the
# output buffer so that the pullback can restore it. `*` allocates that buffer fresh, so
# the copy is pure overhead.
@is_primitive DefaultCtx Tuple{typeof(*),Matrix{P},Matrix{P}} where {P<:BlasRealFloat}
function frule!!(
    ::Lifted{typeof(*),N}, A::Lifted{<:Matrix{P},N}, B::Lifted{<:Matrix{P},N}
) where {N,P<:BlasRealFloat}
    pA, dAs = arrayify(A)
    pB, dBs = arrayify(B)
    C = pA * pB
    V = zero_dual(Val(N), C)
    blk = getfield(V, :partials_block)
    dC = similar(C)                                  # one scratch for every lane
    for k in 1:N
        mul!(dC, dAs[k], pB)
        mul!(dC, pA, dBs[k], one(P), one(P))
        copyto!(view(blk,k,:,:), dC)
    end
    return Lifted{typeof(C),N}(C, V)
end
function rrule!!(
    ::CoDual{typeof(*)}, A::CoDual{<:Matrix{P}}, B::CoDual{<:Matrix{P}}
) where {P<:BlasRealFloat}
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    C = pA * pB
    dC = zero(C)
    function matmul_pb!!(::NoRData)
        mul!(dA, dC, transpose(pB), one(P), one(P))
        mul!(dB, transpose(pA), dC, one(P), one(P))
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(C, dC), matmul_pb!!
end

# Differentiating the generic implementation element by element costs more than the rules
# either side of it; `Distances.pairwise(...; dims=1)` reaches it on every call.
@is_primitive DefaultCtx Tuple{typeof(permutedims),Matrix{P}} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(permutedims),N}, x::Lifted{<:Matrix{P},N}
) where {N,P<:IEEEFloat}
    px, dxs = arrayify(x)
    y = permutedims(px)
    V = zero_dual(Val(N), y)
    blk = getfield(V, :partials_block)
    # Written by hand rather than with `permutedims!`, whose `PermutedDimsArray` over a view
    # infers to a non-concrete type and costs the rule its type stability.
    for k in 1:N
        dk = dxs[k]
        lane = view(blk,k,:,:)
        @inbounds for j in axes(dk, 2), i in axes(dk, 1)
            lane[j, i] = dk[i, j]
        end
    end
    return Lifted{typeof(y),N}(y, V)
end
function rrule!!(::CoDual{typeof(permutedims)}, x::CoDual{<:Matrix{P}}) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    y = permutedims(px)
    dy = zero(y)
    function permutedims_pb!!(::NoRData)
        dx .+= transpose(dy)
        return NoRData(), NoRData()
    end
    return CoDual(y, dy), permutedims_pb!!
end

# Both `kron` pullbacks contract `dy` against one factor to accumulate into the other. The
# contraction is dense, so it goes through `densify` / `accumulate_densified!`. Read
# as `P x M x Q x N`, both contractions read the same element of `dy`, so a single pass in
# memory order serves both. A `gemv` per `(q, n)` block is slower at every shape measured:
# the blocks are small enough that BLAS call overhead dominates.
function _kron_pb!(dx1, dx2, dy, px1, px2)
    T = eltype(px1)
    M, N = size(px1)
    P, Q = size(px2)
    W = reshape(dy, P, M, Q, N)
    t1 = densify(dx1)
    t2 = densify(dx2)
    @inbounds for n in 1:N, q in 1:Q, i in 1:M
        acc = zero(T)
        x1 = px1[i, n]
        @simd for k in 1:P
            w = W[k, i, q, n]
            acc += w * px2[k, q]
            t2[k, q] += w * x1
        end
        t1[i, n] += acc
    end
    accumulate_densified!(dx1, t1)
    accumulate_densified!(dx2, t2)
    return nothing
end

# https://github.com/chalk-lab/Mooncake.jl/issues/526
# Forward mode is split by input shape. Dense inputs (all `Array{T,2}`) are a primitive for every
# real `IEEEFloat`, including Float16: the dense frule below reads the `NDualArray` partials directly
# and never touches `arrayify`. Wrapped inputs (Triangular/Symmetric/Adjoint/…) go through the
# `arrayify` fallback, which only supports `BlasFloat`, so they are a primitive only for `BlasFloat`;
# a Float16 wrapped input is left non-primitive and handled by derived forward mode rather than
# crashing inside `arrayify`. Both are `ForwardMode`-only: the `BlasFloat` widening exists purely for
# the forward wrapper frule and must not touch reverse, whose rrule is real (`IEEEFloat`) only —
# marking complex `_kron!` a reverse primitive would route complex reverse to a `MethodError`.
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(LinearAlgebra._kron!),Array{T,2},Array{T,2},Array{T,2}
} where {T<:IEEEFloat}
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(LinearAlgebra._kron!),AbstractMatrix{T},AbstractMatrix{T},AbstractMatrix{T}
} where {T<:BlasFloat}
# Reverse mode: dense and wrapped, real `IEEEFloat` only. `_kron_pb!` accumulates densely and
# folds the result onto each input's stored entries via `accumulate_densified!`. Complex stays
# derived.
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(LinearAlgebra._kron!),AbstractMatrix{T},AbstractMatrix{T},AbstractMatrix{T}
} where {T<:IEEEFloat}
# A tangent must not carry the primal wrapper's structural constants. Every wrapper `arrayify`
# admits stores structural *zeros* off-pattern, which are correct derivatives; the exceptions are
# the two unit triangulars, whose diagonal reads a constant `1` with derivative zero.
# `_arrayify_lane` cannot mask it upstream: the block scatter writes through its result.
@inline _kron_tangent_mask(z) = z
@inline _kron_tangent_mask(z::UnitUpperTriangular) = triu(parent(z), 1)
@inline _kron_tangent_mask(z::UnitLowerTriangular) = tril(parent(z), -1)

# One lane's Kronecker JVP into `dout_l`, written column-major to match `_kron!`'s fill order:
# d(kron(x1, x2)) = kron(dx1, x2) + kron(x1, dx2), element-wise to avoid allocation.
function _kron!_jvp_lane!(dout_l, px1, dx1_l, px2, dx2_l)
    m = firstindex(dout_l)
    for j in axes(px1, 2), l in axes(px2, 2), i in axes(px1, 1)
        x1ij = px1[i, j]
        dx1ij = dx1_l[i, j]
        for k in axes(px2, 1)
            dout_l[m] = (x1ij * dx2_l[k, l]) + (dx1ij * px2[k, l])
            m += 1
        end
    end
    return dout_l
end

# As `_kron!_jvp_lane!`, but writing lane `lane` straight into the element-major block instead of
# into its own dense array. Element `m`'s lane sits at flat index `(m-1)*N + lane`, and the loop
# walks output elements in the same order `_kron!` fills them, so the cursor just steps by `N`.
# This is what keeps the allocating `kron` frule from paying an output-sized temporary per lane.
function _kron!_jvp_lane_into_block!(
    blk, lane::Int, ::Val{N}, px1, dx1_l, px2, dx2_l
) where {N}
    off = lane
    @inbounds for j in axes(px1, 2), l in axes(px2, 2), i in axes(px1, 1)
        x1ij = px1[i, j]
        dx1ij = dx1_l[i, j]
        for k in axes(px2, 1)
            blk[off] = (x1ij * dx2_l[k, l]) + (dx1ij * px2[k, l])
            off += N
        end
    end
    return blk
end

# Block form of the per-lane JVP: writes all `N` lanes of each output element in one pass over
# the contiguous element-major partials blocks, so the length-`N` lane write vectorises (packed
# `<N x double>`). Blocks are `(N, size...)`; reinterpret their flat parents to `NTuple{N,T}`
# columns, linear-indexed in the same column-major `(j,l,i,k)` order `_kron!` fills. ~6× the
# stride-`N` per-lane loop.
function _kron!_jvp_block!(outb, px1, x1b, px2, x2b, ::Val{N}) where {N}
    outc = reinterpret(NTuple{N,eltype(outb)}, getfield(outb, :parent))
    d1c = reinterpret(NTuple{N,eltype(x1b)}, getfield(x1b, :parent))
    d2c = reinterpret(NTuple{N,eltype(x2b)}, getfield(x2b, :parent))
    m = 1
    @inbounds for j in axes(px1, 2), l in axes(px2, 2), i in axes(px1, 1)
        x1ij = px1[i, j]
        d1 = d1c[(j - 1) * size(px1, 1) + i]
        for k in axes(px2, 1)
            x2kl = px2[k, l]
            d2 = d2c[(l - 1) * size(px2, 1) + k]
            outc[m] = ntuple(t -> x1ij * d2[t] + d1[t] * x2kl, Val(N))
            m += 1
        end
    end
    return outb
end

# Dense fast path: read each lane's partials directly off the `NDualArray` V; the primal
# `LinearAlgebra._kron!(pout, px1, px2)` runs once. Covers every real `IEEEFloat` (including
# Float16, which `arrayify` does not support). Only the matrix (D=2) input shape is supported.
function Mooncake.frule!!(
    ::Lifted{typeof(LinearAlgebra._kron!),N},
    out::Lifted{Aout,N,<:NDualArray{T,N,2,Aout}},
    x1::Lifted{A1,N,<:NDualArray{T,N,2,A1}},
    x2::Lifted{A2,N,<:NDualArray{T,N,2,A2}},
) where {N,T<:IEEEFloat,Aout<:AbstractMatrix{T},A1<:AbstractMatrix{T},A2<:AbstractMatrix{T}}
    pout = primal(out)
    px1 = primal(x1)
    px2 = primal(x2)
    LinearAlgebra._kron!(pout, px1, px2)
    _kron!_jvp_block!(
        getfield(tangent(out), :partials_block),
        px1,
        getfield(tangent(x1), :partials_block),
        px2,
        getfield(tangent(x2), :partials_block),
        Val(N),
    )
    return out
end

# Wrapper fallback: the broad `@is_primitive` admits wrapped inputs (SubArray/Triangular/
# Symmetric/Reshaped/…) whose forward V is the generic struct lift, not `NDualArray`, so the dense
# method above does not match — without this they would `MethodError` at call time while the reverse
# `_kron!` rrule handles them. `arrayify` canonicalises the primal and each lane's partial through
# the wrapper (no copy), mirroring the reverse rrule and the forward `kron` frule. `BlasFloat` only
# (what `arrayify` supports); the dense method is strictly more specific, so dense inputs still take
# it, and dense `out` paired with a wrapped input routes here.
function Mooncake.frule!!(
    ::Lifted{typeof(LinearAlgebra._kron!),N},
    out::Lifted{<:AbstractMatrix{T},N},
    x1::Lifted{<:AbstractMatrix{T},N},
    x2::Lifted{<:AbstractMatrix{T},N},
) where {N,T<:BlasFloat}
    pout, dout_s = arrayify(out)
    px1, dx1_s = arrayify(x1)
    px2, dx2_s = arrayify(x2)
    LinearAlgebra._kron!(pout, px1, px2)
    for lane in 1:N
        _kron!_jvp_lane!(
            dout_s[lane],
            px1,
            _kron_tangent_mask(dx1_s[lane]),
            px2,
            _kron_tangent_mask(dx2_s[lane]),
        )
    end
    return out
end
function Mooncake.rrule!!(
    ::CoDual{typeof(LinearAlgebra._kron!)},
    out::CoDual{<:AbstractMatrix{<:T}},
    x1::CoDual{<:AbstractVecOrMat{<:T}},
    x2::CoDual{<:AbstractVecOrMat{<:T}},
) where {T<:Base.IEEEFloat}
    pout, dout = arrayify(out)
    px1, dx1 = matrixify(x1)
    px2, dx2 = matrixify(x2)
    old_pout = copy(pout)
    LinearAlgebra._kron!(pout, px1, px2)
    function _kron!_pb!!(::NoRData)
        _kron_pb!(dx1, dx2, dout, px1, px2)
        copyto!(pout, old_pout)
        fill!(dout, zero(T))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return out, _kron!_pb!!
end

# Using the rule for `_kron!` above makes performance on `kron` better, but still not as
# good as it _could_ be. To maximise performance we need a rule specifically for `kron`
# itself. See https://github.com/chalk-lab/Mooncake.jl/pull/886
# Primitive only when the output is a dense `Array` — i.e. at least one operand is strided.
# `kron` preserves structure (returns `UpperTriangular`/`Diagonal`/… whose canonical tangent is a
# wrapper, not the dense matrix this rule builds) exactly when BOTH operands are the same
# structured wrapper; those cases route to the derived rule instead. The strided×strided
# declaration is the intersection of the other two, so "≥1 strided operand" carries no
# `_is_primitive` ambiguity. The derived forward path builds the canonical wrapper dual; the
# reverse pullbacks accumulate densely and fold onto a structured fdata via
# `accumulate_densified!`, keeping only the wrapper's stored entries.
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(kron),StridedMatrix{T},AbstractMatrix{T}
} where {T<:IEEEFloat}
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(kron),AbstractMatrix{T},StridedMatrix{T}
} where {T<:IEEEFloat}
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(kron),StridedMatrix{T},StridedMatrix{T}
} where {T<:IEEEFloat}
function Mooncake.rrule!!(
    ::CoDual{typeof(kron)},
    x1::CoDual{<:AbstractVecOrMat{<:T}},
    x2::CoDual{<:AbstractVecOrMat{<:T}},
) where {T<:Base.IEEEFloat}
    px1, dx1 = matrixify(x1)
    px2, dx2 = matrixify(x2)
    y = kron(px1, px2)
    dy = zero(y)
    function kron_pb!!(::NoRData)
        _kron_pb!(dx1, dx2, dy, px1, px2)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), kron_pb!!
end

# Forward analogue of the `kron` rrule above: make `kron` a forward primitive too (mirrors the
# reverse one declared for `ReverseMode`). `arrayify` canonicalises each lane's tangent through the
# input wrapper (Matrix/SubArray/Triangular/…) to a dense partial, then the bilinear product rule
# gives the JVP per lane: `d(kron(x1,x2))ₖ = kron(dx1ₖ, x2) + kron(x1, dx2ₖ)`. Restricted to real
# `BlasFloat` (what `arrayify` and the real `NDualArray{…,NDual{T,N}}` packing support; the tested
# precisions). Float16 / complex `kron` stay derived in forward mode.
# Primitive only for dense `Array` output (≥1 strided operand); structure-preserving kron (both
# the same structured wrapper) falls back to the derived rule, which builds the canonical
# `ImmutableDual` this dense rule cannot. Strided×strided is the intersection — no ambiguity.
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(kron),StridedMatrix{T},AbstractMatrix{T}
} where {T<:Union{Float32,Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(kron),AbstractMatrix{T},StridedMatrix{T}
} where {T<:Union{Float32,Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(kron),StridedMatrix{T},StridedMatrix{T}
} where {T<:Union{Float32,Float64}}
# Dense fast path. Both operands' partials blocks exist, so `_kron!_jvp_block!` can store all `N`
# lanes of an output element in one contiguous write; the generic method below writes lane by lane
# at stride `N`, which grows cache-unfriendly with width (12.3x the primal per lane at width 8
# against 4.4x at width 1). A wrapped operand has no block of its own and keeps the generic path.
function Mooncake.frule!!(
    ::Lifted{typeof(kron),N},
    x1::Lifted{Matrix{T},N,<:NDualArray{T,N,2,Matrix{T}}},
    x2::Lifted{Matrix{T},N,<:NDualArray{T,N,2,Matrix{T}}},
) where {N,T<:Union{Float32,Float64}}
    px1, px2 = primal(x1), primal(x2)
    y = kron(px1, px2)
    A = typeof(y)
    blk = Nfwd._block_type(A)(undef, Nfwd._block_dims(N, y)...)
    _kron!_jvp_block!(
        blk,
        px1,
        getfield(tangent(x1), :partials_block),
        px2,
        getfield(tangent(x2), :partials_block),
        Val(N),
    )
    V = NDualArray{T,N,2,A,Nfwd._wrapped_eltype(T, Val(N)),typeof(blk)}(y, blk)
    return Lifted{A,N}(y, V)
end

# `convert(Matrix, ::Symmetric)` goes through `copytrito!` and LAPACK's `lacpy!`, which demands
# stride-1 columns. A lane of the element-major block is a stride-`N` view, so that path threw for
# every chunk width above 1 -- including the default 8, leaving only an explicit `chunk_size=1`
# working. Densifying the parent first sidesteps LAPACK; measured over the wrappers `arrayify`
# admits, `Symmetric` and `Hermitian` are the only two that cannot convert from a strided view.
@inline _kron_densify(z::AbstractMatrix) = convert(Matrix, z)
@inline _kron_densify(z::Symmetric) = convert(
    Matrix, Symmetric(Matrix(parent(z)), Symbol(z.uplo))
)
@inline _kron_densify(z::Hermitian) = convert(
    Matrix, Hermitian(Matrix(parent(z)), Symbol(z.uplo))
)
function Mooncake.frule!!(
    ::Lifted{typeof(kron),N},
    x1::Lifted{<:AbstractVecOrMat{T},N},
    x2::Lifted{<:AbstractVecOrMat{T},N},
) where {N,T<:Union{Float32,Float64}}
    px1, dx1s = arrayify(x1)
    px2, dx2s = arrayify(x2)
    # `_kron_densify` passes dense `Matrix` inputs through unchanged and materialises wrapped
    # inputs (`view`/`UpperTriangular`/`Symmetric`) once, so the scalar `_kron!_jvp_lane!` loop
    # below indexes plain arrays instead of paying a per-element wrapper branch.
    mx1 = _kron_densify(px1)
    mx2 = _kron_densify(px2)
    y = kron(mx1, mx2)
    A = typeof(y)
    # Fuse the product rule `d(kron(x1,x2))ₖ = kron(dx1ₖ,x2) + kron(x1,dx2ₖ)` into one pass per
    # lane, written straight into the result's block. Going via a per-lane array and then packing
    # cost TWO output-sized allocations per lane, which at chunk width 8 was 87 MB against the
    # 51 MB the block itself needs, and took the per-lane time from 5x the primal to 35x.
    blk = Nfwd._block_type(A)(undef, Nfwd._block_dims(N, y)...)
    bp = Nfwd._block_storage(blk)
    for k in 1:N
        _kron!_jvp_lane_into_block!(
            bp,
            k,
            Val(N),
            mx1,
            _kron_densify(_kron_tangent_mask(dx1s[k])),
            mx2,
            _kron_densify(_kron_tangent_mask(dx2s[k])),
        )
    end
    V = NDualArray{T,N,2,A,Nfwd._wrapped_eltype(T, Val(N)),typeof(blk)}(y, blk)
    return Lifted{A,N}(y, V)
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    sum_sizes = [(11,), (11, 3)]
    precisions = [Float64, Float32, Float16]
    test_cases = vcat(

        # sum(x)
        map_prod(sum_sizes, precisions) do (sz, P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, randn(rng, P, sz...))
        end,

        # sum(view(x, a:b))
        map(precisions) do P
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, view(randn(rng, P, 11), 2:9))
        end,

        # sum(abs2, x)
        map_prod(sum_sizes, precisions) do (sz, P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, abs2, randn(rng, P, sz...))
        end,

        # _kron!(x, y). `interface_only` for `Float16` alone, as the `sum` rows above: finite
        # differences are hopeless at that precision and meaningful at the others.
        map(precisions) do (P)
            return (
                P == Float16,
                :none,
                nothing,
                LinearAlgebra._kron!,
                zeros(P, 50, 50),
                randn(rng, P, 5, 5),
                randn(rng, P, 10, 10),
            )
        end,

        # _kron!(x, y) with a strided wrapper input (SubArray): exercises the `arrayify`
        # fallback frule. The dense-only forward frule matched only `NDualArray` slots, so a
        # wrapper input (forward V is the generic struct lift) that the broad `@is_primitive` and
        # the reverse rrule admit would MethodError. `BlasFloat` only (what `arrayify` supports).
        map([Float64, Float32]) do P
            return (
                false,
                :none,
                nothing,
                LinearAlgebra._kron!,
                zeros(P, 50, 50),
                view(randn(rng, P, 6, 6), 1:5, 1:5),
                randn(rng, P, 10, 10),
            )
        end,

        # The ALLOCATING `kron`, which the in-place cases above do not reach: it has its own frule
        # that builds the result's block and writes each lane into it at stride `N`. Registered so
        # `test_rule` drives it at widths 1-3 — the width > 1 path was previously untested, and an
        # element the lane writer skipped would keep the `undef` the block is allocated with.
        # `interface_only=false` unlike the `_kron!` cases above: the result here is freshly
        # allocated, so finite differences are meaningful and the derivative itself is checked.
        # With `true` these would assert only that the rule runs, which pins none of the above.
        # `Float32`/`Float64` only: unlike the in-place `_kron!`, the allocating frule is bounded to
        # those, so `Float16` has no forward primitive and builds a derived rule instead.
        map([Float64, Float32]) do P
            return (false, :none, nothing, kron, randn(rng, P, 5, 4), randn(rng, P, 3, 6))
        end,
        # Wrapped operands take the `arrayify`/`convert` path into the same lane writer.
        map([Float64, Float32]) do P
            return (
                false,
                :none,
                nothing,
                kron,
                view(randn(rng, P, 6, 6), 1:5, 1:4),
                UpperTriangular(randn(rng, P, 3, 3)),
            )
        end,
        # `Symmetric` is the wrapper whose lane partial cannot be densified through LAPACK, so
        # every width above 1 threw, and whose reverse cotangent needs folding onto the stored
        # triangle. One operand each here because `@is_primitive` wants a `StridedMatrix` on one
        # side; both-`Symmetric` is not primitive and is registered as a derived case instead.
        map([Float64, Float32]) do P
            return (
                false,
                :none,
                nothing,
                kron,
                Symmetric(randn(rng, P, 3, 3)),
                randn(rng, P, 4, 2),
            )
        end,
        map([Float64, Float32]) do P
            return (
                false,
                :none,
                nothing,
                kron,
                randn(rng, P, 3, 4),
                Symmetric(randn(rng, P, 3, 3), :L),
            )
        end,

        # A real `Hermitian` reaches the same two paths through its own `arrayify` overload.
        map([Float64, Float32]) do P
            return (
                false,
                :none,
                nothing,
                kron,
                Hermitian(randn(rng, P, 3, 3)),
                randn(rng, P, 4, 2),
            )
        end,
        map([Float64, Float32]) do P
            return (
                false,
                :none,
                nothing,
                kron,
                randn(rng, P, 3, 4),
                Hermitian(randn(rng, P, 3, 3), :L),
            )
        end,

        # permutedims(x)
        map([Float64, Float32]) do P
            return (false, :stability, nothing, permutedims, randn(rng, P, 7, 11))
        end,

        # x * y
        map([Float64, Float32]) do P
            return (
                false, :stability, nothing, *, randn(rng, P, 7, 11), randn(rng, P, 11, 5)
            )
        end,
    )
    memory = Any[]
    return test_cases, memory
end

_square_matmul(x) = x * x

function derived_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    precisions = [Float64, Float32]
    test_cases = vcat(
        # Both operands `Symmetric`: not primitive (the declaration wants a `StridedMatrix` on one
        # side), and previously untestable at all -- the result is a `Symmetric` whose unstored
        # triangle Base leaves `undef`, which `has_equal_data` compared until it was taught to read
        # through the wrapper. `Float32` is `interface_only`, as the `Float32` `det` cases in
        # `lapack.jl` are: the finite-difference oracle cannot resolve this composite at that
        # precision, though the rule is right -- its `Float32` gradient matches a `Float64`
        # reference to 2.5e-7, and the `Float64` case below checks the same code path against
        # finite differences.
        map([Float64, Float32]) do P
            return (
                P == Float32,
                :none,
                nothing,
                LinearAlgebra.kron,
                Symmetric(randn(rng, P, 3, 3)),
                Symmetric(randn(rng, P, 3, 3), :L),
            )
        end,

        # A COMPLEX `Hermitian`, where the two modes reach the answer differently: forward through
        # the `_arrayify_lane` wrapper (`Hermitian(dA)` is the JVP), reverse through the derived
        # path, since folding a complex cotangent onto the stored triangle needs a conjugation
        # `accumulate_densified!` does not apply. Both are checked here against finite
        # differences.
        map([ComplexF64, ComplexF32]) do C
            return map([:U, :L]) do uplo
                return (
                    false,
                    :none,
                    nothing,
                    LinearAlgebra.kron,
                    Hermitian(randn(rng, C, 3, 3), uplo),
                    randn(rng, C, 4, 2),
                )
            end
        end...,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                randn(rng, P, 5, 5),
                UpperTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                randn(rng, P, 5, 5),
                LowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                UpperTriangular(randn(rng, P, 5, 5)),
                LowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                randn(rng, P, 5, 5),
                UnitUpperTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                randn(rng, P, 5, 5),
                UnitLowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                UnitUpperTriangular(randn(rng, P, 5, 5)),
                UnitLowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                view(randn(rng, P, 5, 5), 1:5, 1:5),
                LowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                view(randn(rng, P, 5, 5), 1:5, 1:5),
                UpperTriangular(randn(rng, P, 10, 10)),
            )
        end,
        # Diagonal operand: the reverse pullback must fold only the diagonal of the dense
        # gradient into the `Diagonal` fdata (off-diagonal are structural zeros, dropped).
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                Diagonal(randn(rng, P, 4)),
                randn(rng, P, 3, 3),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                randn(rng, P, 4, 4),
                Diagonal(randn(rng, P, 3)),
            )
        end,

        # `A * A` aliases the rule's arguments, so `dA === dB` and the pullback must
        # accumulate both terms into the one array.
        map(precisions) do (P)
            return (false, :none, nothing, _square_matmul, randn(rng, P, 5, 5))
        end,
    )
    memory = Any[]
    return test_cases, memory
end
