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
    # Lane-`k` derivative is `Σᵢ blockₖᵢ` (∂(Σx)/∂partialₖ). On 1.11+ fold the contiguous element-
    # major block across lanes (vectorised); on 1.10 (parallel arrays, no block) sum each lane's
    # already-contiguous view. Stack-only / 0-alloc, as the `:allocs` test requires.
    nda = tangent(x)
    pv = sum(getfield(nda, :primal))
    lanes = @static if VERSION >= v"1.11-rc4"
        cols = reinterpret(reshape, NTuple{N,P}, getfield(nda, :partials_block))
        acc = ntuple(_ -> zero(P), Val(N))
        @inbounds for j in eachindex(cols)
            acc = _tadd(acc, cols[j])
        end
        acc
    else
        ntuple(k -> sum(tangent_view(nda, k)), Val(N))
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
@is_primitive(DefaultCtx, Tuple{typeof(sum),typeof(abs2),Array{<:IEEEFloat}})
function frule!!(
    ::Lifted{typeof(sum),N},
    ::Lifted{typeof(abs2),N},
    x::Lifted{Array{P,D},N,<:NDualArray{P,N,D,Array{P,D},NDual{P,N}}},
) where {N,P<:IEEEFloat,D}
    # Chain rule: lane-`k` derivative of `Σᵢ pᵢ²` is `Σᵢ 2pᵢ·blockₖᵢ`. On 1.11+ fold the contiguous
    # block, scaling each element's lane column by `2pᵢ` (vectorised); on 1.10 do `2·dot(p, lane-
    # view)`. Stack-only / 0-alloc (the `:allocs` test).
    nda = tangent(x)
    p = getfield(nda, :primal)
    v = sum(abs2, p)
    lanes = @static if VERSION >= v"1.11-rc4"
        cols = reinterpret(reshape, NTuple{N,P}, getfield(nda, :partials_block))
        acc = ntuple(_ -> zero(P), Val(N))
        @inbounds for j in eachindex(cols)
            acc = _tadd(acc, _tscale(cols[j], 2 * p[j]))
        end
        acc
    else
        ntuple(k -> 2 * dot(p, tangent_view(nda, k)), Val(N))
    end
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
# Reverse mode: dense and wrapped, real `IEEEFloat` only. `_kron_accum!` folds the dense gradient into
# dense/Triangular/Diagonal/Adjoint/Transpose fdata; Symmetric/Hermitian are admitted by the signature
# but their off-diagonal `setindex!` throws. Complex stays derived.
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(LinearAlgebra._kron!),AbstractMatrix{T},AbstractMatrix{T},AbstractMatrix{T}
} where {T<:IEEEFloat}
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

# 1.11+ block form of the per-lane JVP: writes all `N` lanes of each output element in one pass over
# the contiguous element-major partials blocks, so the length-`N` lane write vectorises (packed
# `<N x double>`). Blocks are `(N, size...)`; reinterpret to `NTuple{N,T}` columns, linear-indexed
# in the same column-major `(j,l,i,k)` order `_kron!` fills. ~6× the stride-`N` per-lane loop.
function _kron!_jvp_block!(outb, px1, x1b, px2, x2b, ::Val{N}) where {N}
    outc = reinterpret(reshape, NTuple{N,eltype(outb)}, outb)
    d1c = reinterpret(reshape, NTuple{N,eltype(x1b)}, x1b)
    d2c = reinterpret(reshape, NTuple{N,eltype(x2b)}, x2b)
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
    @static if VERSION >= v"1.11-rc4"
        _kron!_jvp_block!(
            getfield(tangent(out), :partials_block),
            px1,
            getfield(tangent(x1), :partials_block),
            px2,
            getfield(tangent(x2), :partials_block),
            Val(N),
        )
    else
        for lane in 1:N
            _kron!_jvp_lane!(
                tangent_view(out, lane),
                px1,
                tangent_view(x1, lane),
                px2,
                tangent_view(x2, lane),
            )
        end
    end
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
        _kron!_jvp_lane!(dout_s[lane], px1, dx1_s[lane], px2, dx2_s[lane])
    end
    return out
end
# Fold the dense per-entry cotangent `v` at (i,j) into the (possibly structured) input fdata `dx`.
# `matrixify` returns `dx` re-wrapped in the input's wrapper; a triangular/diagonal fdata stores
# only its structural entries, and the off-pattern (i,j) are structurally-zero *non-variables* of
# the primal, so their gradient contribution is dropped (writing them would both throw and be
# wrong — finite differences give zero there). Dense inputs take the first method unchanged, so the
# hot dense path allocates nothing extra.
@inline _kron_accum!(dx::AbstractMatrix, i, j, v) = (@inbounds dx[i, j] += v; nothing)
@inline _kron_tri_stored(::UpperTriangular, i, j) = i <= j
@inline _kron_tri_stored(::UnitUpperTriangular, i, j) = i < j
@inline _kron_tri_stored(::LowerTriangular, i, j) = i >= j
@inline _kron_tri_stored(::UnitLowerTriangular, i, j) = i > j
@inline function _kron_accum!(dx::LinearAlgebra.AbstractTriangular, i, j, v)
    _kron_tri_stored(dx, i, j) && (@inbounds parent(dx)[i, j] += v)
    return nothing
end
@inline _kron_accum!(dx::Diagonal, i, j, v) =
    (i == j && (@inbounds dx.diag[i] += v); nothing)

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
        P, Q = size(px2)
        for m in axes(px1, 1), n in axes(px1, 2)
            _kron_accum!(
                dx1,
                m,
                n,
                dot(
                    (@view dout[((m - 1) * P + 1):(m * P), ((n - 1) * Q + 1):(n * Q)]), px2
                ),
            )
        end
        for p in axes(px2, 1), q in axes(px2, 2)
            _kron_accum!(dx2, p, q, dot((@view dout[p:P:end, q:Q:end]), px1))
        end
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
# reverse pullbacks fold the dense gradient into a structured fdata via `_kron_accum!`, keeping only
# the wrapper's stored entries.
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
        M, N = size(dx1)
        P, Q = size(dx2)
        for m in 1:M, n in 1:N
            _kron_accum!(
                dx1,
                m,
                n,
                dot((@view dy[((m - 1) * P + 1):(m * P), ((n - 1) * Q + 1):(n * Q)]), px2),
            )
        end
        for p in 1:P, q in 1:Q
            _kron_accum!(dx2, p, q, dot((@view dy[p:P:end, q:Q:end]), px1))
        end
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
function Mooncake.frule!!(
    ::Lifted{typeof(kron),N},
    x1::Lifted{<:AbstractVecOrMat{T},N},
    x2::Lifted{<:AbstractVecOrMat{T},N},
) where {N,T<:Union{Float32,Float64}}
    px1, dx1s = arrayify(x1)
    px2, dx2s = arrayify(x2)
    # `convert(Matrix, ·)` passes dense `Matrix` inputs through unchanged and materialises wrapped
    # inputs (`view`/`UpperTriangular`) once, so the scalar `_kron!_jvp_lane!` loop below indexes
    # plain arrays instead of paying a per-element wrapper branch.
    mx1 = convert(Matrix, px1)
    mx2 = convert(Matrix, px2)
    y = kron(mx1, mx2)
    A = typeof(y)
    # Fuse the product rule `d(kron(x1,x2))ₖ = kron(dx1ₖ,x2) + kron(x1,dx2ₖ)` directly into each
    # partial: `kron(dx1ₖ,px2) + kron(px1,dx2ₖ)` allocates two output-sized `kron` temporaries per
    # lane (O(N) waste), whereas `_kron!_jvp_lane!` writes both terms in one pass with none.
    partials = ntuple(
        k -> _kron!_jvp_lane!(
            similar(y), mx1, convert(Matrix, dx1s[k]), mx2, convert(Matrix, dx2s[k])
        ),
        Val(N),
    )
    return Lifted{A,N}(y, NDualArray{T,N,2,A}(y, partials))
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

        # sum(abs2, x)
        map_prod(sum_sizes, precisions) do (sz, P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, abs2, randn(rng, P, sz...))
        end,

        # _kron!(x, y)
        map(precisions) do (P)
            return (
                true,
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
                true,
                :none,
                nothing,
                LinearAlgebra._kron!,
                zeros(P, 50, 50),
                view(randn(rng, P, 6, 6), 1:5, 1:5),
                randn(rng, P, 10, 10),
            )
        end,
    )
    memory = Any[]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    precisions = [Float64, Float32]
    test_cases = vcat(
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
    )
    memory = Any[]
    return test_cases, memory
end
