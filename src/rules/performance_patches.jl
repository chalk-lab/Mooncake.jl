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

# Contiguous NTuple lane loads vectorise across lanes. Keep scalar scale and tuple add
# separate: three-argument tuple muladd boxes.
# Bind only L: NTuple{L,P} becomes Tuple{} at L=0, leaving P unbound.
@inline _tadd(a::NTuple{L}, b::NTuple{L}) where {L} = ntuple(i -> a[i] + b[i], Val(L))
@inline _tscale(a::NTuple{L}, s) where {L} = ntuple(i -> a[i] * s, Val(L))

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
@is_primitive(DefaultCtx, Tuple{typeof(sum),Array{<:IEEEFloat}})
function frule!!(
    ::Lifted{typeof(sum),N},
    x::Lifted{Array{P,D},N,<:NDualArray{P,N,D,Array{P,D},NDual{P,N}}},
) where {N,P<:IEEEFloat,D}
    # Wide tuple loads avoid N scalar loads per element and keep the fold allocation-free.
    nda = tangent(x)
    pv = sum(getfield(nda, :primal))
    blk = getfield(getfield(nda, :partials_block), :parent)
    lanes = if N == 1
        # At width 1 the flat block is the lane; sum supplies a pairwise SIMD reduction.
        (sum(blk),)
    else
        # Four accumulators provide 4N independent adds, avoiding small-width latency stalls.
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

# Selecting from the primal avoids a per-element NDual fold. The NDualArray methods
# preserve the ties: maximum credits the last maximum, minimum the first minimum.
# Forward only; reverse uses derived rules.
@is_primitive MinimalCtx ForwardMode Tuple{typeof(maximum),Array{<:IEEEFloat}}
function frule!!(
    ::Lifted{typeof(maximum),N}, x::Lifted{Array{P,D},N,<:NDualArray{P,N,D}}
) where {N,P<:IEEEFloat,D}
    dy = maximum(tangent(x))
    return Lifted{P,N}(dy.value, dy)
end
@is_primitive MinimalCtx ForwardMode Tuple{typeof(minimum),Array{<:IEEEFloat}}
function frule!!(
    ::Lifted{typeof(minimum),N}, x::Lifted{Array{P,D},N,<:NDualArray{P,N,D}}
) where {N,P<:IEEEFloat,D}
    dy = minimum(tangent(x))
    return Lifted{P,N}(dy.value, dy)
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
    par = tangent(x).fields.parent
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
    nda = tangent(x)
    p = getfield(nda, :primal)
    v = sum(abs2, p)
    blk = getfield(getfield(nda, :partials_block), :parent)
    lanes = if N == 1
        # A scalar SIMD reduction avoids the tuple fold's serial dependency at width 1.
        acc = zero(P)
        @inbounds @simd for j in eachindex(p)
            acc += (2 * p[j]) * blk[j]
        end
        (acc,)
    else
        cols = reinterpret(NTuple{N,P}, blk)
        acc = ntuple(_ -> zero(P), Val(N))
        @inbounds for j in eachindex(p)
            acc = _tadd(acc, _tscale(cols[j], 2 * p[j]))
        end
        acc
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
    # Wider lanes have non-unit row stride; reusable dense buffers keep products on BLAS.
    dA = N == 1 ? dAs[1] : similar(pA)
    dB = N == 1 ? dBs[1] : similar(pB)
    dC = N == 1 ? view(blk,1,:,:) : similar(C)
    for k in 1:N
        if N != 1
            copyto!(dA, dAs[k])
            copyto!(dB, dBs[k])
        end
        mul!(dC, dA, pB)
        mul!(dC, pA, dB, one(P), one(P))
        N == 1 || copyto!(view(blk,k,:,:), dC)
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

# Both pullbacks contract dy in one memory-order pass and fold dense cotangents onto
# stored entries. Small blocks make per-(q,n) BLAS gemv calls slower than this loop.
function _kron_pb!(dx1, dx2, dy, px1, px2)
    T = eltype(px1)
    M, N = size(px1)
    P, Q = size(px2)
    W = reshape(dy, P, M, Q, N)
    t1 = densify_tangent(dx1)
    t2 = densify_tangent(dx2)
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
    increment_densified_tangent!!(dx1, t1)
    increment_densified_tangent!!(dx2, t2)
    return nothing
end

# https://github.com/chalk-lab/Mooncake.jl/issues/526
# Dense forward inputs support IEEEFloat; wrapped inputs need arrayify's BlasFloat.
# Wrapped Float16 stays derived. Keep modes separate: reverse supports only real IEEEFloat.
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(LinearAlgebra._kron!),Array{T,2},Array{T,2},Array{T,2}
} where {T<:IEEEFloat}
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(LinearAlgebra._kron!),AbstractMatrix{T},AbstractMatrix{T},AbstractMatrix{T}
} where {T<:BlasFloat}
# Reverse folds dense cotangents onto stored entries; complex stays derived.
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

# Write lane k at (m-1)*N+k in fill order, avoiding an output-sized temporary per lane.
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

# Tuple columns of the (N, size...) blocks vectorise the contiguous lane writes;
# linear indices follow _kron!'s column-major (j,l,i,k) fill order.
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

# Direct block access supports Float16 without arrayify; the primal runs once.
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

# Struct-lift wrappers need arrayify (BlasFloat only); dense inputs take the method above.
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
            _mask_unit_diagonal(dx1_s[lane]),
            px2,
            _mask_unit_diagonal(dx2_s[lane]),
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
# At least one strided operand guarantees dense output; matching structured wrappers
# can return structured output and must stay derived. The strided×strided intersection
# prevents primitive ambiguity. Reverse folds dense cotangents onto stored entries.
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

# Forward needs real BlasFloat for arrayify and NDualArray packing; Float16/complex
# stay derived. As in reverse, require dense output and declare the strided intersection.
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(kron),StridedMatrix{T},AbstractMatrix{T}
} where {T<:Union{Float32,Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(kron),AbstractMatrix{T},StridedMatrix{T}
} where {T<:Union{Float32,Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(kron),StridedMatrix{T},StridedMatrix{T}
} where {T<:Union{Float32,Float64}}
# Dense operands have partials blocks, allowing contiguous writes across all lanes.
# Wrapped operands keep the generic path because they have no block of their own.
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

# Symmetric/Hermitian conversion uses LAPACK, which requires stride-1 columns.
# Materialise the parent first: element-major lanes have stride N.
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
    # Materialise wrappers once to avoid per-element branches; dense matrices pass through.
    mx1 = _kron_densify(px1)
    mx2 = _kron_densify(px2)
    y = kron(mx1, mx2)
    A = typeof(y)
    # Write the fused JVP into the block to avoid output-sized allocations per lane.
    blk = Nfwd._block_type(A)(undef, Nfwd._block_dims(N, y)...)
    bp = Nfwd._block_storage(blk)
    for k in 1:N
        _kron!_jvp_lane_into_block!(
            bp,
            k,
            Val(N),
            mx1,
            _kron_densify(_mask_unit_diagonal(dx1s[k])),
            mx2,
            _kron_densify(_mask_unit_diagonal(dx2s[k])),
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

        # Forward-only primitive; repeated extrema below pin the tie conventions.
        map(precisions) do P
            flags = (
                P == Float16 ? true : false, :stability_and_allocs, (mode=ForwardMode,)
            )
            return (flags..., maximum, randn(rng, P, 11))
        end,
        # Finite differences cannot resolve ties; pinned partials credit the LAST maximum.
        # The fixed Vector tangent is width 1 only (skip_chunked).
        map([Float64, Float32]) do P
            opts = (mode=ForwardMode, oracle=(value=P(3), deriv=P(40)), skip_chunked=true)
            x = CoDual(P[1.0, 3.0, 2.0, 3.0], P[10.0, 20.0, 30.0, 40.0])
            return (false, :none, opts, maximum, x)
        end,

        # minimum(x), the mirror of the two `maximum` groups above.
        map(precisions) do P
            flags = (
                P == Float16 ? true : false, :stability_and_allocs, (mode=ForwardMode,)
            )
            return (flags..., minimum, randn(rng, P, 11))
        end,
        # Minimum credits the FIRST tied element, unlike maximum.
        map([Float64, Float32]) do P
            opts = (mode=ForwardMode, oracle=(value=P(1), deriv=P(20)), skip_chunked=true)
            x = CoDual(P[3.0, 1.0, 2.0, 1.0], P[10.0, 20.0, 30.0, 40.0])
            return (false, :none, opts, minimum, x)
        end,

        # sum(view(x, a:b))
        map(precisions) do P
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, view(randn(rng, P, 11), 2:9))
        end,

        # sum(abs2, x)
        map_prod(vcat(sum_sizes, [(0,), (0, 3)]), precisions) do (sz, P)
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

        # SubArray exercises the arrayify fallback, restricted to BlasFloat.
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

        # Allocating kron has its own block writer: check derivatives at widths 1 and 8.
        # Only Float32/Float64 are forward primitives; Float16 stays derived.
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
        # Symmetric lanes need densification before LAPACK; reverse folds onto the stored
        # triangle. One strided operand is required; both-Symmetric is tested as derived.
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
        # Both-Symmetric stays derived and leaves the unstored result triangle undef.
        # Float32 needs interface_only: finite differences cannot resolve this composite;
        # Float64 checks its derivative.
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

        # Complex Hermitian: forward wraps lanes, reverse stays derived because folding
        # onto the stored triangle needs conjugation. Check both against finite differences.
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
