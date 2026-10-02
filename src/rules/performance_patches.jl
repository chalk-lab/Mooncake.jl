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
@is_primitive DefaultCtx ReverseMode Tuple{typeof(sum),Array{<:IEEEFloat}}
for T in (Float16, Float32, Float64)
    @eval @is_primitive DefaultCtx ForwardMode Tuple{typeof(sum),Array{$T}}
end
function frule!!(
    ::Lifted{typeof(sum),N},
    x::Lifted{Array{P,D},N,<:NDualArray{P,N,D,Array{P,D},NDual{P,N}}},
) where {N,P<:IEEEFloat,D}
    # Wide tuple loads avoid N scalar loads per element and keep the fold allocation-free.
    nda = tangent(x)
    Nfwd._check_block(nda)
    pv = sum(getfield(nda, :primal))
    blk = getfield(getfield(nda, :partials_block), :parent)
    lanes = if N == 1
        # At width 1 the flat block is the lane; sum supplies a pairwise SIMD reduction.
        (sum(view(blk, 1:length(getfield(nda, :primal)))),)
    else
        # Four accumulators provide 4N independent adds, avoiding small-width latency stalls.
        cols = reinterpret(NTuple{N,P}, blk)
        z = ntuple(_ -> zero(P), Val(N))
        a1, a2, a3, a4 = z, z, z, z
        n = length(getfield(nda, :primal))
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
for f in (maximum, minimum)
    for T in (Float16, Float32, Float64)
        @eval @is_primitive MinimalCtx ForwardMode Tuple{typeof($f),Array{$T}}
    end
    @eval begin
        function frule!!(
            ::Lifted{typeof($f),N}, x::Lifted{Array{P,D},N,<:NDualArray{P,N,D}}
        ) where {N,P<:IEEEFloat,D}
            dy = $f(tangent(x))
            return Lifted{P,N}(dy.value, dy)
        end
    end
end

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
@is_primitive DefaultCtx ReverseMode Tuple{typeof(sum),ContiguousSubVector{<:IEEEFloat}}
for T in (Float16, Float32, Float64)
    @eval @is_primitive DefaultCtx ForwardMode Tuple{typeof(sum),ContiguousSubVector{$T}}
end
# Summed straight off the parent's lane over the view's own index range.
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
@is_primitive DefaultCtx ReverseMode Tuple{typeof(sum),typeof(abs2),Array{<:IEEEFloat}}
for T in (Float16, Float32, Float64)
    @eval @is_primitive DefaultCtx ForwardMode Tuple{typeof(sum),typeof(abs2),Array{$T}}
end
function frule!!(
    ::Lifted{typeof(sum),N},
    ::Lifted{typeof(abs2),N},
    x::Lifted{Array{P,D},N,<:NDualArray{P,N,D,Array{P,D},NDual{P,N}}},
) where {N,P<:IEEEFloat,D}
    nda = tangent(x)
    Nfwd._check_block(nda)
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

# Derived rebuilds a dual at every scalar store; this rule builds numeric storage once.
# The direct rule reduces forward cost on Julia 1.10 through 1.12.
for T in (Float32, Float64)
    @eval begin
        @is_primitive DefaultCtx ForwardMode Tuple{
            typeof(hvcat),Tuple{Vararg{Int}},$T,Vararg{$T}
        }
        @inline function frule!!(
            ::Lifted{typeof(hvcat),N},
            rows::Lifted{<:Tuple{Vararg{Int}},N},
            x::Lifted{$T,N,V},
            xs::Vararg{Lifted{$T,N,V}},
        ) where {N,V}
            rs = primal(rows)
            K = length(xs) + 1
            y =
                if isempty(rs) ||
                    first(rs) < 0 ||
                    length(rs) * first(rs) != K ||
                    !all(==(first(rs)), rs)
                    primals = Vector{$T}(undef, K)
                    for q in 1:K
                        primals[q] = primal(q == 1 ? x : xs[q - 1])
                    end
                    hvcat(rs, primals...)::Matrix{$T}
                else
                    Matrix{$T}(undef, length(rs), first(rs))
                end
            v = uninit_dual(Val(N), y)
            block = getfield(v, :partials_block)
            @inbounds y[1, 1] = primal(x)
            @inbounds for k in 1:N
                block[k, 1, 1] = tangent(x, k)
            end
            @inbounds for j in axes(y, 2), i in (1 + (j == 1)):size(y, 1)
                value = xs[(i - 1) * size(y, 2) + j - 1]::Lifted{$T,N,V}
                y[i, j] = primal(value)
                for k in 1:N
                    block[k, i, j] = tangent(value, k)
                end
            end
            return Lifted{typeof(y),N}(y, v)
        end
    end
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
    if N > 1
        Ab, _ = _partials_block(A)
        Bb, _ = _partials_block(B)
        m, n = size(C)
        p = size(pA, 2)
        BLAS.gemm!(
            'N', 'N', one(P), reshape(Ab, N * m, p), pB, zero(P), reshape(blk, N * m, n)
        )
        for j in 1:n
            BLAS.gemm!('N', 'T', one(P), view(Bb,:,:,j), pA, one(P), view(blk,:,:,j))
        end
    else
        dC = view(blk,1,:,:)
        mul!(dC, dAs[1], pB)
        mul!(dC, pA, dBs[1], one(P), one(P))
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
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(permutedims),Matrix{P}
} where {P<:IEEEFloat}
for T in (Float16, Float32, Float64, Union{})
    @eval @is_primitive DefaultCtx ForwardMode Tuple{typeof(permutedims),Matrix{$T}}
end
function frule!!(
    ::Lifted{typeof(permutedims),N}, x::Lifted{<:Matrix{P},N}
) where {N,P<:IEEEFloat}
    if _storage_kind(P) isa StructuralStorage
        P === Union{} || _throw_nonconcrete_primitive(P)
        return zero_lifted(Val(N), permutedims(primal(x)))
    end
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

# Julia signatures cannot express arbitrary wrapper depth; follow arrayify's parent walk.
function _kron_arrayifiable(mode, ::Type{A}) where {A}
    isconcretetype(A) || return false
    T = eltype(A)
    T <: Union{IEEEFloat,BlasFloat} || return false
    A <: Array && return true
    @static if VERSION >= v"1.11-rc4"
        A <: Memory && return true
    end
    if A <: Union{SubArray,Adjoint,Transpose}
        return _kron_arrayifiable(mode, fieldtype(A, :parent))
    elseif A <: Union{ReshapedArray,Base.ReinterpretArray}
        mode === ReverseMode && !(T <: BlasFloat) && return false
        return _kron_arrayifiable(mode, fieldtype(A, :parent))
    elseif A <: Diagonal
        mode === ReverseMode && !(T <: BlasFloat) && return false
        return _kron_arrayifiable(mode, fieldtype(A, :diag))
    elseif A <: Union{
        UpperTriangular,
        LowerTriangular,
        UnitUpperTriangular,
        UnitLowerTriangular,
        Symmetric,
        Hermitian,
    }
        P = fieldtype(A, :data)
        if mode === ReverseMode
            A <: Symmetric || T <: IEEEFloat || return false
            A <: Union{Symmetric,Hermitian} && !(P <: StridedMatrix) && return false
        end
        return _kron_arrayifiable(mode, P)
    end
    return false
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

@inline _check_kron_destination(::Array) = nothing
@static if VERSION >= v"1.11-rc4"
    @inline _check_kron_destination(::Memory) = nothing
end
@inline function _check_kron_destination(A::SubArray)
    all(
        i -> i isa Union{Integer,AbstractUnitRange,Base.Slice} || allunique(i), A.indices
    ) || throw(
        ArgumentError(
            "_kron! refuses destinations with repeated indices in any wrapper, even if the outer selection is injective",
        ),
    )
    return _check_kron_destination(parent(A))
end
@inline function _check_kron_destination(
    A::Union{
        ReshapedArray,
        Base.ReinterpretArray,
        Adjoint,
        Transpose,
        LinearAlgebra.AbstractTriangular,
        Symmetric,
        Hermitian,
    },
)
    return _check_kron_destination(parent(A))
end
@inline _check_kron_destination(A::Diagonal) = _check_kron_destination(A.diag)

# https://github.com/chalk-lab/Mooncake.jl/issues/526
# Dense forward inputs support IEEEFloat; wrapped Float16 stays derived.
# Keep modes separate: reverse supports only real IEEEFloat.
# Union elements need the derived elementwise rule; arrayify requires numeric storage.
for T in (Float16, Float32, Float64, Union{})
    @eval @is_primitive DefaultCtx ForwardMode Tuple{
        typeof(LinearAlgebra._kron!),Array{$T,2},Array{$T,2},Array{$T,2}
    }
end
for (mode, types) in (
        (ForwardMode, (Float32, Float64, ComplexF32, ComplexF64, Union{})),
        (ReverseMode, (Float16, Float32, Float64, Union{})),
    ),
    T in types

    @eval function _is_primitive(
        ::Type{DefaultCtx}, ::Type{<:$mode}, ::Type{S}
    ) where {
        S<:Tuple{
            typeof(LinearAlgebra._kron!),
            AbstractMatrix{$T},
            AbstractMatrix{$T},
            AbstractMatrix{$T},
        },
    }
        return _kron_arrayifiable($mode, fieldtype(S, 2)) &&
               _kron_arrayifiable($mode, fieldtype(S, 3)) &&
               _kron_arrayifiable($mode, fieldtype(S, 4))
    end
end
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

# Direct block access avoids arrayify; the primal runs once.
function Mooncake.frule!!(
    ::Lifted{typeof(LinearAlgebra._kron!),N},
    out::Lifted{Aout,N,<:NDualArray{T,N,2,Aout}},
    x1::Lifted{A1,N,<:NDualArray{T,N,2,A1}},
    x2::Lifted{A2,N,<:NDualArray{T,N,2,A2}},
) where {N,T<:IEEEFloat,Aout<:AbstractMatrix{T},A1<:AbstractMatrix{T},A2<:AbstractMatrix{T}}
    _check_kron_destination(primal(out))
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

# Struct-lift wrappers need arrayify; dense inputs take the method above.
function Mooncake.frule!!(
    ::Lifted{typeof(LinearAlgebra._kron!),N},
    out::Lifted{<:AbstractMatrix{T},N},
    x1::Lifted{<:AbstractMatrix{T},N},
    x2::Lifted{<:AbstractMatrix{T},N},
) where {N,T<:BlasFloat}
    _check_kron_destination(primal(out))
    if _storage_kind(T) isa StructuralStorage
        T === Union{} || _throw_nonconcrete_primitive(T)
        LinearAlgebra._kron!(primal(out), primal(x1), primal(x2))
        return out
    end
    pout, dout_s = arrayify(out)
    px1, dx1_s = arrayify(x1, Val(:read))
    px2, dx2_s = arrayify(x2, Val(:read))
    LinearAlgebra._kron!(pout, px1, px2)
    for lane in 1:N
        _kron!_jvp_lane!(dout_s[lane], px1, dx1_s[lane], px2, dx2_s[lane])
    end
    return out
end
function Mooncake.rrule!!(
    ::CoDual{typeof(LinearAlgebra._kron!)},
    out::CoDual{<:AbstractMatrix{<:T}},
    x1::CoDual{<:AbstractVecOrMat{<:T}},
    x2::CoDual{<:AbstractVecOrMat{<:T}},
) where {T<:Base.IEEEFloat}
    _check_kron_destination(primal(out))
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
# can return structured output and must stay derived.
for (mode, types) in (
        (ForwardMode, (Float32, Float64, Union{})),
        (ReverseMode, (Float16, Float32, Float64, Union{})),
    ),
    T in types

    @eval function _is_primitive(
        ::Type{DefaultCtx}, ::Type{<:$mode}, ::Type{S}
    ) where {S<:Tuple{typeof(kron),AbstractMatrix{$T},AbstractMatrix{$T}}}
        A, B = fieldtype(S, 2), fieldtype(S, 3)
        return (A <: StridedMatrix || B <: StridedMatrix) &&
               _kron_arrayifiable($mode, A) &&
               _kron_arrayifiable($mode, B)
    end
end
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
for Wrapper in (Symmetric, Hermitian)
    @eval @inline _kron_densify(z::$Wrapper) = convert(
        Matrix, $Wrapper(Matrix(parent(z)), Symbol(z.uplo))
    )
end
function Mooncake.frule!!(
    ::Lifted{typeof(kron),N},
    x1::Lifted{<:AbstractVecOrMat{T},N},
    x2::Lifted{<:AbstractVecOrMat{T},N},
) where {N,T<:Union{Float32,Float64}}
    if _storage_kind(T) isa StructuralStorage
        T === Union{} || _throw_nonconcrete_primitive(T)
        return zero_lifted(Val(N), kron(primal(x1), primal(x2)))
    end
    px1, dx1s = arrayify(x1, Val(:read))
    px2, dx2s = arrayify(x2, Val(:read))
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
            bp, k, Val(N), mx1, _kron_densify(dx1s[k]), mx2, _kron_densify(dx2s[k])
        )
    end
    V = NDualArray{T,N,2,A,Nfwd._wrapped_eltype(T, Val(N)),typeof(blk)}(y, blk)
    return Lifted{A,N}(y, V)
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    sum_sizes = [(11,), (11, 3)]
    precisions = [Float64, Float32, Float16]
    hvcat_throws = map(((), (-1,))) do rows
        try
            Base.hvcat(rows, 1.0)
        catch e
            return (typeof(e), sprint(showerror, e))
        end
        error("Expected Base.hvcat($rows, 1.0) to throw")
    end
    test_cases = vcat(
        map([Float32, Float64]) do T
            TestCase(hvcat, (2, 2), T.(1:4)...; mode=ForwardMode)
        end,
        map([Float32, Float64]) do T
            TestCase(hvcat, (3, 3, 3), T.(1:9)...; mode=ForwardMode)
        end,
        map([Float32, Float64]) do T
            TestCase(hvcat, (1,), T(1); mode=ForwardMode)
        end,
        map([Float32, Float64]) do T
            TestCase(hvcat, (4, 4, 4, 4), T.(1:16)...; mode=ForwardMode)
        end,
        [
            TestCase(
                hvcat,
                (2, 1, 1),
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0;
                mode=ForwardMode,
                throws=DimensionMismatch,
                primal_throws=true,
            ),
            TestCase(
                hvcat,
                (2, 2),
                1.0,
                2.0,
                3.0;
                mode=ForwardMode,
                throws=ArgumentError,
                primal_throws=true,
            ),
            TestCase(
                hvcat,
                (3, 3, 3),
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0;
                mode=ForwardMode,
                throws=ArgumentError,
                primal_throws=true,
            ),
            TestCase(
                hvcat, (), 1.0; mode=ForwardMode, throws=hvcat_throws[1], primal_throws=true
            ),
            TestCase(
                hvcat,
                (-1,),
                1.0;
                mode=ForwardMode,
                throws=hvcat_throws[2],
                primal_throws=true,
            ),
        ],

        # sum(x)
        map_prod(sum_sizes, precisions) do (sz, P)
            flags = (;
                interface_only=P == Float16 ? true : false, perf_flag=:stability_and_allocs
            )
            return TestCase(sum, randn(rng, P, sz...); flags...)
        end,

        # Forward-only primitives. Finite differences cannot resolve ties, so pinned
        # width-1 tangents credit the LAST maximum and FIRST minimum (skip_chunked).
        map((maximum, minimum)) do f
            vcat(
                map(precisions) do P
                    flags = (;
                        interface_only=P == Float16,
                        perf_flag=:stability_and_allocs,
                        mode=ForwardMode,
                    )
                    return TestCase(f, randn(rng, P, 11); flags...)
                end,
                map([Float64, Float32]) do P
                    value, deriv = f === maximum ? (3, 40) : (1, 20)
                    opts = (
                        mode=ForwardMode,
                        oracle=(value=P(value), deriv=P(deriv)),
                        skip_chunked=true,
                    )
                    values = f === maximum ? P[1, 3, 2, 3] : P[3, 1, 2, 1]
                    x = CoDual(values, P[10, 20, 30, 40])
                    return TestCase(f, x; opts...)
                end,
            )
        end...,

        # sum(view(x, a:b))
        map(precisions) do P
            flags = (;
                interface_only=P == Float16 ? true : false, perf_flag=:stability_and_allocs
            )
            return TestCase(sum, view(randn(rng, P, 11), 2:9); flags...)
        end,

        # sum(abs2, x)
        map_prod(vcat(sum_sizes, [(0,), (0, 3)]), precisions) do (sz, P)
            flags = (;
                interface_only=P == Float16 ? true : false, perf_flag=:stability_and_allocs
            )
            return TestCase(sum, abs2, randn(rng, P, sz...); flags...)
        end,

        # _kron!(x, y). `interface_only` for `Float16` alone, as the `sum` test cases above: finite
        # differences are hopeless at that precision and meaningful at the others.
        map(precisions) do (P)
            return TestCase(
                LinearAlgebra._kron!,
                zeros(P, 50, 50),
                randn(rng, P, 5, 5),
                randn(rng, P, 10, 10);
                interface_only=P == Float16,
            )
        end,

        # SubArray exercises the wrapped method, a forward primitive only for BlasFloat.
        map([Float64, Float32]) do P
            return TestCase(
                LinearAlgebra._kron!,
                zeros(P, 50, 50),
                view(randn(rng, P, 6, 6), 1:5, 1:5),
                randn(rng, P, 10, 10),
            )
        end,
        [
            TestCase(
                kron,
                Matrix{Union{}}(undef, 0, 0),
                Matrix{Union{}}(undef, 0, 0);
                interface_only=true,
                perf_flag=:stability,
                mode=ForwardMode,
            ),
        ],
        [
            TestCase(
                LinearAlgebra._kron!,
                Matrix{Union{}}(undef, 0, 0),
                Matrix{Union{}}(undef, 0, 0),
                Matrix{Union{}}(undef, 0, 0);
                perf_flag=:stability,
                mode=ForwardMode,
            ),
        ],
        [
            TestCase(
                permutedims,
                Matrix{Union{}}(undef, 0, 0);
                perf_flag=:stability,
                mode=ForwardMode,
            ),
        ],
        [
            TestCase(
                permutedims,
                Union{Float32,Float64}[2.0f0 3.0];
                mode=ForwardMode,
                throws=(ArgumentError, "for top-level real Arrays"),
            ),
            TestCase(
                kron,
                Union{Float32,Float64}[2.0f0 3.0],
                Union{Float32,Float64}[2.0f0 3.0];
                mode=ForwardMode,
                throws=(ArgumentError, "for top-level real Arrays"),
            ),
        ],
        # Allocating kron has its own block writer: check derivatives at widths 1 and 8.
        # Only Float32/Float64 are forward primitives; Float16 stays derived.
        map([Float64, Float32]) do P
            return TestCase(kron, randn(rng, P, 5, 4), randn(rng, P, 3, 6))
        end,
        # Wrapped operands take the `arrayify`/`convert` path into the same lane writer.
        map([Float64, Float32]) do P
            return TestCase(
                kron,
                view(randn(rng, P, 6, 6), 1:5, 1:4),
                UpperTriangular(randn(rng, P, 3, 3)),
            )
        end,
        # Symmetric/Hermitian lanes need densification before LAPACK; reverse folds
        # onto the stored triangle. Both-Symmetric stays derived and is tested below.
        map((
            (Symmetric, (3, 3), identity, (4, 2)),
            (identity, (3, 4), x -> Symmetric(x, :L), (3, 3)),
            (Hermitian, (3, 3), identity, (4, 2)),
            (identity, (3, 4), x -> Hermitian(x, :L), (3, 3)),
        )) do (wrap1, sz1, wrap2, sz2)
            map([Float64, Float32]) do P
                return TestCase(
                    kron, wrap1(randn(rng, P, sz1...)), wrap2(randn(rng, P, sz2...))
                )
            end
        end...,
        map_prod(
            (
                a -> view(Symmetric(a), :, :),
                a -> UpperTriangular(transpose(a)),
                a -> reshape(Symmetric(a), 1, 4),
                a -> Adjoint(Symmetric(a)),
                a -> Transpose(Hermitian(a)),
                a -> Diagonal(view(vec(a), [1, 2])),
            ),
            (false, true),
            (kron, LinearAlgebra._kron!),
        ) do (wrap, swap, f)
            a, b = wrap([1.0 2.0; 3.0 4.0]), ones(1, 1)
            x, y = swap ? (b, a) : (a, b)
            args = f === kron ? (x, y) : (zeros(size(a)), x, y)
            return TestCase(f, args...)
        end,
        map([
            a -> Diagonal(vec(a)), Symmetric, Hermitian, UpperTriangular, LowerTriangular
        ]) do wrap
            return TestCase(
                LinearAlgebra._kron!, wrap(fill(2.0, 1, 1)), ones(1, 1), ones(1, 1)
            )
        end,
        map([UnitUpperTriangular, UnitLowerTriangular]) do wrap
            i, j = wrap === UnitUpperTriangular ? (1, 2) : (2, 1)
            out = view(wrap(ones(2, 2)), i:i, j:j)
            return TestCase(LinearAlgebra._kron!, out, ones(1, 1), ones(1, 1))
        end,
        map_prod(
            (
                view(zeros(2, 1), [1, 1], :),
                view(reinterpret(reshape, Float64, ComplexF64[0]), [1, 1], :),
                view(zeros(1, 2), :, [1, 1]),
                transpose(view(zeros(2, 1), [1, 1], :)),
                reshape(view(zeros(2, 1), [1, 1], :), 1, 2),
                reinterpret(Float64, view(zeros(ComplexF64, 1, 2), :, [1, 1])),
                view(transpose(view(zeros(2, 1), [1, 1], :)), :, :),
            ),
            (nothing, ForwardMode),
        ) do (out, mode)
            return TestCase(
                LinearAlgebra._kron!,
                out,
                reshape(collect(2.0:(length(out) + 1)), size(out)),
                ones(1, 1);
                mode,
                chunk_size=mode === ForwardMode ? 8 : 1,
                throws=(
                    ArgumentError,
                    "_kron! refuses destinations with repeated indices in any wrapper, even if the outer selection is injective",
                ),
            )
        end,
        map([a -> view(a, [1, 2], :), transpose]) do wrap
            return TestCase(
                LinearAlgebra._kron!, wrap(zeros(2, 2)), [1.0 2.0; 3.0 4.0], ones(1, 1)
            )
        end,
        [
            TestCase(
                LinearAlgebra._kron!,
                zeros(ComplexF64, 2, 2),
                view(Symmetric(ComplexF64[1 2; 3 4]), :, :),
                ones(ComplexF64, 1, 1);
                mode=ForwardMode,
            ),
        ],
        [
            TestCase(
                kron,
                reinterpret(Float64, Symmetric(ComplexF64[1+2im 3+4im; 5+6im 7+8im])),
                ones(1, 1);
                unsafe_perturb=true,
            ),
        ],
        map([kron, LinearAlgebra._kron!]) do f
            a = reshape(
                reinterpret(
                    Float64, reinterpret(reshape, ComplexF64, reshape([1.0, 2.0], 2, 1))
                ),
                1,
                2,
            )
            args = f === kron ? (a, ones(1, 1)) : (zeros(size(a)), a, ones(1, 1))
            return TestCase(f, args...; unsafe_perturb=true)
        end,
        map_prod((1, 2), (kron, LinearAlgebra._kron!)) do (n, f)
            a = view(reinterpret(reshape, Float64, ComplexF64.(1:n)), :, :)
            args = f === kron ? (a, ones(1, 1)) : (zeros(size(a)), a, ones(1, 1))
            return TestCase(f, args...; unsafe_perturb=true)
        end,
        map([1, 2]) do n
            a = view(reinterpret(reshape, Float64, ComplexF64.(1:n)), :, :)
            return TestCase(
                LinearAlgebra._kron!, a, ones(size(a)), ones(1, 1); unsafe_perturb=true
            )
        end,
        map([
            reinterpret(reshape, Float64, ComplexF64[1, 2]),
            reinterpret(Float64, view(ComplexF64[1 2], [1], :)),
        ]) do out
            return TestCase(
                LinearAlgebra._kron!, out, ones(size(out)), ones(1, 1); unsafe_perturb=true
            )
        end,

        # permutedims(x). Float16 finite differences resolve only near zero, and the rule is
        # linear, so the point does not matter.
        map([Float64, Float32, Float16]) do P
            x = P == Float16 ? zeros(P, 7, 11) : randn(rng, P, 7, 11)
            return TestCase(permutedims, x; perf_flag=:stability)
        end,

        # x * y
        map_prod([(7, 11, 5), (1, 7, 1), (7, 1, 5)], [Float64, Float32]) do ((m, p, n), P)
            return TestCase(
                *, randn(rng, P, m, p), randn(rng, P, p, n); perf_flag=:stability
            )
        end,
    )
    # Pin a cached V whose primal grew outside AD: a `throws` row accepts Lifted seeds, so
    # the mismatch survives construction of the tested primitive rule.
    for N in (1, 8)
        x = zero_lifted(Val(N), [3.0])
        push!(primal(x), 7.0)
        opts = (
            mode=ForwardMode,
            chunk_size=N,
            throws=(ArgumentError, "derivative storage covers"),
        )
        push!(test_cases, TestCase(sum, x; opts...))
        push!(test_cases, TestCase(sum, abs2, x; opts...))
    end
    memory = Any[]
    for (f, x, name, direction, reference, cmp) in (
        (
            sum,
            [1.0, 2.0, 3.0],
            "primitive sum",
            [1.0, 0.0, 0.0],
            (value=6.0, gradient=ones(3), hvp=zeros(3)),
            isapprox,
        ),
        (
            sum,
            Float64[],
            "empty sum Hessian",
            Float64[],
            (value=0.0, gradient=Float64[], hessian=zeros(0, 0)),
            (==),
        ),
        (
            x -> sum(abs2, x),
            [2.0, 3.0, 4.0],
            "fused sum abs2 HVP",
            [0.0, 1.0, 0.0],
            (hvp=[0.0, 2.0, 0.0],),
            isapprox,
        ),
    )
        push!(
            test_cases,
            TestCase(
                f,
                x;
                name,
                hvp=(; check=:reference, directions=(direction,), reference, cmp),
            ),
        )
    end

    return test_cases, memory
end

_square_matmul(x) = x * x

function derived_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    precisions = [Float64, Float32]
    test_cases = vcat(
        map_prod(
            (
                Bidiagonal([2.0, 3.0], [4.0], :U),
                Tridiagonal([1.0], [2.0, 3.0], [4.0]),
                SymTridiagonal([2.0, 3.0], [4.0]),
            ),
            (false, true),
            (kron, kron!),
        ) do (a, swap, f)
            b = ones(1, 1)
            x, y = swap ? (b, a) : (a, b)
            args = f === kron ? (x, y) : (zeros(2, 2), x, y)
            return TestCase(f, args...)
        end,
        # Both-Symmetric stays derived and leaves the unstored result triangle undef.
        # Float32 needs interface_only: finite differences cannot resolve this composite;
        # Float64 checks its derivative.
        map([Float64, Float32]) do P
            return TestCase(
                LinearAlgebra.kron,
                Symmetric(randn(rng, P, 3, 3)),
                Symmetric(randn(rng, P, 3, 3), :L);
                interface_only=P == Float32,
            )
        end,

        # Exercise the hvcat primitive inside a derived function.
        TestCase((a, b, c, d) -> hvcat((2, 2), a, b, c, d), 1.0, 2.0, 3.0, 4.0),
        # Check correctness for mixed Float32/Float64 literals.
        TestCase(hvcat, (2, 2), 1.0f0, 2.0, 3.0f0, 4.0),

        # Complex Hermitian: forward wraps lanes, reverse stays derived because folding
        # onto the stored triangle needs conjugation. Check both against finite differences.
        map([ComplexF64, ComplexF32]) do C
            return map([:U, :L]) do uplo
                return TestCase(
                    LinearAlgebra.kron,
                    Hermitian(randn(rng, C, 3, 3), uplo),
                    randn(rng, C, 4, 2),
                )
            end
        end...,
        map((
            (identity, UpperTriangular),
            (identity, LowerTriangular),
            (UpperTriangular, LowerTriangular),
            (identity, UnitUpperTriangular),
            (identity, UnitLowerTriangular),
            (UnitUpperTriangular, UnitLowerTriangular),
            (x -> view(x, 1:5, 1:5), LowerTriangular),
            (x -> view(x, 1:5, 1:5), UpperTriangular),
        )) do (wrap1, wrap2)
            map(precisions) do P
                return TestCase(
                    LinearAlgebra.kron,
                    wrap1(randn(rng, P, 5, 5)),
                    wrap2(randn(rng, P, 10, 10)),
                )
            end
        end...,
        # Diagonal operand: the reverse pullback must fold only the diagonal of the dense
        # gradient into the `Diagonal` fdata (off-diagonal are structural zeros, dropped).
        map(precisions) do (P)
            return TestCase(
                LinearAlgebra.kron, Diagonal(randn(rng, P, 4)), randn(rng, P, 3, 3)
            )
        end,
        map(precisions) do (P)
            return TestCase(
                LinearAlgebra.kron, randn(rng, P, 4, 4), Diagonal(randn(rng, P, 3))
            )
        end,

        # `A * A` aliases the rule's arguments, so `dA === dB` and the pullback must
        # accumulate both terms into the one array.
        map(precisions) do (P)
            return TestCase(_square_matmul, randn(rng, P, 5, 5))
        end,
    )
    memory = Any[]
    return test_cases, memory
end
