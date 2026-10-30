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

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
@is_primitive(DefaultCtx, Tuple{typeof(sum),Array{<:IEEEFloat}})
function frule!!(::Dual{typeof(sum)}, x::Dual{<:Array{P}}) where {P<:IEEEFloat}
    return Dual(sum(primal(x)), sum(tangent(x)))
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
function frule!!(::Dual{typeof(sum)}, x::Dual{ContiguousSubVector{P}}) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    return Dual(sum(px), sum(dx))
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
    ::Dual{typeof(sum)}, ::Dual{typeof(abs2)}, x::Dual{<:Array{P}}
) where {P<:IEEEFloat}
    return Dual(sum(abs2, primal(x)), 2 * dot(primal(x), tangent(x)))
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
    ::Dual{typeof(*)}, A::Dual{<:Matrix{P}}, B::Dual{<:Matrix{P}}
) where {P<:BlasRealFloat}
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    C = pA * pB
    dC = dA * pB
    mul!(dC, pA, dB, one(P), one(P))
    return Dual(C, dC)
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

function _check_mul_alias(C, A, B)
    # Unwrap Transpose/Adjoint: mightalias otherwise rejects disjoint slices of one array.
    C isa Union{Transpose,Adjoint} && return _check_mul_alias(parent(C), A, B)
    A isa Union{Transpose,Adjoint} && return _check_mul_alias(C, parent(A), B)
    B isa Union{Transpose,Adjoint} && return _check_mul_alias(C, A, parent(B))
    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    return nothing
end

# `StridedArray` admits `AbstractGPUArray <: DenseArray`; the scalar loops and `arrayify`
# need `Array` or `Memory` storage, so other dense arrays fall back to stdlib `mul!`.
const _CPUDenseArray = @static VERSION < v"1.11-" ? Array : Union{Array,Memory}
const _CPUContiguousArray = Union{
    _CPUDenseArray,Base.StridedFastContiguousSubArray{T,N,<:_CPUDenseArray} where {T,N}
}
const _CPUReinterpretArray =
    Base.StridedReinterpretArray{T,N,<:_CPUContiguousArray} where {T,N}
const _CPUReshapedArray = Base.StridedReshapedArray{
    T,N,<:Union{_CPUContiguousArray,_CPUReinterpretArray}
} where {T,N}
const _CPUStridedArray{T,N} = Union{
    Array{T,N},
    Base.StridedSubArray{
        T,N,<:Union{_CPUDenseArray,_CPUReinterpretArray,_CPUReshapedArray}
    },
    _CPUReinterpretArray{T,N},
    _CPUReshapedArray{T,N},
}

const _CPUStridedOrAdjTransMatrix{T} = Union{
    _CPUStridedArray{T,2},
    Adjoint{T,<:_CPUStridedArray{T,2}},
    Transpose{T,<:_CPUStridedArray{T,2}},
    Adjoint{T,<:Transpose{T,<:_CPUStridedArray{T,2}}},
    Transpose{T,<:Adjoint{T,<:_CPUStridedArray{T,2}}},
}
@static if VERSION < v"1.11-"
    const _CPUStridedOrAdjTransVecOrMat{T} = Union{
        _CPUStridedArray{T,1},_CPUStridedOrAdjTransMatrix{T}
    }
else
    const _CPUStridedOrAdjTransVecOrMat{T} = Union{
        Memory{T},_CPUStridedArray{T,1},_CPUStridedOrAdjTransMatrix{T}
    }
end

# Finite zero products must retain their derivatives in forward-over-reverse AD.
@inline function _mul_guarded_scale(d, x)
    return iszero(d) && !isfinite(x) ? zero(d*x) : d*x
end

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/1396
# Stdlib `mul!` skips A·B when α = 0; its generic kernels skip α at one and C at β = 0.
# Exception to convention 2: strided coefficient correctness requires an open signature;
# every named element type and wrapper kind is tested.
# The default prepare_derivative_cache nfwd path (enable_nfwd=true) can silently
# lose coefficient derivatives; #1340 removes that path, which fixes this.
# Structured, vector-left or mixed-type operands, wrapped-vector inputs/outputs,
# scalar-array mul! and stdlib internals can lose coefficient derivatives.
const _MulSignature = Tuple{
    typeof(mul!),
    _CPUStridedOrAdjTransVecOrMat{T},
    _CPUStridedOrAdjTransMatrix{T},
    _CPUStridedOrAdjTransVecOrMat{T},
    Union{Bool,IEEEFloat,T},
    Union{Bool,IEEEFloat,T},
} where {T<:Union{BlasFloat,HalfFloat}}

@is_primitive MinimalCtx _MulSignature

function may_match_primitive(
    ::Type{<:Union{MinimalCtx,DefaultCtx}},
    ::Type{<:Mode},
    @nospecialize(S::Type{<:Tuple{typeof(mul!),Any,Any,Any,Any,Any}}),
)
    return typeintersect(S, _MulSignature) !== Union{}
end

@inline function frule!!(
    ::Dual{typeof(mul!)},
    c::Dual{<:_CPUStridedOrAdjTransVecOrMat{T}},
    a::Dual{<:_CPUStridedOrAdjTransMatrix{T}},
    b::Dual{<:_CPUStridedOrAdjTransVecOrMat{T}},
    alpha::Dual{<:Union{Bool,IEEEFloat,T}},
    beta::Dual{<:Union{Bool,IEEEFloat,T}},
) where {T<:Union{BlasFloat,HalfFloat}}
    C, dC = arrayify(c)
    A, dA = arrayify(a)
    B, dB = arrayify(b)
    _check_mul_alias(C, A, B)
    dC = B isa AbstractVector && dC isa AbstractMatrix ? view(dC, :, 1) : dC
    pa, pb = primal(alpha), primal(beta)
    da = pa isa Bool ? zero(T) : tangent(alpha)
    db = pb isa Bool ? zero(T) : tangent(beta)
    mul!(dC, dA, B, pa, pb)
    mul!(dC, A, dB, pa, one(T))
    iszero(da) || mul!(dC, A, B, da, one(T))
    if !iszero(db)
        @inbounds @simd for i in eachindex(IndexLinear(), C, dC)
            dC[i] += db * C[i]
        end
    end
    mul!(C, A, B, pa, pb)
    return c
end

@inline function rrule!!(
    ::CoDual{typeof(mul!)},
    c::CoDual{<:_CPUStridedOrAdjTransVecOrMat{T}},
    a::CoDual{<:_CPUStridedOrAdjTransMatrix{T}},
    b::CoDual{<:_CPUStridedOrAdjTransVecOrMat{T}},
    alpha::CoDual{S},
    beta::CoDual{U},
) where {
    T<:Union{BlasFloat,HalfFloat},S<:Union{Bool,IEEEFloat,T},U<:Union{Bool,IEEEFloat,T}
}
    C, dC = arrayify(c)
    A, dA = arrayify(a)
    B, dB = arrayify(b)
    _check_mul_alias(C, A, B)
    # Small outputs must not allocate a separate restoration buffer.
    # Some avoids a union-typed NamedTuple field, whose tangent type is Any.
    old = (
        small=ntuple(i -> i <= length(C) && length(C) <= 16 ? C[i] : zero(T), Val(16)),
        large=Some{Union{Nothing,Array{T,ndims(C)}}}(length(C) <= 16 ? nothing : copy(C)),
    )
    pa, pb = primal(alpha), primal(beta)
    mul!(C, A, B, pa, pb)
    function pullback(::NoRData)
        # Wider coefficients must not round through an input cotangent product.
        da = pa isa Bool ? NoRData() : zero(T <: HalfFloat ? Float32 : T) + zero(pa)
        from_output = !(pa isa Bool) && isone(pa) && iszero(pb)
        reuse_A = !from_output && da isa T && !iszero(pa) && size(C) == size(A)
        reuse_B = !from_output && da isa T && !iszero(pa) && !reuse_A && size(C) == size(B)
        if !(pa isa Bool) && !(reuse_A || reuse_B)
            dest = B isa AbstractVector && C isa AbstractMatrix ? view(C, :, 1) : C
            from_output || mul!(dest, A, B, one(T), zero(T))
            @inbounds @simd for i in eachindex(C)
                da += _mul_guarded_scale(oftype(da, dC[i]), conj(oftype(da, C[i])))
            end
        end
        # The transposed left factor avoids Julia 1.10's tiled outer-product buffers.
        left, right = if VERSION < v"1.11-" && T <: Float16 && B isa AbstractVector
            transpose(Base.ReshapedArray(dC, (1, length(dC)), ())),
            Base.ReshapedArray(B, (1, length(B)), ())
        else
            dC, adjoint(B)
        end
        # Keep zero coefficients in mul! calls so nested AD retains their directions.
        if reuse_A
            mul!(C, left, right, one(T), zero(T))
            @inbounds @simd for i in eachindex(C)
                v = C[i]
                dA[i] += conj(pa) * v
                da += _mul_guarded_scale(v, conj(A[i]))
            end
        else
            mul!(dA, left, right, conj(pa), one(T))
        end
        if reuse_B
            mul!(C, adjoint(A), dC, one(T), zero(T))
            @inbounds @simd for i in eachindex(C)
                v = C[i]
                dB[i] += conj(pa) * v
                da += _mul_guarded_scale(v, conj(B[i]))
            end
        elseif B isa AbstractVector && dC isa AbstractMatrix
            mul!(dB, adjoint(A), view(dC, :, 1), conj(pa), one(T))
        else
            mul!(dB, adjoint(A), dC, conj(pa), one(T))
        end
        large = something(old.large)
        db = zero(eltype(C) <: HalfFloat ? Float32 : eltype(C)) + zero(pb)
        @inbounds @simd for i in 1:length(C)
            v = (large === nothing ? old.small : large)[i]
            if !(pb isa Bool)
                db += _mul_guarded_scale(oftype(db, dC[i]), conj(oftype(db, v)))
            end
            dC[i] = if !isfinite(pb) && iszero(dC[i])
                zero(dC[i])
            else
                _mul_guarded_scale(conj(pb), dC[i])
            end
            C[i] = v
        end
        db = pb isa Bool ? NoRData() : db
        # Keep α and β in the correction so nested AD sees the unscaled product.
        if from_output
            da =
                (da - (pb isa Bool ? zero(da) : _mul_guarded_scale(conj(pb), db))) /
                conj(pa)
        end
        da = pa isa Bool ? da : S(S <: Real ? real(da) : da)
        db = pb isa Bool ? db : U(U <: Real ? real(db) : db)
        return NoRData(), NoRData(), NoRData(), NoRData(), da, db
    end
    return c, pullback
end

# Differentiating the generic implementation element by element costs more than the rules
# either side of it; `Distances.pairwise(...; dims=1)` reaches it on every call.
@is_primitive DefaultCtx Tuple{typeof(permutedims),Matrix{P}} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(permutedims)}, x::Dual{<:Matrix{P}}) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    return Dual(permutedims(px), permutedims(dx))
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

# Matrices only: `kron(::Vector, ::Vector)` returns a `Vector`, which these rules would
# widen to a `Matrix` by matrixifying their factors.
const KronFactor{T} = Union{
    StridedMatrix{T},
    UpperTriangular{T,<:StridedMatrix{T}},
    LowerTriangular{T,<:StridedMatrix{T}},
}

# Both `kron` pullbacks contract `dy` against one factor to accumulate into the other. The
# contraction is dense, so it goes through `densify_tangent` /
# `increment_densified_tangent!!`. Read as `P x M x Q x N`, both contractions read the same
# element of `dy`, so a single pass in memory order serves both. A `gemv` per `(q, n)` block
# is slower at every shape measured: the blocks are small enough that BLAS call overhead
# dominates.
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
@is_primitive DefaultCtx Tuple{
    typeof(LinearAlgebra._kron!),StridedMatrix{T},KronFactor{T},KronFactor{T}
} where {T<:IEEEFloat}
function Mooncake.frule!!(
    ::Dual{typeof(LinearAlgebra._kron!)},
    out::Dual{<:StridedMatrix{<:T}},
    x1::Dual{<:KronFactor{<:T}},
    x2::Dual{<:KronFactor{<:T}},
) where {T<:Base.IEEEFloat}
    pout, dout = arrayify(out)
    px1, dx1 = arrayify(x1)
    px2, dx2 = arrayify(x2)
    LinearAlgebra._kron!(pout, px1, px2)
    # manually compute dout .= kron(dx1, px2) .+ kron(px1, dx2), otherwise performance
    # suffers
    m = firstindex(dout)
    for j in axes(px1, 2), l in axes(px2, 2), i in axes(px1, 1)
        x1ij = px1[i, j]
        dx1ij = dx1[i, j]
        for k in axes(px2, 1)
            dout[m] = (x1ij * dx2[k, l]) + (dx1ij * px2[k, l])
            m += 1
        end
    end
    return out
end
function Mooncake.rrule!!(
    ::CoDual{typeof(LinearAlgebra._kron!)},
    out::CoDual{<:StridedMatrix{<:T}},
    x1::CoDual{<:KronFactor{<:T}},
    x2::CoDual{<:KronFactor{<:T}},
) where {T<:Base.IEEEFloat}
    pout, dout = arrayify(out)
    px1, dx1 = arrayify(x1)
    px2, dx2 = arrayify(x2)
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
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(kron),KronFactor{T},KronFactor{T}
} where {T<:IEEEFloat}
function Mooncake.rrule!!(
    ::CoDual{typeof(kron)}, x1::CoDual{<:KronFactor{<:T}}, x2::CoDual{<:KronFactor{<:T}}
) where {T<:Base.IEEEFloat}
    px1, dx1 = arrayify(x1)
    px2, dx2 = arrayify(x2)
    y = kron(px1, px2)
    dy = zero(y)
    function kron_pb!!(::NoRData)
        _kron_pb!(dx1, dx2, dy, px1, px2)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), kron_pb!!
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    sum_sizes = [(11,), (11, 3)]
    precisions = [Float64, Float32, Float16]
    test_cases = vcat(
        Any[],
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
    P = Float16
    push!(
        test_cases,
        (
            false,
            :none,
            nothing,
            mul!,
            ones(P, 2, 2),
            zeros(P, 2, 2),
            zeros(P, 2, 2),
            true,
            zero(P),
        ),
    )
    A = P[0.2 0.3; 0.4 0.5]
    B = P[0.3 0.4; 0.5 0.6]
    append!(
        test_cases,
        [
            (true, :none, nothing, mul!, zeros(P, 2, 2), A, A, true, false),
            (true, :none, nothing, mul!, ones(P, 2, 2), A, B, false, true),
            (
                true,
                :none,
                nothing,
                mul!,
                zeros(P, 2, 2),
                transpose(A),
                adjoint(B),
                true,
                false,
            ),
            (true, :none, nothing, mul!, ones(P, 2, 2), A, B, P(0.5), zero(P)),
            (true, :none, nothing, mul!, zeros(P, 2, 2), A, B, P(0.5), zero(P)),
            (true, :none, nothing, mul!, ones(P, 2, 2), A, B, true, P(0.25)),
            (true, :none, nothing, mul!, ones(P, 2, 2), A, B, P(0.5), true),
        ],
    )
    for (wrap_a, wrap_b) in
        ((identity, transpose), (transpose, adjoint), (adjoint, identity))
        X = wrap_a(view(reshape(copy(A), 2, 2, 1),:,:,1))
        Y = wrap_b(view(reshape(copy(B), 2, 2, 1),:,:,1))
        C = view(fill(P(0.1), 2, 2, 1),:,:,1)
        push!(test_cases, (true, :none, nothing, mul!, C, X, Y, P(0.5), P(0.25)))
    end
    memory = Any[]
    return test_cases, memory
end

_square_matmul(x) = x * x

function derived_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    precisions = [Float64, Float32]
    test_cases = vcat(
        Any[],
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
        # `A * A` aliases the rule's arguments, so `dA === dB` and the pullback must
        # accumulate both terms into the one array.
        map(precisions) do (P)
            return (false, :none, nothing, _square_matmul, randn(rng, P, 5, 5))
        end,
    )
    push!(
        test_cases,
        (false, :none, nothing, _mul_guarded_scale, 0.0, 2.0),
        (false, :none, nothing, _mul_guarded_scale, 0.0+0.0im, 2.0+1.0im),
    )
    for T in (Float64, ComplexF64),
        (a, b) in ((one(T), zero(T)), (zero(T), zero(T)), (true, zero(T)))

        n = iszero(a) ? 1 : 2
        A, B = fill(T(0.25), n, n), fill(T(0.5), n, n)
        T <: Complex && (A .+= T(0.2im); B .-= T(0.3im))
        push!(test_cases, (false, :none, nothing, mul!, ones(T, n, n), A, B, a, b))
    end
    let A = Float32[0.25 0.75; 0.5 1.0], B = reverse(A; dims=1)
        push!(
            test_cases, (false, :none, nothing, mul!, ones(Float32, 2, 2), A, B, 1.0, 0.0)
        )
    end
    for (wrap, n, a, b) in (
        (adjoint, 1, 1.0+0.0im, 0.0+0.0im),
        (transpose, 1, 1.0+0.0im, 0.0+0.0im),
        (adjoint, 17, 0.7+0.2im, -0.2+0.3im),
    )
        T = ComplexF64
        push!(
            test_cases,
            (
                false,
                :none,
                nothing,
                mul!,
                wrap(randn(rng, T, 1, n)),
                randn(rng, T, n, 2),
                randn(rng, T, 2, 1),
                a,
                b,
            ),
        )
    end
    for wrap in (adjoint ∘ transpose, transpose ∘ adjoint)
        T = ComplexF64
        push!(
            test_cases,
            (
                false,
                :none,
                nothing,
                mul!,
                wrap(ones(T, 1, 1)),
                fill(T(0.2+0.3im), 1, 1),
                fill(T(0.4-0.1im), 1, 1),
                one(T),
                zero(T),
            ),
        )
    end
    for wrap in (adjoint, transpose)
        T = ComplexF64
        push!(
            test_cases,
            (
                false,
                :none,
                nothing,
                mul!,
                wrap(ones(T, 1, 1)),
                randn(rng, T, 1, 2),
                randn(rng, T, 2),
                one(T),
                zero(T),
            ),
        )
    end
    # The pullback must retain α's direction in wrapped cotangent updates.
    f =
        a -> begin
            T = ComplexF64
            c, x, y = map(
                zero_fcodual,
                (
                    zeros(T, 1, 1),
                    adjoint(fill(T(0.4+0.2im), 1, 1)),
                    transpose(fill(T(1+im), 1, 1)),
                ),
            )
            _, pb = rrule!!(zero_fcodual(mul!), c, x, y, zero_fcodual(a), zero_fcodual(0.0))
            fill!(c.dx, one(T))
            pb(NoRData())
            sum(real, last(arrayify(x))) + sum(real, last(arrayify(y)))
        end
    push!(test_cases, (false, :none, nothing, f, 1.0))
    for shape in ((1,), (2,), (1, 1))
        T, a = ComplexF64, 0.7+0.1im
        m, k = first(shape), length(shape) == 1 ? 1 : 2
        A = fill(T(0.5+0.2im), m, k)
        B = fill(T(1-0.3im), k)
        push!(test_cases, (false, :none, nothing, mul!, ones(T, shape), A, B, a, zero(T)))
    end
    let T = ComplexF64
        push!(
            test_cases,
            (
                false,
                :none,
                nothing,
                mul!,
                view(ones(T, 3, 1), 1:2:3, :),
                view(fill(T(0.5+0.2im), 3, 1), 2:3, :),
                fill(T(1-0.3im), 1, 1),
                0.7+0.2im,
                -0.2+0.3im,
            ),
        )
    end
    for wrap in (transpose, adjoint)
        for (a, b) in ((1.0, 0.0), (0.7+0.2im, -0.2+0.3im))
            n = a isa Real ? 2 : 1
            A, B = fill(0.5+0.2im, n, n), fill(1-0.3im, n, n)
            push!(
                test_cases,
                (
                    false,
                    :none,
                    nothing,
                    mul!,
                    ones(ComplexF64, n, n),
                    wrap(A),
                    wrap(B),
                    a,
                    b,
                ),
            )
        end
    end
    push!(
        test_cases,
        (false, :none, nothing, mul!, ones(1, 1), zeros(1, 0), zeros(0, 1), 0.0, 0.0),
    )
    f = a -> sum(a * ones(2, 2) * ones(2, 2))
    push!(test_cases, (false, :none, nothing, f, 1.0))
    f =
        a -> real(
            sum(
                mul!(
                    fill(oftype(a, NaN), 3, 3),
                    ones(typeof(a), 3, 3),
                    ones(typeof(a), 3, 3),
                    a,
                    zero(a),
                ),
            ),
        )
    for T in (Float64, ComplexF64)
        push!(test_cases, (false, :none, nothing, f, one(T)))
    end
    # Scale β so the smallest finite-difference step survives conversion to Float16.
    for (wrap, k) in ((identity, 0), (identity, 1), (transpose, 1), (adjoint, 1))
        dims = wrap === identity ? (1, k) : (k, 1)
        f =
            b -> only(
                mul!(
                    ones(Float16, 1),
                    wrap(zeros(Float16, dims)),
                    ones(Float16, k),
                    true,
                    Float16(8b),
                ),
            )
        push!(test_cases, (false, :none, nothing, f, 0.0))
    end
    f =
        b -> sum(
            mul!(
                ones(Float16, 2, 1),
                view(zeros(Float16, 3, 1), 1:2:3, :),
                ones(Float16, 1, 1),
                true,
                Float16(8b),
            ),
        )
    push!(test_cases, (false, :none, nothing, f, 0.0))
    f =
        b -> only(
            mul!(
                ones(Float16, 1, 1),
                zeros(Float16, 1, 1),
                ones(Float16, 1),
                true,
                Float16(8b),
            ),
        )
    push!(test_cases, (false, :none, nothing, f, 0.0))
    f = a -> begin
        T = Base.inferencebarrier(Float64)::DataType
        sum(mul!(ones(T, 2, 2), ones(T, 2, 2), ones(T, 2, 2), a, zero(T)))
    end
    push!(test_cases, (false, :none, nothing, f, 0.0))
    f =
        a -> only(
            mul!(
                zeros(Float16, 1),
                reshape(view(ones(Float16, 1), 1:1), 1, 1),
                ones(Float16, 1),
                Float16(8a),
                false,
            ),
        )
    push!(test_cases, (false, :none, nothing, f, 0.0))
    f = z -> mul!(zeros(2, 1), reinterpret(reshape, Float64, z), ones(1, 1), 1.0, false)
    push!(test_cases, (false, :none, nothing, f, ComplexF64[1 + 2im]))
    f =
        z -> only(
            mul!(
                zeros(Float16, 1),
                ones(Float16, 1, 2),
                reinterpret(Float16, z),
                true,
                false,
            ),
        )
    push!(test_cases, (false, :none, nothing, f, zeros(ComplexF16, 1)))
    memory = Any[]
    return test_cases, memory
end
