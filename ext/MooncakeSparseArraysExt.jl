module MooncakeSparseArraysExt

# Rules for multiplying a SparseMatrixCSC by a dense vector or matrix. Without them Mooncake
# differentiates the generic CSC kernel of SparseArrays scalar by scalar, which is one to
# two orders of magnitude slower than the product itself.

using Base: IEEEFloat
using LinearAlgebra: LinearAlgebra, BlasFloat, mul!, dot
using SparseArrays: SparseArrays, SparseMatrixCSC, nonzeros, rowvals, nzrange
import Mooncake:
    MinimalCtx,
    @is_primitive,
    frule!!,
    rrule!!,
    Dual,
    CoDual,
    primal,
    tangent,
    NoRData,
    arrayify,
    Tangent,
    FData,
    HalfFloat

# The five-argument `mul!`; the three-argument one and `*` forward to it. As for the dense
# `mul!` rule, the three arrays share their element type (any type supported by `arrayify`),
# and the coefficients are either of that type, real, or `Bool`.
const SparseMulSig{T} = Tuple{
    typeof(mul!),
    StridedVecOrMat{T},
    SparseMatrixCSC{T},
    StridedVecOrMat{T},
    Union{Bool,IEEEFloat,T},
    Union{Bool,IEEEFloat,T},
}

@is_primitive MinimalCtx SparseMulSig{<:Union{BlasFloat,HalfFloat}}

# The tangent of a SparseMatrixCSC only carries the stored values; its structure is the one
# of the primal.
nzval_tangent(t::Tangent) = t.fields.nzval
nzval_tangent(t::FData) = t.data.nzval
function sparse_tangent(A::SparseMatrixCSC, nzval)
    SparseMatrixCSC(size(A, 1), size(A, 2), SparseArrays.getcolptr(A), rowvals(A), nzval)
end

project_coefficient(::Type{<:Real}, x) = real(x)
project_coefficient(::Type{<:Complex}, x) = x

# dA[i, j] += α (dC * B')[i, j] for every stored entry (i, j) of A.
function accumulate_outer!(dnz, A::SparseMatrixCSC, dC, B, α)
    rows = rowvals(A)
    @inbounds for j in axes(A, 2), k in nzrange(A, j)
        r = rows[k]
        acc = zero(eltype(dnz))
        for t in axes(dC, 2)
            acc += dC[r, t] * conj(B[j, t])
        end
        dnz[k] += α * acc
    end
    return dnz
end

function frule!!(
    ::Dual{typeof(mul!)},
    c::Dual{<:StridedVecOrMat{T}},
    a::Dual{<:SparseMatrixCSC{T}},
    b::Dual{<:StridedVecOrMat{T}},
    alpha::Dual{<:Union{Bool,IEEEFloat,T}},
    beta::Dual{<:Union{Bool,IEEEFloat,T}},
) where {T<:Union{BlasFloat,HalfFloat}}
    C, dC = arrayify(c)
    A = primal(a)
    dA = sparse_tangent(A, nzval_tangent(tangent(a)))
    B, dB = arrayify(b)
    pa, pb = primal(alpha), primal(beta)
    da = pa isa Bool ? zero(T) : tangent(alpha)
    db = pb isa Bool ? zero(T) : tangent(beta)

    # d(α A B + β C) = α (dA B + A dB) + dα A B + β dC + dβ C, with C the old value.
    if !iszero(db)
        dC .= pb .* dC .+ db .* C
        mul!(dC, dA, B, pa, true)
    else
        mul!(dC, dA, B, pa, pb)
    end
    mul!(dC, A, dB, pa, true)
    iszero(da) || mul!(dC, A, B, da, true)
    mul!(C, A, B, pa, pb)
    return c
end

function rrule!!(
    ::CoDual{typeof(mul!)},
    c::CoDual{<:StridedVecOrMat{T}},
    a::CoDual{<:SparseMatrixCSC{T}},
    b::CoDual{<:StridedVecOrMat{T}},
    alpha::CoDual{P},
    beta::CoDual{Q},
) where {
    T<:Union{BlasFloat,HalfFloat},P<:Union{Bool,IEEEFloat,T},Q<:Union{Bool,IEEEFloat,T}
}
    C, dC = arrayify(c)
    A = primal(a)
    dnz = nzval_tangent(tangent(a))
    B, dB = arrayify(b)
    pa, pb = primal(alpha), primal(beta)

    C_old = copy(C)
    mul!(C, A, B, pa, pb)

    function sparse_mul_pullback!!(::NoRData)
        # dα = ⟨A B, dC⟩, computed in C, which is restored below.
        da = if pa isa Bool
            NoRData()
        else
            mul!(C, A, B, true, false)
            project_coefficient(P, dot(C, dC))
        end
        db = pb isa Bool ? NoRData() : project_coefficient(Q, dot(C_old, dC))

        accumulate_outer!(dnz, A, dC, B, conj(pa))
        mul!(dB, adjoint(A), dC, conj(pa), true)

        # Cotangent of the old value of C.
        if pb isa Bool
            pb || fill!(dC, zero(eltype(dC)))
        else
            dC .*= conj(pb)
        end

        copyto!(C, C_old)
        return NoRData(), NoRData(), NoRData(), NoRData(), da, db
    end
    return c, sparse_mul_pullback!!
end

end
