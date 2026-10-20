# Helpers shared by the rule files and by extensions: getting dense arrays and their
# tangents out of `Dual`s and `CoDual`s, and a few type aliases. Rule files should depend on
# these rather than on each other.

function _trans(flag, mat)
    flag === 'T' && return transpose(mat)
    flag === 'C' && return adjoint(mat)
    flag === 'N' && return mat
    throw(error("Unrecognised flag $flag"))
end

function tri!(A, u::Char, d::Char)
    return u == 'L' ? tril!(A, d == 'U' ? -1 : 0) : triu!(A, d == 'U' ? 1 : 0)
end

const BlasRealFloat = Union{Float32,Float64}
const BlasComplexFloat = Union{ComplexF32,ComplexF64}
const FloatOrComplex = Union{IEEEFloat,Complex{<:IEEEFloat}}

# `view(x, a:b)` for an `Array` `x` of any dimensionality: a linear-index view reshapes
# its parent, and reshaping an `Array` yields an `Array`.
const ContiguousSubVector{P} = SubArray{P,1,Vector{P},Tuple{UnitRange{Int}},true}

_fields(x::Tangent) = x.fields
_fields(x::FData) = x.data

const TangentOrFData = Union{Tangent,FData}

"""
    arrayify(x::CoDual{<:AbstractArray{<:BlasFloat}})

Return the primal field of `x`, and convert its fdata into an array of the same type as the
primal. This operation is not guaranteed to be possible for all array types, but seems to be
possible for all array types of interest so far.

## Convention

Every `arrayify` overload preserves the wrapper type: the returned tangent is always wrapped
in the same concrete type as the primal (e.g. `Diagonal` → `Diagonal`, `Adjoint` → `Adjoint`,
`Symmetric` → `Symmetric`). Rules that need to write into the tangent in-place must account
for whether the wrapper supports `setindex!`; if it does not (e.g. `Symmetric`), a dedicated
helper should extract the backing store (see `_accum_sym_logdet!`).

`matrixify` and `viewify` are thin wrappers built on top of `arrayify` and share the same
convention.
"""
function arrayify(
    x::Union{Dual{A},CoDual{A}}
) where {T<:Union{IEEEFloat,BlasFloat},A<:Union{AbstractArray{T},Ptr{<:T}}}
    return arrayify(primal(x), tangent(x))
end
function arrayify(
    x::A, dx::A
) where {T<:Union{IEEEFloat,BlasFloat},A<:Union{Array{<:T},Ptr{<:T}}}
    (x, dx)
end
function arrayify(
    x::Diagonal{P,<:AbstractVector{P}}, dx::TangentOrFData
) where {P<:BlasFloat}
    _, _dx = arrayify(x.diag, _fields(dx).diag)
    return x, Diagonal(_dx)
end
function arrayify(
    x::SubArray{P,B,C,D,E}, dx::TangentOrFData
) where {P<:Union{IEEEFloat,BlasFloat},B,C,D,E}
    _, _dx = arrayify(x.parent, _fields(dx).parent)
    return x, SubArray{P,B,typeof(_dx),D,E}(_dx, x.indices, x.offset1, x.stride1)
end
function arrayify(x::ReshapedArray{P,B,C,D}, dx::TangentOrFData) where {P<:BlasFloat,B,C,D}
    _, _dx = arrayify(x.parent, _fields(dx).parent)
    return x, ReshapedArray{P,B,typeof(_dx),D}(_dx, x.dims, x.mi)
end
function arrayify(x::Base.ReinterpretArray{T}, dx::TangentOrFData) where {T<:BlasFloat}
    _, _dx = arrayify(x.parent, _fields(dx).parent)
    return x, reinterpret(T, _dx)
end
function arrayify(
    x::Tx, dx::TangentOrFData
) where {T<:IEEEFloat,Tx<:LinearAlgebra.AbstractTriangular{T}}
    _, _dx = arrayify(x.data, _fields(dx).data)
    return x, Tx(_dx)
end
function arrayify(
    x::Symmetric{T,<:StridedMatrix{T}}, dx::TangentOrFData
) where {T<:Union{IEEEFloat,BlasFloat}}
    _, _dx = arrayify(x.data, _fields(dx).data)
    return x, Symmetric(_dx, Symbol(x.uplo))
end
function arrayify(
    x::Adjoint{T,<:AbstractArray{T}}, dx::TangentOrFData
) where {T<:Union{IEEEFloat,BlasFloat}}
    _, _dx = arrayify(x.parent, _fields(dx).parent)
    return x, adjoint(_dx)
end
function arrayify(
    x::Transpose{T,<:AbstractArray{T}}, dx::TangentOrFData
) where {T<:Union{IEEEFloat,BlasFloat}}
    _, _dx = arrayify(x.parent, _fields(dx).parent)
    return x, transpose(_dx)
end

@static if VERSION >= v"1.11-rc4"
    arrayify(x::A, dx::A) where {A<:Memory{<:Union{IEEEFloat,BlasFloat}}} = (x, dx)
end

function arrayify(x::A, dx::DA) where {A,DA}
    msg =
        "Encountered unexpected array type in `Mooncake.arrayify`. This error is likely " *
        "due to a call to a BLAS or LAPACK function with an array type that " *
        "Mooncake has not been told about. A new method of `Mooncake.arrayify` is needed." *
        " Please open an issue at " *
        "https://github.com/chalk-lab/Mooncake.jl/issues . " *
        "It should contain this error message and the associated stack trace.\n\n" *
        "Array type: $A\n\nTangent/FData type: $DA."
    return error(msg)
end

"""
    densify_tangent(dx)

Return structurally unrestricted storage in which to increment the tangent `dx`.

[`arrayify`](@ref) returns tangents wrapped in the primal's own structural type, whose
off-structure entries are not parameters: the primal reads a constant there whatever the
storage holds. A rule whose adjoint is a dense expression must therefore accumulate here
and hand the result to [`increment_densified_tangent!!`](@ref), which adds back only the
part `dx` can represent. A tangent backed by strided storage is structurally unrestricted,
even if indexing makes the view itself non-strided, so the common case costs nothing.
"""
densify_tangent(dx::StridedArray) = dx
densify_tangent(dx::SubArray{T,N,A}) where {T,N,A<:StridedArray{T}} = dx
function densify_tangent(dx::Union{UpperTriangular,LowerTriangular,Diagonal,Symmetric})
    return zeros(eltype(dx), size(dx))
end

"""
    increment_densified_tangent!!(dx, dense)

Increment `dx` by the part of `dense` that it can represent. If [`densify_tangent`](@ref)
returned `dx` itself, the increment is already complete.
"""
increment_densified_tangent!!(::StridedArray, dense) = nothing
function increment_densified_tangent!!(
    ::SubArray{T,N,A}, dense
) where {T,N,A<:StridedArray{T}}
    return nothing
end
function increment_densified_tangent!!(
    dx::T, dense
) where {T<:Union{UpperTriangular,LowerTriangular}}
    parent(dx) .+= T(dense)
    return nothing
end
function increment_densified_tangent!!(dx::Diagonal, dense)
    dx.diag .+= view(dense, diagind(dense))
    return nothing
end

# `Symmetric` is the one wrapper for which this is not masking: with `uplo == 'U'`, the
# stored `A[i, j]` is read at both `S[i, j]` and `S[j, i]` when `i < j`, so its adjoint
# picks up both. Dropping the fold would silently halve those gradients rather than throw.
function increment_densified_tangent!!(dx::Symmetric, dense)
    folded = dense .+ transpose(dense)
    folded[diagind(folded)] .= view(dense, diagind(dense))
    parent(dx) .+= dx.uplo == 'U' ? UpperTriangular(folded) : LowerTriangular(folded)
    return nothing
end

"""
    matrixify(x_dx::Union{Dual{<:AbstractVecOrMat{<:BlasFloat}},
                          CoDual{<:AbstractVecOrMat{<:BlasFloat}}})

Normalize a vector or matrix primal–tangent pair into a BLAS-compatible matrix form.

If the primal value is a vector, it is reshaped into a column matrix of size `(length(x), 1)`,
and the associated tangent is reshaped in the same way. If the primal value is already a
matrix, both the primal and tangent are returned unchanged.
"""
function matrixify(
    x_dx::Union{Dual{T},CoDual{T}}
) where {P<:Union{Float16,BlasFloat},T<:AbstractVector{P}}
    x, dx = arrayify(x_dx)
    return reshape(x, :, 1), reshape(dx, :, 1)
end
function matrixify(
    x_dx::Union{Dual{T},CoDual{T}}
) where {P<:Union{Float16,BlasFloat},T<:AbstractMatrix{P}}
    return arrayify(x_dx)
end

function viewify(
    n::BLAS.BlasInt, x_dx::Union{Dual{Ptr{P}},CoDual{Ptr{P}}}, incx::BLAS.BlasInt
) where {P<:BlasFloat}
    x, dx = arrayify(x_dx)
    xinds = 1:incx:(incx * n)
    return (
        view(unsafe_wrap(Vector{P}, x, n * incx), xinds),
        view(unsafe_wrap(Vector{P}, dx, n * incx), xinds),
    )
end
function viewify(
    n::BLAS.BlasInt, x_dx::Union{Dual{A},CoDual{A}}, incx::BLAS.BlasInt
) where {A<:AbstractArray{<:BlasFloat}}
    x, dx = arrayify(x_dx)
    xinds = 1:incx:(incx * n)
    return view(x, xinds), view(dx, xinds)
end
