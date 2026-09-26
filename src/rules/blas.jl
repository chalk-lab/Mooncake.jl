function blas_name(name::Symbol)
    return (BLAS.USE_BLAS64 ? Symbol(name, "64_") : name, Symbol(BLAS.libblastrampoline))
end

function tri!(A, u::Char, d::Char)
    return u == 'L' ? tril!(A, d == 'U' ? -1 : 0) : triu!(A, d == 'U' ? 1 : 0)
end

const BlasRealFloat = Union{Float32,Float64}
const BlasComplexFloat = Union{ComplexF32,ComplexF64}

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
Unit-triangular forward tangents are read-only strict-triangle copies; writers use fdata.

`matrixify` and `viewify` are thin wrappers built on top of `arrayify` and share the same
convention.
"""
function arrayify(
    x::CoDual{A}
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
    dx isa Tangent && x isa UnitUpperTriangular && return x, triu(_dx, 1)
    dx isa Tangent && x isa UnitLowerTriangular && return x, tril(_dx, -1)
    return x, Tx(_dx)
end
function arrayify(
    x::Symmetric{T,<:StridedMatrix{T}}, dx::TangentOrFData
) where {T<:Union{IEEEFloat,BlasFloat}}
    _, _dx = arrayify(x.data, _fields(dx).data)
    return x, Symmetric(_dx, Symbol(x.uplo))
end
# Real Hermitian has the Symmetric tangent map; complex Hermitian also conjugates
# and projects the diagonal to real, so leave its reverse rule to the derived path.
function arrayify(
    x::Hermitian{T,<:StridedMatrix{T}}, dx::TangentOrFData
) where {T<:IEEEFloat}
    _, _dx = arrayify(x.data, _fields(dx).data)
    return x, Hermitian(_dx, Symbol(x.uplo))
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
    arrayify(x::A, dx::A) where {A<:Memory{<:BlasFloat}} = (x, dx)
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

# Return aliased lane views in the primal's wrapper type, so writes reach its partials.
function arrayify(x::Lifted{<:AbstractArray{P},N}) where {P<:BlasFloat,N}
    A = primal(x)
    return A, ntuple(lane -> _arrayify_lane(A, tangent(x), lane), Val(N))
end
# Recurse through the wrapper's ImmutableDual and reconstruct the primal wrapper.
# Val(false) keeps stride-N lane views for block operations; Val(true) copies dense
# leaves so dotc/dotu's raw-memory fallback reads partials in the primal's layout.
# The static Val lets the dense leaf choice constant-fold.
@inline _arrayify_lane(x, V, lane::Integer) = _arrayify_lane(x, V, lane, Val(false))
@inline _dense_lane_partial(x::Lifted, k::Integer) = _arrayify_lane(
    primal(x), tangent(x), k, Val(true)
)
@inline _arrayify_lane(
    ::DenseArray, V::NDualArray, lane::Integer, ::Val{dense}
) where {dense} = dense ? collect(tangent_view(V, lane)) : tangent_view(V, lane)
@inline function _arrayify_lane(x::Ptr, V::NTuple{N,<:Ptr}, lane::Integer, ::Val) where {N}
    # Reject the uninit_* placeholder (the primal address) before BLAS can mutate it.
    dx = V[lane]
    IntrinsicsWrappers._check_tangent_ptr(x, dx)
    return dx
end
@inline function _arrayify_lane(x::SubArray, V::ImmutableDual, lane::Integer, d::Val)
    pp = _arrayify_lane(x.parent, V.fields.parent, lane, d)
    # Flatten nested lane views so strided inputs retain BLAS dispatch.
    return view(pp, x.indices...)
end
@inline function _arrayify_lane(
    x::Base.ReshapedArray{P,B,C,D}, V::ImmutableDual, lane::Integer, d::Val
) where {P,B,C,D}
    pp = _arrayify_lane(x.parent, V.fields.parent, lane, d)
    return Base.ReshapedArray{P,B,typeof(pp),D}(pp, x.dims, x.mi)
end
@inline _arrayify_lane(x::Adjoint, V::ImmutableDual, lane::Integer, d::Val) = adjoint(
    _arrayify_lane(x.parent, V.fields.parent, lane, d)
)
@inline _arrayify_lane(x::Transpose, V::ImmutableDual, lane::Integer, d::Val) = transpose(
    _arrayify_lane(x.parent, V.fields.parent, lane, d)
)
@inline _arrayify_lane(x::Diagonal, V::ImmutableDual, lane::Integer, d::Val) = Diagonal(
    _arrayify_lane(x.diag, V.fields.diag, lane, d)
)
@inline _arrayify_lane(x::Symmetric, V::ImmutableDual, lane::Integer, d::Val) = Symmetric(
    _arrayify_lane(x.data, V.fields.data, lane, d), Symbol(x.uplo)
)
# Hermitian(dA) is the JVP for real and complex eltypes: conjugate the mirrored
# triangle and read the diagonal as real.
@inline _arrayify_lane(x::Hermitian, V::ImmutableDual, lane::Integer, d::Val) = Hermitian(
    _arrayify_lane(x.data, V.fields.data, lane, d), Symbol(x.uplo)
)
# Infer storage type: the primal's concrete parent type can copy or reject a lane.
# Unit triangulars read 1 on the diagonal, whose derivative is zero. Preserve the
# aliased wrapper for block scatter; consumers mask its diagonal on reads
# (_mask_unit_diagonal forward, increment_densified_tangent!! reverse).
for W in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
    @eval @inline _arrayify_lane(x::$W, V::ImmutableDual, lane::Integer, d::Val) = $W(
        _arrayify_lane(x.data, V.fields.data, lane, d)
    )
end
@inline _arrayify_lane(x::Base.ReinterpretArray{T}, V::ImmutableDual, lane::Integer, d::Val) where {T} = reinterpret(
    T, _arrayify_lane(x.parent, V.fields.parent, lane, d)
)

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
function densify_tangent(
    dx::Union{
        UpperTriangular,
        LowerTriangular,
        UnitUpperTriangular,
        UnitLowerTriangular,
        Diagonal,
        Symmetric,
        Hermitian,
        Adjoint,
        Transpose,
        SubArray,
        ReshapedArray,
    },
)
    return zeros(eltype(dx), size(dx))
end

"""
    increment_densified_tangent!!(dx, dense)

Increment `dx` by the part of `dense` that it can represent. If [`densify_tangent`](@ref)
returned `dx` itself, the increment is already complete. Projection recurses through wrapper
parents. Callers must allow `dense` to be overwritten.
"""
function increment_densified_tangent!!(dx::StridedArray, dense)
    dx === dense || (dx .+= dense)
    return nothing
end
# Both view methods accumulate repeated indices, via broadcast or the explicit loop.
function increment_densified_tangent!!(
    dx::SubArray{T,N,A}, dense
) where {T,N,A<:StridedArray{T}}
    dx === dense || (dx .+= dense)
    return nothing
end
function increment_densified_tangent!!(dx::SubArray, dense)
    # Allocates and sweeps the whole parent; project through the view's indices to avoid this.
    p = densify_tangent(parent(dx))
    v = view(p, parentindices(dx)...)
    for i in eachindex(v, dense)
        @inbounds v[i] += dense[i]
    end
    increment_densified_tangent!!(parent(dx), p)
    return nothing
end
function increment_densified_tangent!!(dx::ReshapedArray, dense)
    increment_densified_tangent!!(parent(dx), reshape(dense, size(parent(dx))))
    return nothing
end
function increment_densified_tangent!!(dx::Union{UpperTriangular,LowerTriangular}, dense)
    increment_densified_tangent!!(
        parent(dx), dx isa UpperTriangular ? UpperTriangular(dense) : LowerTriangular(dense)
    )
    return nothing
end
# Unit-triangular tangents store only the strict triangle; their diagonal is constant.
function increment_densified_tangent!!(dx::UnitUpperTriangular, dense)
    p = parent(dx)
    if p isa StridedMatrix
        for j in axes(dense, 2), i in 1:(j - 1)
            @inbounds p[i, j] += dense[i, j]
        end
    else
        increment_densified_tangent!!(p, triu!(dense, 1))
    end
    return nothing
end
function increment_densified_tangent!!(dx::UnitLowerTriangular, dense)
    p = parent(dx)
    if p isa StridedMatrix
        for j in axes(dense, 2), i in (j + 1):size(dense, 1)
            @inbounds p[i, j] += dense[i, j]
        end
    else
        increment_densified_tangent!!(p, tril!(dense, -1))
    end
    return nothing
end
function increment_densified_tangent!!(dx::Diagonal, dense)
    increment_densified_tangent!!(dx.diag, view(dense, diagind(dense)))
    return nothing
end
# `Adjoint`/`Transpose` store every entry, just at the transposed position.
function increment_densified_tangent!!(dx::Adjoint, dense)
    increment_densified_tangent!!(parent(dx), adjoint(dense))
    return nothing
end
function increment_densified_tangent!!(dx::Transpose, dense)
    increment_densified_tangent!!(parent(dx), transpose(dense))
    return nothing
end

# `Symmetric` is the one wrapper for which this is not masking: with `uplo == 'U'`, the
# stored `A[i, j]` is read at both `S[i, j]` and `S[j, i]` when `i < j`, so its adjoint
# picks up both. Dropping the fold would silently halve those gradients rather than throw.
function increment_densified_tangent!!(dx::Union{Symmetric,Hermitian}, dense)
    folded = dense .+ transpose(dense)
    folded[diagind(folded)] .= view(dense, diagind(dense))
    parent(dx) .+= dx.uplo == 'U' ? UpperTriangular(folded) : LowerTriangular(folded)
    return nothing
end

"""
    matrixify(x_dx::CoDual{<:AbstractVecOrMat{<:BlasFloat}})

Normalize a vector or matrix primal–tangent pair into a BLAS-compatible matrix form.

If the primal value is a vector, it is reshaped into a column matrix of size `(length(x), 1)`,
and the associated tangent is reshaped in the same way. If the primal value is already a
matrix, both the primal and tangent are returned unchanged.
"""
function matrixify(x_dx::CoDual{T}) where {P<:Union{Float16,BlasFloat},T<:AbstractVector{P}}
    x, dx = arrayify(x_dx)
    return reshape(x, :, 1), reshape(dx, :, 1)
end
function matrixify(x_dx::CoDual{T}) where {P<:Union{Float16,BlasFloat},T<:AbstractMatrix{P}}
    return arrayify(x_dx)
end

function viewify(
    n::BLAS.BlasInt, x_dx::CoDual{Ptr{P}}, incx::BLAS.BlasInt
) where {P<:BlasFloat}
    x, dx = arrayify(x_dx)
    # Check before unsafe_wrap hides the placeholder's identity: every reverse BLAS
    # pointer rule comes through here, and accumulating into it would mutate the primal.
    IntrinsicsWrappers._check_tangent_ptr(x, dx, n)
    xinds = 1:incx:(incx * n)
    return (
        view(unsafe_wrap(Vector{P}, x, n * incx), xinds),
        view(unsafe_wrap(Vector{P}, dx, n * incx), xinds),
    )
end
@noinline function _throw_no_walk_step(x, incx)
    throw(
        ArgumentError(
            LazyString(
                "BLAS does not support operand `",
                typeof(x),
                "` with strides ",
                strides(x),
                " and `incx = ",
                incx,
                "`: the routine reads raw memory from `pointer(X)`, and no step over this ",
                "operand's own elements follows that walk, so the derivative would be taken of ",
                "different elements from the ones it read.",
            ),
        ),
    )
end

function viewify(
    n::BLAS.BlasInt, x_dx::CoDual{A}, incx::BLAS.BlasInt
) where {A<:Transpose{<:BlasFloat}}
    x = parent(primal(x_dx))
    dx = _fields(tangent(x_dx)).parent
    return viewify(n, CoDual(x, dx), incx)
end

function viewify(
    n::BLAS.BlasInt, x_dx::CoDual{A}, incx::BLAS.BlasInt
) where {A<:AbstractArray{<:BlasFloat}}
    x, dx = arrayify(x_dx)
    if x isa Union{Array,AbstractVector}
        step = _blas_walk_step(x, incx, n)
        step === nothing && _throw_no_walk_step(x, incx)
        xinds = 1:step:(1 + (n - 1) * step)
        return map((x, dx)) do z
            v = z isa Array && ndims(z) > 1 ? Base.ReshapedArray(z, (length(z),), ()) : z
            view(v, xinds)
        end
    end
    incx > 0 || _throw_no_walk_step(x, incx)
    if x isa SubArray && parent(x) isa Array
        p0 = n <= 0 ? 1 : 1 + sum((first.(x.indices) .- 1) .* strides(parent(x)))
        pinds = p0:incx:(p0 + (n - 1) * incx)
        for i in pinds
            checkbounds(Bool, parent(x), i) || _throw_no_walk_step(x, incx)
            coords = Tuple(CartesianIndices(parent(x))[i])
            all(map(in, coords, x.indices)) || _throw_no_walk_step(x, incx)
        end
        return map((x, dx)) do z
            view(Base.ReshapedArray(parent(z), (length(parent(z)),), ()), pinds)
        end
    end
    ranks = ntuple(
        d -> count(e -> (abs(stride(x, e)), -e) > (abs(stride(x, d)), -d), 1:ndims(x)),
        Val(ndims(x)),
    )
    dims = ntuple(i -> something(findfirst(==(i - 1), ranks)), Val(ndims(x)))
    offset = sum(min.(0, (size(x) .- 1) .* strides(x)))
    steps = Base.size_to_strides(1, size(x)...)
    inds = Vector{Int}(undef, max(n, 0))
    for k in 0:(n - 1)
        remaining, ind = k * incx - offset, 1
        # Physical strides decode in descending magnitude, including reversed axes.
        for d in dims
            size(x, d) == 1 && continue
            q, remaining = divrem(remaining, abs(stride(x, d)))
            0 <= q < size(x, d) || _throw_no_walk_step(x, incx)
            ind += (stride(x, d) > 0 ? q : size(x, d) - 1 - q) * steps[d]
        end
        iszero(remaining) || _throw_no_walk_step(x, incx)
        inds[k + 1] = ind
    end
    return view(x, inds), view(dx, inds)
end

#
# Utility
#

@zero_derivative MinimalCtx Tuple{typeof(BLAS.get_num_threads)}
@zero_derivative MinimalCtx Tuple{typeof(BLAS.lbt_get_num_threads)}
@zero_derivative MinimalCtx Tuple{typeof(BLAS.set_num_threads),Union{Integer,Nothing}}
@zero_derivative MinimalCtx Tuple{typeof(BLAS.lbt_set_num_threads),Any}

# Output operands must be disjoint from read-only inputs and their shared tangents.
@inline _check_blas_output_alias(f, output) = nothing
@inline function _check_blas_output_alias(f, output, input, inputs...)
    !isempty(output) &&
        !isempty(input) &&
        Base.mightalias(output, input) &&
        _blas_overlaps(output, input) &&
        _throw_blas_output_alias(f)
    return _check_blas_output_alias(f, output, inputs...)
end
function _blas_overlaps(a, b)
    wrappers = Union{Transpose,Adjoint,LinearAlgebra.AbstractTriangular,Symmetric,Hermitian}
    a isa wrappers && return _blas_overlaps(parent(a), b)
    b isa wrappers && return _blas_overlaps(a, parent(b))
    sa = sizeof(eltype(a)) .* strides(a)
    sb = sizeof(eltype(b)) .* strides(b)
    a0, b0 = Int(pointer(a)), Int(pointer(b))
    alo = a0 + sum(min.(0, (size(a) .- 1) .* sa))
    ahi = a0 + sum(max.(0, (size(a) .- 1) .* sa))
    blo = b0 + sum(min.(0, (size(b) .- 1) .* sb))
    bhi = b0 + sum(max.(0, (size(b) .- 1) .* sb))
    (ahi < blo || bhi < alo) && return false
    length(a) > length(b) && return _blas_overlaps(b, a)
    dims = abs(stride(b, 1)) >= abs(stride(b, 2)) ? (1, 2) : (2, 1)
    for i in CartesianIndices(a)
        offset = a0 + sum((Tuple(i) .- 1) .* sa) - blo
        0 <= offset <= bhi - blo || continue
        for d in dims
            size(b, d) == 1 && continue
            q, offset = divrem(offset, abs(stride(b, d)) * sizeof(eltype(b)))
            if q >= size(b, d)
                offset = -1
                break
            end
        end
        iszero(offset) && return true
    end
    return false
end
@noinline function _throw_blas_output_alias(f)
    throw(
        ArgumentError(
            "Mooncake cannot differentiate $(nameof(f)) with overlapping input and output operands. " *
            "Pass a copy of the input or use an elementwise Julia update.",
        ),
    )
end

# Differentiate the guarded product itself: differentiating its branch would discard
# a live perturbation at a zero multiplier under forward-over-reverse.
_rvs_mul(x::T, y::T) where {T<:BlasFloat} = ifelse(iszero(y), zero(T), x * y)
@is_primitive MinimalCtx ForwardMode Tuple{typeof(_rvs_mul),T,T} where {T<:BlasFloat}
function frule!!(
    ::Lifted{typeof(_rvs_mul),N}, x::Lifted{T,N}, y::Lifted{T,N}
) where {N,T<:BlasFloat}
    a, b = primal(x), primal(y)
    c = _rvs_mul(a, b)
    dc = ntuple(k -> _rvs_mul(a, tangent(y, k)) + _rvs_mul(tangent(x, k), b), Val(N))
    return Lifted{T,N}(c, _scalar_ndual(c, dc))
end

# Mask the value only, so a whole-zero seed's live direction survives forward-over-reverse.
_rvs_zero(x::BlasFloat, zero_seed::Bool) = ifelse(zero_seed, zero(x), x)
@is_primitive MinimalCtx ForwardMode Tuple{typeof(_rvs_zero),BlasFloat,Bool}
function frule!!(::Dual{typeof(_rvs_zero)}, x::Dual, zero_seed::Dual{Bool})
    return Dual(_rvs_zero(primal(x), primal(zero_seed)), tangent(x))
end

# Skip an in-place kernel at a whole-zero seed; its frule still runs the kernel's frule
# and restores only the primal output, so the seed's direction propagates.
@noinline function _rvs_blas!(f::F, zero_seed::Bool, args::Vararg{Any,N}) where {F,N}
    zero_seed || f(args...)
    return last(args)
end
@is_primitive MinimalCtx ForwardMode Tuple{typeof(_rvs_blas!),Any,Bool,Vararg}
function frule!!(
    ::Dual{typeof(_rvs_blas!)}, f::Dual, zero_seed::Dual{Bool}, args::Vararg{Dual,N}
) where {N}
    out = last(args)
    saved = primal(zero_seed) ? copy(primal(out)) : nothing
    frule!!(f, args...)
    saved === nothing || copyto!(primal(out), saved)
    return out
end

# Out of line, with the mask hoisted: fused into a pullback, the strided broadcast
# runs 15-25% slower.
@noinline function _rvs_axpy!(mul::F, Y, X, a, zero_seed::Bool) where {F}
    if zero_seed
        Y .+= _rvs_zero.(mul.(X, a), true)
    else
        Y .+= mul.(X, a)
    end
    return Y
end
_rvs_conj_mul(x, a) = conj(x) * a

# Evaluate the broadcast with the transform's scalar rules, including Base's complex expansion.
struct _RvsScalar{T<:BlasRealFloat} <: Real
    dual::Lifted{T,1,NDual{T,1}}
    _RvsScalar(dual::Lifted{T,1,NDual{T,1}}) where {T<:BlasRealFloat} = new{T}(dual)
end
_RvsScalar(x::T, dx::T) where {T<:BlasRealFloat} = _RvsScalar(lift(x, dx))
function _RvsScalar(x::Complex, dx::Complex)
    return Complex(_RvsScalar(real(x), real(dx)), _RvsScalar(imag(x), imag(dx)))
end
for (op, intrinsic) in ((:+, :add_float), (:-, :sub_float), (:*, :mul_float))
    @eval Base.$op(x::_RvsScalar{T}, y::_RvsScalar{T}) where {T} = _RvsScalar(
        frule!!(zero_dual(IntrinsicsWrappers.$intrinsic), x.dual, y.dual)
    )
end
function Base.:-(x::_RvsScalar)
    return _RvsScalar(frule!!(zero_dual(IntrinsicsWrappers.neg_float), x.dual))
end
_rvs_extract(x::_RvsScalar) = (primal(x.dual), tangent(x.dual, 1))
function _rvs_extract(x::Complex{<:_RvsScalar})
    a, da = _rvs_extract(real(x))
    b, db = _rvs_extract(imag(x))
    return complex(a, b), complex(da, db)
end

@inline function _rvs_product(a, x, y)
    # Inline Base's complex operations on the scalar adapter too.
    return @inline (a * x) * y
end

# Specialize the update modes outside the loop so the scalar rules can vectorize.
function _rvs_vector_frule!(dc, x, dx, y, dy, a, da, tx, ty, ::Val{add}) where {add}
    ad = _RvsScalar(a, da)
    @inbounds for j in axes(dc, 2)
        dcj = view(dc, :, j)
        yd = ty == 'N' ? _RvsScalar(y[1, j], dy[1, j]) : _RvsScalar(y[j, 1], dy[j, 1])
        ty == 'C' && (yd = conj(yd))
        @simd ivdep for i in axes(dc, 1)
            xd = if tx == 'N'
                _RvsScalar(x[i, 1], dx[i, 1])
            else
                _RvsScalar(x[1, i], dx[1, i])
            end
            tx == 'C' && (xd = conj(xd))
            _, v = _rvs_extract(_rvs_product(ad, xd, yd))
            dcj[i] = add ? dcj[i] + v : v
        end
    end
    return nothing
end

# Skip unused operands without losing a live coefficient direction in nested AD.
# Vector callers use coefficient-first products; matrix callers use BLAS scaling.
# Output storage must be disjoint from inputs.
@inline function _rvs_muladd!(
    C::AbstractMatrix{T},
    X::AbstractVecOrMat{T},
    Y::AbstractVecOrMat{T},
    α::T,
    tX::Char,
    tY::Char,
    add::Bool,
    coefficient_first::Bool,
) where {T<:BlasFloat}
    if iszero(α) || all(iszero, X) || all(iszero, Y)
        add || fill!(C, zero(T))
    elseif coefficient_first && (tX == 'N' ? size(X, 2) : size(X, 1)) == 1
        @inbounds for j in axes(C, 2)
            cj = view(C, :, j)
            y = tY == 'N' ? Y[1, j] : Y[j, 1]
            tY == 'C' && (y = conj(y))
            @simd ivdep for i in axes(C, 1)
                x = tX == 'N' ? X[i, 1] : X[1, i]
                tX == 'C' && (x = conj(x))
                v = _rvs_product(α, x, y)
                cj[i] = add ? cj[i] + v : v
            end
        end
    else
        # Avoid the BLAS wrapper call boundary in tiny GEMM pullbacks.
        @inline BLAS.gemm!(tX, tY, α, X, Y, T(add), C)
    end
    return C
end
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(_rvs_muladd!),
    AbstractMatrix{T},
    AbstractVecOrMat{T},
    AbstractVecOrMat{T},
    T,
    Char,
    Char,
    Bool,
    Bool,
} where {T<:BlasFloat}
function frule!!(
    ::Lifted{typeof(_rvs_muladd!),N},
    C::Lifted,
    X::Lifted,
    Y::Lifted,
    α::Lifted,
    tX::Lifted{Char},
    tY::Lifted{Char},
    add::Lifted{Bool},
    coefficient_first::Lifted{Bool},
) where {N}
    c, x, y, a = primal(C), primal(X), primal(Y), primal(α)
    tx, ty = primal(tX), primal(tY)
    for k in 1:N
        dc = _blas_lane_partial(C, k)
        dx = _blas_lane_partial(X, k)
        dy = _blas_lane_partial(Y, k)
        da = tangent(α, k)
        if primal(coefficient_first) && (tx == 'N' ? size(x, 2) : size(x, 1)) == 1
            add_mode = primal(add) ? Val(true) : Val(false)
            _rvs_vector_frule!(dc, x, dx, y, dy, a, da, tx, ty, add_mode)
        elseif primal(coefficient_first)
            _rvs_muladd!(dc, x, y, da, tx, ty, primal(add), true)
            _rvs_muladd!(dc, dx, y, a, tx, ty, true, true)
            _rvs_muladd!(dc, x, dy, a, tx, ty, true, true)
        else
            # Mooncake's BLAS.gemm! frule adds operand directions before the coefficient direction.
            _rvs_muladd!(dc, dx, y, a, tx, ty, primal(add), false)
            _rvs_muladd!(dc, x, dy, a, tx, ty, true, false)
            _rvs_muladd!(dc, x, y, da, tx, ty, true, false)
        end
    end
    _rvs_muladd!(c, x, y, a, tx, ty, primal(add), primal(coefficient_first))
    return C
end

# Strong zero on the cotangent: unused NaN entries must not poison scalar gradients.
# In particular, BLAS permits undefined input y wherever β == 0 discards it.
@inline function _rvs_guarded_dot(y, dy, conjugate::Bool=false)
    s = zero(promote_type(eltype(y), eltype(dy)))
    @inbounds @simd for i in eachindex(y, dy)
        d = conjugate ? conj(dy[i]) : dy[i]
        s += _rvs_mul(y[i]', d)
    end
    return s
end

@inline function _rvs_guarded_dot(A::AbstractMatrix, x::AbstractVector, dy::AbstractVector)
    s = zero(promote_type(eltype(A), eltype(x), eltype(dy)))
    @inbounds for j in axes(A, 2)
        t = zero(s)
        @simd for i in axes(A, 1)
            t += _rvs_mul(A[i, j]', dy[i])
        end
        s += _rvs_mul(x[j]', t)
    end
    return s
end

#
# LEVEL 1
#

for (fname, jlfname, elty) in (
    (:cblas_ddot, :dot, :Float64),
    (:cblas_sdot, :dot, :Float32),
    (:cblas_zdotc_sub, :dotc, :ComplexF64),
    (:cblas_cdotc_sub, :dotc, :ComplexF32),
    (:cblas_zdotu_sub, :dotu, :ComplexF64),
    (:cblas_cdotu_sub, :dotu, :ComplexF32),
)
    isreal = jlfname == :dot

    # Real cblas dot returns by value. Complex dotc/dotu write a Ref and need
    # forward primitives at the BLAS wrapper instead; reverse handles all here.
    if isreal
        @eval @inline function frule!!(
            ::Lifted{typeof(_foreigncall_),Nw},
            ::Lifted{Val{$(blas_name(fname))}},
            ::Lifted, # return type
            ::Lifted, # argument types
            ::Lifted, # nreq
            ::Lifted, # calling convention
            _n::Lifted{BLAS.BlasInt},
            _DX::Lifted{Ptr{$elty},Nw,NTuple{Nw,Ptr{$elty}}},
            _incx::Lifted{BLAS.BlasInt},
            _DY::Lifted{Ptr{$elty},Nw,NTuple{Nw,Ptr{$elty}}},
            _incy::Lifted{BLAS.BlasInt},
            args::Vararg{Any,M},
        ) where {Nw,M}
            GC.@preserve args begin
                n, incx, incy = primal(_n), primal(_incx), primal(_incy)
                DX = primal(_DX)
                DY = primal(_DY)
                result = BLAS.$jlfname(n, DX, incx, DY, incy)
                # _blas_lane_partial checks for the uninit_* primal-address placeholder.
                dresult_lanes = ntuple(Val(Nw)) do lane
                    dDX = _blas_lane_partial(_DX, lane)
                    dDY = _blas_lane_partial(_DY, lane)
                    return BLAS.$jlfname(n, dDX, incx, DY, incy) +
                           BLAS.$jlfname(n, DX, incx, dDY, incy)
                end
                return Lifted{$elty,Nw}(result, _scalar_ndual(result, dresult_lanes))
            end
        end
    end
    @eval @inline function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _n::CoDual{BLAS.BlasInt},
        _DX::CoDual{Ptr{$elty}},
        _incx::CoDual{BLAS.BlasInt},
        _DY::CoDual{Ptr{$elty}},
        _incy::CoDual{BLAS.BlasInt},
        $((isreal ? () : (:(_presult::CoDual{Ptr{$elty}}),))...),
        args::Vararg{Any,N},
    ) where {N}
        GC.@preserve args begin
            # Load in values from pointers.
            n, incx, incy = map(primal, (_n, _incx, _incy))
            DX, _dDX = viewify(n, _DX, incx)
            DY, _dDY = viewify(n, _DY, incy)

            # Run primal computation.
            result = BLAS.$jlfname(DX, DY)

            # For complex numbers the primal result must be stored in the pointer, and the dual must be zeroed
            $(isreal ? :() : quote
                presult, _dpresult = arrayify(_presult)
                Base.unsafe_store!(presult, result)
                Base.unsafe_store!(_dpresult, zero($elty))

                result = nothing
            end)
        end

        $(
            if jlfname == :dot
                quote
                    function dot_pb!!(dv)
                        GC.@preserve args begin
                            _rvs_axpy!(*, _dDX, DY, dv, iszero(dv))
                            _rvs_axpy!(*, _dDY, DX, dv, iszero(dv))
                        end
                        return tuple_fill(NoRData(), Val(N + 11))
                    end
                end
            elseif jlfname == :dotc
                quote
                    function dot_pb!!(::NoRData)
                        GC.@preserve args begin
                            dv = Base.unsafe_load(_dpresult)
                            _rvs_axpy!(*, _dDX, DY, dv', iszero(dv))
                            _rvs_axpy!(*, _dDY, DX, dv, iszero(dv))
                        end
                        return tuple_fill(NoRData(), Val(N + 12))
                    end
                end
            else
                quote
                    function dot_pb!!(::NoRData)
                        GC.@preserve args begin
                            dv = Base.unsafe_load(_dpresult)
                            _rvs_axpy!(_rvs_conj_mul, _dDX, DY, dv, iszero(dv))
                            _rvs_axpy!(_rvs_conj_mul, _dDY, DX, dv, iszero(dv))
                        end
                        return tuple_fill(NoRData(), Val(N + 12))
                    end
                end
            end
        )

        return CoDual(result, NoFData()), dot_pb!!
    end
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.nrm2),Integer,X,Integer
    } where {T<:BlasFloat,X<:Union{Ptr{T},AbstractArray{T}}},
)
# At length >= 32, norm2 inlines nrm2(x) and its ccall past the three-argument
# boundary. Keep a one-argument primitive so the raw pointer stays out of the transform.
@is_primitive(
    MinimalCtx, Tuple{typeof(BLAS.nrm2),X} where {T<:BlasFloat,X<:AbstractArray{T}}
)

function frule!!(
    f::Lifted{typeof(BLAS.nrm2),Nw}, X_dX::Lifted{<:AbstractArray{T}}
) where {Nw,T<:BlasFloat}
    x = primal(X_dX)
    n = Lifted{Int,Nw}(length(x), NoDual())
    return frule!!(f, n, X_dX, Lifted{Int,Nw}(stride(x, 1), NoDual()))
end

function rrule!!(
    f::CoDual{typeof(BLAS.nrm2)}, X_dX::CoDual{<:AbstractArray{T}}
) where {T<:BlasFloat}
    x = primal(X_dX)
    y, pb = rrule!!(f, zero_fcodual(length(x)), X_dX, zero_fcodual(stride(x, 1)))
    # The three-argument pullback accumulates into `X_dX`'s fdata and returns rdata for its four
    # arguments; this form has two, so drop the length and stride slots.
    nrm2_len_pb!!(dy) = (NoRData(), pb(dy)[3])
    return y, nrm2_len_pb!!
end
# Guard scalar cotangent contractions against NaN in unused entries, as in gemv!.
# Complex OpenBLAS kernels can read an overwritten real component when X and Y alias.
# Compare raw walks, including views and pointers; disjoint strided walks remain valid.
@inline function _axpy_overlaps(x, y, n::Integer, incx::Integer, incy::Integer)
    n <= 0 && return false
    px = UInt(x isa Ptr ? x : pointer(x))
    py = UInt(y isa Ptr ? y : pointer(y))
    px == py && return true
    sx, sy = abs(incx) * sizeof(eltype(x)), abs(incy) * sizeof(eltype(y))
    hx, hy = px + (n - 1) * sx, py + (n - 1) * sy
    (hx < py || hy < px) && return false
    sx == sy && return iszero((max(px, py) - min(px, py)) % sx)
    for i in 0:(n - 1)
        p = px + i * sx
        py <= p <= hy && (iszero(sy) || iszero((p - py) % sy)) && return true
    end
    return false
end
@noinline function _throw_axpy_overlap()
    throw(
        ArgumentError(
            "Mooncake cannot differentiate BLAS.axpy! with overlapping complex source and " *
            "destination operands: OpenBLAS can overwrite a component before reading it. " *
            "Pass a copy of the source or use an elementwise Julia update.",
        ),
    )
end
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.axpy!),Integer,P,X,Integer,Y,Integer
    } where {
        P<:BlasFloat,X<:Union{Ptr{P},AbstractArray{P}},Y<:Union{Ptr{P},AbstractArray{P}}
    },
)

function frule!!(
    ::Lifted{typeof(BLAS.axpy!),Nw},
    _n::Lifted,
    a_da::Lifted{P,Nw},
    X_dX::Lifted{<:Union{Ptr{P},AbstractArray{P}}},
    _incx::Lifted,
    Y_dY::Lifted{<:Union{Ptr{P},AbstractArray{P}}},
    _incy::Lifted,
) where {Nw,P<:BlasFloat}
    n, incx, incy = primal(_n), primal(_incx), primal(_incy)
    x, y, a = primal(X_dX), primal(Y_dY), primal(a_da)
    if _axpy_overlaps(x, y, n, incx, incy)
        P <: Complex && _throw_axpy_overlap()
        ((x isa Ptr ? x : pointer(x)) == (y isa Ptr ? y : pointer(y)) && incx == incy) ||
            _throw_blas_output_alias(BLAS.axpy!)
    end
    # `dy := a*dx + da*x + dy`, then the primal ONCE, after every lane has read the original `x`.
    # Broadcast rather than `BLAS.axpy!` on the lane partial: above width 1 a lane is a stride-`Nw`
    # view, which the pointer-based wrapper misreads -- it silently dropped the last element.
    lbl = "Forward-mode `BLAS.axpy!`"
    step_x = _checked_walk_step(lbl, n, x, incx)
    step_y = _checked_walk_step(lbl, n, y, incy)
    xv = _viewify_one(n, x, step_x)
    for k in 1:Nw
        dy_k = _viewify_one(n, _blas_lane_partial(Y_dY, k), step_y)
        dx_k = _viewify_one(n, _blas_lane_partial(X_dX, k), step_x)
        dy_k .= a .* dx_k .+ tangent(a_da, k) .* xv .+ dy_k
    end
    BLAS.axpy!(n, a, x, incx, y, incy)
    return Y_dY
end

function rrule!!(
    ::CoDual{typeof(BLAS.axpy!)},
    _n::CoDual,
    a_da::CoDual{P},
    X_dX::CoDual{<:Union{Ptr{P},AbstractArray{P}}},
    _incx::CoDual,
    Y_dY::CoDual{<:Union{Ptr{P},AbstractArray{P}}},
    _incy::CoDual,
) where {P<:BlasFloat}
    n, incx, incy = primal(_n), primal(_incx), primal(_incy)
    a = primal(a_da)
    xp, yp = primal(X_dX), primal(Y_dY)
    if _axpy_overlaps(xp, yp, n, incx, incy)
        P <: Complex && _throw_axpy_overlap()
        (
            (xp isa Ptr ? xp : pointer(xp)) == (yp isa Ptr ? yp : pointer(yp)) &&
            incx == incy
        ) || _throw_blas_output_alias(BLAS.axpy!)
    end
    x, dx = viewify(n, X_dX, incx)
    y, dy = viewify(n, Y_dY, incy)
    # Restore from a copy, not `y .-= a .* x`: the supported real X === Y case updates
    # `x` too, so the subtraction would not recover the old value.
    y_copy = copy(y)
    BLAS.axpy!(n, a, primal(X_dX), incx, primal(Y_dY), incy)
    function axpy!_pb!!(::NoRData)
        y .= y_copy
        ∇a = _rvs_guarded_dot(x, dy)
        dx .+= _rvs_mul.(dy, a')                    # `y` keeps its own cotangent: d y_new / d y_old is I
        return NoRData(), NoRData(), ∇a, NoRData(), NoRData(), NoRData(), NoRData()
    end
    return Y_dY, axpy!_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.axpy!),Number,X,Y
    } where {P<:BlasFloat,X<:AbstractArray{P},Y<:AbstractArray{P}},
)

# Short forms accept any Number and convert with P(alpha); narrower rules miss
# calls such as axpy!(2, x, y). Convert lane partials too, or zero for NoDual.
# Unsupported tangents (e.g. BigFloat, TwicePrecision) still raise MethodError.
function _fwd_blas_alpha(::Type{P}, a::Lifted{<:Number,Nw,NoDual}) where {P<:BlasFloat,Nw}
    v = P(primal(a))
    return Lifted{P,Nw}(v, zero_dual(Val(Nw), v))
end
function _fwd_blas_alpha(::Type{P}, a::Lifted{<:Number,Nw}) where {P<:BlasFloat,Nw}
    v = P(primal(a))
    return Lifted{P,Nw}(v, dual_type(Val(Nw), P)(tangent(a)))
end

# Reverse needs no matching conversion on the way in -- a scalar carries no fdata, so the converted
# operand is just `zero_fcodual(P(alpha))` -- only the returned rdata converting back.
function _rvs_blas_alpha(::CoDual{A}, ∇::Number) where {A<:Number}
    R = rdata_type(tangent_type(A))
    R === NoRData && return NoRData()
    # A real scalar scaling a complex array moves only along the real axis, so its cotangent is the
    # real part of `dot(x, dy)`; `convert` alone raises `InexactError` on the imaginary component.
    return R <: Real ? convert(R, real(∇)) : convert(R, ∇)
end

function frule!!(
    f::Lifted{typeof(BLAS.axpy!),Nw},
    a_da::Lifted{<:Number,Nw},
    X_dX::Lifted{<:AbstractArray{P}},
    Y_dY::Lifted{<:AbstractArray{P}},
) where {Nw,P<:BlasFloat}
    x = primal(X_dX)
    n = Lifted{Int,Nw}(length(x), NoDual())
    ix = Lifted{Int,Nw}(stride(x, 1), NoDual())
    iy = Lifted{Int,Nw}(stride(primal(Y_dY), 1), NoDual())
    return frule!!(f, n, _fwd_blas_alpha(P, a_da), X_dX, ix, Y_dY, iy)
end

function rrule!!(
    f::CoDual{typeof(BLAS.axpy!)},
    a_da::CoDual{<:Number},
    X_dX::CoDual{<:AbstractArray{P}},
    Y_dY::CoDual{<:AbstractArray{P}},
) where {P<:BlasFloat}
    x = primal(X_dX)
    y, pb = rrule!!(
        f,
        zero_fcodual(length(x)),
        zero_fcodual(P(primal(a_da))),
        X_dX,
        zero_fcodual(stride(x, 1)),
        Y_dY,
        zero_fcodual(stride(primal(Y_dY), 1)),
    )
    # Seven slots `(f, n, a, X, incx, Y, incy)` down to four `(f, a, X, Y)`.
    function axpy!_short_pb!!(dy)
        r = pb(dy)
        return NoRData(), _rvs_blas_alpha(a_da, r[3]), r[4], r[6]
    end
    return y, axpy!_short_pb!!
end

# `axpby!(a, x, b, y)`: `y := a*x + b*y`. `y` is NOT recoverable by inverting when `b == 0`, so the
# pullback works from a copy, as the `gemm!` family does.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.axpby!),Number,X,Number,Y
    } where {P<:BlasFloat,X<:AbstractArray{P},Y<:AbstractArray{P}},
)

function frule!!(
    ::Lifted{typeof(BLAS.axpby!),Nw},
    _a::Lifted{<:Number,Nw},
    X_dX::Lifted{<:AbstractArray{P}},
    _b::Lifted{<:Number,Nw},
    Y_dY::Lifted{<:AbstractArray{P}},
) where {Nw,P<:BlasFloat}
    x, y = primal(X_dX), primal(Y_dY)
    x === y || _check_blas_output_alias(BLAS.axpby!, y, x)
    a_da, b_db = _fwd_blas_alpha(P, _a), _fwd_blas_alpha(P, _b)
    a, b = primal(a_da), primal(b_db)
    for k in 1:Nw
        dx_k = _blas_lane_partial(X_dX, k)
        dy_k = _blas_lane_partial(Y_dY, k)
        da, db = tangent(a_da, k), tangent(b_db, k)
        # `dy := a*dx + da*x + b*dy + db*y`, every term read before the primal overwrites `y`.
        # Both `β` terms carry the strong zero the other `β`-taking rules use: BLAS discards `y`
        # entirely at `β == 0`, so it may hold undefined values there and `0 * NaN` is `NaN`.
        # Unguarded, `axpby!(a, [2.0], 0.0, [NaN])` gave a JVP of `NaN` on a primal of 2.0.
        @inbounds for i in eachindex(dy_k, dx_k, x, y)
            t = a * dx_k[i] + da * x[i]
            iszero(b) || (t += b * dy_k[i])
            if !iszero(db)
                yi = y[i]
                isnan(yi) || (t += db * yi)
            end
            dy_k[i] = t
        end
    end
    BLAS.axpby!(a, x, b, y)
    return Y_dY
end

function rrule!!(
    ::CoDual{typeof(BLAS.axpby!)},
    a_da::CoDual{<:Number},
    X_dX::CoDual{<:AbstractArray{P}},
    b_db::CoDual{<:Number},
    Y_dY::CoDual{<:AbstractArray{P}},
) where {P<:BlasFloat}
    a, b = P(primal(a_da)), P(primal(b_db))
    x, dx = arrayify(X_dX)
    y, dy = arrayify(Y_dY)
    x === y || _check_blas_output_alias(BLAS.axpby!, y, x)
    y_copy = copy(y)
    BLAS.axpby!(a, x, b, y)
    function axpby!_pb!!(::NoRData)
        copyto!(y, y_copy)
        ∇a = _rvs_blas_alpha(a_da, _rvs_guarded_dot(x, dy))
        ∇b = _rvs_blas_alpha(b_db, _rvs_guarded_dot(y_copy, dy))
        if x === y
            # Read the shared cotangent before writing it. Sum first for cancellation,
            # but scale first if the coefficient sum overflows.
            c = (a + b)'
            if isfinite(c)
                dy .*= c
            else
                dy .= a' .* dy .+ b' .* dy
            end
        else
            dx .+= _rvs_mul.(dy, a')
            dy .*= b'
        end
        return NoRData(), ∇a, NoRData(), ∇b, NoRData()
    end
    return Y_dY, axpby!_pb!!
end

# Keep the convenience forms primitive: Julia otherwise inlines through their ccall
# into raw pointers, unsupported above width 1. Both use the array's own stride.
@is_primitive MinimalCtx Tuple{
    typeof(BLAS.scal!),P,X
} where {P<:BlasFloat,X<:AbstractArray{P}}

function frule!!(
    f::Lifted{typeof(BLAS.scal!),Nw}, a_da::Lifted{P,Nw}, X_dX::Lifted{<:AbstractArray{P}}
) where {Nw,P<:BlasFloat}
    x = primal(X_dX)
    n = Lifted{Int,Nw}(length(x), NoDual())
    return frule!!(f, n, a_da, X_dX, Lifted{Int,Nw}(stride(x, 1), NoDual()))
end

function rrule!!(
    f::CoDual{typeof(BLAS.scal!)}, a_da::CoDual{P}, X_dX::CoDual{<:AbstractArray{P}}
) where {P<:BlasFloat}
    x = primal(X_dX)
    y, pb = rrule!!(f, zero_fcodual(length(x)), a_da, X_dX, zero_fcodual(stride(x, 1)))
    # The four-argument pullback returns rdata for `(f, n, a, X, incx)`; this form has
    # `(f, a, X)`, so keep the scaling and array slots and drop the length and stride.
    function scal!_short_pb!!(dy)
        r = pb(dy)
        return NoRData(), r[3], r[4]
    end
    return y, scal!_short_pb!!
end

# The five-argument BLAS.dot has only a raw-pointer rule, so handle arrays here.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.dot),X,Y
    } where {P<:BlasRealFloat,X<:AbstractArray{P},Y<:AbstractArray{P}},
)

function frule!!(
    ::Lifted{typeof(BLAS.dot),Nw},
    X_dX::Lifted{<:AbstractArray{P}},
    Y_dY::Lifted{<:AbstractArray{P}},
) where {Nw,P<:BlasRealFloat}
    x, dxs = arrayify(X_dX)
    y, dys = arrayify(Y_dY)
    v = BLAS.dot(x, y)
    lanes = ntuple(k -> BLAS.dot(dxs[k], y) + BLAS.dot(x, dys[k]), Val(Nw))
    return Lifted{P,Nw}(v, _scalar_ndual(v, lanes))
end

function rrule!!(
    ::CoDual{typeof(BLAS.dot)},
    X_dX::CoDual{<:AbstractArray{P}},
    Y_dY::CoDual{<:AbstractArray{P}},
) where {P<:BlasRealFloat}
    x, dx = arrayify(X_dX)
    y, dy = arrayify(Y_dY)
    function blas_dot_pb!!(dv::P)
        dx .+= y .* dv
        dy .+= x .* dv
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(BLAS.dot(x, y)), blas_dot_pb!!
end

# Wrapper-aware lane extraction for arrays and pointers.
@inline _blas_lane_partial(x::Lifted, lane::Integer) = _arrayify_lane(
    primal(x), tangent(x), lane
)

# ── Lane-leading partials blocks for BLAS/LAPACK forward rules ──────────────────

# Return (block, copied) with shape (N, size(primal(x))...), contiguous lanes per
# element. Dense slots alias their NDualArray block; wrapper slots gather into a
# copy, requiring _write_back_partials! after mutation. Lane views have stride N,
# but the block has unit first-dimension stride for BLAS/LAPACK: right-multiplying
# its (N, len) matrix by a lane-invariant map's transpose processes all lanes.
@inline function _partials_block(
    x::Lifted{P,N,<:NDualArray}
) where {T,D,P<:AbstractArray{T,D},N}
    return getfield(tangent(x), :partials_block), false
end
# Unit triangulars read a structural 1 on the diagonal, whose derivative is zero.
# Mask on reads (_partials_block and kron), not in _arrayify_lane: block scatter
# must write through the aliased wrapper to the slot's storage.
@inline _mask_unit_diagonal(z) = z
@inline _mask_unit_diagonal(z::UnitUpperTriangular) = triu(parent(z), 1)
@inline _mask_unit_diagonal(z::UnitLowerTriangular) = tril(parent(z), -1)

@inline function _partials_block(x::Lifted{P,N}) where {T,D,P<:AbstractArray{T,D},N}
    p = primal(x)
    blk = Array{T,D + 1}(undef, (N, size(p)...))
    colons = ntuple(_ -> Colon(), Val(D))
    for k in 1:N
        copyto!(view(blk, k, colons...), _mask_unit_diagonal(_blas_lane_partial(x, k)))
    end
    return blk, true
end
@inline function _write_back_partials!(
    x::Lifted{P,N}, blk::AbstractArray
) where {T,D,P<:AbstractArray{T,D},N}
    colons = ntuple(_ -> Colon(), Val(D))
    for k in 1:N
        copyto!(_blas_lane_partial(x, k), view(blk, k, colons...))
    end
    return nothing
end

# Strong-zero dot(dy, B, x)', without materialising B*x.
@inline function _rvs_guarded_dot3(dy, B, x)
    s = zero(promote_type(eltype(dy), eltype(B), eltype(x)))
    @inbounds for i in eachindex(dy)
        d = dy[i]
        r = zero(s)
        for j in eachindex(x)
            r += B[i, j] * x[j]
        end
        s += _rvs_mul(r', d)
    end
    return s
end

# BLAS β == 0 overwrites rather than multiplying a possibly NaN tangent.
# A zero contracted dimension quick-returns without applying β (e.g. gemv!).
@inline function _scale_or_zero!(B::AbstractArray{T}, β) where {T}
    B .= _rvs_mul.(B, β)
    return nothing
end
@inline function _scale_or_zero!(B::AbstractArray, β, contracted::Integer)
    contracted == 0 && return nothing
    return _scale_or_zero!(B, β)
end

# `X[i]*dX[i]` overflows once `norm(X)*norm(dX)` leaves `T`'s range, though the JVP it divides down
# to is representable. Scaling `X` by the power of two nearest `y` is exact, so in-range results are
# unchanged; a subnormal `y` has no representable reciprocal, and with every `X[i]` subnormal too it
# needs none. Both lane paths scale by this `r` and divide by `y*r`.
@inline function _nrm2_scale_factor(y)
    r = isfinite(y) && !iszero(y) ? ldexp(one(y), -exponent(y)) : one(y)
    return isfinite(r) ? r : one(y)
end

# Per-lane nrm2 JVP: `dy_k = Σᵢ real(conj(Xᵢ)·dXₖᵢ)/y` with a removable-singularity guard. Fallback
# for the Ptr slot and strided (incx≠1) inputs, where the contiguous-block fast path does not apply.
@inline function _nrm2_lanes_perlane(
    _n, X_dX, step, Xv, y, ::Type{R}, ::Val{Nw}
) where {R,Nw}
    r = _nrm2_scale_factor(y)
    return ntuple(Val(Nw)) do lane
        dX_lane = _blas_lane_partial(X_dX, lane)
        dXv = _viewify_one(_n, dX_lane, step)
        s = zero(R)
        @inbounds for i in eachindex(Xv)
            s += real((Xv[i] * r)' * dXv[i])
        end
        iszero(y) ? zero(R) : s / (y * r)
    end
end

# Contiguous NTuple columns let lane updates vectorise (~4×). Zero sums map to
# zero at the removable singularity. Pass acc by value: capturing a reassigned
# accumulator boxes it to Any. Bind only Nw; NTuple{0,R} leaves R unbound (Aqua).
@inline _nrm2_accum(acc::NTuple{Nw}, xi, col) where {Nw} = ntuple(
    k -> acc[k] + real(xi' * col[k]), Val(Nw)
)
@inline _nrm2_scale(acc::NTuple{Nw}, yr) where {Nw} = ntuple(
    k -> iszero(yr) ? zero(acc[k]) : acc[k] / yr, Val(Nw)
)
@inline function _nrm2_lanes_block(blk, Xv, y, ::Type{R}, ::Val{Nw}) where {R,Nw}
    cols = reinterpret(reshape, NTuple{Nw,eltype(blk)}, blk)
    acc = ntuple(_ -> zero(R), Val(Nw))
    r = _nrm2_scale_factor(y)
    @inbounds for i in eachindex(Xv)
        acc = _nrm2_accum(acc, Xv[i] * r, cols[i])
    end
    return _nrm2_scale(acc, y * r)
end

# Dispatch separates array block access from Ptr lanes to keep the frule type-stable.
# Only logical step 1 permits the block fast path; pointers always use per-lane accumulation.
@inline function _nrm2_lanes(
    X_dX::Lifted{P,Nw}, _n, step, Xv, y, ::Type{R}
) where {T,P<:AbstractArray{T},Nw,R}
    if step == 1
        blk, _ = _partials_block(X_dX)
        return _nrm2_lanes_block(blk, Xv, y, R, Val(Nw))
    end
    return _nrm2_lanes_perlane(_n, X_dX, step, Xv, y, R, Val(Nw))
end
@inline function _nrm2_lanes(
    X_dX::Lifted{P,Nw}, _n, step, Xv, y, ::Type{R}
) where {T,P<:Ptr{T},Nw,R}
    return _nrm2_lanes_perlane(_n, X_dX, step, Xv, y, R, Val(Nw))
end

# nrm2 — output is real (real or complex T); per-lane dy is real.
function frule!!(
    ::Lifted{typeof(BLAS.nrm2),Nw},
    n::Lifted,
    X_dX::Lifted{<:Union{Ptr{T},AbstractArray{T}}},
    incx::Lifted,
) where {Nw,T<:BlasFloat}
    _n = primal(n)
    _inc = primal(incx)
    Xp = primal(X_dX)
    # Lane paths use logical indices, so refuse a raw walk over other elements.
    # Unlike dotc/dotu, nrm2 has no per-lane BLAS fallback.
    step = _blas_walk_step(Xp, _inc, _n)
    if step === nothing
        _throw_no_walk_step(Xp, _inc)
    end
    y = BLAS.nrm2(_n, Xp, _inc)
    Xv = _viewify_one(_n, Xp, step)  # `viewify`-equivalent on the primal side.
    R = typeof(y)  # nrm2 returns the real-valued norm.
    return Lifted{R,Nw}(y, _scalar_ndual(y, _nrm2_lanes(X_dX, _n, step, Xv, y, R)))
end
# Reuse the primal's logical step for its partial: their index spaces match,
# but their strides can differ.
@inline _viewify_one(n::Integer, x::AbstractArray, step::Integer) = view(
    x, 1:step:(1 + (n - 1) * step)
)
@inline _viewify_one(n::Integer, x::Ptr{T}, step::Integer) where {T} = view(
    unsafe_wrap(Vector{T}, x, 1 + (n - 1) * step), 1:step:(1 + (n - 1) * step)
)

# Pair walk validation with computation so invalid layouts get the diagnostic,
# rather than passing nothing to _viewify_one.
@inline function _checked_walk_step(label, n::Integer, x, inc::Integer)
    step = _blas_walk_step(x, inc, n)
    step === nothing && _throw_no_walk_step(x, inc)
    return step
end

# A rule that skipped `_checked_walk_step`, not a user error.
@inline _viewify_one(::Integer, @nospecialize(x), ::Nothing) = error(
    "internal: `_viewify_one` needs a checked walk step; obtain one from `_checked_walk_step`",
)
function rrule!!(
    ::CoDual{typeof(BLAS.nrm2)},
    n::CoDual{<:Integer},
    X_dX::CoDual{<:Union{Ptr{T},AbstractArray{T}} where {T<:BlasFloat}},
    incx::CoDual{<:Integer},
)
    y = BLAS.nrm2(primal(n), primal(X_dX), primal(incx))
    X, dX = viewify(primal(n), X_dX, primal(incx))
    function nrm2_pb!!(dy)
        # Choose the zero subgradient at the zero vector to avoid division by zero.
        # `dy / y` and `X .* (dy / y)` can over- or underflow at extreme finite inputs
        # although the derivative is representable; a range-safe form needs scaled
        # accumulation.
        iszero(y) || (dX .+= _rvs_zero.(X .* (dy / y), iszero(dy)))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, NoFData()), nrm2_pb!!
end

# Intercept concrete real vectors before Julia inlines dot into its raw-pointer
# ccall. Forward-over-reverse cannot address dense lanes in the element-major block;
# array-level access preserves HVP/Hessian at every width. Concrete Vector also
# keeps this disjoint from CUDA's dot rule; strided wrappers retain the pointer guard.
@is_primitive(MinimalCtx, Tuple{typeof(dot),Vector{P},Vector{P}} where {P<:BlasRealFloat})
function frule!!(
    ::Lifted{typeof(dot),Nw}, x_dx::Lifted{Vector{P}}, y_dy::Lifted{Vector{P}}
) where {Nw,P<:BlasRealFloat}
    x, y = primal(x_dx), primal(y_dy)
    result = dot(x, y)
    # Two wide gemv! calls contract contiguous blocks (~3.5× faster than lane dots).
    # The length-Nw output allocation is permitted here (no :allocs guard).
    Xb = getfield(tangent(x_dx), :partials_block)
    Yb = getfield(tangent(y_dy), :partials_block)
    # Empty gemv skips even β scaling, so empty dot needs explicitly zeroed lanes.
    out = zeros(P, Nw)
    BLAS.gemv!('N', one(P), Xb, y, zero(P), out)
    BLAS.gemv!('N', one(P), Yb, x, one(P), out)
    dresult_lanes = ntuple(k -> out[k], Val(Nw))
    return Lifted{P,Nw}(result, _scalar_ndual(result, dresult_lanes))
end
function rrule!!(
    ::CoDual{typeof(dot)}, x_dx::CoDual{Vector{P}}, y_dy::CoDual{Vector{P}}
) where {P<:BlasRealFloat}
    x, dx = arrayify(x_dx)
    y, dy = arrayify(y_dy)
    result = dot(x, y)
    function dot_pb!!(dv)
        dx .+= y .* dv
        dy .+= x .* dv
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(result, NoFData()), dot_pb!!
end

# Forward complex dotc/dotu need an array-level primitive: cblas writes a Ref whose
# Complex{NDual} dual is incompatible with contiguous per-lane scalar cells. Reverse
# still uses the foreigncall rule. Dense operands with positive increments use the
# block loop; other layouts rebuild each lane over contiguous memory and call BLAS,
# preserving its raw pointer walk at the cost of O(Nw) calls and materialisations.
# Unit first-dimension stride alone does not imply density (e.g. a matrix subview).
# Negative increments start at (-n+1)*inc + 1 and also require the fallback.
# _blas_walk_step maps a raw increment to logical indices: dense arrays step by inc,
# strided vectors by inc ÷ stride. If stride does not divide inc, BLAS reads elements
# outside the operand; above one dimension only dense layouts admit a logical step.
@inline function _blas_walk_step(x, inc::Integer, n::Integer)
    inc > 0 || return nothing
    x isa Ptr && return inc
    n <= 1 && return n <= length(x) ? 1 : nothing
    step = if x isa AbstractVector
        st = stride(x, 1)
        (st > 0 && iszero(inc % st)) ? inc ÷ st : nothing
    else
        strides(x) === Base.size_to_strides(1, size(x)...) ? inc : nothing
    end
    step === nothing && return nothing
    return 1 + (n - 1) * step <= length(x) ? step : nothing
end

# Raised where the walk runs off the end of the operand, as distinct from following it in a
# different order -- the two need different messages because the fixes differ.
@noinline function _throw_walk_past_operand(label, x, inc, n)
    throw(
        ArgumentError(
            LazyString(
                label,
                " reads ",
                n,
                " elements from `pointer(X)` with `incx = ",
                inc,
                "`, which runs past the ",
                length(x),
                " elements of `",
                typeof(x),
                "`. The primal may still be defined, since the walk can stay inside a larger " *
                "parent array, but those elements belong to no argument and have no derivative. " *
                "Pass an operand at least `1 + (n-1)*incx` long.",
            ),
        ),
    )
end

# Both dot paths need this span: the fallback rebuilds only length(x) elements.
# Negative increments walk the same span backwards and are valid in the fallback.
@inline _blas_walk_inbounds(x, inc::Integer, n::Integer) =
    1 + (n - 1) * abs(inc) <= length(x)

# Return the primal span, its partials slot, and the partials' leading offset.
# A raw BLAS walk can leave a vector view while staying in its dense parent. Widen
# from the view's first element (same pointer) to include those parent partials,
# matching the reverse pointer rule. Other operands must contain their own walk.
@inline function _dot_walk_widen(label, slot::Lifted, inc::Integer, n::Integer)
    x = primal(slot)
    _blas_walk_inbounds(x, inc, n) || _throw_walk_past_operand(label, x, inc, n)
    return x, slot, 0
end
@inline function _dot_walk_widen(
    label,
    slot::Lifted{<:SubArray{T,1,Vector{T},<:Tuple{AbstractRange}},Nw},
    inc::Integer,
    n::Integer,
) where {T,Nw}
    x = primal(slot)
    p = parent(x)
    a = first(parentindices(x)[1])
    a + (n - 1) * abs(inc) <= length(p) || _throw_walk_past_operand(label, x, inc, n)
    return view(p, a:length(p)), Lifted{typeof(p),Nw}(p, tangent(slot).fields.parent), a - 1
end

@inline _dot_lane_span(a, drop::Integer) = view(a, (firstindex(a) + drop):lastindex(a))

# Cold path for a walk that runs past an operand. Per-lane BLAS calls, as the non-contiguous branch
# does: the widened operand is rare and does not need the block fast path.
@noinline function _dot_widened_acc(
    f::F, ::Val{Nw}, ::Type{E}, label, _DX, _DY, n, incx, incy
) where {F,Nw,E}
    pX, sX, aX = _dot_walk_widen(label, _DX, incx, n)
    pY, sY, aY = _dot_walk_widen(label, _DY, incy, n)
    return ntuple(Val(Nw)) do k
        dX = _dot_lane_span(_dense_lane_partial(sX, k), aX)
        dY = _dot_lane_span(_dense_lane_partial(sY, k), aY)
        f(n, dX, incx, pY, incy) + f(n, pX, incx, dY, incy)
    end::NTuple{Nw,E}
end

@inline function _blas_raw_walk_matches(x, inc::Integer)
    inc > 0 || return false
    x isa Ptr && return true
    return strides(x) === Base.size_to_strides(1, size(x)...)
end

# Pass acc by value to avoid boxing a reassigned closure capture (see _nrm2_accum).
# Share by conjugation, independent of eltype; bind only Nw for the empty tuple (Aqua).
@inline _dotc_accum(acc::NTuple{Nw}, xc, yc, x, y) where {Nw} = ntuple(
    k -> acc[k] + conj(xc[k]) * y + conj(x) * yc[k], Val(Nw)
)
@inline _dotu_accum(acc::NTuple{Nw}, xc, yc, x, y) where {Nw} = ntuple(
    k -> acc[k] + xc[k] * y + x * yc[k], Val(Nw)
)

for (jlfname, elty) in
    ((:dotc, :ComplexF64), (:dotc, :ComplexF32), (:dotu, :ComplexF64), (:dotu, :ComplexF32))
    # Independent X/Y allow mixed array wrappers, matching the frule's coverage.
    @eval @is_primitive(
        MinimalCtx,
        ForwardMode,
        Tuple{
            typeof(BLAS.$jlfname),Integer,X,Integer,Y,Integer
        } where {
            X<:Union{Ptr{$elty},AbstractArray{$elty}},
            Y<:Union{Ptr{$elty},AbstractArray{$elty}},
        }
    )
    # `dotc` conjugates its first argument, `dotu` neither; the JVP is linear either way:
    # d⟨x,y⟩ = ⟨dx,y⟩ + ⟨x,dy⟩.
    accum = jlfname == :dotc ? :_dotc_accum : :_dotu_accum
    @eval @inline function frule!!(
        ::Lifted{typeof(BLAS.$jlfname),Nw},
        _n::Lifted{<:Integer},
        _DX::Lifted{<:AbstractArray{$elty}},
        _incx::Lifted{<:Integer},
        _DY::Lifted{<:AbstractArray{$elty}},
        _incy::Lifted{<:Integer},
    ) where {Nw}
        n, incx, incy = primal(_n), primal(_incx), primal(_incy)
        DX, DY = primal(_DX), primal(_DY)
        result = BLAS.$jlfname(n, DX, incx, DY, incy)
        acc = if !(_blas_walk_inbounds(DX, incx, n) && _blas_walk_inbounds(DY, incy, n))
            # Widen to the parent partials before either path can read past the view.
            _dot_widened_acc(
                BLAS.$jlfname,
                Val(Nw),
                $elty,
                "Forward-mode `BLAS.$($(QuoteNode(jlfname)))`",
                _DX,
                _DY,
                n,
                incx,
                incy,
            )
        elseif _blas_raw_walk_matches(DX, incx) && _blas_raw_walk_matches(DY, incy)
            # Contiguous operands: logical indexing == BLAS's raw walk, so accumulate all lanes
            # in one pass over the element-major block columns.
            Xb, _ = _partials_block(_DX)
            Yb, _ = _partials_block(_DY)
            Xc = reinterpret(reshape, NTuple{Nw,$elty}, Xb)
            Yc = reinterpret(reshape, NTuple{Nw,$elty}, Yb)
            a = ntuple(_ -> zero($elty), Val(Nw))
            @inbounds for t in 1:n
                ix, iy = 1 + (t - 1) * incx, 1 + (t - 1) * incy
                a = $accum(a, Xc[ix], Yc[iy], DX[ix], DY[iy])
            end
            a
        else
            # Dense lane rebuilds preserve BLAS's raw walk for non-contiguous operands.
            ntuple(Val(Nw)) do k
                dX = _dense_lane_partial(_DX, k)
                dY = _dense_lane_partial(_DY, k)
                BLAS.$jlfname(n, dX, incx, DY, incy) + BLAS.$jlfname(n, DX, incx, dY, incy)
            end
        end
        return Lifted{$elty,Nw}(result, _scalar_ndual(result, acc))
    end
    # Ptr lanes are dense by protocol; array lanes are stride-Nw, so mixed calls
    # require width 1. All-array calls use the block method above.
    @eval @inline function frule!!(
        ::Lifted{typeof(BLAS.$jlfname),Nw},
        _n::Lifted{<:Integer},
        _DX::Lifted{<:Union{Ptr{$elty},AbstractArray{$elty}}},
        _incx::Lifted{<:Integer},
        _DY::Lifted{<:Union{Ptr{$elty},AbstractArray{$elty}}},
        _incy::Lifted{<:Integer},
    ) where {Nw}
        if Nw > 1 && !(primal(_DX) isa Ptr && primal(_DY) isa Ptr)
            throw(
                ArgumentError(
                    "`BLAS.$($(QuoteNode(jlfname)))` with mixed raw-pointer and array " *
                    "arguments is unsupported at chunk width $Nw > 1: an array slot's " *
                    "per-lane partials are lane-strided block views that the " *
                    "pointer-based BLAS wrapper cannot read. Differentiate at chunk " *
                    "width 1 (or pass both arguments the same way).",
                ),
            )
        end
        n, incx, incy = primal(_n), primal(_incx), primal(_incy)
        DX, DY = primal(_DX), primal(_DY)
        result = BLAS.$jlfname(n, DX, incx, DY, incy)
        dresult_lanes = ntuple(Val(Nw)) do lane
            dX = _blas_lane_partial(_DX, lane)
            dY = _blas_lane_partial(_DY, lane)
            return BLAS.$jlfname(n, dX, incx, DY, incy) +
                   BLAS.$jlfname(n, DX, incx, dY, incy)
        end
        return Lifted{$elty,Nw}(result, _scalar_ndual(result, dresult_lanes))
    end
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.scal!),Integer,P,X,Integer
    } where {P<:BlasFloat,X<:Union{Ptr{P},AbstractArray{P}}}
)
function frule!!(
    ::Lifted{typeof(BLAS.scal!),Nw},
    _n::Lifted,
    a_da::Lifted{P,Nw},
    X_dX::Lifted{<:AbstractArray{P}},
    _incx::Lifted,
) where {Nw,P<:BlasFloat}
    n = primal(_n)
    incx = primal(_incx)
    a = primal(a_da)
    X = primal(X_dX)
    # Partial blocks follow logical indices, whereas BLAS walks raw memory. Unlike
    # dotc/dotu, this mutating rule cannot use a copied dense-lane fallback.
    step = _blas_walk_step(X, incx, n)
    if step === nothing
        _throw_no_walk_step(X, incx)
    end
    das = ntuple(k -> tangent(a_da, k), Val(Nw))
    Xb, copied = _partials_block(X_dX)
    Xbm = reshape(Xb, Nw, :)
    # Per-lane Frechet dX_k := a·dX_k + da_k·X, all lanes in one pass: each touched
    # element's lanes are one contiguous block column.
    @inbounds for t in 1:n
        i = 1 + (t - 1) * step
        xi = X[i]
        for k in 1:Nw
            Xbm[k, i] = a * Xbm[k, i] + das[k] * xi
        end
    end
    copied && _write_back_partials!(X_dX, Xb)
    BLAS.scal!(n, a, X, incx)
    return X_dX
end
# Raw-pointer path: per-lane tangent pointers are dense buffers by the `Ptr` dual
# protocol, so the per-lane BLAS calls read them correctly at any width.
function frule!!(
    ::Lifted{typeof(BLAS.scal!),Nw},
    _n::Lifted,
    a_da::Lifted{P,Nw},
    X_dX::Lifted{Ptr{P},Nw},
    _incx::Lifted,
) where {Nw,P<:BlasFloat}
    n = primal(_n)
    incx = primal(_incx)
    a = primal(a_da)
    X = primal(X_dX)
    # Per-lane Frechet: dX_lane := a * dX_lane + da_lane * X.
    for lane in 1:Nw
        dX_lane = _blas_lane_partial(X_dX, lane)
        da_lane = tangent(a_da, lane)
        BLAS.scal!(n, a, dX_lane, incx)
        BLAS.axpy!(n, da_lane, X, incx, dX_lane, incx)
    end
    BLAS.scal!(n, a, X, incx)
    return X_dX
end
function rrule!!(
    ::CoDual{typeof(BLAS.scal!)},
    _n::CoDual{<:Integer},
    a_da::CoDual{P},
    X_dX::CoDual{<:Union{Ptr{P},AbstractArray{P}}},
    _incx::CoDual{<:Integer},
) where {P<:BlasFloat}

    # Extract params.
    n = primal(_n)
    incx = primal(_incx)
    a = primal(a_da)
    X, dX = viewify(n, X_dX, incx)

    # Take a copy of previous state in order to recover it on the reverse pass.
    X_copy = copy(X)

    # Run primal computation.
    BLAS.scal!(n, a, primal(X_dX), incx)

    function scal_adjoint(::NoRData)
        zero_seed = all(iszero, dX)
        ∇a = zero(P)
        @inbounds @simd for i in eachindex(X, X_copy, dX)
            X[i] = X_copy[i]
            ∇a += _rvs_mul(X_copy[i]', dX[i])
            P <: BlasRealFloat && (dX[i] = _rvs_zero(_rvs_mul(dX[i], a'), zero_seed))
        end
        # The real loop vectorises; complex scaling vectorises better as a separate broadcast.
        P <: BlasComplexFloat && (dX .= _rvs_zero.(_rvs_mul.(dX, a'), zero_seed))

        return NoRData(), NoRData(), ∇a, NoRData(), NoRData()
    end
    return X_dX, scal_adjoint
end

#
# LEVEL 2
#

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.gemv!),Char,P,AbstractVecOrMat{P},AbstractVector{P},P,AbstractVector{P}
    } where {P<:BlasFloat},
)

# Present a vector operand to the level-3 BLAS kernels as a single-column matrix (used by the
# gemv!/gemm! Lifted frules for both the primal arrays and the per-lane partials).
@inline _as_col(v) = v isa AbstractVector ? reshape(v, :, 1) : v

function frule!!(
    ::Lifted{typeof(BLAS.gemv!),Nw},
    tA::Lifted{Char},
    alpha::Lifted{P,Nw},
    A_dA::Lifted{<:AbstractVecOrMat{P}},
    x_dx::Lifted{<:AbstractVector{P}},
    beta::Lifted{P,Nw},
    y_dy::Lifted{<:AbstractVector{P}},
) where {Nw,P<:BlasFloat}
    _check_blas_output_alias(BLAS.gemv!, primal(y_dy), primal(A_dA), primal(x_dx))
    _tA = _lsame_flag(primal(tA))
    α = primal(alpha)
    β = primal(beta)
    A = _as_col(primal(A_dA))
    x = primal(x_dx)
    y = primal(y_dy)
    dαs = ntuple(k -> tangent(alpha, k), Val(Nw))
    dβs = ntuple(k -> tangent(beta, k), Val(Nw))
    Ab, _ = _partials_block(A_dA)
    Xb, _ = _partials_block(x_dx)
    Yb, ycopied = _partials_block(y_dy)
    M, K = length(y), length(x)
    Xbm, Ybm = reshape(Xb, Nw, K), reshape(Yb, Nw, M)
    # 1) β·dy + α·op(A)·dx: lane `k` is row `k` of the lane matrices, so per-lane
    #    `op(A)·dx_k` is `Xbm·op(A)ᵀ` — one wide gemm over the block, β folded in (applied
    #    exactly once, first; all later terms accumulate). An all-zero `Xbm` means `x` is
    #    constant data — the product term vanishes, leaving the β scaling.
    if !iszero(Xbm)
        if _tA == 'N'
            BLAS.gemm!('N', 'T', α, Xbm, A, β, Ybm)
        elseif _tA == 'T' || P <: BlasRealFloat
            BLAS.gemm!('N', 'N', α, Xbm, A, β, Ybm)
        else
            # Complex 'C': op(A)ᵀ = conj(A), which gemm cannot express on its right
            # operand; the vector-arg wrappers read strided lanes natively, so run the
            # adjoint per lane instead of materialising conj(A).
            _scale_or_zero!(Ybm, β, K)
            for k in 1:Nw
                BLAS.gemv!('C', α, A, view(Xbm, k, :), one(P), view(Ybm, k, :))
            end
        end
    else
        _scale_or_zero!(Ybm, β, K)
    end
    # 2) α·op(dA)·x — skipped when `A` is constant data (all-zero block).
    if !iszero(Ab)
        Abm = reshape(Ab, Nw, size(A)...)
        if _tA == 'N'
            # Contract dA's last axis with x: the (Nw·M, K) flat view of dA's block times
            # x lands lane-major — exactly the flat view of Yb. One wide gemv.
            BLAS.gemv!('N', α, reshape(Abm, Nw * M, K), x, one(P), reshape(Ybm, Nw * M))
        elseif _tA == 'T' || P <: BlasRealFloat
            # Column slab i of dA's block is a contiguous (Nw, K) matrix; `op(dA)·x`
            # lands in Yb's contiguous block column i.
            for i in 1:M
                BLAS.gemv!('N', α, view(Abm,:,:,i), x, one(P), view(Ybm, :, i))
            end
        else
            # Complex 'C': Σⱼ conj(dA[k,j,i])·x[j] = conj(slab_i · conj(x)).
            xc = conj(x)
            # Zeroed for the same reason as in the `dot` frule above: with an empty inner
            # dimension `gemv` skips the write, and the stale buffer would be accumulated.
            wN = zeros(P, Nw)
            for i in 1:M
                BLAS.gemv!('N', one(P), view(Abm,:,:,i), xc, zero(P), wN)
                view(Ybm, :, i) .+= α .* conj.(wN)
            end
        end
    end
    # 3) dα·op(A)·x per seeded lane, gemv straight into the strided lane row (the
    #    vector-arg wrappers pass strides through); usually 0 or 1 such lane.
    for k in 1:Nw
        iszero(dαs[k]) || BLAS.gemv!(_tA, dαs[k], A, x, one(P), view(Ybm, k, :))
    end
    # 4) dβ·y over the original `y`. Strong zero on NaN `y` entries, as in reverse mode:
    #    `y` may hold undefined values wherever `β == 0` discards them. Skipped along with the
    #    `β·dy` term above when the contracted dimension is zero: BLAS quick-returns there without
    #    applying `β`, so the primal does not depend on `β` at all and neither may the tangent.
    if K != 0 && !all(iszero, dβs)
        @inbounds for i in 1:M
            yi = y[i]
            isnan(yi) && continue
            for k in 1:Nw
                Ybm[k, i] += dβs[k] * yi
            end
        end
    end
    ycopied && _write_back_partials!(y_dy, Yb)
    # 5) Primal update AFTER all tangent terms, so every lane's `dβ·y` read the original
    #    `y` and the wide product used the original operands.
    BLAS.gemv!(_tA, α, A, x, β, y)
    return y_dy
end

@inline function rrule!!(
    ::CoDual{typeof(BLAS.gemv!)},
    _tA::CoDual{Char},
    _alpha::CoDual{P},
    _A::CoDual{<:AbstractVecOrMat{P}},
    _x::CoDual{<:AbstractVector{P}},
    _beta::CoDual{P},
    _y::CoDual{<:AbstractVector{P}},
) where {P<:BlasFloat}
    _check_blas_output_alias(BLAS.gemv!, primal(_y), primal(_A), primal(_x))

    # Pull out primals and tangents (the latter only where necessary).
    trans = primal(_tA)
    alpha = _alpha.x
    A, dA = matrixify(_A)
    x, dx = arrayify(_x)
    beta = _beta.x
    y, dy = arrayify(_y)

    pb = _gemv!_rrule_core!(trans, alpha, A, dA, x, dx, beta, y, dy)

    return _y, pb
end

@inline function _gemv!_rrule_core!(
    trans::Char,
    alpha::P,
    A::AbstractMatrix{P},
    dA::AbstractMatrix{P},
    x::AbstractVector{P},
    dx::AbstractVector{P},
    beta::P,
    y::AbstractVector{P},
    dy::AbstractVector{P},
) where {P<:BlasFloat}

    # Take copies before adding.
    y_copy = copy(y)

    # Run primal.
    BLAS.gemv!(trans, alpha, A, x, beta, y)

    function gemv!_pb!!(::NoRData)
        if trans == 'n' && stride(A, 2) < 0
            return _gemv!_pullback(
                'N',
                alpha,
                A,
                dA,
                view(x, length(x):-1:1),
                view(dx, length(dx):-1:1),
                beta,
                view(y, length(y):-1:1),
                view(dy, length(dy):-1:1),
                view(y_copy, length(y_copy):-1:1),
            )
        end
        return _gemv!_pullback(uppercase(trans), alpha, A, dA, x, dx, beta, y, dy, y_copy)
    end
    return gemv!_pb!!
end

@inline function _gemv!_pullback(
    trans, alpha::P, A, dA, x, dx, beta, y, dy, y_copy
) where {P<:BlasFloat}

    # An empty contracted dimension leaves `y` unchanged, independent of the coefficients.
    if isempty(x)
        copyto!(y, y_copy)
        return (NoRData(), NoRData(), zero(P), NoRData(), NoRData(), zero(P), NoRData())
    end

    conjdy = trans == 'T' && P <: BlasComplexFloat ? conj.(dy) : dy
    dalpha = if length(x) == length(y)
        BLAS.gemv!(trans == 'N' ? 'C' : 'N', one(P), A, conjdy, zero(P), y)
        d = _rvs_guarded_dot(x, y, trans == 'T' && P <: BlasComplexFloat)
        isnan(d) ? _rvs_guarded_dot(_trans(trans, A), x, dy) : d
    else
        _rvs_guarded_dot(_trans(trans, A), x, dy)
    end

    # Increment fdata.
    zero_seed = all(iszero, dy)
    if trans == 'N'
        _rvs_muladd!(dA, dy, x, alpha', 'N', 'C', true, true)
        _rvs_blas!(BLAS.gemv!, zero_seed, 'C', alpha', A, dy, one(eltype(A)), dx)
    elseif trans == 'C' || P <: BlasRealFloat
        _rvs_muladd!(dA, x, dy, alpha, 'N', 'C', true, true)
        _rvs_blas!(BLAS.gemv!, zero_seed, 'N', alpha', A, dy, one(eltype(A)), dx)
    else
        _rvs_muladd!(dA, transpose(x), dy, alpha', 'C', 'T', true, true)
        # Should be gemv!("conjugate only", alpha', A, dy, one(eltype(A)), dx)
        # but BLAS has no "conjugate only" gemv
        conj!(dx)
        _rvs_blas!(BLAS.gemv!, zero_seed, 'N', alpha, A, conjdy, one(eltype(A)), dx)
        conj!(dx)
    end
    dbeta = _rvs_guarded_dot(y_copy, dy)
    dy .= _rvs_zero.(_rvs_mul.(dy, beta'), zero_seed)

    # Restore primal.
    copyto!(y, y_copy)

    # Return rdata.
    return (NoRData(), NoRData(), dalpha, NoRData(), NoRData(), dbeta, NoRData())
end

# Note that the complex symv are not BLAS but auxiliary functions in LAPACK
for (fname, elty) in ((:(symv!), BlasFloat), (:(hemv!), BlasComplexFloat))
    isherm = fname == :(hemv!)

    @eval @is_primitive(
        MinimalCtx,
        Tuple{
            typeof(BLAS.$fname),
            Char,
            T,
            AbstractMatrix{T},
            AbstractVector{T},
            T,
            AbstractVector{T},
        } where {T<:$elty},
    )

    @eval function frule!!(
        ::Lifted{typeof(BLAS.$fname),Nw},
        uplo::Lifted{Char},
        alpha::Lifted{T,Nw},
        A_dA::Lifted{<:AbstractMatrix{T}},
        x_dx::Lifted{<:AbstractVector{T}},
        beta::Lifted{T,Nw},
        y_dy::Lifted{<:AbstractVector{T}},
    ) where {Nw,T<:$elty}
        _check_blas_output_alias(BLAS.$fname, primal(y_dy), primal(A_dA), primal(x_dx))
        ul = _lsame_flag(primal(uplo))
        α = primal(alpha)
        β = primal(beta)
        A = primal(A_dA)
        x = primal(x_dx)
        y = primal(y_dy)
        dαs = ntuple(k -> tangent(alpha, k), Val(Nw))
        dβs = ntuple(k -> tangent(beta, k), Val(Nw))
        Ab, _ = _partials_block(A_dA)
        Xb, _ = _partials_block(x_dx)
        Yb, ycopied = _partials_block(y_dy)
        n = length(x)
        Xbm, Ybm = reshape(Xb, Nw, n), reshape(Yb, Nw, n)
        # 1) β·dy + α·A·dx, β folded in (applied exactly once, first). For the symmetric
        #    case Aᵀ = A, so per-lane `A·dx_k` is one wide side-'R' symm over the lane
        #    matrix (reading only the `ul` triangle, like the primal). The hermitian
        #    Aᵀ = conj(A) has no wide form on the right; the vector-arg wrapper reads
        #    strided lanes natively, so run hemv per lane (β folded per lane).
        if !iszero(Xbm)
            $(
                if isherm
                    quote
                        for k in 1:Nw
                            BLAS.hemv!(ul, α, A, view(Xbm, k, :), β, view(Ybm, k, :))
                        end
                    end
                else
                    :(BLAS.symm!('R', ul, α, A, Xbm, β, Ybm))
                end
            )
        else
            _scale_or_zero!(Ybm, β)
        end
        # 2) α·dA·x — skipped when `A` is constant data. dA is symmetric/hermitian with
        #    only the `ul` triangle significant, exactly like `A`; gather each lane into a
        #    dense scratch and apply the same kernel into the strided lane row.
        if !iszero(Ab)
            Abm = reshape(Ab, Nw, n, n)
            Ascr = Matrix{T}(undef, n, n)
            for k in 1:Nw
                copyto!(Ascr, view(Abm,k,:,:))
                BLAS.$fname(ul, α, Ascr, x, one(T), view(Ybm, k, :))
            end
        end
        # 3) dα·A·x per seeded lane, into the strided lane row.
        for k in 1:Nw
            iszero(dαs[k]) || BLAS.$fname(ul, dαs[k], A, x, one(T), view(Ybm, k, :))
        end
        # 4) dβ·y over the original `y`; strong zero on NaN entries (see gemv!).
        if !all(iszero, dβs)
            @inbounds for i in 1:n
                yi = y[i]
                isnan(yi) && continue
                for k in 1:Nw
                    Ybm[k, i] += dβs[k] * yi
                end
            end
        end
        ycopied && _write_back_partials!(y_dy, Yb)
        # Primal hoisted after all tangent terms so every lane's `dβ·y` reads the
        # original `y`.
        BLAS.$fname(ul, α, A, x, β, y)
        return y_dy
    end

    @eval function rrule!!(
        ::CoDual{typeof(BLAS.$fname)},
        uplo::CoDual{Char},
        alpha::CoDual{T},
        A_dA::CoDual{<:AbstractMatrix{T}},
        x_dx::CoDual{<:AbstractVector{T}},
        beta::CoDual{T},
        y_dy::CoDual{<:AbstractVector{T}},
    ) where {T<:$elty}
        _check_blas_output_alias(BLAS.$fname, primal(y_dy), primal(A_dA), primal(x_dx))

        # Extract primals.
        ul = primal(uplo)
        α = primal(alpha)
        β = primal(beta)
        A, dA = arrayify(A_dA)
        x, dx = arrayify(x_dx)
        y, dy = arrayify(y_dy)

        y_copy = copy(y)

        fast = isone(α) && iszero(β)
        BLAS.$fname(ul, α, A, x, β, y)

        function symv!_or_hemv!_adjoint(::NoRData)
            conjdy = T <: BlasRealFloat || $isherm ? dy : conj.(dy)
            dα = if fast
                _rvs_guarded_dot(y, dy)
            elseif all(!iszero, dy)
                BLAS.$fname(ul, one(T), A, conjdy, zero(T), y)
                _rvs_guarded_dot(x, y, T <: BlasComplexFloat && !$isherm)
            else
                _rvs_guarded_dot(
                    $(isherm ? Hermitian : Symmetric)(A, ul == 'U' ? :U : :L), x, dy
                )
            end
            BLAS.copyto!(y, y_copy)

            # gradient w.r.t. A.
            # TODO: could be switched to BLAS.{sy,he}r2! if Julia ever provides it.
            dA_tmp = _rvs_muladd!(similar(dA), dy, x, α', 'N', 'C', false, true)
            if ul == 'L'
                dA .=
                    (dA .+ LowerTriangular(dA_tmp)) .+
                    $(isherm ? adjoint : transpose)(UpperTriangular(dA_tmp))
            else
                dA .=
                    (dA .+ $(isherm ? adjoint : transpose)(LowerTriangular(dA_tmp))) .+
                    UpperTriangular(dA_tmp)
            end
            @inbounds for n in diagind(dA)
                dA[n] -= $(isherm ? :(real(dA_tmp[n])) : :(dA_tmp[n]))
            end

            # gradient w.r.t. x: dx += α' A' dy
            zero_seed = all(iszero, dy)
            if T <: BlasRealFloat || $isherm
                # A' = A for real numbers or for hermitian matrices
                _rvs_blas!(BLAS.$fname, zero_seed, ul, α', A, dy, one(T), dx)
            else
                # A is symmetric but complex so A' = conj(A)
                # Instead we compute conj(dx) += α A conj(dy)
                conj!(dx)
                _rvs_blas!(BLAS.$fname, zero_seed, ul, α, A, conjdy, one(T), dx)
                conj!(dx)
            end

            # gradient w.r.t. beta.
            dβ = _rvs_guarded_dot(y, dy)
            fast && (dα -= _rvs_mul(dβ, β') + _rvs_mul(dα, α' - one(T)))

            # gradient w.r.t. y.
            dy .= _rvs_zero.(_rvs_mul.(dy, β'), zero_seed)

            return (NoRData(), NoRData(), dα, NoRData(), NoRData(), dβ, NoRData())
        end
        return y_dy, symv!_or_hemv!_adjoint
    end
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.trmv!),Char,Char,Char,AbstractMatrix{T},AbstractVector{T}
    } where {T<:BlasFloat},
)

function frule!!(
    ::Lifted{typeof(BLAS.trmv!),Nw},
    _uplo::Lifted{Char},
    _trans::Lifted{Char},
    _diag::Lifted{Char},
    A_dA::Lifted{<:AbstractMatrix{T}},
    x_dx::Lifted{<:AbstractVector{T}},
) where {Nw,T<:BlasFloat}
    _check_blas_output_alias(BLAS.trmv!, primal(x_dx), primal(A_dA))
    uplo = _lsame_flag(primal(_uplo))
    trans = _lsame_flag(primal(_trans))
    diag = _lsame_flag(primal(_diag))
    A = primal(A_dA)
    x = primal(x_dx)
    Ab, _ = _partials_block(A_dA)
    Xb, xcopied = _partials_block(x_dx)
    n = length(x)
    Xbm = reshape(Xb, Nw, n)
    # Frechet: dx_k := op(A)·dx_k + op(dA_k)·x (+ unit-diag adjustment).
    # 1) op(A)·dx_k for all lanes: right-multiply the lane matrix by op(A)ᵀ — one wide
    #    trmm over the block. Complex 'C' (op(A)ᵀ = conj(A), inexpressible on the right)
    #    runs trmv per lane instead: the vector-arg wrapper reads strided lanes natively.
    if trans == 'N'
        BLAS.trmm!('R', uplo, 'T', diag, one(T), A, Xbm)
    elseif trans == 'T' || T <: BlasRealFloat
        BLAS.trmm!('R', uplo, 'N', diag, one(T), A, Xbm)
    else
        for k in 1:Nw
            BLAS.trmv!(uplo, 'C', diag, A, view(Xbm, k, :))
        end
    end
    # 2) op(dA_k)·x — skipped when `A` is constant data. trmv masks dA's triangle (and
    #    implicit unit diagonal, whose derivative the `diag == 'U'` correction removes).
    if !iszero(Ab)
        Abm = reshape(Ab, Nw, n, n)
        Ascr = Matrix{T}(undef, n, n)
        tmp = similar(x, n)
        for k in 1:Nw
            copyto!(Ascr, view(Abm,k,:,:))
            copyto!(tmp, x)
            BLAS.trmv!(uplo, trans, diag, Ascr, tmp)
            diag === 'U' && (tmp .-= x)
            view(Xbm, k, :) .+= tmp
        end
    end
    xcopied && _write_back_partials!(x_dx, Xb)
    BLAS.trmv!(uplo, trans, diag, A, x)
    return x_dx
end
function rrule!!(
    ::CoDual{typeof(BLAS.trmv!)},
    _uplo::CoDual{Char},
    _trans::CoDual{Char},
    _diag::CoDual{Char},
    A_dA::CoDual{<:AbstractMatrix{T}},
    x_dx::CoDual{<:AbstractVector{T}},
) where {T<:BlasFloat}
    _check_blas_output_alias(BLAS.trmv!, primal(x_dx), primal(A_dA))

    # Extract primals.
    uplo = primal(_uplo)
    trans = uppercase(primal(_trans))
    diag = uppercase(primal(_diag))
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)
    x_copy = copy(x)

    # Run primal computation.
    BLAS.trmv!(uplo, primal(_trans), primal(_diag), A, x)

    # Set dx to zero.
    dx .= zero(T)

    function trmv_pb!!(::NoRData)

        # Restore the original value of x.
        x .= x_copy

        # Increment the tangents.
        zero_seed = all(iszero, dx)
        if trans == 'N'
            inc_tri!(dA, dx, x, uplo, diag)
            _rvs_blas!(BLAS.trmv!, zero_seed, uplo, 'C', diag, A, dx)
        elseif trans == 'C' || T <: BlasRealFloat
            inc_tri!(dA, x, dx, uplo, diag)
            _rvs_blas!(BLAS.trmv!, zero_seed, uplo, 'N', diag, A, dx)
        else
            # Equivalent to these two calls:
            # inc_tri!(dA, conj.(x), conj.(dx), uplo, diag)
            # BLAS.trmv!(uplo, "conjugate only", diag, A, dx)

            conj!(x_copy) # Reuse the memory, we don't need it anymore
            conj!(dx)
            inc_tri!(dA, x_copy, dx, uplo, diag)
            _rvs_blas!(BLAS.trmv!, zero_seed, uplo, 'N', diag, A, dx)
            conj!(dx)
        end

        return tuple_fill(NoRData(), Val(6))
    end
    return x_dx, trmv_pb!!
end

function inc_tri!(A, x, y, uplo, diag)
    z = all(iszero, x) || all(iszero, y)
    if uplo == 'L' && diag == 'U'
        @inbounds for q in 1:size(A, 2), p in (q + 1):size(A, 1)
            A[p, q] += _rvs_zero(x[p] * y[q]', z)
        end
    elseif uplo == 'L' && diag == 'N'
        @inbounds for q in 1:size(A, 2), p in q:size(A, 1)
            A[p, q] += _rvs_zero(x[p] * y[q]', z)
        end
    elseif uplo == 'U' && diag == 'U'
        @inbounds for q in 1:size(A, 2), p in 1:(q - 1)
            A[p, q] += _rvs_zero(x[p] * y[q]', z)
        end
    elseif uplo == 'U' && diag == 'N'
        @inbounds for q in 1:size(A, 2), p in 1:q
            A[p, q] += _rvs_zero(x[p] * y[q]', z)
        end
    else
        error("Unexpected uplo $uplo or diag $diag")
    end
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.trsv!),Char,Char,Char,AbstractMatrix{T},AbstractVector{T}
    } where {T<:BlasFloat},
)
function frule!!(
    ::Lifted{typeof(BLAS.trsv!),Nw},
    _uplo::Lifted{Char},
    _trans::Lifted{Char},
    _diag::Lifted{Char},
    A_dA::Lifted{<:AbstractMatrix{T}},
    x_dx::Lifted{<:AbstractVector{T}},
) where {Nw,T<:BlasFloat}
    _check_blas_output_alias(BLAS.trsv!, primal(x_dx), primal(A_dA))
    uplo = _lsame_flag(primal(_uplo))
    trans = _lsame_flag(primal(_trans))
    diag = _lsame_flag(primal(_diag))
    A = primal(A_dA)
    x = primal(x_dx)
    # Primal first — subsequent lane work needs the solved `x`.
    BLAS.trsv!(uplo, trans, diag, A, x)
    Ab, _ = _partials_block(A_dA)
    Xb, xcopied = _partials_block(x_dx)
    n = length(x)
    Xbm = reshape(Xb, Nw, n)
    # d(op(A)⁻¹·x) = op(A)⁻¹·(dx − op(dA)·x). op(A)⁻¹ is linear, so the tangent takes one
    # solve of that combined RHS, not separate solves of `dx` and `op(dA)·x`.
    # 1) dx_k −= op(dA_k)·x — skipped when `A` is constant data.
    if !iszero(Ab)
        Abm = reshape(Ab, Nw, n, n)
        Ascr = Matrix{T}(undef, n, n)
        tmp = similar(x, n)
        for k in 1:Nw
            copyto!(Ascr, view(Abm,k,:,:))
            copyto!(tmp, x)
            BLAS.trmv!(uplo, trans, diag, Ascr, tmp)
            diag === 'U' && (tmp .-= x)
            view(Xbm, k, :) .-= tmp
        end
    end
    # 2) op(A)⁻¹ applied to every lane: right-divide the lane matrix by op(A)ᵀ — one wide
    #    trsm over the block. Complex 'C' runs trsv per lane on the strided lane vectors.
    if trans == 'N'
        BLAS.trsm!('R', uplo, 'T', diag, one(T), A, Xbm)
    elseif trans == 'T' || T <: BlasRealFloat
        BLAS.trsm!('R', uplo, 'N', diag, one(T), A, Xbm)
    else
        for k in 1:Nw
            BLAS.trsv!(uplo, 'C', diag, A, view(Xbm, k, :))
        end
    end
    xcopied && _write_back_partials!(x_dx, Xb)
    return x_dx
end
function rrule!!(
    ::CoDual{typeof(BLAS.trsv!)},
    _uplo::CoDual{Char},
    _trans::CoDual{Char},
    _diag::CoDual{Char},
    A_dA::CoDual{<:AbstractMatrix{T}},
    x_dx::CoDual{<:AbstractVector{T}},
) where {T<:BlasFloat}
    _check_blas_output_alias(BLAS.trsv!, primal(x_dx), primal(A_dA))
    uplo = primal(_uplo)
    trans = uppercase(primal(_trans))
    diag = uppercase(primal(_diag))
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)

    x_copy = copy(x)

    # Primal
    BLAS.trsv!(uplo, primal(_trans), primal(_diag), A, x)

    function trsv_pb!!(::NoRData)

        # Increment dA
        zero_seed = all(iszero, dx)
        if trans == 'N'
            temp = _rvs_blas!(BLAS.trsv!, zero_seed, uplo, 'C', diag, A, copy(dx))
            temp .*= -1
            inc_tri!(dA, temp, x, uplo, diag)
        elseif trans == 'C'
            temp = _rvs_blas!(BLAS.trsv!, zero_seed, uplo, 'N', diag, A, copy(dx))
            temp .*= -1
            inc_tri!(dA, x, temp, uplo, diag)
        else
            temp = _rvs_blas!(BLAS.trsv!, zero_seed, uplo, 'N', diag, A, conj!(copy(dx)))
            temp .*= -1
            inc_tri!(dA, conj!(x), temp, uplo, diag)
        end

        # Restore initial state
        x .= x_copy

        # Compute dx
        if trans == 'T'
            # Equivalent to trsv!(uplo, "conjugate only", diag, A, dx)
            conj!(dx)
            _rvs_blas!(BLAS.trsv!, zero_seed, uplo, 'N', diag, A, dx)
            conj!(dx)
        else
            _rvs_blas!(BLAS.trsv!, zero_seed, uplo, trans == 'N' ? 'C' : 'N', diag, A, dx)
        end

        return tuple_fill(NoRData(), Val(6))
    end

    return x_dx, trsv_pb!!
end

#
# LEVEL 3
#

# Keep primitive coverage in lockstep with both rules: A/B allow vectors, C does
# not. Declaring vector C primitive would prevent fallback and raise MethodError.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.gemm!),
        Char,
        Char,
        T,
        AbstractVecOrMat{T},
        AbstractVecOrMat{T},
        T,
        AbstractMatrix{T},
    } where {T<:BlasFloat},
)

function frule!!(
    ::Lifted{typeof(BLAS.gemm!),Nw},
    transA::Lifted{Char},
    transB::Lifted{Char},
    alpha::Lifted{T,Nw},
    A_dA::Lifted{<:AbstractVecOrMat{T}},
    B_dB::Lifted{<:AbstractVecOrMat{T}},
    beta::Lifted{T,Nw},
    C_dC::Lifted{<:AbstractMatrix{T}},
) where {Nw,T<:BlasFloat}
    _check_blas_output_alias(BLAS.gemm!, primal(C_dC), primal(A_dA), primal(B_dB))
    tA = _lsame_flag(primal(transA))
    tB = _lsame_flag(primal(transB))
    α = primal(alpha)
    β = primal(beta)
    A = _as_col(primal(A_dA))
    B = _as_col(primal(B_dB))
    C = primal(C_dC)
    dαs = ntuple(k -> tangent(alpha, k), Val(Nw))
    dβs = ntuple(k -> tangent(beta, k), Val(Nw))
    Ab_, _ = _partials_block(A_dA)
    Bb_, _ = _partials_block(B_dB)
    Cb, ccopied = _partials_block(C_dC)
    m, n = size(C)
    p = tA == 'N' ? size(A, 2) : size(A, 1)
    Ab = reshape(Ab_, Nw, size(A)...)
    Bb = reshape(Bb_, Nw, size(B)...)
    # Product rule: dC_k = β·dC_k + α·op(dA_k)·op(B) + α·op(A)·op(dB_k) + dα_k·op(A)·op(B)
    # + dβ_k·C. The two matrix-partial terms batch all lanes into wide BLAS calls over the
    # lane-leading blocks; terms whose operand is constant data (all-zero block) vanish.
    # 1) β·dC + α·op(A)·op(dB), β folded in (applied exactly once, first; every later
    #    term accumulates).
    if !iszero(Bb)
        if tB == 'C' && T <: BlasComplexFloat
            # α·op(A)·dB^H: the conj is lane-varying, so per output column j build the
            # conjugated product in a hoisted (Nw, m) scratch and conj-add:
            # t[k,i,j] = conj(conj(α)·Σ_l conj(op(A)[i,l])·dB[k,j,l]).
            fA, Ae = tA == 'N' ? ('C', A) : (tA == 'T' ? ('N', conj(A)) : ('N', A))
            W = Matrix{T}(undef, Nw, m)
            for j in 1:n
                BLAS.gemm!('N', fA, conj(α), view(Bb,:,j,:), Ae, zero(T), W)
                Cslab = view(Cb,:,:,j)
                if iszero(β)
                    Cslab .= conj.(W)
                else
                    Cslab .= β .* Cslab .+ conj.(W)
                end
            end
        else
            # Per output column j: dC slab j (Nw, m) := α·(dB slice j)·op(A)ᵀ + β·(slab j).
            # Slabs are contiguous and slices unit-stride in the lane axis, so both are
            # valid BLAS matrices.
            fA, Ae = if tA == 'N'
                ('T', A)
            elseif tA == 'T' || T <: BlasRealFloat
                ('N', A)
            else
                ('N', conj(A))
            end
            for j in 1:n
                Bslice = tB == 'N' ? view(Bb,:,:,j) : view(Bb,:,j,:)
                BLAS.gemm!('N', fA, α, Bslice, Ae, β, view(Cb,:,:,j))
            end
        end
    else
        _scale_or_zero!(Cb, β)
    end
    # 2) α·op(dA)·op(B).
    if !iszero(Ab)
        if tA == 'N'
            # One flat gemm: contracting dA's last axis with op(B), the (Nw·m, p) flat
            # view of dA's block times op(B) lands lane-major — the flat view of dC's
            # block. gemm applies tB (including 'C') to its right operand natively.
            BLAS.gemm!(
                'N', tB, α, reshape(Ab, Nw * m, p), B, one(T), reshape(Cb, Nw * m, n)
            )
        elseif tA == 'T' || T <: BlasRealFloat
            # Slab i of dA's block (contiguous (Nw, p) — column i of A, i.e. row i of
            # op(A)) times op(B) lands in dC's lane-unit-stride row slice i.
            for i in 1:m
                BLAS.gemm!('N', tB, α, view(Ab,:,:,i), B, one(T), view(Cb,:,i,:))
            end
        else
            # Complex 'C': t[k,i,j] = conj(conj(α)·Σ_l dA[k,l,i]·conj(op(B)[l,j])), and
            # conj(op(B)) re-expresses through gemm flags for tB ∈ {'T','C'}; only
            # tB == 'N' materialises conj(B).
            fB, Be = tB == 'N' ? ('N', conj(B)) : (tB == 'T' ? ('C', B) : ('T', B))
            W = Matrix{T}(undef, Nw, n)
            for i in 1:m
                BLAS.gemm!('N', fB, conj(α), view(Ab,:,:,i), Be, zero(T), W)
                view(Cb,:,i,:) .+= conj.(W)
            end
        end
    end
    # 3) dα·op(A)·op(B): the product is lane-invariant — hoist it once when any lane
    #    seeds α, then accumulate per seeded lane.
    if !all(iszero, dαs)
        AB = BLAS.gemm(tA, tB, one(T), A, B)
        for k in 1:Nw
            iszero(dαs[k]) || (view(Cb,k,:,:) .+= dαs[k] .* AB)
        end
    end
    # 4) dβ·C over the original `C`; strong zero on NaN entries (`C` may hold undefined
    #    values wherever `β == 0` discards them).
    if !all(iszero, dβs)
        Cbm = reshape(Cb, Nw, m * n)
        @inbounds for li in 1:(m * n)
            ci = C[li]
            isnan(ci) && continue
            for k in 1:Nw
                Cbm[k, li] += dβs[k] * ci
            end
        end
    end
    ccopied && _write_back_partials!(C_dC, Cb)
    # 5) Primal update after all tangent terms (they read the original operands).
    BLAS.gemm!(tA, tB, α, A, B, β, C)
    return C_dC
end
@inline function rrule!!(
    ::CoDual{typeof(BLAS.gemm!)},
    transA::CoDual{Char},
    transB::CoDual{Char},
    alpha::CoDual{T},
    A::CoDual{<:AbstractVecOrMat{T}},
    B::CoDual{<:AbstractVecOrMat{T}},
    beta::CoDual{T},
    C::CoDual{<:AbstractMatrix{T}},
) where {T<:BlasFloat}
    _check_blas_output_alias(BLAS.gemm!, primal(C), primal(A), primal(B))
    tA = uppercase(primal(transA))
    tB = uppercase(primal(transB))
    a = primal(alpha)
    b = primal(beta)
    p_A, dA = matrixify(A)
    p_B, dB = matrixify(B)
    p_C, dC = arrayify(C)

    # Save state and run primal
    p_C_copy = copy(p_C)
    fast = isone(a) && iszero(b)
    tmp = if fast
        BLAS.gemm!(primal(transA), primal(transB), a, p_A, p_B, b, p_C)
    else
        BLAS.gemm(primal(transA), primal(transB), one(T), p_A, p_B)
    end
    if !fast && iszero(a)
        BLAS.gemm!(primal(transA), primal(transB), a, p_A, p_B, b, p_C)
    elseif !fast
        p_C .= _rvs_mul.(p_C, b) .+ a .* tmp
    end

    function gemm!_pb!!(::NoRData)
        da = _rvs_guarded_dot(tmp, dC)

        # Restore state
        BLAS.copyto!(p_C, p_C_copy)

        # gradient wrt beta
        db = _rvs_guarded_dot(p_C, dC)
        # At a=1, b=0 these terms vanish but cancel the output's coefficient directions.
        fast && (da -= _rvs_mul(db, b') + _rvs_mul(da, a' - one(T)))

        # gradients wrt A and B (depends on transpose flags tA and tB)
        # C = a * op(A) * op(B) + b * C
        if tA == 'N'
            # A not transposed: C = a*A*op(B) + b*C
            # dA += a' * dC * op(B)'
            Bherm = tB == 'T' ? conj(p_B) : p_B
            _rvs_muladd!(dA, dC, Bherm, a', 'N', tB == 'N' ? 'C' : 'N', true, false)
        elseif tA == 'C'
            # A conjugate transposed: C = a*A'*op(B) + b*C
            # dA += a * op(B) * dC'
            _rvs_muladd!(dA, p_B, dC, a, tB, 'C', true, false)
        else  # tA == 'T'
            # A transposed (complex): C = a*A^T*op(B) + b*C
            # dA += conj(a) * conj(op(B)) * transpose(dC)
            if tB == 'N'
                _rvs_muladd!(dA, conj(p_B), dC, a', 'N', 'T', true, false)
            else
                _rvs_muladd!(dA, p_B, dC, a', tB == 'T' ? 'C' : 'T', 'T', true, false)
            end
        end

        if tB == 'N'
            # B not transposed: C = a*op(A)*B + b*C
            # dB += a' * op(A)' * dC
            Aherm = tA == 'T' ? conj(p_A) : p_A
            _rvs_muladd!(dB, Aherm, dC, a', tA == 'N' ? 'C' : 'N', 'N', true, false)
        elseif tB == 'C'
            # B conjugate transposed: C = a*op(A)*B' + b*C
            # dB += a * dC' * op(A)
            _rvs_muladd!(dB, dC, p_A, a, 'C', tA, true, false)
        else  # tB == 'T'
            # B transposed (complex): C = a*op(A)*B^T + b*C
            # dB += conj(a) * transpose(dC) * conj(op(A))
            if tA == 'N'
                _rvs_muladd!(dB, dC, conj(p_A), a', 'T', 'N', true, false)
            else
                _rvs_muladd!(dB, dC, p_A, a', 'T', tA == 'T' ? 'C' : 'T', true, false)
            end
        end

        # Propagate gradient through beta
        dC .= _rvs_zero.(_rvs_mul.(dC, b'), all(iszero, dC))

        return (NoRData(), NoRData(), NoRData(), da, NoRData(), NoRData(), db, NoRData())
    end

    return C, gemm!_pb!!
end

for (fname, elty) in ((:(symm!), BlasFloat), (:(hemm!), BlasComplexFloat))
    isherm = fname == :(hemm!)

    @eval @is_primitive(
        MinimalCtx,
        Tuple{
            typeof(BLAS.$fname),
            Char,
            Char,
            T,
            AbstractMatrix{T},
            AbstractMatrix{T},
            T,
            AbstractMatrix{T},
        } where {T<:$elty},
    )
    @eval function frule!!(
        ::Lifted{typeof(BLAS.$fname),Nw},
        side::Lifted{Char},
        uplo::Lifted{Char},
        alpha::Lifted{T,Nw},
        A_dA::Lifted{<:AbstractMatrix{T}},
        B_dB::Lifted{<:AbstractMatrix{T}},
        beta::Lifted{T,Nw},
        C_dC::Lifted{<:AbstractMatrix{T}},
    ) where {Nw,T<:$elty}
        _check_blas_output_alias(BLAS.$fname, primal(C_dC), primal(A_dA), primal(B_dB))
        s = _lsame_flag(primal(side))
        ul = _lsame_flag(primal(uplo))
        α = primal(alpha)
        β = primal(beta)
        A = primal(A_dA)
        B = primal(B_dB)
        C = primal(C_dC)
        dαs = ntuple(k -> tangent(alpha, k), Val(Nw))
        dβs = ntuple(k -> tangent(beta, k), Val(Nw))
        Ab, _ = _partials_block(A_dA)
        Bb, _ = _partials_block(B_dB)
        Cb, ccopied = _partials_block(C_dC)
        m, n = size(C)
        # 1) β·dC + α·(A⊛dB) (side-dependent product), β folded in (applied exactly once,
        #    first). Side 'R' contracts dB's last axis with A — one flat wide $fname on
        #    the (Nw·m, n) view. Side 'L' right-multiplies each dC slab by Aᵀ: symmetric
        #    Aᵀ = A directly; hermitian Aᵀ = conj(A), which is hermitian with the same
        #    triangle significant, so a hoisted conj(A) feeds the same kernel.
        if !iszero(Bb)
            if s == 'R'
                BLAS.$fname(
                    'R', ul, α, A, reshape(Bb, Nw * m, n), β, reshape(Cb, Nw * m, n)
                )
            else
                Ae = $(isherm ? :(conj(A)) : :A)
                for j in 1:n
                    BLAS.$fname('R', ul, α, Ae, view(Bb,:,:,j), β, view(Cb,:,:,j))
                end
            end
        else
            _scale_or_zero!(Cb, β)
        end
        # 2) α·(dA⊛B) — skipped when `A` is constant data. dA is symmetric/hermitian with
        #    only the `ul` triangle significant, like `A`: gather each lane into a dense
        #    scratch, apply the same kernel into a hoisted dense product, and accumulate
        #    into the lane's (strided) slice of the block.
        if !iszero(Ab)
            R = size(A, 1)
            Ascr = Matrix{T}(undef, R, R)
            Cscr = Matrix{T}(undef, m, n)
            Abm = reshape(Ab, Nw, R, R)
            for k in 1:Nw
                copyto!(Ascr, view(Abm,k,:,:))
                BLAS.$fname(s, ul, α, Ascr, B, zero(T), Cscr)
                view(Cb,k,:,:) .+= Cscr
            end
        end
        # 3) dα·(A⊛B): lane-invariant product, hoisted once when any lane seeds α.
        if !all(iszero, dαs)
            AB = Matrix{T}(undef, m, n)
            BLAS.$fname(s, ul, one(T), A, B, zero(T), AB)
            for k in 1:Nw
                iszero(dαs[k]) || (view(Cb,k,:,:) .+= dαs[k] .* AB)
            end
        end
        # 4) dβ·C over the original `C`; strong zero on NaN entries.
        if !all(iszero, dβs)
            Cbm = reshape(Cb, Nw, m * n)
            @inbounds for li in 1:(m * n)
                ci = C[li]
                isnan(ci) && continue
                for k in 1:Nw
                    Cbm[k, li] += dβs[k] * ci
                end
            end
        end
        ccopied && _write_back_partials!(C_dC, Cb)
        BLAS.$fname(s, ul, α, A, B, β, C)
        return C_dC
    end
    @eval function rrule!!(
        ::CoDual{typeof(BLAS.$fname)},
        side::CoDual{Char},
        uplo::CoDual{Char},
        alpha::CoDual{T},
        A_dA::CoDual{<:AbstractMatrix{T}},
        B_dB::CoDual{<:AbstractMatrix{T}},
        beta::CoDual{T},
        C_dC::CoDual{<:AbstractMatrix{T}},
    ) where {T<:$elty}
        _check_blas_output_alias(BLAS.$fname, primal(C_dC), primal(A_dA), primal(B_dB))

        # Extract primals.
        s = uppercase(primal(side))
        ul = primal(uplo)
        α = primal(alpha)
        β = primal(beta)
        A, dA = arrayify(A_dA)
        B, dB = arrayify(B_dB)
        C, dC = arrayify(C_dC)

        # In this rule we optimise carefully for the special case a == 1 && b == 0, which
        # corresponds to simply multiplying symm(A) and B together, and writing the result to C.
        # This is an extremely common edge case, so it's important to do well for it.
        C_copy = copy(C)
        fast = isone(α) && iszero(β)
        tmp = if fast
            BLAS.$fname(primal(side), ul, α, A, B, β, C)
        else
            $(isherm ? BLAS.hemm : BLAS.symm)(primal(side), ul, one(T), A, B)
        end
        if !fast
            C .= _rvs_mul.(C, β) .+ _rvs_mul.(tmp, α)
        end

        function symm!_or_hemm!_adjoint(::NoRData)
            dα = _rvs_guarded_dot(tmp, dC)

            BLAS.copyto!(C, C_copy)

            # gradient w.r.t. A.
            # TODO: could be switched to BLAS.{sy,he}r2k! if Julia ever provides it.
            dA_tmp = similar(dA)
            if s == 'L'
                _rvs_muladd!(dA_tmp, dC, B, α', 'N', 'C', false, false)
            else
                _rvs_muladd!(dA_tmp, B, dC, α', 'C', 'N', false, false)
            end
            # Doubling the diagonal below can overflow near `floatmax` although the
            # projected cotangent is representable, and products of opposite-extreme
            # coefficients and operands follow the fixed BLAS evaluation order.
            if ul == 'L'
                dA .=
                    (dA .+ LowerTriangular(dA_tmp)) .+
                    $(isherm ? adjoint : transpose)(UpperTriangular(dA_tmp))
            else
                dA .=
                    (dA .+ $(isherm ? adjoint : transpose)(LowerTriangular(dA_tmp))) .+
                    UpperTriangular(dA_tmp)
            end
            @inbounds for n in diagind(dA)
                dA[n] -= $(isherm ? :(real(dA_tmp[n])) : :(dA_tmp[n]))
            end

            # gradient w.r.t. B: dB += α' A' dC  (or α' dC A' if right)
            # if A is hermitian or real then A' = A, else A' = conj(A)
            zero_seed = all(iszero, dC)
            _rvs_blas!(
                BLAS.$fname,
                zero_seed,
                s,
                ul,
                α',
                $(isherm ? :A : :(conj(A))),
                dC,
                one(T),
                dB,
            )

            # gradient w.r.t. beta.
            dβ = _rvs_guarded_dot(C, dC)
            # Remove the output's coefficient perturbations from the unscaled product.
            fast && (dα -= _rvs_mul(dβ, β') + _rvs_mul(dα, α' - one(T)))

            # gradient w.r.t. C.
            dC .= _rvs_zero.(_rvs_mul.(dC, β'), zero_seed)

            return (
                NoRData(), NoRData(), NoRData(), dα, NoRData(), NoRData(), dβ, NoRData()
            )
        end
        return C_dC, symm!_or_hemm!_adjoint
    end
end

for (fname, elty, relty) in (
    (:(syrk!), Float32, Float32),
    (:(syrk!), Float64, Float64),
    (:(syrk!), ComplexF32, ComplexF32),
    (:(syrk!), ComplexF64, ComplexF64),
    # note that α and β are real for herk
    (:(herk!), ComplexF32, Float32),
    (:(herk!), ComplexF64, Float64),
)
    isherm = fname == :(herk!)
    nonbang = Symbol(chop(string(fname)))  # syrk!/herk! -> syrk/herk (non-mutating product)

    @eval @is_primitive(
        MinimalCtx,
        Tuple{
            typeof(BLAS.$fname),
            Char,
            Char,
            $relty,
            AbstractVecOrMat{$elty},
            $relty,
            AbstractMatrix{$elty},
        }
    )
    @eval function frule!!(
        ::Lifted{typeof(BLAS.$fname),Nw},
        _uplo::Lifted{Char},
        _t::Lifted{Char},
        α_dα::Lifted{$relty,Nw},
        A_dA::Lifted{<:AbstractVecOrMat{$elty}},
        β_dβ::Lifted{$relty,Nw},
        C_dC::Lifted{<:AbstractMatrix{$elty}},
    ) where {Nw}
        _check_blas_output_alias(BLAS.$fname, primal(C_dC), primal(A_dA))
        uplo = _lsame_flag(primal(_uplo))
        t = _lsame_flag(primal(_t))
        α = primal(α_dα)
        A = primal(A_dA)
        β = primal(β_dβ)
        C = primal(C_dC)
        dαs = ntuple(k -> tangent(α_dα, k), Val(Nw))
        dβs = ntuple(k -> tangent(β_dβ, k), Val(Nw))
        Ab, _ = _partials_block(A_dA)
        Cb, ccopied = _partials_block(C_dC)
        nC = size(C, 1)
        Cbm = reshape(Cb, Nw, nC, nC)
        # 1) β·dC + α·(op(dA)·op(A)' + op(A)·op(dA)') on the `uplo` triangle. The rank-2k
        #    update mixes the lane-varying dA into both factors, so it stays per lane:
        #    gather dA's lane and dC's `uplo` triangle into dense scratches, run the same
        #    syr2k!/her2k! the width-1 rule uses, and scatter the triangle back. The
        #    non-`uplo` triangle of dC is never touched, exactly like the primal.
        if !iszero(Ab)
            # `A` may be a vector or a matrix; gather lanes through flat views so the
            # scratch matches either shape.
            Abf = reshape(Ab, Nw, :)
            Cbf = reshape(Cb, Nw, :)
            Ascr = Array{$elty}(undef, size(A))
            Cscr = Matrix{$elty}(undef, nC, nC)
            for k in 1:Nw
                copyto!(Ascr, view(Abf, k, :))
                copyto!(Cscr, view(Cbf, k, :))
                BLAS.$(isherm ? :her2k! : :syr2k!)(uplo, t, $elty(α), A, Ascr, β, Cscr)
                if uplo == 'U'
                    @inbounds for j in 1:nC, i in 1:j
                        Cbm[k, i, j] = Cscr[i, j]
                    end
                else
                    @inbounds for j in 1:nC, i in j:nC
                        Cbm[k, i, j] = Cscr[i, j]
                    end
                end
            end
        else
            # `A` is constant data: only the β scaling remains, on the `uplo` triangle.
            if uplo == 'U'
                @inbounds for j in 1:nC, i in 1:j, k in 1:Nw
                    Cbm[k, i, j] = iszero(β) ? zero($elty) : β * Cbm[k, i, j]
                end
            else
                @inbounds for j in 1:nC, i in j:nC, k in 1:Nw
                    Cbm[k, i, j] = iszero(β) ? zero($elty) : β * Cbm[k, i, j]
                end
            end
        end
        # 2) dα·(op(A)·op(A)') — lane-invariant rank-k product, hoisted once when any lane
        #    seeds α, masked to the `uplo` triangle.
        if !all(iszero, dαs)
            AAt = BLAS.$nonbang(uplo, t, one($relty), A)
            uplo == 'U' ? triu!(AAt) : tril!(AAt)
            for k in 1:Nw
                iszero(dαs[k]) || (view(Cbm,k,:,:) .+= dαs[k] .* AAt)
            end
        end
        # 3) dβ·C over the original `C`'s `uplo` triangle; strong zero on NaN entries
        #    (the β==0 convention lets the caller pass an uninitialised/NaN C).
        if !all(iszero, dβs)
            @inbounds for j in 1:nC
                irange = uplo == 'U' ? (1:j) : (j:nC)
                for i in irange
                    ci = C[i, j]
                    isnan(ci) && continue
                    for k in 1:Nw
                        Cbm[k, i, j] += dβs[k] * ci
                    end
                end
            end
        end
        # herk!'s output diagonal is real; its tangent diagonal must be too.
        $(isherm ? quote
            @inbounds for i in 1:nC, k in 1:Nw
                Cbm[k, i, i] = real(Cbm[k, i, i])
            end
        end : :())
        ccopied && _write_back_partials!(C_dC, Cb)
        BLAS.$fname(uplo, t, α, A, β, C)
        return C_dC
    end
    @eval function rrule!!(
        ::CoDual{typeof(BLAS.$fname)},
        _uplo::CoDual{Char},
        _t::CoDual{Char},
        α_dα::CoDual{$relty},
        A_dA::CoDual{<:AbstractVecOrMat{$elty}},
        β_dβ::CoDual{$relty},
        C_dC::CoDual{<:AbstractMatrix{$elty}},
    )
        _check_blas_output_alias(BLAS.$fname, primal(C_dC), primal(A_dA))

        # Extract values from pairs.
        uplo = primal(_uplo)
        trans = uppercase(primal(_t))
        α = primal(α_dα)
        A, dA = matrixify(A_dA)
        β = primal(β_dβ)
        C, dC = arrayify(C_dC)

        # Run forwards pass, and remember previous value of `C` for the reverse-pass.
        C_copy = collect(C)
        BLAS.$fname(uplo, primal(_t), α, A, β, C)

        function syrk!_or_herk!_adjoint(::NoRData)
            # Restore previous state.
            C .= C_copy

            # Increment gradients.
            $(isherm ? :(real_diag!(dC)) : :())

            B = uplo == 'U' ? triu(dC) : tril(dC)
            ∇β = _rvs_guarded_dot(C, B)
            $(isherm ? :(∇β = real(∇β)) : :())
            ∇α = _rvs_guarded_dot(
                if trans == 'N'
                    A * $(isherm ? adjoint : transpose)(A)
                else
                    $(isherm ? adjoint : transpose)(A) * A
                end,
                B,
            )
            $(isherm ? :(∇α = real(∇α)) : :())

            M1 = B + $(isherm ? adjoint : transpose)(B)
            M2 = $(isherm ? :A : :(conj(A)))
            zero_seed = all(iszero, B)
            dA .+= _rvs_zero.(
                _rvs_mul.(trans == 'N' ? M1 * M2 : M2 * M1, $elty(α')), zero_seed
            )
            dC .=
                (uplo == 'U' ? tril!(dC, -1) : triu!(dC, 1)) .+
                _rvs_zero.(_rvs_mul.(B, $elty(β')), zero_seed)

            return (NoRData(), NoRData(), NoRData(), ∇α, NoRData(), ∇β, NoRData())
        end

        return C_dC, syrk!_or_herk!_adjoint
    end
end

function real_diag!(dA::AbstractMatrix{<:Complex{<:BlasFloat}})
    @inbounds for n in diagind(dA)
        dA[n] = real(dA[n])
    end
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.trmm!),Char,Char,Char,Char,P,AbstractMatrix{P},AbstractMatrix{P}
    } where {P<:BlasFloat}
)
function frule!!(
    ::Lifted{typeof(BLAS.trmm!),Nw},
    _side::Lifted{Char},
    _uplo::Lifted{Char},
    _ta::Lifted{Char},
    _diag::Lifted{Char},
    α_dα::Lifted{P,Nw},
    A_dA::Lifted{<:AbstractMatrix{P}},
    B_dB::Lifted{<:AbstractMatrix{P}},
) where {Nw,P<:BlasFloat}
    _check_blas_output_alias(BLAS.trmm!, primal(B_dB), primal(A_dA))
    side = _lsame_flag(primal(_side))
    uplo = _lsame_flag(primal(_uplo))
    ta = _lsame_flag(primal(_ta))
    diag = _lsame_flag(primal(_diag))
    α = primal(α_dα)
    A = primal(A_dA)
    B = primal(B_dB)
    dαs = ntuple(k -> tangent(α_dα, k), Val(Nw))
    Ab, _ = _partials_block(A_dA)
    Bb, bcopied = _partials_block(B_dB)
    m, n = size(B)
    # dB_k := α·(op(A)⊛dB_k) + α·(op(dA_k)⊛B) + dα_k·(op(A)⊛B), the products on `side`.
    # 1) α·(op(A)⊛dB_k) for all lanes, applied first (it overwrites; later terms add).
    #    Side 'R' contracts dB's last axis with op(A) — one flat wide trmm, flags native.
    #    Side 'L' right-multiplies each dC slab by op(A)ᵀ (flag flip; complex 'C' needs a
    #    hoisted conj(A), whose triangle mirrors A's).
    if !iszero(Bb)
        if side == 'R'
            BLAS.trmm!('R', uplo, ta, diag, α, A, reshape(Bb, Nw * m, n))
        else
            fA, Ae = if ta == 'N'
                ('T', A)
            elseif ta == 'T' || P <: BlasRealFloat
                ('N', A)
            else
                ('N', conj(A))
            end
            for j in 1:n
                BLAS.trmm!('R', uplo, fA, diag, α, Ae, view(Bb,:,:,j))
            end
        end
    end
    # 2) α·(op(dA_k)⊛B) — skipped when `A` is constant data. trmm masks dA's triangle
    #    (and implicit unit diagonal, whose derivative the `diag == 'U'` correction
    #    removes: the stored diagonal never enters the primal, so its partial must not
    #    enter the tangent).
    if !iszero(Ab)
        R = size(A, 1)
        Abm = reshape(Ab, Nw, R, R)
        Ascr = Matrix{P}(undef, R, R)
        Bscr = Matrix{P}(undef, m, n)
        for k in 1:Nw
            copyto!(Ascr, view(Abm,k,:,:))
            copyto!(Bscr, B)
            BLAS.trmm!(side, uplo, ta, diag, α, Ascr, Bscr)
            diag === 'U' && (Bscr .-= α .* B)
            view(Bb,k,:,:) .+= Bscr
        end
    end
    # 3) dα·(op(A)⊛B): lane-invariant product, hoisted once when any lane seeds α.
    if !all(iszero, dαs)
        AopB = Matrix{P}(undef, m, n)
        copyto!(AopB, B)
        BLAS.trmm!(side, uplo, ta, diag, one(P), A, AopB)
        for k in 1:Nw
            iszero(dαs[k]) || (view(Bb,k,:,:) .+= dαs[k] .* AopB)
        end
    end
    bcopied && _write_back_partials!(B_dB, Bb)
    BLAS.trmm!(side, uplo, ta, diag, α, A, B)
    return B_dB
end
function rrule!!(
    ::CoDual{typeof(BLAS.trmm!)},
    _side::CoDual{Char},
    _uplo::CoDual{Char},
    _ta::CoDual{Char},
    _diag::CoDual{Char},
    α_dα::CoDual{P},
    A_dA::CoDual{<:AbstractMatrix{P}},
    B_dB::CoDual{<:AbstractMatrix{P}},
) where {P<:BlasFloat}
    _check_blas_output_alias(BLAS.trmm!, primal(B_dB), primal(A_dA))

    # Extract values.
    side = uppercase(primal(_side))
    uplo = primal(_uplo)
    tA = uppercase(primal(_ta))
    diag = uppercase(primal(_diag))
    α = primal(α_dα)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    B_copy = copy(B)

    # Run primal.
    BLAS.trmm!(primal(_side), uplo, primal(_ta), primal(_diag), α, A, B)

    function trmm_adjoint(::NoRData)

        # Recompute the unscaled output at α == 0 to avoid dividing by zero.
        zero_seed = all(iszero, dB)
        ∇α = if iszero(α)
            M = copy(B_copy)
            BLAS.trmm!(side, uplo, tA, diag, one(P), A, M)
            _rvs_guarded_dot(M, dB)
        else
            _rvs_zero(_rvs_guarded_dot(B, dB) / α', zero_seed)
        end

        # Restore initial state.
        B .= B_copy

        # Increment gradients.
        tmp = if side == 'L'
            if tA == 'T' && P <: BlasComplexFloat
                conj(B) * transpose(dB)
            elseif tA == 'N'
                dB * B'
            else
                B * dB'
            end
        else
            if tA == 'T' && P <: BlasComplexFloat
                transpose(dB) * conj(B)
            elseif tA == 'N'
                B' * dB
            else
                dB' * B
            end
        end
        dA .+= _rvs_zero.(_rvs_mul.(tri!(tmp, uplo, diag), tA == 'C' ? α : α'), zero_seed)

        # Compute dB tangent.
        if tA == 'T' && P <: BlasComplexFloat
            # conjugate-only of A
            _rvs_blas!(BLAS.trmm!, zero_seed, side, uplo, 'N', diag, α', conj(A), dB)
        else
            _rvs_blas!(
                BLAS.trmm!, zero_seed, side, uplo, tA == 'N' ? 'C' : 'N', diag, α', A, dB
            )
        end

        return tuple_fill(NoRData(), Val(5))..., ∇α, NoRData(), NoRData()
    end

    return B_dB, trmm_adjoint
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.trsm!),Char,Char,Char,Char,P,AbstractMatrix{P},AbstractMatrix{P}
    } where {P<:BlasFloat},
)

function frule!!(
    ::Lifted{typeof(BLAS.trsm!),Nw},
    _side::Lifted{Char},
    _uplo::Lifted{Char},
    _t::Lifted{Char},
    _diag::Lifted{Char},
    α_dα::Lifted{P,Nw},
    A_dA::Lifted{<:AbstractMatrix{P}},
    B_dB::Lifted{<:AbstractMatrix{P}},
) where {Nw,P<:BlasFloat}
    _check_blas_output_alias(BLAS.trsm!, primal(B_dB), primal(A_dA))
    side = _lsame_flag(primal(_side))
    uplo = _lsame_flag(primal(_uplo))
    trans = _lsame_flag(primal(_t))
    diag = _lsame_flag(primal(_diag))
    α = primal(α_dα)
    A = primal(A_dA)
    B = primal(B_dB)
    dαs = ntuple(k -> tangent(α_dα, k), Val(Nw))
    Bb, bcopied = _partials_block(B_dB)
    # BLAS's `α == 0` quick return sets `B := 0` without ever referencing `A`, so `A` may
    # legally hold garbage. The JVP is `dα·op(A)⁻¹⊛B`, needing the solve only when some
    # lane seeds α; with none, result and derivative are both identically zero and the
    # solve below would otherwise propagate a legal NaN in `A` into the result.
    if iszero(α)
        fill!(Bb, zero(P))
        if !all(iszero, dαs)
            BLAS.trsm!(side, uplo, trans, diag, one(P), A, B)
            for k in 1:Nw
                iszero(dαs[k]) || (view(Bb,k,:,:) .= dαs[k] .* B)
            end
        end
        fill!(B, zero(P))
        bcopied && _write_back_partials!(B_dB, Bb)
        return B_dB
    end
    Ab, _ = _partials_block(A_dA)
    m, n = size(B)
    # Form α·dB + dα·B before the primal overwrites B. With Y = α·op(A)⁻¹⊛B,
    # dY = op(A)⁻¹⊛(α·dB + dα·B − op(dA)⊛Y): no unscaled primal solve is needed.
    @inbounds for j in 1:n, i in 1:m, k in 1:Nw
        db = α * Bb[k, i, j]
        Bb[k, i, j] = iszero(dαs[k]) ? db : db + dαs[k] * B[i, j]
    end
    BLAS.trsm!(side, uplo, trans, diag, α, A, B)
    # trmm masks dA's triangle; remove the implicit unit diagonal's contribution.
    if !iszero(Ab)
        R = size(A, 1)
        Abm = reshape(Ab, Nw, R, R)
        Ascr = Matrix{P}(undef, R, R)
        tmp = Matrix{P}(undef, m, n)
        for k in 1:Nw
            copyto!(Ascr, view(Abm,k,:,:))
            copyto!(tmp, B)
            BLAS.trmm!(side, uplo, trans, diag, one(P), Ascr, tmp)
            diag == 'U' && (tmp .-= B)
            view(Bb,k,:,:) .-= tmp
        end
    end
    # Apply op(A)⁻¹ to every lane. Side 'R' solves the (Nw·m, n) flat view in one wide
    # trsm, flags native; side 'L' right-divides each slab by op(A)ᵀ (flag flip; complex
    # 'C' needs a hoisted conj(A)).
    if side == 'R'
        BLAS.trsm!('R', uplo, trans, diag, one(P), A, reshape(Bb, Nw * m, n))
    else
        fA, Ae = if trans == 'N'
            ('T', A)
        elseif trans == 'T' || P <: BlasRealFloat
            ('N', A)
        else
            ('N', conj(A))
        end
        for j in 1:n
            BLAS.trsm!('R', uplo, fA, diag, one(P), Ae, view(Bb,:,:,j))
        end
    end
    bcopied && _write_back_partials!(B_dB, Bb)
    return B_dB
end

function rrule!!(
    ::CoDual{typeof(BLAS.trsm!)},
    _side::CoDual{Char},
    _uplo::CoDual{Char},
    _t::CoDual{Char},
    _diag::CoDual{Char},
    α_dα::CoDual{P},
    A_dA::CoDual{<:AbstractMatrix{P}},
    B_dB::CoDual{<:AbstractMatrix{P}},
) where {P<:BlasFloat}
    _check_blas_output_alias(BLAS.trsm!, primal(B_dB), primal(A_dA))

    # Extract parameters.
    side = uppercase(primal(_side))
    uplo = primal(_uplo)
    trans = uppercase(primal(_t))
    diag = uppercase(primal(_diag))
    α = primal(α_dα)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)

    # Copy memory which will be overwritten by primal computation.
    B_copy = copy(B)

    # Run primal computation.
    trsm!(primal(_side), uplo, primal(_t), primal(_diag), α, A, B)

    function trsm_adjoint(::NoRData)
        M = if iszero(α)
            trsm!(side, uplo, trans, diag, one(P), A, copy(B_copy))
        else
            B
        end
        ∇α = _rvs_guarded_dot(M, dB)
        zero_seed = all(iszero, dB)
        iszero(α) || (∇α = _rvs_zero(∇α / α', zero_seed))
        # Keep the zero alpha perturbation live under forward-over-reverse.
        c = iszero(α) ? (trans == 'C' ? α : α') : one(P)

        # Increment cotangents.
        if side == 'L'
            if trans == 'N'
                tmp = trsm!('L', uplo, 'C', diag, -one(P), A, dB * M')
            elseif trans == 'C'
                tmp = trsm!('R', uplo, 'C', diag, -one(P), A, M * dB')
            else
                tmp = trsm!('R', uplo, 'C', diag, -one(P), A, conj(M * dB'))
            end
        else
            if trans == 'N'
                tmp = trsm!('R', uplo, 'C', diag, -one(P), A, M'dB)
            elseif trans == 'C'
                tmp = trsm!('L', uplo, 'C', diag, -one(P), A, dB'M)
            else
                tmp = trsm!('L', uplo, 'C', diag, -one(P), A, conj(dB'M))
            end
        end
        dA .+= _rvs_zero.(_rvs_mul.(tri!(tmp, uplo, diag), c), zero_seed)

        # Restore initial state.
        B .= B_copy

        # Compute dB tangent.
        if trans == 'T'
            # conjugate-only of A
            _rvs_blas!(BLAS.trsm!, zero_seed, side, uplo, 'N', diag, α', conj(A), dB)
        else
            _rvs_blas!(
                BLAS.trsm!, zero_seed, side, uplo, trans == 'N' ? 'C' : 'N', diag, α', A, dB
            )
        end
        return tuple_fill(NoRData(), Val(5))..., ∇α, NoRData(), NoRData()
    end

    return B_dB, trsm_adjoint
end

function blas_matrices(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int, q::Int)
    # blas_matrices must return `Xs` with the same length as blas_vectors.
    Xs = Any[
        randn(rng, P, p, q),
        view(randn(rng, P, p + 5, 2q), 3:(p + 2), 1:2:(2q)),
        view(randn(rng, P, 3p, 3, 2q), (p + 1):(2p), 2, 1:2:(2q)),
        reshape(view(randn(rng, P, p * q + 5), 1:(p * q)), p, q),
    ]
    @static if VERSION >= v"1.11"
        # To match Memory in blas_vectors
        push!(Xs, randn(rng, P, p, q))
    end
    @assert all(X -> size(X) == (p, q), Xs)
    @assert all(Base.Fix2(isa, AbstractMatrix{P}), Xs)
    return Xs
end

function special_matrices(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int, q::Int)
    Xs = map(Diagonal, blas_vectors(rng, P, p))
    @assert all(X -> size(X) == (isa(X, Diagonal) ? (p, p) : (p, q)), Xs)
    @assert all(Base.Fix2(isa, AbstractMatrix{P}), Xs)
    return Xs
end

function invertible_blas_matrices(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int)
    return map(blas_matrices(rng, P, p, p)) do A
        U, _, V = svd(0.1 * A + I)
        λs = p > 1 ? collect(range(1.0, 2.0; length=p)) : [1.0]
        A .= collect(U * Diagonal(λs) * V')
        return A
    end
end

function positive_definite_blas_matrices(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int)
    return map(blas_matrices(rng, P, p, p)) do A
        A .= A'A + I
        return A
    end
end

function blas_vectors(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int; only_contiguous=false)
    xs = Any[
        randn(rng, P, p),
        view(randn(rng, P, p + 5), 3:(p + 2)),
        (only_contiguous ? collect : identity)(view(randn(rng, P, 3p, 3), 1:2:(2p), 2)),
        reshape(view(randn(rng, P, 1, p + 5), 1:1, 1:p), p),
    ]
    @static if VERSION >= v"1.11"
        push!(xs, Memory{P}(randn(rng, P, p)))
    end
    @assert all(x -> length(x) == p, xs)
    @assert all(Base.Fix2(isa, AbstractVector{P}), xs)
    return xs
end

# BLAS tests are split by element type so that arrays for each precision can be GC'd
# before the next precision's arrays are allocated.
function hand_written_rule_test_cases(rng_ctor, ::Val{:blas}, P::Type{<:BlasFloat})
    t_flags = ['N', 'T', 'C']
    αs = [1.0, -0.25, 0.46 + 0.32im]
    βs = [0.0, 0.33, 0.39 + 0.27im]
    uplos = ['L', 'U']
    dAs = ['N', 'U']
    rng = rng_ctor(123456)
    # A float scalar of a DIFFERENT precision from `P`, for the short-form `axpy!` rows below.
    Q = real(P) === Float64 ? Float32 : Float64
    plain = (false, :none, nothing)

    test_cases = vcat(

        #
        # BLAS LEVEL 1
        #

        # nrm2(n, x, incx)
        map_prod([5, 3], [1, 2]) do (n, incx)
            return map([randn(rng, P, 105)]) do x
                (false, :stability, nothing, BLAS.nrm2, n, x, incx)
            end
        end...,

        # Dense/strided long and short forms; width > 1 catches lane-stride misreads.
        Any[
            (plain..., BLAS.axpy!, P(2), randn(rng, P, 5), randn(rng, P, 5)),
            (plain..., BLAS.axpy!, 5, P(2), randn(rng, P, 5), 1, randn(rng, P, 5), 1),
            (
                plain...,
                BLAS.axpy!,
                P(2),
                view(randn(rng, P, 10), 1:2:10),
                view(randn(rng, P, 10), 1:2:10),
            ),
            (plain..., BLAS.axpby!, P(2), randn(rng, P, 5), P(3), randn(rng, P, 5)),
            (
                plain...,
                BLAS.axpby!,
                P(2),
                view(randn(rng, P, 10), 1:2:10),
                P(3),
                view(randn(rng, P, 10), 1:2:10),
            ),
            # Number scalars: NoTangent Int and mixed-precision floating tangents.
            (plain..., BLAS.axpy!, 2, randn(rng, P, 5), randn(rng, P, 5)),
            (plain..., BLAS.axpy!, Q(2), randn(rng, P, 5), randn(rng, P, 5)),
            (plain..., BLAS.axpby!, 2, randn(rng, P, 5), 3, randn(rng, P, 5)),
        ]...,

        # Convenience forms need their own boundary above width 1; cover strides too.
        Any[
            (plain..., BLAS.scal!, P(2), randn(rng, P, 5)),
            (plain..., BLAS.scal!, P(2), view(randn(rng, P, 10), 1:2:10)),
        ]...,
        (
            if P <: Real
                Any[
                    (plain..., BLAS.dot, randn(rng, P, 5), randn(rng, P, 5)),
                    (
                        plain...,
                        BLAS.dot,
                        view(randn(rng, P, 10), 1:2:10),
                        view(randn(rng, P, 10), 1:2:10),
                    ),
                ]
            else
                Any[]
            end
        )...,

        # Strided operands whose stride divides `incx`, which BLAS's raw walk maps onto the
        # operand's own elements one step at a time. `norm(view(A, 1, :))` is the ordinary form of
        # this: the one-argument `nrm2` passes `incx = stride`, giving a step of 1.
        Any[
            (false, :none, nothing, BLAS.nrm2, 5, view(randn(rng, P, 12), 1:2:12), 2),
            (false, :none, nothing, BLAS.nrm2, 4, view(randn(rng, P, 12), 1:3:12), 3),
            (false, :none, nothing, BLAS.nrm2, view(randn(rng, P, 12), 1:2:12)),
            (false, :none, nothing, BLAS.nrm2, 6, view(randn(rng, P, 6, 6), 1, :), 6),
            (
                false,
                :none,
                nothing,
                BLAS.scal!,
                5,
                P(2),
                view(randn(rng, P, 12), 1:2:12),
                2,
            ),
        ]...,

        # Length 40 exceeds norm2's threshold (32) for the inlined nrm2(x) boundary.
        map([randn(rng, P, 40)]) do x
            (false, :stability, nothing, BLAS.nrm2, x)
        end...,

        # Empty dot exercises gemv's quick return without β scaling.
        (
            if P <: BlasRealFloat
                map([0, 3, 5]) do n
                    return (
                        false, :stability, nothing, dot, randn(rng, P, n), randn(rng, P, n)
                    )
                end
            else
                []
            end
        )...,
        map_prod([1, 3, 11], [1, 2, 11]) do (n, incx)
            flags = (false, :stability, nothing)
            return (flags..., BLAS.scal!, n, randn(rng, P), randn(rng, P, n * incx), incx)
        end,

        # Forward primitives; derived rows cover reverse. Keep the direct chunk-width
        # sweep and stability checks for boxed block accumulators; and the strided operand exercises the per-lane fallback.
        (
            if P <: BlasRealFloat
                []
            else
                map([BLAS.dotc, BLAS.dotu]) do f
                    flags = (false, :stability, (; skip_reverse=true))
                    return [
                        (flags..., f, 3, randn(rng, P, 6), 2, randn(rng, P, 6), 2),
                        # Negative increments: BLAS walks the same elements backwards from
                        # `(-n+1)*inc + 1`, so the value matches `inc = +1`, but the block loop's
                        # `1 + (t-1)*inc` would run off the front. Takes the per-lane fallback.
                        (flags..., f, 3, randn(rng, P, 3), -1, randn(rng, P, 3), -1),
                        (
                            flags...,
                            f,
                            3,
                            randn(rng, P, 3),
                            1,
                            view(randn(rng, P, 12), 1:2:12),
                            2,
                        ),
                    ]
                end
            end
        )...,

        # axpy!(n, a, x, incx, y, incy)
        map_prod([1, 3, 11], [1, 2], [1, 2]) do (n, incx, incy)
            flags = (false, :stability, nothing)
            return (
                flags...,
                BLAS.axpy!,
                n,
                randn(rng, P),
                randn(rng, P, n * incx),
                incx,
                randn(rng, P, n * incy),
                incy,
            )
        end,

        # axpy!(n, a, x, 1, y, 1) with mismatched X/Y wrapper types (Vector, SubArray,
        # ReshapedArray, ...) -- the sweep above always pairs same-type Vectors, which
        # would miss a primitive registration that wrongly ties X and Y to one concrete
        # type. `circshift` forces the mismatch: `blas_vectors` returns the same sequence
        # of types every call, so zipping two calls by index would just pair each type with
        # itself. `only_contiguous=true` since incx=incy=1 below isn't valid for
        # `blas_vectors`' one non-contiguous entry.
        let xs = blas_vectors(rng, P, 5; only_contiguous=true)
            map(xs, circshift(xs, 1)) do x, y
                (false, :stability, nothing, BLAS.axpy!, 5, randn(rng, P), x, 1, y, 1)
            end
        end...,

        # axpy!(n, a, x, incx, x, incx): X and Y are the same array.
        map_prod([1, 3, 11], [1, 2]) do (n, incx)
            opts =
                P <: Complex ? (throws=(ArgumentError, "overlapping complex"),) : nothing
            flags = (false, :stability, opts)
            x = randn(rng, P, n * incx)
            return (flags..., BLAS.axpy!, n, randn(rng, P), x, incx, x, incx)
        end,
        (
            if P <: Complex
                x = randn(rng, P, 4)
                [(
                    false,
                    :none,
                    (throws=(ArgumentError, "overlapping complex"),),
                    BLAS.axpy!,
                    P(2),
                    view(x, 1:3),
                    view(x, 2:4),
                )]
            else
                []
            end
        )...,

        #
        # BLAS LEVEL 2
        #

        # gemv!
        map_prod(t_flags, [1, 3], [1, 2], αs, βs) do (tA, M, N, α, β)
            P <: BlasRealFloat && (imag(α) != 0 || imag(β) != 0) && return []

            As = [
                blas_matrices(rng, P, tA == 'N' ? M : N, tA == 'N' ? N : M)
                blas_vectors(rng, P, M; only_contiguous=true)
            ]
            xs = [blas_vectors(rng, P, N); blas_vectors(rng, P, tA == 'N' ? 1 : M)]
            ys = [blas_vectors(rng, P, M); blas_vectors(rng, P, tA == 'N' ? M : 1)]
            flags = (false, :stability, (lb=1e-3, ub=30.0))
            return map(As, xs, ys) do A, x, y
                (flags..., BLAS.gemv!, tA, P(α), A, x, P(β), y)
            end
        end...,

        # Zero-length x quick-returns without β scaling; the size sweep above is nonempty.
        map(βs) do β
            P <: BlasRealFloat && imag(β) != 0 && return []
            return [(
                false,
                :none,
                nothing,
                BLAS.gemv!,
                'N',
                P(1),
                randn(rng, P, 3, 0),
                P[],
                P(β),
                randn(rng, P, 3),
            )]
        end...,

        # symv!, hemv!
        map_prod([BLAS.symv!, BLAS.hemv!], ['L', 'U'], αs, βs) do (f, uplo, α, β)
            P <: BlasRealFloat && f == BLAS.hemv! && return []
            P <: BlasRealFloat && (imag(α) != 0 || imag(β) != 0) && return []

            As = blas_matrices(rng, P, 5, 5)
            ys = blas_vectors(rng, P, 5)
            xs = blas_vectors(rng, P, 5)
            return map(As, xs, ys) do A, x, y
                (false, :stability, nothing, f, uplo, P(α), A, x, P(β), y)
            end
        end...,

        # trmv!
        map_prod(uplos, t_flags, dAs, [1, 3]) do (ul, tA, dA, N)
            As = blas_matrices(rng, P, N, N)
            bs = blas_vectors(rng, P, N)
            return map(As, bs) do A, b
                (false, :stability, nothing, BLAS.trmv!, ul, tA, dA, A, b)
            end
        end...,

        # trsv!
        let
            # This test is sensitive to the random seed
            rng = rng_ctor(123457)
            map_prod(uplos, t_flags, dAs, [1, 3]) do (ul, tA, dA, N)
                As = blas_matrices(rng, P, N, N)
                bs = blas_vectors(rng, P, N)
                return map(As, bs) do A, b
                    (false, :stability, nothing, BLAS.trsv!, ul, tA, dA, A, b)
                end
            end
        end...,
    )

    #
    # BLAS LEVEL 3
    #

    # Pair scalar seeds rather than crossing them: preserve α==1 && β==0, α==0,
    # and β zero/nonzero branches, plus every dα/dβ zero combination. Nonzero
    # imaginary parts catch missing conjugation in complex rules.
    αβs = if P <: BlasComplexFloat
        [(1.0, 0.0), (0.0, 0.33), (0.46 + 0.32im, 0.0), (0.46 + 0.32im, 0.39 + 0.27im)]
    else
        [(1.0, 0.0), (0.0, 0.33), (-0.25, 0.0), (-0.25, 0.33)]
    end
    dαβs = if P <: BlasComplexFloat
        [(0.0, 0.0), (0.44, 0.0), (0.0, -0.11), (-0.20 + 0.38im, 0.86 + 0.44im)]
    else
        [(0.0, 0.0), (0.44, 0.0), (0.0, -0.11), (0.44, -0.11)]
    end
    dαs = [0.0, 0.44, -0.20 + 0.38im]

    # 1.10 fails to infer part of a matmat product in the pullback
    perf_flag = VERSION < v"1.11-" ? :none : :stability

    # The tests are quite sensitive to the random inputs,
    # so each tested gemm! dispatch gets its own rng.

    # gemm! - matrix × matrix
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123456)
            map_prod(t_flags, t_flags, αβs, dαβs) do (tA, tB, (α, β), (dα, dβ))
                As = blas_matrices(rng, P, tA == 'N' ? 3 : 4, tA == 'N' ? 4 : 3)
                Bs = blas_matrices(rng, P, tB == 'N' ? 4 : 5, tB == 'N' ? 5 : 4)
                Cs = blas_matrices(rng, P, 3, 5)

                return map(As, Bs, Cs) do A, B, C
                    a_da = CoDual(P(α), P(dα))
                    b_db = CoDual(P(β), P(dβ))
                    (false, perf_flag, nothing, BLAS.gemm!, tA, tB, a_da, A, B, b_db, C)
                end
            end
        end...,
    )

    # gemm! - matrix × vector
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123457)
            map_prod(t_flags, αβs, dαβs) do (tA, (α, β), (dα, dβ))
                P <: BlasRealFloat && tA == 'C' && return []

                As = blas_matrices(rng, P, tA == 'N' ? 3 : 4, tA == 'N' ? 4 : 3)
                Bs = blas_vectors(rng, P, 4; only_contiguous=true)
                Cs = blas_matrices(rng, P, 3, 1)

                return map(As, Bs, Cs) do A, B, C
                    a_da = CoDual(P(α), P(dα))
                    b_db = CoDual(P(β), P(dβ))
                    (
                        false, perf_flag, nothing, BLAS.gemm!, tA, 'N', a_da, A, B, b_db, C
                    )
                end
            end
        end...,
    )

    # gemm! - vector × matrix
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123458)
            map_prod(['T', 'C'], t_flags, αβs, dαβs) do (tA, tB, (α, β), (dα, dβ))
                P <: BlasRealFloat && (tA == 'C' || tB == 'C') && return []

                As = blas_vectors(rng, P, 3; only_contiguous=true)
                Bs = blas_matrices(rng, P, tB == 'N' ? 3 : 5, tB == 'N' ? 5 : 3)
                Cs = blas_matrices(rng, P, 1, 5)

                return map(As, Bs, Cs) do A, B, C
                    a_da = CoDual(P(α), P(dα))
                    b_db = CoDual(P(β), P(dβ))
                    (false, perf_flag, nothing, BLAS.gemm!, tA, tB, a_da, A, B, b_db, C)
                end
            end
        end...,
    )

    # gemm! - vector × vector
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123459)
            map_prod(['T', 'C'], αβs, dαβs) do (tA, (α, β), (dα, dβ))
                P <: BlasRealFloat && tA == 'C' && return []

                As = blas_vectors(rng, P, 3; only_contiguous=true)
                Bs = blas_vectors(rng, P, 3; only_contiguous=true)
                Cs = blas_matrices(rng, P, 1, 1)

                return map(As, Bs, Cs) do A, B, C
                    a_da = CoDual(P(α), P(dα))
                    b_db = CoDual(P(β), P(dβ))
                    (
                        false, perf_flag, nothing, BLAS.gemm!, tA, 'N', a_da, A, B, b_db, C
                    )
                end
            end
        end...,
    )

    # syrk! / herk! — matrix input
    # syrk! accepts trans ∈ {'N','T'}; herk! (complex) accepts trans ∈ {'N','C'}
    syrk_herk_trans = P <: BlasComplexFloat ? ['N', 'C'] : ['N', 'T']
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123460)
            map_prod(uplos, syrk_herk_trans, αβs, dαβs) do (ul, t, (α, β), (dα, dβ))
                f = P <: BlasComplexFloat ? BLAS.herk! : BLAS.syrk!
                # herk! requires real-valued α, β (relty = real(P) for complex P)
                ra = P <: BlasComplexFloat ? real(P)(real(α)) : P(α)
                rb = P <: BlasComplexFloat ? real(P)(real(β)) : P(β)
                rda = P <: BlasComplexFloat ? real(P)(real(dα)) : P(dα)
                rdb = P <: BlasComplexFloat ? real(P)(real(dβ)) : P(dβ)
                nA, kA = t == 'N' ? (3, 2) : (2, 3)
                As = blas_matrices(rng, P, nA, kA)
                Cs = blas_matrices(rng, P, 3, 3)
                return map(As, Cs) do A, C
                    a_da = CoDual(ra, rda)
                    b_db = CoDual(rb, rdb)
                    (false, perf_flag, nothing, f, ul, t, a_da, A, b_db, C)
                end
            end
        end...,
    )

    # syrk! / herk! — vector input (fixes issue #786: mul!(C, v, v') via BLAS.syrk!)
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123461)
            map_prod(uplos, αβs, dαβs) do (ul, (α, β), (dα, dβ))
                f = P <: BlasComplexFloat ? BLAS.herk! : BLAS.syrk!
                ra = P <: BlasComplexFloat ? real(P)(real(α)) : P(α)
                rb = P <: BlasComplexFloat ? real(P)(real(β)) : P(β)
                rda = P <: BlasComplexFloat ? real(P)(real(dα)) : P(dα)
                rdb = P <: BlasComplexFloat ? real(P)(real(dβ)) : P(dβ)
                vs = blas_vectors(rng, P, 3; only_contiguous=true)
                Cs = blas_matrices(rng, P, 3, 3)
                return map(vs, Cs) do v, C
                    a_da = CoDual(ra, rda)
                    b_db = CoDual(rb, rdb)
                    (false, perf_flag, nothing, f, ul, 'N', a_da, v, b_db, C)
                end
            end
        end...,
    )

    # trmm!
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123456)
            map_prod(
                ['L', 'R'], uplos, t_flags, dAs, [1, 3], [1, 2], dαs
            ) do (side, ul, tA, dA, M, N, dα)
                P <: BlasRealFloat && imag(dα) != 0 && return []

                R = side == 'L' ? M : N
                As = blas_matrices(rng, P, R, R)
                Bs = blas_matrices(rng, P, M, N)
                return map(As, Bs) do A, B
                    α_dα = CoDual(randn(rng, P), P(dα))
                    (
                        false, perf_flag, nothing, BLAS.trmm!, side, ul, tA, dA, α_dα, A, B
                    )
                end
            end
        end...,
    )

    # Lowercase flags must select the same derivative branches as their uppercase forms.
    # Keep uplo uppercase: Julia's wrapper validates it before calling BLAS.
    let
        rng = rng_ctor(123456)
        A = blas_matrices(rng, P, 3, 3)[1]
        B = blas_matrices(rng, P, 3, 2)[1]
        push!(
            test_cases,
            (
                false,
                perf_flag,
                nothing,
                BLAS.trmm!,
                'L',
                'U',
                'n',
                'u',
                randn(rng, P),
                A,
                B,
            ),
        )
    end

    # trsm!
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123456)
            map_prod(
                ['L', 'R'], uplos, t_flags, dAs, [1, 3], [1, 2]
            ) do (side, ul, tA, dA, M, N)
                R = side == 'L' ? M : N
                a = randn(rng, P)
                As = map(blas_matrices(rng, P, R, R)) do A
                    A[diagind(A)] .+= 1
                    return A
                end
                Bs = blas_matrices(rng, P, M, N)
                return map(As, Bs) do A, B
                    (false, perf_flag, nothing, BLAS.trsm!, side, ul, tA, dA, a, A, B)
                end
            end
        end...,
    )

    flags = (false, :stability, (mode=ReverseMode,))
    both_modes = (false, :stability, nothing)
    for n in (0, 1)
        x, y = view(P[3], 1:2:1), view(P[2], 1:2:1)
        append!(
            test_cases,
            [
                (both_modes..., BLAS.nrm2, n, x, 1),
                (flags..., BLAS.scal!, n, P(2), x, 1),
                (flags..., BLAS.axpy!, n, P(2), x, 1, y, 1),
            ],
        )
    end
    append!(
        test_cases,
        [
            (both_modes..., BLAS.nrm2, 2, zeros(P, 2), 1),
            (
                false,
                :stability_and_allocs,
                nothing,
                BLAS.nrm2,
                3,
                transpose(P[1 2; 3 4]),
                1,
            ),
            (flags..., BLAS.axpy!, 3, P(2), transpose(P[1 2; 3 4]), 1, zeros(P, 3), 1),
            (
                false,
                :stability_and_allocs,
                nothing,
                BLAS.nrm2,
                2,
                view(P[3 0; 4 0; 9 0], 1:2, :),
                1,
            ),
            (flags..., BLAS.nrm2, 2, view(P[9 9; 3 4; 9 9], 2:-1:1, :), 3),
            (flags..., BLAS.scal!, 2, P(2), view(P[3 0; 4 0; 9 0], 1:2, :), 1),
            (
                flags...,
                BLAS.axpy!,
                2,
                P(2),
                view(P[3 0; 4 0; 9 0], 1:2, :),
                1,
                view(zeros(P, 3, 2), 1:2, :),
                1,
            ),
            (flags..., BLAS.nrm2, 2, view(P[3, 9, 4, 9], 1:2:4), 2),
            (flags..., BLAS.scal!, 2, P(2), view(P[3, 9, 4, 9], 1:2:4), 2),
            (both_modes..., BLAS.gemv!, 'N', P(2), zeros(P, 2, 0), P[], P(3), ones(P, 2)),
            (both_modes..., BLAS.gemv!, 'n', P(2), P[1 2; 3 4], P[1, 2], P(3), P[3, 4]),
            (
                both_modes...,
                BLAS.gemv!,
                'n',
                P(2),
                view(P[1 2; 3 4], :, 2:-1:1),
                P[1, 2],
                P(3),
                P[3, 4],
            ),
        ],
    )
    for f in (BLAS.trmm!, BLAS.trsm!)
        push!(
            test_cases,
            (flags..., f, 'L', 'U', 'N', 'N', zero(P), P[2 1; 0 3], ones(P, 2, 2)),
        )
    end
    flags = (
        false, :none, (mode=ReverseMode, throws=(ArgumentError, "does not support operand"))
    )
    push!(
        test_cases,
        (flags..., BLAS.nrm2, 2, view(P[3, 9, 4, 9], 1:2:4), 1),
        (flags..., BLAS.scal!, 2, P(2), view(P[3 0; 4 0; 9 0], 1:2, :), 0),
    )
    append!(test_cases, _blas_flag_test_cases(P))
    # The unscaled solve overflows, but the scaled primal and JVP are finite. Pin dα=0
    # so finite differences can perturb A and B without overflowing the scaled result.
    let
        α = P(real(P) === Float32 ? 1e-30 : 1e-300)
        A = fill(P(0.25), 1, 1)
        B = fill(P(real(P) === Float32 ? 1e38 : 1e308), 1, 1)
        opts = (mode=ForwardMode,)
        push!(
            test_cases,
            (false, :none, opts, BLAS.trsm!, 'L', 'U', 'N', 'N', CoDual(α, zero(P)), A, B),
        )
    end

    # trmm!/trsm! reverse ∇α at α=0: the pullback's `dot(B,dB)/α'` is 0/0 there (the primal zeroed
    # B), so it recomputes the finite gradient from the saved input. One α=0 case per op suffices
    # (the rule is linear in α; α≠0 is covered by the random-α cases above).
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123456)
            A = randn(rng, P, 2, 2)
            Ainv = copy(A)
            Ainv[diagind(Ainv)] .+= 1
            B = randn(rng, P, 2, 2)
            [
                (false, :none, nothing, f, 'L', 'U', 'N', 'N', zero(P), M, copy(B)) for
                (f, M) in ((BLAS.trmm!, A), (BLAS.trsm!, Ainv))
            ]
        end,
    )

    # symm! (all BlasFloat) / hemm! (complex only): C ← α·A·B + β·C for side='L' (A is M×M) or
    # α·B·A + β·C for side='R' (A is N×N); A is symmetric (symm!) / Hermitian (hemm!), read through
    # the `uplo` triangle.
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123462)
            fs = P <: BlasComplexFloat ? (BLAS.symm!, BLAS.hemm!) : (BLAS.symm!,)
            map_prod(
                fs, ['L', 'R'], uplos, [1, 3], [1, 2], dαs
            ) do (f, side, ul, M, N, dα)
                P <: BlasRealFloat && imag(dα) != 0 && return []
                R = side == 'L' ? M : N
                As = blas_matrices(rng, P, R, R)
                Bs = blas_matrices(rng, P, M, N)
                Cs = blas_matrices(rng, P, M, N)
                return map(As, Bs, Cs) do A, B, C
                    α_dα = CoDual(randn(rng, P), P(dα))
                    β_dβ = CoDual(randn(rng, P), randn(rng, P))
                    (false, perf_flag, nothing, f, side, ul, α_dα, A, B, β_dβ, C)
                end
            end
        end...,
    )

    # Square operands also satisfy Julia's case-sensitive dimension checks for lowercase flags.
    let
        rng = rng_ctor(123463)
        A = randn(rng, P, 3, 3)
        B = randn(rng, P, 3, 3)
        C = randn(rng, P, 3, 3)
        x, y = randn(rng, P, 3), randn(rng, P, 3)
        for t in ('n', 't', 'c')
            push!(
                test_cases,
                (
                    false,
                    :stability,
                    nothing,
                    BLAS.gemv!,
                    t,
                    P(0.7),
                    copy(A),
                    copy(x),
                    P(0.3),
                    copy(y),
                ),
            )
        end
        fs = P <: BlasComplexFloat ? (BLAS.symm!, BLAS.hemm!) : (BLAS.symm!,)
        for f in fs, side in ('l', 'r')
            push!(
                test_cases,
                (
                    false,
                    perf_flag,
                    nothing,
                    f,
                    side,
                    'U',
                    P(0.7),
                    copy(A),
                    copy(B),
                    P(0.3),
                    copy(C),
                ),
            )
        end
        f = P <: BlasComplexFloat ? BLAS.herk! : BLAS.syrk!
        for t in ('n', P <: BlasComplexFloat ? 'c' : 't')
            push!(
                test_cases,
                (
                    false,
                    perf_flag,
                    nothing,
                    f,
                    'U',
                    t,
                    real(P)(0.7),
                    copy(A),
                    real(P)(0.3),
                    copy(C),
                ),
            )
        end
    end

    throwing_rows, throwing_memory = _blas_throwing_rows(P)
    test_cases = vcat(Any[test_cases...], Any[_throwing_row(c) for c in throwing_rows])
    append!(test_cases, _blas_alias_test_cases(P))
    memory = throwing_memory
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:blas}, P::Type{<:BlasFloat})
    t_flags = ['N', 'T', 'C']
    rng = rng_ctor(123)
    test_cases = Any[]
    for cols in (1:2, 1:2:3)
        output = cols == 1:2 ? (7:8) : (3:4)
        f =
            v -> BLAS.gemv!(
                'N',
                one(P),
                view(reshape(v, 2, 4), :, cols),
                P[1, 2],
                one(P),
                view(v, output),
            )
        push!(test_cases, (false, :none, nothing, f, P.(1:8)))
    end
    push!(
        test_cases,
        (
            false,
            :none,
            nothing,
            A -> BLAS.gemm!('N', 'N', one(P), A, A, zero(P), A),
            zeros(P, 0, 0),
        ),
    )

    #
    # BLAS LEVEL 1
    #

    push!(test_cases, (false, :none, (mode=ReverseMode,), x -> dot(x, x), P[]))

    # dot (real types only)
    if P <: BlasRealFloat
        flags = (false, :none, nothing)
        dot_flags = (false, :none, (skip_chunked=true,))
        append!(
            test_cases,
            [
                # `skip_chunked`: the strided `dot` walks both arguments through raw pointers,
                # which the element-major partials block cannot serve at width > 1 (stride N per
                # lane). The sibling rows below take the same shape but pass, so they are left on.
                (dot_flags..., BLAS.dot, 3, randn(rng, P, 5), 1, randn(rng, P, 4), 1),
                (dot_flags..., BLAS.dot, 3, randn(rng, P, 6), 2, randn(rng, P, 4), 1),
                (dot_flags..., BLAS.dot, 3, randn(rng, P, 6), 1, randn(rng, P, 9), 3),
                (dot_flags..., BLAS.dot, 3, randn(rng, P, 12), 3, randn(rng, P, 9), 2),
            ],
        )
    end

    # dotc, dotu (complex types only)
    if !(P <: BlasRealFloat)
        flags = (false, :none, nothing)
        for f in [BLAS.dotc, BLAS.dotu]
            append!(
                test_cases,
                [
                    (flags..., f, 3, randn(rng, P, 5), 1, randn(rng, P, 4), 1),
                    (flags..., f, 3, randn(rng, P, 6), 2, randn(rng, P, 4), 1),
                    (flags..., f, 3, randn(rng, P, 6), 1, randn(rng, P, 9), 3),
                    (flags..., f, 3, randn(rng, P, 12), 3, randn(rng, P, 9), 2),
                    # Differently-typed pair (dense Vector + strided SubArray): the @is_primitive
                    # binds the two array args to independent type vars, so the pair stays a forward
                    # primitive. A strided operand is read out of view order by the block loop, so
                    # this hits the per-lane BLAS fallback (correct at all widths, less efficient).
                    (flags..., f, 4, randn(rng, P, 4), 1, view(randn(rng, P, 8), 1:2:8), 1),
                    # Walk running past the view into its parent: BLAS reads raw memory, so the
                    # elements beyond the view belong to the parent and carry partials.
                    # `_dot_walk_widen` re-expresses the operand over that parent.
                    (
                        flags...,
                        f,
                        30,
                        view(randn(rng, P, 40), 1:4),
                        1,
                        view(randn(rng, P, 40), 1:4),
                        1,
                    ),
                ],
            )
        end
    end

    # nrm2
    push!(
        test_cases,
        (
            false,
            :none,
            (mode=ReverseMode,),
            A -> BLAS.nrm2(2, view(A, 1:2, :), 1),
            P[3 0; 4 0; 9 0],
        ),
    )
    push!(test_cases, (false, :none, nothing, BLAS.nrm2, randn(rng, P, 105)))

    #
    # BLAS LEVEL 3
    #

    # aliased gemm! — uses a fresh rng to avoid depending on the state left by the
    # level-1/2 tests above.
    aliased_gemm! = (tA, tB, a, b, A, C) -> BLAS.gemm!(tA, tB, a, A, A, b, C)
    rng_gemm = rng_ctor(123)
    append!(
        test_cases,
        map_prod(t_flags, t_flags) do (tA, tB)
            As = blas_matrices(rng_gemm, P, 5, 5)
            Bs = blas_matrices(rng_gemm, P, 5, 5)
            a = randn(rng_gemm, P)
            b = randn(rng_gemm, P)
            return map_prod(As, Bs) do (A, B)
                (false, :none, nothing, aliased_gemm!, tA, tB, a, b, A, B)
            end
        end...,
    )

    # Build the view inside the call: the direct numerical harness copies immutable
    # view arguments separately, losing their shared primal storage.
    self_axpby! = (a, x, b) -> begin
        v = view(x, 1:2:length(x))
        BLAS.axpby!(a, v, b, v)
        return x
    end
    for b in (0, 3)
        push!(test_cases, (false, :none, nothing, self_axpby!, P(2), P[1, 2, 3, 4], P(b)))
    end

    # Summing the coefficients first overflows, although each scaled cotangent is finite.
    scaled_self_axpby = x -> begin
        R = real(eltype(x))
        a = eltype(x)(R(0.75) * floatmax(R))
        y = copy(x)
        BLAS.axpby!(a, y, a, y)
        return real(first(y)) / real(a)
    end
    push!(test_cases, (false, :none, (mode=ReverseMode,), scaled_self_axpby, zeros(P, 2)))
    cancelled_self_axpby = x -> begin
        R = real(eltype(x))
        a = eltype(x)(2)
        y = copy(x)
        BLAS.axpby!(a, y, -a, y)
        return (R(0.75) * floatmax(R)) * real(first(y))
    end
    push!(
        test_cases,
        (false, :none, (output_tangent=one(real(P)),), cancelled_self_axpby, zeros(P, 2)),
    )

    memory = Any[]
    return test_cases, memory
end

# Tests that are not specific to any BlasFloat precision.
function hand_written_rule_test_cases(rng_ctor, ::Val{:blas_basic})
    # Removable singularity at the zero vector: the nrm2 frule (`s/(2y)`) and reverse pullback
    # (`X*(dy/y)`) are both 0/0 there, so every lane's partial and the gradient must be 0, not NaN.
    return Any[(false, :none, nothing, BLAS.nrm2, 3, zeros(3), 1)], Any[]
end
function derived_rule_test_cases(rng_ctor, ::Val{:blas_basic})
    test_cases = Any[
        (false, :stability, nothing, BLAS.get_num_threads),
        (false, :stability, nothing, BLAS.lbt_get_num_threads),
        (false, :stability, nothing, BLAS.set_num_threads, 1),
        (false, :stability, nothing, BLAS.lbt_set_num_threads, 1),
        (false, :none, nothing, x -> sum(complex(x) * x), rand(rng_ctor(123), 5, 5)),
    ]
    return test_cases, Any[]
end

function _blas_throwing_rows(P::Type{<:BlasFloat})
    # Nondivisible stride: BLAS walks elements outside the view. Divisible strides
    # and nrm2's one-argument form are covered as working cases.
    x = view(P[i for i in 1:10], 1:2:10)
    # Unit first-dimension stride but non-dense: logical index 4 is raw offset 6.
    m = view(reshape(P[i for i in 1:25], 5, 5), 1:3, 1:2)
    # Widening cannot support a walk that also leaves the parent.
    short = view(P[i for i in 1:40], 36:39)
    cases = Any[
        ((ArgumentError, "does not support operand"), BLAS.scal!, (5, P(2), x, 1), (;)),
        ((ArgumentError, "does not support operand"), BLAS.scal!, (6, P(2), m, 1), (;)),
        ((ArgumentError, "does not support operand"), BLAS.nrm2, (6, m, 1), (;)),
        ((ArgumentError, "does not support operand"), BLAS.nrm2, (5, x, 1), (;)),
    ]
    # A bare `Ptr` input can only be seeded with the `uninit_*` placeholder -- its own primal
    # address -- so without the guards both modes write derivatives over `xs`/`ys` themselves.
    # Real `BLAS.dot` has a reverse pointer rule but no forward one; `dotc`/`dotu` have both.
    xs = P[i for i in 1:3]
    ys = P[i for i in 4:6]
    placeholder = (ArgumentError, "tangent is the placeholder")
    two_operand = if P <: Real
        ((BLAS.dot, (; mode=ReverseMode)),)
    else
        ((BLAS.dotc, (;)), (BLAS.dotu, (;)))
    end
    append!(
        cases,
        Any[
            (placeholder, BLAS.nrm2, (3, pointer(xs), 1), (;)),
            (placeholder, BLAS.scal!, (3, P(2), pointer(xs), 1), (;)),
            (placeholder, BLAS.axpy!, (3, P(2), pointer(xs), 1, pointer(ys), 1), (;)),
            ((placeholder, f, (3, pointer(xs), 1, pointer(ys), 1), opts) for
             (f, opts) in two_operand)...,
        ],
    )
    # Complex forward dot must reject walks past the parent. Real dot takes a
    # different path; reverse uses tangent pointers with no operand length to check.
    P <: Complex && append!(
        cases,
        Any[
            (
                (ArgumentError, "runs past the"),
                f,
                (30, short, 1, short, 1),
                (; mode=ForwardMode),
            ) for f in (BLAS.dotc, BLAS.dotu)
        ],
    )

    return cases, Any[x, m, xs, ys, short]
end

# One Val per BlasFloat precision; each runs all BLAS tests for that type so GC can
# reclaim one precision's arrays before the next is allocated.
for P in (Float64, Float32, ComplexF64, ComplexF32)
    sym = Symbol(:blas_, P)
    @eval function hand_written_rule_test_cases(rng_ctor, ::Val{$(QuoteNode(sym))})
        return hand_written_rule_test_cases(rng_ctor, Val(:blas), $P)
    end
    @eval function derived_rule_test_cases(rng_ctor, ::Val{$(QuoteNode(sym))})
        return derived_rule_test_cases(rng_ctor, Val(:blas), $P)
    end
end

# Aliases are intentional: the registry seeds and copies them with shared caches.
function _blas_alias_test_cases(P)
    rows = Any[]
    for b in (0, 3)
        x = P[1, 2, 3, 4]
        a = P <: Complex ? P(2 + im) : P(2)
        push!(rows, (false, :stability, nothing, BLAS.axpby!, a, x, P(b), x))
    end
    flags = (false, :none, (throws=(ArgumentError, "overlapping input and output"),))
    x = P[1, 2, 3]
    P <: Real && push!(rows, (flags..., BLAS.axpy!, P(2), view(x, 1:2), view(x, 2:3)))
    push!(rows, (flags..., BLAS.axpby!, P(2), view(x, 1:2), P(3), view(x, 2:3)))
    A = P[2 1; 1 3]
    v = P[1, 2]
    for f in (BLAS.gemv!, BLAS.symv!, (P <: Complex ? (BLAS.hemv!,) : ())...)
        flag = f === BLAS.gemv! ? 'N' : 'U'
        push!(rows, (flags..., f, flag, P(2), A, v, P(3), v))
        push!(rows, (flags..., f, flag, P(2), A, v, P(3), view(A, :, 1)))
    end
    for f in (BLAS.gemm!, BLAS.symm!, (P <: Complex ? (BLAS.hemm!,) : ())...)
        chars = f === BLAS.gemm! ? ('N', 'N') : ('L', 'U')
        B = copy(A)
        for C in (A, B)
            push!(rows, (flags..., f, chars..., P(2), A, B, P(3), C))
        end
    end
    for f in (BLAS.syrk!, (P <: Complex ? (BLAS.herk!,) : ())...)
        Q = f === BLAS.herk! ? real(P) : P
        push!(rows, (flags..., f, 'U', 'N', Q(2), A, Q(3), A))
    end
    for f in (BLAS.trmv!, BLAS.trsv!)
        push!(rows, (flags..., f, 'U', 'N', 'N', A, view(A, :, 1)))
    end
    for f in (BLAS.trmm!, BLAS.trsm!)
        push!(rows, (flags..., f, 'L', 'U', 'N', 'N', P(2), A, A))
    end
    return rows
end

function _blas_flag_test_cases(P)
    A = P[2 1; 0 3]
    B = P[1 2; 3 4]
    x = P[1, 2]
    flags = (false, :none, (throws=(ArgumentError, "uplo argument must be"),))
    rows = Any[]
    for f in (BLAS.symv!, (P <: Complex ? (BLAS.hemv!,) : ())...)
        push!(rows, (flags..., f, 'u', P(2), A, x, P(3), copy(x)))
    end
    for f in (BLAS.trmv!, BLAS.trsv!)
        push!(rows, (flags..., f, 'u', 'N', 'N', A, x))
    end
    for f in (BLAS.symm!, (P <: Complex ? (BLAS.hemm!,) : ())...)
        push!(rows, (flags..., f, 'L', 'u', P(2), A, B, P(3), copy(B)))
    end
    for f in (BLAS.syrk!, (P <: Complex ? (BLAS.herk!,) : ())...)
        Q = f === BLAS.herk! ? real(P) : P
        push!(rows, (flags..., f, 'u', 'N', Q(2), A, Q(3), copy(B)))
    end
    for f in (BLAS.trmm!, BLAS.trsm!)
        push!(rows, (flags..., f, 'L', 'u', 'N', 'N', P(2), A, B))
    end
    flags = (false, :none, (throws=(DimensionMismatch, nothing),))
    push!(
        rows,
        (
            flags...,
            BLAS.gemm!,
            'n',
            'N',
            P(2),
            ones(P, 2, 3),
            ones(P, 3, 4),
            P(1),
            ones(P, 2, 4),
        ),
    )
    for f in (BLAS.symm!, (P <: Complex ? (BLAS.hemm!,) : ())...)
        push!(rows, (flags..., f, 'l', 'U', P(2), A, ones(P, 2, 3), P(1), ones(P, 2, 3)))
    end
    for f in (BLAS.syrk!, (P <: Complex ? (BLAS.herk!,) : ())...)
        Q = f === BLAS.herk! ? real(P) : P
        push!(rows, (flags..., f, 'U', 'n', Q(2), ones(P, 2, 3), Q(1), copy(A)))
    end
    for f in (BLAS.trmm!, BLAS.trsm!)
        push!(rows, (flags..., f, 'l', 'U', 'N', 'N', P(2), A, ones(P, 2, 3)))
    end
    flags = (false, :stability, nothing)
    for trans in ('n', 't', 'c')
        push!(rows, (flags..., BLAS.gemm!, trans, trans, P(2), A, B, P(3), copy(B)))
        for f in (BLAS.trmv!, BLAS.trsv!)
            push!(rows, (flags..., f, 'U', trans, 'u', A, copy(x)))
        end
        for f in (BLAS.trmm!, BLAS.trsm!), side in ('l', 'r')
            # Julia 1.10 cannot infer the triangular pullback's matrix product.
            perf_flag = VERSION < v"1.11-" ? :none : :stability
            push!(
                rows,
                (false, perf_flag, nothing, f, side, 'U', trans, 'u', P(2), A, copy(B)),
            )
        end
    end
    for f in (BLAS.symm!, (P <: Complex ? (BLAS.hemm!,) : ())...), side in ('l', 'r')
        push!(rows, (flags..., f, side, 'U', P(2), A, B, P(3), copy(B)))
    end
    for f in (BLAS.syrk!, (P <: Complex ? (BLAS.herk!,) : ())...)
        Q = f === BLAS.herk! ? real(P) : P
        for trans in ('n', f === BLAS.herk! ? 'c' : 't')
            push!(rows, (flags..., f, 'U', trans, Q(2), A, Q(3), copy(B)))
        end
    end
    return rows
end
