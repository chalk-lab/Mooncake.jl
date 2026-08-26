function blas_name(name::Symbol)
    return (BLAS.USE_BLAS64 ? Symbol(name, "64_") : name, Symbol(BLAS.libblastrampoline))
end

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
function arrayify(x::SubArray{P,B,C,D,E}, dx::TangentOrFData) where {P<:BlasFloat,B,C,D,E}
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

# Forward-mode analogue of `arrayify(::CoDual)`. Returns the primal array and an N-tuple of
# per-lane tangent arrays, each canonicalised to the primal's wrapper. This mirrors the reverse
# `arrayify` wrapper methods, applied across the parallel per-lane partials: no copy — BLAS/LAPACK run on the
# (possibly strided) views directly, and in-place writes flow back through the view into the
# parent's partials.
function arrayify(x::Lifted{<:AbstractArray{P},N}) where {P<:BlasFloat,N}
    A = primal(x)
    return A, ntuple(lane -> _arrayify_lane(A, tangent(x), lane), Val(N))
end
# `_arrayify_lane` is the per-wrapper analogue of a reverse `arrayify` method, applied per lane: it
# recurses through the wrapper's V (`ImmutableDual` whose NamedTuple mirrors the wrapper's fields)
# and re-wraps, exactly as reverse `arrayify` recurses through the tangent. Base case: a dense
# `NDualArray`'s lane partial.
# The trailing `Val{dense}` selects the dense leaf's backing: `Val(false)` (the 3-arg default,
# used by the block-op callers gemm/gemv/_partials_block) returns the lazy stride-`N` block-row
# view; `Val(true)` `collect`s it, so the reconstructed wrapper is backed by CONTIGUOUS memory —
# needed by the `dotc`/`dotu` non-contiguous fallback, where BLAS must read an operand's partials
# exactly as it reads the primal. `dense` is a static parameter, so the leaf branch constant-folds.
@inline _arrayify_lane(x, V, lane::Integer) = _arrayify_lane(x, V, lane, Val(false))
@inline _dense_lane_partial(x::Lifted, k::Integer) = _arrayify_lane(
    primal(x), tangent(x), k, Val(true)
)
@inline _arrayify_lane(
    ::DenseArray, V::NDualArray, lane::Integer, ::Val{dense}
) where {dense} = dense ? collect(tangent_view(V, lane)) : tangent_view(V, lane)
@inline _arrayify_lane(::Ptr, V::NTuple{N,<:Ptr}, lane::Integer, ::Val) where {N} = V[lane]
@inline function _arrayify_lane(
    x::SubArray{P,B,C,D,E}, V::ImmutableDual, lane::Integer, d::Val
) where {P,B,C,D,E}
    pp = _arrayify_lane(x.parent, V.value.parent, lane, d)
    return SubArray{P,B,typeof(pp),D,E}(pp, x.indices, x.offset1, x.stride1)
end
@inline function _arrayify_lane(
    x::Base.ReshapedArray{P,B,C,D}, V::ImmutableDual, lane::Integer, d::Val
) where {P,B,C,D}
    pp = _arrayify_lane(x.parent, V.value.parent, lane, d)
    return Base.ReshapedArray{P,B,typeof(pp),D}(pp, x.dims, x.mi)
end
@inline _arrayify_lane(x::Adjoint, V::ImmutableDual, lane::Integer, d::Val) = adjoint(
    _arrayify_lane(x.parent, V.value.parent, lane, d)
)
@inline _arrayify_lane(x::Transpose, V::ImmutableDual, lane::Integer, d::Val) = transpose(
    _arrayify_lane(x.parent, V.value.parent, lane, d)
)
@inline _arrayify_lane(x::Diagonal, V::ImmutableDual, lane::Integer, d::Val) = Diagonal(
    _arrayify_lane(x.diag, V.value.diag, lane, d)
)
@inline _arrayify_lane(x::Symmetric, V::ImmutableDual, lane::Integer, d::Val) = Symmetric(
    _arrayify_lane(x.data, V.value.data, lane, d), Symbol(x.uplo)
)
# All four triangular wrappers (Upper/Lower and the Unit variants) share a `.data` field and a
# `Tx(data)` constructor, so one `AbstractTriangular` method covers them — mirroring the reverse
# `arrayify(::AbstractTriangular)`.
@inline _arrayify_lane(x::Tx, V::ImmutableDual, lane::Integer, d::Val) where {Tx<:LinearAlgebra.AbstractTriangular} = Tx(
    _arrayify_lane(x.data, V.value.data, lane, d)
)
@inline _arrayify_lane(x::Base.ReinterpretArray{T}, V::ImmutableDual, lane::Integer, d::Val) where {T} = reinterpret(
    T, _arrayify_lane(x.parent, V.value.parent, lane, d)
)

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
    xinds = 1:incx:(incx * n)
    return (
        view(unsafe_wrap(Vector{P}, x, n * incx), xinds),
        view(unsafe_wrap(Vector{P}, dx, n * incx), xinds),
    )
end
function viewify(
    n::BLAS.BlasInt, x_dx::CoDual{A}, incx::BLAS.BlasInt
) where {A<:AbstractArray{<:BlasFloat}}
    x, dx = arrayify(x_dx)
    xinds = 1:incx:(incx * n)
    return view(x, xinds), view(dx, xinds)
end

#
# Utility
#

@zero_derivative MinimalCtx Tuple{typeof(BLAS.get_num_threads)}
@zero_derivative MinimalCtx Tuple{typeof(BLAS.lbt_get_num_threads)}
@zero_derivative MinimalCtx Tuple{typeof(BLAS.set_num_threads),Union{Integer,Nothing}}
@zero_derivative MinimalCtx Tuple{typeof(BLAS.lbt_set_num_threads),Any}

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

    # Forward mode: only real `dot` (cblas returns the result by value) is handled at
    # the foreigncall boundary. Complex `dotc`/`dotu` (which write into a scalar
    # result `Ref`) are forward primitives at the `BLAS.dotc`/`dotu` level instead —
    # see below. Reverse mode handles all three here.
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
                dDX_partials = tangent(_DX)
                dDY_partials = tangent(_DY)

                result = BLAS.$jlfname(n, DX, incx, DY, incy)
                dresult_lanes = ntuple(Val(Nw)) do lane
                    return BLAS.$jlfname(n, dDX_partials[lane], incx, DY, incy) +
                           BLAS.$jlfname(n, DX, incx, dDY_partials[lane], incy)
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
                            _dDX .+= DY .* dv
                            _dDY .+= DX .* dv
                        end
                        return tuple_fill(NoRData(), Val(N + 11))
                    end
                end
            elseif jlfname == :dotc
                quote
                    function dot_pb!!(::NoRData)
                        GC.@preserve args begin
                            dv = Base.unsafe_load(_dpresult)
                            _dDX .+= DY .* dv'
                            _dDY .+= DX .* dv
                        end
                        return tuple_fill(NoRData(), Val(N + 12))
                    end
                end
            else
                quote
                    function dot_pb!!(::NoRData)
                        GC.@preserve args begin
                            dv = Base.unsafe_load(_dpresult)
                            _dDX .+= conj.(DY) .* dv
                            _dDY .+= conj.(DX) .* dv
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
# `LinearAlgebra.norm2` calls the ONE-argument `BLAS.nrm2(x)` for length >= 32, and Julia inlines
# both it and the `ccall` it wraps, so the three-argument boundary declared above no longer exists
# by the time a rule could fire and the raw pointer reaches the transform. Declaring the
# one-argument form keeps a boundary that survives inlining; the rules below just supply the length
# and stride the wrapper would have computed.
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
# BLAS Lifted parallels — each rule iterates lanes and calls the BLAS
# routine on the per-lane partial array (or Ptr) directly. Supports both
# `Array{T, D}` slots (NDualArray V) and `Ptr{T}` slots (NTuple{N, Ptr{T}}
# V); real and complex element types both routed.
#
# `_blas_lane_partial` extracts a Lifted matrix/vector/Ptr slot's per-lane partial
# in the right shape, via the wrapper-aware `_arrayify_lane`. This covers dense
# `NDualArray`, the `NTuple{N, Ptr}` V, and the wrapper structural-lift V's
# (SubArray/ReshapedArray/Adjoint/…), so the slot may be any `AbstractVecOrMat`.
@inline _blas_lane_partial(x::Lifted, lane::Integer) = _arrayify_lane(
    primal(x), tangent(x), lane
)

# ── Lane-leading partials blocks for BLAS/LAPACK forward rules ──────────────────

# `_partials_block(x)` returns `(block, copied)`: a dense lane-leading array of shape
# `(N, size(primal(x))...)` holding the slot's per-lane partials — lane `k` of element `i`
# is `block[k, i...]`, and each element's `N` lanes are one contiguous column. For a
# dense-primal slot this is the `NDualArray`'s own block (`copied == false`): mutations
# land in the slot directly. Wrapper primals (SubArray/Reshaped/Adjoint/…) gather the
# per-lane views into a fresh block (`copied == true`); a rule that mutates such a block
# must `_write_back_partials!` it into the slot afterwards.
#
# Why dense blocks: BLAS/LAPACK matrix arguments need unit first-dim stride. A single lane
# of the element-major block is stride-`N` (never BLAS-compatible for `N > 1`), but the
# block itself IS BLAS-compatible with the lane axis leading, so a lane-invariant linear
# map applies to all `N` lanes in one wide call: right-multiplying the `(N, len)` lane
# matrix by the map's transpose batches every lane (see the per-rule comments).
@inline function _partials_block(
    x::Lifted{P,N,<:NDualArray}
) where {T,D,P<:AbstractArray{T,D},N}
    return getfield(tangent(x), :partials_block), false
end
@inline function _partials_block(x::Lifted{P,N}) where {T,D,P<:AbstractArray{T,D},N}
    p = primal(x)
    blk = Array{T,D + 1}(undef, (N, size(p)...))
    colons = ntuple(_ -> Colon(), Val(D))
    for k in 1:N
        copyto!(view(blk, k, colons...), _blas_lane_partial(x, k))
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

# `β`-scale a tangent block with BLAS's β semantics: `β == 0` overwrites (strong zero)
# rather than multiplying, so a NaN already in the tangent cannot leak through `0 * NaN`.
# `contracted` is the dimension BLAS sums over. BLAS takes a quick return when it is zero and does
# NOT apply `β` there, so a rule folding `β` in by hand must skip it too — otherwise the tangent is
# scaled where the primal was left alone, which is a silently wrong derivative. Reproduced for
# `gemv!` with a 0-column `A`: the primal `y` is untouched while the tangent came back `β * dy`.
@inline function _scale_or_zero!(B::AbstractArray{T}, β) where {T}
    iszero(β) ? fill!(B, zero(T)) : (B .*= β)
    return nothing
end
@inline function _scale_or_zero!(B::AbstractArray, β, contracted::Integer)
    contracted == 0 && return nothing
    return _scale_or_zero!(B, β)
end

# `X[i]*dX[i]` overflows once `norm(X)*norm(dX)` leaves `T`'s range, though the JVP it divides down
# to is representable. Scaling `X` by the power of two nearest `y` is exact, so in-range results are
# unchanged; a subnormal `y` has no representable reciprocal, and with every `X[i]` subnormal too it
# needs none. Both lane paths scale by this `r` and divide by `2(y*r)`.
@inline function _nrm2_scale_factor(y)
    r = isfinite(y) && !iszero(y) ? ldexp(one(y), -exponent(y)) : one(y)
    return isfinite(r) ? r : one(y)
end

# Per-lane nrm2 JVP: `dy_k = Σᵢ real(conj(Xᵢ)·dXₖᵢ)/y` with a removable-singularity guard. Fallback
# for the Ptr slot and strided (incx≠1) inputs, where the contiguous-block fast path does not apply.
@inline function _nrm2_lanes_perlane(
    _n, X_dX, _inc, Xv, y, ::Type{R}, ::Val{Nw}
) where {R,Nw}
    r = _nrm2_scale_factor(y)
    return ntuple(Val(Nw)) do lane
        dX_lane = _blas_lane_partial(X_dX, lane)
        dXv = _viewify_one(_n, dX_lane, _inc)
        s = zero(R)
        @inbounds for i in eachindex(Xv)
            # real(a·conj(b)) and real(conj(a)·b) are bit-identical, so their sum is exactly 2×.
            s += 2 * real((Xv[i] * r)' * dXv[i])
        end
        iszero(s) ? zero(R) : s / (2 * (y * r))
    end
end

# Block fast path: accumulate all Nw lanes in one pass over the contiguous element-major partials
# block (each element's Nw lanes are a contiguous NTuple column, so the length-Nw update vectorises,
# ~4×). Removable singularity: `s == 0` ⇒ `s/(2y)` is `0/0` ⇒ 0.
# Lane accumulate / final scale as helpers taking `acc` by value: an inline `acc = ntuple(k -> acc[k]
# + …)` would capture the reassigned `acc` in the closure, boxing it to `Any` (runtime dispatch JET
# flags). Bind only `Nw` — `NTuple{Nw,R}` degenerates to `Tuple{}` at `Nw=0`, unbinding `R` (Aqua).
@inline _nrm2_accum(acc::NTuple{Nw}, xi, col) where {Nw} = ntuple(
    k -> acc[k] + 2 * real(xi' * col[k]), Val(Nw)
)
@inline _nrm2_scale(acc::NTuple{Nw}, yr) where {Nw} = ntuple(
    k -> iszero(acc[k]) ? acc[k] : acc[k] / (2 * yr), Val(Nw)
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

# Dispatch the lane JVP on the slot kind — a function barrier that keeps the frule type-stable
# despite its `Union{Ptr,AbstractArray}` signature (`getfield(tangent(X_dX), :partials_block)` on the
# Union is a dynamic access JET flags). Array slot: block fast path with `incx == 1` (the
# block accessor `_partials_block` needs a dense, unit-stride layout); per-lane otherwise. Ptr slot:
# always per-lane (raw pointers, no block).
@inline function _nrm2_lanes(
    X_dX::Lifted{P,Nw}, _n, _inc, Xv, y, ::Type{R}
) where {T,P<:AbstractArray{T},Nw,R}
    if _inc == 1
        blk, _ = _partials_block(X_dX)
        return _nrm2_lanes_block(blk, Xv, y, R, Val(Nw))
    end
    return _nrm2_lanes_perlane(_n, X_dX, _inc, Xv, y, R, Val(Nw))
end
@inline function _nrm2_lanes(
    X_dX::Lifted{P,Nw}, _n, _inc, Xv, y, ::Type{R}
) where {T,P<:Ptr{T},Nw,R}
    return _nrm2_lanes_perlane(_n, X_dX, _inc, Xv, y, R, Val(Nw))
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
    # Both lane paths index the partials by LOGICAL linear position, as `_viewify_one` does on the
    # primal side, so neither can follow BLAS's raw walk over a non-dense operand: the derivative
    # would then be taken of different elements from the ones `nrm2` summed. Unlike `dotc`/`dotu`
    # there is no per-lane BLAS fallback to fall to, so refuse.
    _blas_raw_walk_matches(Xp, _inc) || throw(
        ArgumentError(
            LazyString(
                "Forward-mode `BLAS.nrm2` does not support operand `",
                typeof(Xp),
                "` with strides ",
                strides(Xp),
                " and `incx = ",
                _inc,
                "`: `nrm2` reads raw memory from `pointer(X)`, which follows the operand's own ",
                "elements only when it is dense and the increment is positive, so otherwise the ",
                "partials walked would not be those of the elements summed. Pass a dense operand ",
                "with a positive increment, or the raw-pointer form.",
            ),
        ),
    )
    y = BLAS.nrm2(_n, Xp, _inc)
    Xv = _viewify_one(_n, Xp, _inc)  # `viewify`-equivalent on the primal side.
    R = typeof(y)  # nrm2 returns the real-valued norm.
    return Lifted{R,Nw}(y, _scalar_ndual(y, _nrm2_lanes(X_dX, _n, _inc, Xv, y, R)))
end
# Shared single-side viewify: handles both Ptr and Array uniformly so the
# Lifted bodies don't have to branch on input shape.
@inline _viewify_one(n::Integer, x::AbstractArray, incx::Integer) = view(
    x, 1:incx:(incx * n)
)
@inline _viewify_one(n::Integer, x::Ptr{T}, incx::Integer) where {T} = view(
    unsafe_wrap(Vector{T}, x, n * incx), 1:incx:(incx * n)
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
        # Removable singularity at the zero vector: there `y == 0` (all Xᵢ == 0), so
        # `X * (dy / y)` would be `0 * Inf = NaN`. The gradient x/‖x‖ is taken as 0
        # there, matching the frule's `iszero(s)` guard.
        iszero(y) || (dX .+= X .* (dy / y))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, NoFData()), nrm2_pb!!
end

# dot(x, y) — real inner product, intercepted at the array level. `LinearAlgebra.dot` on a pair of
# strided `BlasReal` vectors inlines the cblas `ccall` directly (reading each operand's
# `.ref.ptr_or_offset`), so without this it is differentiated only at that raw-pointer
# `_foreigncall_` (below). That descends to raw pointers in *forward-over-reverse* (HVP/Hessian):
# the reverse pullback reads its fdata through those pointers, and lifting a pointer out of an
# element-major partials block is unsupported — a lane is strided (stride N), so there is no dense
# per-lane buffer to address (see `_get_lifted_field(::NDualMemoryRef, :ptr_or_offset)`).
# Intercepting `dot` at the array level keeps every access on the `NDualArray`/view path, which
# handles the block, so `dot`'s HVP/Hessian works at any chunk width. The reverse is a cheap axpy,
# so this matches the foreigncall's cost. Complex inner products are `dotc`/`dotu` (below),
# unaffected. Scoped to the concrete `Vector` (the inlining case above): disjoint from the CUDA
# extension's `dot(::CuArray, ::CuArray)` rule (no method ambiguity), and strided-wrapper operands
# — never supported through the raw-pointer path at width > 1 — stay guarded by the same throw.
@is_primitive(MinimalCtx, Tuple{typeof(dot),Vector{P},Vector{P}} where {P<:BlasRealFloat})
function frule!!(
    ::Lifted{typeof(dot),Nw}, x_dx::Lifted{Vector{P}}, y_dy::Lifted{Vector{P}}
) where {Nw,P<:BlasRealFloat}
    x, y = primal(x_dx), primal(y_dy)
    result = dot(x, y)
    # Bilinear JVP: d⟨x,y⟩ = ⟨dx,y⟩ + ⟨x,dy⟩. Each term is one matvec of the contiguous
    # (Nw, K) partials block against the primal — `out = Xblock·y + Yblock·x` in two `gemv!`s,
    # replacing the Nw strided per-lane dots (~3.5×; the length-Nw output alloc is fine — no
    # `:allocs` guard here).
    Xb = getfield(tangent(x_dx), :partials_block)
    Yb = getfield(tangent(y_dy), :partials_block)
    # Zeroed, not `undef`: `gemv` returns early on an empty operand WITHOUT applying `beta`, so
    # for `dot(P[], P[])` neither call below writes `out` and the lanes would be whatever the
    # allocator handed back — a garbage derivative, and an intermittent one.
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

# dotc/dotu (complex) — forward mode only. Unlike real `dot` (which the cblas
# routine returns by value), the complex routines write into a scalar `result =
# Ref{T}()` passed to the ccall. The canonical NDualArray-style dual of that Ref stores a
# `Complex{NDual{R,Nw}}`, which is not layout-compatible with the `Nw` contiguous
# `T`-cells the foreigncall needs, so the `_foreigncall_` frule cannot land the
# per-lane partials there. Instead we make `BLAS.dotc`/`dotu` themselves forward
# primitives and assemble the result directly, bypassing the Ref roundtrip. The
# JVP is linear: d(⟨x,y⟩) = ⟨dx,y⟩ + ⟨x,dy⟩ (with conjugation folded into the same
# routine). Reverse mode is unaffected — it still descends to the `_foreigncall_`
# rrule above.
# `BLAS.dotc`/`dotu` read each operand as a raw `pointer(op) + increment` walk, ignoring the
# operand's own memory stride. A contiguous (unit-stride) array or a `Ptr` is read the same as
# logical indexing, so the width-`N` block loop below is exact and fast. A non-contiguous operand
# (e.g. `view(v, 1:2:end)`) is read out of view order by BLAS; the element-major partials block
# cannot be walked raw, so those operands take the per-lane BLAS fallback: each lane's partials
# are rebuilt in the operand's own structure but over CONTIGUOUS memory (`_dense_lane_partial`),
# so `BLAS.$fname` reads the partial exactly as it reads the primal operand. Correct at all widths,
# but O(Nw) BLAS calls plus a materialisation per lane — hence only for the non-contiguous case.
#
# Two things must hold for the logical walk to match the raw one. The operand must be genuinely
# DENSE — unit first-dim stride is not enough above one dimension, since `view(A, 1:3, 1:2)` over a
# 5x5 has strides `(1, 5)` where a dense `(3, 2)` has `(1, 3)`, so its logical index 4 is raw offset
# 6. And the increment must be POSITIVE: BLAS starts a negative walk at `(-n+1)*inc + 1` and runs
# backwards over the same elements, where `1 + (t-1)*inc` would run off the front of the block.
@inline function _blas_raw_walk_matches(x, inc::Integer)
    inc > 0 || return false
    x isa Ptr && return true
    return strides(x) === Base.size_to_strides(1, size(x)...)
end
# `_dense_lane_partial` (defined with `_arrayify_lane` above) rebuilds a lane over CONTIGUOUS
# memory, so BLAS reads it identically to the primal operand.

# Lane accumulate as a helper taking `acc` BY VALUE, for the reason given at `_nrm2_accum` above: an
# inline `a = ntuple(k -> a[k] + …, Val(Nw))` captures the reassigned `a` and boxes it to `Any`,
# which cost ~195 B per element here (49776 B at n = 256, Nw = 1, against 64 B for this form). One
# helper per conjugation rather than per `(name, eltype)` pair — the body does not depend on the
# element type. Bind only `Nw`, as there (Aqua).
@inline _dotc_accum(acc::NTuple{Nw}, xc, yc, x, y) where {Nw} = ntuple(
    k -> acc[k] + conj(xc[k]) * y + conj(x) * yc[k], Val(Nw)
)
@inline _dotu_accum(acc::NTuple{Nw}, xc, yc, x, y) where {Nw} = ntuple(
    k -> acc[k] + xc[k] * y + x * yc[k], Val(Nw)
)

for (jlfname, elty) in
    ((:dotc, :ComplexF64), (:dotc, :ComplexF32), (:dotu, :ComplexF64), (:dotu, :ComplexF32))
    # Two independent type vars (X, Y): the two array arguments need not share a concrete type
    # (e.g. a dense `Vector` dotted with a strided `SubArray`/`Adjoint`). A single shared `X` would
    # leave differently-typed pairs non-primitive, falling to the derived forward path that cannot
    # land complex per-lane partials (the reason these are primitives at all) — the frule method
    # below already binds the two arguments independently.
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
        acc = if _blas_raw_walk_matches(DX, incx) && _blas_raw_walk_matches(DY, incy)
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
            # Non-contiguous operand: BLAS reads it out of view order, so let BLAS itself
            # contract each lane's partials (rebuilt over contiguous memory) against the primal
            # — d⟨x,y⟩ = ⟨dx,y⟩ + ⟨x,dy⟩. Correct at all widths; O(Nw) BLAS calls, less efficient.
            ntuple(Val(Nw)) do k
                dX = _dense_lane_partial(_DX, k)
                dY = _dense_lane_partial(_DY, k)
                BLAS.$jlfname(n, dX, incx, DY, incy) + BLAS.$jlfname(n, DX, incx, dY, incy)
            end
        end
        return Lifted{$elty,Nw}(result, _scalar_ndual(result, acc))
    end
    # Raw-pointer path: per-lane tangent pointers are dense buffers by the `Ptr` dual
    # protocol, so the per-lane BLAS calls below read them correctly at any width. A
    # strided block lane, by contrast, cannot be read by the pointer-based wrapper, so a
    # mixed Ptr/array call keeps a loud width guard (the all-array case takes the
    # block-based method above).
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
    # The block is element-major by LOGICAL index while BLAS scales raw memory from
    # `pointer(X)` by `incx`; those agree only for a contiguous operand. `dotc`/`dotu` fall
    # back to `_dense_lane_partial` here, but that rebuild is a copy and this rule mutates in
    # place. Supporting strided operands means updating the PARENT block at its raw offset.
    _blas_raw_walk_matches(X, incx) || throw(
        ArgumentError(
            LazyString(
                "Forward-mode `BLAS.scal!` does not support operand `",
                typeof(X),
                "` with strides ",
                strides(X),
                " and `incx = ",
                incx,
                "`: BLAS scales raw memory from `pointer(X)`, which follows the operand's own ",
                "elements only when it is dense and the increment is positive, so otherwise its ",
                "partials block cannot be updated to match. Pass a dense operand with a positive ",
                "increment, or the raw-pointer form.",
            ),
        ),
    )
    das = ntuple(k -> tangent(a_da, k), Val(Nw))
    Xb, copied = _partials_block(X_dX)
    Xbm = reshape(Xb, Nw, :)
    # Per-lane Frechet dX_k := a·dX_k + da_k·X, all lanes in one pass: each touched
    # element's lanes are one contiguous block column.
    @inbounds for t in 1:n
        i = 1 + (t - 1) * incx
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

        # Set primal to previous state.
        X .= X_copy

        # Compute gradient w.r.t. scaling.
        ∇a = dot(X, dX)

        # Compute gradient w.r.t. DX.
        BLAS.scal!(a', dX)

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
    _tA = primal(tA)
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

    # Pull out primals and tangents (the latter only where necessary).
    trans = _tA.x
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

        # BLAS quick-returns when the contracted dimension is zero and leaves `y` untouched, so the
        # primal is the identity on `y`: nothing depends on `α`, `β`, `A` or `x`. Returning here
        # also avoids the 3-arg `dot` below, which reads `first(A)` on Julia 1.10 and throws for an
        # empty `A`.
        if isempty(x)
            copyto!(y, y_copy)
            return (NoRData(), NoRData(), zero(P), NoRData(), NoRData(), zero(P), NoRData())
        end

        # Increment fdata.
        if trans == 'N'
            dalpha = dot(dy, A, x)'
            dA .+= alpha' .* dy .* x'
            BLAS.gemv!('C', alpha', A, dy, one(eltype(A)), dx)
        elseif trans == 'C' || P <: BlasRealFloat
            dalpha = dot(dy, A', x)'
            dA .+= alpha .* x .* dy'
            BLAS.gemv!('N', alpha', A, dy, one(eltype(A)), dx)
        else
            dalpha = dot(dy, transpose(A), x)'
            dA .+= alpha' .* conj.(x) .* transpose(dy)
            # Should be gemv!("conjugate only", alpha', A, dy, one(eltype(A)), dx)
            # but BLAS has no "conjugate only" gemv
            conj!(dx)
            BLAS.gemv!('N', alpha, A, conj.(dy), one(eltype(A)), dx)
            conj!(dx)
        end
        dbeta = dot(y_copy, dy)
        dy .*= beta'

        # Restore primal.
        copyto!(y, y_copy)

        # Return rdata.
        return (NoRData(), NoRData(), dalpha, NoRData(), NoRData(), dbeta, NoRData())
    end

    return gemv!_pb!!
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
        ul = primal(uplo)
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

        # Extract primals.
        ul = primal(uplo)
        α = primal(alpha)
        β = primal(beta)
        A, dA = arrayify(A_dA)
        x, dx = arrayify(x_dx)
        y, dy = arrayify(y_dy)

        y_copy = copy(y)

        BLAS.$fname(ul, α, A, x, β, y)

        function symv!_or_hemv!_adjoint(::NoRData)
            # dα = <dy, Ax>'
            if (α == 1 && β == 0)
                # Don't recompute Ax, it's already in y.
                dα = dot(dy, y)'
                BLAS.copyto!(y, y_copy)
            else
                # Reset y.
                BLAS.copyto!(y, y_copy)

                # First compute Ax with {sy,he}mv!: safe to write into memory for copy of y.
                BLAS.$fname(ul, one(T), A, x, zero(T), y_copy)
                dα = dot(dy, y_copy)'
            end

            # gradient w.r.t. A.
            # TODO: could be switched to BLAS.{sy,he}r2! if Julia ever provides it.
            dA_tmp = α' * dy * x'
            if ul == 'L'
                dA .+= LowerTriangular(dA_tmp)
                dA .+= $(isherm ? adjoint : transpose)(UpperTriangular(dA_tmp))
            else
                dA .+= $(isherm ? adjoint : transpose)(LowerTriangular(dA_tmp))
                dA .+= UpperTriangular(dA_tmp)
            end
            @inbounds for n in diagind(dA)
                dA[n] -= $(isherm ? :(real(dA_tmp[n])) : :(dA_tmp[n]))
            end

            # gradient w.r.t. x: dx += α' A' dy
            if T <: BlasRealFloat || $isherm
                # A' = A for real numbers or for hermitian matrices
                BLAS.$fname(ul, α', A, dy, one(T), dx)
            else
                # A is symmetric but complex so A' = conj(A)
                # Instead we compute conj(dx) += α A conj(dy)
                conj!(dx)
                BLAS.$fname(ul, α, A, conj.(dy), one(T), dx)
                conj!(dx)
            end

            # gradient w.r.t. beta.
            dβ = dot(y, dy)

            # gradient w.r.t. y.
            BLAS.scal!(β', dy)

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
    uplo = primal(_uplo)
    trans = primal(_trans)
    diag = primal(_diag)
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

    # Extract primals.
    uplo = primal(_uplo)
    trans = primal(_trans)
    diag = primal(_diag)
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)
    x_copy = copy(x)

    # Run primal computation.
    BLAS.trmv!(uplo, trans, diag, A, x)

    # Set dx to zero.
    dx .= zero(T)

    function trmv_pb!!(::NoRData)

        # Restore the original value of x.
        x .= x_copy

        # Increment the tangents.
        if trans == 'N'
            inc_tri!(dA, dx, x, uplo, diag)
            BLAS.trmv!(uplo, 'C', diag, A, dx)
        elseif trans == 'C' || T <: BlasRealFloat
            inc_tri!(dA, x, dx, uplo, diag)
            BLAS.trmv!(uplo, 'N', diag, A, dx)
        else
            # Equivalent to these two calls:
            # inc_tri!(dA, conj.(x), conj.(dx), uplo, diag)
            # BLAS.trmv!(uplo, "conjugate only", diag, A, dx)

            conj!(x_copy) # Reuse the memory, we don't need it anymore
            conj!(dx)
            inc_tri!(dA, x_copy, dx, uplo, diag)
            BLAS.trmv!(uplo, 'N', diag, A, dx)
            conj!(dx)
        end

        return tuple_fill(NoRData(), Val(6))
    end
    return x_dx, trmv_pb!!
end

function inc_tri!(A, x, y, uplo, diag)
    if uplo == 'L' && diag == 'U'
        @inbounds for q in 1:size(A, 2), p in (q + 1):size(A, 1)
            A[p, q] += x[p] * y[q]'
        end
    elseif uplo == 'L' && diag == 'N'
        @inbounds for q in 1:size(A, 2), p in q:size(A, 1)
            A[p, q] += x[p] * y[q]'
        end
    elseif uplo == 'U' && diag == 'U'
        @inbounds for q in 1:size(A, 2), p in 1:(q - 1)
            A[p, q] += x[p] * y[q]'
        end
    elseif uplo == 'U' && diag == 'N'
        @inbounds for q in 1:size(A, 2), p in 1:q
            A[p, q] += x[p] * y[q]'
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
    uplo = primal(_uplo)
    trans = primal(_trans)
    diag = primal(_diag)
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
    uplo = primal(_uplo)
    trans = primal(_trans)
    diag = primal(_diag)
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)

    x_copy = copy(x)

    # Primal
    BLAS.trsv!(uplo, trans, diag, A, x)

    function trsv_pb!!(::NoRData)

        # Increment dA
        if trans == 'N'
            temp = BLAS.trsv(uplo, 'C', diag, A, dx)
            temp .*= -1
            inc_tri!(dA, temp, x, uplo, diag)
        elseif trans == 'C'
            temp = BLAS.trsv(uplo, 'N', diag, A, dx)
            temp .*= -1
            inc_tri!(dA, x, temp, uplo, diag)
        else
            temp = BLAS.trsv(uplo, 'N', diag, A, conj(dx))
            temp .*= -1
            inc_tri!(dA, conj!(x), temp, uplo, diag)
        end

        # Restore initial state
        x .= x_copy

        # Compute dx
        if trans == 'T'
            # Equivalent to trsv!(uplo, "conjugate only", diag, A, dx)
            conj!(dx)
            BLAS.trsv!(uplo, 'N', diag, A, dx)
            conj!(dx)
        else
            BLAS.trsv!(uplo, trans == 'N' ? 'C' : 'N', diag, A, dx)
        end

        return tuple_fill(NoRData(), Val(6))
    end

    return x_dx, trsv_pb!!
end

#
# LEVEL 3
#

# A and B may be vectors (the rules reshape them to matrices), but the output C must be a
# matrix: both frule!! and rrule!! take `C::AbstractMatrix{T}`. Keeping the C slot at the
# broader `AbstractVecOrMat{T}` would declare a vector-C `gemm!` primitive with no matching
# rule method, giving a `MethodError` at call time instead of falling back to recursion.
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
    tA = primal(transA)
    tB = primal(transB)
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
    tA = primal(transA)
    tB = primal(transB)
    a = primal(alpha)
    b = primal(beta)
    p_A, dA = matrixify(A)
    p_B, dB = matrixify(B)
    p_C, dC = arrayify(C)

    # Save state and run primal
    p_C_copy = copy(p_C)
    tmp_ref = Ref{Matrix{T}}()

    if (a == 1 && b == 0)
        BLAS.gemm!(tA, tB, a, p_A, p_B, b, p_C)
    else
        tmp = BLAS.gemm(tA, tB, one(T), p_A, p_B)
        tmp_ref[] = tmp
        # BLAS leaves `C` unreferenced at `b == 0` and `A`/`B` at `a == 0`, so either may
        # legally hold garbage; recomputing as `a*tmp + b*C` turns a NaN there into a NaN
        # RESULT, where BLAS returns the finite value. `tmp` is still needed for `da`.
        _scale_or_zero!(p_C, b)
        iszero(a) || (p_C .+= a .* tmp)
    end

    function gemm!_pb!!(::NoRData)
        # gradient wrt alpha
        da = (a == 1 && b == 0) ? dot(p_C, dC) : dot(tmp_ref[], dC)

        # Restore state
        BLAS.copyto!(p_C, p_C_copy)

        # gradient wrt beta
        db = dot(p_C, dC)

        # gradients wrt A and B (depends on transpose flags tA and tB)
        # C = a * op(A) * op(B) + b * C
        if tA == 'N'
            # A not transposed: C = a*A*op(B) + b*C
            # dA += a' * dC * op(B)'
            Bherm = tB == 'T' ? conj(p_B) : p_B
            BLAS.gemm!('N', tB == 'N' ? 'C' : 'N', a', dC, Bherm, one(T), dA)
        elseif tA == 'C'
            # A conjugate transposed: C = a*A'*op(B) + b*C
            # dA += a * op(B) * dC'
            BLAS.gemm!(tB, 'C', a, p_B, dC, one(T), dA)
        else  # tA == 'T'
            # A transposed (complex): C = a*A^T*op(B) + b*C
            # dA += conj(a) * conj(op(B)) * transpose(dC)
            if tB == 'N'
                BLAS.gemm!('N', 'T', a', conj(p_B), dC, one(T), dA)
            else
                BLAS.gemm!(tB == 'T' ? 'C' : 'T', 'T', a', p_B, dC, one(T), dA)
            end
        end

        if tB == 'N'
            # B not transposed: C = a*op(A)*B + b*C
            # dB += a' * op(A)' * dC
            Aherm = tA == 'T' ? conj(p_A) : p_A
            BLAS.gemm!(tA == 'N' ? 'C' : 'N', 'N', a', Aherm, dC, one(T), dB)
        elseif tB == 'C'
            # B conjugate transposed: C = a*op(A)*B' + b*C
            # dB += a * dC' * op(A)
            BLAS.gemm!('C', tA, a, dC, p_A, one(T), dB)
        else  # tB == 'T'
            # B transposed (complex): C = a*op(A)*B^T + b*C
            # dB += conj(a) * transpose(dC) * conj(op(A))
            if tA == 'N'
                BLAS.gemm!('T', 'N', a', dC, conj(p_A), one(T), dB)
            else
                BLAS.gemm!('T', tA == 'T' ? 'C' : 'T', a', dC, p_A, one(T), dB)
            end
        end

        # Propagate gradient through beta
        dC .*= b'

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
        s = primal(side)
        ul = primal(uplo)
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

        # Extract primals.
        s = primal(side)
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
        tmp_ref = Ref{Matrix{T}}()
        if (α == 1 && β == 0)
            BLAS.$fname(s, ul, α, A, B, β, C)
        else
            tmp = $(isherm ? BLAS.hemm : BLAS.symm)(s, ul, one(T), A, B)
            tmp_ref[] = tmp
            # Strong zeros, as in the `gemm!` pullback above.
            _scale_or_zero!(C, β)
            iszero(α) || (C .+= α .* tmp)
        end

        function symm!_or_hemm!_adjoint(::NoRData)
            dα = (α == 1 && β == 0) ? dot(C, dC) : dot(tmp_ref[], dC)

            BLAS.copyto!(C, C_copy)

            # gradient w.r.t. A.
            # TODO: could be switched to BLAS.{sy,he}r2k! if Julia ever provides it.
            dA_tmp = s == 'L' ? α' * dC * B' : α' * B' * dC
            if ul == 'L'
                dA .+= LowerTriangular(dA_tmp)
                dA .+= $(isherm ? adjoint : transpose)(UpperTriangular(dA_tmp))
            else
                dA .+= $(isherm ? adjoint : transpose)(LowerTriangular(dA_tmp))
                dA .+= UpperTriangular(dA_tmp)
            end
            @inbounds for n in diagind(dA)
                dA[n] -= $(isherm ? :(real(dA_tmp[n])) : :(dA_tmp[n]))
            end

            # gradient w.r.t. B: dB += α' A' dC  (or α' dC A' if right)
            # if A is hermitian or real then A' = A, else A' = conj(A)
            BLAS.$fname(s, ul, α', $(isherm ? :A : :(conj(A))), dC, one(T), dB)

            # gradient w.r.t. beta.
            dβ = dot(C, dC)

            # gradient w.r.t. C.
            dC .*= β'

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
        uplo = primal(_uplo)
        t = primal(_t)
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

        # Extract values from pairs.
        uplo = primal(_uplo)
        trans = primal(_t)
        α = primal(α_dα)
        A, dA = matrixify(A_dA)
        β = primal(β_dβ)
        C, dC = arrayify(C_dC)

        # Run forwards pass, and remember previous value of `C` for the reverse-pass.
        C_copy = collect(C)
        BLAS.$fname(uplo, trans, α, A, β, C)

        function syrk!_or_herk!_adjoint(::NoRData)
            # Restore previous state.
            C .= C_copy

            # Increment gradients.
            $(isherm ? :(real_diag!(dC)) : :())

            B = uplo == 'U' ? triu(dC) : tril(dC)
            ∇β = dot(C, B)
            $(isherm ? :(∇β = real(∇β)) : :())
            ∇α = dot(
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
            dA .+= α' .* (trans == 'N' ? M1 * M2 : M2 * M1)
            dC .= (uplo == 'U' ? tril!(dC, -1) : triu!(dC, 1)) .+ β' .* B

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
    side = primal(_side)
    uplo = primal(_uplo)
    ta = primal(_ta)
    diag = primal(_diag)
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

    # Extract values.
    side = primal(_side)
    uplo = primal(_uplo)
    tA = primal(_ta)
    diag = primal(_diag)
    α = primal(α_dα)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    B_copy = copy(B)

    # Run primal.
    BLAS.trmm!(side, uplo, tA, diag, α, A, B)

    function trmm_adjoint(::NoRData)

        # Compute α gradient. `B` holds `α·op(A)·B_old`, and `dot` conjugates its first argument, so
        # `dot(B, dB)/α' = dot(op(A)·B_old, dB)` — the true, finite ∇α. But at α==0 the primal zeroed
        # `B`, making that `0/0 = NaN`; recompute the unscaled `op(A)·B_old` from the saved input in
        # that case (the mathematically-defined limit), keeping the cheap division for α≠0.
        ∇α = if iszero(α)
            M = copy(B_copy)
            BLAS.trmm!(side, uplo, tA, diag, one(P), A, M)
            dot(M, dB)
        else
            dot(B, dB) / α'
        end

        # Restore initial state.
        B .= B_copy

        # Increment gradients.
        if side == 'L'
            if tA == 'T' && P <: BlasComplexFloat
                dA .+= α' .* tri!(conj(B) * transpose(dB), uplo, diag)
            elseif tA == 'N'
                dA .+= α' .* tri!(dB * B', uplo, diag)
            else
                dA .+= α .* tri!(B * dB', uplo, diag)
            end
        else
            if tA == 'T' && P <: BlasComplexFloat
                dA .+= α' .* tri!(transpose(dB) * conj(B), uplo, diag)
            elseif tA == 'N'
                dA .+= α' .* tri!(B' * dB, uplo, diag)
            else
                dA .+= α .* tri!(dB' * B, uplo, diag)
            end
        end

        # Compute dB tangent.
        if tA == 'T' && P <: BlasComplexFloat
            # conjugate-only of A
            BLAS.trmm!(side, uplo, 'N', diag, α', conj(A), dB)
        else
            BLAS.trmm!(side, uplo, tA == 'N' ? 'C' : 'N', diag, α', A, dB)
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
    side = primal(_side)
    uplo = primal(_uplo)
    trans = primal(_t)
    diag = primal(_diag)
    α = primal(α_dα)
    A = primal(A_dA)
    B = primal(B_dB)
    dαs = ntuple(k -> tangent(α_dα, k), Val(Nw))
    Bb, bcopied = _partials_block(B_dB)
    # BLAS's `α == 0` quick return sets `B := 0` without ever referencing `A`, so `A` may
    # legally hold garbage. The JVP is `dα·op(A)⁻¹⊛B`, needing the solve only when some
    # lane seeds α; with none, result and derivative are both identically zero and the
    # solve below would otherwise propagate a legal NaN in `A` into the result.
    if iszero(α) && all(iszero, dαs)
        fill!(B, zero(P))
        fill!(Bb, zero(P))
        bcopied && _write_back_partials!(B_dB, Bb)
        return B_dB
    end
    Ab, _ = _partials_block(A_dA)
    m, n = size(B)
    # `X = op(A)⁻¹⊛B` (the primal RHS solve) is lane-invariant: hoist it.
    X = copy(B)
    trsm!(side, uplo, trans, diag, one(P), A, X)
    # d(α·op(A)⁻¹⊛B) = dα·X + α·op(A)⁻¹⊛(dB − op(dA)⊛X). op(A)⁻¹ is linear, so the
    # tangent takes one solve of that combined RHS, not separate solves of `dB` and
    # `op(dA)⊛X`.
    # 1) dB_k −= op(dA_k)⊛X — skipped when `A` is constant data. trmm masks dA's triangle
    #    (and implicit unit diagonal, whose derivative the `diag == 'U'` correction
    #    removes).
    if !iszero(Ab)
        R = size(A, 1)
        Abm = reshape(Ab, Nw, R, R)
        Ascr = Matrix{P}(undef, R, R)
        tmp = Matrix{P}(undef, m, n)
        for k in 1:Nw
            copyto!(Ascr, view(Abm,k,:,:))
            copyto!(tmp, X)
            BLAS.trmm!(side, uplo, trans, diag, one(P), Ascr, tmp)
            diag == 'U' && (tmp .-= X)
            view(Bb,k,:,:) .-= tmp
        end
    end
    # 2) α·op(A)⁻¹ applied to every lane (α folded into the solve). Side 'R' solves the
    #    (Nw·m, n) flat view in one wide trsm, flags native; side 'L' right-divides each
    #    slab by op(A)ᵀ (flag flip; complex 'C' needs a hoisted conj(A)).
    if side == 'R'
        BLAS.trsm!('R', uplo, trans, diag, α, A, reshape(Bb, Nw * m, n))
    else
        fA, Ae = if trans == 'N'
            ('T', A)
        elseif trans == 'T' || P <: BlasRealFloat
            ('N', A)
        else
            ('N', conj(A))
        end
        for j in 1:n
            BLAS.trsm!('R', uplo, fA, diag, α, Ae, view(Bb,:,:,j))
        end
    end
    # 3) dα·X per seeded lane.
    for k in 1:Nw
        iszero(dαs[k]) || (view(Bb,k,:,:) .+= dαs[k] .* X)
    end
    bcopied && _write_back_partials!(B_dB, Bb)
    # Primal result α·op(A)⁻¹⊛B = α·X, and X already holds the unscaled solve: scale,
    # don't re-solve. At `α == 0` BLAS returns `B := 0` by a quick return that never
    # references `A`, so `A` may legally hold garbage and `0 * X` would turn it into a NaN
    # primal. The early return above cannot cover this: a seeded `dα` still needs the solve,
    # because the derivative genuinely depends on `A`.
    if iszero(α)
        fill!(B, zero(P))
    else
        B .= α .* X
    end
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

    # Extract parameters.
    side = primal(_side)
    uplo = primal(_uplo)
    trans = primal(_t)
    diag = primal(_diag)
    α = primal(α_dα)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)

    # Copy memory which will be overwritten by primal computation.
    B_copy = copy(B)

    # Run primal computation.
    trsm!(side, uplo, trans, diag, α, A, B)

    function trsm_adjoint(::NoRData)
        # Compute α gradient. `B` holds `α·op(A)⁻¹·B_old`; `dot(B, dB)/α' = dot(op(A)⁻¹·B_old, dB)` is
        # the true finite ∇α, but α==0 zeroes `B` → `0/0 = NaN`. Recompute the unscaled
        # `op(A)⁻¹·B_old` from the saved input in that case; keep the cheap division for α≠0.
        ∇α = if iszero(α)
            M = copy(B_copy)
            trsm!(side, uplo, trans, diag, one(P), A, M)
            dot(M, dB)
        else
            dot(B, dB) / α'
        end

        # Increment cotangents.
        if side == 'L'
            if trans == 'N'
                tmp = trsm!('L', uplo, 'C', diag, -one(P), A, dB * B')
            elseif trans == 'C'
                tmp = trsm!('R', uplo, 'C', diag, -one(P), A, B * dB')
            else
                tmp = trsm!('R', uplo, 'C', diag, -one(P), A, conj(B * dB'))
            end
            dA .+= tri!(tmp, uplo, diag)
        else
            if trans == 'N'
                tmp = trsm!('R', uplo, 'C', diag, -one(P), A, B'dB)
            elseif trans == 'C'
                tmp = trsm!('L', uplo, 'C', diag, -one(P), A, dB'B)
            else
                tmp = trsm!('L', uplo, 'C', diag, -one(P), A, conj(dB'B))
            end
            dA .+= tri!(tmp, uplo, diag)
        end

        # Restore initial state.
        B .= B_copy

        # Compute dB tangent.
        if trans == 'T'
            # conjugate-only of A
            BLAS.trsm!(side, uplo, 'N', diag, α', conj(A), dB)
        else
            BLAS.trsm!(side, uplo, trans == 'N' ? 'C' : 'N', diag, α', A, dB)
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

        # nrm2(x) — the one-argument form `LinearAlgebra.norm2` calls at length >= 32. Julia inlines
        # it and its `ccall`, so the three-argument primitive above never sees a boundary and the
        # raw pointer reached the transform: `norm` of any array that size threw at chunk width > 1,
        # which is the DEFAULT width for it. Length 40 to stay above LinearAlgebra's threshold.
        map([randn(rng, P, 40)]) do x
            (false, :stability, nothing, BLAS.nrm2, x)
        end...,

        # dot(x, y) — real only (complex inner products are dotc/dotu). `n = 0` is the case
        # `gemv` skips without applying `beta`, which left the forward lanes reading uninitialised
        # memory; the derivative there is exactly zero, so garbage fails against finite differences.
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

        # dotc, dotu — complex only, and forward primitives only, so `skip_reverse`. The derived
        # rows below cover reverse. These exist for what a derived row cannot check: widths 2-3 (a
        # derived case never runs them) and `:stability`, which is what catches a boxed lane
        # accumulator in the block fast path. The strided second operand takes the per-lane BLAS
        # fallback rather than the block loop.
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

        # gemv! with a zero-length `x`. BLAS takes its quick return there and never applies `β`, so
        # a rule that folds `β` in by hand must skip it too: with `β` applied unconditionally the
        # tangent came back scaled while the primal `y` was left untouched. The `M`/`N` product
        # above never reaches a zero dimension.
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

    dαs = [0.0, 0.44, -0.20 + 0.38im]
    dβs = [0.0, -0.11, 0.86 + 0.44im]

    # 1.10 fails to infer part of a matmat product in the pullback
    perf_flag = VERSION < v"1.11-" ? :none : :stability

    # The tests are quite sensitive to the random inputs,
    # so each tested gemm! dispatch gets its own rng.

    # gemm! - matrix × matrix
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123456)
            map_prod(t_flags, t_flags, αs, βs, dαs, dβs) do (tA, tB, α, β, dα, dβ)
                P <: BlasRealFloat && (imag(α) != 0 || imag(β) != 0) && return []
                P <: BlasRealFloat && (imag(dα) != 0 || imag(dβ) != 0) && return []

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
            map_prod(t_flags, αs, βs, dαs, dβs) do (tA, α, β, dα, dβ)
                P <: BlasRealFloat && (imag(α) != 0 || imag(β) != 0) && return []
                P <: BlasRealFloat && (imag(dα) != 0 || imag(dβ) != 0) && return []
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
            map_prod(['T', 'C'], t_flags, αs, βs, dαs, dβs) do (tA, tB, α, β, dα, dβ)
                P <: BlasRealFloat && (imag(α) != 0 || imag(β) != 0) && return []
                P <: BlasRealFloat && (imag(dα) != 0 || imag(dβ) != 0) && return []
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
            map_prod(['T', 'C'], αs, βs, dαs, dβs) do (tA, α, β, dα, dβ)
                P <: BlasRealFloat && (imag(α) != 0 || imag(β) != 0) && return []
                P <: BlasRealFloat && (imag(dα) != 0 || imag(dβ) != 0) && return []
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
            map_prod(uplos, syrk_herk_trans, αs, βs, dαs, dβs) do (ul, t, α, β, dα, dβ)
                P <: BlasRealFloat && (imag(α) != 0 || imag(β) != 0) && return []
                P <: BlasRealFloat && (imag(dα) != 0 || imag(dβ) != 0) && return []
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
            map_prod(uplos, αs, βs, dαs, dβs) do (ul, α, β, dα, dβ)
                P <: BlasRealFloat && (imag(α) != 0 || imag(β) != 0) && return []
                P <: BlasRealFloat && (imag(dα) != 0 || imag(dβ) != 0) && return []
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

                t = tA == 'N'
                R = side == 'L' ? M : N
                As = blas_matrices(rng, P, R, R)
                Bs = blas_matrices(rng, P, M, N)
                return map(As, Bs) do A, B
                    α_dα = CoDual(randn(rng, P), P(dα))
                    # 1.10 fails to infer part of a matmat product in the pullback
                    perf_flag = VERSION < v"1.11-" ? :none : :stability
                    (
                        false, perf_flag, nothing, BLAS.trmm!, side, ul, tA, dA, α_dα, A, B
                    )
                end
            end
        end...,
    )

    # trsm!
    test_cases = append!(
        test_cases,
        let
            rng = rng_ctor(123456)
            map_prod(
                ['L', 'R'], uplos, t_flags, dAs, [1, 3], [1, 2]
            ) do (side, ul, tA, dA, M, N)
                t = tA == 'N'
                R = side == 'L' ? M : N
                a = randn(rng, P)
                As = map(blas_matrices(rng, P, R, R)) do A
                    A[diagind(A)] .+= 1
                    return A
                end
                Bs = blas_matrices(rng, P, M, N)
                return map(As, Bs) do A, B
                    # 1.10 fails to infer part of a matmat product in the pullback
                    perf_flag = VERSION < v"1.11-" ? :none : :stability
                    (false, perf_flag, nothing, BLAS.trsm!, side, ul, tA, dA, a, A, B)
                end
            end
        end...,
    )

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
                (
                    false,
                    :none,
                    nothing,
                    BLAS.trmm!,
                    'L',
                    'U',
                    'N',
                    'N',
                    zero(P),
                    A,
                    copy(B),
                ),
                (
                    false,
                    :none,
                    nothing,
                    BLAS.trsm!,
                    'L',
                    'U',
                    'N',
                    'N',
                    zero(P),
                    Ainv,
                    copy(B),
                ),
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
                    # 1.10 fails to infer part of a matmat product in the pullback
                    perf_flag = VERSION < v"1.11-" ? :none : :stability
                    (false, perf_flag, nothing, f, side, ul, α_dα, A, B, β_dβ, C)
                end
            end
        end...,
    )

    memory = Any[]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:blas}, P::Type{<:BlasFloat})
    t_flags = ['N', 'T', 'C']
    rng = rng_ctor(123)
    test_cases = Any[]

    #
    # BLAS LEVEL 1
    #

    # dot (real types only)
    if P <: BlasRealFloat
        flags = (false, :none, nothing)
        append!(
            test_cases,
            [
                (flags..., BLAS.dot, 3, randn(rng, P, 5), 1, randn(rng, P, 4), 1),
                (flags..., BLAS.dot, 3, randn(rng, P, 6), 2, randn(rng, P, 4), 1),
                (flags..., BLAS.dot, 3, randn(rng, P, 6), 1, randn(rng, P, 9), 3),
                (flags..., BLAS.dot, 3, randn(rng, P, 12), 3, randn(rng, P, 9), 2),
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
                ],
            )
        end
    end

    # nrm2
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

function throwing_rule_test_cases(::Val{:blas}, P::Type{<:BlasFloat})
    # The registered `scal!`/`nrm2` cases all pass a DENSE vector with a positive increment, where
    # logical index and raw walk coincide, so none reach these guards. `incx == stride` is the
    # well-formed call: its primal is correct, and it previously indexed the operand out of range.
    x = view(P[i for i in 1:10], 1:2:10)
    # A >=2-D operand with unit FIRST-dim stride but a non-dense layout: `strides` is `(1, 5)` where
    # a dense `(3, 2)` has `(1, 3)`, so logical index 4 is raw offset 6. The old first-dim-stride
    # test admitted it and the rule then scaled the partials of the wrong elements, silently.
    m = view(reshape(P[i for i in 1:25], 5, 5), 1:3, 1:2)
    cases = Any[
        (
            (ArgumentError, "does not support operand"),
            BLAS.scal!,
            (5, P(2), x, 2),
            (; mode=ForwardMode),
        ),
        (
            (ArgumentError, "does not support operand"),
            BLAS.scal!,
            (6, P(2), m, 1),
            (; mode=ForwardMode),
        ),
        (
            (ArgumentError, "does not support operand"),
            BLAS.nrm2,
            (6, m, 1),
            (; mode=ForwardMode),
        ),
    ]
    return cases, Any[x, m]
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
    @eval function throwing_rule_test_cases(::Val{$(QuoteNode(sym))})
        return throwing_rule_test_cases(Val(:blas), $P)
    end
end
