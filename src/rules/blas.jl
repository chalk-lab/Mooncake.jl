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
    x::Union{Dual{A},CoDual{A}}
) where {T<:Union{IEEEFloat,BlasFloat},A<:Union{AbstractArray{T},Ptr{<:T}}}
    return arrayify(primal(x), tangent(x))
end
function arrayify(
    x::A, dx::A
) where {T<:Union{IEEEFloat,BlasFloat},A<:Union{Array{<:T},Ptr{<:T}}}
    (x, dx)
end
# `tangent(::Dual{Ptr{T}, NTangent{Tuple{Ptr{T}}}})` returns the
# NTangent-wrapped Ptr tangent. Unwrap the singleton lane to match the
# `(x::Ptr, dx::Ptr)` pair the rest of arrayify expects.
function arrayify(
    x::Ptr{T}, dx::Mooncake.NTangent{Tuple{Ptr{T}}}
) where {T<:Union{IEEEFloat,BlasFloat}}
    return (x, dx.lanes[1])
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

# ── Unified Dual / NDual array extract & writeback ──────────────────────────
#
# Array arguments arrive as one of two shapes: a `Dual{Wrapper{P,...}, ...}`
# for struct wrappers (`ReshapedArray`, `SubArray`, `ReinterpretArray`) or an
# `Array{NDual{T,N}}` / `Array{Complex{NDual{T,N}}}` for plain `Array` primals
# that lift elementwise. `_arr_extract` / `_mat_extract` / `_scalar_extract`
# return a `(primal, tangent)` pair for the width-1 shapes (arrayify-views for
# the Dual case so mutations propagate; `map`-allocated copies for the NDual
# case, paired with `_arr_writeback!` to push results back element-wise).
# The `_*_extract_n` / `_arr_writeback_n!` family below mirrors this for
# width-N callers, returning per-lane tangent tuples.
@inline _arr_extract(x::Dual{<:AbstractArray}) = arrayify(x)
@inline function _arr_extract(x::AbstractArray{NDual{T,1}}) where {T}
    return (map(d -> d.value, x), map(d -> d.partials[1], x))
end
# Complex-element overload: each element is `Complex{NDual{R,1}}` with the
# canonical width-1 representation putting real & imag parts in separate
# NDuals. Primals/tangents are reconstructed elementwise as `Complex{R}`.
@inline function _arr_extract(x::AbstractArray{Complex{NDual{T,1}}}) where {T<:IEEEFloat}
    return (
        map(c -> Complex(c.re.value, c.im.value), x),
        map(c -> Complex(c.re.partials[1], c.im.partials[1]), x),
    )
end

@inline _arr_writeback!(::Dual, _, _) = nothing
@inline function _arr_writeback!(x::AbstractArray{NDual{T,1}}, p, t) where {T}
    @inbounds for i in eachindex(x)
        x[i] = NDual{T,1}(p[i], (t[i],))
    end
    return nothing
end
@inline function _arr_writeback!(
    x::AbstractArray{Complex{NDual{T,1}}}, p, t
) where {T<:IEEEFloat}
    @inbounds for i in eachindex(x)
        x[i] = Complex(
            NDual{T,1}(real(p[i]), (real(t[i]),)), NDual{T,1}(imag(p[i]), (imag(t[i]),))
        )
    end
    return nothing
end

# Width-N extract / writeback: returns (p, ts::NTuple{N, AbstractArray})
# where each `ts[n]` is the n-th lane tangent (separate array per lane).
# Used by rules that compute per-lane derivatives separately and then
# reassemble lanes at the output.
@inline function _arr_extract_n(x::AbstractArray{NDual{T,N}}) where {T,N}
    p = map(d -> d.value, x)
    ts = ntuple(n -> map(d -> d.partials[n], x), Val(N))
    return (p, ts)
end
@inline function _arr_extract_n(
    x::AbstractArray{Complex{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    p = map(c -> Complex(c.re.value, c.im.value), x)
    ts = ntuple(Val(N)) do n
        map(c -> Complex(c.re.partials[n], c.im.partials[n]), x)
    end
    return (p, ts)
end

@inline function _arr_writeback_n!(
    x::AbstractArray{NDual{T,N}}, p, ts::NTuple{N,<:AbstractArray}
) where {T,N}
    @inbounds for i in eachindex(x)
        partials = ntuple(n -> ts[n][i], Val(N))
        x[i] = NDual{T,N}(p[i], partials)
    end
    return nothing
end
@inline function _arr_writeback_n!(
    x::AbstractArray{Complex{NDual{T,N}}}, p, ts::NTuple{N,<:AbstractArray}
) where {T<:IEEEFloat,N}
    @inbounds for i in eachindex(x)
        re_partials = ntuple(n -> real(ts[n][i]), Val(N))
        im_partials = ntuple(n -> imag(ts[n][i]), Val(N))
        x[i] = Complex(
            NDual{T,N}(real(p[i]), re_partials), NDual{T,N}(imag(p[i]), im_partials)
        )
    end
    return nothing
end

# Width-N scalar extract: returns (primal, NTuple{N, T_tangent}).
@inline _scalar_extract_n(x::NDual{T,N}) where {T,N} = (x.value, x.partials)
@inline function _scalar_extract_n(x::Complex{NDual{R,N}}) where {R<:IEEEFloat,N}
    p = Complex(x.re.value, x.im.value)
    ts = ntuple(n -> Complex(x.re.partials[n], x.im.partials[n]), Val(N))
    return (p, ts)
end

# Width-N matrix extract: like `_mat_extract` but per-lane tangents. Reshape
# Vector inputs to M×1 columns so BLAS Level 2/3 callers can rely on the
# `AbstractMatrix` shape regardless of input rank.
@inline function _mat_extract_n(x::AbstractVector{NDual{T,N}}) where {T,N}
    p, ts = _arr_extract_n(x)
    return (reshape(p, :, 1), map(t -> reshape(t, :, 1), ts))
end
@inline _mat_extract_n(x::AbstractMatrix{NDual{T,N}}) where {T,N} = _arr_extract_n(x)
@inline function _mat_extract_n(
    x::AbstractVector{Complex{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    p, ts = _arr_extract_n(x)
    return (reshape(p, :, 1), map(t -> reshape(t, :, 1), ts))
end
@inline function _mat_extract_n(
    x::AbstractMatrix{Complex{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    return _arr_extract_n(x)
end

@inline _mat_extract(x::Dual{<:AbstractVecOrMat}) = matrixify(x)
@inline function _mat_extract(x::AbstractVector{NDual{T,1}}) where {T}
    p, t = _arr_extract(x)
    return reshape(p, :, 1), reshape(t, :, 1)
end
@inline _mat_extract(x::AbstractMatrix{NDual{T,1}}) where {T} = _arr_extract(x)
@inline function _mat_extract(x::AbstractVector{Complex{NDual{T,1}}}) where {T<:IEEEFloat}
    p, t = _arr_extract(x)
    return reshape(p, :, 1), reshape(t, :, 1)
end
@inline function _mat_extract(x::AbstractMatrix{Complex{NDual{T,1}}}) where {T<:IEEEFloat}
    return _arr_extract(x)
end

# Scalar counterpart: at width 1, an `IEEEFloat` slot may arrive as either
# `NDual{T,1}` (the canonical IEEEFloat lifted form) or `Dual{T,T}` (when
# constructed by the test framework or by zero_dual). Both unwrap to a
# `(primal, tangent)` pair the same way.
@inline _scalar_extract(x::Dual{<:Number}) = (primal(x), tangent(x))
@inline _scalar_extract(x::NDual{T,1}) where {T} = (x.value, x.partials[1])
# Complex-scalar canonical width-1 form: `Complex{NDual{R, 1}}` with real
# and imag NDuals.
@inline _scalar_extract(x::Complex{NDual{R,1}}) where {R<:IEEEFloat} = (
    Complex(x.re.value, x.im.value), Complex(x.re.partials[1], x.im.partials[1])
)

# Slot-level Union that matches either a struct-wrapped Dual array or a
# plain NDual-elementwise array, per the AGENTS.md "NDual shapes" rule.
const _MatLikeWidth1{P} = Union{Dual{<:AbstractMatrix{P}},AbstractMatrix{NDual{P,1}}}
const _VecOrMatLikeWidth1{P} = Union{
    Dual{<:AbstractVecOrMat{P}},AbstractVecOrMat{NDual{P,1}}
}
const _ArrLikeWidth1{P} = Union{Dual{<:AbstractArray{P}},AbstractArray{NDual{P,1}}}
const _ScalarLikeWidth1{T} = Union{Dual{T},NDual{T,1}}

# Complex-element analogues: the canonical width-1 lift of
# `Matrix{Complex{R}}` is `Matrix{Complex{NDual{R, 1}}}` (real and imag
# parts each carried by an NDual), NOT `Matrix{NDual{Complex{R}, 1}}`.
# Existing `_MatLikeWidth1{Complex{R}}` accepts only the Dual-wrapper form;
# the NDual-element form needs these separate aliases.
const _MatLikeWidth1Complex{R} = Union{
    Dual{<:AbstractMatrix{Complex{R}}},AbstractMatrix{Complex{NDual{R,1}}}
}
const _VecOrMatLikeWidth1Complex{R} = Union{
    Dual{<:AbstractVecOrMat{Complex{R}}},AbstractVecOrMat{Complex{NDual{R,1}}}
}
const _ArrLikeWidth1Complex{R} = Union{
    Dual{<:AbstractArray{Complex{R}}},AbstractArray{Complex{NDual{R,1}}}
}
const _ScalarLikeWidth1Complex{R} = Union{Dual{Complex{R}},Complex{NDual{R,1}}}

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

    @eval @inline function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{$(blas_name(fname))}},
        ::Dual, # return type
        ::Dual, # argument types
        ::Dual, # nreq
        ::Dual, # calling convention
        _n::Dual{BLAS.BlasInt},
        _DX::Dual{Ptr{$elty}},
        _incx::Dual{BLAS.BlasInt},
        _DY::Dual{Ptr{$elty}},
        _incy::Dual{BLAS.BlasInt},
        # For complex numbers the result is stored in an extra pointer
        $((isreal ? () : (:(_presult::Dual{Ptr{$elty}}),))...),
        args::Vararg{Any,N},
    ) where {N}
        GC.@preserve args begin
            # Load in values from pointers.
            n, incx, incy = map(primal, (_n, _incx, _incy))
            DX, _dDX = arrayify(_DX)
            DY, _dDY = arrayify(_DY)

            result = BLAS.$jlfname(n, DX, incx, DY, incy)
            _dresult =
                BLAS.$jlfname(n, _dDX, incx, DY, incy) +
                BLAS.$jlfname(n, DX, incx, _dDY, incy)

            # For complex numbers the result must be stored in the pointer
            $(
                if isreal
                    quote
                        Dual(result, _dresult)
                    end
                else
                    quote
                        presult, _dpresult = arrayify(_presult)
                        Base.unsafe_store!(presult, result)
                        Base.unsafe_store!(_dpresult, _dresult)

                        Dual(nothing, NoTangent())
                    end
                end
            )
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

    # Do NOT add a Lifted-aware delegator for the `cblas_*dot*` foreigncall:
    # past attempts segfaulted inside libopenblas64_'s `sdot_k_COOPERLAKE`
    # when called with the canonical width-1 NTangent-wrapped
    # `Dual{Ptr{T}, NTangent{Tuple{Ptr{T}}}}` tangent pointer (likely the
    # tangent Ptr isn't a valid readable address). Flipping `_is_lifted_aware`
    # on this foreigncall sig also doesn't help — the IR-emit's foreigncall
    # dispatch path ignores the trait. With no delegator, the foreigncall
    # fallback returns a controlled `MissingForeigncallRuleError` instead of
    # crashing.
    #
    # The rule-level workaround is the high-level `BLAS.dot` /
    # `BLAS.dotc` / `BLAS.dotu` primitive (below) that intercepts user calls
    # BEFORE the BLAS shim reaches the foreigncall. Reverse mode still uses
    # the foreigncall rrule above for `Vector{T}` (non-NDual) inputs.
end

# High-level `BLAS.dot` / `BLAS.dotc` / `BLAS.dotu` ForwardMode primitives.
# These intercept user calls before the BLAS shim emits the foreigncall,
# eliminating the need for the IR-emit to materialise `Ptr{T}`-typed tangents
# for canonical width-1 NDual containers. ForwardMode only — the existing
# foreigncall rrule above handles reverse-mode `Vector{T}` inputs. See the
# comment in the cblas rrule above for why the foreigncall path can't be
# Lifted-aware-registered.
@is_primitive(
    MinimalCtx,
    ForwardMode,
    Tuple{
        typeof(BLAS.dot),
        Integer,
        Union{Ptr{T},AbstractArray{T}},
        Integer,
        Union{Ptr{T},AbstractArray{T}},
        Integer,
    } where {T<:BlasRealFloat},
)
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.dot),Vararg}}) = true
@inline function frule!!(
    ::Dual{typeof(BLAS.dot)},
    _n::Dual{<:Integer},
    X_dX::Dual{<:Union{Ptr{T},AbstractArray{T}}},
    _incx::Dual{<:Integer},
    Y_dY::Dual{<:Union{Ptr{T},AbstractArray{T}}},
    _incy::Dual{<:Integer},
) where {T<:BlasRealFloat}
    n, incx, incy = primal(_n), primal(_incx), primal(_incy)
    X, dX = arrayify(X_dX)
    Y, dY = arrayify(Y_dY)
    val = BLAS.dot(n, X, incx, Y, incy)
    dval = BLAS.dot(n, dX, incx, Y, incy) + BLAS.dot(n, X, incx, dY, incy)
    return Dual(val, dval)
end
# Width-N: NDual element containers.
@inline function frule!!(
    ::Dual{typeof(BLAS.dot)},
    _n::Dual{<:Integer},
    X_dX::AbstractArray{NDual{T,N}},
    _incx::Dual{<:Integer},
    Y_dY::AbstractArray{NDual{T,N}},
    _incy::Dual{<:Integer},
) where {T<:BlasRealFloat,N}
    n, incx, incy = primal(_n), primal(_incx), primal(_incy)
    X, dXs = _arr_extract_n(X_dX)
    Y, dYs = _arr_extract_n(Y_dY)
    val = BLAS.dot(n, X, incx, Y, incy)
    dvals = ntuple(
        lane ->
            BLAS.dot(n, dXs[lane], incx, Y, incy) + BLAS.dot(n, X, incx, dYs[lane], incy),
        Val(N),
    )
    return NDual(val, dvals)
end
# `BLAS.dotc(n, X, incx, Y, incy) = conj(X) ⋅ Y` (Complex only).
@is_primitive(
    MinimalCtx,
    ForwardMode,
    Tuple{
        typeof(BLAS.dotc),
        Integer,
        Union{Ptr{T},AbstractArray{T}},
        Integer,
        Union{Ptr{T},AbstractArray{T}},
        Integer,
    } where {T<:BlasComplexFloat},
)
# `BLAS.dotu(n, X, incx, Y, incy) = X ⋅ Y` without conjugation (Complex only).
@is_primitive(
    MinimalCtx,
    ForwardMode,
    Tuple{
        typeof(BLAS.dotu),
        Integer,
        Union{Ptr{T},AbstractArray{T}},
        Integer,
        Union{Ptr{T},AbstractArray{T}},
        Integer,
    } where {T<:BlasComplexFloat},
)
# dotc and dotu width-1 bare-Dual frules share byte-identical body structure
# (apart from the BLAS function name): the derivative of `f(X, Y)` w.r.t.
# `(X, Y)` is `f(dX, Y) + f(X, dY)` for the same `f`. dotc carries the
# conjugation through `f = BLAS.dotc`; dotu uses `f = BLAS.dotu`.
for fname in (:dotc, :dotu)
    @eval @inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.$fname),Vararg}}) =
        true
    @eval @inline function frule!!(
        ::Dual{typeof(BLAS.$fname)},
        _n::Dual{<:Integer},
        X_dX::Dual{<:Union{Ptr{T},AbstractArray{T}}},
        _incx::Dual{<:Integer},
        Y_dY::Dual{<:Union{Ptr{T},AbstractArray{T}}},
        _incy::Dual{<:Integer},
    ) where {T<:BlasComplexFloat}
        n, incx, incy = primal(_n), primal(_incx), primal(_incy)
        X, dX = arrayify(X_dX)
        Y, dY = arrayify(Y_dY)
        val = BLAS.$fname(n, X, incx, Y, incy)
        dval = BLAS.$fname(n, dX, incx, Y, incy) + BLAS.$fname(n, X, incx, dY, incy)
        return Dual(val, dval)
    end
end

# Lifted-aware delegators for `BLAS.dot` / `dotc` / `dotu`: `_unlift` and
# delegate to the bare frules above, then `_wrap_rule_result`.
for fname in (:dot, :dotc, :dotu)
    @eval @inline function frule!!(
        f::Mooncake.Lifted{typeof(BLAS.$fname),N},
        _n::Mooncake.Lifted{<:Integer},
        X_dX::Mooncake.Lifted{<:Union{Ptr,AbstractArray}},
        _incx::Mooncake.Lifted{<:Integer},
        Y_dY::Mooncake.Lifted{<:Union{Ptr,AbstractArray}},
        _incy::Mooncake.Lifted{<:Integer},
    ) where {N}
        bare_result = frule!!(
            Mooncake._unlift(f),
            Mooncake._unlift(_n),
            Mooncake._unlift(X_dX),
            Mooncake._unlift(_incx),
            Mooncake._unlift(Y_dY),
            Mooncake._unlift(_incy),
        )
        P_out = __primal_type(_typeof(bare_result))
        return _wrap_rule_result(P_out, Val(N), bare_result)
    end
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.nrm2),Int,X,Int
    } where {T<:BlasFloat,X<:Union{Ptr{T},AbstractArray{T}}},
)
# 1-arg convenience form: `BLAS.nrm2(X) === BLAS.nrm2(length(X), X, 1)`.
@is_primitive(
    MinimalCtx, Tuple{typeof(BLAS.nrm2),X} where {T<:BlasFloat,X<:AbstractArray{T}},
)
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.nrm2),Vararg}}) = true
function frule!!(
    ::Dual{typeof(BLAS.nrm2)},
    n::Dual{<:Integer},
    X_dX::Dual{<:Union{Ptr{T},AbstractArray{T}}},
    incx::Dual{<:Integer},
) where {T<:BlasFloat}
    y = BLAS.nrm2(primal(n), primal(X_dX), primal(incx))
    X, dX = viewify(primal(n), X_dX, primal(incx))
    dy = zero(y)
    @inbounds for i in eachindex(X)
        dy = dy + real(X[i] * dX[i]') + real(X[i]' * dX[i])
    end
    return Dual(y, dy / 2y)
end
# 1-arg `BLAS.nrm2(X)` convenience form. Delegates to the 3-arg rule with
# `n = length(X)` and `incx = 1`, matching Julia's BLAS shim.
function frule!!(
    f::Dual{typeof(BLAS.nrm2)}, X_dX::Dual{<:Union{Ptr{T},AbstractArray{T}}}
) where {T<:BlasFloat}
    return frule!!(f, Dual(length(primal(X_dX)), NoTangent()), X_dX, Dual(1, NoTangent()))
end
# Consolidated 1-arg nrm2 convenience form: delegates to the 3-arg form
# which already handles both Real (NDual{T,N}) and Complex (Complex{NDual{R,N}})
# canonical V shapes via dispatch.
function frule!!(
    f::Dual{typeof(BLAS.nrm2)}, X_dX::AbstractArray{<:Union{NDual{T,N},Complex{NDual{T,N}}}}
) where {T<:IEEEFloat,N}
    return frule!!(f, Dual(length(X_dX), NoTangent()), X_dX, Dual(1, NoTangent()))
end
# Width-1 NDual overload — extracts primal/tangent via element-wise map
# (allocates) and reuses the bare body's dot-style accumulation.
# Consolidated width-N nrm2: covers Real (NDual{T,N}) and Complex
# (Complex{NDual{R,N}}). Replaces the previous element-type-specific
# extract lambdas with `_arr_extract_n`, which dispatches on input type
# and returns `(primal, NTuple{N, lane_tangent})` uniformly.
#
# Per-lane tangent: nrm2 derivative w.r.t. each direction is
# (X' * dX + dX' * X) / (2 * nrm). Each lane uses real() for both Real
# and Complex (real() is identity on Float, takes Real-component on Complex).
function frule!!(
    ::Dual{typeof(BLAS.nrm2)},
    n::Dual{<:Integer},
    X_dX::AbstractArray{<:Union{NDual{T,N},Complex{NDual{T,N}}}},
    incx::Dual{<:Integer},
) where {T<:IEEEFloat,N}
    _n = primal(n)
    _incx = primal(incx)
    X, dXs = _arr_extract_n(X_dX)
    y = BLAS.nrm2(_n, X, _incx)
    Xinds = 1:_incx:(_incx * _n)
    Xv = view(X, Xinds)
    partials = ntuple(Val(N)) do lane
        dXv = view(dXs[lane], Xinds)
        dy = zero(y)
        @inbounds for i in eachindex(Xv)
            dy = dy + real(Xv[i] * dXv[i]') + real(Xv[i]' * dXv[i])
        end
        dy / 2y
    end
    return NDual{T,N}(y, partials)
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.nrm2),N},
    n::Mooncake.Lifted{<:Integer},
    X_dX::Mooncake.Lifted,
    incx::Mooncake.Lifted{<:Integer},
) where {N}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(n),
        Mooncake._unlift(X_dX),
        Mooncake._unlift(incx),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
# Lifted-aware 1-arg convenience form.
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.nrm2),N}, X_dX::Mooncake.Lifted
) where {N}
    bare_result = frule!!(Mooncake._unlift(f), Mooncake._unlift(X_dX))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
function rrule!!(
    ::CoDual{typeof(BLAS.nrm2)},
    n::CoDual{<:Integer},
    X_dX::CoDual{<:Union{Ptr{T},AbstractArray{T}} where {T<:BlasFloat}},
    incx::CoDual{<:Integer},
)
    y = BLAS.nrm2(primal(n), primal(X_dX), primal(incx))
    X, dX = viewify(primal(n), X_dX, primal(incx))
    function nrm2_pb!!(dy)
        dX .+= X .* (dy / y)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, NoFData()), nrm2_pb!!
end
# 1-arg `BLAS.nrm2(X)` convenience form for reverse mode. Inline the
# 3-arg core logic with `n = length(X)`, `incx = 1`; the pullback returns 2
# values (function + arg) matching the 2-arg call signature.
function rrule!!(
    ::CoDual{typeof(BLAS.nrm2)}, X_dX::CoDual{<:AbstractArray{T} where {T<:BlasFloat}}
)
    n = length(primal(X_dX))
    y = BLAS.nrm2(n, primal(X_dX), 1)
    X, dX = viewify(n, X_dX, 1)
    function nrm2_pb!!(dy)
        dX .+= X .* (dy / y)
        return NoRData(), NoRData()
    end
    return CoDual(y, NoFData()), nrm2_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.scal!),Integer,P,X,Integer
    } where {P<:BlasFloat,X<:Union{Ptr{P},AbstractArray{P}}}
)
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.scal!),Vararg}}) = true
# Note: the bare body covers both `Dual{<:Array{P}}` (struct-wrapped via
# `_arr_extract`) and `AbstractArray{NDual{P,1}}` (plain NDual, with
# `_arr_writeback!` pushing results back). The legacy `Ptr{P}` argument shape
# is not handled here — `BLAS.scal!(n, a, ::Ptr, incx)` calls go through the
# `_foreigncall_` dot loop above.
# Consolidated width-1 entry: covers Real (BlasFloat) and Complex
# (canonical Complex{NDual{R,1}}). Body identical for both V shapes.
function frule!!(
    ::Dual{typeof(BLAS.scal!)},
    _n::Dual{<:Integer},
    a_da::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    X_dX::Union{_ArrLikeWidth1,_ArrLikeWidth1Complex},
    _incx::Dual{<:Integer},
)
    n = primal(_n)
    incx = primal(_incx)
    a, da = _scalar_extract(a_da)
    X, dX = _arr_extract(X_dX)

    # Compute Frechet derivative.
    BLAS.scal!(n, a, dX, incx)
    BLAS.axpy!(n, da, X, incx, dX, incx)

    # Perform primal computation.
    BLAS.scal!(n, a, X, incx)
    _arr_writeback!(X_dX, X, dX)
    return X_dX
end
# Width-N NDual scal!: per-lane Frechet via N
# `BLAS.scal!` + `BLAS.axpy!` calls on independent lane tangent arrays,
# then a single `BLAS.scal!` for the primal and a width-N writeback. The
# real path matches `NDual{T, N}`; the complex path mirrors with
# `Complex{NDual{R, N}}`.
# Consolidated width-N scal!: covers Real (NDual{P,N}) and Complex
# (Complex{NDual{P,N}}). Bodies were byte-identical.
@inline function frule!!(
    ::Dual{typeof(BLAS.scal!)},
    _n::Dual{<:Integer},
    a_da::Union{NDual{P,N},Complex{NDual{P,N}}},
    X_dX::AbstractArray{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    _incx::Dual{<:Integer},
) where {P<:IEEEFloat,N}
    nn = primal(_n)
    incx = primal(_incx)
    a, das = _scalar_extract_n(a_da)
    X, dXs = _arr_extract_n(X_dX)
    @inbounds for lane in 1:N
        BLAS.scal!(nn, a, dXs[lane], incx)
        BLAS.axpy!(nn, das[lane], X, incx, dXs[lane], incx)
    end
    BLAS.scal!(nn, a, X, incx)
    _arr_writeback_n!(X_dX, X, dXs)
    return X_dX
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.scal!),N},
    _n::Mooncake.Lifted{<:Integer},
    a_da::Mooncake.Lifted{P},
    X_dX::Mooncake.Lifted{<:AbstractArray{P}},
    _incx::Mooncake.Lifted{<:Integer},
) where {N,P<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(_n),
        Mooncake._unlift(a_da),
        Mooncake._unlift(X_dX),
        Mooncake._unlift(_incx),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.gemv!),Vararg}}) = true

# Width-1 path: container args narrowed to bare-`Dual{<:Abstract...}` to
# disambiguate from the width-N rule at N=1 (NDual array inputs route to
# width-N). Scalars keep the `_ScalarLikeWidth1` Union so mixed
# (NDual scalar + Dual structural-wrapper container) inputs dispatch
# here. `P<:BlasFloat` covers real and complex element types.
# Consolidated width-1 gemv!: covers Real and Complex (the wrapper-exception
# Dual-typed slots). `A_dA` can be a Vector or Matrix. The core helper
# requires a Matrix, so route through `_mat_extract` which reshapes 1D
# inputs to `M×1` matrices. `P<:BlasFloat` covers real and complex element types.
@inline function frule!!(
    ::Dual{typeof(BLAS.gemv!)},
    tA::Dual{Char},
    alpha::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    A_dA::Dual{<:AbstractVecOrMat{<:BlasFloat}},
    x_dx::Dual{<:AbstractVector{<:BlasFloat}},
    beta::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    y_dy::Dual{<:AbstractVector{<:BlasFloat}},
)
    A, dA = _mat_extract(A_dA)
    x, dx = _arr_extract(x_dx)
    y, dy = _arr_extract(y_dy)
    α, dα = _scalar_extract(alpha)
    β, dβ = _scalar_extract(beta)

    _gemv!_frule_core!(primal(tA), α, dα, A, dA, x, dx, β, dβ, y, dy)

    _arr_writeback!(y_dy, y, dy)
    return y_dy
end
# Width-N NDual gemv!: per-lane Frechet computed N times
# (each with independent lane tangents), then primal computed once. We
# inline the per-lane Frechet to avoid `_gemv!_frule_core!`'s baked-in
# primal computation (which would accumulate y N times if called per-lane).
@inline function _gemv_frechet_lane!(tA, α::P, dα, A, dA, x, dx, β, dβ, y, dy) where {P}
    BLAS.gemv!(tA, dα, A, x, β, dy)
    BLAS.gemv!(tA, α, dA, x, one(P), dy)
    BLAS.gemv!(tA, α, A, dx, one(P), dy)
    if !iszero(dβ)
        @inbounds for n in eachindex(y)
            tmp = dβ * y[n]
            dy[n] = ifelse(isnan(y[n]), dy[n], tmp + dy[n])
        end
    end
    return nothing
end
# Consolidated width-N gemv!: covers Real (NDual{P,N}) and Complex
# (Complex{NDual{P,N}}). Bodies were byte-identical.
@inline function frule!!(
    ::Dual{typeof(BLAS.gemv!)},
    tA::Dual{Char},
    alpha::Union{NDual{P,N},Complex{NDual{P,N}}},
    A_dA::AbstractVecOrMat{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    x_dx::AbstractArray{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    beta::Union{NDual{P,N},Complex{NDual{P,N}}},
    y_dy::AbstractArray{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
) where {P<:IEEEFloat,N}
    A, dAs = _mat_extract_n(A_dA)
    x, dxs = _arr_extract_n(x_dx)
    y, dys = _arr_extract_n(y_dy)
    α, dαs = _scalar_extract_n(alpha)
    β, dβs = _scalar_extract_n(beta)
    ta = primal(tA)
    @inbounds for lane in 1:N
        _gemv_frechet_lane!(
            ta, α, dαs[lane], A, dAs[lane], x, dxs[lane], β, dβs[lane], y, dys[lane]
        )
    end
    BLAS.gemv!(ta, α, A, x, β, y)
    _arr_writeback_n!(y_dy, y, dys)
    return y_dy
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.gemv!),N},
    tA::Mooncake.Lifted{Char},
    alpha::Mooncake.Lifted{P},
    A_dA::Mooncake.Lifted{<:AbstractVecOrMat{P}},
    x_dx::Mooncake.Lifted{<:AbstractArray{P}},
    beta::Mooncake.Lifted{P},
    y_dy::Mooncake.Lifted{<:AbstractArray{P}},
) where {N,P<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(tA),
        Mooncake._unlift(alpha),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(x_dx),
        Mooncake._unlift(beta),
        Mooncake._unlift(y_dy),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end

@inline function _gemv!_frule_core!(
    tA::Char,
    α::P,
    dα::P,
    A::AbstractMatrix{P},
    dA::AbstractMatrix{P},
    x::AbstractVector{P},
    dx::AbstractVector{P},
    β::P,
    dβ::P,
    y::AbstractVector{P},
    dy::AbstractVector{P},
) where {P<:BlasFloat}
    # Derivative computation.
    BLAS.gemv!(tA, dα, A, x, β, dy)
    BLAS.gemv!(tA, α, dA, x, one(P), dy)
    BLAS.gemv!(tA, α, A, dx, one(P), dy)

    # Strong zero is essential here, in case `y` has undefined element values.
    if !iszero(dβ)
        @inbounds for n in eachindex(y)
            tmp = dβ * y[n]
            dy[n] = ifelse(isnan(y[n]), dy[n], tmp + dy[n])
        end
    end

    # Primal computation.
    BLAS.gemv!(tA, α, A, x, β, y)
    return nothing
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

# Note that the complex symv are not BLAS but auxiliary functions in LAPACK.
# `symv!` is primitive for `T<:BlasFloat`, but its bare-`Dual` `frule!!` only
# needs to cover `BlasComplexFloat` because the width1-aware method below
# handles `BlasRealFloat` (its `Union{Dual,NDual}` arg shapes subsume the
# pure-`Dual` case there). `hemv!` is restricted to `BlasComplexFloat`.
for (fname, prim_elty, frule_elty) in (
    (:(symv!), BlasFloat, BlasComplexFloat), (:(hemv!), BlasComplexFloat, BlasComplexFloat)
)
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
        } where {T<:$prim_elty},
    )

    @eval function frule!!(
        ::Dual{typeof(BLAS.$fname)},
        uplo::Dual{Char},
        alpha::Dual{T},
        A_dA::Dual{<:AbstractMatrix{T}},
        x_dx::Dual{<:AbstractVector{T}},
        beta::Dual{T},
        y_dy::Dual{<:AbstractVector{T}},
    ) where {T<:$frule_elty}
        # Extract primals.
        ul = primal(uplo)
        α, dα = extract(alpha)
        β, dβ = extract(beta)
        A, dA = arrayify(A_dA)
        x, dx = arrayify(x_dx)
        y, dy = arrayify(y_dy)

        # Compute Frechet derivative.
        BLAS.$fname(ul, dα, A, x, β, dy)
        BLAS.$fname(ul, α, dA, x, one(T), dy)
        BLAS.$fname(ul, α, A, dx, one(T), dy)
        if !iszero(dβ)
            @inbounds for n in eachindex(y)
                tmp = dβ * y[n]
                dy[n] = ifelse(isnan(y[n]), dy[n], tmp + dy[n])
            end
        end

        # Run primal computation.
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
    ) where {T<:$prim_elty}

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

# Width-1 path with mixed-wrapper support: container args narrowed to
# bare-`Dual{<:Abstract...}` to disambiguate from the width-N rule at N=1
# (NDual array inputs route to width-N), while scalar args use Union
# `Dual{T} | NDual{T,1}` for mixed-wrapper inputs. Consolidated to cover
# both Real and Complex element types — `BLAS.symv!` natively dispatches on
# element type and `one(eltype(y))` selects the correct β scalar shape.
function frule!!(
    ::Dual{typeof(BLAS.symv!)},
    uplo::Dual{Char},
    alpha::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    A_dA::Dual{<:AbstractMatrix{<:BlasFloat}},
    x_dx::Dual{<:AbstractVector{<:BlasFloat}},
    beta::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    y_dy::Dual{<:AbstractVector{<:BlasFloat}},
)
    ul = primal(uplo)
    α, dα = _scalar_extract(alpha)
    β, dβ = _scalar_extract(beta)
    A, dA = _arr_extract(A_dA)
    x, dx = _arr_extract(x_dx)
    y, dy = _arr_extract(y_dy)

    BLAS.symv!(ul, dα, A, x, β, dy)
    BLAS.symv!(ul, α, dA, x, one(eltype(y)), dy)
    BLAS.symv!(ul, α, A, dx, one(eltype(y)), dy)
    if !iszero(dβ)
        @inbounds for n in eachindex(y)
            tmp = dβ * y[n]
            dy[n] = ifelse(isnan(y[n]), dy[n], tmp + dy[n])
        end
    end

    BLAS.symv!(ul, α, A, x, β, y)
    _arr_writeback!(y_dy, y, dy)
    return y_dy
end

@inline Mooncake._is_lifted_aware(
    ::Type{
        <:Tuple{
            typeof(BLAS.symv!),
            Char,
            T,
            AbstractMatrix{T},
            AbstractVector{T},
            T,
            AbstractVector{T},
        },
    },
) where {T<:BlasFloat} = true
@inline Mooncake._is_lifted_aware(
    ::Type{
        <:Tuple{
            typeof(BLAS.hemv!),
            Char,
            T,
            AbstractMatrix{T},
            AbstractVector{T},
            T,
            AbstractVector{T},
        },
    },
) where {T<:BlasComplexFloat} = true
# hemv! width-1 Complex canonical form. The Real and Complex symv! width-1
# entries above are consolidated into a single BlasFloat-typed body; hemv!
# is Complex-only so its width-1 entry stays separate.
@inline function frule!!(
    ::Dual{typeof(BLAS.hemv!)},
    uplo::Dual{Char},
    alpha::_ScalarLikeWidth1Complex{R},
    A_dA::Dual{<:AbstractMatrix{Complex{R}}},
    x_dx::Dual{<:AbstractVector{Complex{R}}},
    beta::_ScalarLikeWidth1Complex{R},
    y_dy::Dual{<:AbstractVector{Complex{R}}},
) where {R<:IEEEFloat}
    ul = primal(uplo)
    α, dα = _scalar_extract(alpha)
    β, dβ = _scalar_extract(beta)
    A, dA = _arr_extract(A_dA)
    x, dx = _arr_extract(x_dx)
    y, dy = _arr_extract(y_dy)

    BLAS.hemv!(ul, dα, A, x, β, dy)
    BLAS.hemv!(ul, α, dA, x, one(eltype(y)), dy)
    BLAS.hemv!(ul, α, A, dx, one(eltype(y)), dy)
    if !iszero(dβ)
        @inbounds for n in eachindex(y)
            tmp = dβ * y[n]
            dy[n] = ifelse(isnan(y[n]), dy[n], tmp + dy[n])
        end
    end

    BLAS.hemv!(ul, α, A, x, β, y)
    _arr_writeback!(y_dy, y, dy)
    return y_dy
end

# Width-N NDual symv!/hemv!: per-lane Frechet then primal
# once. The Frechet body matches the existing width-1 inline. Two separate
# per-op helpers (`symv!`/`hemv!`) because BLAS dispatch by name differs.
@inline function _symv_frechet_lane!(ul, α::P, dα, A, dA, x, dx, β, dβ, y, dy) where {P}
    BLAS.symv!(ul, dα, A, x, β, dy)
    BLAS.symv!(ul, α, dA, x, one(P), dy)
    BLAS.symv!(ul, α, A, dx, one(P), dy)
    if !iszero(dβ)
        @inbounds for n in eachindex(y)
            tmp = dβ * y[n]
            dy[n] = ifelse(isnan(y[n]), dy[n], tmp + dy[n])
        end
    end
    return nothing
end
@inline function _hemv_frechet_lane!(ul, α::P, dα, A, dA, x, dx, β, dβ, y, dy) where {P}
    BLAS.hemv!(ul, dα, A, x, β, dy)
    BLAS.hemv!(ul, α, dA, x, one(P), dy)
    BLAS.hemv!(ul, α, A, dx, one(P), dy)
    if !iszero(dβ)
        @inbounds for n in eachindex(y)
            tmp = dβ * y[n]
            dy[n] = ifelse(isnan(y[n]), dy[n], tmp + dy[n])
        end
    end
    return nothing
end
# Consolidated width-N symv!: covers Real and Complex.
@inline function frule!!(
    ::Dual{typeof(BLAS.symv!)},
    uplo::Dual{Char},
    alpha::Union{NDual{P,N},Complex{NDual{P,N}}},
    A_dA::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    x_dx::AbstractVector{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    beta::Union{NDual{P,N},Complex{NDual{P,N}}},
    y_dy::AbstractVector{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
) where {P<:IEEEFloat,N}
    ul = primal(uplo)
    α, dαs = _scalar_extract_n(alpha)
    β, dβs = _scalar_extract_n(beta)
    A, dAs = _arr_extract_n(A_dA)
    x, dxs = _arr_extract_n(x_dx)
    y, dys = _arr_extract_n(y_dy)
    @inbounds for lane in 1:N
        _symv_frechet_lane!(
            ul, α, dαs[lane], A, dAs[lane], x, dxs[lane], β, dβs[lane], y, dys[lane]
        )
    end
    BLAS.symv!(ul, α, A, x, β, y)
    _arr_writeback_n!(y_dy, y, dys)
    return y_dy
end
@inline function frule!!(
    ::Dual{typeof(BLAS.hemv!)},
    uplo::Dual{Char},
    alpha::Complex{NDual{R,N}},
    A_dA::AbstractMatrix{Complex{NDual{R,N}}},
    x_dx::AbstractVector{Complex{NDual{R,N}}},
    beta::Complex{NDual{R,N}},
    y_dy::AbstractVector{Complex{NDual{R,N}}},
) where {R<:IEEEFloat,N}
    ul = primal(uplo)
    α, dαs = _scalar_extract_n(alpha)
    β, dβs = _scalar_extract_n(beta)
    A, dAs = _arr_extract_n(A_dA)
    x, dxs = _arr_extract_n(x_dx)
    y, dys = _arr_extract_n(y_dy)
    @inbounds for lane in 1:N
        _hemv_frechet_lane!(
            ul, α, dαs[lane], A, dAs[lane], x, dxs[lane], β, dβs[lane], y, dys[lane]
        )
    end
    BLAS.hemv!(ul, α, A, x, β, y)
    _arr_writeback_n!(y_dy, y, dys)
    return y_dy
end

@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.symv!),N},
    uplo::Mooncake.Lifted{Char},
    alpha::Mooncake.Lifted{T},
    A_dA::Mooncake.Lifted{<:AbstractMatrix{T}},
    x_dx::Mooncake.Lifted{<:AbstractVector{T}},
    beta::Mooncake.Lifted{T},
    y_dy::Mooncake.Lifted{<:AbstractVector{T}},
) where {N,T<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(uplo),
        Mooncake._unlift(alpha),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(x_dx),
        Mooncake._unlift(beta),
        Mooncake._unlift(y_dy),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.hemv!),N},
    uplo::Mooncake.Lifted{Char},
    alpha::Mooncake.Lifted{T},
    A_dA::Mooncake.Lifted{<:AbstractMatrix{T}},
    x_dx::Mooncake.Lifted{<:AbstractVector{T}},
    beta::Mooncake.Lifted{T},
    y_dy::Mooncake.Lifted{<:AbstractVector{T}},
) where {N,T<:BlasComplexFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(uplo),
        Mooncake._unlift(alpha),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(x_dx),
        Mooncake._unlift(beta),
        Mooncake._unlift(y_dy),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.trmv!),Char,Char,Char,AbstractMatrix{T},AbstractVector{T}
    } where {T<:BlasFloat},
)
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.trmv!),Vararg}}) = true

# Width-1 path: container args narrowed to bare-`Dual{<:Abstract...}` to
# disambiguate from the width-N rule at N=1 (NDual array inputs route to
# width-N).
function frule!!(
    ::Dual{typeof(BLAS.trmv!)},
    _uplo::Dual{Char},
    _trans::Dual{Char},
    _diag::Dual{Char},
    A_dA::Dual{<:AbstractMatrix{T}},
    x_dx::Dual{<:AbstractVector{T}},
) where {T<:BlasFloat}
    A, dA = _arr_extract(A_dA)
    x, dx = _arr_extract(x_dx)
    _trmv!_frule_core!(primal(_uplo), primal(_trans), primal(_diag), A, dA, x, dx)
    _arr_writeback!(x_dx, x, dx)
    return x_dx
end
# Width-N NDual trmv!: per-lane Frechet (which depends on
# pre-primal x), then primal once.
@inline function _trmv_frechet_lane!(uplo, trans, diag, A, dA, x, dx)
    BLAS.trmv!(uplo, trans, diag, A, dx)
    tmp = copy(x)
    BLAS.trmv!(uplo, trans, diag, dA, tmp)
    dx .+= tmp
    if diag === 'U'
        dx .-= x
    end
    return nothing
end
# Consolidated width-N trmv!: covers Real (NDual{T,N}) and Complex
# (Complex{NDual{R,N}}). Bodies were byte-identical.
@inline function frule!!(
    ::Dual{typeof(BLAS.trmv!)},
    _uplo::Dual{Char},
    _trans::Dual{Char},
    _diag::Dual{Char},
    A_dA::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    x_dx::AbstractVector{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
) where {P<:IEEEFloat,N}
    A, dAs = _arr_extract_n(A_dA)
    x, dxs = _arr_extract_n(x_dx)
    uplo = primal(_uplo)
    trans = primal(_trans)
    diag = primal(_diag)
    @inbounds for lane in 1:N
        _trmv_frechet_lane!(uplo, trans, diag, A, dAs[lane], x, dxs[lane])
    end
    BLAS.trmv!(uplo, trans, diag, A, x)
    _arr_writeback_n!(x_dx, x, dxs)
    return x_dx
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.trmv!),N},
    _uplo::Mooncake.Lifted{Char},
    _trans::Mooncake.Lifted{Char},
    _diag::Mooncake.Lifted{Char},
    A_dA::Mooncake.Lifted{<:AbstractMatrix{T}},
    x_dx::Mooncake.Lifted{<:AbstractVector{T}},
) where {N,T<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(_uplo),
        Mooncake._unlift(_trans),
        Mooncake._unlift(_diag),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(x_dx),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
@inline function _trmv!_frule_core!(
    uplo::Char,
    trans::Char,
    diag::Char,
    A::AbstractMatrix{T},
    dA::AbstractMatrix{T},
    x::AbstractVector{T},
    dx::AbstractVector{T},
) where {T<:BlasFloat}
    BLAS.trmv!(uplo, trans, diag, A, dx)
    tmp = copy(x)
    BLAS.trmv!(uplo, trans, diag, dA, tmp)
    dx .+= tmp
    if diag === 'U'
        dx .-= x
    end
    BLAS.trmv!(uplo, trans, diag, A, x)
    return nothing
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
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.trsv!),Vararg}}) = true
# Width-1 path: container args narrowed to bare-`Dual{<:Abstract...}` to
# disambiguate from the width-N rule at N=1 (NDual array inputs route to
# width-N). `T<:BlasFloat` covers both real and complex element types.
function frule!!(
    ::Dual{typeof(BLAS.trsv!)},
    _uplo::Dual{Char},
    _trans::Dual{Char},
    _diag::Dual{Char},
    A_dA::Dual{<:AbstractMatrix{T}},
    x_dx::Dual{<:AbstractVector{T}},
) where {T<:BlasFloat}
    A, dA = _arr_extract(A_dA)
    x, dx = _arr_extract(x_dx)
    _trsv!_frule_core!(primal(_uplo), primal(_trans), primal(_diag), A, dA, x, dx)
    _arr_writeback!(x_dx, x, dx)
    return x_dx
end
# Width-N NDual trsv!: primal first (x ← A^{-1} x), then
# per-lane Frechet (which uses the post-primal x).
@inline function _trsv_frechet_lane!(uplo, trans, diag, A, dA, x, dx)
    BLAS.trsv!(uplo, trans, diag, A, dx)
    tmp = BLAS.trmv(uplo, trans, diag, dA, x)
    if diag == 'U'
        tmp .-= x
    end
    BLAS.trsv!(uplo, trans, diag, A, tmp)
    dx .-= tmp
    return nothing
end
@inline function frule!!(
    ::Dual{typeof(BLAS.trsv!)},
    _uplo::Dual{Char},
    _trans::Dual{Char},
    _diag::Dual{Char},
    A_dA::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    x_dx::AbstractVector{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
) where {P<:BlasRealFloat,N}
    A, dAs = _arr_extract_n(A_dA)
    x, dxs = _arr_extract_n(x_dx)
    uplo = primal(_uplo)
    trans = primal(_trans)
    diag = primal(_diag)
    BLAS.trsv!(uplo, trans, diag, A, x)
    @inbounds for lane in 1:N
        _trsv_frechet_lane!(uplo, trans, diag, A, dAs[lane], x, dxs[lane])
    end
    _arr_writeback_n!(x_dx, x, dxs)
    return x_dx
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.trsv!),N},
    _uplo::Mooncake.Lifted{Char},
    _trans::Mooncake.Lifted{Char},
    _diag::Mooncake.Lifted{Char},
    A_dA::Mooncake.Lifted{<:AbstractMatrix{T}},
    x_dx::Mooncake.Lifted{<:AbstractVector{T}},
) where {N,T<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(_uplo),
        Mooncake._unlift(_trans),
        Mooncake._unlift(_diag),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(x_dx),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
@inline function _trsv!_frule_core!(
    uplo::Char,
    trans::Char,
    diag::Char,
    A::AbstractMatrix{T},
    dA::AbstractMatrix{T},
    x::AbstractVector{T},
    dx::AbstractVector{T},
) where {T<:BlasFloat}
    BLAS.trsv!(uplo, trans, diag, A, x)
    BLAS.trsv!(uplo, trans, diag, A, dx)
    tmp = BLAS.trmv(uplo, trans, diag, dA, x)
    if diag == 'U'
        tmp .-= x
    end
    BLAS.trsv!(uplo, trans, diag, A, tmp)
    dx .-= tmp
    return nothing
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
        AbstractVecOrMat{T},
    } where {T<:BlasFloat},
)
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.gemm!),Vararg}}) = true

# Helper function to avoid NaN poisoning caused due to adding undef or non initialized C matrices.
function ifelse_nan(cond, left::P, right::P) where {P<:BlasFloat}
    return isnan(cond) * left + !isnan(cond) * right
end

# Consolidated width-1 gemm!: covers Real and Complex via element-type
# union signatures. `one(eltype(C))` works for both `Float64` and `Complex{Float64}`.
@inline function frule!!(
    ::Dual{typeof(BLAS.gemm!)},
    transA::Dual{Char},
    transB::Dual{Char},
    alpha::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    A_dA::Union{_VecOrMatLikeWidth1,_VecOrMatLikeWidth1Complex},
    B_dB::Union{_VecOrMatLikeWidth1,_VecOrMatLikeWidth1Complex},
    beta::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    C_dC::Union{_MatLikeWidth1,_MatLikeWidth1Complex},
)
    tA = primal(transA)
    tB = primal(transB)
    α, dα = _scalar_extract(alpha)
    β, dβ = _scalar_extract(beta)
    A, dA = _arr_extract(A_dA)
    B, dB = _arr_extract(B_dB)
    C, dC = _arr_extract(C_dC)
    one_T = one(eltype(C))

    # Tangents (product rule)
    # d(α*op(A)*op(B) + β*C) = dα*op(A)*op(B) + α*op(dA)*op(B) + α*op(A)*op(dB) + dβ*C + β*dC
    BLAS.gemm!(tA, tB, α, dA, B, β, dC)      # α*op(dA)*op(B) + β*dC
    BLAS.gemm!(tA, tB, α, A, dB, one_T, dC)  # α*op(A)*op(dB) + 1*dC

    if !iszero(dα)
        BLAS.gemm!(tA, tB, dα, A, B, one_T, dC)  # dα*op(A)*op(B) + 1*dC
    end

    if !iszero(dβ)
        @inbounds for n in eachindex(C)
            dC[n] = ifelse_nan(C[n], dC[n], dC[n] + dβ * C[n])
        end
    end

    # Primal
    BLAS.gemm!(tA, tB, α, A, B, β, C)

    _arr_writeback!(C_dC, C, dC)
    return C_dC
end
# Width-N NDual gemm!: per-lane Frechet then primal once.
@inline function _gemm_frechet_lane!(tA, tB, α::P, dα, A, dA, B, dB, β, dβ, C, dC) where {P}
    BLAS.gemm!(tA, tB, α, dA, B, β, dC)
    BLAS.gemm!(tA, tB, α, A, dB, one(P), dC)
    if !iszero(dα)
        BLAS.gemm!(tA, tB, dα, A, B, one(P), dC)
    end
    if !iszero(dβ)
        @inbounds for n in eachindex(C)
            dC[n] = ifelse_nan(C[n], dC[n], dC[n] + dβ * C[n])
        end
    end
    return nothing
end
# Consolidated width-N gemm!: covers Real and Complex via element-type
# union with shared `P<:IEEEFloat` typevar.
@inline function frule!!(
    ::Dual{typeof(BLAS.gemm!)},
    transA::Dual{Char},
    transB::Dual{Char},
    alpha::Union{NDual{P,N},Complex{NDual{P,N}}},
    A_dA::AbstractVecOrMat{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    B_dB::AbstractVecOrMat{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    beta::Union{NDual{P,N},Complex{NDual{P,N}}},
    C_dC::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
) where {P<:IEEEFloat,N}
    tA = primal(transA)
    tB = primal(transB)
    α, dαs = _scalar_extract_n(alpha)
    β, dβs = _scalar_extract_n(beta)
    A, dAs = _arr_extract_n(A_dA)
    B, dBs = _arr_extract_n(B_dB)
    C, dCs = _arr_extract_n(C_dC)
    @inbounds for lane in 1:N
        _gemm_frechet_lane!(
            tA, tB, α, dαs[lane], A, dAs[lane], B, dBs[lane], β, dβs[lane], C, dCs[lane]
        )
    end
    BLAS.gemm!(tA, tB, α, A, B, β, C)
    _arr_writeback_n!(C_dC, C, dCs)
    return C_dC
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.gemm!),N},
    transA::Mooncake.Lifted{Char},
    transB::Mooncake.Lifted{Char},
    alpha::Mooncake.Lifted{T},
    A_dA::Mooncake.Lifted{<:AbstractVecOrMat{T}},
    B_dB::Mooncake.Lifted{<:AbstractVecOrMat{T}},
    beta::Mooncake.Lifted{T},
    C_dC::Mooncake.Lifted{<:AbstractMatrix{T}},
) where {N,T<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(transA),
        Mooncake._unlift(transB),
        Mooncake._unlift(alpha),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(B_dB),
        Mooncake._unlift(beta),
        Mooncake._unlift(C_dC),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
        p_C .= a .* tmp .+ b .* p_C
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
        ::Dual{typeof(BLAS.$fname)},
        side::Dual{Char},
        uplo::Dual{Char},
        alpha::Dual{T},
        A_dA::Dual{<:AbstractMatrix{T}},
        B_dB::Dual{<:AbstractMatrix{T}},
        beta::Dual{T},
        C_dC::Dual{<:AbstractMatrix{T}},
    ) where {T<:$elty}

        # Extract primals.
        s = primal(side)
        ul = primal(uplo)
        α, dα = extract(alpha)
        β, dβ = extract(beta)
        A, dA = arrayify(A_dA)
        B, dB = arrayify(B_dB)
        C, dC = arrayify(C_dC)

        # Compute Frechet derivative.
        BLAS.$fname(s, ul, α, A, dB, β, dC)
        BLAS.$fname(s, ul, α, dA, B, one(T), dC)
        if !iszero(dα)
            BLAS.$fname(s, ul, dα, A, B, one(T), dC)
        end
        if !iszero(dβ)
            @inbounds for n in eachindex(C)
                dC[n] = ifelse_nan(C[n], dC[n], dC[n] + dβ * C[n])
            end
        end

        # Run primal computation.
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
            C .= α .* tmp .+ β .* C
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

# Width-N NDual symm!/hemm!: per-lane Frechet then primal
# once. Mirrors the legacy `Dual{<:AbstractMatrix{T}}` rule body but
# operates on bare NDual-element matrices and uses per-lane assembly.
@inline function _symm_frechet_lane!(s, ul, α::P, dα, A, dA, B, dB, β, dβ, C, dC) where {P}
    BLAS.symm!(s, ul, α, A, dB, β, dC)
    BLAS.symm!(s, ul, α, dA, B, one(P), dC)
    if !iszero(dα)
        BLAS.symm!(s, ul, dα, A, B, one(P), dC)
    end
    if !iszero(dβ)
        @inbounds for n in eachindex(C)
            dC[n] = ifelse_nan(C[n], dC[n], dC[n] + dβ * C[n])
        end
    end
    return nothing
end
@inline function _hemm_frechet_lane!(s, ul, α::P, dα, A, dA, B, dB, β, dβ, C, dC) where {P}
    BLAS.hemm!(s, ul, α, A, dB, β, dC)
    BLAS.hemm!(s, ul, α, dA, B, one(P), dC)
    if !iszero(dα)
        BLAS.hemm!(s, ul, dα, A, B, one(P), dC)
    end
    if !iszero(dβ)
        @inbounds for n in eachindex(C)
            dC[n] = ifelse_nan(C[n], dC[n], dC[n] + dβ * C[n])
        end
    end
    return nothing
end
# Consolidated width-N symm!: covers Real (NDual{P,N}) and Complex
# (Complex{NDual{P,N}}). Bodies were byte-identical; collapsed via the
# element-type Union pattern with shared P<:BlasRealFloat typevar.
@inline function frule!!(
    ::Dual{typeof(BLAS.symm!)},
    side::Dual{Char},
    uplo::Dual{Char},
    alpha::Union{NDual{P,N},Complex{NDual{P,N}}},
    A_dA::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    B_dB::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    beta::Union{NDual{P,N},Complex{NDual{P,N}}},
    C_dC::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
) where {P<:BlasRealFloat,N}
    s = primal(side)
    ul = primal(uplo)
    α, dαs = _scalar_extract_n(alpha)
    β, dβs = _scalar_extract_n(beta)
    A, dAs = _arr_extract_n(A_dA)
    B, dBs = _arr_extract_n(B_dB)
    C, dCs = _arr_extract_n(C_dC)
    @inbounds for lane in 1:N
        _symm_frechet_lane!(
            s, ul, α, dαs[lane], A, dAs[lane], B, dBs[lane], β, dβs[lane], C, dCs[lane]
        )
    end
    BLAS.symm!(s, ul, α, A, B, β, C)
    _arr_writeback_n!(C_dC, C, dCs)
    return C_dC
end
@inline function frule!!(
    ::Dual{typeof(BLAS.hemm!)},
    side::Dual{Char},
    uplo::Dual{Char},
    alpha::Complex{NDual{R,N}},
    A_dA::AbstractMatrix{Complex{NDual{R,N}}},
    B_dB::AbstractMatrix{Complex{NDual{R,N}}},
    beta::Complex{NDual{R,N}},
    C_dC::AbstractMatrix{Complex{NDual{R,N}}},
) where {R<:IEEEFloat,N}
    s = primal(side)
    ul = primal(uplo)
    α, dαs = _scalar_extract_n(alpha)
    β, dβs = _scalar_extract_n(beta)
    A, dAs = _arr_extract_n(A_dA)
    B, dBs = _arr_extract_n(B_dB)
    C, dCs = _arr_extract_n(C_dC)
    @inbounds for lane in 1:N
        _hemm_frechet_lane!(
            s, ul, α, dαs[lane], A, dAs[lane], B, dBs[lane], β, dβs[lane], C, dCs[lane]
        )
    end
    BLAS.hemm!(s, ul, α, A, B, β, C)
    _arr_writeback_n!(C_dC, C, dCs)
    return C_dC
end

# Lifted-typed delegators for symm!/hemm!: route through the bare-Dual
# entries (canonical width-1 NDual via the width-N rule at N=1, or the
# @eval-generated wrapper-exception entry for Dual{<:AbstractMatrix{T}}).
# Trait registrations enable Lifted-aware IR emission.
@inline Mooncake._is_lifted_aware(
    ::Type{
        <:Tuple{
            typeof(BLAS.symm!),
            Char,
            Char,
            T,
            AbstractMatrix{T},
            AbstractMatrix{T},
            T,
            AbstractMatrix{T},
        },
    },
) where {T<:BlasFloat} = true
@inline Mooncake._is_lifted_aware(
    ::Type{
        <:Tuple{
            typeof(BLAS.hemm!),
            Char,
            Char,
            T,
            AbstractMatrix{T},
            AbstractMatrix{T},
            T,
            AbstractMatrix{T},
        },
    },
) where {T<:BlasComplexFloat} = true
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.symm!),N},
    side::Mooncake.Lifted{Char},
    uplo::Mooncake.Lifted{Char},
    alpha::Mooncake.Lifted{T},
    A_dA::Mooncake.Lifted{<:AbstractMatrix{T}},
    B_dB::Mooncake.Lifted{<:AbstractMatrix{T}},
    beta::Mooncake.Lifted{T},
    C_dC::Mooncake.Lifted{<:AbstractMatrix{T}},
) where {N,T<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(side),
        Mooncake._unlift(uplo),
        Mooncake._unlift(alpha),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(B_dB),
        Mooncake._unlift(beta),
        Mooncake._unlift(C_dC),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.hemm!),N},
    side::Mooncake.Lifted{Char},
    uplo::Mooncake.Lifted{Char},
    alpha::Mooncake.Lifted{T},
    A_dA::Mooncake.Lifted{<:AbstractMatrix{T}},
    B_dB::Mooncake.Lifted{<:AbstractMatrix{T}},
    beta::Mooncake.Lifted{T},
    C_dC::Mooncake.Lifted{<:AbstractMatrix{T}},
) where {N,T<:BlasComplexFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(side),
        Mooncake._unlift(uplo),
        Mooncake._unlift(alpha),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(B_dB),
        Mooncake._unlift(beta),
        Mooncake._unlift(C_dC),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
        ::Dual{typeof(BLAS.$fname)},
        _uplo::Dual{Char},
        _t::Dual{Char},
        α_dα::Dual{$relty},
        A_dA::Dual{<:AbstractVecOrMat{$elty}},
        β_dβ::Dual{$relty},
        C_dC::Dual{<:AbstractMatrix{$elty}},
    )

        # Extract values from pairs.
        uplo = primal(_uplo)
        t = primal(_t)
        α, dα = extract(α_dα)
        A, dA = matrixify(A_dA)
        β, dβ = extract(β_dβ)
        C, dC = arrayify(C_dC)

        # Compute Frechet derivative.
        BLAS.$(isherm ? :her2k! : :syr2k!)(uplo, t, $elty(α), A, dA, β, dC)
        iszero(dα) || BLAS.$fname(uplo, t, dα, A, one($relty), dC)
        if !iszero(dβ)
            dC .+= dβ .* (uplo == 'U' ? triu(C) : tril(C))
        end
        # BLAS will zero out the imaginary parts on the diagonal of C,
        # do the same on the tangent
        $(isherm ? :(real_diag!(dC)) : :())

        # Run primal computation.
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

# Width-N NDual `syrk!` / `herk!`. α and β are always real
# (BlasRealFloat); matrix elements are real for syrk-real-elty and complex
# for syrk-complex-elty / herk-complex-elty. We handle each combination
# separately so the runtime dispatch sees concrete element types.
@inline function _syrk_frechet_lane!(uplo, t, α::P, dα, A, dA, β, dβ, C, dC) where {P}
    BLAS.syr2k!(uplo, t, P(α), A, dA, β, dC)
    iszero(dα) || BLAS.syrk!(uplo, t, dα, A, one(P), dC)
    if !iszero(dβ)
        dC .+= dβ .* (uplo == 'U' ? triu(C) : tril(C))
    end
    return nothing
end
@inline function _herk_frechet_lane!(
    uplo, t, α::R, dα, A::AbstractMatrix{Complex{R}}, dA, β::R, dβ, C, dC
) where {R<:Real}
    BLAS.her2k!(uplo, t, Complex{R}(α), A, dA, β, dC)
    iszero(dα) || BLAS.herk!(uplo, t, dα, A, one(R), dC)
    if !iszero(dβ)
        dC .+= dβ .* (uplo == 'U' ? triu(C) : tril(C))
    end
    real_diag!(dC)
    return nothing
end
# syrk!: covers real-elty (α/β/A/C all real) and complex-elty (α/β real, A/C complex) paths.
@inline function frule!!(
    ::Dual{typeof(BLAS.syrk!)},
    _uplo::Dual{Char},
    _t::Dual{Char},
    α_dα::NDual{R,N},
    A_dA::AbstractVecOrMat{<:Union{NDual{R,N},Complex{NDual{R,N}}}},
    β_dβ::NDual{R,N},
    C_dC::AbstractMatrix{<:Union{NDual{R,N},Complex{NDual{R,N}}}},
) where {R<:BlasRealFloat,N}
    uplo = primal(_uplo)
    t = primal(_t)
    α, dαs = _scalar_extract_n(α_dα)
    β, dβs = _scalar_extract_n(β_dβ)
    A, dAs = _mat_extract_n(A_dA)
    C, dCs = _arr_extract_n(C_dC)
    @inbounds for lane in 1:N
        _syrk_frechet_lane!(uplo, t, α, dαs[lane], A, dAs[lane], β, dβs[lane], C, dCs[lane])
    end
    BLAS.syrk!(uplo, t, α, A, β, C)
    _arr_writeback_n!(C_dC, C, dCs)
    return C_dC
end
# herk!: complex matrix, real α/β.
@inline function frule!!(
    ::Dual{typeof(BLAS.herk!)},
    _uplo::Dual{Char},
    _t::Dual{Char},
    α_dα::NDual{R,N},
    A_dA::AbstractVecOrMat{Complex{NDual{R,N}}},
    β_dβ::NDual{R,N},
    C_dC::AbstractMatrix{Complex{NDual{R,N}}},
) where {R<:BlasRealFloat,N}
    uplo = primal(_uplo)
    t = primal(_t)
    α, dαs = _scalar_extract_n(α_dα)
    β, dβs = _scalar_extract_n(β_dβ)
    A, dAs = _mat_extract_n(A_dA)
    C, dCs = _arr_extract_n(C_dC)
    @inbounds for lane in 1:N
        _herk_frechet_lane!(uplo, t, α, dαs[lane], A, dAs[lane], β, dβs[lane], C, dCs[lane])
    end
    BLAS.herk!(uplo, t, α, A, β, C)
    real_diag!(C)
    _arr_writeback_n!(C_dC, C, dCs)
    return C_dC
end

# Consolidated width-1 syrk!: covers Real and Complex. The underlying
# `BLAS.syr2k!` and `BLAS.syrk!` calls natively dispatch on element type;
# `one(eltype(C))` yields the correct β scalar shape for either.
function frule!!(
    ::Dual{typeof(BLAS.syrk!)},
    _uplo::Dual{Char},
    _t::Dual{Char},
    α_dα::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    A_dA::Union{_VecOrMatLikeWidth1,_VecOrMatLikeWidth1Complex},
    β_dβ::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    C_dC::Union{_MatLikeWidth1,_MatLikeWidth1Complex},
)
    uplo = primal(_uplo)
    t = primal(_t)
    α, dα = _scalar_extract(α_dα)
    A, dA = _mat_extract(A_dA)
    β, dβ = _scalar_extract(β_dβ)
    C, dC = _arr_extract(C_dC)

    BLAS.syr2k!(uplo, t, α, A, dA, β, dC)
    iszero(dα) || BLAS.syrk!(uplo, t, dα, A, one(eltype(C)), dC)
    if !iszero(dβ)
        dC .+= dβ .* (uplo == 'U' ? triu(C) : tril(C))
    end

    BLAS.syrk!(uplo, t, α, A, β, C)
    _arr_writeback!(C_dC, C, dC)
    return C_dC
end

@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(BLAS.syrk!),Char,Char,T,AbstractVecOrMat{T},T,AbstractMatrix{T}}}
) where {T<:BlasFloat} = true

@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.syrk!),N},
    uplo::Mooncake.Lifted{Char},
    t::Mooncake.Lifted{Char},
    α_dα::Mooncake.Lifted{T},
    A_dA::Mooncake.Lifted{<:AbstractVecOrMat{T}},
    β_dβ::Mooncake.Lifted{T},
    C_dC::Mooncake.Lifted{<:AbstractMatrix{T}},
) where {N,T<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(uplo),
        Mooncake._unlift(t),
        Mooncake._unlift(α_dα),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(β_dβ),
        Mooncake._unlift(C_dC),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end

# herk! Lifted delegator + trait. herk! takes Complex matrices but Real
# scalars (`relty = real(T)`); the @eval-generated bare-Dual frule!! and
# the width-N canonical-NDual entry (line ~2294) both already handle the
# math. Without this Lifted entry, IR-emit's Lifted-typed callsites errored.
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(BLAS.herk!),Char,Char,R,AbstractVecOrMat{T},R,AbstractMatrix{T}}}
) where {T<:BlasComplexFloat,R<:BlasRealFloat} = true
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.herk!),N},
    uplo::Mooncake.Lifted{Char},
    t::Mooncake.Lifted{Char},
    α_dα::Mooncake.Lifted{R},
    A_dA::Mooncake.Lifted{<:AbstractVecOrMat{T}},
    β_dβ::Mooncake.Lifted{R},
    C_dC::Mooncake.Lifted{<:AbstractMatrix{T}},
) where {N,T<:BlasComplexFloat,R<:BlasRealFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(uplo),
        Mooncake._unlift(t),
        Mooncake._unlift(α_dα),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(β_dβ),
        Mooncake._unlift(C_dC),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.trmm!),Vararg}}) = true
# Consolidated width-1 trmm!: covers Real and Complex.
function frule!!(
    ::Dual{typeof(BLAS.trmm!)},
    _side::Dual{Char},
    _uplo::Dual{Char},
    _ta::Dual{Char},
    _diag::Dual{Char},
    α_dα::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    A_dA::Union{_MatLikeWidth1,_MatLikeWidth1Complex},
    B_dB::Union{_MatLikeWidth1,_MatLikeWidth1Complex},
)
    α, dα = _scalar_extract(α_dα)
    A, dA = _arr_extract(A_dA)
    B, dB = _arr_extract(B_dB)
    _trmm!_frule_core!(
        primal(_side), primal(_uplo), primal(_ta), primal(_diag), α, dα, A, dA, B, dB
    )
    _arr_writeback!(B_dB, B, dB)
    return B_dB
end
# Width-N NDual trmm!: per-lane Frechet (pre-primal B) then primal once.
@inline function _trmm_frechet_lane!(side, uplo, ta, diag, α::P, dα, A, dA, B, dB) where {P}
    BLAS.trmm!(side, uplo, ta, diag, α, A, dB)
    dB .+= BLAS.trmm!(side, uplo, ta, diag, α, dA, copy(B))
    if diag == 'U'
        dB .-= α .* B
    end
    if !iszero(dα)
        dB .+= BLAS.trmm!(side, uplo, ta, diag, dα, A, copy(B))
    end
    return nothing
end
# Consolidated width-N trmm!: covers Real (NDual{P,N}) and Complex
# (Complex{NDual{R,N}}). Bodies were byte-identical; type-binding via
# matched element-type unions on scalar/matrix args.
@inline function frule!!(
    ::Dual{typeof(BLAS.trmm!)},
    _side::Dual{Char},
    _uplo::Dual{Char},
    _ta::Dual{Char},
    _diag::Dual{Char},
    α_dα::Union{NDual{P,N},Complex{NDual{P,N}}},
    A_dA::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    B_dB::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
) where {P<:IEEEFloat,N}
    side = primal(_side)
    uplo = primal(_uplo)
    ta = primal(_ta)
    diag = primal(_diag)
    α, dαs = _scalar_extract_n(α_dα)
    A, dAs = _arr_extract_n(A_dA)
    B, dBs = _arr_extract_n(B_dB)
    @inbounds for lane in 1:N
        _trmm_frechet_lane!(side, uplo, ta, diag, α, dαs[lane], A, dAs[lane], B, dBs[lane])
    end
    BLAS.trmm!(side, uplo, ta, diag, α, A, B)
    _arr_writeback_n!(B_dB, B, dBs)
    return B_dB
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.trmm!),N},
    _side::Mooncake.Lifted{Char},
    _uplo::Mooncake.Lifted{Char},
    _ta::Mooncake.Lifted{Char},
    _diag::Mooncake.Lifted{Char},
    α_dα::Mooncake.Lifted{P},
    A_dA::Mooncake.Lifted{<:AbstractMatrix{P}},
    B_dB::Mooncake.Lifted{<:AbstractMatrix{P}},
) where {N,P<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(_side),
        Mooncake._unlift(_uplo),
        Mooncake._unlift(_ta),
        Mooncake._unlift(_diag),
        Mooncake._unlift(α_dα),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(B_dB),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
@inline function _trmm!_frule_core!(
    side::Char,
    uplo::Char,
    ta::Char,
    diag::Char,
    α::P,
    dα::P,
    A::AbstractMatrix{P},
    dA::AbstractMatrix{P},
    B::AbstractMatrix{P},
    dB::AbstractMatrix{P},
) where {P<:BlasFloat}
    BLAS.trmm!(side, uplo, ta, diag, α, A, dB)
    dB .+= BLAS.trmm!(side, uplo, ta, diag, α, dA, copy(B))
    if diag == 'U'
        dB .-= α .* B
    end
    if !iszero(dα)
        dB .+= BLAS.trmm!(side, uplo, ta, diag, dα, A, copy(B))
    end
    BLAS.trmm!(side, uplo, ta, diag, α, A, B)
    return nothing
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

        # Compute α gradient.
        ∇α = dot(B, dB) / α'

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
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(BLAS.trsm!),Vararg}}) = true

# Consolidated width-1 trsm!: covers Real and Complex.
function frule!!(
    ::Dual{typeof(BLAS.trsm!)},
    _side::Dual{Char},
    _uplo::Dual{Char},
    _t::Dual{Char},
    _diag::Dual{Char},
    α_dα::Union{_ScalarLikeWidth1,_ScalarLikeWidth1Complex},
    A_dA::Union{_MatLikeWidth1,_MatLikeWidth1Complex},
    B_dB::Union{_MatLikeWidth1,_MatLikeWidth1Complex},
)
    α, dα = _scalar_extract(α_dα)
    A, dA = _arr_extract(A_dA)
    B, dB = _arr_extract(B_dB)
    _trsm!_frule_core!(
        primal(_side), primal(_uplo), primal(_t), primal(_diag), α, dα, A, dA, B, dB
    )
    _arr_writeback!(B_dB, B, dB)
    return B_dB
end
# Width-N NDual trsm!: per-lane Frechet (which uses pre-primal B for the dα
# branch via `tmp = copy(B)`), then primal once. Reusing pre-primal B is
# safe because each lane recopies B independently.
@inline function _trsm_frechet_lane!(
    side, uplo, trans, diag, α::P, dα, A, dA, B, dB
) where {P}
    BLAS.trsm!(side, uplo, trans, diag, α, A, dB)
    tmp = copy(B)
    BLAS.trsm!(side, uplo, trans, diag, one(P), A, tmp)
    dB .+= dα .* tmp
    tmp2 = copy(tmp)
    BLAS.trmm!(side, uplo, trans, diag, α, dA, tmp)
    if diag == 'U'
        tmp .-= α .* tmp2
    end
    BLAS.trsm!(side, uplo, trans, diag, one(P), A, tmp)
    dB .-= tmp
    return nothing
end
# Consolidated width-N trsm!: covers Real and Complex via element-type
# union with shared `P<:IEEEFloat` typevar.
@inline function frule!!(
    ::Dual{typeof(BLAS.trsm!)},
    _side::Dual{Char},
    _uplo::Dual{Char},
    _t::Dual{Char},
    _diag::Dual{Char},
    α_dα::Union{NDual{P,N},Complex{NDual{P,N}}},
    A_dA::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
    B_dB::AbstractMatrix{<:Union{NDual{P,N},Complex{NDual{P,N}}}},
) where {P<:IEEEFloat,N}
    side = primal(_side)
    uplo = primal(_uplo)
    trans = primal(_t)
    diag = primal(_diag)
    α, dαs = _scalar_extract_n(α_dα)
    A, dAs = _arr_extract_n(A_dA)
    B, dBs = _arr_extract_n(B_dB)
    @inbounds for lane in 1:N
        _trsm_frechet_lane!(
            side, uplo, trans, diag, α, dαs[lane], A, dAs[lane], B, dBs[lane]
        )
    end
    BLAS.trsm!(side, uplo, trans, diag, α, A, B)
    _arr_writeback_n!(B_dB, B, dBs)
    return B_dB
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(BLAS.trsm!),N},
    _side::Mooncake.Lifted{Char},
    _uplo::Mooncake.Lifted{Char},
    _t::Mooncake.Lifted{Char},
    _diag::Mooncake.Lifted{Char},
    α_dα::Mooncake.Lifted{P},
    A_dA::Mooncake.Lifted{<:AbstractMatrix{P}},
    B_dB::Mooncake.Lifted{<:AbstractMatrix{P}},
) where {N,P<:BlasFloat}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(_side),
        Mooncake._unlift(_uplo),
        Mooncake._unlift(_t),
        Mooncake._unlift(_diag),
        Mooncake._unlift(α_dα),
        Mooncake._unlift(A_dA),
        Mooncake._unlift(B_dB),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
@inline function _trsm!_frule_core!(
    side::Char,
    uplo::Char,
    trans::Char,
    diag::Char,
    α::P,
    dα::P,
    A::AbstractMatrix{P},
    dA::AbstractMatrix{P},
    B::AbstractMatrix{P},
    dB::AbstractMatrix{P},
) where {P<:BlasFloat}
    BLAS.trsm!(side, uplo, trans, diag, α, A, dB)
    tmp = copy(B)
    trsm!(side, uplo, trans, diag, one(P), A, tmp)
    dB .+= dα .* tmp
    tmp2 = copy(tmp)
    BLAS.trmm!(side, uplo, trans, diag, α, dA, tmp)
    if diag == 'U'
        tmp .-= α .* tmp2
    end
    BLAS.trsm!(side, uplo, trans, diag, one(P), A, tmp)
    dB .-= tmp
    BLAS.trsm!(side, uplo, trans, diag, α, A, B)
    return nothing
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
        # Compute α gradient.
        ∇α = dot(B, dB) / α'

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
        map_prod([1, 3, 11], [1, 2, 11]) do (n, incx)
            flags = (false, :stability, nothing)
            return (flags..., BLAS.scal!, n, randn(rng, P), randn(rng, P, n * incx), incx)
        end,

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
    return Any[], Any[]
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
