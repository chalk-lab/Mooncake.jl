module MooncakeSparseArraysExt

using LinearAlgebra, Mooncake, SelectedInversion, SparseArrays
using Random: AbstractRNG

# Import only names we extend with new methods.
import Mooncake:
    frule!!,
    rrule!!,
    tangent_type,
    fdata_type,
    rdata_type,
    fdata,
    rdata,
    tangent,
    zero_tangent_internal,
    randn_tangent_internal,
    increment_internal!!,
    set_to_zero_internal!!,
    _add_to_primal_internal,
    tangent_to_primal_internal!!,
    primal_to_tangent_internal!!,
    _dot_internal,
    _scale_internal

const CHOLMOD = SparseArrays.CHOLMOD
const Factor = CHOLMOD.Factor

# =============================================================================
# Ptr{cholmod_factor_struct} registration
#
# CHOLMOD.Factor wraps a Ptr{cholmod_factor_struct} — an opaque C handle to
# CHOLMOD-managed memory. The Ptr type must be registered as NoTangent to
# prevent Mooncake from recursing into cholmod_factor_struct's many Ptr{Nothing}
# fields (which would trigger "zero_tangent not available for pointers").
# =============================================================================

const _cholmod_factor_struct = SparseArrays.LibSuiteSparse.cholmod_factor_struct
tangent_type(::Type{Ptr{_cholmod_factor_struct}}) = Mooncake.NoTangent
function zero_tangent_internal(::Ptr{_cholmod_factor_struct}, ::Mooncake.MaybeCache)
    Mooncake.NoTangent()
end

# =============================================================================
# CholmodFactorTangent: tangent type for CHOLMOD.Factor
#
# Stores the derivative w.r.t. the *original matrix A* (not the Cholesky factor L).
# This representation avoids the expensive Cholesky adjoint formula entirely:
#   - logdet pullback uses selected inversion → O(nnz(L))
#   - \ pullback uses IFT: d̄A = -(F\d̄x)*x' → O(nnz(A))
#   - cholesky pullback trivially passes d̄A upstream
# =============================================================================

mutable struct CholmodFactorTangent{T,Ti<:Integer}
    dA::SparseMatrixCSC{T,Ti}  # same sparsity pattern as original A
end
function CholmodFactorTangent(dA::SparseMatrixCSC{T,Ti}) where {T,Ti}
    return CholmodFactorTangent{T,Ti}(dA)
end

# --- tangent_type ---

tangent_type(::Type{<:Factor{T,Ti}}) where {T,Ti} = CholmodFactorTangent{T,Ti}

# fdata/rdata reconstruction: CholmodFactorTangent is mutable, so it IS the fdata.
# Same pattern as MutableTangent (see src/tangents/fwds_rvs_data.jl:944).
Mooncake.@foldable tangent_type(
    ::Type{F}, ::Type{Mooncake.NoRData}
) where {F<:CholmodFactorTangent} = F

# --- Helpers ---

function _zero_sparse_like(A::SparseMatrixCSC{T}) where {T}
    # Structural arrays (colptr, rowval) are shared — only nzval is ever mutated.
    return SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, zeros(T, nnz(A)))
end

function _reconstruct_A(F::Factor)
    return sparse(F)
end

# --- Core tangent interface ---

function zero_tangent_internal(F::Factor{T}, d::Mooncake.MaybeCache) where {T}
    A = _reconstruct_A(F)
    return CholmodFactorTangent(_zero_sparse_like(A))
end

function randn_tangent_internal(
    rng::AbstractRNG, F::Factor{T}, d::Mooncake.MaybeCache
) where {T}
    A = _reconstruct_A(F)
    dA = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, randn(rng, T, nnz(A)))
    return CholmodFactorTangent(dA)
end

function increment_internal!!(
    ::Mooncake.IncCache, x::CholmodFactorTangent{T,Ti}, y::CholmodFactorTangent{T,Ti}
) where {T,Ti}
    x.dA.nzval .+= y.dA.nzval
    return x
end

function set_to_zero_internal!!(::Mooncake.SetToZeroCache, x::CholmodFactorTangent)
    x.dA.nzval .= zero(eltype(x.dA))
    return x
end

function _scale_internal(
    ::Mooncake.MaybeCache, a::Float64, t::CholmodFactorTangent{T}
) where {T}
    dA_new = SparseMatrixCSC(t.dA.m, t.dA.n, t.dA.colptr, t.dA.rowval, T(a) .* t.dA.nzval)
    return CholmodFactorTangent(dA_new)
end

function _dot_internal(
    ::Mooncake.MaybeCache, t::CholmodFactorTangent{T,Ti}, s::CholmodFactorTangent{T,Ti}
) where {T,Ti}
    return Float64(dot(t.dA.nzval, s.dA.nzval))
end

function _add_to_primal_internal(
    ::Mooncake.MaybeCache, F::Factor{T}, t::CholmodFactorTangent, ::Bool
) where {T}
    A = _reconstruct_A(F)
    A_perturbed = A + t.dA
    return cholesky(A_perturbed)
end

function tangent_to_primal_internal!!(
    F::Factor{T}, t::CholmodFactorTangent, ::Mooncake.MaybeCache
) where {T}
    return cholesky(t.dA)
end

function primal_to_tangent_internal!!(
    t::CholmodFactorTangent, F::Factor{T}, ::Mooncake.MaybeCache
) where {T}
    A = _reconstruct_A(F)
    t.dA.nzval .= A.nzval
    return t
end

# --- FData / RData ---

fdata_type(::Type{CholmodFactorTangent{T,Ti}}) where {T,Ti} = CholmodFactorTangent{T,Ti}
rdata_type(::Type{<:CholmodFactorTangent}) = Mooncake.NoRData

fdata(t::CholmodFactorTangent) = t
rdata(::CholmodFactorTangent) = Mooncake.NoRData()
tangent(f::CholmodFactorTangent, ::Mooncake.NoRData) = f

# =============================================================================
# rrule helpers
# =============================================================================

function _accumulate_selinv!(dA::SparseMatrixCSC{T}, Z, scale::T) where {T}
    # Convert Z to SparseMatrixCSC to avoid per-element indexing overhead
    # (Z may be a SupernodalMatrix with expensive scalar getindex).
    Z_sparse = Z isa SparseMatrixCSC ? Z : SparseMatrixCSC(Z)
    rows = rowvals(dA)
    vals = nonzeros(dA)
    for j in 1:size(dA, 2)
        for k in nzrange(dA, j)
            i = rows[k]
            vals[k] += scale * Z_sparse[i, j]
        end
    end
end

function _accumulate_into_triangle!(
    dst_nzval::Vector{T}, dst::SparseMatrixCSC, src::SparseMatrixCSC{T}, uplo::Char
) where {T}
    src_rows = rowvals(src)
    src_vals = nonzeros(src)
    for j in 1:size(src, 2)
        for k in nzrange(src, j)
            i = src_rows[k]
            if uplo == 'L'
                r, c = max(i, j), min(i, j)
            else
                r, c = min(i, j), max(i, j)
            end
            for k2 in nzrange(dst, c)
                if rowvals(dst)[k2] == r
                    dst_nzval[k2] += src_vals[k]
                    break
                end
            end
        end
    end
end

function _tangent_from_symmetric_input(
    nzval_src::Vector{T}, S_tri::SparseMatrixCSC, A_full::SparseMatrixCSC
) where {T}
    dA = _zero_sparse_like(A_full)
    rows_full = rowvals(dA)
    vals_full = nonzeros(dA)
    rows_tri = rowvals(S_tri)
    for j in 1:size(S_tri, 2)
        for k in nzrange(S_tri, j)
            i = rows_tri[k]
            v = nzval_src[k]
            for k2 in nzrange(dA, j)
                if rows_full[k2] == i
                    vals_full[k2] = v
                    break
                end
            end
            if i != j
                for k2 in nzrange(dA, i)
                    if rows_full[k2] == j
                        vals_full[k2] = v
                        break
                    end
                end
            end
        end
    end
    return CholmodFactorTangent(dA)
end

function _accumulate_rank1!(
    dA::SparseMatrixCSC{T}, f::AbstractVector{T}, x::AbstractVector{T}, scale::T
) where {T}
    rows = rowvals(dA)
    vals = nonzeros(dA)
    for j in 1:size(dA, 2)
        for k in nzrange(dA, j)
            i = rows[k]
            vals[k] += scale * f[i] * x[j]
        end
    end
end

# =============================================================================
# rrules
# =============================================================================

# --- cholesky ---

for _Ti in (Int32, Int64)
    @eval Mooncake.@is_primitive(
        Mooncake.MinimalCtx, Tuple{typeof(cholesky),SparseMatrixCSC{T,$_Ti}} where {T<:Real}
    )
    @eval Mooncake.@is_primitive(
        Mooncake.MinimalCtx,
        Tuple{typeof(cholesky),Symmetric{T,SparseMatrixCSC{T,$_Ti}}} where {T<:Real},
    )
    @eval Mooncake.@is_primitive(
        Mooncake.MinimalCtx,
        Tuple{
            typeof(Core.kwcall),NamedTuple,typeof(cholesky),SparseMatrixCSC{T,$_Ti}
        } where {T<:Real},
    )
    @eval Mooncake.@is_primitive(
        Mooncake.MinimalCtx,
        Tuple{
            typeof(Core.kwcall),
            NamedTuple,
            typeof(cholesky),
            Symmetric{T,SparseMatrixCSC{T,$_Ti}},
        } where {T<:Real},
    )
end

function rrule!!(
    ::Mooncake.CoDual{typeof(cholesky)}, _A::Mooncake.CoDual{<:SparseMatrixCSC{T}}
) where {T<:Real}
    A = Mooncake.primal(_A)
    dA_fdata = tangent(_A)
    F = cholesky(A)
    dF = zero_tangent_internal(F, Mooncake.NoCache())

    function cholesky_pb!!(::Mooncake.NoRData)
        nzval_tangent = Mooncake._fields(dA_fdata).nzval
        nzval_tangent .+= dF.dA.nzval
        return Mooncake.NoRData(), Mooncake.NoRData()
    end

    return Mooncake.CoDual(F, dF), cholesky_pb!!
end

function rrule!!(
    ::Mooncake.CoDual{typeof(cholesky)},
    _A::Mooncake.CoDual{<:Symmetric{T,<:SparseMatrixCSC{T}}},
) where {T<:Real}
    A_sym = Mooncake.primal(_A)
    A_sparse = parent(A_sym)
    dA_tangent = tangent(_A)
    F = cholesky(A_sym)
    dF = zero_tangent_internal(F, Mooncake.NoCache())
    uplo = A_sym.uplo

    function cholesky_sym_pb!!(::Mooncake.NoRData)
        data_fdata = Mooncake._fields(dA_tangent).data
        nzval_tangent = Mooncake._fields(data_fdata).nzval
        if length(nzval_tangent) == length(dF.dA.nzval)
            nzval_tangent .+= dF.dA.nzval
        else
            _accumulate_into_triangle!(nzval_tangent, A_sparse, dF.dA, uplo)
        end
        return Mooncake.NoRData(), Mooncake.NoRData()
    end

    return Mooncake.CoDual(F, dF), cholesky_sym_pb!!
end

# kwcall variants delegate to the non-kwcall rules.

function rrule!!(
    ::Mooncake.CoDual{typeof(Core.kwcall)},
    _kw::Mooncake.CoDual{<:NamedTuple},
    ::Mooncake.CoDual{typeof(cholesky)},
    _A::Mooncake.CoDual{<:SparseMatrixCSC{T}},
) where {T<:Real}
    result, pb = rrule!!(Mooncake.CoDual(cholesky, Mooncake.NoFData()), _A)
    function cholesky_kw_pb!!(::Mooncake.NoRData)
        _, dA_r = pb(Mooncake.NoRData())
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), dA_r
    end
    return result, cholesky_kw_pb!!
end

function rrule!!(
    ::Mooncake.CoDual{typeof(Core.kwcall)},
    _kw::Mooncake.CoDual{<:NamedTuple},
    ::Mooncake.CoDual{typeof(cholesky)},
    _A::Mooncake.CoDual{<:Symmetric{T,<:SparseMatrixCSC{T}}},
) where {T<:Real}
    result, pb = rrule!!(Mooncake.CoDual(cholesky, Mooncake.NoFData()), _A)
    function cholesky_sym_kw_pb!!(::Mooncake.NoRData)
        _, dA_r = pb(Mooncake.NoRData())
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), dA_r
    end
    return result, cholesky_sym_kw_pb!!
end

function frule!!(
    ::Mooncake.Dual{typeof(Core.kwcall)},
    _kw::Mooncake.Dual{<:NamedTuple},
    ::Mooncake.Dual{typeof(cholesky)},
    _A::Mooncake.Dual{<:SparseMatrixCSC{T}},
) where {T<:Real}
    return frule!!(Mooncake.Dual(cholesky, Mooncake.NoTangent()), _A)
end

function frule!!(
    ::Mooncake.Dual{typeof(Core.kwcall)},
    _kw::Mooncake.Dual{<:NamedTuple},
    ::Mooncake.Dual{typeof(cholesky)},
    _A::Mooncake.Dual{<:Symmetric{T,<:SparseMatrixCSC{T}}},
) where {T<:Real}
    return frule!!(Mooncake.Dual(cholesky, Mooncake.NoTangent()), _A)
end

# --- logdet ---

for _Ti in (Int32, Int64)
    @eval Mooncake.@is_primitive(
        Mooncake.MinimalCtx, Tuple{typeof(logdet),Factor{T,$_Ti}} where {T<:Real}
    )
end

function rrule!!(
    ::Mooncake.CoDual{typeof(logdet)}, _F::Mooncake.CoDual{<:Factor{T}}
) where {T<:Real}
    F = Mooncake.primal(_F)
    dF = tangent(_F)
    val = logdet(F)

    function logdet_pb!!(d̄val::T)
        result = selinv(F; depermute=true)
        _accumulate_selinv!(dF.dA, result.Z, d̄val)
        return Mooncake.NoRData(), Mooncake.NoRData()
    end

    return Mooncake.CoDual(val, Mooncake.NoFData()), logdet_pb!!
end

# --- \ (vector RHS) ---

for _Ti in (Int32, Int64)
    @eval Mooncake.@is_primitive(
        Mooncake.MinimalCtx,
        Tuple{typeof(\),Factor{T,$_Ti},StridedVector{T}} where {T<:Real},
    )
end

function rrule!!(
    ::Mooncake.CoDual{typeof(\)},
    _F::Mooncake.CoDual{<:Factor{T}},
    _b::Mooncake.CoDual{<:StridedVector{T}},
) where {T<:Real}
    F = Mooncake.primal(_F)
    dF = tangent(_F)
    b = Mooncake.primal(_b)
    x = F \ b
    dx = zero(x)
    db = tangent(_b)

    function solve_pb!!(::Mooncake.NoRData)
        adj = F \ dx
        _accumulate_rank1!(dF.dA, adj, x, T(-1))
        db .+= adj
        dx .= zero(T)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end

    return Mooncake.CoDual(x, dx), solve_pb!!
end

# --- \ (matrix RHS) ---

for _Ti in (Int32, Int64)
    @eval Mooncake.@is_primitive(
        Mooncake.MinimalCtx,
        Tuple{typeof(\),Factor{T,$_Ti},StridedMatrix{T}} where {T<:Real},
    )
end

function rrule!!(
    ::Mooncake.CoDual{typeof(\)},
    _F::Mooncake.CoDual{<:Factor{T}},
    _B::Mooncake.CoDual{<:StridedMatrix{T}},
) where {T<:Real}
    F = Mooncake.primal(_F)
    dF = tangent(_F)
    B = Mooncake.primal(_B)
    X = F \ B
    dX = zero(X)
    dB = tangent(_B)

    function solve_mat_pb!!(::Mooncake.NoRData)
        Adj = F \ dX
        for j in 1:size(X, 2)
            _accumulate_rank1!(dF.dA, view(Adj, :, j), view(X, :, j), T(-1))
        end
        dB .+= Adj
        dX .= zero(T)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end

    return Mooncake.CoDual(X, dX), solve_mat_pb!!
end

# =============================================================================
# frules
# =============================================================================

function frule!!(
    ::Mooncake.Dual{typeof(cholesky)}, _A::Mooncake.Dual{<:SparseMatrixCSC{T}}
) where {T<:Real}
    A = Mooncake.primal(_A)
    dA_tangent = tangent(_A)
    F = cholesky(A)

    # Use A directly for sparsity pattern — avoids expensive sparse(F) reconstruction.
    dA_nzval = Mooncake._fields(dA_tangent).nzval
    dF_sparse = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, copy(dA_nzval))
    return Mooncake.Dual(F, CholmodFactorTangent(dF_sparse))
end

function frule!!(
    ::Mooncake.Dual{typeof(cholesky)},
    _A::Mooncake.Dual{<:Symmetric{T,<:SparseMatrixCSC{T}}},
) where {T<:Real}
    A_sym = Mooncake.primal(_A)
    S_tri = parent(A_sym)
    dA_tangent = tangent(_A)
    F = cholesky(A_sym)

    data_tangent = Mooncake._fields(dA_tangent).data
    dA_nzval = Mooncake._fields(data_tangent).nzval

    # Use sparse(F) only when the input triangle pattern differs from the full pattern.
    A_recon = _reconstruct_A(F)
    if length(dA_nzval) == nnz(A_recon)
        dF_sparse = SparseMatrixCSC(
            A_recon.m, A_recon.n, A_recon.colptr, A_recon.rowval, copy(dA_nzval)
        )
        dF = CholmodFactorTangent(dF_sparse)
    else
        dF = _tangent_from_symmetric_input(dA_nzval, S_tri, A_recon)
    end

    return Mooncake.Dual(F, dF)
end

function frule!!(
    ::Mooncake.Dual{typeof(logdet)}, _F::Mooncake.Dual{<:Factor{T}}
) where {T<:Real}
    F = Mooncake.primal(_F)
    dF = tangent(_F)
    val = logdet(F)
    Z = selinv(F; depermute=true)
    return Mooncake.Dual(val, T(dot(Z.Z, dF.dA)))
end

function frule!!(
    ::Mooncake.Dual{typeof(\)},
    _F::Mooncake.Dual{<:Factor{T}},
    _b::Mooncake.Dual{<:StridedVecOrMat{T}},
) where {T<:Real}
    F = Mooncake.primal(_F)
    dF = tangent(_F)
    b = Mooncake.primal(_b)
    db = tangent(_b)
    x = F \ b
    return Mooncake.Dual(x, F \ (db - dF.dA * x))
end

end # module MooncakeSparseArraysExt
