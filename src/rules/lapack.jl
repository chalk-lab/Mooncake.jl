# See https://sethaxen.com/blog/2021/02/differentiating-the-lu-decomposition/ for details.
@is_primitive(MinimalCtx, Tuple{typeof(LAPACK.getrf!),AbstractMatrix{<:BlasFloat}})
function frule!!(
    ::Dual{typeof(LAPACK.getrf!)}, A_dA::Dual{<:AbstractMatrix{P}}
) where {P<:BlasFloat}
    _, ipiv, info = LAPACK.getrf!(primal(A_dA))
    return _getrf_fwd(A_dA, ipiv, info)
end
function rrule!!(
    ::CoDual{typeof(LAPACK.getrf!)}, _A::CoDual{<:AbstractMatrix{P}}
) where {P<:BlasFloat}
    A, dA = arrayify(_A)
    A_copy = copy(A)

    # Run the primal.
    _, ipiv, code = LAPACK.getrf!(A)

    # Zero out the tangent.
    dA .= zero(P)

    function getrf_pb!!(::NoRData)
        _getrf_pb!(A, dA, ipiv, A_copy)
        return NoRData(), NoRData()
    end
    dipiv = zero_tangent(ipiv)
    return CoDual((_A.x, ipiv, code), (_A.dx, dipiv, NoFData())), getrf_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{typeof(Core.kwcall),NamedTuple,typeof(LAPACK.getrf!),AbstractMatrix{<:BlasFloat}},
)
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    _kwargs::Dual{<:NamedTuple},
    ::Dual{typeof(getrf!)},
    A_dA::Dual{<:AbstractMatrix{P}},
) where {P<:BlasFloat}
    check = primal(_kwargs).check
    _, ipiv, info = LAPACK.getrf!(primal(A_dA); check)
    return _getrf_fwd(A_dA, ipiv, info)
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    _kwargs::CoDual{<:NamedTuple},
    ::CoDual{typeof(getrf!)},
    _A::CoDual{<:AbstractMatrix{P}},
) where {P<:BlasFloat}
    check = _kwargs.x.check
    A, dA = arrayify(_A)
    A_copy = copy(A)

    # Run the primal.
    _, ipiv, code = LAPACK.getrf!(A; check)

    # Zero out the tangent.
    dA .= zero(P)

    function getrf_pb!!(::NoRData)
        _getrf_pb!(A, dA, ipiv, A_copy)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    dipiv = zero_tangent(ipiv)
    return CoDual((_A.x, ipiv, code), (_A.dx, dipiv, NoFData())), getrf_pb!!
end

function _getrf_fwd(A_dA, ipiv, info)
    A, dA = arrayify(A_dA)

    # Compute Frechet derivative.
    L = UnitLowerTriangular(A)
    U = UpperTriangular(A)
    p = LinearAlgebra.ipiv2perm(ipiv, size(A, 2))
    F = rdiv!(ldiv!(L, dA[p, :]), U)
    dA .= L * tril(F, -1) + triu(F) * U

    return Dual((A, ipiv, info), (tangent(A_dA), zero_tangent(ipiv), NoTangent()))
end

function _getrf_pb!(A, dA, ipiv, A_copy)

    # Run reverse-pass.
    L = UnitLowerTriangular(A)
    U = UpperTriangular(A)
    dL = tril(dA, -1)
    dU = UpperTriangular(dA)

    # Figure out the pivot matrix used.
    p = LinearAlgebra.ipiv2perm(ipiv, size(A, 2))

    # Compute pullback using Seth's method.
    _dF = tril(L'dL, -1) + UpperTriangular(dU * U')
    dA .= (inv(L') * _dF * inv(U'))[invperm(p), :]

    # Restore initial state.
    A .= A_copy

    return nothing
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(trtrs!),Char,Char,Char,AbstractMatrix{P},AbstractVecOrMat{P}
    } where {P<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(trtrs!)},
    _uplo::Dual{Char},
    _trans::Dual{Char},
    _diag::Dual{Char},
    A_dA::Dual{<:AbstractMatrix{P}},
    B_dB::Dual{<:AbstractVecOrMat{P}},
) where {P<:BlasRealFloat}

    # Extract data.
    uplo = primal(_uplo)
    trans = primal(_trans)
    diag = primal(_diag)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)

    # Compute Frechet derivative.
    LAPACK.trtrs!(uplo, trans, diag, A, dB)
    tmp = copy(B)
    LAPACK.trtrs!(uplo, trans, diag, A, tmp) # tmp now contains inv(A) B.

    tmp2 = copy(tmp)
    if diag == 'N'
        a = uplo == 'L' ? LowerTriangular(dA) : UpperTriangular(dA)
        lmul!(trans == 'N' ? a : a', tmp)
    else
        a = uplo == 'L' ? UnitLowerTriangular(dA) : UnitUpperTriangular(dA)
        lmul!(trans == 'N' ? a : a', tmp)
        tmp .-= tmp2
    end
    LAPACK.trtrs!(uplo, trans, diag, A, tmp) # tmp is now α inv(A) dA inv(A) B.
    dB .-= tmp

    # Run primal computation.
    LAPACK.trtrs!(uplo, trans, diag, A, B)
    return B_dB
end
function rrule!!(
    ::CoDual{typeof(trtrs!)},
    _uplo::CoDual{Char},
    _trans::CoDual{Char},
    _diag::CoDual{Char},
    _A::CoDual{<:AbstractMatrix{P}},
    _B::CoDual{<:AbstractVecOrMat{P}},
) where {P<:BlasRealFloat}
    # Extract everything and make a copy of B for the reverse-pass.
    uplo, trans, diag = primal(_uplo), primal(_trans), primal(_diag)
    A, dA = arrayify(_A)
    B, dB = arrayify(_B)
    B_copy = copy(B)

    # Run primal.
    trtrs!(uplo, trans, diag, A, B)

    function trtrs_pb!!(::NoRData)

        # Compute cotangent of B.
        LAPACK.trtrs!(uplo, trans == 'N' ? 'T' : 'N', diag, A, dB)

        # Compute cotangent of A.
        if trans == 'N'
            dA .-= tri!(dB * B', uplo, diag)
        else
            dA .-= tri!(B * dB', uplo, diag)
        end

        # Restore initial state.
        B .= B_copy

        return tuple_fill(NoRData(), Val(6))
    end
    return _B, trtrs_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(getrs!),Char,AbstractMatrix{P},AbstractVector{Int},AbstractVecOrMat{P}
    } where {P<:BlasRealFloat}
)
function frule!!(
    ::Dual{typeof(getrs!)},
    _trans::Dual{Char},
    A_dA::Dual{<:AbstractMatrix{P}},
    _ipiv::Dual{<:AbstractVector{Int}},
    B_dB::Dual{<:AbstractVecOrMat{P}},
) where {P<:BlasRealFloat}

    # Extract data.
    trans = primal(_trans)
    A, dA = arrayify(A_dA)
    ipiv = primal(_ipiv)
    B, dB = arrayify(B_dB)

    # Run primal computation.
    LAPACK.getrs!(trans, A, ipiv, B)

    # Compute Frechet derivative.
    L = UnitLowerTriangular(A)
    dL_plus_I = UnitLowerTriangular(dA)
    U = UpperTriangular(A)
    dU = UpperTriangular(dA)
    p = LinearAlgebra.ipiv2perm(ipiv, size(dB, 1))
    tmp = dL_plus_I * U
    tmp .-= U
    tmp2 = mul!(tmp, L, dU, one(P), one(P))[invperm(p), :]
    if trans == 'N'
        mul!(dB, tmp2, B, -one(P), one(P))
    else
        mul!(dB, tmp2', B, -one(P), one(P))
    end
    LAPACK.getrs!(trans, A, ipiv, dB)

    return B_dB
end
function rrule!!(
    ::CoDual{typeof(getrs!)},
    _trans::CoDual{Char},
    _A::CoDual{<:AbstractMatrix{P}},
    _ipiv::CoDual{<:AbstractVector{Int}},
    _B::CoDual{<:AbstractVecOrMat{P}},
) where {P<:BlasRealFloat}

    # Extract data.
    trans = _trans.x
    A, dA = arrayify(_A)
    ipiv = _ipiv.x
    B, dB = arrayify(_B)
    B0 = copy(B)

    # Pivot B.
    p = LinearAlgebra.ipiv2perm(ipiv, size(A, 1))
    ip = invperm(p)

    # Pre-allocate B1 with concrete type before the if/else to avoid Core.Box in the
    # pullback closure. B2 is always just an alias for B, so we use B directly below.
    B1 = similar(B)

    if trans == 'N'
        # Apply permutation matrix.
        B .= B[p, :]

        # Run inv(L) * B and write result to B.
        LAPACK.trtrs!('L', 'N', 'U', A, B)
        copyto!(B1, B) # record intermediate state for use in pullback.

        # Run inv(U) * B and write result to B.
        LAPACK.trtrs!('U', 'N', 'N', A, B)
    else
        # Run inv(U)^T * B and write result to B.
        LAPACK.trtrs!('U', 'T', 'N', A, B)
        copyto!(B1, B) # record intermediate state for use in pullback.

        # Run inv(L)^T * B and write result to B.
        LAPACK.trtrs!('L', 'T', 'U', A, B)

        # Apply permutation matrix.
        B .= B[ip, :]
    end

    function getrs_pb!!(::NoRData)
        if trans == 'N'

            # Run pullback for inv(U) * B.
            LAPACK.trtrs!('U', 'T', 'N', A, dB)
            dA .-= tri!(dB * B', 'U', 'N')

            # Run pullback for inv(L) * B.
            LAPACK.trtrs!('L', 'T', 'U', A, dB)
            dA .-= tri!(dB * B1', 'L', 'U')

            # Undo permutation.
            dB .= dB[ip, :]
        else

            # Undo permutation.
            dB .= dB[p, :]
            B .= B[p, :]

            # Run pullback for inv(L^T) * B.
            LAPACK.trtrs!('L', 'N', 'U', A, dB)
            dA .-= tri!(B * dB', 'L', 'U')

            # Run pullback for inv(U^T) * B.
            LAPACK.trtrs!('U', 'N', 'N', A, dB)
            dA .-= tri!(B1 * dB', 'U', 'N')
        end

        # Restore initial state.
        B .= B0
        return tuple_fill(NoRData(), Val(5))
    end
    return _B, getrs_pb!!
end

@is_primitive(
    MinimalCtx, Tuple{typeof(getri!),AbstractMatrix{<:BlasRealFloat},AbstractVector{Int}},
)
function frule!!(
    ::Dual{typeof(getri!)},
    A_dA::Dual{<:AbstractMatrix{P}},
    _ipiv::Dual{<:AbstractVector{Int}},
) where {P<:BlasRealFloat}
    # Extract args.
    A, dA = arrayify(A_dA)
    ipiv = primal(_ipiv)

    # Compute part of Frechet derivative.
    L = UnitLowerTriangular(A)
    dL_plus_I = UnitLowerTriangular(dA)
    U = UpperTriangular(A)
    dU = UpperTriangular(dA)
    p = LinearAlgebra.ipiv2perm(ipiv, size(dA, 1))
    tmp = dL_plus_I * U
    tmp .-= U
    tmp2 = mul!(tmp, L, dU, one(P), one(P))[invperm(p), :]

    # Perform primal computation.
    LAPACK.getri!(A, ipiv)

    # Compute Frechet derivative.
    dA .= (-A * tmp2 * A)

    return A_dA
end
function rrule!!(
    ::CoDual{typeof(getri!)},
    _A::CoDual{<:AbstractMatrix{<:BlasRealFloat}},
    _ipiv::CoDual{<:AbstractVector{Int}},
)
    # Extract args and copy A for reverse-pass.
    A, dA = arrayify(_A)
    ipiv = _ipiv.x
    A_copy = copy(A)

    # Run primal.
    getri!(A, ipiv)
    p = LinearAlgebra.ipiv2perm(ipiv, size(A, 1))

    function getri_pb!!(::NoRData)
        # Pivot.
        A .= A[:, p]
        dA .= dA[:, p]

        # Cotangent w.r.t. L.
        dL = -(A' * dA) / UnitLowerTriangular(A_copy)'
        dU = -(UpperTriangular(A_copy)' \ (dA * A'))
        dA .= tri!(dL, 'L', 'U') .+ tri!(dU, 'U', 'N')

        # Restore initial state.
        A .= A_copy
        return NoRData(), NoRData(), NoRData()
    end
    return _A, getri_pb!!
end

function __sym!(X::Matrix)
    X .= (X .+ X') ./ 2
    return X
end

@is_primitive(MinimalCtx, Tuple{typeof(potrf!),Char,AbstractMatrix{<:BlasRealFloat}})
function frule!!(
    ::Dual{typeof(potrf!)}, _uplo::Dual{Char}, A_dA::Dual{<:AbstractMatrix{<:BlasRealFloat}}
)
    # Extract args and take a copy of A.
    uplo = primal(_uplo)
    A, dA = arrayify(A_dA)

    # Run primal computation.
    _, info = LAPACK.potrf!(uplo, A)

    # Compute Frechet derivative.
    if uplo == 'L'
        L = LowerTriangular(A)
        tmp = LowerTriangular(ldiv!(L, Symmetric(dA, :L) / L'))
        @inbounds for n in 1:size(A, 1)
            tmp[n, n] = tmp[n, n] / 2
        end
        _copytrito!(dA, lmul!(L, tmp), 'L')
    else
        U = UpperTriangular(A)
        tmp = UpperTriangular(rdiv!(U' \ Symmetric(dA, :U), U))
        @inbounds for n in 1:size(A, 1)
            tmp[n, n] = tmp[n, n] / 2
        end
        _copytrito!(dA, rmul!(tmp, U), 'U')
    end

    return Dual((A, info), (tangent(A_dA), NoTangent()))
end
function rrule!!(
    ::CoDual{typeof(potrf!)}, _uplo::CoDual{Char}, _A::CoDual{<:AbstractMatrix{P}}
) where {P<:BlasRealFloat}

    # Extract args and take a copy of A.
    uplo = _uplo.x
    A, dA = arrayify(_A)
    A_copy = copy(A)

    # Run primal.
    _, info = potrf!(uplo, A)

    function potrf_pb!!(::NoRData)
        dA2 = dA

        # Compute cotangents.
        N = size(A, 1)
        if Char(uplo) == 'L'
            E = LowerTriangular(__E(P, N))
            L = LowerTriangular(A)
            tmp = dA2'L
            tmp .*= E'
            B = rdiv!(ldiv!(L', tmp), L)
            dA .= __sym_lower!(B) .* E ./ 2 .+ triu!(dA2, 1)
        else
            E = UpperTriangular(__E(P, N))
            U = UpperTriangular(A)
            tmp = U * dA2'
            tmp .*= E'
            B = rdiv!(ldiv!(U, tmp), U')
            dA .= __sym_upper!(B) .* E ./ 2 .+ tril!(dA2, -1)
        end

        # Restore initial state.
        A .= A_copy

        return NoRData(), NoRData(), NoRData()
    end
    return CoDual((_A.x, info), (_A.dx, NoFData())), potrf_pb!!
end

function __sym_lower!(X::Matrix)
    @inbounds for q in 1:size(X, 2), p in (q + 1):size(X, 1)
        X[p, q] = (X[p, q] + X[q, p]) / 2
    end
    return X
end

function __sym_upper!(X::Matrix)
    @inbounds for q in 1:size(X, 2), p in 1:(q - 1)
        X[p, q] = (X[p, q] + X[q, p]) / 2
    end
    return X
end

@inline function __E(P::Type, N::Int)
    E = fill(P(2), (N, N))
    for n in diagind(E)
        E[n] -= P(1)
    end
    return E
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(potrs!),Char,AbstractMatrix{P},AbstractVecOrMat{P}
    } where {P<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(potrs!)},
    _uplo::Dual{Char},
    A_dA::Dual{<:AbstractMatrix{P}},
    B_dB::Dual{<:AbstractVecOrMat{P}},
) where {P<:BlasRealFloat}

    # Extract args and take a copy of B.
    uplo = primal(_uplo)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)

    # Run primal computation.
    LAPACK.potrs!(uplo, A, B)

    # Compute Frechet derivative.
    if uplo == 'L'
        L = LowerTriangular(A)
        dL = LowerTriangular(dA)
        mul!(dB, Symmetric(dL * L' + L * dL'), B, -one(P), one(P))
        LAPACK.potrs!(uplo, A, dB)
    else
        U = UpperTriangular(A)
        dU = UpperTriangular(dA)
        mul!(dB, Symmetric(U'dU + dU'U), B, -one(P), one(P))
        LAPACK.potrs!(uplo, A, dB)
    end

    return B_dB
end
function rrule!!(
    ::CoDual{typeof(potrs!)},
    _uplo::CoDual{Char},
    _A::CoDual{<:AbstractMatrix{P}},
    _B::CoDual{<:AbstractVecOrMat{P}},
) where {P<:BlasRealFloat}

    # Extract args and take a copy of B.
    uplo = _uplo.x
    A, dA = arrayify(_A)
    B, dB = arrayify(_B)
    B_copy = copy(B)

    # Run the primal.
    potrs!(uplo, A, B)

    function potrs_pb!!(::NoRData)

        # Compute cotangents.
        if uplo == 'L'
            tmp = __sym!(B_copy * dB') / LowerTriangular(A)'
            dA .-= 2 .* tril!(LinearAlgebra.LAPACK.potrs!('L', A, tmp))
            LinearAlgebra.LAPACK.potrs!('L', A, dB)
        else
            tmp = UpperTriangular(A)' \ __sym!(B_copy * dB')
            dA .-= 2 .* triu!((tmp / UpperTriangular(A)) / UpperTriangular(A)')
            LinearAlgebra.LAPACK.potrs!('U', A, dB)
        end

        # Restore initial state.
        B .= B_copy

        return tuple_fill(NoRData(), Val(4))
    end
    return _B, potrs_pb!!
end

@static if VERSION > v"1.11-"
    @is_primitive(
        MinimalCtx,
        Tuple{
            typeof(LAPACK.lacpy!),AbstractMatrix{P},AbstractMatrix{P},Char
        } where {P<:BlasFloat},
    )
    function frule!!(
        ::Dual{typeof(LAPACK.lacpy!)},
        B_dB::Dual{<:AbstractMatrix{P}},
        A_dA::Dual{<:AbstractMatrix{P}},
        _uplo::Dual{Char},
    ) where {P<:BlasFloat}
        B, dB = arrayify(B_dB)
        A, dA = arrayify(A_dA)

        LAPACK.lacpy!(B, A, primal(_uplo))
        LAPACK.lacpy!(dB, dA, primal(_uplo))
        return B_dB
    end
    function rrule!!(
        ::CoDual{typeof(LAPACK.lacpy!)},
        B_dB::CoDual{<:AbstractMatrix{P}},
        A_dA::CoDual{<:AbstractMatrix{P}},
        _uplo::CoDual{Char},
    ) where {P<:BlasFloat}
        B, dB = arrayify(B_dB)
        A, dA = arrayify(A_dA)
        uplo = primal(_uplo)

        B_copy = copy(B)
        LAPACK.lacpy!(B, A, uplo)
        # fill dB with zeros in the copied region
        zero_tri!(dB, uplo)

        function lacpy_pb!!(::NoRData)
            if uplo == 'U'
                dA .+= UpperTriangular(dB)
            elseif uplo == 'L'
                dA .+= LowerTriangular(dB)
            else
                dA .+= dB
            end
            zero_tri!(dB, uplo)

            # undo the primal change
            LAPACK.lacpy!(B, B_copy, uplo)

            return NoRData(), NoRData(), NoRData(), NoRData()
        end
        return B_dB, lacpy_pb!!
    end
end

# sytrf! (Bunch-Kaufman factorization)
#
# Issue: https://github.com/chalk-lab/Mooncake.jl/issues/819
# `logdet(Symmetric(A))` fails because it calls `LAPACK.sytrf!` (Bunch-Kaufman factorization),
# which has no AD rule.
#
# All of the following user-facing calls hit sytrf! for BlasFloat symmetric matrices:
#
#   Use case                         | Call path                              | Rule added?
#   ---------------------------------|----------------------------------------|------------
#   logdet(Symmetric(A))             | → _factorize → bunchkaufman → sytrf!   | yes
#   det(Symmetric(A))                | same                                   | yes
#   logabsdet(Symmetric(A))          | same                                   | yes
#   inv(Symmetric(A))                | → bunchkaufman → sytrf!, then sytri!   | no
#   Symmetric(A) \ b                 | → bunchkaufman → sytrf!, then sytrs!   | no
#   factorize(Symmetric(A))          | → bunchkaufman → sytrf!                | no
#   bunchkaufman(Symmetric(A))       | → sytrf! directly                      | no
#   isposdef(Symmetric(A))           | → _factorize → bunchkaufman → sytrf!   | no
#
# Possible fix strategies (in order of increasing complexity):
#
#   1. Direct rules for logdet / logabsdet / det on Symmetric:  ← DONE (below)
#      d logdet(Sym(A)) / dSym(A) = Sym(A)⁻¹ (off-diagonal stored entries scaled ×2).
#      Covers logdet/det/logabsdet only; does not fix inv or \.
#
#   2. Rule for bunchkaufman(::Symmetric) returning a BunchKaufman struct:
#      More involved, but covers all downstream uses (logdet, inv, \).
#
#   3. Rule for LAPACK.sytrf! directly (frule!! + rrule!! on packed LD storage):
#      Maximal coverage, but requires careful handling of LAPACK.syconv! row/col
#      swaps when converting packed LD → clean unit-triangular factor T.
#      Specifically: syconv!(way='C') applies forward row swaps on the strict
#      triangular part; tangents/cotangents computed in T-ordering must be
#      mapped back to A_LD-ordering before storing.
#      No existing public implementation in any AD framework (ChainRules.jl,
#      JAX, PyTorch) — this is novel; see Seeger et al. arXiv:1710.08717 which
#      covers Cholesky/LQ/eigensym but explicitly omits pivoted LDL.
#
# Strategy 2 or 3 required for full coverage (inv, \, factorize, etc.).

function zero_tri!(A, uplo::Char)
    if uplo == 'U'
        tril!(A, -1)
    elseif uplo == 'L'
        triu!(A, 1)
    else
        A .= zero(eltype(A))
    end
    return nothing
end

# Symmetric stores uplo as a Char, but its constructor takes a Symbol.
# The generic _add_to_primal_internal tries P(fields...) which breaks for Symmetric
# because it passes a Char where a Symbol is expected.  Override it here.
function _add_to_primal_internal(
    c::MaybeCache, p::Symmetric{P,M}, t::Tangent, unsafe::Bool
) where {P,M}
    new_data = _add_to_primal_internal(c, p.data, _fields(t).data, unsafe)
    return Symmetric(new_data, Symbol(p.uplo))
end

"""
    _accum_sym_logdet!(ddata::StridedMatrix, Sinv::StridedMatrix, ȳ, uplo)

Accumulate `ȳ * ∂logdet(Symmetric(A, uplo))/∂A` into `ddata` in-place, where
`Sinv = inv(Symmetric(A, uplo))`.

The gradient of `logdet(S)` w.r.t. the stored data array `A` of `S = Symmetric(A, uplo)` is:

    ∂logdet(S)/∂A[i,j] = S⁻¹[i,j]    for i = j  (diagonal)
                        = 2·S⁻¹[i,j]  for i ≠ j, (i,j) in the active triangle
                        = 0            otherwise

The factor of 2 for off-diagonal entries arises because `A[i,j]` represents both
`S[i,j]` and `S[j,i]`. Equivalently, in forward mode: `ḋ = dot(S⁻¹, Symmetric(dA, uplo))`.

This accumulator is shared by the `logdet`, `det`, and `logabsdet` rules:
- `logdet`:     calls with scalar `ȳ`
- `det`:        calls with scalar `ȳ * det(S)`  (chain rule through `exp ∘ logdet`)
- `logabsdet`:  calls with scalar `ȳ[1]`        (sign component has zero derivative)

When `ddata` is a `Symmetric` matrix, `uplo` and the backing store are extracted
automatically via the two-argument overload below.
"""
function _accum_sym_logdet!(
    ddata::StridedMatrix{P}, Sinv::StridedMatrix{P}, ȳ::P, uplo::Char
) where {P}
    n = size(ddata, 1)
    if uplo == 'U'
        @inbounds for j in 1:n
            for i in 1:(j - 1)
                ddata[i, j] += 2 * ȳ * Sinv[i, j]
            end
            ddata[j, j] += ȳ * Sinv[j, j]
        end
    else
        @inbounds for j in 1:n
            ddata[j, j] += ȳ * Sinv[j, j]
            for i in (j + 1):n
                ddata[i, j] += 2 * ȳ * Sinv[i, j]
            end
        end
    end
    return nothing
end
function _accum_sym_logdet!(ddata::Symmetric{P}, Sinv::StridedMatrix{P}, ȳ::P) where {P}
    _accum_sym_logdet!(ddata.data, Sinv, ȳ, ddata.uplo)
end

"""
    logdet(S::Symmetric{<:BlasRealFloat})

Primitive rule for `logdet` of a real symmetric matrix.

Given `S = Symmetric(A, uplo)`, the Fréchet derivative is:

    d/dt logdet(S + t·dS)|_{t=0} = dot(S⁻¹, Symmetric(dA, uplo))

which equals `tr(S⁻¹ · sym(dA))`. See [`_accum_sym_logdet!`](@ref) for the gradient
w.r.t. the underlying data array `A`.
"""
@is_primitive(
    MinimalCtx,
    Tuple{typeof(logdet),Symmetric{P,<:StridedMatrix{P}}} where {P<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(logdet)}, _S::Dual{<:Symmetric{P,<:StridedMatrix{P}}}
) where {P<:BlasRealFloat}
    S, d_data = arrayify(_S)
    F = bunchkaufman(S)
    Sinv = inv(F)
    return Dual(logdet(F), dot(Sinv, d_data))
end
function rrule!!(
    ::CoDual{typeof(logdet)}, _S::CoDual{<:Symmetric{P,<:StridedMatrix{P}}}
) where {P<:BlasRealFloat}
    S, ddata = arrayify(_S)
    F = bunchkaufman(S)
    ld = logdet(F)
    Sinv = inv(F)
    function logdet_sym_pb!!(ȳ::P)
        _accum_sym_logdet!(ddata, Sinv, ȳ)
        return NoRData(), NoRData()
    end
    return CoDual(ld, NoFData()), logdet_sym_pb!!
end

"""
    det(S::Symmetric{<:BlasRealFloat})

Primitive rule for `det` of a real symmetric matrix.

Given `S = Symmetric(A, uplo)`, the Fréchet derivative follows from `det = exp ∘ logdet`:

    d/dt det(S + t·dS)|_{t=0} = det(S) · dot(S⁻¹, Symmetric(dA, uplo))

The reverse-mode cotangent is accumulated via [`_accum_sym_logdet!`](@ref) with scalar
`ȳ · det(S)`.
"""
@is_primitive(
    MinimalCtx, Tuple{typeof(det),Symmetric{P,<:StridedMatrix{P}}} where {P<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(det)}, _S::Dual{<:Symmetric{P,<:StridedMatrix{P}}}
) where {P<:BlasRealFloat}
    S, d_data = arrayify(_S)
    F = bunchkaufman(S; check=false)
    d = det(F)
    # Zero tangent for singular S. Strictly correct only for rank ≤ n-2; at rank n-1
    # the true derivative is the adjugate (nonzero), but exact floating-point zeros are
    # measure-zero in practice.
    iszero(d) && return Dual(d, zero(P))
    Sinv = inv(F)
    return Dual(d, d * dot(Sinv, d_data))
end
function rrule!!(
    ::CoDual{typeof(det)}, _S::CoDual{<:Symmetric{P,<:StridedMatrix{P}}}
) where {P<:BlasRealFloat}
    S, ddata = arrayify(_S)
    F = bunchkaufman(S; check=false)
    d = det(F)
    Sinv = iszero(d) ? nothing : inv(F)
    function det_sym_pb!!(ȳ::P)
        # Zero gradient for singular S (approximate; see frule!! for details).
        isnothing(Sinv) && return NoRData(), NoRData()
        _accum_sym_logdet!(ddata, Sinv, ȳ * d)
        return NoRData(), NoRData()
    end
    return CoDual(d, NoFData()), det_sym_pb!!
end

"""
    logabsdet(S::Symmetric{<:BlasRealFloat})

Primitive rule for `logabsdet` of a real symmetric matrix. Returns `(log|det(S)|, sign(det(S)))`.

Given `S = Symmetric(A, uplo)`, the Fréchet derivative of the first output is identical
to that of `logdet`:

    d/dt log|det(S + t·dS)||_{t=0} = dot(S⁻¹, Symmetric(dA, uplo))

The sign component has zero derivative w.r.t. `A`. In reverse mode only `ȳ[1]` (the
cotangent of the log-magnitude) contributes; `ȳ[2]` is ignored.
"""
@is_primitive(
    MinimalCtx,
    Tuple{typeof(logabsdet),Symmetric{P,<:StridedMatrix{P}}} where {P<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(logabsdet)}, _S::Dual{<:Symmetric{P,<:StridedMatrix{P}}}
) where {P<:BlasRealFloat}
    S, d_data = arrayify(_S)
    F = bunchkaufman(S; check=false)
    ld, s = logabsdet(F)
    iszero(s) && return Dual((ld, s), (zero(P), zero(P)))
    Sinv = inv(F)
    return Dual((ld, s), (dot(Sinv, d_data), zero(P)))
end
function rrule!!(
    ::CoDual{typeof(logabsdet)}, _S::CoDual{<:Symmetric{P,<:StridedMatrix{P}}}
) where {P<:BlasRealFloat}
    S, ddata = arrayify(_S)
    F = bunchkaufman(S; check=false)
    ld, s = logabsdet(F)
    Sinv = iszero(s) ? nothing : inv(F)
    function logabsdet_sym_pb!!(ȳ::Tuple{P,P})
        isnothing(Sinv) && return NoRData(), NoRData()
        _accum_sym_logdet!(ddata, Sinv, ȳ[1])
        return NoRData(), NoRData()
    end
    return CoDual((ld, s), NoFData()), logabsdet_sym_pb!!
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:lapack})
    rng = rng_ctor(123)
    Ps = [Float64, Float32]
    complexPs = [Float64, Float32, ComplexF64, ComplexF32]
    bools = [false, true]
    uplos = ['U', 'L', 'N']
    test_cases = vcat(

        # getrf!
        map_prod(Ps) do (P,)
            As = blas_matrices(rng, P, 5, 5)
            ipiv = Vector{Int}(undef, 5)
            return map(As) do A
                (false, :stability, nothing, getrf!, A)
            end
        end...,
        map_prod(bools, complexPs) do (check, P)
            As = blas_matrices(rng, P, 5, 5)
            ipiv = Vector{Int}(undef, 5)
            return map(As) do A
                (false, :stability, nothing, Core.kwcall, (; check), getrf!, A)
            end
        end...,

        # trtrs!
        map_prod(
            ['U', 'L'], ['N', 'T', 'C'], ['N', 'U'], [1, 3], [-1, 1, 2], Ps
        ) do (ul, tA, diag, N, Nrhs, P)
            As = invertible_blas_matrices(rng, P, N)
            Bs = Nrhs == -1 ? blas_vectors(rng, P, N) : blas_matrices(rng, P, N, Nrhs)
            Bs = filter(B -> stride(B, 1) == 1, Bs)
            return map_prod(As, Bs) do (A, B)
                (false, :none, nothing, trtrs!, ul, tA, diag, A, B)
            end
        end...,

        # getrs
        map_prod(['N', 'T', 'C'], [1, 5], [-1, 1, 2], Ps) do (trans, N, Nrhs, P)
            As = map(LAPACK.getrf!, invertible_blas_matrices(rng, P, N))
            Bs = Nrhs == -1 ? [randn(rng, P, N)] : blas_matrices(rng, P, N, Nrhs)
            return map_prod(As, Bs) do ((A, _), B)
                ipiv = fill(N, N)
                (false, :none, nothing, getrs!, trans, A, ipiv, B)
            end
        end...,

        # getri
        map_prod([1, 9], Ps) do (N, P)
            As = map(LAPACK.getrf!, invertible_blas_matrices(rng, P, N))
            return map(As) do (A, _)
                ipiv = fill(N, N)
                (false, :none, nothing, getri!, A, ipiv)
            end
        end...,

        # potrf
        map_prod([1, 3, 9], Ps) do (N, P)
            As = map(blas_matrices(rng, P, N, N)) do A
                A .= A * A' + I
                return A
            end
            return map_prod(['L', 'U'], As) do (uplo, A)
                return (false, :stability, nothing, potrf!, uplo, A)
            end
        end...,

        # potrs
        map_prod([1, 3, 9], [-1, 1, 2], Ps) do (N, Nrhs, P)
            X = randn(rng, P, N, N)
            A = X * X' + I
            Bs = Nrhs == -1 ? blas_vectors(rng, P, N) : blas_matrices(rng, P, N, Nrhs)
            return map_prod(['L', 'U'], Bs) do (uplo, B)
                tmp = potrf!(uplo, copy(A))[1]
                (false, :none, nothing, potrs!, uplo, tmp, copy(B))
            end
        end...,

        # lacpy!
        (@static if VERSION > v"1.11-"
            map_prod(complexPs, uplos) do (P, uplo)
                As = blas_matrices(rng, P, 5, 5)
                Bs = blas_matrices(rng, P, 5, 5)
                return map_prod(As, Bs) do (A, B)
                    (false, :none, nothing, LAPACK.lacpy!, B, A, uplo)
                end
            end
        else
            []
        end)...,

        # logdet / det / logabsdet on Symmetric
        # Positive-definite inputs: valid for all three functions.
        map_prod([1, 3, 5], ['U', 'L'], Ps) do (N, uplo, P)
            As = positive_definite_blas_matrices(rng, P, N)
            Ss = map(A -> Symmetric(A, Symbol(uplo)), As)
            # For Float32 det, the FD correctness check is unreliable:
            # - Non-contiguous arrays: the FD test normalises the perturbation over the full
            #   parent, so the effective step in the submatrix is O(ε/√parent_size) — too
            #   small for Float32's precision.
            # - Contiguous arrays with large N (e.g. N=5): det can reach O(10³), causing
            #   Float32 cancellation in (det(A+εδ)−det(A−εδ)) to dominate at every step size.
            # Mark all Float32 det tests as interface_only. The gradient is verified
            # indirectly: Float64 det tests exercise the same frule!!/rrule!! code paths, and
            # Float32 logdet/logabsdet pass full FD checks using the same accumulator.
            det_interface_only = P == Float32
            return vcat(
                map(S -> (false, :none, nothing, logdet, S), Ss),
                map(S -> (det_interface_only, :none, nothing, det, S), Ss),
                map(S -> (false, :none, nothing, logabsdet, S), Ss),
            )
        end...,

        # Negative-definite inputs: det < 0 for odd N, det > 0 for even N.
        # logdet is not tested here (requires det > 0).
        map_prod([2, 3], ['U', 'L'], Ps) do (N, uplo, P)
            As = map(positive_definite_blas_matrices(rng, P, N)) do A
                A .= -A
                return A
            end
            Ss = map(A -> Symmetric(A, Symbol(uplo)), As)
            # Same Float32 FD limitations as positive-definite above — use interface_only.
            det_interface_only = P == Float32
            return vcat(
                map(S -> (det_interface_only, :none, nothing, det, S), Ss),
                map(S -> (false, :none, nothing, logabsdet, S), Ss),
            )
        end...,

        # Indefinite inputs: eigenvalues alternate ±1, ±2, …
        # Covers mixed-sign-determinant cases for det and logabsdet.
        map_prod([2, 4], ['U', 'L'], Ps) do (N, uplo, P)
            As = map(invertible_blas_matrices(rng, P, N)) do V
                λs = P[isodd(i) ? P(i) : -P(i) for i in 1:N]
                return collect(V * Diagonal(λs) * V')
            end
            Ss = map(A -> Symmetric(A, Symbol(uplo)), As)
            return vcat(
                map(S -> (false, :none, nothing, det, S), Ss),
                map(S -> (false, :none, nothing, logabsdet, S), Ss),
            )
        end...,

        # Singular inputs: logabsdet returns (-Inf, 0.0) without throwing.
        # FD is not meaningful at a singular point, so interface_only = true.
        # The gradient is zero (iszero(s) guard), which is also not FD-verifiable.
        map_prod([2, 3], ['U', 'L'], Ps) do (N, uplo, P)
            # rank-1 outer-product: v*v' is symmetric and singular for N ≥ 2
            v = ones(P, N)
            A = v * v'
            S = Symmetric(A, Symbol(uplo))
            return [(true, :none, nothing, logabsdet, S)]
        end...,
    )
    memory = Any[]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:lapack})
    rng = rng_ctor(123)
    complexPs = [Float64, Float32, ComplexF64, ComplexF32]
    getrf_wrapper!(x, check) = getrf!(x; check)
    test_cases = vcat(
        # getrf
        map_prod([false, true], complexPs) do (check, P)
            As = blas_matrices(rng, P, 5, 5)
            return map(As) do A
                (false, :none, nothing, getrf_wrapper!, A, check)
            end
        end...,

        # real logdet
        map([Float64, Float32]) do P
            As = positive_definite_blas_matrices(rng, P, 3)
            return map(As) do A
                (false, :none, nothing, logdet, A)
            end
        end...,

        # complex logdet
        map(complexPs) do P
            As = blas_matrices(rng, P, 3, 3)
            return map(As) do A
                (false, :none, nothing, real ∘ logdet ∘ complex, A)
            end
        end...,
    )
    memory = Any[]
    return test_cases, memory
end
