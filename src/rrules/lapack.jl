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

    if trans == 'N'
        # Apply permutation matrix.
        B .= B[p, :]

        # Run inv(L) * B and write result to B.
        LAPACK.trtrs!('L', 'N', 'U', A, B)
        B1 = copy(B) # record intermediate state for use in pullback.

        # Run inv(U) * B and write result to B.
        LAPACK.trtrs!('U', 'N', 'N', A, B)
        B2 = B
    else
        # Run inv(U)^T * B and write result to B.
        LAPACK.trtrs!('U', 'T', 'N', A, B)
        B1 = copy(B) # record intermediate state for use in pullback.

        # Run inv(L)^T * B and write result to B.
        LAPACK.trtrs!('L', 'T', 'U', A, B)
        B2 = B

        # Apply permutation matrix.
        B2 .= B2[invperm(p), :]
    end

    function getrs_pb!!(::NoRData)
        if trans == 'N'

            # Run pullback for inv(U) * B.
            LAPACK.trtrs!('U', 'T', 'N', A, dB)
            dA .-= tri!(dB * B2', 'U', 'N')

            # Run pullback for inv(L) * B.
            LAPACK.trtrs!('L', 'T', 'U', A, dB)
            dA .-= tri!(dB * B1', 'L', 'U')

            # Undo permutation.
            dB .= dB[invperm(p), :]
        else

            # Undo permutation.
            dB .= dB[p, :]
            B2 .= B2[p, :]

            # Run pullback for inv(L^T) * B.
            LAPACK.trtrs!('L', 'N', 'U', A, dB)
            dA .-= tri!(B2 * dB', 'L', 'U')

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

function hand_written_rule_test_cases(rng_ctor, ::Val{:lapack})
    rng = rng_ctor(123)
    Ps = [Float64, Float32]
    complexPs = [Float64, Float32, ComplexF64, ComplexF32]
    bools = [false, true]
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
