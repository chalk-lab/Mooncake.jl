@is_primitive(MinimalCtx, Tuple{typeof(LAPACK.getrf!),AbstractMatrix{<:BlasFloat}})
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
    # ipiv .= ipiv_copy
    A .= A_copy

    return nothing
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(trtrs!),Char,Char,Char,AbstractMatrix{P},AbstractVecOrMat{P}
    } where {P<:BlasRealFloat},
)
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

    function trtrs_pb!!(::NoRData)
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
    return _B, trtrs_pb!!
end

@is_primitive(
    MinimalCtx, Tuple{typeof(getri!),AbstractMatrix{<:BlasRealFloat},AbstractVector{Int}},
)
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

__sym(X) = (X + X') / 2

@is_primitive(MinimalCtx, Tuple{typeof(potrf!),Char,AbstractMatrix{<:BlasRealFloat}})
function rrule!!(
    ::CoDual{typeof(potrf!)},
    _uplo::CoDual{Char},
    _A::CoDual{<:AbstractMatrix{<:BlasRealFloat}},
)
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
            E = LowerTriangular(2 * ones(N, N)) - Diagonal(ones(N))
            L = LowerTriangular(A)
            B = L' \ (E' .* (dA2'L)) / L
            dA .= 0.5 * __sym(B) .* E .+ triu!(dA2, 1)
        else
            E = UpperTriangular(2 * ones(N, N) - Diagonal(ones(N)))
            U = UpperTriangular(A)
            B = U \ ((U * dA2') .* E') / U'
            dA .= 0.5 * __sym(B) .* E .+ tril!(dA2, -1)
        end

        # Restore initial state.
        A .= A_copy

        return NoRData(), NoRData(), NoRData()
    end
    return CoDual((_A.x, info), (_A.dx, NoFData())), potrf_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(potrs!),Char,AbstractMatrix{P},AbstractVecOrMat{P}
    } where {P<:BlasRealFloat},
)
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
            tmp = __sym(B_copy * dB') / LowerTriangular(A)'
            dA .-= 2 .* tril!(LinearAlgebra.LAPACK.potrs!('L', A, tmp))
            LinearAlgebra.LAPACK.potrs!('L', A, dB)
        else
            tmp = UpperTriangular(A)' \ __sym(B_copy * dB')
            dA .-= 2 .* triu!((tmp / UpperTriangular(A)) / UpperTriangular(A)')
            LinearAlgebra.LAPACK.potrs!('U', A, dB)
        end

        # Restore initial state.
        B .= B_copy

        return tuple_fill(NoRData(), Val(4))
    end
    return _B, potrs_pb!!
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:lapack})
    rng = rng_ctor(123)
    Ps = [Float64, Float32]
    complexPs = [Float64, Float32, ComplexF64, ComplexF32]
    bools = [false, true]
    test_cases = vcat(

        # getrf!
        map_prod(bools, complexPs) do (check, P)
            As = blas_matrices(rng, P, 5, 5)
            ipiv = Vector{Int}(undef, 5)
            return map(As) do A
                (false, :none, nothing, getrf!, A)
            end
        end...,

        # trtrs
        map_prod(
            ['U', 'L'], ['N', 'T', 'C'], ['N', 'U'], [1, 3], [1, 2], Ps
        ) do (ul, tA, diag, N, Nrhs, P)
            As = blas_matrices(rng, P, N, N)
            Bs = blas_matrices(rng, P, N, Nrhs)
            return map(As, Bs) do A, B
                (false, :none, nothing, trtrs!, ul, tA, diag, A, B)
            end
        end...,

        # getrs
        map_prod(['N', 'T'], [1, 9], [1, 2], Ps) do (trans, N, Nrhs, P)
            As = map(blas_matrices(rng, P, N, N)) do A
                A[diagind(A)] .+= 5
                return getrf!(A)
            end
            Bs = blas_matrices(rng, P, N, Nrhs)
            return map(As, Bs) do (A, ipiv), B
                (false, :none, nothing, getrs!, trans, A, ipiv, B)
            end
        end...,

        # getri
        map_prod([1, 9], Ps) do (N, P)
            As = map(blas_matrices(rng, P, N, N)) do A
                A[diagind(A)] .+= 5
                return getrf!(A)
            end
            return map(As) do (A, ipiv)
                (false, :none, nothing, getri!, A, ipiv)
            end
        end...,

        # potrf
        map_prod([1, 3, 9], Ps) do (N, P)
            As = map(blas_matrices(rng, P, N, N)) do A
                A .= A * A' + I
                return A
            end
            return map(['L', 'U'], As) do uplo, A
                return (false, :none, nothing, potrf!, uplo, A)
            end
        end...,

        # potrs
        map_prod([1, 3, 9], [1, 2], Ps) do (N, Nrhs, P)
            X = randn(rng, P, N, N)
            A = X * X' + I
            Bs = blas_matrices(rng, P, N, Nrhs)
            return map(['L', 'U'], Bs) do uplo, B
                (false, :none, nothing, potrs!, uplo, potrf!(uplo, copy(A))[1], copy(B))
            end
        end...,
    )
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:lapack})
    rng = rng_ctor(123)
    getrf_wrapper!(x, check) = getrf!(x; check)
    test_cases = vcat(
        map_prod([false, true], [Float64, Float32, ComplexF64, ComplexF32]) do (check, P)
            As = blas_matrices(rng, P, 5, 5)
            # ipiv = Vector{Int}(undef, 5)
            return map(As) do A
                (false, :none, nothing, getrf_wrapper!, A, check)
            end
        end...
    )
    memory = Any[]
    return test_cases, memory
end
