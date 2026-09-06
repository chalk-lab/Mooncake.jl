# See https://sethaxen.com/blog/2021/02/differentiating-the-lu-decomposition/ for details.
# Shared width-N Fréchet-derivative body for the `getrf!` frules (plain and `Core.kwcall`),
# which differ only in the primal call. `A`/`dA_lanes` come from `arrayify(A_dA)` and `A` is
# already overwritten by the in-place `getrf!`.
function _getrf_fwd(A_dA::Lifted{<:AbstractMatrix,Nw}, A, dA_lanes, ipiv, info) where {Nw}
    T = eltype(A)
    # `ipiv` permutes ROWS: the column count reads `p` out of range for a tall `A` below.
    p = LinearAlgebra.ipiv2perm(ipiv, size(A, 1))
    n = size(A, 1)
    # F = L \ (P·dA) / U, then dA = L*tril(F,-1) + triu(F)*U. Two scratches reused across lanes;
    # the solves/products run in place via direct BLAS (LinearAlgebra's triangular `\`/`*` and
    # `[p,:]`/`tril`/`triu` allocate per call). `A` is the packed LU factor: unit-lower `L`
    # (BLAS diag 'U') and upper `U` (diag 'N').
    Fbuf = similar(A)
    buf = similar(A)
    @inbounds for lane in 1:Nw
        dA_lane = dA_lanes[lane]
        for i in 1:n
            @views Fbuf[i, :] .= dA_lane[p[i], :]
        end
        BLAS.trsm!('L', 'L', 'N', 'U', one(T), A, Fbuf)
        BLAS.trsm!('R', 'U', 'N', 'N', one(T), A, Fbuf)
        copyto!(buf, Fbuf)
        tril!(buf, -1)
        BLAS.trmm!('L', 'L', 'N', 'U', one(T), A, buf)
        triu!(Fbuf)
        BLAS.trmm!('R', 'U', 'N', 'N', one(T), A, Fbuf)
        dA_lane .= buf .+ Fbuf
    end
    y = (A, ipiv, info)
    return Lifted{typeof(y),Nw}(y, (tangent(A_dA), zero_dual(Val(Nw), ipiv), NoDual()))
end

@is_primitive(MinimalCtx, Tuple{typeof(LAPACK.getrf!),AbstractMatrix{<:BlasFloat}})
function frule!!(
    ::Lifted{typeof(LAPACK.getrf!),Nw}, A_dA::Lifted{<:AbstractMatrix{P},Nw}
) where {Nw,P<:BlasFloat}
    A, dA_lanes = arrayify(A_dA)
    _, ipiv, info = LAPACK.getrf!(A)
    return _getrf_fwd(A_dA, A, dA_lanes, ipiv, info)
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
    ::Lifted{typeof(Core.kwcall),Nw},
    _kwargs::Lifted{<:NamedTuple},
    ::Lifted{typeof(getrf!),Nw},
    A_dA::Lifted{<:AbstractMatrix{P},Nw},
) where {Nw,P<:BlasFloat}
    check = primal(_kwargs).check
    A, dA_lanes = arrayify(A_dA)
    _, ipiv, info = LAPACK.getrf!(A; check)
    return _getrf_fwd(A_dA, A, dA_lanes, ipiv, info)
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

function _getrf_pb!(A, dA, ipiv, A_copy)

    # Run reverse-pass.
    L = UnitLowerTriangular(A)
    U = UpperTriangular(A)
    dL = tril(dA, -1)
    dU = UpperTriangular(dA)

    # Figure out the pivot matrix used; `ipiv` permutes rows.
    p = LinearAlgebra.ipiv2perm(ipiv, size(A, 1))

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
    ::Lifted{typeof(trtrs!),Nw},
    _uplo::Lifted{Char},
    _trans::Lifted{Char},
    _diag::Lifted{Char},
    A_dA::Lifted{<:AbstractMatrix{P},Nw},
    B_dB::Lifted{<:AbstractVecOrMat{P},Nw},
) where {Nw,P<:BlasRealFloat}
    uplo = _lsame_flag(primal(_uplo))
    trans = _lsame_flag(primal(_trans))
    diag = _lsame_flag(primal(_diag))
    A = primal(A_dA)
    B = primal(B_dB)
    Ab, _ = _partials_block(A_dA)
    Bb, bcopied = _partials_block(B_dB)
    m, nrhs = size(B, 1), size(B, 2)
    Bb3 = reshape(Bb, Nw, m, nrhs)
    # `X = op(A)⁻¹·B` (the primal RHS solve) is lane-invariant: hoist it.
    X = copy(B)
    LAPACK.trtrs!(uplo, trans, diag, A, X)
    # d(op(A)⁻¹·B) = op(A)⁻¹·(dB − op(dA)·X). op(A)⁻¹ is linear, so the tangent takes one
    # solve of that combined RHS, not separate solves of `dB` and `op(dA)·X`.
    # 1) dB_k −= op(dA_k)·X — skipped when `A` is constant data. trmm masks dA's triangle
    #    (and implicit unit diagonal, whose derivative the `diag == 'U'` correction
    #    removes).
    if !iszero(Ab)
        R = size(A, 1)
        Abm = reshape(Ab, Nw, R, R)
        Ascr = Matrix{P}(undef, R, R)
        tmp = Matrix{P}(undef, m, nrhs)
        for k in 1:Nw
            copyto!(Ascr, view(Abm,k,:,:))
            copyto!(tmp, X)
            BLAS.trmm!('L', uplo, trans, diag, one(P), Ascr, tmp)
            diag == 'N' || (tmp .-= reshape(X, m, nrhs))
            view(Bb3,k,:,:) .-= tmp
        end
    end
    # 2) op(A)⁻¹ applied to every lane: right-divide each dB slab by op(A)ᵀ (real
    #    element types only, so a flag flip suffices).
    fA = trans == 'N' ? 'T' : 'N'
    for j in 1:nrhs
        BLAS.trsm!('R', uplo, fA, diag, one(P), A, view(Bb3,:,:,j))
    end
    bcopied && _write_back_partials!(B_dB, Bb)
    # Primal result op(A)⁻¹·B = X, already solved above and unmutated: copy it, don't
    # re-solve.
    copyto!(B, X)
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
    ::Lifted{typeof(getrs!),Nw},
    _trans::Lifted{Char},
    A_dA::Lifted{<:AbstractMatrix{P},Nw},
    _ipiv::Lifted{<:AbstractVector{Int}},
    B_dB::Lifted{<:AbstractVecOrMat{P},Nw},
) where {Nw,P<:BlasRealFloat}
    trans = _lsame_flag(primal(_trans))
    ipiv = primal(_ipiv)
    A = primal(A_dA)
    B = primal(B_dB)
    Ab, _ = _partials_block(A_dA)
    Bb, bcopied = _partials_block(B_dB)
    Bbf = reshape(Bb, Nw, :)
    LAPACK.getrs!(trans, A, ipiv, B)
    U = UpperTriangular(A)
    p = LinearAlgebra.ipiv2perm(ipiv, size(B, 1))
    invp = invperm(p)
    # d(LU) = dL*U + L*dU (dL strict-lower via the unit-diagonal factor, dU upper). Build
    # it into `tmp` with in-place `BLAS.trmm!`, then row-permute by `invp` into `buf`
    # (both reused across lanes). The per-lane getrs! solve needs a dense RHS, so each
    # lane's dB round-trips through the `dBscr` scratch (matching `B`'s shape).
    n = size(A, 1)
    tmp = similar(A)
    buf = similar(A)
    dBscr = Array{P}(undef, size(B))
    danonzero = !iszero(Ab)
    Abm = reshape(Ab, Nw, n, n)
    Ascr = danonzero ? Matrix{P}(undef, n, n) : Matrix{P}(undef, 0, 0)
    @inbounds for lane in 1:Nw
        copyto!(dBscr, view(Bbf, lane, :))
        if danonzero
            copyto!(Ascr, view(Abm,lane,:,:))
            copyto!(tmp, U)
            BLAS.trmm!('L', 'L', 'N', 'U', one(P), Ascr, tmp)
            tmp .-= U
            copyto!(buf, UpperTriangular(Ascr))
            BLAS.trmm!('L', 'L', 'N', 'U', one(P), A, buf)
            tmp .+= buf
            for i in 1:n
                @views buf[i, :] .= tmp[invp[i], :]
            end
            if trans == 'N'
                mul!(dBscr, buf, B, -one(P), one(P))
            else
                mul!(dBscr, buf', B, -one(P), one(P))
            end
        end
        LAPACK.getrs!(trans, A, ipiv, dBscr)
        copyto!(view(Bbf, lane, :), dBscr)
    end
    bcopied && _write_back_partials!(B_dB, Bb)
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
    ::Lifted{typeof(getri!),Nw},
    A_dA::Lifted{<:AbstractMatrix{P},Nw},
    _ipiv::Lifted{<:AbstractVector{Int}},
) where {Nw,P<:BlasRealFloat}
    A = primal(A_dA)
    ipiv = primal(_ipiv)
    Ab, acopied = _partials_block(A_dA)
    U = UpperTriangular(A)
    p = LinearAlgebra.ipiv2perm(ipiv, size(A, 1))
    invp = invperm(p)
    n = size(A, 1)
    Abm = reshape(Ab, Nw, n, n)
    buf1 = similar(A)
    buf2 = similar(A)
    Ascr = Matrix{P}(undef, n, n)
    # Phase 1 (before getri! destroys the LU factor A): store tmp2 = (dL*U + L*dU)[invp,:]
    # for each lane INTO its own block slice (the output partial), via the dense `Ascr`
    # scratch (the BLAS calls need dense operands; a block lane is stride-Nw).
    @inbounds for lane in 1:Nw
        copyto!(Ascr, view(Abm,lane,:,:))
        copyto!(buf1, U)
        BLAS.trmm!('L', 'L', 'N', 'U', one(P), Ascr, buf1)
        buf1 .-= U
        copyto!(buf2, UpperTriangular(Ascr))
        BLAS.trmm!('L', 'L', 'N', 'U', one(P), A, buf2)
        buf1 .+= buf2
        for i in 1:n
            @views view(Abm, lane, i, :) .= buf1[invp[i], :]
        end
    end
    LAPACK.getri!(A, ipiv)
    # Phase 2: lane := -A⁻¹ * tmp2 * A⁻¹, with tmp2 currently held in the lane and A now
    # holding A⁻¹.
    @inbounds for lane in 1:Nw
        copyto!(Ascr, view(Abm,lane,:,:))
        mul!(buf1, A, Ascr)
        mul!(Ascr, buf1, A, -one(P), zero(P))
        copyto!(view(Abm,lane,:,:), Ascr)
    end
    acopied && _write_back_partials!(A_dA, Ab)
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
    ::Lifted{typeof(potrf!),Nw}, _uplo::Lifted{Char}, A_dA::Lifted{<:AbstractMatrix{P},Nw}
) where {Nw,P<:BlasRealFloat}
    uplo = _lsame_flag(primal(_uplo))
    A = primal(A_dA)
    Ab, acopied = _partials_block(A_dA)
    _, info = LAPACK.potrf!(uplo, A)
    N = size(A, 1)
    Abm = reshape(Ab, Nw, N, N)
    # Left and right solves stack lanes differently; at width 1 both layouts share storage.
    # Write back only the factor's triangle, preserving the untouched triangle's partials.
    S = Array{P}(undef, N, Nw, N)
    T = Nw == 1 ? reshape(S, Nw, N, N) : Array{P}(undef, Nw, N, N)
    if uplo == 'U'
        @inbounds for j in 1:N, lane in 1:Nw, i in 1:N
            S[i, lane, j] = i <= j ? Abm[lane, i, j] : Abm[lane, j, i]
        end
        BLAS.trsm!('L', 'U', 'T', 'N', one(P), A, reshape(S, N, Nw * N))
        if Nw != 1
            @inbounds for j in 1:N, lane in 1:Nw, i in 1:N
                T[lane, i, j] = S[i, lane, j]
            end
        end
        Tf = reshape(T, Nw * N, N)
        BLAS.trsm!('R', 'U', 'N', 'N', one(P), A, Tf)
        @inbounds for lane in 1:Nw
            for n in 1:N
                T[lane, n, n] /= 2
            end
            for j in 1:N, i in (j + 1):N
                T[lane, i, j] = zero(P)
            end
        end
        BLAS.trmm!('R', 'U', 'N', 'N', one(P), A, Tf)
        @inbounds for lane in 1:Nw, q in 1:N, i in 1:q
            Abm[lane, i, q] = T[lane, i, q]
        end
    else
        @inbounds for lane in 1:Nw, i in 1:N, j in 1:N
            T[lane, i, j] = i >= j ? Abm[lane, i, j] : Abm[lane, j, i]
        end
        Tf = reshape(T, Nw * N, N)
        BLAS.trsm!('R', 'L', 'T', 'N', one(P), A, Tf)
        if Nw != 1
            @inbounds for j in 1:N, lane in 1:Nw, i in 1:N
                S[i, lane, j] = T[lane, i, j]
            end
        end
        Sf = reshape(S, N, Nw * N)
        BLAS.trsm!('L', 'L', 'N', 'N', one(P), A, Sf)
        @inbounds for lane in 1:Nw
            for n in 1:N
                S[n, lane, n] /= 2
            end
            for j in 1:N, i in 1:(j - 1)
                S[i, lane, j] = zero(P)
            end
        end
        BLAS.trmm!('L', 'L', 'N', 'N', one(P), A, Sf)
        @inbounds for lane in 1:Nw, q in 1:N, i in q:N
            Abm[lane, i, q] = S[i, lane, q]
        end
    end
    acopied && _write_back_partials!(A_dA, Ab)
    y = (A, info)
    return Lifted{typeof(y),Nw}(y, (tangent(A_dA), NoDual()))
end
function rrule!!(
    ::CoDual{typeof(potrf!)}, _uplo::CoDual{Char}, _A::CoDual{<:AbstractMatrix{P}}
) where {P<:BlasRealFloat}

    # Extract args and take a copy of A.
    uplo = _uplo.x
    A, dA = arrayify(_A)
    # Keep `copy` in the IR so forward-over-reverse AD can dispatch to its `frule!!`, as `trtrs!`
    # does: inlined, its internal `jl_genericmemory_copy_slice` ccall has no forward rule and an
    # HVP through `cholesky` dies. `A_copy` is live — the pullback restores the primal with it.
    A_copy = Base.@noinline copy(A)

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
    ::Lifted{typeof(potrs!),Nw},
    _uplo::Lifted{Char},
    A_dA::Lifted{<:AbstractMatrix{P},Nw},
    B_dB::Lifted{<:AbstractVecOrMat{P},Nw},
) where {Nw,P<:BlasRealFloat}
    uplo = _lsame_flag(primal(_uplo))
    A = primal(A_dA)
    B = primal(B_dB)
    Ab, _ = _partials_block(A_dA)
    Bb, bcopied = _partials_block(B_dB)
    Bbf = reshape(Bb, Nw, :)
    n = size(A, 1)
    Abm = reshape(Ab, Nw, n, n)
    LAPACK.potrs!(uplo, A, B)
    # dS = dL*L' + L*dL' (resp. U'dU + dU'U). Build its two (triangular*triangular) terms
    # into hoisted scratches via in-place `BLAS.trmm!` (materialize one factor, apply the
    # other in place); the symmetric `mul!` then runs BLAS symm!/symv! in place. The sum
    # is symmetric (`X + X'`), so `Symmetric(buf1)` (uplo `:U`) reads it exactly. The
    # BLAS/LAPACK calls need dense operands, so each lane's dA and dB round-trip through
    # the `Ascr`/`dBscr` scratches (a block lane is stride-Nw).
    buf1 = similar(A)
    buf2 = similar(A)
    Ascr = Matrix{P}(undef, n, n)
    dBscr = Array{P}(undef, size(B))
    @inbounds for lane in 1:Nw
        copyto!(Ascr, view(Abm,lane,:,:))
        copyto!(dBscr, view(Bbf, lane, :))
        if uplo == 'L'
            copyto!(buf1, adjoint(LowerTriangular(A)))
            BLAS.trmm!('L', 'L', 'N', 'N', one(P), Ascr, buf1)
            copyto!(buf2, adjoint(LowerTriangular(Ascr)))
            BLAS.trmm!('L', 'L', 'N', 'N', one(P), A, buf2)
        else
            copyto!(buf1, UpperTriangular(Ascr))
            BLAS.trmm!('L', 'U', 'T', 'N', one(P), A, buf1)
            copyto!(buf2, UpperTriangular(A))
            BLAS.trmm!('L', 'U', 'T', 'N', one(P), Ascr, buf2)
        end
        buf1 .+= buf2
        mul!(dBscr, Symmetric(buf1), B, -one(P), one(P))
        LAPACK.potrs!(uplo, A, dBscr)
        copyto!(view(Bbf, lane, :), dBscr)
    end
    bcopied && _write_back_partials!(B_dB, Bb)
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
        ::Lifted{typeof(LAPACK.lacpy!),Nw},
        B_dB::Lifted{<:AbstractMatrix{P},Nw},
        A_dA::Lifted{<:AbstractMatrix{P},Nw},
        _uplo::Lifted{Char},
    ) where {Nw,P<:BlasFloat}
        uplo = _lsame_flag(primal(_uplo))
        B = primal(B_dB)
        A = primal(A_dA)
        Ab, _ = _partials_block(A_dA)
        Bb, bcopied = _partials_block(B_dB)
        LAPACK.lacpy!(B, A, uplo)
        # The tangent copy mirrors the primal's triangle selection, applied to all lanes
        # at once: for each copied element the `Nw` lanes are one contiguous block
        # column, so the copies below move whole lane columns.
        m, n = size(A)
        Ab3 = reshape(Ab, Nw, size(A)...)
        Bb3 = reshape(Bb, Nw, size(B)...)
        if uplo == 'U'
            for j in 1:n
                r = 1:min(j, m)
                view(Bb3, :, r, j) .= view(Ab3, :, r, j)
            end
        elseif uplo == 'L'
            for j in 1:n
                r = j:m
                view(Bb3, :, r, j) .= view(Ab3, :, r, j)
            end
        else
            view(Bb3, :, 1:m, 1:n) .= Ab3
        end
        bcopied && _write_back_partials!(B_dB, Bb)
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
        uplo = _lsame_flag(primal(_uplo))

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
# A real `Hermitian` IS the `Symmetric` matrix with the same stored triangle, takes the same
# `bunchkaufman`/`sytrf!` path, and gives bit-identical `logdet`, `inv` and `dot` weighting, so the
# determinant rules below serve both. `BlasRealFloat` keeps complex out, where the two differ.
const _SymHerm{P} = Union{Symmetric{P,<:StridedMatrix{P}},Hermitian{P,<:StridedMatrix{P}}}

function _accum_sym_logdet!(
    ddata::Union{Symmetric{P},Hermitian{P}}, Sinv::StridedMatrix{P}, ȳ::P
) where {P}
    _accum_sym_logdet!(ddata.data, Sinv, ȳ, ddata.uplo)
end

"""
    logdet(S::Union{Symmetric,Hermitian}{<:BlasRealFloat})

Primitive rule for `logdet` of a real symmetric matrix. A real `Hermitian` is the same matrix
and takes the same path, so it is served here too.

Given `S = Symmetric(A, uplo)`, the Fréchet derivative is:

    d/dt logdet(S + t·dS)|_{t=0} = dot(S⁻¹, Symmetric(dA, uplo))

which equals `tr(S⁻¹ · sym(dA))`. See [`_accum_sym_logdet!`](@ref) for the gradient
w.r.t. the underlying data array `A`.
"""
@is_primitive(MinimalCtx, Tuple{typeof(logdet),_SymHerm{P}} where {P<:BlasRealFloat})
function frule!!(
    ::Lifted{typeof(logdet),Nw}, _S::Lifted{<:_SymHerm{P},Nw,<:ImmutableDual}
) where {Nw,P<:BlasRealFloat}
    S, d_lanes = arrayify(_S)
    F = bunchkaufman(S)
    Sinv = inv(F)
    y = logdet(F)
    # `arrayify` re-wraps each lane's `.data` partial as `Symmetric(·, uplo)`, applying the storage
    # weighting (2× off-diagonals, 1× diagonal, 0 off-triangle) the reverse `rrule!!` encodes via
    # `_accum_sym_logdet!`; a plain `dot` over the full matrix would be wrong.
    dy_lanes = ntuple(k -> dot(Sinv, d_lanes[k]), Val(Nw))
    return Lifted{P,Nw}(y, _scalar_ndual(y, dy_lanes))
end
function rrule!!(
    ::CoDual{typeof(logdet)}, _S::CoDual{<:_SymHerm{P}}
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
    _sym_adjugate(S::Union{Symmetric,Hermitian}{<:BlasRealFloat})

Adjugate of a real symmetric matrix, valid at a singular `S`.

`adj(S) = det(S)·S⁻¹` whenever `S` is invertible, which is how the rules below obtain it. At a
singular `S` that product is `0·Inf`, so take the eigendecomposition instead: for `S = QΛQᵀ`,
`adj(S) = Q·diag(∏_{j≠i} λⱼ)·Qᵀ`. That is zero at rank ≤ n-2 and rank one at rank n-1, which is
the derivative the product form cannot express.
"""
function _sym_adjugate(S::_SymHerm{P}) where {P<:BlasRealFloat}
    F = eigen(S)
    λ = F.values
    cofactors = similar(λ)
    @inbounds for i in eachindex(λ)
        c = one(P)
        for j in eachindex(λ)
            j == i || (c *= λ[j])
        end
        cofactors[i] = c
    end
    return F.vectors * Diagonal(cofactors) * transpose(F.vectors)
end

"""
    det(S::Union{Symmetric,Hermitian}{<:BlasRealFloat})

Primitive rule for `det` of a real symmetric matrix, `Hermitian` included.

Given `S = Symmetric(A, uplo)`, the Fréchet derivative is the adjugate contraction:

    d/dt det(S + t·dS)|_{t=0} = dot(adj(S), Symmetric(dA, uplo))

For invertible `S` this is `det(S) · dot(S⁻¹, Symmetric(dA, uplo))`, since `adj(S) = det(S)·S⁻¹`,
and the reverse-mode cotangent is accumulated via [`_accum_sym_logdet!`](@ref) with scalar
`ȳ · det(S)`. At a singular `S` that product is `0·Inf`, so both rules obtain the adjugate from
[`_sym_adjugate`](@ref) and accumulate with scalar `ȳ`. The derivative is well defined there: it
vanishes at rank ≤ n-2 and is rank one at rank n-1.
"""
@is_primitive(MinimalCtx, Tuple{typeof(det),_SymHerm{P}} where {P<:BlasRealFloat},)
function frule!!(
    ::Lifted{typeof(det),Nw}, _S::Lifted{<:_SymHerm{P},Nw,<:ImmutableDual}
) where {Nw,P<:BlasRealFloat}
    S = primal(_S)
    F = bunchkaufman(S; check=false)
    d = det(F)
    # See `logdet` frule: `arrayify` applies the symmetric-storage weighting to each lane.
    _, d_lanes = arrayify(_S)
    # `ḋ = dot(adj(S), dS)`. Keep the cheap `d·S⁻¹` form off the singular path.
    dy_lanes = if iszero(d)
        adj = _sym_adjugate(S)
        ntuple(k -> dot(adj, d_lanes[k]), Val(Nw))
    else
        Sinv = inv(F)
        ntuple(k -> d * dot(Sinv, d_lanes[k]), Val(Nw))
    end
    return Lifted{P,Nw}(d, _scalar_ndual(d, dy_lanes))
end
function rrule!!(::CoDual{typeof(det)}, _S::CoDual{<:_SymHerm{P}}) where {P<:BlasRealFloat}
    S, ddata = arrayify(_S)
    F = bunchkaufman(S; check=false)
    d = det(F)
    # `S̄ += ȳ·adj(S)`, weighted for symmetric storage. Keep the cheap `d·S⁻¹` form off the
    # singular path, where it is `0·Inf`.
    G, scale = iszero(d) ? (_sym_adjugate(S), one(P)) : (inv(F), d)
    function det_sym_pb!!(ȳ::P)
        _accum_sym_logdet!(ddata, G, ȳ * scale)
        return NoRData(), NoRData()
    end
    return CoDual(d, NoFData()), det_sym_pb!!
end

"""
    logabsdet(S::Union{Symmetric,Hermitian}{<:BlasRealFloat})

Primitive rule for `logabsdet` of a real symmetric matrix, `Hermitian` included. Returns
`(log|det(S)|, sign(det(S)))`.

Given `S = Symmetric(A, uplo)`, the Fréchet derivative of the first output is identical
to that of `logdet`:

    d/dt log|det(S + t·dS)||_{t=0} = dot(S⁻¹, Symmetric(dA, uplo))

The sign component has zero derivative w.r.t. `A`. In reverse mode only `ȳ[1]` (the
cotangent of the log-magnitude) contributes; `ȳ[2]` is ignored.
"""
@is_primitive(MinimalCtx, Tuple{typeof(logabsdet),_SymHerm{P}} where {P<:BlasRealFloat},)
function frule!!(
    ::Lifted{typeof(logabsdet),Nw}, _S::Lifted{<:_SymHerm{P},Nw,<:ImmutableDual}
) where {Nw,P<:BlasRealFloat}
    S = primal(_S)
    F = bunchkaufman(S; check=false)
    ld, s = logabsdet(F)
    y = (ld, s)
    # The sign `s` always has zero derivative; a singular S (s==0) zeros `ld`'s derivative too.
    iszero(s) && return zero_lifted(Val(Nw), y)
    Sinv = inv(F)
    # See `logdet` frule: `arrayify` applies the symmetric-storage weighting to each lane.
    _, d_lanes = arrayify(_S)
    ld_lanes = ntuple(k -> dot(Sinv, d_lanes[k]), Val(Nw))
    return Lifted{typeof(y),Nw}(y, (_scalar_ndual(ld, ld_lanes), zero_dual(Val(Nw), s)))
end
function rrule!!(
    ::CoDual{typeof(logabsdet)}, _S::CoDual{<:_SymHerm{P}}
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

# `getrf!`'s derivative is derived for a square factor, so every rectangular shape must refuse. A
# tall `A` gathered the row permutation out of range under `@inbounds` and segfaulted instead.
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

        # The same three on a real `Hermitian`, which reaches the identical `bunchkaufman`/`sytrf!`
        # path and gave `MissingForeigncallRuleError` before the rules admitted it. One size only:
        # the numerics are the `Symmetric` ones above (measured bit-identical), so what these pin is
        # that dispatch reaches the rule at all, at both uplos and both precisions.
        map_prod(['U', 'L'], Ps) do (uplo, P)
            Hs = map(
                A -> Hermitian(A, Symbol(uplo)), positive_definite_blas_matrices(rng, P, 3)
            )
            return vcat(
                map(H -> (false, :none, nothing, logdet, H), Hs),
                map(H -> (P == Float32, :none, nothing, det, H), Hs),
                map(H -> (false, :none, nothing, logabsdet, H), Hs),
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
            # `Float32` `det` is `interface_only` for the same reason as the definite and
            # `Hermitian` blocks above: the FD check, not the rule, is what fails. On an
            # ill-conditioned draw the estimates scatter over four orders of magnitude around a
            # stable rule value, and the rule agrees with the same computation in `Float64` to
            # a relative 2e-7. `logabsdet` below keeps its full `Float32` FD check.
            return vcat(
                map(S -> (P == Float32, :none, nothing, det, S), Ss),
                map(S -> (false, :none, nothing, logabsdet, S), Ss),
            )
        end...,

        # Singular inputs. `logabsdet` returns (-Inf, 0.0) without throwing and its gradient is
        # zero by the `iszero(s)` guard; neither is FD-verifiable, so it stays interface_only.
        # `det` IS differentiable at a singular point -- the derivative is the adjugate -- so it
        # is FD-checked here: N = 2 is rank n-1, where the adjugate is nonzero, and N = 3 is
        # rank n-2, where it vanishes. `Float32` det is interface_only for the same
        # FD-cancellation reason as the definite and indefinite cases above.
        map_prod([2, 3], ['U', 'L'], Ps) do (N, uplo, P)
            # rank-1 outer-product: v*v' is symmetric and singular for N ≥ 2
            v = ones(P, N)
            A = v * v'
            S = Symmetric(A, Symbol(uplo))
            return [
                (true, :none, nothing, logabsdet, S), (P == Float32, :none, nothing, det, S)
            ]
        end...,
    )
    # `getrf!` must refuse a non-square matrix. Forward only: the reverse leg drives the primal
    # through `value_and_gradient!!`, which refuses `getrf!`'s tuple return before any rule runs.
    tall = Float64[1 1 1; 1 2 1; 1 1 3; 1 1 1; 9 1 1]
    wide = collect(transpose(tall))
    # `vcat` above types the opts slot from rows that all carry `nothing`, so a row with a
    # `NamedTuple` there needs an `Any` element type rather than `push!`.
    test_cases = vcat(
        Any[test_cases...],
        Any[
            (
                false,
                :none,
                (throws=(DimensionMismatch, "matrix is not square"), mode=ForwardMode),
                LAPACK.getrf!,
                A,
            ) for A in (tall, wide)
        ],
    )
    memory = Any[tall, wide]
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
