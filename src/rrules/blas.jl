function blas_name(name::Symbol)
    return (BLAS.USE_BLAS64 ? Symbol(name, "64_") : name, Symbol(BLAS.libblastrampoline))
end

function wrap_ptr_as_view(ptr::Ptr{T}, N::Int, inc::Int) where {T}
    return view(unsafe_wrap(Vector{T}, ptr, N * inc), 1:inc:(N * inc))
end

function wrap_ptr_as_view(ptr::Ptr{T}, buffer_nrows::Int, nrows::Int, ncols::Int) where {T}
    return view(unsafe_wrap(Matrix{T}, ptr, (buffer_nrows, ncols)), 1:nrows, :)
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

const MatrixOrView{T} = Union{Matrix{T},SubArray{T,2,<:Array{T}}}
const VecOrView{T} = Union{Vector{T},SubArray{T,1,<:Array{T}}}
const BlasRealFloat = Union{Float32,Float64}
const BlasComplexFloat = Union{ComplexF32,ComplexF64}

"""
    arrayify(x::CoDual{<:AbstractArray{<:BlasFloat}})

Return the primal field of `x`, and convert its fdata into an array of the same type as the
primal. This operation is not guaranteed to be possible for all array types, but seems to be
possible for all array types of interest so far.
"""
function arrayify(x::CoDual{A}) where {A<:AbstractArray{<:BlasFloat}}
    return arrayify(primal(x), tangent(x))  # NOTE: for complex number, the tangent is a reinterpreted version of the primal
end
arrayify(x::Array{P}, dx::Array{P}) where {P<:BlasRealFloat} = (x, dx)
function arrayify(x::Array{P}, dx::Array{<:Tangent}) where {P<:BlasComplexFloat}
    return x, reinterpret(P, dx)
end
function arrayify(x::A, dx::FData) where {A<:SubArray{<:BlasFloat}}
    _, _dx = arrayify(x.parent, dx.data.parent)
    return x, A(_dx, x.indices, x.offset1, x.stride1)
end
function arrayify(x::A, dx::FData) where {A<:Base.ReshapedArray{<:BlasFloat}}
    _, _dx = arrayify(x.parent, dx.data.parent)
    return x, A(_dx, x.dims, x.mi)
end
function arrayify(x::Base.ReinterpretArray{T}, dx::FData) where {T<:BlasFloat}
    _, _dx = arrayify(x.parent, dx.data.parent)
    return x, reinterpret(T, _dx)
end

function arrayify(x::A, dx::DA) where {A,DA}
    msg =
        "Encountered unexpected array type in `Mooncake.arrayify`. This error is likely " *
        "due to a call to a BLAS or LAPACK function with an array type that " *
        "Mooncake has not been told about. A new method of `Mooncake.arrayify` is needed." *
        " Please open an issue at " *
        "https://github.com/chalk-lab/Mooncake.jl/issues . " *
        "It should contain this error message and the associated stack trace.\n\n" *
        "Array type: $A\n\nFData type: $DA."
    return error(msg)
end

#
# Utility
#

@zero_adjoint MinimalCtx Tuple{typeof(BLAS.get_num_threads)}
@zero_adjoint MinimalCtx Tuple{typeof(BLAS.lbt_get_num_threads)}
@zero_adjoint MinimalCtx Tuple{typeof(BLAS.set_num_threads),Union{Integer,Nothing}}
@zero_adjoint MinimalCtx Tuple{typeof(BLAS.lbt_set_num_threads),Any}

#
# LEVEL 1
#

for (fname, elty) in ((:cblas_ddot, :Float64), (:cblas_sdot, :Float32))
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
        args::Vararg{Any,N},
    ) where {N}
        GC.@preserve args begin
            # Load in values from pointers.
            n, incx, incy = map(primal, (_n, _incx, _incy))
            xinds = 1:incx:(incx * n)
            yinds = 1:incy:(incy * n)
            DX = view(unsafe_wrap(Vector{$elty}, primal(_DX), n * incx), xinds)
            DY = view(unsafe_wrap(Vector{$elty}, primal(_DY), n * incy), yinds)

            _dDX = view(unsafe_wrap(Vector{$elty}, tangent(_DX), n * incx), xinds)
            _dDY = view(unsafe_wrap(Vector{$elty}, tangent(_DY), n * incy), yinds)

            out = dot(DX, DY)
        end

        function ddot_pb!!(dv)
            GC.@preserve args begin
                _dDX .+= DY .* dv
                _dDY .+= DX .* dv
            end
            return tuple_fill(NoRData(), Val(N + 11))
        end

        # Run primal computation.
        return zero_fcodual(out), ddot_pb!!
    end
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.nrm2),Int,X,Int
    } where {T<:BlasFloat,X<:Union{Ptr{T},AbstractArray{T}}},
)
function rrule!!(
    ::CoDual{typeof(BLAS.nrm2)},
    n::CoDual{<:Integer},
    X_dX::CoDual{<:Union{Ptr{T},AbstractArray{T}} where {T<:BlasFloat}},
    incx::CoDual{<:Integer},
)
    X, dX = arrayify(X_dX)
    y = BLAS.nrm2(n.x, X, incx.x)
    function nrm2_pb!!(dy)
        view(dX, 1:(incx.x):(incx.x * n.x)) .+=
            view(X, 1:(incx.x):(incx.x * n.x)) .* (dy / y)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, NoFData()), nrm2_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{typeof(BLAS.nrm2),X} where {T<:BlasFloat,X<:Union{Ptr{T},AbstractArray{T}}},
)
function rrule!!(
    ::CoDual{typeof(BLAS.nrm2)},
    X_dX::CoDual{<:Union{Ptr{T},AbstractArray{T}} where {T<:BlasFloat}},
)
    X, dX = arrayify(X_dX)
    y = BLAS.nrm2(X)
    function nrm2_pb!!(dy)
        dX .+= X .* (dy / y)
        return NoRData(), NoRData()
    end
    return CoDual(y, NoFData()), nrm2_pb!!
end

for (fname, elty) in ((:dscal_, :Float64), (:sscal_, :Float32))
    @eval @inline function Mooncake.rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        n::CoDual{Ptr{BLAS.BlasInt}},
        DA::CoDual{Ptr{$elty}},
        DX::CoDual{Ptr{$elty}},
        incx::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,N},
    ) where {N}
        GC.@preserve args begin

            # Load in values from pointers, and turn pointers to memory buffers into Vectors.
            _n = unsafe_load(primal(n))
            _incx = unsafe_load(primal(incx))
            _DA = unsafe_load(primal(DA))
            _DX = unsafe_wrap(Vector{$elty}, primal(DX), _n * _incx)
            _DX_s = unsafe_wrap(Vector{$elty}, tangent(DX), _n * _incx)

            inds = 1:_incx:(_incx * _n)
            DX_copy = _DX[inds]
            BLAS.scal!(_n, _DA, _DX, _incx)

            dDA = tangent(DA)
            dDX = tangent(DX)
        end

        function dscal_pullback!!(::NoRData)
            GC.@preserve args begin

                # Set primal to previous state.
                _DX[inds] .= DX_copy

                # Compute cotangent w.r.t. scaling.
                unsafe_store!(dDA, BLAS.dot(_n, _DX, _incx, dDX, _incx) + unsafe_load(dDA))

                # Compute cotangent w.r.t. DX.
                BLAS.scal!(_n, _DA, _DX_s, _incx)
            end

            return tuple_fill(NoRData(), Val(10 + N))
        end
        return zero_fcodual(Cvoid()), dscal_pullback!!
    end
end

#
# LEVEL 2
#

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.gemv!),Char,P,AbstractMatrix{P},AbstractVector{P},P,AbstractVector{P}
    } where {P<:BlasRealFloat},
)

@inline function rrule!!(
    ::CoDual{typeof(BLAS.gemv!)},
    _tA::CoDual{Char},
    _alpha::CoDual{P},
    _A::CoDual{<:AbstractMatrix{P}},
    _x::CoDual{<:AbstractVector{P}},
    _beta::CoDual{P},
    _y::CoDual{<:AbstractVector{P}},
) where {P<:BlasRealFloat}

    # Pull out primals and tangents (the latter only where necessary).
    trans = _tA.x
    alpha = _alpha.x
    A, dA = arrayify(_A)
    x, dx = arrayify(_x)
    beta = _beta.x
    y, dy = arrayify(_y)

    # Take copies before adding.
    y_copy = copy(y)

    # Run primal.
    BLAS.gemv!(trans, alpha, A, x, beta, y)

    function gemv!_pb!!(::NoRData)

        # Increment fdata.
        if trans == 'N'
            dalpha = dot(dy, A, x)
            dA .+= alpha .* dy .* x'
            BLAS.gemv!('T', alpha, A, dy, one(eltype(A)), dx)
        else
            dalpha = dot(dy, A', x)
            dA .+= alpha .* x .* dy'
            BLAS.gemv!('N', alpha, A, dy, one(eltype(A)), dx)
        end
        dbeta = dot(y_copy, dy)
        dy .*= beta

        # Restore primal.
        copyto!(y, y_copy)

        # Return rdata.
        return NoRData(), NoRData(), dalpha, NoRData(), NoRData(), dbeta, NoRData()
    end

    return _y, gemv!_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.symv!),Char,T,AbstractMatrix{T},AbstractVector{T},T,AbstractVector{T}
    } where {T<:BlasRealFloat},
)

function rrule!!(
    ::CoDual{typeof(BLAS.symv!)},
    uplo::CoDual{Char},
    alpha::CoDual{T},
    A_dA::CoDual{<:AbstractMatrix{T}},
    x_dx::CoDual{<:AbstractVector{T}},
    beta::CoDual{T},
    y_dy::CoDual{<:AbstractVector{T}},
) where {T<:BlasRealFloat}

    # Extract primals.
    ul = primal(uplo)
    α = primal(alpha)
    β = primal(beta)
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)
    y, dy = arrayify(y_dy)

    # In this rule we optimise carefully for the special case a == 1 && b == 0, which
    # corresponds to simply multiplying symm(A) and x together, and writing the result to y.
    # This is an extremely common edge case, so it's important to do well for it.
    y_copy = copy(y)
    tmp_ref = Ref{Vector{T}}()
    if (α == 1 && β == 0)
        BLAS.symv!(ul, α, A, x, β, y)
    else
        tmp = BLAS.symv(ul, one(T), A, x)
        tmp_ref[] = tmp
        BLAS.axpby!(α, tmp, β, y)
    end

    function symv!_adjoint(::NoRData)
        if (α == 1 && β == 0)
            dα = dot(dy, y)
            BLAS.copyto!(y, y_copy)
        else
            # Reset y.
            BLAS.copyto!(y, y_copy)

            # gradient w.r.t. α. Safe to write into memory for copy of y.
            BLAS.symv!(ul, one(T), A, x, zero(T), y_copy)
            dα = dot(dy, y_copy)
        end

        # gradient w.r.t. A.
        dA_tmp = dy * x'
        if ul == 'L'
            dA .+= α .* LowerTriangular(dA_tmp)
            dA .+= α .* UpperTriangular(dA_tmp)'
        else
            dA .+= α .* LowerTriangular(dA_tmp)'
            dA .+= α .* UpperTriangular(dA_tmp)
        end
        @inbounds for n in diagind(dA)
            dA[n] -= α * dA_tmp[n]
        end

        # gradient w.r.t. x.
        BLAS.symv!(ul, α, A, dy, one(T), dx)

        # gradient w.r.t. beta.
        dβ = dot(dy, y)

        # gradient w.r.t. y.
        BLAS.scal!(β, dy)

        return NoRData(), NoRData(), dα, NoRData(), NoRData(), dβ, NoRData()
    end
    return y_dy, symv!_adjoint
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.trmv!),Char,Char,Char,AbstractMatrix{T},AbstractVector{T}
    } where {T<:BlasRealFloat},
)

function rrule!!(
    ::CoDual{typeof(BLAS.trmv!)},
    _uplo::CoDual{Char},
    _trans::CoDual{Char},
    _diag::CoDual{Char},
    A_dA::CoDual{<:AbstractMatrix{T}},
    x_dx::CoDual{<:AbstractVector{T}},
) where {T<:BlasRealFloat}

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
        trans == 'N' ? inc_tri!(dA, dx, x, uplo, diag) : inc_tri!(dA, x, dx, uplo, diag)
        BLAS.trmv!(uplo, trans == 'N' ? 'T' : 'N', diag, A, dx)

        return tuple_fill(NoRData(), Val(6))
    end
    return x_dx, trmv_pb!!
end

function inc_tri!(A, x, y, uplo, diag)
    if uplo == 'L' && diag == 'U'
        @inbounds for q in 1:size(A, 2), p in (q + 1):size(A, 1)
            A[p, q] = fma(x[p], y[q], A[p, q])
        end
    elseif uplo == 'L' && diag == 'N'
        @inbounds for q in 1:size(A, 2), p in q:size(A, 1)
            A[p, q] = fma(x[p], y[q], A[p, q])
        end
    elseif uplo == 'U' && diag == 'U'
        @inbounds for q in 1:size(A, 2), p in 1:(q - 1)
            A[p, q] = fma(x[p], y[q], A[p, q])
        end
    elseif uplo == 'U' && diag == 'N'
        @inbounds for q in 1:size(A, 2), p in 1:q
            A[p, q] = fma(x[p], y[q], A[p, q])
        end
    else
        error("Unexpected uplo $uplo or diag $diag")
    end
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
        AbstractMatrix{T},
        AbstractMatrix{T},
        T,
        AbstractMatrix{T},
    } where {T<:BlasRealFloat},
)

function rrule!!(
    ::CoDual{typeof(BLAS.gemm!)},
    transA::CoDual{Char},
    transB::CoDual{Char},
    alpha::CoDual{T},
    A::CoDual{<:AbstractMatrix{T}},
    B::CoDual{<:AbstractMatrix{T}},
    beta::CoDual{T},
    C::CoDual{<:AbstractMatrix{T}},
) where {T<:BlasRealFloat}
    tA = primal(transA)
    tB = primal(transB)
    a = primal(alpha)
    b = primal(beta)
    p_A, dA = arrayify(A)
    p_B, dB = arrayify(B)
    p_C, dC = arrayify(C)

    # In this rule we optimise carefully for the special case a == 1 && b == 0, which
    # corresponds to simply multiplying A and B together, and writing the result to C.
    # This is an extremely common edge case, so it's important to do well for it.
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

        # Compute pullback w.r.t. alpha.
        da = (a == 1 && b == 0) ? dot(dC, p_C) : dot(dC, tmp_ref[])

        # Restore previous state.
        BLAS.copyto!(p_C, p_C_copy)

        # Compute pullback w.r.t. beta.
        db = dot(dC, p_C)

        # Increment cotangents.
        if tA == 'N'
            BLAS.gemm!('N', tB == 'N' ? 'T' : 'N', a, dC, p_B, one(T), dA)
        else
            BLAS.gemm!(tB == 'N' ? 'N' : 'T', 'T', a, p_B, dC, one(T), dA)
        end
        if tB == 'N'
            BLAS.gemm!(tA == 'N' ? 'T' : 'N', 'N', a, p_A, dC, one(T), dB)
        else
            BLAS.gemm!('T', tA == 'N' ? 'N' : 'T', a, dC, p_A, one(T), dB)
        end
        dC .*= b

        return NoRData(), NoRData(), NoRData(), da, NoRData(), NoRData(), db, NoRData()
    end
    return C, gemm!_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.symm!),
        Char,
        Char,
        T,
        AbstractMatrix{T},
        AbstractMatrix{T},
        T,
        AbstractMatrix{T},
    } where {T<:BlasRealFloat},
)

function rrule!!(
    ::CoDual{typeof(BLAS.symm!)},
    side::CoDual{Char},
    uplo::CoDual{Char},
    alpha::CoDual{T},
    A_dA::CoDual{<:AbstractMatrix{T}},
    B_dB::CoDual{<:AbstractMatrix{T}},
    beta::CoDual{T},
    C_dC::CoDual{<:AbstractMatrix{T}},
) where {T<:BlasRealFloat}

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
        BLAS.symm!(s, ul, α, A, B, β, C)
    else
        tmp = BLAS.symm(s, ul, one(T), A, B)
        tmp_ref[] = tmp
        C .= α .* tmp .+ β .* C
    end

    function symm!_adjoint(::NoRData)
        if (α == 1 && β == 0)
            dα = dot(dC, C)
            BLAS.copyto!(C, C_copy)
        else
            # Reset C.
            BLAS.copyto!(C, C_copy)

            # gradient w.r.t. α. Safe to write into memory for copy of C.
            BLAS.symm!(s, ul, one(T), A, B, zero(T), C_copy)
            dα = dot(dC, C_copy)
        end

        # gradient w.r.t. A.
        dA_tmp = s == 'L' ? dC * B' : B' * dC
        if ul == 'L'
            dA .+= α .* LowerTriangular(dA_tmp)
            dA .+= α .* UpperTriangular(dA_tmp)'
        else
            dA .+= α .* LowerTriangular(dA_tmp)'
            dA .+= α .* UpperTriangular(dA_tmp)
        end
        @inbounds for n in diagind(dA)
            dA[n] -= α * dA_tmp[n]
        end

        # gradient w.r.t. B.
        BLAS.symm!(s, ul, α, A, dC, one(T), dB)

        # gradient w.r.t. beta.
        dβ = dot(dC, C)

        # gradient w.r.t. C.
        dC .*= β

        return NoRData(), NoRData(), NoRData(), dα, NoRData(), NoRData(), dβ, NoRData()
    end
    return C_dC, symm!_adjoint
end

for (syrk, elty) in ((:dsyrk_, :Float64), (:ssyrk_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(syrk))}},
        ::CoDual{Val{Cvoid}},
        ::CoDual, # arg types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        uplo::CoDual{Ptr{UInt8}},
        trans::CoDual{Ptr{UInt8}},
        n::CoDual{Ptr{BLAS.BlasInt}},
        k::CoDual{Ptr{BLAS.BlasInt}},
        alpha::CoDual{Ptr{$elty}},
        A::CoDual{Ptr{$elty}},
        LDA::CoDual{Ptr{BLAS.BlasInt}},
        beta::CoDual{Ptr{$elty}},
        C::CoDual{Ptr{$elty}},
        LDC::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,Nargs},
    ) where {Nargs}
        GC.@preserve args begin
            _uplo = Char(unsafe_load(primal(uplo)))
            _t = Char(unsafe_load(primal(trans)))
            _n = unsafe_load(primal(n))
            _k = unsafe_load(primal(k))
            _alpha = unsafe_load(primal(alpha))
            _A = primal(A)
            _LDA = unsafe_load(primal(LDA))
            _beta = unsafe_load(primal(beta))
            _C = primal(C)
            _LDC = unsafe_load(primal(LDC))

            A_mat = wrap_ptr_as_view(primal(A), _LDA, (_t == 'N' ? (_n, _k) : (_k, _n))...)
            C_mat = wrap_ptr_as_view(primal(C), _LDC, _n, _n)
            C_copy = collect(C_mat)

            BLAS.syrk!(_uplo, _t, _alpha, A_mat, _beta, C_mat)

            dalpha = tangent(alpha)
            dA = tangent(A)
            dbeta = tangent(beta)
            dC = tangent(C)
        end

        function syrk!_pullback!!(::NoRData)
            GC.@preserve args begin
                # Restore previous state.
                C_mat .= C_copy

                # Convert pointers to views.
                dA_mat = wrap_ptr_as_view(dA, _LDA, (_t == 'N' ? (_n, _k) : (_k, _n))...)
                dC_mat = wrap_ptr_as_view(dC, _LDC, _n, _n)

                # Increment cotangents.
                B = _uplo == 'U' ? triu(dC_mat) : tril(dC_mat)
                unsafe_store!(dbeta, unsafe_load(dbeta) + sum(B .* C_mat))
                dalpha_inc = tr(B' * _trans(_t, A_mat) * _trans(_t, A_mat)')
                unsafe_store!(dalpha, unsafe_load(dalpha) + dalpha_inc)
                dA_mat .+= _alpha * (_t == 'N' ? (B + B') * A_mat : A_mat * (B + B'))
                dC_mat .=
                    (_uplo == 'U' ? tril!(dC_mat, -1) : triu!(dC_mat, 1)) .+ _beta .* B
            end

            return tuple_fill(NoRData(), Val(16 + Nargs))
        end
        return zero_fcodual(Cvoid()), syrk!_pullback!!
    end
end

for (trmm, elty) in ((:dtrmm_, :Float64), (:strmm_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(trmm))}},
        ::CoDual,
        ::CoDual, # arg types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _side::CoDual{Ptr{UInt8}},
        _uplo::CoDual{Ptr{UInt8}},
        _trans::CoDual{Ptr{UInt8}},
        _diag::CoDual{Ptr{UInt8}},
        _M::CoDual{Ptr{BLAS.BlasInt}},
        _N::CoDual{Ptr{BLAS.BlasInt}},
        _alpha::CoDual{Ptr{$elty}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BLAS.BlasInt}},
        _B::CoDual{Ptr{$elty}},
        _ldb::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,Nargs},
    ) where {Nargs}
        GC.@preserve args begin

            # Load in data and store B for the reverse-pass.
            side, ul, tA, diag = map(
                Char ∘ unsafe_load ∘ primal, (_side, _uplo, _trans, _diag)
            )
            M, N, lda, ldb = map(unsafe_load ∘ primal, (_M, _N, _lda, _ldb))
            alpha = unsafe_load(primal(_alpha))
            R = side == 'L' ? M : N
            A = wrap_ptr_as_view(primal(_A), lda, R, R)
            B = wrap_ptr_as_view(primal(_B), ldb, M, N)
            B_copy = copy(B)

            # Run primal.
            BLAS.trmm!(side, ul, tA, diag, alpha, A, B)

            dalpha = tangent(_alpha)
            _dA = tangent(_A)
            _dB = tangent(_B)
        end

        function trmm!_pullback!!(::NoRData)
            GC.@preserve args begin
                # Convert pointers to views.
                dA = wrap_ptr_as_view(_dA, lda, R, R)
                dB = wrap_ptr_as_view(_dB, ldb, M, N)

                # Increment alpha tangent.
                alpha != 0 && unsafe_store!(dalpha, unsafe_load(dalpha) + tr(dB'B) / alpha)

                # Restore initial state.
                B .= B_copy

                # Increment cotangents.
                if side == 'L'
                    dA .+= alpha .* tri!(tA == 'N' ? dB * B' : B * dB', ul, diag)
                else
                    dA .+= alpha .* tri!(tA == 'N' ? B'dB : dB'B, ul, diag)
                end

                # Compute dB tangent.
                BLAS.trmm!(side, ul, tA == 'N' ? 'T' : 'N', diag, alpha, A, dB)
            end

            return tuple_fill(NoRData(), Val(17 + Nargs))
        end

        return zero_fcodual(Cvoid()), trmm!_pullback!!
    end
end

for (trsm, elty) in ((:dtrsm_, :Float64), (:strsm_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(trsm))}},
        ::CoDual,
        ::CoDual, # arg types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _side::CoDual{Ptr{UInt8}},
        _uplo::CoDual{Ptr{UInt8}},
        _trans::CoDual{Ptr{UInt8}},
        _diag::CoDual{Ptr{UInt8}},
        _M::CoDual{Ptr{BLAS.BlasInt}},
        _N::CoDual{Ptr{BLAS.BlasInt}},
        _alpha::CoDual{Ptr{$elty}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BLAS.BlasInt}},
        _B::CoDual{Ptr{$elty}},
        _ldb::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,Nargs},
    ) where {Nargs}
        GC.@preserve args begin
            side = Char(unsafe_load(primal(_side)))
            uplo = Char(unsafe_load(primal(_uplo)))
            trans = Char(unsafe_load(primal(_trans)))
            diag = Char(unsafe_load(primal(_diag)))
            M = unsafe_load(primal(_M))
            N = unsafe_load(primal(_N))
            R = side == 'L' ? M : N
            alpha = unsafe_load(primal(_alpha))
            lda = unsafe_load(primal(_lda))
            ldb = unsafe_load(primal(_ldb))
            A = wrap_ptr_as_view(primal(_A), lda, R, R)
            B = wrap_ptr_as_view(primal(_B), ldb, M, N)
            B_copy = copy(B)

            trsm!(side, uplo, trans, diag, alpha, A, B)

            dalpha = tangent(_alpha)
            _dA = tangent(_A)
            _dB = tangent(_B)
        end

        function trsm_pb!!(::NoRData)
            GC.@preserve args begin
                # Convert pointers to views.
                dA = wrap_ptr_as_view(_dA, lda, R, R)
                dB = wrap_ptr_as_view(_dB, ldb, M, N)

                # Increment alpha tangent.
                alpha != 0 && unsafe_store!(dalpha, unsafe_load(dalpha) + tr(dB'B) / alpha)

                # Increment cotangents.
                if side == 'L'
                    if trans == 'N'
                        tmp = trsm!('L', uplo, 'T', diag, -one($elty), A, dB * B')
                        dA .+= tri!(tmp, uplo, diag)
                    else
                        tmp = trsm!('R', uplo, 'T', diag, -one($elty), A, B * dB')
                        dA .+= tri!(tmp, uplo, diag)
                    end
                else
                    if trans == 'N'
                        tmp = trsm!('R', uplo, 'T', diag, -one($elty), A, B'dB)
                        dA .+= tri!(tmp, uplo, diag)
                    else
                        tmp = trsm!('L', uplo, 'T', diag, -one($elty), A, dB'B)
                        dA .+= tri!(tmp, uplo, diag)
                    end
                end

                # Restore initial state.
                B .= B_copy

                # Compute dB tangent.
                BLAS.trsm!(side, uplo, trans == 'N' ? 'T' : 'N', diag, alpha, A, dB)
            end

            return tuple_fill(NoRData(), Val(17 + Nargs))
        end
        return zero_fcodual(Cvoid()), trsm_pb!!
    end
end

function blas_matrices(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int, q::Int)
    Xs = Any[
        randn(rng, P, p, q),
        view(randn(rng, P, p + 5, 2q), 3:(p + 2), 1:2:(2q)),
        view(randn(rng, P, 3p, 3, 2q), (p + 1):(2p), 2, 1:2:(2q)),
        reshape(view(randn(rng, P, p * q + 5), 1:(p * q)), p, q),
    ]
    @assert all(X -> size(X) == (p, q), Xs)
    @assert all(Base.Fix2(isa, AbstractMatrix{P}), Xs)
    return Xs
end

function blas_vectors(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int)
    xs = Any[
        randn(rng, P, p),
        view(randn(rng, P, p + 5), 3:(p + 2)),
        view(randn(rng, P, 3p, 3), 1:2:(2p), 2),
        reshape(view(randn(rng, P, 1, p + 5), 1:1, 1:p), p),
    ]
    @assert all(x -> length(x) == p, xs)
    @assert all(Base.Fix2(isa, AbstractVector{P}), xs)
    return xs
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:blas})
    t_flags = ['N', 'T', 'C']
    alphas = [1.0, -0.25]
    betas = [0.0, 0.33]
    uplos = ['L', 'U']
    dAs = ['N', 'U']
    Ps = [Float64, Float32]
    rng = rng_ctor(123456)

    test_cases = vcat(
        # nrm2(x)
        map_prod([Ps..., ComplexF64, ComplexF32]) do (P,)
            return map([randn(rng, P, 105)]) do x
                (false, :none, nothing, BLAS.nrm2, x)
            end
        end...,

        # nrm2(n, x, incx)
        map_prod([Ps..., ComplexF64, ComplexF32], [5, 3], [1, 2]) do (P, n, incx)
            return map([randn(rng, P, 105)]) do x
                (false, :none, nothing, BLAS.nrm2, n, x, incx)
            end
        end...,

        # gemv!
        map_prod(t_flags, [1, 3], [1, 2], Ps) do (tA, M, N, P)
            As = blas_matrices(rng, P, tA == 'N' ? M : N, tA == 'N' ? N : M)
            xs = blas_vectors(rng, P, N)
            ys = blas_vectors(rng, P, M)
            flags = (false, :stability, (lb=1e-3, ub=10.0))
            return map(As, xs, ys) do A, x, y
                (flags..., BLAS.gemv!, tA, randn(rng, P), A, x, randn(rng, P), y)
            end
        end...,

        # symv!
        map_prod(['L', 'U'], alphas, betas, Ps) do (uplo, α, β, P)
            As = blas_matrices(rng, P, 5, 5)
            ys = blas_vectors(rng, P, 5)
            xs = blas_vectors(rng, P, 5)
            return map(As, xs, ys) do A, x, y
                (false, :stability, nothing, BLAS.symv!, uplo, P(α), A, x, P(β), y)
            end
        end...,

        # trmv!
        map_prod(uplos, t_flags, dAs, [1, 3], Ps) do (ul, tA, dA, N, P)
            As = blas_matrices(rng, P, N, N)
            bs = blas_vectors(rng, P, N)
            return map(As, bs) do A, b
                (false, :stability, nothing, BLAS.trmv!, ul, tA, dA, A, b)
            end
        end...,

        # gemm!
        map_prod(t_flags, t_flags, alphas, betas, Ps) do (tA, tB, a, b, P)
            As = blas_matrices(rng, P, tA == 'N' ? 3 : 4, tA == 'N' ? 4 : 3)
            Bs = blas_matrices(rng, P, tB == 'N' ? 4 : 5, tB == 'N' ? 5 : 4)
            Cs = blas_matrices(rng, P, 3, 5)
            return map(As, Bs, Cs) do A, B, C
                (false, :stability, nothing, BLAS.gemm!, tA, tB, P(a), A, B, P(b), C)
            end
        end...,

        # symm!
        map_prod(['L', 'R'], ['L', 'U'], alphas, betas, Ps) do (side, ul, α, β, P)
            nA = side == 'L' ? 5 : 7
            As = blas_matrices(rng, P, nA, nA)
            Bs = blas_matrices(rng, P, 5, 7)
            Cs = blas_matrices(rng, P, 5, 7)
            return map(As, Bs, Cs) do A, B, C
                (false, :stability, nothing, BLAS.symm!, side, ul, P(α), A, B, P(β), C)
            end
        end...,
    )

    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:blas})
    t_flags = ['N', 'T', 'C']
    aliased_gemm! = (tA, tB, a, b, A, C) -> BLAS.gemm!(tA, tB, a, A, A, b, C)
    Ps = [Float32, Float64]
    uplos = ['L', 'U']
    dAs = ['N', 'U']
    rng = rng_ctor(123)

    test_cases = vcat(

        # Utility
        (false, :stability, nothing, BLAS.get_num_threads),
        (false, :stability, nothing, BLAS.lbt_get_num_threads),
        (false, :stability, nothing, BLAS.set_num_threads, 1),
        (false, :stability, nothing, BLAS.lbt_set_num_threads, 1),

        #
        # BLAS LEVEL 1
        #

        map(Ps) do P
            flags = (false, :none, nothing)
            Any[
                (flags..., BLAS.dot, 3, randn(rng, P, 5), 1, randn(rng, P, 4), 1),
                (flags..., BLAS.dot, 3, randn(rng, P, 6), 2, randn(rng, P, 4), 1),
                (flags..., BLAS.dot, 3, randn(rng, P, 6), 1, randn(rng, P, 9), 3),
                (flags..., BLAS.dot, 3, randn(rng, P, 12), 3, randn(rng, P, 9), 2),
                (flags..., BLAS.scal!, 10, P(2.4), randn(rng, P, 30), 2),
            ]
        end...,

        #
        # BLAS LEVEL 3
        #

        # aliased gemm!
        map_prod(t_flags, t_flags, Ps) do (tA, tB, P)
            As = blas_matrices(rng, P, 5, 5)
            Bs = blas_matrices(rng, P, 5, 5)
            a = randn(rng, P)
            b = randn(rng, P)
            return map_prod(As, Bs) do (A, B)
                (false, :none, nothing, aliased_gemm!, tA, tB, a, b, A, B)
            end
        end...,

        # syrk!
        map_prod(uplos, t_flags, Ps) do (uplo, t, P)
            As = blas_matrices(rng, P, t == 'N' ? 3 : 4, t == 'N' ? 4 : 3)
            C = randn(rng, P, 3, 3)
            a = randn(rng, P)
            b = randn(rng, P)
            return map(As) do A
                (false, :none, nothing, BLAS.syrk!, uplo, t, a, A, b, C)
            end
        end...,

        # trmm!
        map_prod(
            ['L', 'R'], uplos, t_flags, dAs, [1, 3], [1, 2], Ps
        ) do (side, ul, tA, dA, M, N, P)
            t = tA == 'N'
            R = side == 'L' ? M : N
            a = randn(rng, P)
            As = blas_matrices(rng, P, R, R)
            Bs = blas_matrices(rng, P, M, N)
            return map(As, Bs) do A, B
                (false, :none, nothing, BLAS.trmm!, side, ul, tA, dA, a, A, B)
            end
        end...,

        # trsm!
        map_prod(
            ['L', 'R'], uplos, t_flags, dAs, [1, 3], [1, 2], Ps
        ) do (side, ul, tA, dA, M, N, P)
            t = tA == 'N'
            R = side == 'L' ? M : N
            a = randn(rng, P)
            As = map(blas_matrices(rng, P, R, R)) do A
                A[diagind(A)] .+= 1
                return A
            end
            Bs = blas_matrices(rng, P, M, N)
            return map(As, Bs) do A, B
                (false, :none, nothing, BLAS.trsm!, side, ul, tA, dA, a, A, B)
            end
        end...,
    )
    memory = Any[]
    return test_cases, memory
end
