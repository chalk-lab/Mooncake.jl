module MooncakeCUDAExt

using LinearAlgebra, Random, Mooncake

using Base: IEEEFloat
using CUDA:
    CuArray,
    CuRefValue,
    CuPtr,
    CuContext,
    CuStream,
    CUmemPoolHandle_st,
    CUdevice_attribute_enum,
    cudaError_enum,
    libraryPropertyType,
    TaskLocalState,
    task_local_state!,
    active_state,
    CuDevice,
    attribute,
    cuDeviceGetAttribute,
    DeviceMemory,
    UnifiedMemory,
    HostMemory
using CUDA: CUBLAS
using CUDA: CUSPARSE

import Mooncake:
    MinimalCtx,
    DefaultCtx,
    frule!!,
    rrule!!,
    @is_primitive,
    @unstable,
    @foldable,
    @from_rrule,
    @zero_derivative,
    tangent_type,
    fdata_type,
    rdata_type,
    primal,
    tangent,
    lgetfield,
    zero_fcodual,
    zero_tangent_internal,
    randn_tangent_internal,
    increment_internal!!,
    set_to_zero_internal!!,
    _add_to_primal_internal,
    tangent_to_primal_internal!!,
    primal_to_tangent_internal!!,
    _dot_internal,
    _scale_internal,
    _new_,
    TestUtils,
    Dual,
    CoDual,
    NoTangent,
    NoPullback,
    NoFData,
    to_cr_tangent,
    mooncake_tangent,
    increment_and_get_rdata!,
    MaybeCache,
    IncCache,
    NoRData,
    arrayify

import Mooncake.TestUtils:
    populate_address_map_internal, AddressMap, __increment_should_allocate

const CuFloatArray = CuArray{<:IEEEFloat}
const CuComplexArray = CuArray{<:Complex{<:IEEEFloat}}
const CuMaybeComplexArray = Union{CuFloatArray,CuComplexArray}

# CuArray{T,N,M}.data is a DataRef — a reference-counted handle to the GPU memory buffer.
# Operations like reshape and view reconstruct a CuArray from its components.
# CuDataRef expanded to cover DeviceMemory, UnifiedMemory, and HostMemory variants.
const CuDataRef = Union{
    fieldtype(CuArray{Float32,1,DeviceMemory}, :data),
    fieldtype(CuArray{Float32,1,UnifiedMemory}, :data),
    fieldtype(CuArray{Float32,1,HostMemory}, :data),
}

# DataRef values are mutable reference-counted handles; preserve them as-is for fdata.
@foldable tangent_type(::Type{P}) where {P<:CuDataRef} = P
@foldable tangent_type(::Type{P}, ::Type{NoRData}) where {P<:CuDataRef} = P
tangent(p::CuDataRef, ::NoRData) = p
Mooncake.__verify_fdata_value(::IdDict{Any,Nothing}, ::CuDataRef, ::CuDataRef) = nothing

@foldable fdata_type(::Type{CuPtr{T}}) where {T} = CuPtr{T}
@foldable rdata_type(::Type{CuPtr{T}}) where {T} = NoRData
@foldable tangent_type(::Type{P}) where {P<:CuMaybeComplexArray} = P
@foldable tangent_type(::Type{P}, ::Type{NoRData}) where {P<:CuMaybeComplexArray} = P
@unstable @foldable tangent_type(::Type{CuPtr{P}}) where {P} = CuPtr{tangent_type(P)}
@unstable @foldable tangent_type(::Type{CuRefValue{P}}) where {P} = CuRefValue{
    tangent_type(P)
}

# Non-differentiable CUDA handle, enum, and state types.
#
# Opaque pointer types (Ptr{X}): Mooncake's default tangent_type(::Type{Ptr{P}}) returns
# Ptr{tangent_type(P)}, and zero_tangent_internal(::Ptr, ::MaybeCache) throws
# unconditionally.  Both must be overridden for each concrete opaque pointer type.
#
# Plain struct/enum types: only tangent_type needs overriding (NoTangent stops traversal).
for (_cuda_opaque_t, _is_ptr) in [
    # --- opaque C handle/descriptor Ptr types (CUBLAS) ---
    (CUmemPoolHandle_st, true),
    (CUBLAS.cublasContext, true),
    (CUBLAS.cublasXtContext, true),
    # --- opaque C handle/descriptor Ptr types (CUSPARSE) ---
    (CUSPARSE.cusparseContext, true),
    (CUSPARSE.cusparseMatDescr, true),
    (CUSPARSE.bsrsv2Info, true),
    (CUSPARSE.bsrsm2Info, true),
    (CUSPARSE.csric02Info, true),
    (CUSPARSE.bsric02Info, true),
    (CUSPARSE.csrilu02Info, true),
    (CUSPARSE.bsrilu02Info, true),
    (CUSPARSE.csru2csrInfo, true),
    (CUSPARSE.cusparseColorInfo, true),
    (CUSPARSE.pruneInfo, true),
    (CUSPARSE.cusparseSpVecDescr, true),
    (CUSPARSE.cusparseDnVecDescr, true),
    (CUSPARSE.cusparseSpMatDescr, true),
    (CUSPARSE.cusparseDnMatDescr, true),
    (CUSPARSE.cusparseSpSVDescr, true),
    (CUSPARSE.cusparseSpSMDescr, true),
    (CUSPARSE.cusparseSpGEMMDescr, true),
    (CUSPARSE.cusparseSpMMOpPlan, true),
    # --- CUBLAS enum/status types ---
    (CUBLAS.cublasOperation_t, false),
    (CUBLAS.cublasComputeType_t, false),
    (CUBLAS.cublasStatus_t, false),
    (CUBLAS.cublasGemmAlgo_t, false),
    # --- other non-differentiable CUDA types ---
    (libraryPropertyType, false),
    (CUdevice_attribute_enum, false),
    (cudaError_enum, false),
    # CuStream contains Ptr/Bool/CuContext fields; without NoTangent Mooncake generates a
    # MutableTangent that propagates into task-local CUDA state → SIGILL at runtime.
    (CuStream, false),
    # TaskLocalState bundles device index, stream handles, and library contexts — all
    # non-differentiable CUDA runtime state.
    (TaskLocalState, false),
    # CuContext wraps an opaque Ptr{Cvoid} to the CUDA context — no differentiable content.
    (CuContext, false),
]
    if _is_ptr
        @eval tangent_type(::Type{Ptr{$_cuda_opaque_t}}) = NoTangent
        @eval zero_tangent_internal(::Ptr{$_cuda_opaque_t}, ::MaybeCache) = NoTangent()
    else
        @eval tangent_type(::Type{$_cuda_opaque_t}) = NoTangent
    end
end

# CUDA runtime state functions — non-differentiable, must be registered as primitives.
@zero_derivative MinimalCtx Tuple{typeof(task_local_state!)}
@zero_derivative MinimalCtx Tuple{typeof(active_state)}
@zero_derivative MinimalCtx Tuple{typeof(CUBLAS.version)}
@zero_derivative MinimalCtx Tuple{typeof(CUBLAS.handle)}
@zero_derivative MinimalCtx Tuple{typeof(CUSPARSE.handle)}
@zero_derivative MinimalCtx Tuple{
    typeof(cuDeviceGetAttribute),Base.RefValue{Int32},CUdevice_attribute_enum,CuDevice
}
@zero_derivative MinimalCtx Tuple{typeof(attribute),CuDevice,CUdevice_attribute_enum}

# CuArray{<:Integer} and CuArray{<:Bool} are index/mask arrays — not differentiable.
tangent_type(::Type{<:CuArray{<:Union{Integer,Bool}}}) = NoTangent
tangent_type(::Type{<:CuArray{<:Union{Integer,Bool}}}, ::Type{NoRData}) = NoTangent

tangent(p::CuMaybeComplexArray, ::NoRData) = p

function arrayify(x::A, dx::A) where {A<:CuMaybeComplexArray}
    (x, dx)
end

function zero_tangent_internal(x::CuMaybeComplexArray, dict::MaybeCache)
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = zero(x)
    dict[x] = t
    return t
end
function randn_tangent_internal(rng::AbstractRNG, x::CuMaybeComplexArray, dict::MaybeCache)
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = CuArray(randn(rng, eltype(x), size(x)...))
    dict[x] = t
    return t
end
function TestUtils.has_equal_data_internal(
    x::P, y::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P<:CuMaybeComplexArray}
    # allow nan comparisons to return true, real() to cover complex case
    return isapprox(x, y; atol=(√eps(real(eltype(P)))), nans=true)
end
function increment_internal!!(c::IncCache, x::A, y::A) where {A<:CuMaybeComplexArray}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x .+= y
    return x
end
__increment_should_allocate(::Type{<:CuMaybeComplexArray}) = true
set_to_zero_internal!!(::Mooncake.SetToZeroCache, x::CuMaybeComplexArray) = x .= 0

function _add_to_primal_internal(
    c::MaybeCache, x::P, y::P, unsafe::Bool
) where {P<:CuMaybeComplexArray}
    key = (x, y, unsafe)
    haskey(c, key) && return c[key]::P
    x′ = x + y
    c[(x, y, unsafe)] = x′
    return x′
end
function primal_to_tangent_internal!!(t, x::CuMaybeComplexArray, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(t)
    c[x] = t
    t .= x
    return t
end
function tangent_to_primal_internal!!(x::CuMaybeComplexArray, t, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(x)
    c[x] = x
    x .= t
    return x
end
function _dot_internal(c::MaybeCache, x::P, y::P) where {P<:CuMaybeComplexArray}
    key = (x, y)
    haskey(c, key) && return c[key]::Float64
    return Float64(real(dot(x, y)))
end
function _scale_internal(c::MaybeCache, x::Float64, y::P) where {P<:CuMaybeComplexArray}
    haskey(c, y) && return c[y]::P
    t′ = eltype(P)(x) * y
    c[y] = t′
    return t′
end
function populate_address_map_internal(m::AddressMap, p::CuArray, t::CuArray)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    return m
end
function Mooncake.__verify_fdata_value(::IdDict{Any,Nothing}, p::CuArray, f::CuArray)
    if size(p) != size(f)
        throw(InvalidFDataException("p has size $(size(p)) but f has size $(size(f))"))
    end
    return nothing
end

# @from_chainrules tools
# CuArray is its own tangent type and ChainRules also uses CuArray directly as the
# tangent, so mooncake_tangent is the identity.
mooncake_tangent(::CuMaybeComplexArray, t::CuMaybeComplexArray) = t
to_cr_tangent(x::CuMaybeComplexArray) = x
function increment_and_get_rdata!(f::T, ::NoRData, t::T) where {T<:CuMaybeComplexArray}
    f .+= t
    return NoRData()
end

# Basic rules for operating on CuArrays.

# Primitive (not _new_) because GPU allocation happens inside the constructor body before
# the `new` call; tracing through it would hit CUDA-internal machinery.
@zero_derivative MinimalCtx Tuple{Type{<:CuArray},UndefInitializer,NTuple{N,Int}} where {N}

# Primitive because CUDA.jl's reshape body calls copy(DataRef) for reference counting,
# which uses llvmcall. reshape returns a view, so the tangent is a reshaped view of
# x.dx and gradient accumulation propagates automatically — NoPullback is correct.
@is_primitive(
    MinimalCtx, Tuple{typeof(reshape),CuMaybeComplexArray,NTuple{N,Int}} where {N},
)
function frule!!(
    ::Dual{typeof(reshape)}, x::Dual{<:CuMaybeComplexArray}, dims::Dual{<:NTuple}
)
    return Dual(reshape(primal(x), primal(dims)), reshape(tangent(x), primal(dims)))
end
function rrule!!(
    ::CoDual{typeof(reshape)}, x::CoDual{<:CuMaybeComplexArray}, dims::CoDual{<:NTuple}
)
    _dims = primal(dims)
    return CoDual(reshape(primal(x), _dims), reshape(x.dx, _dims)),
    NoPullback(ntuple(_ -> NoRData(), 3))
end

# `_new_` rules for the DataRef-based inner CuArray constructor (used by views and
# similar operations). The tangent reuses the DataRef from the input tangent so that
# gradient accumulation propagates automatically.
function frule!!(
    ::Dual{typeof(_new_)},
    ::Dual{Type{P}},
    data::Dual,
    maxsize::Dual,
    offset::Dual,
    dims::Dual,
) where {P<:CuMaybeComplexArray}
    y = _new_(P, primal(data), primal(maxsize), primal(offset), primal(dims))
    dy = _new_(P, tangent(data), primal(maxsize), primal(offset), primal(dims))
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(_new_)},
    ::CoDual{Type{P}},
    data::CoDual,
    maxsize::CoDual,
    offset::CoDual,
    dims::CoDual,
) where {P<:CuMaybeComplexArray}
    y = _new_(P, primal(data), primal(maxsize), primal(offset), primal(dims))
    dy = _new_(P, data.dx, primal(maxsize), primal(offset), primal(dims))
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 6))
end

# getfield / lgetfield rules for CuArray.
# CuArray has 4 fields: data (1, differentiable DataRef), maxsize (2), offset (3), dims (4).
function frule!!(
    ::Dual{typeof(lgetfield)},
    x::Dual{<:CuArray,<:CuArray},
    ::Dual{Val{name}},
    ::Dual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    is_data = name === 1 || name === :data
    dy = is_data ? tangent(x).data : NoTangent()
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:CuArray,<:CuArray},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    is_data = name === 1 || name === :data
    dy = is_data ? x.dx.data : NoFData()
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{<:CuArray,<:CuArray}, ::Dual{Val{name}}
) where {name}
    y = getfield(primal(x), name)
    is_data = name === 1 || name === :data
    dy = is_data ? tangent(x).data : NoTangent()
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{<:CuArray,<:CuArray}, ::CoDual{Val{name}}
) where {name}
    y = getfield(primal(x), name)
    is_data = name === 1 || name === :data
    dy = is_data ? x.dx.data : NoFData()
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 3))
end

# Rule for `sum` is defined as a performance rule.
# See also `src/rules/performance_patches`.
@is_primitive(DefaultCtx, Tuple{typeof(sum),CuFloatArray})
function frule!!(::Dual{typeof(sum)}, x::Dual{<:CuFloatArray})
    px, dx = arrayify(x)
    return Dual(sum(px), sum(dx))
end
function rrule!!(::CoDual{typeof(sum)}, x::CoDual{<:CuFloatArray})
    _, dx = arrayify(x)
    function sum_pb!!(dz)
        dx .+= dz
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(primal(x))), sum_pb!!
end

# Rules for the 3-arg mul!(C, A, B) on GPU arrays.
@is_primitive(
    MinimalCtx,
    Tuple{typeof(mul!),<:CuMaybeComplexArray,<:CuMaybeComplexArray,<:CuMaybeComplexArray},
)
function frule!!(
    ::Dual{typeof(mul!)},
    C::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    A::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    B::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
)
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    pC, dC = arrayify(C)
    mul!(dC, dA, pB)             # dC  = dA*B   (product rule; overwrites since β=0)
    mul!(dC, pA, dB, true, true) # dC += A*dB
    mul!(pC, pA, pB)
    return C
end
function rrule!!(
    ::CoDual{typeof(mul!)},
    C::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    A::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    B::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
)
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    pC, dC = arrayify(C)
    pC_copy = copy(pC)
    mul!(pC, pA, pB)
    function mul!_pb!!(::NoRData)
        mul!(dA, dC, pB', true, true)  # dA += dC * B^H
        mul!(dB, pA', dC, true, true)  # dB += A^H * dC
        copyto!(pC, pC_copy)           # restore C's primal (it was overwritten above)
        dC .= 0                        # β=0: C_old does not contribute, so ∂L/∂C_old = 0
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return C, mul!_pb!!
end

end
