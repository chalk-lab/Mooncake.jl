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
    CuArrayStyle,
    CUdevice_attribute_enum,
    cu,
    TaskLocalState,
    task_local_state!,
    active_state,
    CuDevice,
    attribute,
    cuDeviceGetAttribute,
    DeviceMemory,
    UnifiedMemory,
    HostMemory,
    is_capturing,
    capture_status
using CUDA: CUBLAS
using CUDA: CUSPARSE
using CUDA: CUSOLVER
using Base.Broadcast: Broadcasted
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
    FData,
    Tangent,
    to_cr_tangent,
    mooncake_tangent,
    increment_and_get_rdata!,
    MaybeCache,
    IncCache,
    NoRData,
    arrayify,
    _fields,
    zero_rdata,
    RData

import Mooncake.TestUtils:
    populate_address_map_internal, AddressMap, __increment_should_allocate

include("ndual.jl")

const CuFloatArray = CuArray{<:IEEEFloat}
const CuComplexArray = CuArray{<:Complex{<:IEEEFloat}}
const CuMaybeComplexArray = Union{CuFloatArray,CuComplexArray}
const CuFloatOrComplex = Union{IEEEFloat,Complex{<:IEEEFloat}}
# CuArray{T,N,M}.data is a DataRef — a reference-counted handle to the GPU memory buffer.
# Operations like reshape and view reconstruct a CuArray from its components:
#   `y = _new_(typeof(y), getfield(x, :data), getfield(x, :maxsize), getfield(x, :offset), dims)`.
# The tangent of data flows through these _new_ calls, so Mooncake needs lgetfield and
# _new_ rules for DataRef.
#
# CuArray{T,N,M} uses a different DataRef concrete type for each memory kind M:
#   DeviceMemory  → DataRef{Managed{DeviceMemory}}
#   UnifiedMemory → DataRef{Managed{UnifiedMemory}}
#   HostMemory    → DataRef{Managed{HostMemory}}
# DataRef does NOT depend on T or N — only on M — so three entries cover every
# CuArray{T,N,M} combination.  Missing a variant causes Mooncake to fall through to the
# generic struct handler, which tries to build tangents for DataRef's internal Ptr fields.
const CuDataRef = Union{
    fieldtype(CuArray{Float32,1,DeviceMemory}, :data),   # DataRef{Managed{DeviceMemory}}
    fieldtype(CuArray{Float32,1,UnifiedMemory}, :data),   # DataRef{Managed{UnifiedMemory}}
    fieldtype(CuArray{Float32,1,HostMemory}, :data),   # DataRef{Managed{HostMemory}}
}

# DataRef is treated as an opaque handle: its tangent type is DataRef itself.
# The three fields (:rc, :freed, :cached) are reference-counting internals — not
# differentiable.  lgetfield rules return NoTangent/NoFData for all field accesses.
@foldable tangent_type(::Type{P}) where {P<:CuDataRef} = P
@foldable tangent_type(::Type{P}, ::Type{NoRData}) where {P<:CuDataRef} = P
tangent(p::CuDataRef, ::NoRData) = p
Mooncake.__verify_fdata_value(::IdDict{Any,Nothing}, ::CuDataRef, ::CuDataRef) = nothing

# CuPtr and CuArray tangent types.
# CuPtr carries no differentiable content (it's a device address), so rdata is NoRData.
# CuMaybeComplexArray (float/complex GPU arrays) is its own tangent — gradient arrays
# have the same shape and element type as the primal.

# For CuPtr{T}: if T has no differentiable content (tangent_type(T) = NoTangent) then the
# pointer itself carries no gradient — e.g. CuPtr{Nothing} is a raw void pointer used only
# for memory management.  For differentiable T (e.g. Float32) the CuPtr IS the fdata
# (pointing to the tangent buffer on-device), so fdata = primal CuPtr.
@unstable @foldable tangent_type(::Type{CuPtr{P}}) where {P} =
    tangent_type(P) === NoTangent ? NoTangent : CuPtr{tangent_type(P)}
@foldable fdata_type(::Type{CuPtr{T}}) where {T} =
    tangent_type(T) === NoTangent ? NoFData : CuPtr{T}
@foldable rdata_type(::Type{CuPtr{T}}) where {T} = NoRData
@foldable tangent_type(::Type{P}) where {P<:CuMaybeComplexArray} = P
@foldable tangent_type(::Type{P}, ::Type{NoRData}) where {P<:CuMaybeComplexArray} = P
@unstable @foldable tangent_type(::Type{CuRefValue{P}}) where {P} = CuRefValue{
    tangent_type(P)
}

# CuPtr{T} wraps a device address (an integer).  The generic zero_tangent_internal for
# immutable structs does not apply here — construct a null device pointer directly.
function zero_tangent_internal(x::CuPtr{T}, ::MaybeCache) where {T}
    tangent_type(T) === NoTangent && return NoTangent()
    CuPtr{tangent_type(T)}(UInt64(0))
end

# Non-differentiable CUDA handle, enum, and state types.
#
# Opaque pointer types (Ptr{X}): Mooncake's default tangent_type(::Type{Ptr{P}}) returns
# Ptr{tangent_type(P)}, and zero_tangent_internal(::Ptr, ::MaybeCache) throws
# unconditionally.  Both must be overridden for each concrete opaque pointer type.
#
# Only the non-primitive opaque C pointer types need explicit registration here; all
# @cenum (primitive) types are handled by the programmatic loop further below.
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
    # CuStream contains Ptr/Bool/CuContext fields; without NoTangent, Mooncake generates a
    # MutableTangent that propagates into task-local CUDA state → SIGILL at runtime.
    (CuStream, false),
    # TaskLocalState bundles device index, stream handles, and library contexts — all
    # non-differentiable CUDA runtime state.
    (TaskLocalState, false),
    # CuContext wraps an opaque Ptr{Cvoid} to the CUDA context — no differentiable content.
    (CuContext, false),
    # --- opaque C handle/descriptor Ptr types (CUSOLVER) ---
    (CUSOLVER.syevjInfo_t, true),
    (CUSOLVER.gesvdjInfo_t, true),
    (CUSOLVER.cusolverDnIRSParams_t, true),
    (CUSOLVER.cusolverDnIRSInfos_t, true),
    (CUSOLVER.cusolverDnParams_t, true),
]
    if _is_ptr
        @eval tangent_type(::Type{Ptr{$_cuda_opaque_t}}) = NoTangent
        @eval zero_tangent_internal(::Ptr{$_cuda_opaque_t}, ::MaybeCache) = NoTangent()
    else
        @eval tangent_type(::Type{$_cuda_opaque_t}) = NoTangent
    end
end

# CUDA @cenum types are primitive types (integer-backed C enums) — never differentiable.
# Mooncake's generic tangent_type @generated function errors on primitive types with no
# registered method, so we register all of them here programmatically.
# Filter: parentmodule(T) must be one of the CUDA family modules, to avoid accidentally
# re-registering standard Julia primitive types (Bool, Int32, Float64, ...) that happen
# to be visible in the CUDA namespace.
let _cuda_family = (parentmodule(CUBLAS), CUBLAS, CUSPARSE, CUSOLVER)
    _cenum_seen = Set{DataType}()
    for _mod in _cuda_family
        for _nm in names(_mod; all=true)
            _T = try
                getfield(_mod, _nm)
            catch
                nothing
            end
            _T isa DataType || continue
            isprimitivetype(_T) || continue
            parentmodule(_T) in _cuda_family || continue
            _T in _cenum_seen && continue
            push!(_cenum_seen, _T)
            # tangent_type throws for unregistered primitives — catch means not yet registered
            (
                try
                    tangent_type(_T) === NoTangent
                catch
                    false
                end
            ) && continue
            @eval tangent_type(::Type{$_T}) = NoTangent
        end
    end
end

# Concrete field types of each CuDataRef (e.g. RefCounted, Managed, ...) are also
# non-differentiable memory-management internals.  Without this, Mooncake infers
# MutableTangent for them structurally, conflicting with the NoFData our lgetfield rules
# return and causing a TypeError typeassert at runtime.  We recurse into each registered
# type's fields to catch arbitrarily nested mutable structs (e.g. Managed inside
# RefCounted).
#
# _seen is pre-seeded with the CuDataRef root types — those are already registered with
# tangent_type = P (opaque/self) above, so must not be overwritten with NoTangent here.
# The tangent_type(T) === NoTangent guard additionally skips types already registered by
# the main opaque-types loop (e.g. CuStream), preventing duplicate-method errors.
let _seen = Set{DataType}(Base.uniontypes(CuDataRef))
    function _register_cuda_internal!(T)
        T isa DataType || return nothing
        T ∈ _seen && return nothing
        push!(_seen, T)
        isconcretetype(T) && ismutabletype(T) || return nothing
        already_registered = try
            tangent_type(T) === NoTangent
        catch
            false
        end
        already_registered && return nothing
        @eval tangent_type(::Type{$T}) = NoTangent
        @eval tangent_type(::Type{$T}, ::Type{NoRData}) = NoTangent
        for _i in 1:fieldcount(T)
            _register_cuda_internal!(fieldtype(T, _i))
        end
    end
    for _T in Base.uniontypes(CuDataRef)
        for _i in 1:fieldcount(_T)
            _register_cuda_internal!(fieldtype(_T, _i))
        end
    end
end

# CUDA runtime state functions — non-differentiable, must be registered as primitives.
# Without this, Mooncake's forward-mode interpreter traces into CUDA's task-local-storage
# machinery.  Those internals contain type assertions on the concrete stored types; when
# called with Dual-wrapped arguments the assertions fail, producing `Unreachable` in
# generated IR → SIGILL at runtime.
#
# task_local_state!() is the root entry point: all library handle() functions and
# active_state() call it to retrieve the per-task device/context/stream state.
@zero_derivative MinimalCtx Tuple{typeof(task_local_state!)}
# active_state() wraps task_local_state!() and returns a NamedTuple{device,context,stream,
# math_mode}.  Registering it separately covers call sites that bypass task_local_state!.
@zero_derivative MinimalCtx Tuple{typeof(active_state)}
# CUBLAS.version() queries the runtime library version via cublasGetProperty (a ccall).
# Returns a constant VersionNumber — not differentiable.
@zero_derivative MinimalCtx Tuple{typeof(CUBLAS.version)}
# Library handle() functions retrieve per-task C pointers to CUBLAS/CUSPARSE contexts.
@zero_derivative MinimalCtx Tuple{typeof(CUBLAS.handle)}
@zero_derivative MinimalCtx Tuple{typeof(CUSPARSE.handle)}
# cuDeviceGetAttribute queries a static integer device property (e.g. warp size, max
# threads per block).  Returns an Int — not differentiable.  Signature matches the
# internal call: cuDeviceGetAttribute(Ref{Cint}(), attrib, dev) from CUDA.attribute.
@zero_derivative MinimalCtx Tuple{
    typeof(cuDeviceGetAttribute),Base.RefValue{Int32},CUdevice_attribute_enum,CuDevice
}
# attribute() is the public wrapper around cuDeviceGetAttribute; registering it avoids
# tracing into the ccall at call sites that use the high-level API.
@zero_derivative MinimalCtx Tuple{typeof(attribute),CuDevice,CUdevice_attribute_enum}
# is_capturing / capture_status query whether the current stream is being graph-captured.
# They create Ref{CUstreamCaptureStatus_enum}() locally for a ccall output parameter.
# Without these rules, Mooncake traces into them and attempts to compute
# tangent_type(CUstreamCaptureStatus_enum), which fails for primitive types with no
# registered method.  Registering @cenum types above handles the type-level issue, but
# these @zero_derivative rules additionally avoid any tracing overhead.
@zero_derivative MinimalCtx Tuple{typeof(is_capturing)}
@zero_derivative MinimalCtx Tuple{typeof(is_capturing),CuStream}
@zero_derivative MinimalCtx Tuple{typeof(capture_status)}
@zero_derivative MinimalCtx Tuple{typeof(capture_status),CuStream}
# CuArray{<:Integer} and CuArray{<:Bool} are index/mask arrays — not differentiable.
# Assigning NoTangent stops Mooncake from building a struct tangent from CuArray's
# internal fields (data::CuDataRef, maxsize::Int, offset::Int, dims::NTuple).
# The CuMaybeComplexArray rule above takes priority for float and complex arrays.
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
function TestUtils.has_equal_data_internal(
    x::P, y::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P<:CuArray{<:Union{Integer,Bool}}}
    # For integer/bool CuArrays, compare by content by downloading to CPU.
    size(x) != size(y) && return false
    return Array(x) == Array(y)
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
    c[key] = x′
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

# ChainRules interop.  CuArray is its own tangent in both Mooncake and ChainRules,
# so to_cr_tangent and mooncake_tangent are identity operations.
mooncake_tangent(::CuMaybeComplexArray, t::CuMaybeComplexArray) = t
to_cr_tangent(x::CuMaybeComplexArray) = x
function increment_and_get_rdata!(f::T, ::NoRData, t::T) where {T<:CuMaybeComplexArray}
    f .+= t
    return NoRData()
end

# CuArray construction and reshape.

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

# lgetfield rules for CuArray.  CuArray has 4 fields:
#   :data (field 1) — the DataRef handle; tangent flows here
#   :maxsize (field 2), :offset (field 3), :dims (field 4) — non-differentiable metadata
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
