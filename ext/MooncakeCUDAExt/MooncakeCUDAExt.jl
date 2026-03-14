module MooncakeCUDAExt

using LinearAlgebra, Random, Mooncake

using Base: IEEEFloat, unsafe_convert
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
    matrixify,
    _fields,
    zero_rdata,
    RData,
    nan_tangent_guard

import Mooncake.TestUtils:
    populate_address_map_internal, AddressMap, __increment_should_allocate

include("ndual.jl")

const CuFloatArray = CuArray{<:IEEEFloat}
const CuComplexArray = CuArray{<:Complex{<:IEEEFloat}}
const CuMaybeComplexArray = Union{CuFloatArray,CuComplexArray}
const CuMaybeComplexVec = Union{CuArray{<:IEEEFloat,1},CuArray{<:Complex{<:IEEEFloat},1}}
const CuMaybeComplexMat = Union{CuArray{<:IEEEFloat,2},CuArray{<:Complex{<:IEEEFloat},2}}
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

# copy(::CuDataRef) is called by GPUArrays.derive (which backs view, reinterpret, and
# similar operations) for reference-count management.  It is a bookkeeping operation —
# the primal copy increments the refcount; the tangent DataRef is also copied so that the
# new CuArray's .data field holds a separate handle to the same tangent GPU memory.
@is_primitive(MinimalCtx, Tuple{typeof(copy),<:CuDataRef})
function frule!!(::Dual{typeof(copy)}, x::Dual{<:CuDataRef,<:CuDataRef})
    return Dual(copy(primal(x)), copy(tangent(x)))
end
function rrule!!(::CoDual{typeof(copy)}, x::CoDual{<:CuDataRef,<:CuDataRef})
    return CoDual(copy(primal(x)), copy(tangent(x))), NoPullback(ntuple(_ -> NoRData(), 2))
end

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

# unsafe_convert(::Type{CuPtr{T}}, x::CuArray{T}):
# Returns a raw device pointer to x's data buffer.  For AD, the fdata of the returned
# CuPtr is the pointer to the tangent buffer — both primal and tangent CuArrays have
# the same layout, so unsafe_convert on the tangent gives the correct tangent pointer.
# Needed because the traced body accesses DataRef internals (llvmcall) and loses the
# tangent, causing a CoDual{CuPtr{T}, CuPtr{T}} ← CoDual{CuPtr{T}, NoFData} TypeError.
@is_primitive(
    MinimalCtx,
    Tuple{typeof(unsafe_convert),Type{CuPtr{T}},CuArray{T}} where {T<:IEEEFloat},
)
@is_primitive(
    MinimalCtx,
    Tuple{typeof(unsafe_convert),Type{CuPtr{T}},CuArray{T}} where {T<:Complex{<:IEEEFloat}},
)
function frule!!(
    ::Dual{typeof(unsafe_convert)}, ::Dual{Type{CuPtr{T}}}, x::Dual{CuArray{T},CuArray{T}}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    return Dual(unsafe_convert(CuPtr{T}, primal(x)), unsafe_convert(CuPtr{T}, tangent(x)))
end
function rrule!!(
    ::CoDual{typeof(unsafe_convert)},
    ::CoDual{Type{CuPtr{T}}},
    x::CoDual{CuArray{T},CuArray{T}},
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    return CoDual(unsafe_convert(CuPtr{T}, primal(x)), unsafe_convert(CuPtr{T}, x.dx)),
    NoPullback(ntuple(_ -> NoRData(), 3))
end

# CuPtr arithmetic: (p::CuPtr{T}) + (n::Integer) offsets a device pointer by n bytes.
# For differentiable T the tangent is also a CuPtr; it must be offset by the same amount
# since primal and tangent arrays are laid out identically.
# For non-differentiable T (e.g. CuPtr{Cvoid} used in memory management), the tangent
# is NoTangent and the pointer arithmetic carries no gradient.
@is_primitive(MinimalCtx, Tuple{typeof(+),CuPtr{T},Integer} where {T})
function frule!!(
    ::Dual{typeof(+)}, p::Dual{CuPtr{T},CuPtr{T}}, n::Dual{<:Integer,NoTangent}
) where {T}
    return Dual(primal(p) + primal(n), tangent(p) + primal(n))
end
function frule!!(
    ::Dual{typeof(+)}, p::Dual{CuPtr{T},NoTangent}, n::Dual{<:Integer,NoTangent}
) where {T}
    return Dual(primal(p) + primal(n), NoTangent())
end
function rrule!!(
    ::CoDual{typeof(+)}, p::CoDual{CuPtr{T},CuPtr{T}}, n::CoDual{<:Integer,NoFData}
) where {T}
    return CoDual(primal(p) + primal(n), tangent(p) + primal(n)),
    NoPullback(ntuple(_ -> NoRData(), 3))
end
function rrule!!(
    ::CoDual{typeof(+)}, p::CoDual{CuPtr{T},NoFData}, n::CoDual{<:Integer,NoFData}
) where {T}
    return CoDual(primal(p) + primal(n), NoFData()), NoPullback(ntuple(_ -> NoRData(), 3))
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
# Base.mightalias(A::CuArray, B::CuArray) checks whether two GPU arrays share memory.
# It is called internally by copyto!.  Without this rule, forward-mode tracing enters
# mightalias's body where it accesses DataRef fields: our lgetfield rule returns NoFData
# for those, but Mooncake may infer MutableTangent for the inner RefCounted struct,
# causing a tangent type mismatch.
@zero_derivative MinimalCtx Tuple{typeof(Base.mightalias),T,S} where {T<:CuArray,S<:CuArray}
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
    # Use `similar` to match the memory kind of `x` (DeviceMemory, UnifiedMemory, or
    # HostMemory), then populate from a CPU-side randn so we don't need a GPU RNG.
    t = copyto!(similar(x), randn(rng, eltype(x), size(x)...))
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

# lgetfield rules for DataRef.  DataRef has three fields: :rc (ref count Atomic{Int}),
# :freed (Bool), :cached (the wrapped memory object, e.g. Managed{DeviceMemory}).
# All are reference-counting internals — no derivative flows through them.
# tangent_type(DataRef) = DataRef (opaque handle), so the tangent is the DataRef itself;
# field accesses return NoTangent/NoFData.
function frule!!(
    ::Dual{typeof(lgetfield)},
    x::Dual{<:CuDataRef,<:CuDataRef},
    ::Dual{Val{name}},
    ::Dual{Val{order}},
) where {name,order}
    return Dual(getfield(primal(x), name, order), NoTangent())
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:CuDataRef,<:CuDataRef},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    return CoDual(getfield(primal(x), name, order), NoFData()),
    NoPullback(ntuple(_ -> NoRData(), 4))
end
function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{<:CuDataRef,<:CuDataRef}, ::Dual{Val{name}}
) where {name}
    return Dual(getfield(primal(x), name), NoTangent())
end
function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{<:CuDataRef,<:CuDataRef}, ::CoDual{Val{name}}
) where {name}
    return CoDual(getfield(primal(x), name), NoFData()),
    NoPullback(ntuple(_ -> NoRData(), 3))
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

# Scalar indexing on CuArrays (e.g. x[1]) requires device→host round-trips and is
# disallowed by CUDA.jl by default.  Give a clear AD error rather than a cryptic one.
const _SCALAR_IDX_MSG =
    "Mooncake: scalar indexing of CuArray is not differentiable. " *
    "Rewrite using vectorised indexing (e.g. x[idx] with idx::AbstractVector) or " *
    "broadcasting. Add a new rule or open an issue at " *
    "https://github.com/chalk-lab/Mooncake.jl."
@is_primitive(MinimalCtx, Tuple{typeof(getindex),CuArray,Integer})
function frule!!(::Dual{typeof(getindex)}, x::Dual{<:CuArray}, i::Dual{<:Integer})
    throw(ArgumentError(_SCALAR_IDX_MSG))
end
function rrule!!(::CoDual{typeof(getindex)}, x::CoDual{<:CuArray}, i::CoDual{<:Integer})
    throw(ArgumentError(_SCALAR_IDX_MSG))
end

@is_primitive(MinimalCtx, Tuple{typeof(setindex!),CuArray,Any,Integer})
function frule!!(::Dual{typeof(setindex!)}, x::Dual{<:CuArray}, v::Dual, i::Dual{<:Integer})
    throw(ArgumentError(_SCALAR_IDX_MSG))
end
function rrule!!(
    ::CoDual{typeof(setindex!)}, x::CoDual{<:CuArray}, v::CoDual, i::CoDual{<:Integer}
)
    throw(ArgumentError(_SCALAR_IDX_MSG))
end

# Vector indexing: y = x[idx] where idx is a vector of integers (gather).
#
# frule:    dy = dx[idx]          (gather tangents)
# pullback: dx[idx] .+= dy_out   (scatter-add cotangents)
#
# Note: repeated indices in idx are undefined (last write wins on GPU without atomics).
# Distinct-index usage (e.g. embedding lookup, slicing) is safe.
@is_primitive(
    MinimalCtx, Tuple{typeof(getindex),CuMaybeComplexArray,AbstractVector{<:Integer}}
)
function frule!!(
    ::Dual{typeof(getindex)},
    x::Dual{<:CuMaybeComplexArray},
    idx::Dual{<:AbstractVector{<:Integer}},
)
    px, dx = arrayify(x)
    return Dual(px[primal(idx)], dx[primal(idx)])
end
function rrule!!(
    ::CoDual{typeof(getindex)},
    x::CoDual{<:CuMaybeComplexArray},
    idx::CoDual{<:AbstractVector{<:Integer}},
)
    px, dx = arrayify(x)
    pidx = primal(idx)
    y = px[pidx]
    dy_out = zero(y)
    function getindex_pb!!(::NoRData)
        dx[pidx] .+= dy_out
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy_out), getindex_pb!!
end

# norm: d(norm(x)) = Re(dot(x, dx)) / norm(x)  (valid for both real and complex x)
#       pullback:  dx += (dy / norm(x)) * x
#
# dot (real): d(dot(x,y)) = dot(dx,y) + dot(x,dy)
#             pullback:     dx += dz*y,  dy += dz*x
@is_primitive(MinimalCtx, Tuple{typeof(norm),CuMaybeComplexArray})
function frule!!(::Dual{typeof(norm)}, x::Dual{<:CuMaybeComplexArray})
    px, dx = arrayify(x)
    y = norm(px)
    dy = iszero(y) ? zero(real(eltype(px))) : real(dot(px, dx)) / y
    return Dual(y, dy)
end
function rrule!!(::CoDual{typeof(norm)}, x::CoDual{<:CuMaybeComplexArray})
    px, dx = arrayify(x)
    y = norm(px)
    function norm_pb!!(dy)
        # iszero triggers a device→host sync — inherent since we branch on the scalar result.
        iszero(y) || (dx .+= (dy / y) .* px)
        return NoRData(), NoRData()
    end
    return zero_fcodual(y), norm_pb!!
end

@is_primitive(MinimalCtx, Tuple{typeof(dot),CuFloatArray,CuFloatArray})
function frule!!(::Dual{typeof(dot)}, x::Dual{<:CuFloatArray}, y::Dual{<:CuFloatArray})
    px, dx = arrayify(x)
    py, dy = arrayify(y)
    return Dual(dot(px, py), dot(dx, py) + dot(px, dy))
end
function rrule!!(
    ::CoDual{typeof(dot)}, x::CoDual{<:CuFloatArray}, y::CoDual{<:CuFloatArray}
)
    px, dx = arrayify(x)
    py, dy = arrayify(y)
    function dot_pb!!(dz)
        dx .+= dz .* py
        dy .+= dz .* px
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(dot(px, py)), dot_pb!!
end

# Catch-all error rules for GPU reductions that use opaque CUDA kernels.
# These ops are differentiable in principle but lack explicit rules.
const _UNIMPL_MSG = "Add a new rule or open an issue at https://github.com/chalk-lab/Mooncake.jl."
for _fn in (:maximum, :minimum, :diff, :sort, :sortperm)
    @eval @is_primitive(MinimalCtx, Tuple{typeof($_fn),CuArray})
    @eval function frule!!(::Dual{typeof($_fn)}, x::Dual{<:CuArray}; kwargs...)
        throw(
            ArgumentError(
                "Mooncake: $_fn on CuArray is not yet differentiable. " * _UNIMPL_MSG
            ),
        )
    end
    @eval function rrule!!(::CoDual{typeof($_fn)}, x::CoDual{<:CuArray}; kwargs...)
        throw(
            ArgumentError(
                "Mooncake: $_fn on CuArray is not yet differentiable. " * _UNIMPL_MSG
            ),
        )
    end
end

# Rules for `prod(x)` on GPU arrays.
#
# prod(x) = x₁·x₂·…·xₙ,  ∂prod/∂xᵢ = prod(x)/xᵢ
# frule:    dy = prod(x) · sum(dx ./ x)
# pullback: dx[i] += dy · prod(x) / x[i]
#
# Note: undefined when any element of x is zero (gradient is skipped in that case).
@is_primitive(MinimalCtx, Tuple{typeof(prod),CuMaybeComplexArray})
function frule!!(::Dual{typeof(prod)}, x::Dual{<:CuMaybeComplexArray})
    px, dx = arrayify(x)
    y = prod(px)
    dy = iszero(y) ? zero(y) : y * sum(dx ./ px)
    return Dual(y, dy)
end
function rrule!!(::CoDual{typeof(prod)}, x::CoDual{<:CuMaybeComplexArray})
    px, dx = arrayify(x)
    y = prod(px)
    function prod_pb!!(dy)
        # Wirtinger chain rule for holomorphic prod: Δxᵢ = Δy · conj(y/xᵢ)
        # For real inputs conj is a no-op, so this is backward compatible.
        # iszero triggers a device→host sync — inherent since we branch on the scalar result.
        iszero(y) || (dx .+= dy .* conj.(y ./ px))
        return NoRData(), NoRData()
    end
    return zero_fcodual(y), prod_pb!!
end

# Rules for `cumsum(x)` on GPU arrays.
#
# y[k] = Σᵢ₌₁ᵏ x[i],  so ∂y[k]/∂x[i] = 1 if i≤k else 0
# frule:    dy = cumsum(dx)
# pullback: dx[i] += Σₖ≥ᵢ dy[k]  =  reverse(cumsum(reverse(dy)))
#
# Supports the optional `dims` keyword (passed through to CUDA's cumsum).
@is_primitive(MinimalCtx, Tuple{typeof(cumsum),CuMaybeComplexArray})
function frule!!(::Dual{typeof(cumsum)}, x::Dual{<:CuMaybeComplexArray}; kw...)
    px, dx = arrayify(x)
    return Dual(cumsum(px; kw...), cumsum(dx; kw...))
end
function rrule!!(::CoDual{typeof(cumsum)}, x::CoDual{<:CuMaybeComplexArray}; kw...)
    px, dx = arrayify(x)
    y = cumsum(px; kw...)
    dy_out = zero(y)
    d = get(kw, :dims, 1)
    function cumsum_pb!!(::NoRData)
        dx .+= reverse(cumsum(reverse(dy_out; dims=d); dims=d); dims=d)
        return NoRData(), NoRData()
    end
    return CoDual(y, dy_out), cumsum_pb!!
end

# Rules for `cumprod(x)` on GPU arrays.
#
# y[k] = Πᵢ₌₁ᵏ x[i],  ∂y[k]/∂x[i] = y[k]/x[i] if i≤k else 0
# frule:    dy[k] = y[k] · cumsum(dx ./ x)[k]
# pullback: dx[i] += (1/x[i]) · Σₖ≥ᵢ dy[k]·y[k]
#           i.e.  dx .+= reverse(cumsum(reverse(dy .* y))) ./ x
#
# Zero elements: when x[i] == 0 the cumulative product y[k] == 0 for all k ≥ i,
# so the Jacobian at that position is zero (the zero annihilates the product).
# nan_tangent_guard is used to return zero instead of NaN/Inf from 0/0 or x/0.
@is_primitive(MinimalCtx, Tuple{typeof(cumprod),CuMaybeComplexArray})
function frule!!(::Dual{typeof(cumprod)}, x::Dual{<:CuMaybeComplexArray}; kw...)
    px, dx = arrayify(x)
    y = cumprod(px; kw...)
    inv_px = nan_tangent_guard.(px, inv.(px))
    dy = y .* cumsum(dx .* inv_px; kw...)
    return Dual(y, dy)
end
function rrule!!(::CoDual{typeof(cumprod)}, x::CoDual{<:CuMaybeComplexArray}; kw...)
    px, dx = arrayify(x)
    y = cumprod(px; kw...)
    dy_out = zero(y)
    d = get(kw, :dims, 1)
    function cumprod_pb!!(::NoRData)
        # Wirtinger chain rule: Δxᵢ = (1/conj(xᵢ)) · Σₖ≥ᵢ Δyₖ · conj(yₖ)
        # i.e. dx .+= reverse(cumsum(reverse(dy .* conj.(y)))) ./ conj.(px)
        # For real inputs conj is a no-op, so this is backward compatible.
        # nan_tangent_guard: where px == 0 the product is annihilated (zero gradient).
        inv_cx_px = nan_tangent_guard.(px, inv.(conj.(px)))
        dx .+=
            reverse(cumsum(reverse(dy_out .* conj.(y); dims=d); dims=d); dims=d) .*
            inv_cx_px
        return NoRData(), NoRData()
    end
    return CoDual(y, dy_out), cumprod_pb!!
end

# Rules for `accumulate(+, x)` — identical to cumsum but via the accumulate interface.
# Other operators are not supported and throw an informative error (catch-all below).
@is_primitive(MinimalCtx, Tuple{typeof(accumulate),typeof(+),CuMaybeComplexArray})
function frule!!(
    ::Dual{typeof(accumulate)}, ::Dual{typeof(+)}, x::Dual{<:CuMaybeComplexArray}; kw...
)
    px, dx = arrayify(x)
    return Dual(accumulate(+, px; kw...), cumsum(dx; kw...))
end
function rrule!!(
    ::CoDual{typeof(accumulate)},
    ::CoDual{typeof(+)},
    x::CoDual{<:CuMaybeComplexArray};
    kw...,
)
    px, dx = arrayify(x)
    y = accumulate(+, px; kw...)
    dy_out = zero(y)
    d = get(kw, :dims, 1)
    function accumulate_plus_pb!!(::NoRData)
        dx .+= reverse(cumsum(reverse(dy_out; dims=d); dims=d); dims=d)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy_out), accumulate_plus_pb!!
end
@is_primitive(MinimalCtx, Tuple{typeof(accumulate),Any,CuArray})
function frule!!(::Dual{typeof(accumulate)}, op::Dual, x::Dual{<:CuArray}; kwargs...)
    throw(
        ArgumentError(
            "Mooncake: accumulate on CuArray only supports op=+; got op=$(primal(op)). " *
            _UNIMPL_MSG,
        ),
    )
end
function rrule!!(::CoDual{typeof(accumulate)}, op::CoDual, x::CoDual{<:CuArray}; kwargs...)
    throw(
        ArgumentError(
            "Mooncake: accumulate on CuArray only supports op=+; got op=$(primal(op)). " *
            _UNIMPL_MSG,
        ),
    )
end

# Rule for `sum(x)` — widened from CuFloatArray to also cover complex CuArrays.
# See also `src/rules/performance_patches`.
@is_primitive(DefaultCtx, Tuple{typeof(sum),CuMaybeComplexArray})
function frule!!(::Dual{typeof(sum)}, x::Dual{<:CuMaybeComplexArray})
    px, dx = arrayify(x)
    return Dual(sum(px), sum(dx))
end
function rrule!!(::CoDual{typeof(sum)}, x::CoDual{<:CuMaybeComplexArray})
    _, dx = arrayify(x)
    function sum_pb!!(dz)
        dx .+= dz
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(primal(x))), sum_pb!!
end

# Rule for `unsafe_copyto!(dest, doffs, src, soffs, n)` on GPU arrays.
# This function contains try/catch blocks (UpsilonNodes) from `context!(...)` that
# Mooncake cannot trace. It implements a GPU memcpy — the gradient is identity:
# accumulate the destination tangent into the source tangent over the same range.
#
# Forward: copy both primal and tangent with the same offsets.
# Backward: accumulate ddest[doffs:doffs+n-1] into dsrc[soffs:soffs+n-1], then zero ddest range.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(unsafe_copyto!),
        <:CuMaybeComplexArray,
        Integer,
        <:CuMaybeComplexArray,
        Integer,
        Integer,
    },
)
function frule!!(
    ::Dual{typeof(unsafe_copyto!)},
    dest::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    doffs::Dual{<:Integer,NoTangent},
    src::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    soffs::Dual{<:Integer,NoTangent},
    n::Dual{<:Integer,NoTangent},
)
    pdest, ddest = arrayify(dest)
    psrc, dsrc = arrayify(src)
    doffs_v, soffs_v, n_v = primal(doffs), primal(soffs), primal(n)
    unsafe_copyto!(pdest, doffs_v, psrc, soffs_v, n_v)
    unsafe_copyto!(ddest, doffs_v, dsrc, soffs_v, n_v)
    return dest
end
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)},
    dest::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    doffs::CoDual{<:Integer,NoFData},
    src::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    soffs::CoDual{<:Integer,NoFData},
    n::CoDual{<:Integer,NoFData},
)
    pdest, ddest = arrayify(dest)
    psrc, dsrc = arrayify(src)
    doffs_v, soffs_v, n_v = primal(doffs), primal(soffs), primal(n)
    dest_range = doffs_v:(doffs_v + n_v - 1)
    src_range = soffs_v:(soffs_v + n_v - 1)
    # Save the overwritten slice of dest (primal + tangent) so the pullback can restore it.
    pdest_copy = copy(view(pdest, dest_range))
    ddest_copy = copy(view(ddest, dest_range))
    unsafe_copyto!(pdest, doffs_v, psrc, soffs_v, n_v)
    function unsafe_copyto!_pb!!(::NoRData)
        # Accumulate gradient into src tangent, then restore dest to pre-mutation state.
        view(dsrc, src_range) .+= view(ddest, dest_range)
        copyto!(view(pdest, dest_range), pdest_copy)
        copyto!(view(ddest, dest_range), ddest_copy)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return dest, unsafe_copyto!_pb!!
end

# Rule for unsafe_copyto!(dest, doffs, src, soffs, n) where dest is a GPU array but src
# is a CPU Array (cross-device: host → device).  This path is taken e.g. when a Lux
# StatefulRecurrentCell initialises its hidden state from zeros32(...) and copies it to
# the GPU.  The pullback accumulates the GPU cotangent of the overwritten region back
# into the CPU src tangent via a synchronous device-to-host transfer.
@is_primitive(
    MinimalCtx,
    Tuple{typeof(unsafe_copyto!),<:CuMaybeComplexArray,Integer,<:Array,Integer,Integer},
)
function frule!!(
    ::Dual{typeof(unsafe_copyto!)},
    dest::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    doffs::Dual{<:Integer,NoTangent},
    src::Dual{<:Array,<:Array},
    soffs::Dual{<:Integer,NoTangent},
    n::Dual{<:Integer,NoTangent},
)
    pdest, ddest = arrayify(dest)
    psrc, dsrc = primal(src), tangent(src)
    doffs_v, soffs_v, n_v = primal(doffs), primal(soffs), primal(n)
    unsafe_copyto!(pdest, doffs_v, psrc, soffs_v, n_v)
    unsafe_copyto!(ddest, doffs_v, dsrc, soffs_v, n_v)
    return dest
end
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)},
    dest::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    doffs::CoDual{<:Integer,NoFData},
    src::CoDual{<:Array,<:Array},
    soffs::CoDual{<:Integer,NoFData},
    n::CoDual{<:Integer,NoFData},
)
    pdest, ddest = arrayify(dest)
    psrc, dsrc = primal(src), tangent(src)
    doffs_v, soffs_v, n_v = primal(doffs), primal(soffs), primal(n)
    dest_range = doffs_v:(doffs_v + n_v - 1)
    src_range = soffs_v:(soffs_v + n_v - 1)
    # Save overwritten slice via host copies (avoids scalar indexing on GPU).
    pdest_copy = Array(view(pdest, dest_range))
    ddest_copy = Array(view(ddest, dest_range))
    unsafe_copyto!(pdest, doffs_v, psrc, soffs_v, n_v)
    function mixed_copyto!_pb!!(::NoRData)
        # Propagate GPU cotangent back to CPU src tangent.
        view(dsrc, src_range) .+= Array(view(ddest, dest_range))
        # Restore dest primal and tangent to their pre-copy state.
        unsafe_copyto!(pdest, doffs_v, pdest_copy, 1, n_v)
        unsafe_copyto!(ddest, doffs_v, ddest_copy, 1, n_v)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return dest, mixed_copyto!_pb!!
end

# fill! on a GPU array has an internal try/catch block (for GPU error handling) that
# generates an UpsilonNode in the IR, which Mooncake cannot differentiate through.
# Provide explicit rules.
#
# Semantics: fill!(a, x) sets every element of a to x, so:
#   - d(output_i)/d(a_input_j) = 0  → tangent of a's prior content does not flow forward
#   - d(output_i)/d(x) = 1          → tangent(x) (if any) broadcasts into tangent(a)
# For integer x the tangent is NoTangent, so the tangent array is zeroed.
# For float x the tangent array is filled with tangent(x).
@is_primitive MinimalCtx Tuple{typeof(fill!),CuMaybeComplexArray,Any}
function frule!!(
    ::Dual{typeof(fill!)}, a::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray}, x::Dual
)
    fill!(primal(a), primal(x))
    tx = tangent(x)
    fill!(tangent(a), tx isa NoTangent ? zero(eltype(tangent(a))) : eltype(tangent(a))(tx))
    return a
end
function rrule!!(
    ::CoDual{typeof(fill!)},
    a::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    x::CoDual,
)
    pa, da = primal(a), tangent(a)
    old = copy(pa)
    fill!(pa, primal(x))
    function fill!_gpu_pb!!(::NoRData)
        copyto!(pa, old)
        # Gradient of x: ∂loss/∂x = Σ ∂loss/∂a_i = sum(da).
        # For non-differentiable x (tangent_type = NoTangent, e.g. integers) return NoRData.
        # Must use tangent_type here — rdata_type throws for primitive non-float types.
        dx = if tangent_type(typeof(primal(x))) == NoTangent
            NoRData()
        else
            rdata_type(typeof(primal(x)))(sum(da))
        end
        fill!(da, zero(eltype(da)))
        return NoRData(), NoRData(), dx
    end
    return a, fill!_gpu_pb!!
end

# _fields overload for CuArray tangents: the tangent of a plain CuArray is itself.
# for Adjoint/Transpose wrappers (tangent = Tangent/FData with a .parent field).
_fields(x::CuMaybeComplexArray) = (parent=x,)

# sum(A') / sum(transpose(A)) for CuArrays — real and complex unified.
#
# sum(transpose(A)) = sum(A) for both real and complex (permuting indices preserves total).
# frule: dy = sum(t_parent),  pullback: dx_parent .+= dy.
#
# sum(A') = conj(sum(A)) for complex A; for real A conj is identity, so the same formula
# holds for both.  frule: dy = conj(sum(t_parent)),  pullback: dx_parent .+= conj(dy).
#
# The real/complex unification works naturally: conj(x::Real) == x in Julia, so the
# complex Adjoint formula is a no-op on the real branch — no special-casing required.
@is_primitive(
    DefaultCtx, Tuple{typeof(sum),<:Transpose{<:CuFloatOrComplex,<:CuMaybeComplexArray}},
)
@is_primitive(
    DefaultCtx, Tuple{typeof(sum),<:Adjoint{<:CuFloatOrComplex,<:CuMaybeComplexArray}},
)
function frule!!(
    ::Dual{typeof(sum)}, x::Dual{<:Transpose{<:CuFloatOrComplex,<:CuMaybeComplexArray}}
)
    return Dual(sum(primal(x)), sum(_fields(tangent(x)).parent))
end
function frule!!(
    ::Dual{typeof(sum)}, x::Dual{<:Adjoint{<:CuFloatOrComplex,<:CuMaybeComplexArray}}
)
    return Dual(sum(primal(x)), conj(sum(_fields(tangent(x)).parent)))
end
function rrule!!(
    ::CoDual{typeof(sum)}, x::CoDual{<:Transpose{<:CuFloatOrComplex,<:CuMaybeComplexArray}}
)
    dx_parent = _fields(tangent(x)).parent
    function sum_tr_pb!!(dy)
        dx_parent .+= dy
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(primal(x))), sum_tr_pb!!
end
function rrule!!(
    ::CoDual{typeof(sum)}, x::CoDual{<:Adjoint{<:CuFloatOrComplex,<:CuMaybeComplexArray}}
)
    dx_parent = _fields(tangent(x)).parent
    function sum_adj_pb!!(dy)
        dx_parent .+= conj(dy)
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(primal(x))), sum_adj_pb!!
end

# Rules for `sum(f, x)` — applies f element-wise then reduces.
#
# Performance note: differentiation through f uses NDual numbers inside a
# single GPU kernel (via _gpu_broadcast_dual).  The cost is therefore similar to running
# NDual over f directly: one kernel launch that evaluates f once per element and
# returns both the value and the scalar partial df/dx simultaneously.
#
# Real arrays: one Dual slot per element (standard forward-mode chain rule).
# Complex arrays: two Dual slots per element (one for Re, one for Im) — see the
# CuComplexArray overload below.  This correctly handles non-holomorphic f (e.g. abs2)
# via Wirtinger calculus.
@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,CuFloatArray})
@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,<:Adjoint{<:IEEEFloat,<:CuFloatArray}})
@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,<:Transpose{<:IEEEFloat,<:CuFloatArray}})
function frule!!(
    ::Dual{typeof(sum)}, f::Dual, x::Dual{T}
) where {
    T<:Union{
        CuFloatArray,
        Adjoint{<:IEEEFloat,<:CuFloatArray},
        Transpose{<:IEEEFloat,<:CuFloatArray},
    },
}
    flat_px = parent(primal(x))
    flat_dx = _fields(tangent(x)).parent
    out = _gpu_broadcast_dual(primal(f), flat_px)
    y = sum(_gpu_dual_val, out)
    # JVP: d(sum(f(x))) = sum(f'(x) · dx) element-wise
    dy = if _is_gpu_differentiable(eltype(out)) && !(flat_dx isa NoTangent)
        sum(broadcast((o, t) -> _gpu_dual_part_cx(o, 1) * t, out, flat_dx))
    else
        zero(real(y))
    end
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(sum)}, f::CoDual, x::CoDual{T}
) where {
    T<:Union{
        CuFloatArray,
        Adjoint{<:IEEEFloat,<:CuFloatArray},
        Transpose{<:IEEEFloat,<:CuFloatArray},
    },
}
    flat_px = parent(primal(x))
    flat_dx = _fields(tangent(x)).parent
    out = _gpu_broadcast_dual(primal(f), flat_px)
    y = sum(_gpu_dual_val, out)
    function sum_f_pb!!(dy)
        if _is_gpu_differentiable(eltype(out))
            flat_dx .+= dy .* broadcast(o -> _gpu_dual_part_cx(o, 1), out)
        end
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), sum_f_pb!!
end

# Rules for `sum(f, x)` on complex CuArrays — extends the real rule above to ℂ.
#
# Each complex element z = Re(z) + i·Im(z) is assigned two Dual slots (one per real
# DOF), so a single GPU kernel pass gives both ∂f/∂Re(z) and ∂f/∂Im(z).  The
# Euclidean complex gradient is then:
#   grad[i] = ∂(Re·f + Im·f)/∂Re(zᵢ) + i · ∂(Re·f + Im·f)/∂Im(zᵢ)
# which handles non-holomorphic f (e.g. abs2) correctly via Wirtinger calculus.
#
# Works for both f: ℂ→ℝ (e.g. abs2, real, imag) and f: ℂ→ℂ (e.g. sin, exp).
# Performance: equivalent to NDual with 2-wide Duals — one kernel pass.
@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,CuComplexArray})
function frule!!(::Dual{typeof(sum)}, f::Dual, x::Dual{<:CuComplexArray})
    pf, px, dx = primal(f), primal(x), tangent(x)
    out = _gpu_broadcast_dual(pf, px)
    y = sum(_gpu_dual_val, out)
    dy = if _is_gpu_differentiable(eltype(out)) && !(dx isa NoTangent)
        sum(
            broadcast(
                (o, t) ->
                    _gpu_dual_part_cx(o, 1) * real(t) + _gpu_dual_part_cx(o, 2) * imag(t),
                out,
                dx,
            ),
        )
    else
        zero(real(y))
    end
    return Dual(y, dy)
end
function rrule!!(::CoDual{typeof(sum)}, f::CoDual, x::CoDual{<:CuComplexArray})
    pf, px, dx = primal(f), primal(x), tangent(x)
    out = _gpu_broadcast_dual(pf, px)
    y = sum(_gpu_dual_val, out)
    function sum_f_cx_pb!!(dy)
        if _is_gpu_differentiable(eltype(out))
            dx .+= broadcast(
                o -> complex(_gpu_adj_part(o, dy, 1), _gpu_adj_part(o, dy, 2)), out
            )
        end
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), sum_f_cx_pb!!
end

# Rules for `mapreduce(f, op, x)` on GPU arrays.
#
# CUDA.jl uses opaque reduction kernels that Mooncake cannot trace.  We intercept
# the op=+ and op=Base.add_sum cases by delegating to the sum frule!!/rrule!! above.
#
#   mapreduce(f, +, x)        ≡  sum(f, x)
#   mapreduce(f, add_sum, x)  ≡  sum(f, x)   (add_sum is Base's internal alias for +)
#
# Both operators must be covered: Base.sum(f, x) dispatches through
#   Base._sum(f, x, :) → mapreduce(f, add_sum, x)
# in Julia 1.11, so op=+ alone is insufficient.
#
# The mapreduce pullback returns one extra NoRData for the `op` argument compared
# to the sum pullback.
for _op in (:(+), :(Base.add_sum))
    @eval @is_primitive(
        MinimalCtx, Tuple{typeof(mapreduce),Any,typeof($_op),CuMaybeComplexArray}
    )
    @eval function frule!!(
        ::Dual{typeof(mapreduce)},
        f::Dual,
        ::Dual{typeof($_op)},
        x::Dual{<:CuMaybeComplexArray},
    )
        return frule!!(Dual(sum, NoTangent()), f, x)
    end
    @eval function rrule!!(
        ::CoDual{typeof(mapreduce)},
        f::CoDual,
        ::CoDual{typeof($_op)},
        x::CoDual{<:CuMaybeComplexArray},
    )
        y, sum_pb!! = rrule!!(zero_fcodual(sum), f, x)
        function mapreduce_pb!!(dy)
            _, r_f, r_x = sum_pb!!(dy)          # sum pullback: (sum, f, x)
            return NoRData(), r_f, NoRData(), r_x  # mapreduce: (mapreduce, f, op, x)
        end
        return y, mapreduce_pb!!
    end
end

# Rules for `reduce(op, x)` on GPU arrays.
#
#   reduce(+, x)  ≡  sum(x),   delegated to the sum rrule
#   reduce(*, x)  ≡  prod(x),  delegated to the prod rrule
#
# Unlike mapreduce, reduce is user-facing and Base does not route through the
# add_sum / mul_prod aliases here, so only the literal + and * are needed.
# The reduce pullback returns one extra NoRData for `op` compared to sum/prod.
for (_op, _fn) in ((:(+), :sum), (:(Base.:*), :prod))
    @eval @is_primitive(MinimalCtx, Tuple{typeof(reduce),typeof($_op),CuMaybeComplexArray})
    @eval function frule!!(
        ::Dual{typeof(reduce)}, ::Dual{typeof($_op)}, x::Dual{<:CuMaybeComplexArray}
    )
        return frule!!(Dual($_fn, NoTangent()), x)
    end
    @eval function rrule!!(
        ::CoDual{typeof(reduce)}, ::CoDual{typeof($_op)}, x::CoDual{<:CuMaybeComplexArray}
    )
        y, pb!! = rrule!!(zero_fcodual($_fn), x)
        function reduce_pb!!(dy)
            _, r_x = pb!!(dy)              # delegate pullback: (fn, x)
            return NoRData(), NoRData(), r_x  # reduce: (reduce, op, x)
        end
        return y, reduce_pb!!
    end
end

# Catch-all rules for unsupported operators — give a clear error rather than letting
# Mooncake attempt to trace into an opaque CUDA reduction kernel.
@is_primitive(MinimalCtx, Tuple{typeof(mapreduce),Any,Any,CuArray})
function frule!!(::Dual{typeof(mapreduce)}, f::Dual, op::Dual, x::Dual{<:CuArray})
    throw(
        ArgumentError(
            "Mooncake: mapreduce on CuArray only supports op=+ or op=Base.add_sum; " *
            "got op=$(primal(op)). " *
            _UNIMPL_MSG,
        ),
    )
end
function rrule!!(::CoDual{typeof(mapreduce)}, f::CoDual, op::CoDual, x::CoDual{<:CuArray})
    throw(
        ArgumentError(
            "Mooncake: mapreduce on CuArray only supports op=+ or op=Base.add_sum; " *
            "got op=$(primal(op)). " *
            _UNIMPL_MSG,
        ),
    )
end

@is_primitive(MinimalCtx, Tuple{typeof(reduce),Any,CuArray})
function frule!!(::Dual{typeof(reduce)}, op::Dual, x::Dual{<:CuArray})
    throw(
        ArgumentError(
            "Mooncake: reduce on CuArray only supports op=+ (sum) or op=* (prod); " *
            "got op=$(primal(op)). " *
            _UNIMPL_MSG,
        ),
    )
end
function rrule!!(::CoDual{typeof(reduce)}, op::CoDual, x::CoDual{<:CuArray})
    throw(
        ArgumentError(
            "Mooncake: reduce on CuArray only supports op=+ (sum) or op=* (prod); " *
            "got op=$(primal(op)). " *
            _UNIMPL_MSG,
        ),
    )
end

# vcat / hcat / cat on CuArrays are not yet supported — give a clear error rather than
# letting Mooncake attempt to trace through opaque CUDA memory kernels.
for _fn in (:vcat, :hcat)
    @eval @is_primitive(MinimalCtx, Tuple{typeof($_fn),Vararg{Union{CuArray,Number}}})
    @eval function frule!!(::Dual{typeof($_fn)}, args::Dual...)
        throw(
            ArgumentError(
                "Mooncake: $($_fn) on CuArray is not yet differentiable. " * _UNIMPL_MSG
            ),
        )
    end
    @eval function rrule!!(::CoDual{typeof($_fn)}, args::CoDual...)
        throw(
            ArgumentError(
                "Mooncake: $($_fn) on CuArray is not yet differentiable. " * _UNIMPL_MSG
            ),
        )
    end
end
@is_primitive(MinimalCtx, Tuple{typeof(cat),Vararg{Union{CuArray,Number}}})
function frule!!(::Dual{typeof(cat)}, args::Dual...; kwargs...)
    throw(
        ArgumentError("Mooncake: cat on CuArray is not yet differentiable. " * _UNIMPL_MSG)
    )
end
function rrule!!(::CoDual{typeof(cat)}, args::CoDual...; kwargs...)
    throw(
        ArgumentError("Mooncake: cat on CuArray is not yet differentiable. " * _UNIMPL_MSG)
    )
end

# Rules are written at the `generic_matmatmul!` / `generic_matvecmul!` level rather
# than at the individual CUBLAS primitive level (gemm!, gemv!, gemmEx!, symm!, ...).
# This gives broad coverage of the LinearAlgebra.mul! dispatch chain with just two
# rules, and is correct for all practical ML workloads (dense real/complex arrays).
# The tradeoff: symmetric/Hermitian cases (tA='S'/'H', dispatching to symv!/hemv!
# in the primal) use gemm!/gemv! in the backward, which is mathematically correct
# only when the full matrix is populated. Direct CUBLAS calls that bypass
# LinearAlgebra.mul! are not covered; add lower-level rules if that becomes needed.

# Rule for `LinearAlgebra.generic_matmatmul!` on real and complex GPU arrays.
#
# `generic_matmatmul!(C, tA, tB, A, B)` computes C = op_A(A) * op_B(B) in-place,
# where tA, tB ∈ {'N','T','C'} are BLAS transpose flags. It is the generic fallback
# that LinearAlgebra dispatches to when CUBLAS has no specific method — for example,
# `adjoint(CuVector) * CuMatrix` falls through here because CUBLAS.gemm! only accepts
# CuMatrix inputs.
#
# Strategy: reshape any CuVector to (n,1) CuMatrix via `matrixify` (zero-copy), then
# delegate to CUBLAS.gemm! which is differentiable and avoids scalar GPU indexing.
#
# Backward formulas for C = op_A(A) * op_B(B) (real and complex; uses '^H' = Hermitian
# conjugate, which CUBLAS flag 'C' handles; for real 'C' == 'T'):
#   tA='N': dA += dC * op_B(B)^H    (flags: 'N', tB=='N' ? 'C' : 'N')
#   tA≠'N': dA += op_B(B) * dC^H   (flags: tB, 'C')
#   tB='N': dB += op_A(A)^H * dC   (flags: tA=='N' ? 'C' : 'N', 'N')
#   tB≠'N': dB += dC^H * op_A(A)   (flags: 'C', tA)
#
# Limitation: the 'T' (plain transpose) flag is only correct for real arrays.
# For complex arrays, 'T' would require element-wise conjugation (conj(B)) in the
# backward, which cannot be expressed as a single CUBLAS GEMM call. A runtime guard
# below rejects complex + 'T' rather than silently returning incorrect gradients.

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(LinearAlgebra.generic_matmatmul!),
        <:CuMaybeComplexArray,
        Char,
        Char,
        <:CuMaybeComplexArray,
        <:CuMaybeComplexArray,
    },
)
function frule!!(
    ::Dual{typeof(LinearAlgebra.generic_matmatmul!)},
    C::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    tA::Dual{Char,NoTangent},
    tB::Dual{Char,NoTangent},
    A::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    B::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
)
    pC, dC = matrixify(C)
    pA, dA = matrixify(A)
    pB, dB = matrixify(B)
    tAv = primal(tA)
    tBv = primal(tB)
    T = eltype(pA)
    T <: Complex &&
        (tAv == 'T' || tBv == 'T') &&
        throw(
            ArgumentError(
                "Mooncake: generic_matmatmul! with the 'T' (plain transpose) flag is not " *
                "supported for complex CuArrays — the backward requires element-wise " *
                "conjugation, which cannot be expressed as a single CUBLAS GEMM. " *
                "Use adjoint ('C') instead of transpose ('T').",
            ),
        )
    _1 = one(T)
    _0 = zero(T)
    # primal: C = op_A(A) * op_B(B)
    CUBLAS.gemm!(tAv, tBv, _1, pA, pB, _0, pC)
    # tangent (product rule): dC = op_A(dA)*op_B(pB) + op_A(pA)*op_B(dB)
    CUBLAS.gemm!(tAv, tBv, _1, dA, pB, _0, dC)
    CUBLAS.gemm!(tAv, tBv, _1, pA, dB, _1, dC)
    return C
end
function rrule!!(
    ::CoDual{typeof(LinearAlgebra.generic_matmatmul!)},
    C::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    tA::CoDual{Char,NoFData},
    tB::CoDual{Char,NoFData},
    A::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    B::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
)
    pC, dC = matrixify(C)
    pA, dA = matrixify(A)
    pB, dB = matrixify(B)
    tAv = primal(tA)
    tBv = primal(tB)
    T = eltype(pA)
    T <: Complex &&
        (tAv == 'T' || tBv == 'T') &&
        throw(
            ArgumentError(
                "Mooncake: generic_matmatmul! with the 'T' (plain transpose) flag is not " *
                "supported for complex CuArrays — the backward requires element-wise " *
                "conjugation, which cannot be expressed as a single CUBLAS GEMM. " *
                "Use adjoint ('C') instead of transpose ('T').",
            ),
        )
    _1 = one(T)
    _0 = zero(T)
    pC_copy = copy(pC)
    CUBLAS.gemm!(tAv, tBv, _1, pA, pB, _0, pC)
    function generic_matmatmul!_pb!!(::NoRData)
        if tAv == 'N'
            CUBLAS.gemm!('N', tBv == 'N' ? 'C' : 'N', _1, dC, pB, _1, dA) # dA += dC * op_B(B)^H
        else
            CUBLAS.gemm!(tBv, 'C', _1, pB, dC, _1, dA)                     # dA += op_B(B) * dC^H
        end
        if tBv == 'N'
            CUBLAS.gemm!(tAv == 'N' ? 'C' : 'N', 'N', _1, pA, dC, _1, dB) # dB += op_A(A)^H * dC
        else
            CUBLAS.gemm!('C', tAv, _1, dC, pA, _1, dB)                     # dB += dC^H * op_A(A)
        end
        copyto!(pC, pC_copy)
        dC .= _0
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return C, generic_matmatmul!_pb!!
end

# 7-arg version of `generic_matmatmul!`: used by CUDA.jl's override of the LinearAlgebra
# function, which always passes explicit alpha and beta scalars.  The 5-arg rule above
# covers the pure LinearAlgebra fallback path; this rule covers the CUDA.jl path
# (cublas/linalg.jl line 349) that is reached from `A * B` → `mul!` → matmul dispatch.
#
# alpha / beta are treated as non-differentiable (NoTangent / NoFData): they are
# typically `true`/`false` (from `MulAddMul`) and we never differentiate w.r.t. them.

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(LinearAlgebra.generic_matmatmul!),
        <:CuMaybeComplexArray,
        Char,
        Char,
        <:CuMaybeComplexArray,
        <:CuMaybeComplexArray,
        Number,
        Number,
    },
)
function frule!!(
    ::Dual{typeof(LinearAlgebra.generic_matmatmul!)},
    C::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    tA::Dual{Char,NoTangent},
    tB::Dual{Char,NoTangent},
    A::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    B::Dual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    alpha::Dual{<:Number,NoTangent},
    beta::Dual{<:Number,NoTangent},
)
    pC, dC = matrixify(C)
    pA, dA = matrixify(A)
    pB, dB = matrixify(B)
    tAv = primal(tA)
    tBv = primal(tB)
    T = eltype(pA)
    T <: Complex &&
        (tAv == 'T' || tBv == 'T') &&
        throw(
            ArgumentError(
                "Mooncake: generic_matmatmul! with the 'T' (plain transpose) flag is not " *
                "supported for complex CuArrays — the backward requires element-wise " *
                "conjugation, which cannot be expressed as a single CUBLAS GEMM. " *
                "Use adjoint ('C') instead of transpose ('T').",
            ),
        )
    _α = T(primal(alpha))
    _β = T(primal(beta))
    _1 = one(T)
    # primal: C := α*op_A(A)*op_B(B) + β*C
    CUBLAS.gemm!(tAv, tBv, _α, pA, pB, _β, pC)
    # tangent: dC := α*(op_A(dA)*op_B(pB) + op_A(pA)*op_B(dB)) + β*dC
    CUBLAS.gemm!(tAv, tBv, _α, dA, pB, _β, dC)
    CUBLAS.gemm!(tAv, tBv, _α, pA, dB, _1, dC)
    return C
end
function rrule!!(
    ::CoDual{typeof(LinearAlgebra.generic_matmatmul!)},
    C::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    tA::CoDual{Char,NoFData},
    tB::CoDual{Char,NoFData},
    A::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    B::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    alpha::CoDual{<:Number,NoFData},
    beta::CoDual{<:Number,NoFData},
)
    pC, dC = matrixify(C)
    pA, dA = matrixify(A)
    pB, dB = matrixify(B)
    tAv = primal(tA)
    tBv = primal(tB)
    T = eltype(pA)
    T <: Complex &&
        (tAv == 'T' || tBv == 'T') &&
        throw(
            ArgumentError(
                "Mooncake: generic_matmatmul! with the 'T' (plain transpose) flag is not " *
                "supported for complex CuArrays — the backward requires element-wise " *
                "conjugation, which cannot be expressed as a single CUBLAS GEMM. " *
                "Use adjoint ('C') instead of transpose ('T').",
            ),
        )
    _α = T(primal(alpha))
    _β = T(primal(beta))
    _1 = one(T)
    pC_copy = copy(pC)
    CUBLAS.gemm!(tAv, tBv, _α, pA, pB, _β, pC)
    function generic_matmatmul!_7arg_pb!!(::NoRData)
        # Adjoint of C = α*op_A(A)*op_B(B) + β*C_old requires conj(α) and conj(β).
        # For real scalars conj is identity, so this is backward-compatible.
        _cα = conj(_α)
        _cβ = conj(_β)
        if tAv == 'N'
            CUBLAS.gemm!('N', tBv == 'N' ? 'C' : 'N', _cα, dC, pB, _1, dA) # dA += conj(α)*dC*op_B(B)^H
        else
            CUBLAS.gemm!(tBv, 'C', _cα, pB, dC, _1, dA)                     # dA += conj(α)*op_B(B)*dC^H
        end
        if tBv == 'N'
            CUBLAS.gemm!(tAv == 'N' ? 'C' : 'N', 'N', _cα, pA, dC, _1, dB) # dB += conj(α)*op_A(A)^H*dC
        else
            CUBLAS.gemm!('C', tAv, _cα, dC, pA, _1, dB)                     # dB += conj(α)*dC^H*op_A(A)
        end
        copyto!(pC, pC_copy)
        dC .*= _cβ  # gradient w.r.t. C_old: ΔC_old = conj(β) * ΔC_new
        return NoRData(),
        NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(),
        NoRData()
    end
    return C, generic_matmatmul!_7arg_pb!!
end

# Rule for `LinearAlgebra.generic_matvecmul!` on real and complex GPU arrays.
#
# `generic_matvecmul!(Y, tA, A, B, alpha, beta)` computes Y = alpha*op(A)*B + beta*Y
# in-place, where tA ∈ {'N','T','C'} is the BLAS transpose flag.
# CUDA.jl overrides this to call CUBLAS.gemv! directly (cublas/linalg.jl), bypassing
# `mul!`. Without this rule, Mooncake's forward-mode interpreter traces into CUDA's
# task-local-storage machinery (CUBLAS.handle → task_local_state!) which contains
# `Unreachable` code paths when called with dual types → SIGILL.
#
# Strategy: for the primal and tangent pass use CUBLAS.gemv!; for the dA update
# (an outer product) reshape both vectors to (n,1) matrices and use CUBLAS.gemm!.
#
# Backward formulas for Y = alpha*op(A)*B + beta*Y_old (ȳ = cotangent of Y):
#   tA='N': dA += alpha * ȳ * B^H  (outer product via gemm!('N','C'))
#   tA≠'N': dA += alpha * B * ȳ^H  (outer product via gemm!('N','C'), roles swapped)
#   tA='N': dB += alpha * A^H * ȳ  (gemv!('C'))
#   tA≠'N': dB += alpha * A   * ȳ  (gemv!('N'), since op(A)^H = A)
#   dY_old  = beta * ȳ             (pass-through scaled by beta)
#
# Limitation: 'T' flag for complex arrays is rejected (same as generic_matmatmul!).

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(LinearAlgebra.generic_matvecmul!),
        <:CuMaybeComplexVec,
        <:AbstractChar,
        <:CuMaybeComplexMat,
        <:CuMaybeComplexVec,
        Number,
        Number,
    },
)
function frule!!(
    ::Dual{typeof(LinearAlgebra.generic_matvecmul!)},
    Y::Dual{<:CuMaybeComplexVec,<:CuMaybeComplexVec},
    tA::Dual{<:AbstractChar,NoTangent},
    A::Dual{<:CuMaybeComplexMat,<:CuMaybeComplexMat},
    B::Dual{<:CuMaybeComplexVec,<:CuMaybeComplexVec},
    alpha::Dual{<:Number,NoTangent},
    beta::Dual{<:Number,NoTangent},
)
    pY, dY = primal(Y), tangent(Y)
    pA, dA = primal(A), tangent(A)
    pB, dB = primal(B), tangent(B)
    tAv = primal(tA)
    av = primal(alpha)
    bv = primal(beta)
    T = eltype(pA)
    eltype(pB) == T || throw(
        ArgumentError(
            "Mooncake: GPU gemv with mismatched element types " *
            "(A=$(T), B=$(eltype(pB))) is not supported. " *
            "Cast all arrays to the same element type before multiplying. " *
            "(Note: cu() downcasts Float64/ComplexF64 to Float32/ComplexF32 by default; " *
            "use CuArray(x) to preserve the element type.)",
        ),
    )
    T <: Complex &&
        tAv == 'T' &&
        throw(
            ArgumentError(
                "Mooncake: generic_matvecmul! with the 'T' (plain transpose) flag is not " *
                "supported for complex CuArrays. Use adjoint ('C') instead.",
            ),
        )
    _1 = one(T)
    # tangent (product rule): dY = av*op(dA)*pB + av*op(pA)*dB + bv*dY
    CUBLAS.gemv!(tAv, av, dA, pB, bv, dY) # dY  = av*op(dA)*pB + bv*dY
    CUBLAS.gemv!(tAv, av, pA, dB, _1, dY) # dY += av*op(pA)*dB
    # primal: pY = av*op(pA)*pB + bv*pY
    CUBLAS.gemv!(tAv, av, pA, pB, bv, pY)
    return Y
end
function rrule!!(
    ::CoDual{typeof(LinearAlgebra.generic_matvecmul!)},
    Y::CoDual{<:CuMaybeComplexVec,<:CuMaybeComplexVec},
    tA::CoDual{<:AbstractChar,NoFData},
    A::CoDual{<:CuMaybeComplexMat,<:CuMaybeComplexMat},
    B::CoDual{<:CuMaybeComplexVec,<:CuMaybeComplexVec},
    alpha::CoDual{<:Number,NoFData},
    beta::CoDual{<:Number,NoFData},
)
    pY, dY = primal(Y), tangent(Y)
    pA, dA = primal(A), tangent(A)
    pB, dB = primal(B), tangent(B)
    tAv = primal(tA)
    av = primal(alpha)
    bv = primal(beta)
    T = eltype(pA)
    eltype(pB) == T || throw(
        ArgumentError(
            "Mooncake: GPU gemv with mismatched element types " *
            "(A=$(T), B=$(eltype(pB))) is not supported. " *
            "Cast all arrays to the same element type before multiplying. " *
            "(Note: cu() downcasts Float64/ComplexF64 to Float32/ComplexF32 by default; " *
            "use CuArray(x) to preserve the element type.)",
        ),
    )
    T <: Complex &&
        tAv == 'T' &&
        throw(
            ArgumentError(
                "Mooncake: generic_matvecmul! with the 'T' (plain transpose) flag is not " *
                "supported for complex CuArrays. Use adjoint ('C') instead.",
            ),
        )
    _1 = one(T)
    pY_copy = copy(pY)
    CUBLAS.gemv!(tAv, av, pA, pB, bv, pY)
    function generic_matvecmul!_pb!!(::NoRData)
        # dA update: outer product — reshape vectors to (n,1) matrices for gemm!
        dY_mat = reshape(dY, :, 1)
        pB_mat = reshape(pB, :, 1)
        if tAv == 'N'
            CUBLAS.gemm!('N', 'C', av, dY_mat, pB_mat, _1, dA) # dA += av * ȳ * B^H
        else
            CUBLAS.gemm!('N', 'C', av, pB_mat, dY_mat, _1, dA) # dA += av * B * ȳ^H
        end
        # dB update: gemv with Hermitian conjugate of op(A)
        if tAv == 'N'
            CUBLAS.gemv!('C', av, pA, dY, _1, dB) # dB += av * A^H * ȳ
        else
            CUBLAS.gemv!('N', av, pA, dY, _1, dB) # dB += av * A   * ȳ  (op(A)^H = A)
        end
        # Y tangent passes through scaled by beta
        dY .*= bv
        copyto!(pY, pY_copy)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return Y, generic_matvecmul!_pb!!
end
# The tangent of Array{T} is Array{T} (fdata, accumulated in-place).
# The tangent of CuArray{T} is CuArray{T} (fdata, accumulated in-place).
@is_primitive(MinimalCtx, Tuple{typeof(cu),AbstractArray{<:CuFloatOrComplex}})
function frule!!(::Dual{typeof(cu)}, x::Dual{<:AbstractArray{<:CuFloatOrComplex}})
    return Dual(cu(primal(x)), cu(tangent(x)))
end
function rrule!!(::CoDual{typeof(cu)}, x::CoDual{<:AbstractArray{<:CuFloatOrComplex}})
    dx = tangent(x)
    dy_gpu = cu(zero(primal(x)))  # output fdata, accumulated into by downstream
    function cu_pb!!(::NoRData)
        dx .+= Array(dy_gpu)      # transfer gradient back to CPU in-place
        return NoRData(), NoRData()
    end
    return CoDual(cu(primal(x)), dy_gpu), cu_pb!!
end

# Rule for `Array(x::CuArray)` — GPU→CPU transfer.
# Symmetric to the `cu` rule: tangent stays on CPU, accumulated into by the pullback.
@is_primitive(
    MinimalCtx, Tuple{Type{Array{T,N}},CuArray{T,N}} where {T<:CuFloatOrComplex,N}
)
function frule!!(
    ::Dual{Type{Array{T,N}}}, x::Dual{<:CuArray{T,N}}
) where {T<:CuFloatOrComplex,N}
    return Dual(Array(primal(x)), Array(tangent(x)))
end
function rrule!!(
    ::CoDual{Type{Array{T,N}}}, x::CoDual{<:CuArray{T,N}}
) where {T<:CuFloatOrComplex,N}
    dx = tangent(x)
    dy_cpu = Array(zero(primal(x)))  # output fdata, accumulated into by downstream
    function array_pb!!(::NoRData)
        dx .+= cu(dy_cpu)            # transfer gradient back to GPU in-place
        return NoRData(), NoRData()
    end
    return CoDual(Array(primal(x)), dy_cpu), array_pb!!
end

# Rule for `Diagonal(v::CuMaybeComplexArray)` — construction of a GPU diagonal matrix.
# Diagonal is a thin wrapper: its only differentiable field is `.diag`.
# frule:    d(Diagonal(v)) = Diagonal(dv)
# pullback: dv += diag(dD)  (i.e. extract the diagonal from the output cotangent)
@is_primitive(MinimalCtx, Tuple{Type{<:Diagonal},CuMaybeComplexArray})
function frule!!(::Dual{<:Type{<:Diagonal}}, v::Dual{<:CuMaybeComplexArray})
    # Diagonal is a non-mutable struct; its tangent type is Tangent{(; diag::CuArray)}.
    return Dual(Diagonal(primal(v)), Tangent((; diag=tangent(v))))
end
function rrule!!(::CoDual{<:Type{<:Diagonal}}, v::CoDual{<:CuMaybeComplexArray})
    pv, dv = arrayify(v)
    dD = zero(pv)  # fdata for .diag of the Diagonal output
    function diagonal_pb!!(::NoRData)
        dv .+= dD
        return NoRData(), NoRData()
    end
    # fdata_type(Diagonal{T, CuArray{T,1}}) = FData{(; diag::CuArray{T,1})}
    return CoDual(Diagonal(pv), FData((; diag=dD))), diagonal_pb!!
end

# ===== GPU broadcasting rule (materialize-level, NDual-based forward pass) =====
#
# --- How it works ---
#
# Goal: given y = f.(x1, x2, ...) on CuArrays, compute both y and the gradient
# dy/dx_i in a single GPU kernel pass.
#
# The key idea is NDual arithmetic.  A dual number carries a primal value
# and a vector of N partial derivatives ("partials"):
#
#   NDual(v, (p1, p2, ..., pN))   represents   v + p1*e1 + p2*e2 + ... + pN*eN
#
# where e1..eN are symbolic infinitesimals.  Any function f defined in terms of
# arithmetic and standard math ops propagates them exactly via the chain rule —
# no source transformation required.
#
# We assign one slot per real DOF of each differentiable broadcast argument:
#   real arg x_i  -> slot k,   Dual(x_i[j], one_hot(k, N))
#   complex arg z_i -> slots k,k+1, Complex(Dual(Re(z_i[j]), e_k), Dual(Im(z_i[j]), e_{k+1}))
#
# Then the GPU kernel evaluates f element-wise on these Duals.  By the chain rule:
#   result[j] = Dual(f(x1[j],...), (df/dx1[j], df/dx2[j], ..., df/dxN[j]))
#
# In one kernel pass we get:
#   primal:    value(result[j])        = f(x1[j], x2[j], ...)
#   partials:  partials(result[j])[k] = df/dx_k at element j
#
# Reverse mode (rrule!!): given upstream gradient dy_out, accumulate
#   dx_k[j] += Re(conj(dy_out[j]) * df/dx_k[j])   for real or complex
#
# Forward mode (frule!!): given tangents dt_k, compute
#   dy[j] = sum_k  df/dx_k[j] * dt_k[j]            (JVP, chain rule)
#
# For Adjoint/Transpose leaves (A' or transpose(A)): the kernel sees A'[i,j] as a
# plain scalar, so Dual wrapping is unchanged.  Only the gradient accumulation differs:
# the contribution is transposed (and conjugated for complex Adjoint) before being
# added to the parent array's gradient.
#
# Intercept point: `Base.Broadcast.materialize` (not `broadcasted`) because:
#   - `materialize` : Broadcasted -> CuArray (types match rrule signature)
#   - `Base.Broadcast.flatten` fuses nested broadcast trees into one function,
#     so a single kernel handles arbitrarily deep `.`-fusion (e.g. sin.(x .^ 2)).
#
# Cost: one fused GPU kernel evaluating f with N extra NDual slots (N = total real DOFs
# across all CuArray args).  Comparable to a single NDual pass over f.
#
# Analogy with JAX vmap: JAX's vmap lifts f(x_scalar) -> f(x_batch) by adding a batch
# dimension, using a single kernel where each thread handles one element.  We do the
# same thing but widen the scalar *type* instead of adding a dimension: each thread
# evaluates f(Dual(x[j], partials)) rather than f(x[j]).  Both exploit the same GPU
# property — threads are independent — so the kernel shape is unchanged; only the
# per-thread arithmetic is wider.  The difference is what is being lifted: batch
# dimension (vmap) vs. tangent dimension (NDual).
#
# Supported primitives inside f (Julia CUDA kernel constraints):
# f must compile to PTX: no heap allocation, no dynamic dispatch, no cross-element ops.
#
#   Primitive                  Julia CUDA kernel    JAX (inside jit/vmap)
#   ─────────────────────────────────────────────────────────────────────
#   Scalar math (sin/exp/...)  yes                  yes
#   Complex arithmetic         yes                  yes
#   Plain if/while             yes (warp diverge)   yes
#   NDual                      yes (plain bitstype) n/a
#   Data-dep. conditionals     warning: warp div.   yes  (lax.cond)
#   Loops with carry / scan    must fully unroll    yes  (lax.scan)
#   Bounded while              must fully unroll    yes  (lax.while_loop)
#   Reductions inside f        no (needs 2nd kern.) yes  (lax.reduce)
#   Gather / scatter           no (no autodiff)     yes  (lax.gather/scatter)
#   Heap allocation            no                   no
#
# The fundamental gap vs JAX: control flow and reductions are first-class differentiable
# ops in JAX/XLA (traced into a Jaxpr with known derivative rules).  Julia evaluates
# eagerly, so Mooncake only sees an unrolled execution trace.
#
# Scalar IEEEFloat and Complex{<:IEEEFloat} variables (e.g. `c` in `c .* x`) get a
# Dual slot in the same kernel pass.  They have NoFData so can't use in-place
# accumulation; instead their gradient (sum of the partial over all output elements)
# is packed into the Broadcasted rdata via _gpu_fill_scalar_rdata.
# Other scalar types (e.g. Int, Bool) have dof=0 and are not differentiated.
# To support a new scalar type T: add a _broadcast_elem_dof_type(::Type{T}) method,
# handle it in _leaf_effective_tangent / materialize_pb!! / _gpu_fill_args_rdata.

# ── Dual-wrapping helpers for GPU kernels ────────────────────────────────────────────
#
# LIMITATION: this forward-mode broadcast strategy works for pure elementwise Julia
# functions, but breaks down for operations that are NOT implemented as Julia broadcasts:
#
#   • cuDNN-backed layers (BatchNorm, InstanceNorm, LayerNorm via cudnnNormForward!) call
#     C++ library kernels that receive raw Float32/Float64 device pointers.  They never
#     see the NDual-element CuArrays that Mooncake inserts, so the GPU compiler fails to
#     generate a kernel for NDual{Float32, N} element types.
#
#   • Any Lux/Flux layer that dispatches to a specialised CUDA primitive (softmax via
#     NNlib.softmax!, attention scoring, etc.) hits the same wall: the primitive expects
#     plain float arrays, not NDual arrays.
#
# The failure mode is a GPU kernel-compilation error at trace time, e.g.:
#   "LLVM error: ... cannot select: ... NDual{Float32, 3}"
# (N = total real DOFs across all broadcast inputs; 3 arises for BatchNorm as
#  scale + input + bias each contribute one real DOF.)
#
# Fix: add an explicit rrule!! for the cuDNN / NNlib primitive so Mooncake never tries
# to trace through it with NDual inputs.  See the unsafe_copyto! and fill! rules above
# for the pattern to follow.

# Wrap a real differentiable scalar as an NDual with a one-hot partial at
# `slot` (1-indexed, out of N total slots).  Non-differentiable types (Int, Bool, …)
# pass through unchanged so NDual arithmetic still works (e.g. x .^ 7).
@inline function _gpu_bcast_dual(x::T, slot::Int, ::Val{N}) where {T<:IEEEFloat,N}
    NDual{T,N}(x, ntuple(j -> T(j == slot), Val(N)))
end
@inline _gpu_bcast_dual(x, ::Int, ::Any) = x  # non-differentiable: pass through

@inline function _gpu_bcast_dual(
    x::Complex{ET}, slot_re::Int, slot_im::Int, ::Val{N}
) where {ET<:IEEEFloat,N}
    Complex(
        NDual{ET,N}(real(x), ntuple(j -> ET(j == slot_re), Val(N))),
        NDual{ET,N}(imag(x), ntuple(j -> ET(j == slot_im), Val(N))),
    )
end

# At Julia-compile time, compute the total number of Dual slots N from the argument
# types (real → 1 slot, complex → 2 slots, other → 0) and generate code that wraps
# each differentiable arg as the appropriate Dual before calling f.
# This produces a fixed-width Dual<N> for the GPU compiler; no runtime branching.
@generated function _gpu_apply_with_duals(f::F, args...) where {F}
    N = 0
    offsets = Int[]
    for ET in args
        push!(offsets, N)
        if ET <: IEEEFloat
            N += 1
        elseif ET <: Complex{<:IEEEFloat}
            N += 2
        end
    end
    N == 0 && return :(f(args...))
    body = Expr[]
    wrapped = Symbol[]
    for (i, (ET, off)) in enumerate(zip(args, offsets))
        sym = Symbol(:_w, i)
        push!(wrapped, sym)
        if ET <: IEEEFloat
            push!(body, :($sym = _gpu_bcast_dual(args[$i], $(off + 1), Val{$N}())))
        elseif ET <: Complex{<:IEEEFloat}
            push!(
                body, :($sym = _gpu_bcast_dual(args[$i], $(off + 1), $(off + 2), Val{$N}()))
            )
        else
            push!(body, :($sym = args[$i]))
        end
    end
    return quote
        $(body...)
        f($(wrapped...))
    end
end

# GPU-compilable closure that broadcast dispatches element-wise over CuArrays.
@inline _gpu_dual_fn(f::F) where {F} = @inline (args...) ->
    _gpu_apply_with_duals(f, args...)

# One fused GPU kernel: evaluates f and all partial derivatives simultaneously.
# Real args use 1 Dual slot each; complex args use 2 (one per real DOF).
function _gpu_broadcast_dual(f::F, args...) where {F}
    _gpu_dual_fn(f).(args...)
end

# Extract primal value from a Dual or Complex{Dual} element; pass others through.
@inline _gpu_dual_val(x::NDual) = ndual_value(x)
@inline _gpu_dual_val(x::Complex{<:NDual}) = complex(
    ndual_value(real(x)), ndual_value(imag(x))
)
@inline _gpu_dual_val(x) = x

@inline _gpu_dual_part_cx(x::NDual{T,N}, k::Int) where {T,N} = ndual_partial(x, k)
@inline _gpu_dual_part_cx(x::Complex{NDual{T,N}}, k::Int) where {T,N} = complex(
    ndual_partial(real(x), k), ndual_partial(imag(x), k)
)
@inline _gpu_dual_part_cx(x, ::Int) = false

# Adjoint contribution to Dual slot k from output cotangent `dy`.
# Computes Re(conj(dy) · P_k) where P_k = _gpu_dual_part_cx(o, k).
# For real dy and real P_k this reduces to dy * P_k (standard chain rule).
@inline _gpu_adj_part(o, dy, k) = real(conj(dy) * _gpu_dual_part_cx(o, k))

# True when the broadcast output element type carries NDual partial information.
@inline _is_gpu_differentiable(::Type{<:NDual}) = true
@inline _is_gpu_differentiable(::Type{<:Complex{<:NDual}}) = true
@inline _is_gpu_differentiable(::Type) = false

# Number of Dual slots contributed by a broadcast leaf arg.  Matches the slot
# assignment in _gpu_apply_with_duals by dispatching on the broadcast element type
# (scalar, CuArray, or Ref all handled via eltype).
@inline _broadcast_elem_dof(x) = _broadcast_elem_dof_type(eltype(x))
@inline _broadcast_elem_dof_type(::Type{<:IEEEFloat}) = 1
@inline _broadcast_elem_dof_type(::Type{<:Complex{<:IEEEFloat}}) = 2
@inline _broadcast_elem_dof_type(::Type) = 0

# ── Adjoint / Transpose leaf helpers ─────────────────────────────────────────────────
#
# When a broadcast leaf is `A'` or `transpose(A)` the GPU kernel element is A'[i,j]
# (a scalar), so the Dual wrapping and partials work unchanged.  The difference is in
# how the gradient is accumulated:
#
#   Plain CuArray:                   fd .+= contrib               (direct, same layout)
#   Transpose{T, CuArray{T}}:        fd.parent .+= transpose(contrib)
#   Adjoint{T, CuArray{T}}  (T<:IEEEFloat):          fd.parent .+= adjoint(contrib)    (= transpose since conj = id for real)
#   Adjoint{T, CuArray{Complex{T}}} (T<:IEEEFloat):  fd.parent .+= adjoint(contrib)    (conj + transpose)
#
# and the JVP tangent must be reindexed the same way:
#   Plain CuArray:   t_eff = t               (t is a CuArray)
#   Transpose:       t_eff = transpose(t)    (t is the parent CuArray tangent)
#   Adjoint:         t_eff = adjoint(t)      (t is the parent CuArray tangent)
# because d(A'[i,j]) = conj(t[j,i]) = adjoint(t)[i,j], d(Aᵀ[i,j]) = t[j,i] = transpose(t)[i,j].

# Forward mode: return the effective tangent seen by the broadcast kernel for leaf pa.
# For Adjoint/Transpose, raw_t is a Tangent{@NamedTuple{parent::CuArray}}; extract parent.
@inline _leaf_effective_tangent(::CuMaybeComplexArray, t::CuArray) = t
@inline _leaf_effective_tangent(::Adjoint{<:CuFloatOrComplex,<:CuMaybeComplexArray}, t) = adjoint(
    _fields(t).parent
)
@inline _leaf_effective_tangent(::Transpose{<:CuFloatOrComplex,<:CuMaybeComplexArray}, t) = transpose(
    _fields(t).parent
)
# Scalar variables broadcast as a uniform constant; their tangent is the scalar itself.
@inline _leaf_effective_tangent(::IEEEFloat, t) = t
@inline _leaf_effective_tangent(::Complex{<:IEEEFloat}, t) = t
@inline _leaf_effective_tangent(_, _) = nothing  # non-differentiable

# Reduce `dx` (broadcast-output shape) back to `sz` by summing over any dimensions that
# were singleton-expanded or added during broadcasting.  Mirrors ChainRules' `unbroadcast`.
#
# Julia broadcasting is left-aligned: a 1D array (n,) broadcast against (n,p) is treated
# as (n,1) — extra trailing dimensions, not extra leading ones.  So "extra" dims are those
# at positions d > length(sz), not d <= n_extra.
function _unbroadcast(dx::CuArray, sz::Tuple)
    size(dx) == sz && return dx
    dims = ntuple(ndims(dx)) do d
        d > length(sz) || sz[d] == 1
    end
    reduce_dims = filter(d -> dims[d], 1:ndims(dx))
    return isempty(reduce_dims) ? reshape(dx, sz) : reshape(sum(dx; dims=reduce_dims), sz)
end

# Reverse mode: accumulate `contrib` (same shape as broadcast output) into leaf fdata.
# Unbroadcast before accumulating so that broadcast-expanded inputs get the correct shape.
@inline function _leaf_accum_fdata!(pa::CuMaybeComplexArray, fd::CuArray, contrib)
    fd .+= _unbroadcast(contrib, size(pa))
end
@inline function _leaf_accum_fdata!(
    pa::Adjoint{<:CuFloatOrComplex,<:CuMaybeComplexArray}, fd, contrib
)
    _fields(fd).parent .+= adjoint(_unbroadcast(contrib, size(pa)))
end
@inline function _leaf_accum_fdata!(
    pa::Transpose{<:CuFloatOrComplex,<:CuMaybeComplexArray}, fd, contrib
)
    _fields(fd).parent .+= transpose(_unbroadcast(contrib, size(pa)))
end
@inline _leaf_accum_fdata!(_, _, _) = nothing  # non-differentiable

# Recursively extract leaf (non-Broadcasted) arg primals and their tangent data from a
# possibly-nested Broadcasted / tangent pair.  Works for both reverse mode (FData, uses
# _fields(td).args) and forward mode (Tangent, uses _fields(td).args) because _fields
# abstracts over both.
@inline _gpu_bcast_leaves(bc, td) = _gpu_bcast_leaves_args(bc.args, _fields(td).args)
@inline _gpu_bcast_leaves_args(::Tuple{}, ::Tuple{}) = ((), ())
@inline function _gpu_bcast_leaves_args(args::Tuple, tds::Tuple)
    a1 = first(args)
    td1 = first(tds)
    rest_ps, rest_ts = _gpu_bcast_leaves_args(Base.tail(args), Base.tail(tds))
    if a1 isa Broadcasted
        inner_ps, inner_ts = _gpu_bcast_leaves(a1, td1)
        return (inner_ps..., rest_ps...), (inner_ts..., rest_ts...)
    else
        return (a1, rest_ps...), (td1, rest_ts...)
    end
end

@is_primitive(
    MinimalCtx, Tuple{typeof(Base.Broadcast.materialize),<:Broadcasted{<:CuArrayStyle}},
)

# Build rdata for bc_primal by walking its args tree depth-first (same order as
# _gpu_bcast_leaves), replacing scalar IEEEFloat/Complex leaves with their actual
# gradients from scalar_grads and filling everything else with zero_rdata.
function _gpu_fill_scalar_rdata(
    bc::Broadcasted, scalar_grads::AbstractVector, idx::Ref{Int}
)
    r_args = _gpu_fill_args_rdata(bc.args, scalar_grads, idx)
    return RData((;
        style=zero_rdata(bc.style),
        f=zero_rdata(bc.f),
        args=r_args,
        axes=zero_rdata(bc.axes),
    ))
end
function _gpu_fill_args_rdata(args::Tuple, scalar_grads, idx::Ref{Int})
    a1 = first(args)
    r1 = if a1 isa Broadcasted
        _gpu_fill_scalar_rdata(a1, scalar_grads, idx)
    elseif a1 isa IEEEFloat || a1 isa Complex{<:IEEEFloat}
        g = scalar_grads[idx[]]
        idx[] += 1
        g
    else
        zero_rdata(a1)
    end
    return (r1, _gpu_fill_args_rdata(Base.tail(args), scalar_grads, idx)...)
end
function _gpu_fill_args_rdata(::Tuple{}, ::Any, ::Ref{Int})
    return ()
end

# Detect mixed-eltype GPU broadcasts: when CuArray leaves have different element types
# (e.g. Float32 and Float64 in the same broadcast), the Dual wrapping would produce
# incompatible Dual widths and cause a cryptic GPU compiler error.  Raise a clear error.
# Note: scalar args (IEEEFloat/Complex) are not checked here; a Float64 scalar mixed
# with a Float32 CuArray silently promotes the broadcast to Float64, which may be slow
# or unsupported on some GPUs.  Cast the scalar explicitly if needed.
function _check_mixed_gpu_eltype(flat_pargs)
    gpu_ets = [
        eltype(pa) for pa in flat_pargs if pa isa CuMaybeComplexArray ||
        pa isa Adjoint{<:CuFloatOrComplex,<:CuMaybeComplexArray} ||
        pa isa Transpose{<:CuFloatOrComplex,<:CuMaybeComplexArray}
    ]
    length(unique(gpu_ets)) <= 1 && return nothing
    throw(
        ArgumentError(
            "Mooncake: GPU broadcast over arrays with mixed element types " *
            "($(join(gpu_ets, ", "))) is not supported. " *
            "Cast all inputs to the same type before broadcasting.",
        ),
    )
end

function frule!!(
    ::Dual{typeof(Base.Broadcast.materialize)}, bc::Dual{<:Broadcasted{<:CuArrayStyle}}
)
    bc_primal = primal(bc)
    flat_bc = Base.Broadcast.flatten(bc_primal)
    # flat_pargs == flat_bc.args (both depth-first leaves of bc_primal); _gpu_bcast_leaves
    # is used here solely to also obtain the paired tangent data flat_ts.
    flat_pargs, flat_ts = _gpu_bcast_leaves(bc_primal, tangent(bc))
    _check_mixed_gpu_eltype(flat_pargs)

    # One GPU kernel: compute primal AND all partial derivatives simultaneously.
    # Real args use 1 Dual slot each; complex args use 2 (one per real DOF).
    out = _gpu_broadcast_dual(flat_bc.f, flat_pargs...)

    # Non-differentiable output (e.g. Bool from comparisons): zero tangent.
    if !_is_gpu_differentiable(eltype(out))
        return Dual(out, NoTangent())
    end

    y = broadcast(_gpu_dual_val, out)
    # JVP: dy = Σᵢ Σₛ∈slots(i) (∂f/∂xₛ · δxₛ)
    # Real CuArray input: 1 slot, tangent t is real.
    # Complex CuArray input: 2 slots, tangent t is complex; δRe=real(t), δIm=imag(t).
    dy = zero(y)
    offset = 0
    for (pa, t) in zip(flat_pargs, flat_ts)
        dof = _broadcast_elem_dof(pa)
        t_eff = _leaf_effective_tangent(pa, t)
        if t_eff !== nothing
            if dof == 1
                k = offset + 1
                dy .+= broadcast(o -> _gpu_dual_part_cx(o, k), out) .* t_eff
            elseif dof == 2
                k1, k2 = offset + 1, offset + 2
                dy .+= broadcast(o -> _gpu_dual_part_cx(o, k1), out) .* real.(t_eff)
                dy .+= broadcast(o -> _gpu_dual_part_cx(o, k2), out) .* imag.(t_eff)
            end
        end
        offset += dof
    end
    return Dual(y, dy)
end

function rrule!!(
    mat_fn::CoDual{typeof(Base.Broadcast.materialize)},
    bc::CoDual{<:Broadcasted{<:CuArrayStyle}},
)
    bc_primal = primal(bc)
    bc_fdata = tangent(bc)

    # Flatten nested Broadcasted trees into a single composed function + leaf args.
    # flat_pargs == flat_bc.args; _gpu_bcast_leaves is used to also obtain flat_fdatas.
    flat_bc = Base.Broadcast.flatten(bc_primal)
    flat_pf = flat_bc.f
    flat_pargs, flat_fdatas = _gpu_bcast_leaves(bc_primal, bc_fdata)
    _check_mixed_gpu_eltype(flat_pargs)

    # One GPU kernel: compute primal AND all N partial derivatives simultaneously.
    out = _gpu_broadcast_dual(flat_pf, flat_pargs...)

    # Non-differentiable output (e.g. Bool from x .!= 0): return zero gradients.
    if !_is_gpu_differentiable(eltype(out))
        return CoDual(out, NoFData()), NoPullback(mat_fn, bc)
    end

    y = broadcast(_gpu_dual_val, out)
    dy_out = zero(y)  # accumulated into by the downstream reverse pass

    # Pre-extract per-slot partial arrays from the NDual output.  Each pullback kernel
    # then reads a plain CuArray{T} instead of the (N+1)-wide NDual array, reducing GPU
    # memory bandwidth in the pullback by ~(N+1)× per slot iteration.
    n_slots = sum(_broadcast_elem_dof, flat_pargs)
    partial_slots = [broadcast(o -> _gpu_dual_part_cx(o, k), out) for k in 1:n_slots]
    out = nothing  # primal is in y; partials are in partial_slots — NDual array can be freed

    # Pre-check at rrule construction time to avoid Any[] allocation in the common case.
    has_scalars = any(pa isa CuFloatOrComplex for pa in flat_pargs)

    function materialize_pb!!(::NoRData)
        # Walk flat_pargs in order, tracking the cumulative Dual-slot offset.
        # CuArray / Adjoint / Transpose inputs accumulate via _leaf_accum_fdata!.
        # Scalar IEEEFloat / Complex inputs have no fdata slot; their gradients are
        # collected here and returned via rdata (packed into the Broadcasted rdata tree).
        scalar_grads = has_scalars ? Any[] : nothing
        offset = 0
        for (pa, fd) in zip(flat_pargs, flat_fdatas)
            dof = _broadcast_elem_dof(pa)
            if dof == 1
                k = offset + 1
                contrib = broadcast((p, d) -> real(conj(d) * p), partial_slots[k], dy_out)
                if pa isa IEEEFloat
                    push!(scalar_grads::Vector{Any}, sum(contrib))
                else
                    _leaf_accum_fdata!(pa, fd, contrib)
                end
            elseif dof == 2
                k1, k2 = offset + 1, offset + 2
                contrib = broadcast(
                    (p1, p2, d) -> complex(real(conj(d) * p1), real(conj(d) * p2)),
                    partial_slots[k1],
                    partial_slots[k2],
                    dy_out,
                )
                if pa isa Complex{<:IEEEFloat}
                    push!(scalar_grads::Vector{Any}, sum(contrib))
                else
                    _leaf_accum_fdata!(pa, fd, contrib)
                end
            end
            offset += dof
        end
        r_bc = if isnothing(scalar_grads)
            zero_rdata(bc_primal)
        else
            _gpu_fill_scalar_rdata(bc_primal, scalar_grads, Ref(1))
        end
        return NoRData(), r_bc
    end

    return CoDual(y, dy_out), materialize_pb!!
end

# In-place GPU broadcast: Base.Broadcast.materialize!(dest, bc) is what
# broadcast!(f, dest, args...) calls after constructing bc = broadcasted(f, args...).
#
# Intercepting here (rather than at broadcast! level) is cleaner: we receive an
# already-constructed Broadcasted and can reuse _gpu_bcast_leaves exactly like the
# materialize rrule, with no need to manually rebuild the Broadcasted from raw args.
#
# The rule mirrors the materialize rrule but writes the primal result into the
# pre-allocated `dest` and uses tangent(dest) as the gradient accumulator.
#
# ALIASING: `dest` may appear in bc.args (e.g. x .= f.(x, y)).  The pullback
# handles this correctly: contribs are computed from dual_out + dout, captured in
# the closure BEFORE dout is zeroed.  The frule accumulates contributions into a
# temporary before writing to dout, for the same reason.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(Base.Broadcast.materialize!),P,<:Broadcasted{<:CuArrayStyle}
    } where {P<:CuMaybeComplexArray},
)
function frule!!(
    ::Dual{typeof(Base.Broadcast.materialize!)},
    dest::Dual{P,P},
    bc::Dual{<:Broadcasted{<:CuArrayStyle}},
) where {P<:CuMaybeComplexArray}
    bc_primal = primal(bc)
    flat_bc = Base.Broadcast.flatten(bc_primal)
    flat_pargs, flat_ts = _gpu_bcast_leaves(bc_primal, tangent(bc))
    _check_mixed_gpu_eltype(flat_pargs)

    dual_out = _gpu_broadcast_dual(flat_bc.f, flat_pargs...)
    pout, dout = primal(dest), tangent(dest)

    # Write primal result in-place into dest.
    broadcast!(_gpu_dual_val, pout, dual_out)

    # Non-differentiable output (e.g. Bool arrays): zero the tangent and return.
    if !_is_gpu_differentiable(eltype(dual_out))
        fill!(dout, 0)
        return dest
    end

    # JVP: accumulate into a temporary to handle aliasing (dest may appear in
    # bc.args, so flat_ts may contain a reference to dout; we must not overwrite
    # dout until all contributions have been read from the old tangent values).
    dy = zero(pout)
    offset = 0
    for (pa, t) in zip(flat_pargs, flat_ts)
        dof = _broadcast_elem_dof(pa)
        t_eff = _leaf_effective_tangent(pa, t)
        if t_eff !== nothing
            if dof == 1
                k = offset + 1
                dy .+= broadcast(o -> _gpu_dual_part_cx(o, k), dual_out) .* t_eff
            elseif dof == 2
                k1, k2 = offset + 1, offset + 2
                dy .+= broadcast(o -> _gpu_dual_part_cx(o, k1), dual_out) .* real.(t_eff)
                dy .+= broadcast(o -> _gpu_dual_part_cx(o, k2), dual_out) .* imag.(t_eff)
            end
        end
        offset += dof
    end
    copyto!(dout, dy)
    return dest
end
function rrule!!(
    ::CoDual{typeof(Base.Broadcast.materialize!),NoFData},
    dest::CoDual{P,P},
    bc::CoDual{<:Broadcasted{<:CuArrayStyle}},
) where {P<:CuMaybeComplexArray}
    pout, dout = primal(dest), tangent(dest)
    bc_primal = primal(bc)
    bc_fdata = tangent(bc)

    flat_bc = Base.Broadcast.flatten(bc_primal)
    flat_pf = flat_bc.f
    flat_pargs, flat_fdatas = _gpu_bcast_leaves(bc_primal, bc_fdata)
    _check_mixed_gpu_eltype(flat_pargs)

    # Save primal for restoration in the pullback.
    old_pout = copy(pout)

    # Single GPU kernel: primal + all partial derivatives simultaneously.
    dual_out = _gpu_broadcast_dual(flat_pf, flat_pargs...)

    # Write primal result in-place into dest.
    broadcast!(_gpu_dual_val, pout, dual_out)

    # Non-differentiable output (e.g. Bool arrays): no gradient to propagate.
    # Check eltype(dual_out) (NDual elements), NOT eltype(pout) (plain floats after
    # _gpu_dual_val extraction): eltype(pout) is always IEEEFloat for CuMaybeComplexArray.
    if !_is_gpu_differentiable(eltype(dual_out))
        function materialize!_nodiff_pb!!(::NoRData)
            copyto!(pout, old_pout)
            return NoRData(), NoRData(), zero_rdata(bc_primal)
        end
        return dest, materialize!_nodiff_pb!!
    end

    has_scalars = any(pa isa CuFloatOrComplex for pa in flat_pargs)

    # Pre-extract per-slot partial arrays (same bandwidth rationale as materialize rrule).
    n_slots = sum(_broadcast_elem_dof, flat_pargs)
    partial_slots = [broadcast(o -> _gpu_dual_part_cx(o, k), dual_out) for k in 1:n_slots]
    dual_out = nothing  # primal written to pout; partials in partial_slots — NDual array can be freed

    function materialize!_pb!!(::NoRData)
        # Snapshot dout before any modifications. When dest appears in bc.args
        # (e.g. x .= x .+ y), flat_fdatas contains fd = dout for x's slot.
        # Without a snapshot, _leaf_accum_fdata!(x, dout, contrib) would corrupt
        # dout mid-loop, causing subsequent slots to read a doubled value, and
        # fill!(dout, 0) at the end would then zero x's gradient entirely.
        # Zeroing dout first and then accumulating avoids both problems.
        g = copy(dout)
        fill!(dout, 0)
        scalar_grads = has_scalars ? Any[] : nothing
        offset = 0
        for (pa, fd) in zip(flat_pargs, flat_fdatas)
            dof = _broadcast_elem_dof(pa)
            if dof == 1
                k = offset + 1
                contrib = broadcast((p, d) -> real(conj(d) * p), partial_slots[k], g)
                if pa isa IEEEFloat
                    push!(scalar_grads::Vector{Any}, sum(contrib))
                else
                    _leaf_accum_fdata!(pa, fd, contrib)
                end
            elseif dof == 2
                k1, k2 = offset + 1, offset + 2
                contrib = broadcast(
                    (p1, p2, d) -> complex(real(conj(d) * p1), real(conj(d) * p2)),
                    partial_slots[k1],
                    partial_slots[k2],
                    g,
                )
                if pa isa Complex{<:IEEEFloat}
                    push!(scalar_grads::Vector{Any}, sum(contrib))
                else
                    _leaf_accum_fdata!(pa, fd, contrib)
                end
            end
            offset += dof
        end
        r_bc = if isnothing(scalar_grads)
            zero_rdata(bc_primal)
        else
            _gpu_fill_scalar_rdata(bc_primal, scalar_grads, Ref(1))
        end
        # Restore primal to allow the reverse pass to see the pre-broadcast value.
        copyto!(pout, old_pout)
        return NoRData(), NoRData(), r_bc
    end

    return dest, materialize!_pb!!
end

end
