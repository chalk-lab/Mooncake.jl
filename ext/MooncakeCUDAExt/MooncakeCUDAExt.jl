module MooncakeCUDAExt

using LinearAlgebra, Random, Mooncake

using Base: IEEEFloat, unsafe_convert
using CUDA.CUDACore:
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
    capture_status,
    hasfieldcount
using CUDA: CUDACore
using CUDA: cuBLAS
using CUDA: cuSPARSE
using CUDA: cuSOLVER
using CUDA.CUDACore.GPUArrays: unsafe_free!
using Base.Broadcast: Broadcasted
# Statistics is a weakdep triggering this extension alongside CUDA; always loaded
# transitively via GPUArrays anyway, just needed for Aqua's stale-deps check.
import Statistics: mean, varm
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
    nan_tangent_guard,
    NDual,
    Nfwd

import Mooncake.TestUtils:
    populate_address_map_internal, AddressMap, __increment_should_allocate

# NDual lives in Mooncake.Nfwd and is loaded as part of Mooncake core.

const CuFloatArray = CuArray{<:IEEEFloat}
const CuComplexArray = CuArray{<:Complex{<:IEEEFloat}}
const CuMaybeComplexArray = Union{CuFloatArray,CuComplexArray}

# Without these overloads the generic struct handler would recurse into CuMaybeComplexArray's
# Julia-visible fields — wrong for GPU arrays.
Mooncake._copy_output(x::CuMaybeComplexArray) = copy(x)
function Mooncake._copy_to_output!!(dst::P, src::P) where {P<:CuMaybeComplexArray}
    copyto!(dst, src)
    return dst
end

const CuMaybeComplexVec = Union{CuArray{<:IEEEFloat,1},CuArray{<:Complex{<:IEEEFloat},1}}
const CuMaybeComplexMat = Union{CuArray{<:IEEEFloat,2},CuArray{<:Complex{<:IEEEFloat},2}}
const CuFloatOrComplex = Union{IEEEFloat,Complex{<:IEEEFloat}}
const CuGpuSumFArray = Union{
    CuFloatArray,
    CuComplexArray,
    Adjoint{<:IEEEFloat,<:CuFloatArray},
    Transpose{<:IEEEFloat,<:CuFloatArray},
}

# vcat/hcat/cat/permutedims also accept these wrapper types; without them here the wrappers
# fall through to the interpreter and hit the same untraceable `cufunction` try/finally the
# rules below exist to avoid. `Mooncake.arrayify` canonicalises each argument (generic
# Adjoint/Transpose/SubArray overloads in `src/rules/blas.jl`), and the bounds mirror its
# own — `Union{IEEEFloat,BlasFloat}` for Adjoint/Transpose, `BlasFloat` alone for SubArray —
# since a wider bound would claim a call as primitive only to fail inside the rule with a
# misdirected `arrayify` error instead of falling through to the interpreter.
const CuMaybeWrappedArray = Union{
    CuMaybeComplexArray,
    Adjoint{<:Union{IEEEFloat,LinearAlgebra.BlasFloat},<:CuMaybeComplexArray},
    Transpose{<:Union{IEEEFloat,LinearAlgebra.BlasFloat},<:CuMaybeComplexArray},
    SubArray{T,N,P} where {T<:LinearAlgebra.BlasFloat,N,P<:CuMaybeComplexArray},
}

@inline _nopb(::Val{N}) where {N} = NoPullback(ntuple(_ -> NoRData(), N))
@noinline _throw_gpu_argument_error(msg::AbstractString) = throw(ArgumentError(msg))

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
# zero_tangent_internal for CuDataRef: returns copy(x), which increments the refcount and
# shares the same underlying GPU buffer as the primal.  This is NOT a true zero buffer —
# it is an alias of the primal's memory.  It is safe only because DataRef tangents are
# fully opaque: lgetfield returns NoTangent for every field, so no gradient operation ever
# writes through a DataRef tangent directly.  All actual gradient accumulation goes via the
# enclosing CuMaybeComplexArray rule, which allocates its own freshly-zeroed GPU array and
# sets .data to a DataRef for that new buffer — so by the time gradient accumulation runs,
# the DataRef tangent has already been replaced and copy(x) is never written to.
zero_tangent_internal(x::T, ::MaybeCache) where {T<:CuDataRef} = copy(x)

# Base._check_mutable(o) is called by GPUArrays.derive on Julia 1.10 to assert that the
# array is mutable before constructing a view.  It returns nothing and has no differentiable
# content; Mooncake cannot trace it on 1.10 because it contains an internal intrinsic.
# Removed from the call path in Julia 1.11+.
@static if VERSION < v"1.11-"
    @zero_derivative MinimalCtx Tuple{typeof(Base._check_mutable),Any}
end

# copy(::CuDataRef) is called by GPUArrays.derive (which backs view, reinterpret, and
# similar operations) for reference-count management.  It is a bookkeeping operation —
# the primal copy increments the refcount; the tangent DataRef is also copied so that the
# new CuArray's .data field holds a separate handle to the same tangent GPU memory.
@is_primitive(MinimalCtx, Tuple{typeof(copy),<:CuDataRef})
function frule!!(::Dual{typeof(copy)}, x::Dual{<:CuDataRef,<:CuDataRef})
    return Dual(copy(primal(x)), copy(tangent(x)))
end
function rrule!!(::CoDual{typeof(copy)}, x::CoDual{<:CuDataRef,<:CuDataRef})
    return CoDual(copy(primal(x)), copy(tangent(x))), _nopb(Val(2))
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
    return CuPtr{tangent_type(T)}(UInt64(0))
end

# unsafe_convert(::Type{CuPtr{T}}, x::CuArray{T}):
# Returns a raw device pointer to x's data buffer.  For AD, the fdata of the returned
# CuPtr is the pointer to the tangent buffer — both primal and tangent CuArrays have
# the same layout, so unsafe_convert on the tangent gives the correct tangent pointer.
# Needed because the traced body accesses DataRef internals (llvmcall) and loses the
# tangent, causing a CoDual{CuPtr{T}, CuPtr{T}} ← CoDual{CuPtr{T}, NoFData} TypeError.
#
# The rules use CoDual{X,X} where X<:CuArray{T} rather than CoDual{CuArray{T},CuArray{T}}
# because Julia's type parameters are invariant: CuArray{Float32,2,Mem} ≠ CuArray{Float32}
# as a parameter, so the latter signature would never match a concrete CuArray argument.
@is_primitive(
    MinimalCtx,
    Tuple{typeof(unsafe_convert),Type{CuPtr{T}},CuArray{T}} where {T<:IEEEFloat},
)
@is_primitive(
    MinimalCtx,
    Tuple{typeof(unsafe_convert),Type{CuPtr{T}},CuArray{T}} where {T<:Complex{<:IEEEFloat}},
)
function frule!!(
    ::Dual{typeof(unsafe_convert)}, ::Dual{Type{CuPtr{T}}}, x::Dual{X,X}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},X<:CuArray{T}}
    return Dual(unsafe_convert(CuPtr{T}, primal(x)), unsafe_convert(CuPtr{T}, tangent(x)))
end
function rrule!!(
    ::CoDual{typeof(unsafe_convert)}, ::CoDual{Type{CuPtr{T}}}, x::CoDual{X,X}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},X<:CuArray{T}}
    return CoDual(unsafe_convert(CuPtr{T}, primal(x)), unsafe_convert(CuPtr{T}, x.dx)),
    _nopb(Val(3))
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
    return CoDual(primal(p) + primal(n), tangent(p) + primal(n)), _nopb(Val(3))
end
function rrule!!(
    ::CoDual{typeof(+)}, p::CoDual{CuPtr{T},NoFData}, n::CoDual{<:Integer,NoFData}
) where {T}
    return CoDual(primal(p) + primal(n), NoFData()), _nopb(Val(3))
end

# Non-differentiable CUDA handle, enum, and state types.
#
# Opaque pointer types (Ptr{X}): Mooncake's default tangent_type(::Type{Ptr{P}}) returns
# Ptr{tangent_type(P)}, and zero_tangent_internal(::Ptr, ::MaybeCache) throws
# unconditionally.  Both must be overridden for each concrete opaque pointer type.
#
# Only the non-primitive opaque C pointer types need explicit registration here; all
# @cenum (primitive) types are handled by the programmatic loop further below.
function _register_cuda_opaque_types!()
    for (_cuda_opaque_t, _is_ptr) in [
        # Opaque C handle/descriptor Ptr types (cuBLAS)
        (CUmemPoolHandle_st, true),
        (cuBLAS.cublasContext, true),
        (cuBLAS.cublasXtContext, true),
        # Opaque C handle/descriptor Ptr types (cuSPARSE)
        (cuSPARSE.cusparseContext, true),
        (cuSPARSE.cusparseMatDescr, true),
        (cuSPARSE.bsrsv2Info, true),
        (cuSPARSE.bsrsm2Info, true),
        (cuSPARSE.csric02Info, true),
        (cuSPARSE.bsric02Info, true),
        (cuSPARSE.csrilu02Info, true),
        (cuSPARSE.bsrilu02Info, true),
        (cuSPARSE.csru2csrInfo, true),
        (cuSPARSE.cusparseColorInfo, true),
        (cuSPARSE.pruneInfo, true),
        (cuSPARSE.cusparseSpVecDescr, true),
        (cuSPARSE.cusparseDnVecDescr, true),
        (cuSPARSE.cusparseSpMatDescr, true),
        (cuSPARSE.cusparseDnMatDescr, true),
        (cuSPARSE.cusparseSpSVDescr, true),
        (cuSPARSE.cusparseSpSMDescr, true),
        (cuSPARSE.cusparseSpGEMMDescr, true),
        (cuSPARSE.cusparseSpMMOpPlan, true),
        # CuStream contains Ptr/Bool/CuContext fields; without NoTangent, Mooncake
        # generates a MutableTangent that propagates into task-local CUDA state.
        (CuStream, false),
        # TaskLocalState bundles device index, stream handles, and library contexts.
        (TaskLocalState, false),
        # CuContext wraps an opaque Ptr{Cvoid} to the CUDA context.
        (CuContext, false),
        # Opaque C handle/descriptor Ptr types (cuSOLVER)
        (cuSOLVER.syevjInfo_t, true),
        (cuSOLVER.gesvdjInfo_t, true),
        (cuSOLVER.cusolverDnIRSParams_t, true),
        (cuSOLVER.cusolverDnIRSInfos_t, true),
        (cuSOLVER.cusolverDnParams_t, true),
    ]
        if _is_ptr
            @eval tangent_type(::Type{Ptr{$_cuda_opaque_t}}) = NoTangent
            @eval zero_tangent_internal(::Ptr{$_cuda_opaque_t}, ::MaybeCache) = NoTangent()
        else
            @eval tangent_type(::Type{$_cuda_opaque_t}) = NoTangent
        end
    end
    return nothing
end

_register_cuda_opaque_types!()

# CUDA @cenum types are primitive types (integer-backed C enums) — never differentiable.
# Mooncake's generic tangent_type @generated function errors on primitive types with no
# registered method, so we register all of them here programmatically.
# Covers: CUDACore, cuBLAS, cuSPARSE, cuSOLVER.
# cuDNN enums are handled in MooncakeCUDNNExt (loaded only when cuDNN is available).
# Filter: parentmodule(T) must be one of the CUDA family modules, to avoid accidentally
# re-registering standard Julia primitive types (Bool, Int32, Float64, ...) that happen
# to be visible in the CUDA namespace.
function _register_cuda_enum_types!()
    let _cuda_family = (CUDACore, cuBLAS, cuSPARSE, cuSOLVER)
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
                _T <: CUDACore.CEnum.Cenum || continue
                parentmodule(_T) in _cuda_family || continue
                _T in _cenum_seen && continue
                push!(_cenum_seen, _T)
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
    return nothing
end

_register_cuda_enum_types!()

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
function _register_cudataref_internal_types!()
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
            return nothing
        end
        for _T in Base.uniontypes(CuDataRef)
            for _i in 1:fieldcount(_T)
                _register_cuda_internal!(fieldtype(_T, _i))
            end
        end
    end
    return nothing
end

_register_cudataref_internal_types!()

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
# cuBLAS.version() queries the runtime library version via cublasGetProperty (a ccall).
# Returns a constant VersionNumber — not differentiable.
@zero_derivative MinimalCtx Tuple{typeof(cuBLAS.version)}
# Library handle() functions retrieve per-task C pointers to cuBLAS/cuSPARSE contexts.
@zero_derivative MinimalCtx Tuple{typeof(cuBLAS.handle)}
@zero_derivative MinimalCtx Tuple{typeof(cuSPARSE.handle)}
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
    return (x, dx)
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
    x::P, y::P, equal_undefs::Bool, d::IdDict{Any,Bool}
) where {P<:CuMaybeComplexArray}
    # allow nan comparisons to return true, real() to cover complex case
    return isapprox(x, y; atol=(√eps(real(eltype(P)))), nans=true)
end
function TestUtils.has_equal_data_internal(
    x::P, y::P, equal_undefs::Bool, d::IdDict{Any,Bool}
) where {P<:CuArray{<:Union{Integer,Bool}}}
    # For integer/bool CuArrays, compare by content by downloading to CPU.
    size(x) != size(y) && return false
    return Array(x) == Array(y)
end
# The array-array bookkeeping broadcasts here and below (`x .+= y`, `t .= x`, `x .= t`)
# need no `set_to_zero!!`-style `fill!` workaround: they form `Broadcasted{CuArrayStyle}`,
# so under HVP they hit the `materialize!` rule and route to the `_gpu_broadcast_dual`
# chokepoint (a clear error), never a raw kernel escaping to `cufunction`.
function increment_internal!!(c::IncCache, x::A, y::A) where {A<:CuMaybeComplexArray}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x .+= y
    return x
end
__increment_should_allocate(::Type{<:CuMaybeComplexArray}) = true
# Use `fill!` (has Mooncake rules), not `x .= 0`: under HVP `set_to_zero!!` is itself
# differentiated, and the broadcast inlines to a raw kernel launch whose forward-mode
# trace descends into `cufunction`; `fill!` dispatches to its frule!! instead.
set_to_zero_internal!!(::Mooncake.SetToZeroCache, x::CuMaybeComplexArray) = fill!(x, 0)

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
    return CoDual(reshape(primal(x), _dims), reshape(x.dx, _dims)), _nopb(Val(3))
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
    return CoDual(y, dy), _nopb(Val(6))
end

# lgetfield rules for DataRef.  DataRef has three fields: :rc (ref count Atomic{Int}),
# :freed (Bool), :cached (the wrapped memory object, e.g. Managed{DeviceMemory}).
# All are reference-counting internals — no derivative flows through them.
# tangent_type(DataRef) = DataRef (opaque handle), so the tangent is the DataRef itself;
# field accesses return NoTangent/NoFData.
@inline _cu_lgetfield_primal(x, name, ::Nothing) = getfield(x, name)
@inline _cu_lgetfield_primal(x, name, order) = getfield(x, name, order)
@inline _cuarray_is_data_field(name) = name === 1 || name === :data
@inline _cu_lgetfield_data_tangent(dx::CuArray, name) =
    _cuarray_is_data_field(name) ? dx.data : NoTangent()
@inline _cu_lgetfield_data_fdata(dx::CuArray, name) =
    _cuarray_is_data_field(name) ? dx.data : NoFData()

@inline _cudataref_lgetfield_fwd(x_primal, name, order=nothing) = Dual(
    _cu_lgetfield_primal(x_primal, name, order), NoTangent()
)
@inline _cudataref_lgetfield_rev(x_primal, name, order=nothing) = CoDual(
    _cu_lgetfield_primal(x_primal, name, order), NoFData()
)
@inline _cuarray_lgetfield_fwd(x_primal, x_tangent, name, order=nothing) = Dual(
    _cu_lgetfield_primal(x_primal, name, order), _cu_lgetfield_data_tangent(x_tangent, name)
)
@inline _cuarray_lgetfield_rev(x_primal, x_fdata, name, order=nothing) = CoDual(
    _cu_lgetfield_primal(x_primal, name, order), _cu_lgetfield_data_fdata(x_fdata, name)
)

function frule!!(
    ::Dual{typeof(lgetfield)},
    x::Dual{<:CuDataRef,<:CuDataRef},
    ::Dual{Val{name}},
    ::Dual{Val{order}},
) where {name,order}
    return _cudataref_lgetfield_fwd(primal(x), name, order)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:CuDataRef,<:CuDataRef},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    return _cudataref_lgetfield_rev(primal(x), name, order), _nopb(Val(4))
end
function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{<:CuDataRef,<:CuDataRef}, ::Dual{Val{name}}
) where {name}
    return _cudataref_lgetfield_fwd(primal(x), name)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{<:CuDataRef,<:CuDataRef}, ::CoDual{Val{name}}
) where {name}
    return _cudataref_lgetfield_rev(primal(x), name), _nopb(Val(3))
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
    return _cuarray_lgetfield_fwd(primal(x), tangent(x), name, order)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:CuArray,<:CuArray},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    return _cuarray_lgetfield_rev(primal(x), x.dx, name, order), _nopb(Val(4))
end

function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{<:CuArray,<:CuArray}, ::Dual{Val{name}}
) where {name}
    return _cuarray_lgetfield_fwd(primal(x), tangent(x), name)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{<:CuArray,<:CuArray}, ::CoDual{Val{name}}
) where {name}
    return _cuarray_lgetfield_rev(primal(x), x.dx, name), _nopb(Val(3))
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
    return _throw_gpu_argument_error(_SCALAR_IDX_MSG)
end
function rrule!!(::CoDual{typeof(getindex)}, x::CoDual{<:CuArray}, i::CoDual{<:Integer})
    return _throw_gpu_argument_error(_SCALAR_IDX_MSG)
end

@is_primitive(MinimalCtx, Tuple{typeof(setindex!),CuArray,Any,Integer})
function frule!!(::Dual{typeof(setindex!)}, x::Dual{<:CuArray}, v::Dual, i::Dual{<:Integer})
    return _throw_gpu_argument_error(_SCALAR_IDX_MSG)
end
function rrule!!(
    ::CoDual{typeof(setindex!)}, x::CoDual{<:CuArray}, v::CoDual, i::CoDual{<:Integer}
)
    return _throw_gpu_argument_error(_SCALAR_IDX_MSG)
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
    @eval frule!!(::Dual{typeof($_fn)}, x::Dual{<:CuArray}; kwargs...) = _throw_gpu_argument_error(
        "Mooncake: $_fn on CuArray is not yet differentiable. " * _UNIMPL_MSG
    )
    @eval rrule!!(::CoDual{typeof($_fn)}, x::CoDual{<:CuArray}; kwargs...) = _throw_gpu_argument_error(
        "Mooncake: $_fn on CuArray is not yet differentiable. " * _UNIMPL_MSG
    )
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
    # Pre-compute once at rule construction time: reused on every pullback call.
    # nan_tangent_guard: where px == 0 the product is annihilated (zero gradient).
    inv_cx_px = nan_tangent_guard.(px, inv.(conj.(px)))
    function cumprod_pb!!(::NoRData)
        # Wirtinger chain rule: Δxᵢ = (1/conj(xᵢ)) · Σₖ≥ᵢ Δyₖ · conj(yₖ)
        # i.e. dx .+= reverse(cumsum(reverse(dy .* conj.(y)))) ./ conj.(px)
        # For real inputs conj is a no-op, so this is backward compatible.
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
    return _throw_gpu_argument_error(
        "Mooncake: accumulate on CuArray only supports op=+; got op=$(primal(op)). " *
        _UNIMPL_MSG,
    )
end
function rrule!!(::CoDual{typeof(accumulate)}, op::CoDual, x::CoDual{<:CuArray}; kwargs...)
    return _throw_gpu_argument_error(
        "Mooncake: accumulate on CuArray only supports op=+; got op=$(primal(op)). " *
        _UNIMPL_MSG,
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

# unsafe_free! releases GPU memory early (normally handled by GC finalizer).
# It is a pure side-effect with no mathematical output — gradient is zero.
# Both the primal and its fdata (if any) are independent GPU allocations; free both.
@is_primitive MinimalCtx Tuple{typeof(unsafe_free!),CuArray}
function frule!!(::Dual{typeof(unsafe_free!)}, x::Dual{<:CuArray})
    unsafe_free!(primal(x))
    dx = tangent(x)
    dx isa NoFData || unsafe_free!(dx)
    return Dual(nothing, NoTangent())
end
function rrule!!(::CoDual{typeof(unsafe_free!)}, x::CoDual{<:CuArray})
    unsafe_free!(primal(x))
    dx = tangent(x)
    dx isa NoFData || unsafe_free!(dx)
    return CoDual(nothing, NoFData()), _nopb(Val(2))
end

# Core.finalizer(f, x) registers f as a GC finalizer for x. This is a pure side-effect
# (no mathematical output) encountered inside CuArray constructors (e.g. view/derive).
# The primal registration must happen; the gradient is zero.
@is_primitive MinimalCtx Tuple{typeof(Core.finalizer),Any,Any}
function frule!!(::Dual{typeof(Core.finalizer)}, f::Dual, x::Dual)
    Core.finalizer(primal(f), primal(x))
    return Dual(nothing, NoTangent())
end
function rrule!!(::CoDual{typeof(Core.finalizer)}, f::CoDual, x::CoDual)
    Core.finalizer(primal(f), primal(x))
    return CoDual(nothing, NoFData()), _nopb(Val(3))
end

# CUDA.hasfieldcount (imported as hasfieldcount) checks whether fieldcount(T) is valid for
# type T.
# It contains a try/catch block which causes Mooncake's IR transformation to produce
# invalid IR ("terminator not last in block"). Mark as primitive: returns Bool, no gradient.
@is_primitive MinimalCtx Tuple{typeof(hasfieldcount),Type}
function frule!!(::Dual{typeof(hasfieldcount)}, T::Dual{<:Type})
    return Dual(hasfieldcount(primal(T)), NoTangent())
end
function rrule!!(::CoDual{typeof(hasfieldcount)}, T::CoDual{<:Type})
    return CoDual(hasfieldcount(primal(T)), NoFData()), _nopb(Val(2))
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
@is_primitive MinimalCtx Tuple{typeof(fill!),CuMaybeWrappedArray,Any}
function frule!!(::Dual{typeof(fill!)}, a::Dual{<:CuMaybeWrappedArray}, x::Dual)
    pa, da = arrayify(a)
    fill!(pa, primal(x))
    tx = tangent(x)
    fill!(da, tx isa NoTangent ? zero(eltype(da)) : eltype(da)(tx))
    return a
end
function rrule!!(::CoDual{typeof(fill!)}, a::CoDual{<:CuMaybeWrappedArray}, x::CoDual)
    # Through the wrapper, so `sum(da)` sums the destination's own elements and the restore
    # writes back through it — conjugating for an `Adjoint`, as its `setindex!` does.
    pa, da = arrayify(a)
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
#
# Limitation: the NDual pass threads duals only through the CuArray *elements*.
# Scalars or other state captured inside f's closure are invisible to the kernel
# and receive no gradient.  If f has differentiable captured variables
# (rdata_type(f) ≠ NoRData), the pullback would silently return zero for them,
# producing wrong gradients and a type mismatch in increment!!.
# _check_gpu_sum_f detects this case early and raises an informative error.
function _check_gpu_sum_f(f)
    F = typeof(f)
    # Zero-field types (singletons such as typeof(abs2), typeof(sin)) have no captured
    # state and are always safe. Calling rdata_type on them can hit an internal
    # fields_type MethodError for plain function types, so we short-circuit here.
    fieldcount(F) == 0 && return nothing
    RT = rdata_type(F)
    if RT !== NoRData
        throw(
            ArgumentError(
                "Mooncake GPU sum/mapreduce rule does not support $F as the mapping " *
                "function because it has rdata type $RT, meaning it captures " *
                "differentiable state (e.g. a closed-over Float32 scalar). The GPU rule " *
                "threads NDuals only through the CuArray elements and cannot propagate " *
                "gradients back through captured variables. To fix this, implement a " *
                "custom rrule!! for the enclosing function (e.g. Statistics.varm) or " *
                "restructure the computation to avoid differentiable closures.",
            ),
        )
    end
    return nothing
end

function _gpu_sum_f_frule(f, x)
    _check_gpu_sum_f(f)
    flat_px = parent(primal(x))
    flat_dx = _fields(tangent(x)).parent
    flat_pargs = (flat_px,)
    flat_tangents = (flat_dx,)
    out = _gpu_broadcast_dual(f, flat_px)
    decoded = _gpu_decode_ndual_output(Val(:sum), out, flat_pargs)
    dy = if decoded.is_diff && !(flat_dx isa NoTangent)
        _gpu_accumulate_reduced_jvp(out, flat_pargs, flat_tangents, decoded.primal_out)
    else
        zero(decoded.primal_out)
    end
    return Dual(decoded.primal_out, dy)
end

function _gpu_sum_f_rrule(f, x)
    _check_gpu_sum_f(f)
    flat_px = parent(primal(x))
    flat_dx = _fields(tangent(x)).parent
    flat_pargs = (flat_px,)
    flat_fdatas = (flat_dx,)
    out = _gpu_broadcast_dual(f, flat_px)
    decoded = _gpu_decode_ndual_output(Val(:sum), out, flat_pargs; extract_partials=true)
    function sum_f_pb!!(dy)
        isnothing(decoded.partial_slots) || _gpu_accumulate_reduced_pullback!(
            flat_pargs, flat_fdatas, decoded.partial_slots, dy
        )
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(decoded.primal_out), sum_f_pb!!
end

@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,CuFloatArray})
@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,<:Adjoint{<:IEEEFloat,<:CuFloatArray}})
@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,<:Transpose{<:IEEEFloat,<:CuFloatArray}})

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
function frule!!(::Dual{typeof(sum)}, f::Dual, x::Dual{<:CuGpuSumFArray})
    return _gpu_sum_f_frule(primal(f), x)
end
function rrule!!(::CoDual{typeof(sum)}, f::CoDual, x::CoDual{<:CuGpuSumFArray})
    return _gpu_sum_f_rrule(primal(f), x)
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
    return _throw_gpu_argument_error(
        "Mooncake: mapreduce on CuArray only supports op=+ or op=Base.add_sum; " *
        "got op=$(primal(op)). " *
        _UNIMPL_MSG,
    )
end
function rrule!!(::CoDual{typeof(mapreduce)}, f::CoDual, op::CoDual, x::CoDual{<:CuArray})
    return _throw_gpu_argument_error(
        "Mooncake: mapreduce on CuArray only supports op=+ or op=Base.add_sum; " *
        "got op=$(primal(op)). " *
        _UNIMPL_MSG,
    )
end

@is_primitive(MinimalCtx, Tuple{typeof(reduce),Any,CuArray})
function frule!!(::Dual{typeof(reduce)}, op::Dual, x::Dual{<:CuArray})
    return _throw_gpu_argument_error(
        "Mooncake: reduce on CuArray only supports op=+ (sum) or op=* (prod); " *
        "got op=$(primal(op)). " *
        _UNIMPL_MSG,
    )
end
function rrule!!(::CoDual{typeof(reduce)}, op::CoDual, x::CoDual{<:CuArray})
    return _throw_gpu_argument_error(
        "Mooncake: reduce on CuArray only supports op=+ (sum) or op=* (prod); " *
        "got op=$(primal(op)). " *
        _UNIMPL_MSG,
    )
end

# vcat / hcat / cat on CuMaybeWrappedArray (see its definition above for what that
# includes and why).
# frule:    tangent of concatenation = concatenation of tangents (concat is linear).
# pullback: selectdim returns a view (no allocation per slice); running-offset loop
#           avoids pre-allocating an offsets array.
@inline function _cu_concat_pb!(fdatas, dy_out, dim::Integer)
    offset = 0
    for i in eachindex(fdatas)
        n = size(fdatas[i], dim)
        fdatas[i] .+= selectdim(dy_out, dim, (offset + 1):(offset + n))
        offset += n
    end
    return nothing
end

# `dims` a Tuple of K distinct dimensions: cat(A, B, ...; dims=(d1, d2, ...)) builds a
# block-diagonal result, growing every listed dimension simultaneously and zero-filling
# elsewhere. Each input's gradient is the sub-block of dy_out at its own running offset in
# each listed dimension, full range elsewhere (`cat` requires equal sizes there).
@inline function _cu_concat_pb!(fdatas, dy_out, dims::NTuple{K,Integer}) where {K}
    offsets = ntuple(_ -> 0, Val(K))
    nd = Val(ndims(dy_out))
    for i in eachindex(fdatas)
        fi = fdatas[i]
        # `let`, so the closure below captures a binding that is never reassigned:
        # capturing `offsets` itself boxes it and costs the loop its zero-allocation.
        offsets = let offs = offsets
            ranges = ntuple(nd) do d
                k = findfirst(==(d), dims)
                k === nothing ? Colon() : (offs[k] + 1):(offs[k] + size(fi, d))
            end
            fi .+= view(dy_out, ranges...)
            ntuple(k -> offs[k] + size(fi, dims[k]), Val(K))
        end
    end
    return nothing
end

_unwrap_cat_dim(d::Integer) = d
_unwrap_cat_dim(::Val{N}) where {N} = N
_unwrap_cat_dim(d::Tuple{Vararg{Integer}}) = d
# `Base.dims2cat` takes any iterable, so `cat(A, B; dims=1:2)` and `dims=[1, 2]` are valid
# calls the primal and the frule both accept; normalise to the Tuple the pullback needs.
_unwrap_cat_dim(d::Union{AbstractRange{<:Integer},AbstractVector{<:Integer}}) = Tuple(d)
function _unwrap_cat_dim(d)
    return throw(
        ArgumentError(
            "Mooncake: cat requires dims to be an Integer, Val{N}, or an iterable of " *
            "Integers; got dims=$(d).",
        ),
    )
end

@is_primitive(
    MinimalCtx, Tuple{typeof(vcat),CuMaybeWrappedArray,Vararg{CuMaybeWrappedArray}}
)
function frule!!(::Dual{typeof(vcat)}, args::Dual{<:CuMaybeWrappedArray}...)
    pairs = map(arrayify, args)
    return Dual(vcat(map(first, pairs)...), vcat(map(last, pairs)...))
end
function rrule!!(::CoDual{typeof(vcat)}, args::CoDual{<:CuMaybeWrappedArray}...)
    pairs = map(arrayify, args)
    primals = map(first, pairs)
    fdatas = map(last, pairs)
    out = vcat(primals...)
    dy_out = zero(out)
    pb!!(::NoRData) =
        (_cu_concat_pb!(fdatas, dy_out, 1); (NoRData(), map(_ -> NoRData(), args)...))
    return CoDual(out, dy_out), pb!!
end

@is_primitive(
    MinimalCtx, Tuple{typeof(hcat),CuMaybeWrappedArray,Vararg{CuMaybeWrappedArray}}
)
function frule!!(::Dual{typeof(hcat)}, args::Dual{<:CuMaybeWrappedArray}...)
    pairs = map(arrayify, args)
    return Dual(hcat(map(first, pairs)...), hcat(map(last, pairs)...))
end
function rrule!!(::CoDual{typeof(hcat)}, args::CoDual{<:CuMaybeWrappedArray}...)
    pairs = map(arrayify, args)
    primals = map(first, pairs)
    fdatas = map(last, pairs)
    out = hcat(primals...)
    dy_out = zero(out)
    pb!!(::NoRData) =
        (_cu_concat_pb!(fdatas, dy_out, 2); (NoRData(), map(_ -> NoRData(), args)...))
    return CoDual(out, dy_out), pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(Core.kwcall),
        NamedTuple,
        typeof(cat),
        CuMaybeWrappedArray,
        Vararg{CuMaybeWrappedArray},
    },
)
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    kw::Dual{<:NamedTuple},
    ::Dual{typeof(cat)},
    args::Dual{<:CuMaybeWrappedArray}...,
)
    pkw = primal(kw)
    pairs = map(arrayify, args)
    return Dual(cat(map(first, pairs)...; pkw...), cat(map(last, pairs)...; pkw...))
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(cat)},
    args::CoDual{<:CuMaybeWrappedArray}...,
)
    pkw = primal(kw)
    dim = _unwrap_cat_dim(pkw.dims)
    pairs = map(arrayify, args)
    primals = map(first, pairs)
    fdatas = map(last, pairs)
    out = cat(primals...; pkw...)
    dy_out = zero(out)
    pb!!(::NoRData) = (
        _cu_concat_pb!(fdatas, dy_out, dim);
        (NoRData(), NoRData(), NoRData(), map(_ -> NoRData(), args)...)
    )
    return CoDual(out, dy_out), pb!!
end

@noinline function _throw_mixed_cat_error(fn)
    _throw_gpu_argument_error(
        "Mooncake: cannot differentiate $fn with a mix of GPU (CuArray) and non-GPU " *
        "arguments. Without this check, this would previously crash with an opaque " *
        "CUDA kernel error instead. Likely causes: (a) some arguments are still on " *
        "the CPU, or are plain Numbers: move arrays to the GPU with `gpu(array)` " *
        "(CUDA.jl / MLDataDevices.jl); or (b) a GPU array is wrapped in a type not " *
        "yet recognised here (e.g. `Diagonal`), so it isn't detected as GPU.",
    )
end

# Mixed CPU/GPU device guard for vcat/hcat/cat, at any arity and any argument order, arrays
# or scalars: only genuinely mixed calls make `_is_primitive` return true, at which point
# frule!!/rrule!! below throw a clear error instead of the opaque `cufunction` crash.
# Pure-GPU calls match the strictly more specific GPU-only signature above (same
# function-argument type, so specificity decides); pure-CPU calls do reach here, get
# has_gpu=false, and fall through to the interpreter, which keeps CPU code such as NNlib's
# softmax working. `@is_primitive` can't express a decision that depends on the concrete
# argument types, hence the direct `_is_primitive` method.
_cu_isa_gpu_side(::Type{T}) where {T} = T <: CuMaybeWrappedArray

# `any_matches_primitive` feeds every call site's type into `_is_primitive`, including
# splatted calls (`Core.TypeofVararg`, not a `Type`) and imprecise-eltype calls
# (`UnionAll`). A `@generated` version crashes on both, which Mooncake's own ambiguity
# workaround then silently turns into an incorrect `true`. This plain method instead
# bails out to `false` (conservative) on either case.
function _cu_mixed_device_is_primitive(T::Type, from::Int)
    T isa DataType || return false
    has_gpu = false
    has_other = false
    for t in T.parameters[from:end]
        t isa Core.TypeofVararg && return false
        _cu_isa_gpu_side(t) ? (has_gpu = true) : (has_other = true)
    end
    return has_gpu && has_other
end

for _fn in (:vcat, :hcat)
    @eval function Mooncake._is_primitive(
        ::Type{MinimalCtx}, ::Type{<:Mooncake.Mode}, ::Type{T}
    ) where {T<:Tuple{typeof($_fn),Vararg{Union{AbstractArray,Number}}}}
        return _cu_mixed_device_is_primitive(T, 2)
    end
    @eval function frule!!(::Dual{typeof($_fn)}, ::Dual{<:Union{AbstractArray,Number}}...)
        return _throw_mixed_cat_error($_fn)
    end
    @eval function rrule!!(
        ::CoDual{typeof($_fn)}, ::CoDual{<:Union{AbstractArray,Number}}...
    )
        return _throw_mixed_cat_error($_fn)
    end
end

# cat(; dims=...) goes through Core.kwcall so cannot share the loop above.
function Mooncake._is_primitive(
    ::Type{MinimalCtx}, ::Type{<:Mooncake.Mode}, ::Type{T}
) where {
    T<:Tuple{typeof(Core.kwcall),NamedTuple,typeof(cat),Vararg{Union{AbstractArray,Number}}}
}
    return _cu_mixed_device_is_primitive(T, 4)
end
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    ::Dual{<:NamedTuple},
    ::Dual{typeof(cat)},
    ::Dual{<:Union{AbstractArray,Number}}...,
)
    return _throw_mixed_cat_error(cat)
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    ::CoDual{<:NamedTuple},
    ::CoDual{typeof(cat)},
    ::CoDual{<:Union{AbstractArray,Number}}...,
)
    return _throw_mixed_cat_error(cat)
end

# Rules are written at the `generic_matmatmul!` / `generic_matvecmul!` level rather
# than at the individual cuBLAS primitive level (gemm!, gemv!, gemmEx!, symm!, ...).
# This gives broad coverage of the LinearAlgebra.mul! dispatch chain with just two
# rules, and is correct for all practical ML workloads (dense real/complex arrays).
# The tradeoff: symmetric/Hermitian cases (tA='S'/'H', dispatching to symv!/hemv!
# in the primal) use gemm!/gemv! in the backward, which is mathematically correct
# only when the full matrix is populated. Direct cuBLAS calls that bypass
# LinearAlgebra.mul! are not covered; add lower-level rules if that becomes needed.

# Guard helpers shared by the generic_matmatmul! and generic_matvecmul! rules.

@inline function _check_complex_transpose_flag(T, tAv, tBv)
    T <: Complex &&
        (tAv == 'T' || tBv == 'T') &&
        throw(
            ArgumentError(
                "Mooncake: generic_matmatmul! with the 'T' (plain transpose) flag is not " *
                "supported for complex CuArrays — the backward requires element-wise " *
                "conjugation, which cannot be expressed as a single cuBLAS GEMM. " *
                "Use adjoint ('C') instead of transpose ('T').",
            ),
        )
    return nothing
end

@inline function _check_gemv_eltypes(T, T_B)
    T_B == T || throw(
        ArgumentError(
            "Mooncake: GPU gemv with mismatched element types " *
            "(A=$(T), B=$(T_B)) is not supported. " *
            "Cast all arrays to the same element type before multiplying. " *
            "(Note: cu() downcasts Float64/ComplexF64 to Float32/ComplexF32 by default; " *
            "use CuArray(x) to preserve the element type.)",
        ),
    )
    return nothing
end

@inline function _check_complex_matvecmul_transpose(T, tAv)
    T <: Complex &&
        tAv == 'T' &&
        throw(
            ArgumentError(
                "Mooncake: generic_matvecmul! with the 'T' (plain transpose) flag is not " *
                "supported for complex CuArrays. Use adjoint ('C') instead.",
            ),
        )
    return nothing
end

# Rule for `LinearAlgebra.generic_matmatmul!` on real and complex GPU arrays.
#
# `generic_matmatmul!(C, tA, tB, A, B)` computes C = op_A(A) * op_B(B) in-place,
# where tA, tB ∈ {'N','T','C'} are BLAS transpose flags. It is the generic fallback
# that LinearAlgebra dispatches to when cuBLAS has no specific method — for example,
# `adjoint(CuVector) * CuMatrix` falls through here because cuBLAS.gemm! only accepts
# CuMatrix inputs.
#
# Strategy: reshape any CuVector to (n,1) CuMatrix via `matrixify` (zero-copy), then
# delegate to cuBLAS.gemm! which is differentiable and avoids scalar GPU indexing.
#
# Backward formulas for C = op_A(A) * op_B(B) (real and complex; uses '^H' = Hermitian
# conjugate, which cuBLAS flag 'C' handles; for real 'C' == 'T'):
#   tA='N': dA += dC * op_B(B)^H    (flags: 'N', tB=='N' ? 'C' : 'N')
#   tA≠'N': dA += op_B(B) * dC^H   (flags: tB, 'C')
#   tB='N': dB += op_A(A)^H * dC   (flags: tA=='N' ? 'C' : 'N', 'N')
#   tB≠'N': dB += dC^H * op_A(A)   (flags: 'C', tA)
#
# Limitation: the 'T' (plain transpose) flag is only correct for real arrays.
# For complex arrays, 'T' would require element-wise conjugation (conj(B)) in the
# backward, which cannot be expressed as a single cuBLAS GEMM call. A runtime guard
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
    _check_complex_transpose_flag(T, tAv, tBv)
    _1 = one(T)
    _0 = zero(T)
    # primal: C = op_A(A) * op_B(B)
    cuBLAS.gemm!(tAv, tBv, _1, pA, pB, _0, pC)
    # tangent (product rule): dC = op_A(dA)*op_B(pB) + op_A(pA)*op_B(dB)
    cuBLAS.gemm!(tAv, tBv, _1, dA, pB, _0, dC)
    cuBLAS.gemm!(tAv, tBv, _1, pA, dB, _1, dC)
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
    _check_complex_transpose_flag(T, tAv, tBv)
    _1 = one(T)
    _0 = zero(T)
    pC_copy = copy(pC)
    cuBLAS.gemm!(tAv, tBv, _1, pA, pB, _0, pC)
    function generic_matmatmul!_pb!!(::NoRData)
        if tAv == 'N'
            cuBLAS.gemm!('N', tBv == 'N' ? 'C' : 'N', _1, dC, pB, _1, dA) # dA += dC * op_B(B)^H
        else
            cuBLAS.gemm!(tBv, 'C', _1, pB, dC, _1, dA)                     # dA += op_B(B) * dC^H
        end
        if tBv == 'N'
            cuBLAS.gemm!(tAv == 'N' ? 'C' : 'N', 'N', _1, pA, dC, _1, dB) # dB += op_A(A)^H * dC
        else
            cuBLAS.gemm!('C', tAv, _1, dC, pA, _1, dB)                     # dB += dC^H * op_A(A)
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
    _check_complex_transpose_flag(T, tAv, tBv)
    _α = T(primal(alpha))
    _β = T(primal(beta))
    _1 = one(T)
    # primal: C := α*op_A(A)*op_B(B) + β*C
    cuBLAS.gemm!(tAv, tBv, _α, pA, pB, _β, pC)
    # tangent: dC := α*(op_A(dA)*op_B(pB) + op_A(pA)*op_B(dB)) + β*dC
    cuBLAS.gemm!(tAv, tBv, _α, dA, pB, _β, dC)
    cuBLAS.gemm!(tAv, tBv, _α, pA, dB, _1, dC)
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
    _check_complex_transpose_flag(T, tAv, tBv)
    _α = T(primal(alpha))
    _β = T(primal(beta))
    _1 = one(T)
    pC_copy = copy(pC)
    cuBLAS.gemm!(tAv, tBv, _α, pA, pB, _β, pC)
    function generic_matmatmul!_7arg_pb!!(::NoRData)
        # Adjoint of C = α*op_A(A)*op_B(B) + β*C_old requires conj(α) and conj(β).
        # For real scalars conj is identity, so this is backward-compatible.
        _cα = conj(_α)
        _cβ = conj(_β)
        if tAv == 'N'
            cuBLAS.gemm!('N', tBv == 'N' ? 'C' : 'N', _cα, dC, pB, _1, dA) # dA += conj(α)*dC*op_B(B)^H
        else
            cuBLAS.gemm!(tBv, 'C', _cα, pB, dC, _1, dA)                     # dA += conj(α)*op_B(B)*dC^H
        end
        if tBv == 'N'
            cuBLAS.gemm!(tAv == 'N' ? 'C' : 'N', 'N', _cα, pA, dC, _1, dB) # dB += conj(α)*op_A(A)^H*dC
        else
            cuBLAS.gemm!('C', tAv, _cα, dC, pA, _1, dB)                     # dB += conj(α)*dC^H*op_A(A)
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
# CUDA.jl overrides this to call cuBLAS.gemv! directly (cublas/linalg.jl), bypassing
# `mul!`. Without this rule, Mooncake's forward-mode interpreter traces into CUDA's
# task-local-storage machinery (cuBLAS.handle → task_local_state!) which contains
# `Unreachable` code paths when called with dual types → SIGILL.
#
# Strategy: for the primal and tangent pass use cuBLAS.gemv!; for the dA update
# (an outer product) reshape both vectors to (n,1) matrices and use cuBLAS.gemm!.
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
    _check_gemv_eltypes(T, eltype(pB))
    _check_complex_matvecmul_transpose(T, tAv)
    _1 = one(T)
    # tangent (product rule): dY = av*op(dA)*pB + av*op(pA)*dB + bv*dY
    cuBLAS.gemv!(tAv, av, dA, pB, bv, dY) # dY  = av*op(dA)*pB + bv*dY
    cuBLAS.gemv!(tAv, av, pA, dB, _1, dY) # dY += av*op(pA)*dB
    # primal: pY = av*op(pA)*pB + bv*pY
    cuBLAS.gemv!(tAv, av, pA, pB, bv, pY)
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
    _check_gemv_eltypes(T, eltype(pB))
    _check_complex_matvecmul_transpose(T, tAv)
    _1 = one(T)
    pY_copy = copy(pY)
    cuBLAS.gemv!(tAv, av, pA, pB, bv, pY)
    function generic_matvecmul!_pb!!(::NoRData)
        # dA update: outer product — reshape vectors to (n,1) matrices for gemm!
        dY_mat = reshape(dY, :, 1)
        pB_mat = reshape(pB, :, 1)
        if tAv == 'N'
            cuBLAS.gemm!('N', 'C', av, dY_mat, pB_mat, _1, dA) # dA += av * ȳ * B^H
        else
            cuBLAS.gemm!('N', 'C', av, pB_mat, dY_mat, _1, dA) # dA += av * B * ȳ^H
        end
        # dB update: gemv with Hermitian conjugate of op(A)
        if tAv == 'N'
            cuBLAS.gemv!('C', av, pA, dY, _1, dB) # dB += av * A^H * ȳ
        else
            cuBLAS.gemv!('N', av, pA, dY, _1, dB) # dB += av * A   * ȳ  (op(A)^H = A)
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
# How it works
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
# To support a new scalar type T: extend Nfwd's internal leaf-DOF helpers so it contributes
# the correct slot count, then handle it in _leaf_effective_tangent / materialize_pb!! /
# _gpu_fill_args_rdata.

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
    return NDual{T,N}(x, ntuple(j -> T(j == slot), Val(N)))
end
@inline _gpu_bcast_dual(x, ::Int, ::Any) = x  # non-differentiable: pass through

@inline function _gpu_bcast_dual(
    x::Complex{ET}, slot_re::Int, slot_im::Int, ::Val{N}
) where {ET<:IEEEFloat,N}
    return Complex(
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
        N += Nfwd._nfwd_type_dof(ET)
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

# One fused GPU kernel: evaluates f and all partial derivatives simultaneously.
# Real args use 1 Dual slot each; complex args use 2 (one per real DOF).
function _gpu_broadcast_dual(f::F, args...) where {F}
    return ((args...) -> _gpu_apply_with_duals(f, args...)).(args...)
end

# Single second-order chokepoint: the NDual-based GPU rules (`sum(f, x)`, `materialize`,
# `materialize!`) all launch their kernel through `_gpu_broadcast_dual`, traced only when
# nested AD differentiates a rule body. Tracing it nests `NDual` over `NDual`, which errors
# (perturbation confusion). Both modes get a throwing rule so the unsupported
# reverse-over-* direction fails with the same clear message rather than a MethodError.
const _GPU_SECOND_ORDER_MSG =
    "Mooncake: HVP / Hessian over a custom CUDA kernel (broadcasting, or " *
    "`sum(f, x)`-style mapped reductions) is not yet supported — the kernel is a " *
    "foreign call with no Julia IR to differentiate through at second order. " *
    "Gradient and JVP are unaffected; for HVP/Hessian, use array-level ops " *
    "(`sum(x)`, `dot`, matmul)."
@is_primitive MinimalCtx Tuple{typeof(_gpu_broadcast_dual),Vararg}
function frule!!(::Dual{typeof(_gpu_broadcast_dual)}, ::Vararg{Dual})
    return _throw_gpu_argument_error(_GPU_SECOND_ORDER_MSG)
end
function rrule!!(::CoDual{typeof(_gpu_broadcast_dual)}, ::Vararg{CoDual})
    return _throw_gpu_argument_error(_GPU_SECOND_ORDER_MSG)
end

# Map each broadcast leaf arg to a representative scalar element so that
# _nfwd_input_dof counts per-broadcast-element DOFs.
@inline _gpu_rep_element(x::CuFloatOrComplex) = x
@inline _gpu_rep_element(x::AbstractArray{T}) where {T<:IEEEFloat} = zero(T)
@inline _gpu_rep_element(x::AbstractArray{Complex{T}}) where {T<:IEEEFloat} = zero(
    Complex{T}
)
@inline _gpu_rep_element(::Any) = ()

@inline _gpu_total_slots(flat_pargs) = Nfwd._nfwd_input_dof(
    map(_gpu_rep_element, flat_pargs)
)

@inline function _gpu_leaf_slot_meta(pa, offset)
    dof = Nfwd._nfwd_input_dof(_gpu_rep_element(pa))
    return (; dof, slot1=offset + 1, slot2=offset + 2, is_scalar=pa isa CuFloatOrComplex)
end

@inline function _gpu_extract_partial_slots(out, n_slots::Int)
    return [broadcast(o -> Nfwd._nfwd_dual_partial(o, k), out) for k in 1:n_slots]
end

@inline function _gpu_decode_ndual_meta(out, flat_pargs; extract_partials::Bool=false)
    is_diff = Nfwd._nfwd_dual_has_partials(eltype(out))
    n_slots = is_diff ? _gpu_total_slots(flat_pargs) : 0
    partial_slots =
        extract_partials && is_diff ? _gpu_extract_partial_slots(out, n_slots) : nothing
    return (; is_diff, n_slots, partial_slots)
end

@inline function _gpu_decode_ndual_output(
    ::Val{:broadcast},
    out,
    flat_pargs;
    extract_partials::Bool=false,
    extract_primal::Bool=true,
)
    decoded = _gpu_decode_ndual_meta(out, flat_pargs; extract_partials)
    primal_out = if extract_primal
        decoded.is_diff ? broadcast(Nfwd._nfwd_dual_value, out) : out
    else
        nothing
    end
    return (; decoded..., primal_out)
end

@inline function _gpu_write_broadcast_primal!(dest, out, is_diff::Bool)
    if is_diff
        broadcast!(Nfwd._nfwd_dual_value, dest, out)
    else
        copyto!(dest, out)
    end
    return dest
end

@inline function _gpu_decode_ndual_output(
    ::Val{:sum}, out, flat_pargs; extract_partials::Bool=false
)
    decoded = _gpu_decode_ndual_meta(out, flat_pargs; extract_partials)
    primal_out = sum(
        Nfwd._nfwd_dual_value, out; init=zero(Nfwd._nfwd_dual_primal_type(eltype(out)))
    )
    return (; decoded..., primal_out)
end

# Replace any nested Broadcasted sub-expression whose tangent/fdata tree is
# `NoTangent`/`NoFData`, or whose broadcast tree has zero effective differentiable
# degrees of freedom and flattens to a non-isbits function, with its primal materialized
# value. This catches zero-DOF subtrees such as `Float64.(b .> 0)`, where flattening the
# nested broadcast embeds `Type{Float64}` in the composed function object and makes the
# GPU kernel argument non-isbits.
#
# Note: the resulting plain CuArray leaf may still have a differentiable eltype, so the
# GPU dual kernel may reserve a slot for it. `_leaf_effective_tangent` returns `nothing`
# for the paired `NoTangent`, so the slot contribution is discarded. That is slightly
# wasteful but keeps the kernel function isbits and GPU-compilable.
@inline _gpu_bcast_has_nondiff_result(::typeof(>)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(<)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(>=)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(<=)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(==)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(!=)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(iszero)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(signbit)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(isfinite)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(isinf)) = true
@inline _gpu_bcast_has_nondiff_result(::typeof(isnan)) = true
@inline _gpu_bcast_has_nondiff_result(::Any) = false
@inline _gpu_is_simple_cast_broadcast(::Any) = false
@inline function _gpu_is_simple_cast_broadcast(bc::Broadcasted)
    return bc.f isa Type{<:CuFloatOrComplex} &&
           length(bc.args) == 1 &&
           !(first(bc.args) isa Broadcasted)
end

@inline _gpu_bcast_arg_dof(x::IEEEFloat) = 1
@inline _gpu_bcast_arg_dof(x::Complex{<:IEEEFloat}) = 2
@inline _gpu_bcast_arg_dof(x::AbstractArray{<:IEEEFloat}) = 1
@inline _gpu_bcast_arg_dof(x::AbstractArray{<:Complex{<:IEEEFloat}}) = 1
@inline _gpu_bcast_arg_dof(x::Base.Broadcast.Extruded) = _gpu_bcast_arg_dof(x.x)
@inline _gpu_bcast_arg_dof(x::Adjoint{<:CuFloatOrComplex,<:AbstractArray}) = 1
@inline _gpu_bcast_arg_dof(x::Transpose{<:CuFloatOrComplex,<:AbstractArray}) = 1
@inline _gpu_bcast_arg_dof(x::Broadcasted) = _gpu_bcast_effective_dof(x)
@inline _gpu_bcast_arg_dof(::Any) = 0

function _gpu_bcast_effective_dof(bc::Broadcasted)
    _gpu_bcast_has_nondiff_result(bc.f) && return 0
    return any(!iszero, map(_gpu_bcast_arg_dof, bc.args)) ? 1 : 0
end

@inline _gpu_bcast_needs_premat(bc::Broadcasted) =
    (_gpu_bcast_effective_dof(bc) == 0 || _gpu_is_simple_cast_broadcast(bc)) &&
    !isbitstype(typeof(Base.Broadcast.flatten(bc).f))

_premat_nondiff_args(bc::Broadcasted) = _premat_nondiff_args(bc, NoTangent())

function _premat_nondiff_args(bc::Broadcasted, td)
    targs = if td isa Union{NoTangent,NoFData}
        ntuple(_ -> NoTangent(), length(bc.args))
    else
        _fields(td).args
    end
    new_args = ntuple(length(bc.args)) do i
        a = bc.args[i]
        ta = targs[i]
        if a isa Broadcasted
            # Scalars always report NoFData (their grad flows via RData), even when
            # differentiable — so also require zero DOF before collapsing, or a
            # differentiable scalar leaf here silently drops out of flat_pargs.
            if ta isa Union{NoTangent,NoFData} && _gpu_bcast_effective_dof(a) == 0
                Base.Broadcast.materialize(a)
            else
                a_prepared = _premat_nondiff_args(a, ta)
                if _gpu_bcast_needs_premat(a_prepared)
                    Base.Broadcast.materialize(a_prepared)
                else
                    a_prepared
                end
            end
        else
            a
        end
    end
    return Broadcasted(bc.f, new_args, bc.axes)
end

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

struct _GpuBroadcastCastDiff{T,PA,D}
    primal_arg::PA
    diff_arg::D
end
@inline function _gpu_broadcast_cast_diff(::Type{T}, primal_arg, diff_arg) where {T}
    return _GpuBroadcastCastDiff{T,typeof(primal_arg),typeof(diff_arg)}(
        primal_arg, diff_arg
    )
end

@inline _gpu_cast_like(::Type{T}, x::AbstractArray) where {T} = T.(x)
@inline _gpu_cast_like(::Type{T}, x::CuFloatOrComplex) where {T} = convert(T, x)
@inline _gpu_cast_back_like(pa::AbstractArray, contrib) = eltype(pa).(contrib)
@inline _gpu_cast_back_like(pa::CuFloatOrComplex, contrib) = convert(typeof(pa), contrib)

@inline function _leaf_effective_tangent(_, diff::_GpuBroadcastCastDiff{T}) where {T}
    t_eff = _leaf_effective_tangent(diff.primal_arg, diff.diff_arg)
    return t_eff === nothing ? nothing : _gpu_cast_like(T, t_eff)
end
@inline _leaf_effective_tangent(_, _) = nothing  # non-differentiable

# Reduce `dx` (broadcast-output shape) back to `sz` by summing over any dimensions that
# were singleton-expanded or added during broadcasting.  Mirrors ChainRules' `unbroadcast`.
#
# Julia broadcasting is left-aligned: a 1D array (n,) broadcast against (n,p) is treated
# as (n,1) — extra trailing dimensions, not extra leading ones.  So "extra" dims are those
# at positions d > length(sz), not d <= n_extra.
function _unbroadcast(dx::CuArray, sz::Tuple)
    size(dx) == sz && return dx
    # Collect reduction dims as a Tuple (stack-allocated) to avoid filter's heap Vector.
    reduce_dims = ntuple(ndims(dx)) do d
        return d > length(sz) || sz[d] == 1 ? d : 0
    end
    reduce_dims = filter(!iszero, reduce_dims)  # Tuple filter — no heap alloc
    return isempty(reduce_dims) ? reshape(dx, sz) : reshape(sum(dx; dims=reduce_dims), sz)
end

# Reverse mode: accumulate `contrib` (same shape as broadcast output) into leaf fdata.
# Unbroadcast before accumulating so that broadcast-expanded inputs get the correct shape.
@inline function _leaf_accum_fdata!(pa::CuMaybeComplexArray, fd::CuArray, contrib)
    return fd .+= _unbroadcast(contrib, size(pa))
end
@inline function _leaf_accum_fdata!(
    pa::Adjoint{<:CuFloatOrComplex,<:CuMaybeComplexArray}, fd, contrib
)
    return _fields(fd).parent .+= adjoint(_unbroadcast(contrib, size(pa)))
end
@inline function _leaf_accum_fdata!(
    pa::Transpose{<:CuFloatOrComplex,<:CuMaybeComplexArray}, fd, contrib
)
    return _fields(fd).parent .+= transpose(_unbroadcast(contrib, size(pa)))
end
@inline function _leaf_accum_fdata!(_, diff::_GpuBroadcastCastDiff, contrib)
    return _leaf_accum_fdata!(
        diff.primal_arg, diff.diff_arg, _gpu_cast_back_like(diff.primal_arg, contrib)
    )
end
@inline _leaf_accum_fdata!(_, _, _) = nothing  # non-differentiable

# Recursively extract leaf (non-Broadcasted) arg primals and their tangent data from a
# possibly-nested Broadcasted / tangent pair.  Works for both reverse mode (FData, uses
# _fields(td).args) and forward mode (Tangent, uses _fields(td).args) because _fields
# abstracts over both.
#
# When td is NoTangent or NoFData the whole sub-expression has no differentiable content.
# We still extract the primal leaves (the GPU kernel needs them) paired with NoTangent;
# contributions from non-differentiable leaves are filtered out downstream via the
# _leaf_effective_tangent / _leaf_accum_fdata! catch-all methods (which return nothing
# when the tangent is not a CuArray or IEEEFloat scalar).
@inline function _gpu_bcast_leaves(bc_prepared, bc_primal, td)
    return _gpu_bcast_leaves_args(bc_prepared.args, bc_primal.args, _fields(td).args)
end
@inline _gpu_bcast_leaves(bc_prepared, _, ::Union{NoTangent,NoFData}) = _gpu_bcast_leaves_nots(
    bc_prepared.args
)
@inline _gpu_bcast_leaves_nots(::Tuple{}) = ((), ())
@inline function _gpu_bcast_leaves_nots(args::Tuple)
    a1 = first(args)
    rest_ps, rest_ts = _gpu_bcast_leaves_nots(Base.tail(args))
    if a1 isa Broadcasted
        inner_ps, inner_ts = _gpu_bcast_leaves(a1, a1, NoTangent())
        return (inner_ps..., rest_ps...), (inner_ts..., rest_ts...)
    else
        return (a1, rest_ps...), (NoTangent(), rest_ts...)
    end
end
@inline _gpu_bcast_leaves_args(::Tuple{}, ::Tuple{}, ::Tuple{}) = ((), ())
@inline _gpu_bcast_leaves_args(args_prepared::Tuple, ::Tuple, ::Tuple{}) = _gpu_bcast_leaves_nots(
    args_prepared
)
@inline function _gpu_cast_diff_arg(bc::Broadcasted, td)
    return _gpu_broadcast_cast_diff(bc.f, first(bc.args), first(_fields(td).args))
end
@inline function _gpu_bcast_leaves_args(
    args_prepared::Tuple, args_primal::Tuple, tds::Tuple
)
    a1_prepared = first(args_prepared)
    a1_primal = first(args_primal)
    td1 = first(tds)
    rest_ps, rest_ts = _gpu_bcast_leaves_args(
        Base.tail(args_prepared), Base.tail(args_primal), Base.tail(tds)
    )
    if a1_prepared isa Broadcasted
        inner_ps, inner_ts = _gpu_bcast_leaves(a1_prepared, a1_primal, td1)
        return (inner_ps..., rest_ps...), (inner_ts..., rest_ts...)
    elseif a1_primal isa Broadcasted
        # `_premat_nondiff_args` collapsed a zero-DOF nested Broadcasted subtree to a plain
        # leaf. For zero-DOF subtrees the prepared leaf is constant; for simple numeric
        # casts like `Float64.(x32)` we keep the underlying leaf tangent/fdata and apply
        # the cast explicitly in the JVP/pullback.
        diff = if td1 isa Union{NoTangent,NoFData}
            NoTangent()
        elseif _gpu_is_simple_cast_broadcast(a1_primal)
            _gpu_cast_diff_arg(a1_primal, td1)
        else
            NoTangent()
        end
        return (a1_prepared, rest_ps...), (diff, rest_ts...)
    else
        return (a1_prepared, rest_ps...), (td1, rest_ts...)
    end
end

function _prepare_gpu_broadcast(bc_primal, tangent_or_fdata)
    bc_prepared = _premat_nondiff_args(bc_primal, tangent_or_fdata)
    flat_bc = Base.Broadcast.flatten(bc_prepared)
    flat_pargs, flat_tangent_or_fdata = _gpu_bcast_leaves(
        bc_prepared, bc_primal, tangent_or_fdata
    )
    _check_mixed_gpu_eltype(flat_pargs)
    return bc_prepared, flat_bc, flat_pargs, flat_tangent_or_fdata
end

function _gpu_collect_scalar_map(bc::Broadcasted)
    scalar_index = Ref(1)
    scalar_map = _gpu_collect_scalar_map_args(bc.args, scalar_index)
    return scalar_map, scalar_index[] - 1
end

function _gpu_collect_scalar_map(bc::Broadcasted, scalar_index::Ref{Int})
    return _gpu_collect_scalar_map_args(bc.args, scalar_index)
end

function _gpu_collect_scalar_map_args(args::Tuple, scalar_index::Ref{Int})
    return ntuple(length(args)) do i
        a = args[i]
        if a isa Broadcasted
            _gpu_collect_scalar_map(a, scalar_index)
        elseif a isa CuFloatOrComplex
            idx = scalar_index[]
            scalar_index[] += 1
            idx
        else
            nothing
        end
    end
end

@is_primitive(
    MinimalCtx, Tuple{typeof(Base.Broadcast.materialize),<:Broadcasted{<:CuArrayStyle}},
)

# Build rdata for bc_primal from explicit scalar metadata collected at rule-construction
# time. This avoids rebuilding a path lookup table on every pullback.
function _gpu_fill_scalar_rdata(
    bc::Broadcasted, scalar_map::Tuple, scalar_grads::AbstractVector
)
    zbc = zero_rdata(bc)
    zbc isa NoRData && return zbc
    return _gpu_fill_scalar_rdata(zbc, bc, scalar_map, scalar_grads)
end

function _gpu_fill_scalar_rdata(
    zbc::RData, bc::Broadcasted, scalar_map::Tuple, scalar_grads::AbstractVector
)
    r_args = _gpu_fill_args_rdata(zbc.data.args, bc.args, scalar_map, scalar_grads)
    return RData((; style=zbc.data.style, f=zbc.data.f, args=r_args, axes=zbc.data.axes))
end

function _gpu_fill_args_rdata(
    zargs::Tuple, args::Tuple, scalar_map::Tuple, scalar_grads::AbstractVector
)
    return ntuple(length(args)) do i
        za = zargs[i]
        a = args[i]
        scalar_meta = scalar_map[i]
        if a isa Broadcasted
            za isa NoRData ? za : _gpu_fill_scalar_rdata(za, a, scalar_meta, scalar_grads)
        elseif scalar_meta isa Int
            scalar_grads[scalar_meta]
        else
            za
        end
    end
end

_gpu_fill_args_rdata(::NoRData, ::Tuple, ::Tuple, ::AbstractVector) = NoRData()

function _gpu_foreach_jvp_leaf(flat_pargs, flat_tangents, visit!)
    offset = 0
    for (pa, t) in zip(flat_pargs, flat_tangents)
        meta = _gpu_leaf_slot_meta(pa, offset)
        t_eff = _leaf_effective_tangent(pa, t)
        t_eff === nothing || visit!(meta, t_eff)
        offset += meta.dof
    end
    return nothing
end

function _gpu_accumulate_jvp!(dy, flat_pargs, flat_tangents, dual_out)
    _gpu_foreach_jvp_leaf(
        flat_pargs,
        flat_tangents,
        (meta, t_eff) -> begin
            if meta.dof == 1
                dy .+=
                    broadcast(o -> Nfwd._nfwd_dual_partial(o, meta.slot1), dual_out) .*
                    t_eff
            elseif meta.dof == 2
                dy .+=
                    broadcast(o -> Nfwd._nfwd_dual_partial(o, meta.slot1), dual_out) .*
                    real.(t_eff)
                dy .+=
                    broadcast(o -> Nfwd._nfwd_dual_partial(o, meta.slot2), dual_out) .*
                    imag.(t_eff)
            end
        end,
    )
    return dy
end

function _gpu_accumulate_reduced_jvp(out, flat_pargs, flat_tangents, y)
    dy = zero(y)
    _gpu_foreach_jvp_leaf(
        flat_pargs,
        flat_tangents,
        (meta, t_eff) -> begin
            if meta.dof == 1
                dy += sum(
                    broadcast(
                        (o, tt) -> Nfwd._nfwd_dual_partial(o, meta.slot1) * tt,
                        out,
                        t_eff,
                    ),
                )
            elseif meta.dof == 2
                dy += sum(
                    broadcast(
                        (o, tt) ->
                            Nfwd._nfwd_dual_partial(o, meta.slot1) * real(tt) +
                            Nfwd._nfwd_dual_partial(o, meta.slot2) * imag(tt),
                        out,
                        t_eff,
                    ),
                )
            end
        end,
    )
    return dy
end

# Detect mixed-eltype GPU broadcasts: when CuArray leaves have different element types
# (e.g. Float32 and Float64 in the same broadcast), the Dual wrapping would produce
# incompatible Dual widths and cause a cryptic GPU compiler error.  Raise a clear error.
# Note: scalar args (IEEEFloat/Complex) are not checked here; a Float64 scalar mixed
# with a Float32 CuArray silently promotes the broadcast to Float64, which may be slow
# or unsupported on some GPUs.  Cast the scalar explicitly if needed.

# Shared pullback accumulation for materialize and materialize! rrules.
#
# Walks flat_pargs in order, computing the contribution from each arg's partial
# slot(s) and gradient dy, then accumulating into the arg's fdata via
# _leaf_accum_fdata!.  Scalar IEEEFloat/Complex args have no fdata slot; their
# gradients are returned in a Vector that the caller uses to build r_bc via
# _gpu_fill_scalar_rdata.
#
# Keep the contraction inline here rather than reusing `_nfwd_real_dot`: these
# GPU pullbacks need mixed-precision support (e.g. Float64 cotangent against
# Float32 partials) and CUDA-friendly codegen for complex broadcasts.
#
# Returns r_bc (the Broadcasted rdata), or zero_rdata(bc_primal) if no scalars.
function _gpu_accum_pullback!(
    flat_pargs,
    flat_fdatas,
    partial_slots::AbstractVector,
    dy_out,
    bc_primal,
    scalar_map,
    scalar_count,
)
    scalar_grads = isnothing(scalar_map) ? nothing : Vector{Any}(undef, scalar_count)
    scalar_index = 1
    offset = 0
    for (pa, fd) in zip(flat_pargs, flat_fdatas)
        meta = _gpu_leaf_slot_meta(pa, offset)
        if meta.dof == 1
            contrib = broadcast(
                (p, d) -> real(conj(d) * p), partial_slots[meta.slot1], dy_out
            )
            if meta.is_scalar
                (scalar_grads::Vector{Any})[scalar_index] = sum(contrib)
                scalar_index += 1
            else
                _leaf_accum_fdata!(pa, fd, contrib)
            end
        elseif meta.dof == 2
            contrib = broadcast(
                (p1, p2, d) -> complex(real(conj(d) * p1), real(conj(d) * p2)),
                partial_slots[meta.slot1],
                partial_slots[meta.slot2],
                dy_out,
            )
            if meta.is_scalar
                (scalar_grads::Vector{Any})[scalar_index] = sum(contrib)
                scalar_index += 1
            else
                _leaf_accum_fdata!(pa, fd, contrib)
            end
        end
        offset += meta.dof
    end
    return if isnothing(scalar_grads)
        zero_rdata(bc_primal)
    else
        _gpu_fill_scalar_rdata(bc_primal, scalar_map, scalar_grads)
    end
end

function _gpu_accumulate_reduced_pullback!(flat_pargs, flat_fdatas, partial_slots, dy)
    offset = 0
    for (pa, fd) in zip(flat_pargs, flat_fdatas)
        meta = _gpu_leaf_slot_meta(pa, offset)
        if meta.dof == 1
            contrib = broadcast(p -> real(conj(dy) * p), partial_slots[meta.slot1])
            _leaf_accum_fdata!(pa, fd, contrib)
        elseif meta.dof == 2
            cdy = conj(dy)
            contrib = broadcast(
                (p1, p2) -> complex(real(cdy * p1), real(cdy * p2)),
                partial_slots[meta.slot1],
                partial_slots[meta.slot2],
            )
            _leaf_accum_fdata!(pa, fd, contrib)
        end
        offset += meta.dof
    end
    return nothing
end

function _check_mixed_gpu_eltype(flat_pargs)
    # Walk flat_pargs with an early-exit loop rather than building a temporary array.
    # In the common case (all same element type), this allocates nothing.
    first_et = nothing
    for pa in flat_pargs
        (
            pa isa CuMaybeComplexArray ||
            pa isa Adjoint{<:CuFloatOrComplex,<:CuMaybeComplexArray} ||
            pa isa Transpose{<:CuFloatOrComplex,<:CuMaybeComplexArray}
        ) || continue
        et = eltype(pa)
        if first_et === nothing
            first_et = et
        elseif et !== first_et
            throw(
                ArgumentError(
                    "Mooncake: GPU broadcast over arrays with mixed element types " *
                    "($first_et and $et) is not supported. " *
                    "Cast all inputs to the same type before broadcasting.",
                ),
            )
        end
    end
    return nothing
end

function frule!!(
    ::Dual{typeof(Base.Broadcast.materialize)}, bc::Dual{<:Broadcasted{<:CuArrayStyle}}
)
    bc_primal = primal(bc)
    _, flat_bc, flat_pargs, flat_ts = _prepare_gpu_broadcast(bc_primal, tangent(bc))

    # One GPU kernel: compute primal AND all partial derivatives simultaneously.
    # Real args use 1 Dual slot each; complex args use 2 (one per real DOF).
    out = _gpu_broadcast_dual(flat_bc.f, flat_pargs...)
    decoded = _gpu_decode_ndual_output(Val(:broadcast), out, flat_pargs)

    # Non-differentiable output (e.g. Bool from comparisons): zero tangent.
    if !decoded.is_diff
        return Dual(out, NoTangent())
    end

    dy = _gpu_accumulate_jvp!(zero(decoded.primal_out), flat_pargs, flat_ts, out)
    return Dual(decoded.primal_out, dy)
end

function rrule!!(
    mat_fn::CoDual{typeof(Base.Broadcast.materialize)},
    bc::CoDual{<:Broadcasted{<:CuArrayStyle}},
)
    bc_primal = primal(bc)
    bc_fdata = tangent(bc)
    bc_prepared, flat_bc, flat_pargs, flat_fdatas = _prepare_gpu_broadcast(
        bc_primal, bc_fdata
    )
    scalar_map, scalar_count = _gpu_collect_scalar_map(bc_primal)
    scalar_map = iszero(scalar_count) ? nothing : scalar_map

    # One GPU kernel: compute primal AND all N partial derivatives simultaneously.
    out = _gpu_broadcast_dual(flat_bc.f, flat_pargs...)
    decoded = _gpu_decode_ndual_output(
        Val(:broadcast), out, flat_pargs; extract_partials=true
    )

    # Non-differentiable output (e.g. Bool from x .!= 0): return zero gradients.
    if !decoded.is_diff
        return CoDual(out, NoFData()), NoPullback(mat_fn, bc)
    end

    dy_out = zero(decoded.primal_out)  # accumulated into by the downstream reverse pass

    function materialize_pb!!(::NoRData)
        r_bc = _gpu_accum_pullback!(
            flat_pargs,
            flat_fdatas,
            decoded.partial_slots,
            dy_out,
            bc_primal,
            scalar_map,
            scalar_count,
        )
        return NoRData(), r_bc
    end

    return CoDual(decoded.primal_out, dy_out), materialize_pb!!
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
#
# A broadcast into a wrapped `CuArray` still forms a `Broadcasted{CuArrayStyle}`, so a bare
# `CuArray` bound left it to the interpreter, which died tracing the kernel launch.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(Base.Broadcast.materialize!),P,<:Broadcasted{<:CuArrayStyle}
    } where {P<:CuMaybeWrappedArray},
)
function frule!!(
    ::Dual{typeof(Base.Broadcast.materialize!)},
    dest::Dual{P},
    bc::Dual{<:Broadcasted{<:CuArrayStyle}},
) where {P<:CuMaybeWrappedArray}
    bc_primal = primal(bc)
    _, flat_bc, flat_pargs, flat_ts = _prepare_gpu_broadcast(bc_primal, tangent(bc))

    dual_out = _gpu_broadcast_dual(flat_bc.f, flat_pargs...)
    pout, dout = arrayify(dest)
    decoded = _gpu_decode_ndual_output(
        Val(:broadcast), dual_out, flat_pargs; extract_primal=false
    )

    # Write primal result in-place into dest.
    _gpu_write_broadcast_primal!(pout, dual_out, decoded.is_diff)

    # Non-differentiable output (e.g. Bool arrays): zero the tangent and return.
    if !decoded.is_diff
        fill!(dout, zero(eltype(dout)))
        return dest
    end

    # JVP: accumulate into a temporary to handle aliasing (dest may appear in
    # bc.args, so flat_ts may contain a reference to dout; we must not overwrite
    # dout until all contributions have been read from the old tangent values).
    dy = _gpu_accumulate_jvp!(zero(pout), flat_pargs, flat_ts, dual_out)
    copyto!(dout, dy)
    return dest
end
function rrule!!(
    ::CoDual{typeof(Base.Broadcast.materialize!),NoFData},
    dest::CoDual{P},
    bc::CoDual{<:Broadcasted{<:CuArrayStyle}},
) where {P<:CuMaybeWrappedArray}
    pout, dout = arrayify(dest)
    bc_primal = primal(bc)
    bc_fdata = tangent(bc)
    bc_prepared, flat_bc, flat_pargs, flat_fdatas = _prepare_gpu_broadcast(
        bc_primal, bc_fdata
    )
    scalar_map, scalar_count = _gpu_collect_scalar_map(bc_primal)
    scalar_map = iszero(scalar_count) ? nothing : scalar_map

    # Save primal for restoration in the pullback.
    old_pout = copy(pout)

    # Single GPU kernel: primal + all partial derivatives simultaneously.
    dual_out = _gpu_broadcast_dual(flat_bc.f, flat_pargs...)
    decoded = _gpu_decode_ndual_output(
        Val(:broadcast), dual_out, flat_pargs; extract_partials=true, extract_primal=false
    )

    # Write primal result in-place into dest.
    _gpu_write_broadcast_primal!(pout, dual_out, decoded.is_diff)

    # Non-differentiable output (e.g. Bool arrays): no gradient to propagate.
    # Check eltype(dual_out) (NDual elements), NOT eltype(pout) (plain floats after
    # shared NDual-value extraction): eltype(pout) is always IEEEFloat for CuMaybeComplexArray.
    if !decoded.is_diff
        function materialize!_nodiff_pb!!(::NoRData)
            copyto!(pout, old_pout)
            return NoRData(), NoRData(), zero_rdata(bc_primal)
        end
        return dest, materialize!_nodiff_pb!!
    end

    function materialize!_pb!!(::NoRData)
        # Snapshot dout before any modifications. When dest appears in bc.args
        # (e.g. x .= x .+ y), flat_fdatas contains fd = dout for x's slot.
        # Without a snapshot, _leaf_accum_fdata!(x, dout, contrib) would corrupt
        # dout mid-loop, causing subsequent slots to read a doubled value.
        g = copy(dout)
        fill!(dout, 0)
        r_bc = _gpu_accum_pullback!(
            flat_pargs,
            flat_fdatas,
            decoded.partial_slots,
            g,
            bc_primal,
            scalar_map,
            scalar_count,
        )
        # Restore primal to allow the reverse pass to see the pre-broadcast value.
        copyto!(pout, old_pout)
        return NoRData(), NoRData(), r_bc
    end

    return dest, materialize!_pb!!
end

# Rules for `permutedims(x, perm)` on CuArrays, and on Adjoint/Transpose/SubArray
# wrapping a CuArray (via `arrayify`, same convention as vcat/hcat/cat above).
# frule:    permute the tangent with the same permutation — permutedims is linear.
# pullback: permute the output cotangent with the inverse permutation.
@is_primitive(MinimalCtx, Tuple{typeof(permutedims),CuMaybeWrappedArray,Any})
function frule!!(::Dual{typeof(permutedims)}, x::Dual{<:CuMaybeWrappedArray}, perm::Dual)
    px, dx = arrayify(x)
    pperm = primal(perm)
    return Dual(permutedims(px, pperm), permutedims(dx, pperm))
end
function rrule!!(
    ::CoDual{typeof(permutedims)}, x::CoDual{<:CuMaybeWrappedArray}, perm::CoDual
)
    px, dx = arrayify(x)
    pperm = primal(perm)
    y = permutedims(px, pperm)
    dy_out = zero(y)
    iperm = invperm(pperm)
    function permutedims_pb!!(::NoRData)
        dx .+= permutedims(dy_out, iperm)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy_out), permutedims_pb!!
end

# varm(x, m; corrected, dims) on CuArrays. LayerNorm/GroupNorm/InstanceNorm reach this
# via LuxLib.Impl.mean_var → var → varm; the no-dims path uses a mapreduce closure
# Mooncake can't trace. Split into two primitives: this one for array-valued m (dims
# gives a CuArray output), and one below for scalar m (dims=: gives a scalar output).
#
# Two conventions throughout the varm/mean rules below: the primal value comes from calling
# the real function (rules run natively, so the closure only blocks tracing), inheriting its
# kwarg validation and NaN-on-empty convention; and each hand-rolled derivative uses the same
# arithmetic as its own primal method — λ-prescaled where the primal prescales, divide-after
# where it divides the sum — so value and derivative degrade identically in Float16 edge cases.

# A real-valued fdata/rdata slot can't hold a complex value. When exactly one of x/m
# is complex, the true partial derivative of |x-m|² is the real part of the naive
# complex expression below, so real slots need that projection. Dispatched on Type,
# not applied to a materialized array, so it fuses into the caller's broadcast instead
# of adding a separate kernel.
_realprojector(::Type{T}) where {T<:Real} = real
_realprojector(::Type) = identity

# Real eltypes only, unlike the scalar-m rules below: GPUArrays' accelerated method
# is `varm(A::AbstractGPUArray{<:Real}, M::AbstractArray{<:Real}; dims, corrected)`.
# Complex arrays fall through to Statistics' generic centralize_sumabs2!, which
# scalar-indexes and cannot run on GPU, so a rule covering them would make AD succeed
# where the primal throws. Mixed real precisions need no tie here, unlike scalar-m
# below: this method has no n==0 type bifurcation, so its output type is always
# concrete.
@is_primitive(
    MinimalCtx,
    Tuple{typeof(Core.kwcall),NamedTuple,typeof(varm),CuFloatArray,CuFloatArray},
)
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    kw::Dual{<:NamedTuple},
    ::Dual{typeof(varm)},
    x::Dual{<:CuFloatArray},
    m::Dual{<:CuFloatArray},
)
    pkw = primal(kw)
    px = primal(x)
    dx = tangent(x)
    pm = primal(m)
    dm = tangent(m)
    σ² = varm(px, pm; pkw...)
    # `dims` has no default in GPUArrays' method, so the call above has already thrown
    # unless it is present; `corrected` does default to true there.
    _raw_dims = pkw.dims
    corrected = get(pkw, :corrected, true)
    T = eltype(px)
    # As in the primal: λ-prescale before reducing (a raw product sum could overflow
    # Float16), with λ inverted in Float64 then converted. `one(T) / n` would convert
    # n to T first, turning n > 65504 into Inf16 and silently zeroing every Float16
    # gradient.
    if _raw_dims isa Colon
        λ = convert(T, inv(length(px) - Int(corrected)))
        return Dual(σ², sum((2λ) .* (px .- pm) .* (dx .- dm)))
    end
    # The denominator mirrors GPUArrays' _mean_denom: a repeated dim (dims=(1,1))
    # counts once, and `init=1` keeps an empty dims=() from throwing on the empty
    # prod. sum(...; dims) already deduplicates repeated dims on its own.
    _dims = _raw_dims isa Integer ? (_raw_dims,) : _raw_dims
    n = prod(d -> size(px, d), unique(_dims); init=1)
    λ = convert(T, inv(n - Int(corrected)))
    return Dual(σ², sum((2λ) .* (px .- pm) .* (dx .- dm); dims=_dims))
end

function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(varm)},
    x::CoDual{<:CuFloatArray},
    m::CoDual{<:CuFloatArray},
)
    pkw = primal(kw)
    px, dx = arrayify(x)
    pm, dm = arrayify(m)
    σ² = varm(px, pm; pkw...)
    _raw_dims = pkw.dims  # required by GPUArrays' method; validated by the call above
    corrected = get(pkw, :corrected, true)
    T = eltype(px)
    diff = px .- pm
    _dims = _raw_dims isa Integer ? (_raw_dims,) : _raw_dims
    # Same denominator and Float64-inverted λ as the frule above. Assigned once,
    # outside the branch, so the pullbacks capture coeff as a typed closure field
    # rather than a Core.Box.
    n = _dims isa Colon ? length(px) : prod(d -> size(px, d), unique(_dims); init=1)
    coeff = 2 * convert(T, inv(n - Int(corrected)))
    # Either operand may be broadcast-expanded against the other in the primal, so
    # the pullbacks form the elementwise gradient g at the broadcast shape, then
    # unbroadcast it back to each operand's own shape. Forming g before any reduction
    # also covers empty x: g is empty and the unbroadcast sums give clean zeros,
    # instead of coeff(=Inf at n==0, corrected=false) * 0 = NaN.
    if _dims isa Colon
        # Scalar output: σ² gets NoFData, and the pullback receives its cotangent
        # functionally rather than via a pre-accumulated fdata mutation.
        function varm_colon_pb!!(dσ²)
            g = coeff * dσ² .* diff
            dx .+= _unbroadcast(g, size(px))
            dm .-= _unbroadcast(g, size(pm))
            return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(σ², NoFData()), varm_colon_pb!!
    end
    dσ² = zero(σ²)
    function varm_pb!!(::NoRData)
        # dσ² varies along the non-reduced dims, so it must sit inside the
        # unbroadcast reductions rather than factoring out.
        g = coeff .* dσ² .* diff
        dx .+= _unbroadcast(g, size(px))
        dm .-= _unbroadcast(g, size(pm))
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(σ², dσ²), varm_pb!!
end

# NOTE: no bare `varm(x, m::AbstractArray)` primitive here, unlike the scalar-m case
# below. GPUArrays' override, `varm(A::AbstractGPUArray, M::AbstractArray; dims,
# corrected=true)`, has no default for `dims`, so for the Real eltypes these rules
# cover, this 2-arg spelling always throws `UndefKeywordError` even without AD. A
# rule here would make AD succeed where the real function throws.

# varm(x, m::scalar; corrected): the no-dims case, m from mean(x; dims=:). Same
# closure-mapreduce problem as above, but m here is a plain scalar, not an array.
# Complex x/m are supported here, unlike the array-m rules above: the primal path is
# Statistics' generic scalar-m mapreduce, which GPUArrays handles for any eltype.

# JVP for the scalar-m Colon case, shared by the kwcall and bare rules. The primal is
# Statistics' generic `centralize_sumabs2(A, m) / (n - Int(corrected))`, which divides
# AFTER summing in T/Int promotion arithmetic. The tangent does the same: this method
# also runs (traced) on the CPU, and a CUDA-only rule must not give it a different
# derivative — at Float16 n > 65504 the promoted denominator is Inf16, zeroing the
# quotient on both backends.
function _varm_scalar_colon_tangent(px, dx, pm, dm, corrected::Bool)
    # n == 0: the primal takes Statistics' constant-NaN branch, so the derivative is
    # the zero map (dividing the empty sum by 0 would instead manufacture a NaN).
    n = length(px)
    n == 0 && return zero(real(eltype(px)))
    # real(conj(diff)*...) is the JVP for both real and complex diff (σ² is always
    # real, so dσ² must be too). conj/real are dot-called so they fuse into one kernel
    # with the subtraction/product, instead of allocating a separate array just to
    # support the complex case.
    return 2 * sum(real.(conj.(px .- pm) .* (dx .- dm))) / (n - Int(corrected))
end

# Pullback for the scalar-m Colon case, with the same divide-after arithmetic as the
# tangent above. coeff is applied elementwise BEFORE reducing, as in the array-m
# pullbacks: a precomputed raw sum(diff) can overflow Float16 to Inf while coeff is
# zeroed by its Inf16-promoted denominator, turning m's cotangent into 0 * Inf = NaN
# where the CPU-traced pullback gives 0. The same ordering makes n == 0 guard-free:
# g is then empty, so both cotangents reduce to clean zeros even though
# coeff = dσ² / 0 = Inf at corrected=false.
function _varm_scalar_colon_pb(px, dx, pm, corrected::Bool)
    denom = length(px) - Int(corrected)
    diff = px .- pm
    function pb!!(dσ²)
        coeff = 2 * (dσ² / denom)
        g = coeff .* diff
        dx .+= _realprojector(eltype(dx)).(g)
        return typeof(pm)(_realprojector(typeof(pm))(-sum(g)))
    end
    return pb!!
end

# Scalar m must share x's underlying precision: with e.g. Float32 data and a Float64 mean,
# Statistics' scalar-m varm infers Union{Float32,Float64} (the n==0 branch types σ² off x
# alone, the main branch promotes with m), and Mooncake's rule builder cannot handle a
# Union-typed primitive output. Mismatched combinations are left to fail in the interpreter.
# Hence the tie takes four explicit eltype combinations, not one
# `Tuple{..., CuArray{<:Union{P,Complex{P}}}, Union{P,Complex{P}}} where P`: there P never
# occurs twice covariantly, so it is not diagonal and subtyping may bind it to
# Union{Float32,Float64}, matching mixed precision anyway; the invariant CuArray{P} pins it
# to one concrete eltype. frule!!/rrule!! below keep broad signatures; they are only reached
# for claimed primitives.
for A in (:(CuArray{P}), :(CuArray{Complex{P}})), M in (:P, :(Complex{P}))
    @eval @is_primitive(
        MinimalCtx,
        Tuple{typeof(Core.kwcall),NamedTuple,typeof(varm),$A,$M} where {P<:IEEEFloat},
    )
    @eval @is_primitive(MinimalCtx, Tuple{typeof(varm),$A,$M} where {P<:IEEEFloat})
end
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    kw::Dual{<:NamedTuple},
    ::Dual{typeof(varm)},
    x::Dual{<:CuMaybeComplexArray},
    m::Dual{<:CuFloatOrComplex},
)
    pkw = primal(kw)
    px = primal(x)
    pm = primal(m)
    # Statistics' scalar-m method has no dims kwarg, so a stray dims=1 throws here
    # exactly as it does without AD.
    σ² = varm(px, pm; pkw...)
    corrected = get(pkw, :corrected, true)
    dσ² = _varm_scalar_colon_tangent(px, tangent(x), pm, tangent(m), corrected)
    return Dual(σ², dσ²)
end

function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(varm)},
    x::CoDual{<:CuMaybeComplexArray},
    m::CoDual{<:CuFloatOrComplex},
)
    pkw = primal(kw)
    px, dx = arrayify(x)
    pm = primal(m)
    σ² = varm(px, pm; pkw...)
    corrected = get(pkw, :corrected, true)
    pb!! = _varm_scalar_colon_pb(px, dx, pm, corrected)
    function varm_scalar_mean_pb!!(dσ²)
        return NoRData(), NoRData(), NoRData(), NoRData(), pb!!(dσ²)
    end
    return CoDual(σ², NoFData()), varm_scalar_mean_pb!!
end

# Bare `varm(x, m::scalar)`: a kwarg-less call dispatches straight to varm rather than
# through Core.kwcall, so it needs its own rules. Statistics.jl's scalar-m method has no
# `dims` kwarg, so this spelling is always the Colon case with corrected=true. Its
# @is_primitive declarations live in the loop above so the precision tie is stated once.
function frule!!(
    ::Dual{typeof(varm)}, x::Dual{<:CuMaybeComplexArray}, m::Dual{<:CuFloatOrComplex}
)
    px = primal(x)
    pm = primal(m)
    σ² = varm(px, pm)
    dσ² = _varm_scalar_colon_tangent(px, tangent(x), pm, tangent(m), true)
    return Dual(σ², dσ²)
end
function rrule!!(
    ::CoDual{typeof(varm)}, x::CoDual{<:CuMaybeComplexArray}, m::CoDual{<:CuFloatOrComplex}
)
    px, dx = arrayify(x)
    pm = primal(m)
    σ² = varm(px, pm)
    pb!! = _varm_scalar_colon_pb(px, dx, pm, true)
    varm_scalar_bare_pb!!(dσ²) = (NoRData(), NoRData(), pb!!(dσ²))
    return CoDual(σ², NoFData()), varm_scalar_bare_pb!!
end

# mean(x; dims) on CuArrays, real and complex. GPUArrays' _mean uses a mapreduce with
# a captured scalar Mooncake can't trace. dims=: returns a scalar (NoFData tangent);
# an explicit dims returns a CuArray.

@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(mean),CuMaybeComplexArray},
)
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    kw::Dual{<:NamedTuple},
    ::Dual{typeof(mean)},
    x::Dual{<:CuMaybeComplexArray},
)
    pkw = primal(kw)
    px = primal(x)
    dx = tangent(x)
    μ = mean(px; pkw...)
    raw_dims = get(pkw, :dims, :)
    if raw_dims isa Colon
        # The primal's Colon branch is sum(A)/length(A); dividing after summing keeps
        # the gradient identical to the traced bare-mean decomposition (at Float16
        # n > 65504 both give exactly zero, where a λ-prescale would not). n == 0:
        # the primal is a constant NaN, so the JVP is the zero map, consistent with
        # the rrule — 0/0 would manufacture a NaN tangent.
        n = length(px)
        return Dual(μ, n == 0 ? zero(μ) : sum(dx) / n)
    end
    _dims = raw_dims isa Integer ? (raw_dims,) : raw_dims
    n = prod(d -> size(px, d), unique(_dims); init=1)
    λ = eltype(px)(inv(n))
    return Dual(μ, sum(λ .* dx; dims=_dims))
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(mean)},
    x::CoDual{<:CuMaybeComplexArray},
)
    pkw = primal(kw)
    px, dx = arrayify(x)
    μ = mean(px; pkw...)
    raw_dims = get(pkw, :dims, :)
    _dims = raw_dims isa Integer ? (raw_dims,) : raw_dims
    # Assigned once, outside the branch, so the pullbacks capture n un-boxed.
    n = _dims isa Colon ? length(px) : prod(d -> size(px, d), unique(_dims); init=1)
    if _dims isa Colon
        # Scalar output: pullback receives scalar rdata. Same divide-after
        # arithmetic as the frule above.
        function mean_scalar_pb!!(dμ)
            dx .+= dμ / n
            return NoRData(), NoRData(), NoRData(), NoRData()
        end
        return CoDual(μ, NoFData()), mean_scalar_pb!!
    end
    λ = eltype(px)(inv(n))
    dμ = zero(μ)
    function mean_array_pb!!(::NoRData)
        dx .+= dμ .* λ
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(μ, dμ), mean_array_pb!!
end

# NOTE: no bare `mean(x)` primitive here, unlike scalar-m `varm` above. `mean`'s
# Colon branch is just `sum(A)/length(A)`, with no closure over a differentiable
# value, so ordinary decomposition already differentiates it correctly.

end
