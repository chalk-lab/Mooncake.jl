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
        scalar_grads = has_scalars ? Any[] : nothing
        offset = 0
        for (pa, fd) in zip(flat_pargs, flat_fdatas)
            dof = _broadcast_elem_dof(pa)
            if dof == 1
                k = offset + 1
                contrib = broadcast((p, d) -> real(conj(d) * p), partial_slots[k], dout)
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
                    dout,
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
        # Zero dout: gradient has been propagated; earlier ops accumulate fresh.
        fill!(dout, 0)
        # Restore primal to allow the reverse pass to see the pre-broadcast value.
        copyto!(pout, old_pout)
        return NoRData(), NoRData(), r_bc
    end

    return dest, materialize!_pb!!
end

end
