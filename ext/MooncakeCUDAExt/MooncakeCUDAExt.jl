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
using CUDA.CUDACore.GPUArrays: derive, unsafe_free!
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
    zero_adjoint,
    zero_derivative,
    zero_fcodual,
    zero_tangent,
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
    NDual,
    NDualArray,
    Lifted,
    NoDual,
    dual_type,
    Nfwd,
    ImmutableDual,
    MutableDual,
    zero_lifted,
    _typeof

import Mooncake.TestUtils:
    populate_address_map_internal, AddressMap, __increment_should_allocate

# NDual lives in Mooncake.Nfwd and is loaded as part of Mooncake core.

const CuFloatArray = CuArray{<:IEEEFloat}
const CuComplexArray = CuArray{<:Complex{<:IEEEFloat}}
const CuMaybeComplexArray = Union{CuFloatArray,CuComplexArray}
# Index and mask arrays. Deliberately a whitelist rather than a predicate on
# tangent_type(eltype), which cannot be written as a dispatch signature without also
# capturing the differentiable eltypes handled above. An eltype outside it whose
# derivative is nonetheless structurally zero (Char, Symbol, Enum) therefore falls to the
# generic struct handler, which fails building CuArray's inner DataRef; extend the union
# if such a case turns up.
const CuNonDiffArray = CuArray{<:Union{Integer,Bool,CartesianIndex}}
# `x .= 0f0` reaches the two-argument `materialize!` as a zero-dimensional Broadcasted: it is
# handed the right-hand side's style alone, and the destination is combined in only inside
# the three-argument method, so a scalar right-hand side never carries `CuArrayStyle`. A
# `DefaultArrayStyle{N}` for N > 0 means host arrays are involved and stays out.
const _GpuMaterializeStyle = Union{CuArrayStyle,Base.Broadcast.DefaultArrayStyle{0}}

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
# CuDataRef has NoDual V (opaque handle); copy primal only — the V is
# already a no-derivative sentinel and needs no per-lane work.
function frule!!(::Lifted{typeof(copy),Nw}, x::Lifted{<:CuDataRef,Nw,NoDual}) where {Nw}
    return Lifted{_typeof(primal(x)),Nw}(copy(primal(x)), NoDual())
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

# Keep GPU-backed `NDualArray`s out of the nfwd-native path: running the primal element-wise over
# the dual array scalar-indexes the device array (disallowed). Forcing `_nfwd_projectable` false
# routes every CuArray forward op to the transform, which uses the device frules below. (Without
# this, on Julia 1.10 the classifier admitted composed reductions like `sum(f, ::CuArray)` to nfwd
# and they scalar-indexed; 1.12's block layout differs but the hazard is identical.)
Mooncake._nfwd_backing_projectable(::Type{<:CuArray}) = false

# Forward-mode canonical V for CUDA primitives — mirrors the host
# (`Array{T,D}` / `Ptr{T}` / etc.) V shapes:
#
#   CuArray{T<:IEEEFloat,D}            → NDualArray{T,N,D,CuArray{T,D},NDual{T,N},CuArray{T,D+1}}
#   CuArray{Complex{R<:IEEEFloat},D}   → NDualArray{Complex{R},N,D,…,Complex{NDual{R,N}},CuArray{Complex{R},D+1}}
#   CuPtr{T}                            → NTuple{N, CuPtr{T}}
#   CuDataRef (any memory-kind variant) → NoDual (opaque handle)
#
# The concrete block type `B` (6th param) must be spelled out — `Lifted`/`NDualArray` are
# invariant in it, so a `B`-free `dual_type` would fail the `::dual_type(...)` seed typeasserts.
# `NDualArray` accepts any `AbstractArray{T,D}` storage by construction, including `CuArray`; the
# block is LANE-MAJOR `(dims..., N)` for `CuArray` (see `_block_dims`/`tangent_view` below).
@foldable @inline function dual_type(
    ::Val{N}, ::Type{P}
) where {N,T<:IEEEFloat,D,P<:CuArray{T,D}}
    return NDualArray{T,N,D,P,NDual{T,N},Nfwd._block_type(P)}
end
@foldable @inline function dual_type(
    ::Val{N}, ::Type{P}
) where {N,R<:IEEEFloat,D,P<:CuArray{Complex{R},D}}
    return NDualArray{Complex{R},N,D,P,Complex{NDual{R,N}},Nfwd._block_type(P)}
end

# LANE-MAJOR block for a `CuArray`: `CuArray{T,D+1,M}` of size `(dims..., N)`. The host block is
# element-major `(N, dims...)` so scalar `getindex` reads a contiguous lane column — but a GPU
# never scalar-indexes (CUDA forbids it), so that layout would only make each lane a stride-`N`
# view, which the low-level CUDA primitives (`unsafe_copyto!`/`unsafe_free!`/cuBLAS batch) reject.
# With the lane as the LAST dimension, lane `k` is the contiguous slice `view(block, colons..., k)`
# — accepted by those primitives and by per-lane broadcasts with no gather (and it lines up with
# the batched-cuBLAS follow-on). `_block_dims`/`_block_shape_ok`/`tangent_view` below carry this
# orientation; `_block_type` is orientation-free (the type is the same either way).
@inline Nfwd._block_type(::Type{CuArray{T,D,M}}) where {T,D,M} = CuArray{T,D + 1,M}
@inline Nfwd._block_dims(N::Int, p::CuArray) = (size(p)..., N)
@inline Nfwd._block_shape_ok(block::CuArray, N::Int, p) = size(block) == (size(p)..., N)
# Lane `k`'s partial: the contiguous last-dim slice. Overrides the host element-major
# `tangent_view` (which views the leading lane axis); `_lane_views` builds on this.
@inline function Nfwd.tangent_view(
    a::NDualArray{E,N,D,A}, k::Integer
) where {E,N,D,A<:CuArray}
    return view(getfield(a, :partials_block), ntuple(_ -> Colon(), Val(D))..., k)
end
# Whole per-lane tuple. Overrides the host generic, which slices the block's LEADING axis
# (element-major `(N, dims...)`); the CuArray block is lane-major `(dims..., N)`, so build
# from the lane-major `tangent_view` above instead.
@inline Nfwd._lane_views(a::NDualArray{E,N,D,A}) where {E,N,D,A<:CuArray} = ntuple(
    k -> Nfwd.tangent_view(a, k), Val(N)
)
# GPU-friendly pack: the generic `_pack_block` fills element-by-element (scalar `setindex!`,
# which a `CuArray` forbids). Copy each lane's partial into its block slice `[dims..., k]` — one
# device copy per lane, no scalar indexing. `copyto!` accepts any-typed source, so lane views of
# another block (`SubArray`, e.g. reconstructing a result from an input's lane views) work too.
@inline function _cu_pack_lane_major(p::CuArray, ts, Nw::Int)
    block = Nfwd._block_type(typeof(p))(undef, Nfwd._block_dims(Nw, p)...)
    colons = ntuple(_ -> Colon(), Val(ndims(p)))
    for k in 1:Nw
        copyto!(view(block, colons..., k), ts[k])
    end
    return block
end
# `CuArray` tangents: strictly out-specialises the generic `_pack_block(::A, ::NTuple{N,A})`.
function Nfwd._pack_block(p::A, ts::NTuple{Nw,A}) where {T,Nw,A<:CuArray{T}}
    _cu_pack_lane_major(p, ts, Nw)
end
# View / other-array tangents (the generic, which binds the tuple eltype to `A`, does not apply).
function Nfwd._pack_block(p::CuArray, ts::NTuple{Nw,<:AbstractArray}) where {Nw}
    _cu_pack_lane_major(p, ts, Nw)
end

# The last two block accessors address ONE element's lane. The host versions index the
# element-major block at `(elem-1)*N + lane`; on the lane-major CuArray block that reaches lane
# `((p-1) ÷ n)+1` of element `((p-1) mod n)+1` — the wrong axis, so the assembled gradient is
# scrambled at every chunk width except `W == n`, where the two orderings coincide. Lane `lane`'s
# element `elem` is at `(lane-1)*n + elem` here. Both remain scalar `CuArray` accesses, so the
# seeding fast paths still need `CUDA.allowscalar(true)` and cost a launch per element per lane;
# what changes is that they no longer return a wrong answer to callers who have enabled it.
@inline function Nfwd._set_partial!(
    a::NDualArray{E,N,D,A}, elem::Int, lane::Int, v
) where {E,N,D,A<:CuArray}
    @inbounds getfield(a, :partials_block)[(lane - 1) * length(getfield(a, :primal)) + elem] =
        v
    return a
end
@inline function Nfwd._get_partial(
    a::NDualArray{E,N,D,A}, elem::Int, lane::Int
) where {E,N,D,A<:CuArray}
    return @inbounds getfield(a, :partials_block)[(lane - 1) * length(getfield(a, :primal)) + elem]
end

# Seed factories for CuArray (mirror the host `Array{T,D}` overloads in `src/lifted.jl`):
# the @generated struct-lift fallback would recurse into CuArray's internal `Ptr` fields
# and fail; an explicit `NDualArray` seed keeps `zero_dual` / `uninit_dual` / `randn_dual`
# coherent with `dual_type`.
@inline function Mooncake.zero_dual(::Val{N}, x::A) where {N,A<:CuMaybeComplexArray}
    return NDualArray{eltype(A),N,ndims(A),A}(x)
end
# `uninit_dual` coincides with `zero_dual`: the single-arg `NDualArray` constructor zero-inits
# the slot-local partials.
@inline function Mooncake.uninit_dual(w::Val{N}, x::A) where {N,A<:CuMaybeComplexArray}
    return Mooncake.zero_dual(w, x)
end
@inline function Mooncake.randn_dual(
    ::Val{N}, rng::Random.AbstractRNG, x::A
) where {N,A<:CuMaybeComplexArray}
    partials = ntuple(_ -> A(randn(rng, eltype(A), size(x)...)), Val(N))
    return NDualArray{eltype(A),N,ndims(A),A}(x, partials)
end
# Cache-aware seed delegations: a `CuArray` has a custom `NDualArray` V, so the cache-aware
# `_*_dual_internal` must delegate to the cache-free factory above (like core's `Array`
# delegation) rather than fall to the generic struct-lift @generated, which would recurse into
# CuArray's internal `DataRef`/`Ptr` fields. Register by primal identity so aliased CuArrays
# (e.g. from `reshape`/`view`) share one V.
const _CuDualArray = Union{CuArray{<:IEEEFloat},CuArray{<:Complex{<:IEEEFloat}}}
for (factory, internal) in
    ((:zero_dual, :_zero_dual_internal), (:uninit_dual, :_uninit_dual_internal))
    @eval function Mooncake.$internal(w::Val{N}, x::_CuDualArray, d::MaybeCache) where {N}
        haskey(d, x) && return d[x]::dual_type(Val(N), typeof(x))
        v = Mooncake.$factory(w, x)
        d[x] = v
        return v
    end
end
function Mooncake._randn_dual_internal(
    w::Val{N}, rng::Random.AbstractRNG, x::_CuDualArray, d::MaybeCache
) where {N}
    haskey(d, x) && return d[x]::dual_type(Val(N), typeof(x))
    v = Mooncake.randn_dual(w, rng, x)
    d[x] = v
    return v
end
# Non-differentiable T (`tangent_type(T) === NoTangent`) makes the whole CuPtr a
# `NoDual` slot — coherent with reverse-mode (`tangent_type(CuPtr{Cvoid}) === NoTangent`).
# Differentiable T keeps the per-lane tangent pointer V.
@foldable @inline function dual_type(::Val{N}, ::Type{CuPtr{T}}) where {N,T}
    return tangent_type(T) === NoTangent ? NoDual : NTuple{N,CuPtr{T}}
end

# `zero_dual`/`uninit_dual` build the same per-lane null-pointer seed; generate both from the name
# (mirroring the `_*_dual_internal` delegation loop above and core's `zero_dual`/`uninit_dual`
# loop). `randn_dual` shares the body but keeps its own method for the extra `rng` argument.
for factory in (:zero_dual, :uninit_dual)
    @eval @inline function Mooncake.$factory(::Val{N}, x::CuPtr{T}) where {N,T}
        tangent_type(T) === NoTangent && return NoDual()
        return ntuple(_ -> CuPtr{T}(UInt64(0)), Val(N))
    end
end
@inline function Mooncake.randn_dual(
    ::Val{N}, ::Random.AbstractRNG, x::CuPtr{T}
) where {N,T}
    tangent_type(T) === NoTangent && return NoDual()
    return ntuple(_ -> CuPtr{T}(UInt64(0)), Val(N))
end

# Width-1 `lift` overloads for CuPtr / CuArray — mirror the host `Ptr` / `Array`
# `lift` overloads in `src/lifted.jl`. Without these, the test-side `lift(p, ẋ)`
# boundary call MethodErrors for CuPtr / CuArray inputs.
@inline function Mooncake.lift(x::CuPtr{T}, ẋ::CuPtr{T}) where {T}
    return Mooncake.Lifted{CuPtr{T},1}(x, (ẋ,))
end
@inline function Mooncake.lift(x::A, ẋ::A) where {A<:CuMaybeComplexArray}
    return Mooncake.Lifted{A,1}(x, NDualArray{eltype(A),1,ndims(A),A}(x, (ẋ,)))
end
# CuDataRef is non-differentiable (V === NoDual). This `lift(x::A, ::A)` method accepts a
# same-typed second argument (the tangent the test harness supplies) and discards it, producing
# the canonical NoDual V.
@inline function Mooncake.lift(x::A, ::A) where {A<:CuDataRef}
    return Mooncake.Lifted{A,1}(x, NoDual())
end
# Forward uses NoDual; reverse uses `tangent_type === P` (above). The difference is the aliasing model:
# reverse reuses the handle as *shared* cotangent storage so aliased CuArrays/views accumulate gradient
# into one place (the reverse aliasing invariant). Forward tangents are slot-local — nothing is shared —
# and a CuArray's JVP lives at the array level in the result's `NDualArray` partials (views build that
# via the `view` frule, never through a tangent on the DataRef). So the handle carries no forward derivative.
@foldable @inline dual_type(::Val{N}, ::Type{P}) where {N,P<:CuDataRef} = NoDual
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
    ::Lifted{typeof(unsafe_convert),Nw},
    ::Lifted{Type{CuPtr{T}},Nw},
    x::Lifted{X,Nw,<:NDualArray},
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},X<:CuArray{T},Nw}
    y = unsafe_convert(CuPtr{T}, primal(x))
    x_partials = Nfwd._lane_views(tangent(x))
    dy = ntuple(k -> unsafe_convert(CuPtr{T}, x_partials[k]), Val(Nw))
    return Lifted{CuPtr{T},Nw}(y, dy)
end
function rrule!!(
    ::CoDual{typeof(unsafe_convert)}, ::CoDual{Type{CuPtr{T}}}, x::CoDual{X,X}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},X<:CuArray{T}}
    return CoDual(unsafe_convert(CuPtr{T}, primal(x)), unsafe_convert(CuPtr{T}, x.dx)),
    _nopb(Val(3))
end

# Chunked-Hessian basis seeding for device-resident arrays. The generic
# `_basis_seed!!(::NDualArray)` writes each lane's one-hot with scalar `setindex!`, which a
# CuArray forbids. Each lane consumes one degree of freedom per element (numbered by `cursor`
# in `eachindex` order, i.e. column-major linear for a dense CuArray), so lane `k`'s partial is
# a one-hot at element `slots[k] - base`. Zero the whole block, then write each lane's single
# one-hot into the contiguous block at its LANE-MAJOR position `(k - 1) * n + hot` (lane `k`'s
# slice is the contiguous run `[(k-1)*n + 1 : k*n]`) — a 1-element host→device `copyto!`, no
# scalar indexing.
function Mooncake._basis_seed!!(
    v::NDualArray{T,N,D,A}, slots::NTuple{N,Int}, cursor, dict
) where {T<:IEEEFloat,N,D,A<:CuArray}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    n = length(v.primal)
    base = cursor[]
    onehot = [one(T)]
    block = getfield(v, :partials_block)
    fill!(block, zero(T))
    for k in 1:N
        hot = slots[k] - base
        1 <= hot <= n && copyto!(block, (k - 1) * n + hot, onehot, 1, 1)
    end
    cursor[] += n
    return v
end
# Complex device arrays consume two degrees of freedom per element (real then imaginary).
function Mooncake._basis_seed!!(
    v::NDualArray{Complex{R},N,D,A}, slots::NTuple{N,Int}, cursor, dict
) where {R<:IEEEFloat,N,D,A<:CuArray}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    n = length(v.primal)
    base = cursor[]
    block = getfield(v, :partials_block)
    fill!(block, zero(Complex{R}))
    for k in 1:N
        off = slots[k] - base
        if 1 <= off <= 2n
            j = cld(off, 2)
            val = isodd(off) ? Complex(one(R), zero(R)) : Complex(zero(R), one(R))
            copyto!(block, (k - 1) * n + j, [val], 1, 1)
        end
    end
    cursor[] += 2n
    return v
end

# CuPtr arithmetic: (p::CuPtr{T}) + (n::Integer) offsets a device pointer by n bytes.
# For differentiable T the tangent is also a CuPtr; it must be offset by the same amount
# since primal and tangent arrays are laid out identically.
# For non-differentiable T (e.g. CuPtr{Cvoid} used in memory management), the tangent
# is NoTangent and the pointer arithmetic carries no gradient.
@is_primitive(MinimalCtx, Tuple{typeof(+),CuPtr{T},Integer} where {T})
# Differentiable T: per-lane CuPtr offset.
function frule!!(
    ::Lifted{typeof(+),Nw}, p::Lifted{CuPtr{T},Nw,NTuple{Nw,CuPtr{T}}}, n::Lifted{<:Integer}
) where {Nw,T}
    np = primal(n)
    new_primal = primal(p) + np
    p_partials = tangent(p)
    new_partials = ntuple(k -> p_partials[k] + np, Val(Nw))
    return Lifted{CuPtr{T},Nw}(new_primal, new_partials)
end
# Non-differentiable T: NoDual tangent.
function frule!!(
    ::Lifted{typeof(+),Nw}, p::Lifted{CuPtr{T},Nw,NoDual}, n::Lifted{<:Integer}
) where {Nw,T}
    return Lifted{CuPtr{T},Nw}(primal(p) + primal(n), NoDual())
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
# Assigning NoTangent stops Mooncake from building a struct tangent from CuArray's
# internal fields (data::CuDataRef, maxsize::Int, offset::Int, dims::NTuple).
# The CuMaybeComplexArray rule above takes priority for float and complex arrays.
tangent_type(::Type{<:CuNonDiffArray}) = NoTangent
tangent_type(::Type{<:CuNonDiffArray}, ::Type{NoRData}) = NoTangent

tangent(p::CuMaybeComplexArray, ::NoRData) = p

function arrayify(x::A, dx::A) where {A<:CuMaybeComplexArray}
    return (x, dx)
end
# Forward-mode `arrayify` for GPU arrays. The generic `arrayify(::Lifted)` (blas.jl) is bounded to
# `BlasFloat`, excluding Float16/ComplexF16, but the concat/permutedims frules admit them via
# `CuMaybeWrappedArray`. Delegate to the eltype-agnostic `_arrayify_lane` recursion — which handles a
# dense `CuArray`'s `NDualArray` V and the Adjoint/Transpose/SubArray wrappers alike, with no
# `BlasFloat` bound — mirroring the reverse `arrayify(::A,::A)` above. More specific than the generic
# method on the array type, so it also takes `BlasFloat` GPU arrays (no ambiguity, same result).
function arrayify(x::Lifted{<:CuMaybeWrappedArray,N}) where {N}
    A = Mooncake.primal(x)
    return A, ntuple(lane -> Mooncake._arrayify_lane(A, Mooncake.tangent(x), lane), Val(N))
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
) where {P<:CuNonDiffArray}
    # For non-differentiable CuArrays, compare by content by downloading to CPU.
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
    c[key] = 0.0
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
    ::Lifted{typeof(reshape),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    dims::Lifted{<:NTuple},
) where {Nw}
    _dims = primal(dims)
    y = reshape(primal(x), _dims)
    # The result shares the input's memory, so its block must share the input's block, not copy
    # it — otherwise an in-place write through the reshape never reaches the input's tangent and
    # the JVP is silently wrong. A reshape keeps every element, so the lane-major block simply
    # reshapes to `(_dims..., Nw)` over the same device memory.
    blk = reshape(getfield(tangent(x), :partials_block), (_dims..., Nw))
    Y = typeof(y)
    Element = eltype(y)
    V = NDualArray{Element,Nw,ndims(y),Y,Nfwd._wrapped_eltype(Element, Val(Nw)),typeof(blk)}(
        y, blk
    )
    return Lifted{Y,Nw}(y, V)
end
function rrule!!(
    ::CoDual{typeof(reshape)}, x::CoDual{<:CuMaybeComplexArray}, dims::CoDual{<:NTuple}
)
    _dims = primal(dims)
    return CoDual(reshape(primal(x), _dims), reshape(x.dx, _dims)), _nopb(Val(3))
end

# GPUArrays.derive is the single funnel for `view`, `reshape` and both `reinterpret`
# spellings. Claiming it keeps the derivation on the arrays, where the offset it adds is
# relative to whichever array it is handed, so the tangent is sliced from its own base. The
# `_new_` rules below cannot do that: they see the primal's offset already resolved against
# the primal's allocation, while a tangent from `zero_tangent` starts at offset 0, and
# applying one to the other lands `x.offset` bytes too far in — silently shifted while it
# stays inside the buffer, and writing into whatever the pool handed out next once it does
# not. The derived tangent shares the source tangent's DataRef, so accumulation into a view
# reaches the parent's tangent and the pullback has nothing to do.
# `derive` also backs both `reinterpret` spellings, which may change the element type. The
# tangent is derived the same way, which is right exactly when the underlying real field is
# unchanged: same eltype for a view or reshape, and Complex{T} against T for a reinterpret
# that interleaves the parts, whose tangent interleaves identically. Anything else -- Float64
# read as Float32 pairs, a float read as an integer -- hands back a tangent whose elements are
# bit-halves of the real one, which accumulates into the parent as silent corruption.
function _check_derive_eltype(::Type{T}, pa) where {T}
    real(T) === real(eltype(pa)) && return nothing
    return _throw_gpu_argument_error(
        "Mooncake: reinterpreting a CuArray of $(eltype(pa)) as $T is not differentiable. " *
        "The tangent would be reinterpreted the same way, so its elements would no longer " *
        "line up with the primal's. Reinterpreting between a complex eltype and its own " *
        "real part is supported. " *
        _UNIMPL_MSG,
    )
end
@is_primitive(MinimalCtx, Tuple{typeof(derive),Type,CuMaybeComplexArray,Dims,Int})
# Forward: derive each lane's partial exactly as the primal is derived. A lane view is a
# contiguous slice of the lane-major block, hence a `CuArray` in its own right, so `derive`
# applies to it directly.
function frule!!(
    ::Lifted{typeof(derive),Nw},
    ::Lifted{Type{T},Nw},
    a::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    dims::Lifted,
    offset::Lifted,
) where {T,Nw}
    pa = primal(a)
    _check_derive_eltype(T, pa)
    d, o = primal(dims), primal(offset)
    y = derive(T, pa, d, o)
    Y = typeof(y)
    # `derive` is the funnel for `view`, `reshape` and `reinterpret`, so the same rule applies here
    # as in the `view` frule: a result covering the whole parent at the same element type shares the
    # parent's block, reshaped. Packing a fresh block instead detaches the tangent, and a write
    # through the result then never reaches the parent's — silently.
    # Full byte coverage from offset 0 can share the block even when the element type changes: the
    # lane-major block's slabs are per-lane and contiguous, so reinterpreting it to `T` splits or
    # merges elements exactly as it does in the primal. That keeps real<->complex `reinterpret`
    # aliased rather than snapshotted.
    if o == 0 && length(y) * sizeof(T) == length(pa) * sizeof(eltype(pa))
        pblk = getfield(tangent(a), :partials_block)
        blk = reshape(reinterpret(T, pblk), (size(y)..., Nw))
        V = NDualArray{T,Nw,ndims(y),Y,Nfwd._wrapped_eltype(T, Val(Nw)),typeof(blk)}(y, blk)
        return Lifted{Y,Nw}(y, V)
    end
    # As in the `view` frule: anything short of full same-eltype coverage cannot share the block,
    # and a copied block is a snapshot that decays the moment the parent is written.
    throw(
        ArgumentError(
            "Forward mode cannot derive a partial or element-retyped `CuArray` (offset $o, " *
            "eltype $T from $(eltype(pa))): the result's per-lane partials cannot share the " *
            "parent's block, so its derivative would silently detach. Materialise the result " *
            "instead of viewing/reinterpreting in place.",
        ),
    )
end
function rrule!!(
    ::CoDual{typeof(derive)},
    ::CoDual{Type{T}},
    a::CoDual{<:CuMaybeComplexArray},
    dims::CoDual,
    offset::CoDual,
) where {T}
    pa, da = arrayify(a)
    _check_derive_eltype(T, pa)
    d, o = primal(dims), primal(offset)
    return CoDual(derive(T, pa, d, o), derive(T, da, d, o)), _nopb(Val(5))
end

# A contiguous `view(::CuArray, range)` hands back a `CuArray` that is a strict sub-region of the
# parent's allocation. Its forward block cannot alias the parent's: the block is lane-major, so the
# parent bytes belonging to the view are strided across lanes and no `CuArray` describes them. The
# `view` frule below therefore copies, which reads correctly and writes wrongly — the write lands
# in the copy and never reaches the parent's tangent. Refuse such a write rather than return a
# silently wrong JVP. `reshape`/`vec` keep every element and so alias their block outright.
# `view(::CuArray, inds...)` of a contiguous range reconstructs a CuArray via GPU pointer
# arithmetic (`unsafe_contiguous_view` → `_new_(CuArray, parent.data, …)`). Made a
# FORWARD-mode primitive so the forward transform does not trace into that primal-only
# reconstruction and drop the parallel per-lane partials: view the primal and each partial
# alike, mirroring `reshape` above. Reverse mode is NOT a primitive — the traced path
# already produces the canonical tangent for both contiguous (CuArray) and non-contiguous
# (SubArray) results; a hand-written rrule!! here returned a malformed SubArray CoDual.
@is_primitive(
    MinimalCtx, Mooncake.ForwardMode, Tuple{typeof(view),CuMaybeComplexArray,Vararg}
)
function frule!!(
    ::Lifted{typeof(view),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    inds::Vararg{Lifted,M},
) where {Nw,M}
    _inds = map(primal, inds)
    y = view(primal(x), _inds...)
    x_partials = Nfwd._lane_views(tangent(x))
    if y isa CuMaybeComplexArray
        Y = typeof(y)
        # A view spanning the whole parent covers the entire lane-major block, so it can share
        # that block — as `reshape` does. Copying it instead would detach the tangent and a write
        # through the view would never reach the parent's, silently. The test is COVERAGE, not
        # shape: `view(M, :)` over a matrix spans the whole allocation while changing rank, so
        # requiring equal shapes left exactly that case detached and silently wrong. Reshaping the
        # parent's block to the view's shape covers both, and is a no-op when the shape is equal.
        # `y.offset == primal(x).offset`, not `== 0`: the test is whether the view STARTS where
        # the parent does, and a parent that is itself an offset view carries a non-zero offset that
        # every full-coverage view of it inherits. Requiring zero refused exactly those — a
        # `view(v, 1:length(v))` over `v = view(a, 3:6)` covers all of `v` and was rejected.
        if length(y) == length(primal(x)) && y.offset == primal(x).offset
            blk = reshape(getfield(tangent(x), :partials_block), (size(y)..., Nw))
            V = NDualArray{
                eltype(y),Nw,ndims(y),Y,Nfwd._wrapped_eltype(eltype(y), Val(Nw)),typeof(blk)
            }(
                y, blk
            )
            return Lifted{Y,Nw}(y, V)
        end
        # An empty result has no element whose partial could be stranded, so the copy below is
        # harmless and the refusal would only reject a no-op.
        if !isempty(y)
            # A strict sub-range cannot share the block as ONE array: it is strided across lanes
            # in the lane-major layout. Copying instead makes the block a snapshot, wrong in both
            # directions — a write through the view never reaches the parent's tangent, and a write
            # to the PARENT leaves the snapshot stale, so even reading through the view returns a
            # pre-mutation derivative. Only the first is detectable at the write, so the view is
            # refused where it is taken.
            #
            # Holding `N` borrowed per-lane arrays would work instead — each lane's sub-range IS a
            # contiguous `CuArray` — and is declined deliberately: a borrowed block and an owned one
            # would share a type and support the same operations, differing only in whether they
            # alias, which no signature can express. A site written for one then returns a plausible
            # wrong derivative for the other rather than failing. Refusing here keeps one kind of
            # block, so no site can mishandle a second.
            throw(
                ArgumentError(
                    "Forward mode cannot take a partial view of a `CuArray`: the view's per-lane " *
                    "partials cannot share the parent's block (a sub-range is strided across " *
                    "lanes), so its derivative would silently detach. Materialise the " *
                    "slice instead (`y = x[inds]`), or take a view spanning the whole array.",
                ),
            )
        end
        y_partials = ntuple(k -> view(x_partials[k], _inds...), Val(Nw))
        return Lifted{Y,Nw}(y, NDualArray{eltype(y),Nw,ndims(y),Y}(y, y_partials))
    end
    # Non-contiguous indices yield a SubArray, whose canonical V is the struct lift
    # through the parent — the parent field must alias THIS slot's V so the derivative
    # stays connected. Indices/offset/stride are non-differentiable metadata.
    y.parent === primal(x) || throw(
        ArgumentError(
            "view(::CuArray, …) returned a SubArray whose parent is not the input " *
            "array; cannot construct a coherent forward V for index types $(typeof(_inds)).",
        ),
    )
    V = ImmutableDual((
        parent=tangent(x), indices=NoDual(), offset1=NoDual(), stride1=NoDual()
    ))
    return Lifted{typeof(y),Nw}(y, V)
end

# Reverse `_new_` rule for the DataRef-based inner CuArray constructor. The tangent reuses the
# input tangent's DataRef (shared cotangent storage), so gradient accumulation propagates
# automatically. There is deliberately NO forward parallel: `dual_type(CuDataRef) === NoDual` makes
# the handle forward-opaque (the JVP lives at the array level in the result's `NDualArray`, not in
# the DataRef), so a forward `_new_(CuArray, DataRef, …)` would have no tangent to propagate — and
# it is never needed, because forward views/reshapes build the result's `NDualArray` directly via
# the `view` frule above, never through this constructor.
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
@inline _cu_lgetfield_data_fdata(dx::CuArray, name) =
    _cuarray_is_data_field(name) ? dx.data : NoFData()

@inline _cudataref_lgetfield_rev(x_primal, name, order=nothing) = CoDual(
    _cu_lgetfield_primal(x_primal, name, order), NoFData()
)
@inline _cuarray_lgetfield_rev(x_primal, x_fdata, name, order=nothing) = CoDual(
    _cu_lgetfield_primal(x_primal, name, order), _cu_lgetfield_data_fdata(x_fdata, name)
)

# CuDataRef field access — fields (`rc`, `freed`, `cached`) are
# reference-counting internals with no derivative flow; Lifted V is `NoDual`.
function frule!!(
    ::Lifted{typeof(lgetfield),Nw},
    x::Lifted{<:CuDataRef,Nw,NoDual},
    ::Lifted{Val{name},Nw},
    ::Lifted{Val{order},Nw},
) where {Nw,name,order}
    y = _cu_lgetfield_primal(primal(x), name, order)
    return Lifted{typeof(y),Nw}(y, NoDual())
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
    ::Lifted{typeof(lgetfield),Nw}, x::Lifted{<:CuDataRef,Nw,NoDual}, ::Lifted{Val{name},Nw}
) where {Nw,name}
    y = _cu_lgetfield_primal(primal(x), name, nothing)
    return Lifted{typeof(y),Nw}(y, NoDual())
end
function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{<:CuDataRef,<:CuDataRef}, ::CoDual{Val{name}}
) where {name}
    return _cudataref_lgetfield_rev(primal(x), name), _nopb(Val(3))
end

# lgetfield rules for CuArray (4 fields: `:data` the DataRef handle, then `:maxsize`/`:offset`/
# `:dims` metadata). Reverse mode (rrule) routes the cotangent through `:data`; the metadata fields
# are non-differentiable. Forward mode (frule) returns a `NoDual` result V for every field — the
# JVP lives in the `NDualArray` partials, not behind the `.data` handle.
function frule!!(
    ::Lifted{typeof(lgetfield),Nw},
    x::Lifted{<:CuArray,Nw,<:NDualArray},
    ::Lifted{Val{name},Nw},
    ::Lifted{Val{order},Nw},
) where {Nw,name,order}
    y = _cu_lgetfield_primal(primal(x), name, order)
    return Lifted{typeof(y),Nw}(y, NoDual())
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
    ::Lifted{typeof(lgetfield),Nw},
    x::Lifted{<:CuArray,Nw,<:NDualArray},
    ::Lifted{Val{name},Nw},
) where {Nw,name}
    y = _cu_lgetfield_primal(primal(x), name, nothing)
    return Lifted{typeof(y),Nw}(y, NoDual())
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
function frule!!(
    ::Lifted{typeof(getindex),Nw}, x::Lifted{<:CuArray}, i::Lifted{<:Integer}
) where {Nw}
    _throw_gpu_argument_error(_SCALAR_IDX_MSG)
end
function rrule!!(::CoDual{typeof(getindex)}, x::CoDual{<:CuArray}, i::CoDual{<:Integer})
    return _throw_gpu_argument_error(_SCALAR_IDX_MSG)
end

@is_primitive(MinimalCtx, Tuple{typeof(setindex!),CuArray,Any,Integer})
function frule!!(
    ::Lifted{typeof(setindex!),Nw}, x::Lifted{<:CuArray}, v::Lifted, i::Lifted{<:Integer}
) where {Nw}
    _throw_gpu_argument_error(_SCALAR_IDX_MSG)
end
function rrule!!(
    ::CoDual{typeof(setindex!)}, x::CoDual{<:CuArray}, v::CoDual, i::CoDual{<:Integer}
)
    return _throw_gpu_argument_error(_SCALAR_IDX_MSG)
end

# Vector indexing: y = x[idx] where idx is a vector of linear or Cartesian indices
# (gather).  Without the CartesianIndex arm the trace falls into `checkbounds`, which
# reduces with `&` and hits the mapreduce catch-all.
#
# frule:    dy = dx[idx]          (gather tangents)
# pullback: scatter-add the output cotangents back to the elements they were read from
#
# A repeated index means several outputs read one element, so that element is owed the sum of
# their cotangents.  `dx[idx] .+= dy` is a read-modify-write per output: where an index
# repeats the reads race and a single contribution survives, silently.  An embedding lookup is
# exactly a repeated gather, so this is scattered atomically instead.  `@atomic` has no
# Complex method, so a complex buffer is scattered as its interleaved real and imaginary
# halves, which are independent sums.
function _gpu_scatter_add_kernel!(dx, lin, dy)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    @inbounds if i <= length(lin)
        CUDACore.@atomic dx[lin[i]] += dy[i]
    end
    return nothing
end
function _gpu_scatter_add_complex_kernel!(rdx, lin, rdy)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    @inbounds if i <= length(lin)
        j = lin[i]
        CUDACore.@atomic rdx[2j - 1] += rdy[2i - 1]
        CUDACore.@atomic rdx[2j] += rdy[2i]
    end
    return nothing
end
function _gpu_scatter_add!(dx, pidx, dy)
    # A logical mask is a claimed spelling as well, since Bool <: Integer, but it names its
    # positions by where it sits rather than by value: scattering it as written would send
    # every selected element to index 1 or to one slot before the buffer. Base converts it
    # for the primal and for the frule's indexing; the kernel needs it done here.
    idx = eltype(pidx) === Bool ? findall(pidx) : pidx
    # The output is what sets the launch size — a mask is longer than the gather it selects.
    n = length(dy)
    iszero(n) && return dx
    # The complex kernel indexes the reinterpreted halves arithmetically, so both paths take
    # linear indices; a Cartesian index vector is converted once, on the device.
    li = LinearIndices(size(dx))
    idx_lin = eltype(idx) <: Integer ? idx : map(I -> li[I], idx)
    # The index vector may live on the host — `x[[1, 2, 3]]` is a perfectly ordinary
    # spelling — and a kernel argument has to be on the device.
    lin = idx_lin isa CuArray ? idx_lin : CuArray(idx_lin)
    threads = min(n, 256)
    blocks = cld(n, threads)
    if eltype(dx) <: Complex
        R = real(eltype(dx))
        CUDACore.@cuda threads = threads blocks = blocks _gpu_scatter_add_complex_kernel!(
            reinterpret(R, dx), lin, reinterpret(R, dy)
        )
    else
        CUDACore.@cuda threads = threads blocks = blocks _gpu_scatter_add_kernel!(
            dx, lin, dy
        )
    end
    return dx
end
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(getindex),CuMaybeComplexArray,AbstractVector{<:Union{Integer,CartesianIndex}}
    },
)
function frule!!(
    ::Lifted{typeof(getindex),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    idx::Lifted{<:AbstractVector{<:Union{Integer,CartesianIndex}}},
) where {Nw}
    pidx = primal(idx)
    px = primal(x)
    y = px[pidx]
    x_partials = Nfwd._lane_views(tangent(x))
    y_partials = ntuple(k -> x_partials[k][pidx], Val(Nw))
    Y = typeof(y)
    Element = eltype(y)
    return Lifted{Y,Nw}(y, NDualArray{Element,Nw,ndims(y),Y}(y, y_partials))
end
function rrule!!(
    ::CoDual{typeof(getindex)},
    x::CoDual{<:CuMaybeComplexArray},
    idx::CoDual{<:AbstractVector{<:Union{Integer,CartesianIndex}}},
)
    px, dx = arrayify(x)
    pidx = primal(idx)
    y = px[pidx]
    dy_out = zero(y)
    function getindex_pb!!(::NoRData)
        _gpu_scatter_add!(dx, pidx, dy_out)
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
function frule!!(
    ::Lifted{typeof(norm),Nw}, x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray}
) where {Nw}
    px = primal(x)
    y = norm(px)
    R = real(eltype(px))
    x_partials = Nfwd._lane_views(tangent(x))
    if iszero(y)
        dy_lanes = ntuple(_ -> zero(R), Val(Nw))
    else
        s_lanes = if Nw == 1
            (real(dot(px, x_partials[1])),)
        else
            # Batch the N per-lane dots (px shared) into one gemv_batched!: dot(px, xpₖ) = conj(px)'·xpₖ,
            # i.e. gemv 'C' with px as a length×1 column (vec covers matrix/Frobenius too). One
            # concatenated readback keeps the N host scalars to a single device transfer (~2.3x vs N dots).
            T = eltype(px)
            pxm = reshape(px, :, 1)
            xvs = [reshape(xp, :) for xp in x_partials]
            outs = [similar(px, T, 1) for _ in 1:Nw]  # beta=0 overwrites, no zeroing needed
            cuBLAS.gemv_batched!('C', one(T), fill(pxm, Nw), xvs, zero(T), outs)
            host = Array(reduce(vcat, outs))
            ntuple(k -> real(host[k]), Val(Nw))
        end
        # dot overflows once norm(px)*norm(dxₖ) leaves the eltype's range, while the JVP it
        # divides down to is still representable — reachable in Float16, whose dot saturates
        # at 65504. The core `BLAS.nrm2` frule scales every element as it accumulates, which
        # here would mean either a temporary or a fused mapreduce costing 25x a cuBLAS dot,
        # so this one pays for the rescale only on the lanes whose dot has already overflowed.
        y_finite = isfinite(y)
        dy_lanes = ntuple(Val(Nw)) do k
            s = s_lanes[k]
            (isfinite(s) || !y_finite) ? s / y : real(dot(px ./ y, x_partials[k]))
        end
    end
    return Lifted{R,Nw}(y, NDual{R,Nw}(y, dy_lanes))
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
function frule!!(
    ::Lifted{typeof(dot),Nw},
    x::Lifted{<:CuFloatArray,Nw,<:NDualArray},
    y::Lifted{<:CuFloatArray,Nw,<:NDualArray},
) where {Nw}
    px = primal(x)
    py = primal(y)
    z = dot(px, py)
    R = eltype(px)
    x_partials = Nfwd._lane_views(tangent(x))
    y_partials = Nfwd._lane_views(tangent(y))
    # `isempty` routes past the batched path: gemv_batched! with a zero-length operand quick-returns
    # before its `beta=0` overwrite, leaving output buffers uninitialised — so dot over empty operands
    # (derivative 0) must use the per-lane form, which is 0 for every lane.
    if Nw == 1 || isempty(px)
        dz_lanes = ntuple(k -> dot(x_partials[k], py) + dot(px, y_partials[k]), Val(Nw))
    else
        # Batch the 2N per-lane dots into 2 gemv_batched! (py, px shared): dot(a,b) = conj(a)'·b via
        # gemv 'C' with the varying operand as a length×1 column; one concatenated readback per term.
        pxm = reshape(px, :, 1)
        pyv = reshape(py, :)
        xms = [reshape(xp, :, 1) for xp in x_partials]
        yvs = [reshape(yp, :) for yp in y_partials]
        o1 = [similar(px, R, 1) for _ in 1:Nw]
        o2 = [similar(px, R, 1) for _ in 1:Nw]
        cuBLAS.gemv_batched!('C', one(R), xms, fill(pyv, Nw), zero(R), o1)  # dot(x_partials[k], py)
        cuBLAS.gemv_batched!('C', one(R), fill(pxm, Nw), yvs, zero(R), o2)  # dot(px, y_partials[k])
        h1 = Array(reduce(vcat, o1))
        h2 = Array(reduce(vcat, o2))
        dz_lanes = ntuple(k -> h1[k] + h2[k], Val(Nw))
    end
    return Lifted{R,Nw}(z, NDual{R,Nw}(z, dz_lanes))
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
# NB: `$_fn` inside a string literal under @eval is runtime interpolation of a
# global that never exists, so build each message here and splice it whole.
# sortperm is not in this list: it returns a permutation of Int indices, so it has no
# derivative for any element type, and the program that uses those indices to gather is
# differentiable through the gather rule.  sort and diff return values and do belong here.
@zero_derivative MinimalCtx Tuple{typeof(sortperm),CuArray}
@zero_derivative MinimalCtx Tuple{typeof(Core.kwcall),NamedTuple,typeof(sortperm),CuArray}

for _fn in (:maximum, :minimum, :diff, :sort)
    _msg = "Mooncake: $_fn on CuArray is not yet differentiable. " * _UNIMPL_MSG
    @eval @is_primitive(MinimalCtx, Tuple{typeof($_fn),CuArray})
    @eval @is_primitive(
        MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof($_fn),CuArray}
    )
    @eval frule!!(::Lifted{typeof($_fn),Nw}, x::Lifted{<:CuArray}; kwargs...) where {Nw} = _throw_gpu_argument_error(
        $_msg
    )
    @eval rrule!!(::CoDual{typeof($_fn)}, x::CoDual{<:CuArray}; kwargs...) = _throw_gpu_argument_error(
        $_msg
    )
    @eval frule!!(::Lifted{typeof(Core.kwcall),Nw}, ::Lifted{<:NamedTuple}, ::Lifted{typeof($_fn),Nw}, x::Lifted{<:CuArray}) where {Nw} = _throw_gpu_argument_error(
        $_msg
    )
    @eval rrule!!(::CoDual{typeof(Core.kwcall)}, ::CoDual{<:NamedTuple}, ::CoDual{typeof($_fn)}, x::CoDual{<:CuArray}) = _throw_gpu_argument_error(
        $_msg
    )
end

# Rules for `maximum`/`minimum` on real CuArrays, bare and `dims` spellings.
# Claimed on CuFloatArray, narrower than the CuArray catch-alls above, so
# integer and complex arrays keep the friendly error.
#
# Ties go to the lowest linear index, because that is what findmax/findmin do.
# Mooncake's CPU path disagrees: it decomposes to `max(acc, xᵢ)`, which returns
# `xᵢ` when they are equal, so the gradient lands on the last tied element.
# Both are valid subgradients; this one matches Base and ChainRules.
#
# The one-hot mask must stay fused inside the enclosing GPU broadcast.
# `CartesianIndices(px) .== _winner(ind)` alone has two host operands, so it
# evaluates to a BitArray that cannot then combine with a CuArray.
#
# GPUArrays returns a linear `Int` index when `ndims(x) == 1` and a
# `CartesianIndex` otherwise; comparing the two is silently `false`, which would
# give an all-zero mask and a silently zero gradient.
_winner(i::Integer) = Ref(CartesianIndex(i))
_winner(i::CartesianIndex) = Ref(i)
_winner(i::AbstractArray{<:CartesianIndex}) = i
_winner(i::AbstractArray{<:Integer}) = CartesianIndex.(i)

# Two reductions have an `init` whose derivative survives the folding, because theirs is
# idempotent where sum's and prod's is not: maximum/minimum, where it competes with the
# elements and takes the whole derivative of every slice it beats, and accumulate(+, …),
# where it adds to each element.  `from_init` says where `init` decided the output: a mask
# for max/min, `true` throughout for accumulate.  A tie counts against `init`, as findmax
# breaks them.
_kw_init_tangent(dkw) = dkw isa NamedTuple ? get(dkw, :init, NoTangent()) : NoTangent()
_kw_init_jvp(dy, ::NoTangent, from_init) = dy
_kw_init_jvp(dy, dinit::Number, from_init) = dy .+ dinit .* from_init
# Takes dy rather than a finished cotangent so that a call with no differentiable `init`
# never launches the reduction.  `sum` also covers the scalar branch, where dy is a number.
_kw_init_rdata(kw_rdata::NoRData, dy, from_init) = kw_rdata
function _kw_init_rdata(kw_rdata::NamedTuple, dy, from_init)
    (haskey(kw_rdata, :init) && !(kw_rdata.init isa NoRData)) || return kw_rdata
    return merge(kw_rdata, (; init=oftype(kw_rdata.init, sum(dy .* from_init))))
end
# findmax answers `dims` and nothing else, so those calls keep the plain argmax path.
_minmax_kw_is_plain(::NamedTuple{K}) where {K} = K === () || K === (:dims,)

# Reducing over an empty extent has a well-defined primal — GPUArrays fills each slice with
# `init`, or with the eltype's identity when there is none — but no argmax to report, and
# findmax/findmin read out of bounds there: a device-side BoundsError that surfaces at some
# later synchronisation.  A Colon reduction over an empty non-vector instead hands `_winner`
# a linear index outside CartesianIndices.  Since no element competes, x's derivative is
# zero and `init` takes all of it.
_empty_minmax_slice(px, ::Colon) = isempty(px)
_empty_minmax_slice(px, dims::Integer) = size(px, dims) == 0
_empty_minmax_slice(px, dims) = any(d -> size(px, d) == 0, dims)

for (_fn, _find, _beat) in ((:maximum, :findmax, :<), (:minimum, :findmin, :>))
    @eval @is_primitive(MinimalCtx, Tuple{typeof($_fn),CuFloatArray})
    @eval @is_primitive(
        MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof($_fn),CuFloatArray}
    )

    @eval function frule!!(
        ::Lifted{typeof($_fn),Nw}, x::Lifted{<:CuFloatArray,Nw}
    ) where {Nw}
        px, x_partials = arrayify(x)
        if isempty(px)
            y_e = $_fn(px)
            zs = ntuple(_ -> zero(eltype(px)), Val(Nw))
            return Lifted{typeof(y_e),Nw}(y_e, _wrap_scalar_v_lanes(y_e, zs))
        end
        y, ind = $_find(px)
        # The one-hot comparison stays inside each lane's broadcast: on its own it has two
        # host operands and evaluates to a BitArray, which no kernel can take (see above).
        dy_lanes = ntuple(
            k -> sum(x_partials[k] .* (CartesianIndices(px) .== _winner(ind))), Val(Nw)
        )
        return Lifted{typeof(y),Nw}(y, _wrap_scalar_v_lanes(y, dy_lanes))
    end
    @eval function rrule!!(::CoDual{typeof($_fn)}, x::CoDual{<:CuFloatArray})
        px, dx = arrayify(x)
        isempty(px) && return CoDual($_fn(px), NoFData()), _nopb(Val(2))
        y, ind = $_find(px)
        function minmax_pb!!(dy)
            dx .+= dy .* (CartesianIndices(px) .== _winner(ind))
            return NoRData(), NoRData()
        end
        return CoDual(y, NoFData()), minmax_pb!!
    end

    # Any keyword beyond `dims` needs the value from the primal call: `init` can beat every
    # element, leaving the result independent of x, and it also fixes the output eltype, so
    # the tangent is converted to follow it.  A keyword the primal rejects then still raises
    # its MethodError.  `won` is a negated comparison so that a slice whose max is NaN keeps
    # the argmax gradient, where the keyword-free rule puts it.
    @eval function frule!!(
        ::Lifted{typeof(Core.kwcall),Nw},
        kw::Lifted{<:NamedTuple},
        ::Lifted{typeof($_fn),Nw},
        x::Lifted{<:CuFloatArray,Nw},
    ) where {Nw}
        pkw = primal(kw)
        px, x_partials = arrayify(x)
        dims = get(pkw, :dims, :)
        if _empty_minmax_slice(px, dims)
            y_e = $_fn(px; pkw...)
            lanes = ntuple(
                k -> _kw_init_jvp(zero(y_e), _kw_init_tangent(tangent(kw, k)), true),
                Val(Nw),
            )
            return Lifted{typeof(y_e),Nw}(y_e, _wrap_v_lanes(y_e, lanes))
        end
        m, ind = $_find(px; dims=dims)
        if _minmax_kw_is_plain(pkw)
            lanes = ntuple(
                k -> sum(
                    x_partials[k] .* (CartesianIndices(px) .== _winner(ind)); dims=dims
                ),
                Val(Nw),
            )
            return Lifted{typeof(m),Nw}(m, _wrap_v_lanes(m, lanes))
        end
        y = $_fn(px; pkw...)
        won = .!broadcast($_beat, m, y)
        lanes = ntuple(Val(Nw)) do k
            dy = sum(x_partials[k] .* (CartesianIndices(px) .== _winner(ind)); dims=dims)
            _kw_init_jvp(eltype(y).(dy) .* won, _kw_init_tangent(tangent(kw, k)), .!won)
        end
        return Lifted{typeof(y),Nw}(y, _wrap_v_lanes(y, lanes))
    end
    @eval function rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        kw::CoDual{<:NamedTuple},
        ::CoDual{typeof($_fn)},
        x::CoDual{<:CuFloatArray},
    )
        pkw = primal(kw)
        kw_rdata = zero_rdata(pkw)
        px, dx = arrayify(x)
        dims = get(pkw, :dims, :)
        if _empty_minmax_slice(px, dims)
            y_e = $_fn(px; pkw...)
            if dims isa Colon
                function minmax_empty_scalar_pb!!(dy)
                    dkw = _kw_init_rdata(kw_rdata, dy, true)
                    return NoRData(), dkw, NoRData(), NoRData()
                end
                return CoDual(y_e, NoFData()), minmax_empty_scalar_pb!!
            end
            dy_e = zero(y_e)
            function minmax_empty_array_pb!!(::NoRData)
                dkw = _kw_init_rdata(kw_rdata, dy_e, true)
                return NoRData(), dkw, NoRData(), NoRData()
            end
            return CoDual(y_e, dy_e), minmax_empty_array_pb!!
        end
        m, ind = $_find(px; dims=dims)
        y, won = if _minmax_kw_is_plain(pkw)
            m, true
        else
            y_kw = $_fn(px; pkw...)
            y_kw, .!broadcast($_beat, m, y_kw)
        end
        if dims isa Colon
            function minmax_kw_scalar_pb!!(dy)
                dx .+= dy .* (CartesianIndices(px) .== _winner(ind)) .* won
                dkw = _kw_init_rdata(kw_rdata, dy, .!won)
                return NoRData(), dkw, NoRData(), NoRData()
            end
            return CoDual(y, NoFData()), minmax_kw_scalar_pb!!
        end
        dy_out = zero(y)
        function minmax_kw_array_pb!!(::NoRData)
            dx .+= dy_out .* (CartesianIndices(px) .== _winner(ind)) .* won
            dkw = _kw_init_rdata(kw_rdata, dy_out, .!won)
            return NoRData(), dkw, NoRData(), NoRData()
        end
        return CoDual(y, dy_out), minmax_kw_array_pb!!
    end
end

# maximum(f, x) / minimum(f, x) are separate methods that escape the claims
# above. A real rule needs sum(f, x)'s NDual machinery plus winner selection.
for _fn in (:maximum, :minimum)
    _msg = "Mooncake: $_fn(f, x) on CuArray is not yet differentiable. " * _UNIMPL_MSG
    @eval @is_primitive(MinimalCtx, Tuple{typeof($_fn),Any,CuArray})
    @eval @is_primitive(
        MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof($_fn),Any,CuArray}
    )
    @eval frule!!(::Lifted{typeof($_fn),Nw}, ::Lifted, ::Lifted{<:CuArray}) where {Nw} = _throw_gpu_argument_error(
        $_msg
    )
    @eval rrule!!(::CoDual{typeof($_fn)}, ::CoDual, ::CoDual{<:CuArray}) = _throw_gpu_argument_error(
        $_msg
    )
    @eval frule!!(::Lifted{typeof(Core.kwcall),Nw}, ::Lifted{<:NamedTuple}, ::Lifted{typeof($_fn),Nw}, ::Lifted, ::Lifted{<:CuArray}) where {Nw} = _throw_gpu_argument_error(
        $_msg
    )
    @eval rrule!!(::CoDual{typeof(Core.kwcall)}, ::CoDual{<:NamedTuple}, ::CoDual{typeof($_fn)}, ::CoDual, ::CoDual{<:CuArray}) = _throw_gpu_argument_error(
        $_msg
    )
end

# Rules for `prod(x)` on GPU arrays, bare and `dims`.
#
# ∂prod/∂xᵢ is the product of xᵢ's slice excluding xᵢ.  Reading that off as y/xᵢ divides by
# zero precisely where the answer is not zero, so it is built from two reductions instead: a
# nonzero xᵢ contributes only when its slice holds no zero at all, and a zero xᵢ contributes
# the product of the rest when it is its slice's only zero.  Two zeros in a slice leave every
# derivative in it zero.  `prod` is a polynomial, so none of this is a non-differentiable
# point — the exclusion is what the division could not express.
#
# Both quantities come from the same reduction the primal ran, over `px` with its zeros
# replaced by ones, so a folded `init` scales the derivative exactly as it scaled the value.
function _prod_exclusive(px, dims, kw)
    T = eltype(px)
    nonzero = ifelse.(iszero.(px), one(T), px)
    pnz = prod(nonzero; kw...)
    nzeros = sum(iszero.(px); dims=dims)
    contributes = ifelse.(iszero.(px), nzeros .== 1, iszero.(nzeros))
    # `pnz` follows `init`'s type when there is one, since GPUArrays takes the output eltype
    # from it, so the zero has to come from the quotient rather than from either side: `T`
    # would join two eltypes into an `AbstractFloat` the kernel cannot hold, and `eltype(pnz)`
    # would do the same the other way round when `init` is the narrower of the two.
    q = pnz ./ nonzero
    return ifelse.(contributes, q, zero(eltype(q)))
end

@is_primitive(MinimalCtx, Tuple{typeof(prod),CuMaybeComplexArray})
function frule!!(
    ::Lifted{typeof(prod),Nw}, x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray}
) where {Nw}
    px = primal(x)
    y = prod(px)
    x_partials = Nfwd._lane_views(tangent(x))
    # ∂prod/∂xᵢ does not depend on the lane, so the exclusive product is formed once and each
    # lane's JVP is its inner product with it.
    excl = _prod_exclusive(px, :, (;))
    if isempty(px)
        # prod over no elements has derivative 0, and gemv_batched! quick-returns on a
        # zero-length operand before its `beta=0` overwrite, leaving the outputs uninitialised.
        dy_lanes = ntuple(_ -> zero(y), Val(Nw))
    elseif Nw == 1
        dy_lanes = (sum(x_partials[1] .* excl),)
    else
        # Per lane dyₖ = exclᵀ·xpₖ with `excl` lane-invariant; batch the N reductions into one
        # gemv_batched! ('T' avoids conjugating complex `excl`).
        T = eltype(px)
        wcol = reshape(excl, :, 1)
        xvs = [reshape(xp, :) for xp in x_partials]
        outs = [similar(px, T, 1) for _ in 1:Nw]
        cuBLAS.gemv_batched!('T', one(T), fill(wcol, Nw), xvs, zero(T), outs)
        host = Array(reduce(vcat, outs))
        dy_lanes = ntuple(k -> host[k], Val(Nw))
    end
    return Lifted{typeof(y),Nw}(y, _wrap_scalar_v_lanes(y, dy_lanes))
end
function rrule!!(::CoDual{typeof(prod)}, x::CoDual{<:CuMaybeComplexArray})
    px, dx = arrayify(x)
    excl = _prod_exclusive(px, :, (;))
    function prod_pb!!(dy)
        # Wirtinger chain rule for holomorphic prod: Δxᵢ = Δy · conj(∂prod/∂xᵢ).
        # For real inputs conj is a no-op.
        dx .+= dy .* conj.(excl)
        return NoRData(), NoRData()
    end
    return zero_fcodual(prod(px)), prod_pb!!
end

# The `dims` spelling: the same exclusive product per reduced slice, which broadcasts because
# the reduced dimensions stay singleton.
@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(prod),CuMaybeComplexArray}
)
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple},
    ::Lifted{typeof(prod),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw},
) where {Nw}
    pkw = primal(kw)
    _check_reduction_identity(prod, pkw)
    for k in 1:Nw
        _check_reduction_init(tangent(kw, k))
    end
    px, x_partials = arrayify(x)
    dims = get(pkw, :dims, :)
    y = prod(px; pkw...)
    excl = _prod_exclusive(px, dims, pkw)
    # `init` fixes the output eltype, and the tangent has to follow it.  Converting after
    # the reduction rather than seeding it: a widening seed over a broadcast fails to
    # compile a kernel.
    lanes = ntuple(k -> eltype(y).(sum(x_partials[k] .* excl; dims=dims)), Val(Nw))
    return Lifted{typeof(y),Nw}(y, _wrap_v_lanes(y, lanes))
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(prod)},
    x::CoDual{<:CuMaybeComplexArray},
)
    pkw = primal(kw)
    _check_reduction_identity(prod, pkw)
    kw_rdata = zero_rdata(pkw)
    px, dx = arrayify(x)
    dims = get(pkw, :dims, :)
    y = prod(px; pkw...)
    excl = conj.(_prod_exclusive(px, dims, pkw))
    if dims isa Colon
        function prod_kw_scalar_pb!!(dy)
            dx .+= dy .* excl
            return NoRData(), kw_rdata, NoRData(), NoRData()
        end
        return CoDual(y, NoFData()), prod_kw_scalar_pb!!
    end
    dy_out = zero(y)
    function prod_kw_array_pb!!(::NoRData)
        dx .+= dy_out .* excl
        return NoRData(), kw_rdata, NoRData(), NoRData()
    end
    return CoDual(y, dy_out), prod_kw_array_pb!!
end

# Rules for `cumsum(x)` on GPU arrays.
#
# y[k] = Σᵢ₌₁ᵏ x[i],  so ∂y[k]/∂x[i] = 1 if i≤k else 0
# frule:    dy = cumsum(dx)
# pullback: dx[i] += Σₖ≥ᵢ dy[k]  =  reverse(cumsum(reverse(dy)))
#
# Supports the optional `dims` keyword (passed through to CUDA's cumsum).
@is_primitive(MinimalCtx, Tuple{typeof(cumsum),CuMaybeComplexArray})
function frule!!(
    ::Lifted{typeof(cumsum),Nw}, x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray}; kw...
) where {Nw}
    px = primal(x)
    y = cumsum(px; kw...)
    x_partials = Nfwd._lane_views(tangent(x))
    y_partials = ntuple(k -> cumsum(x_partials[k]; kw...), Val(Nw))
    Y = typeof(y)
    Element = eltype(y)
    return Lifted{Y,Nw}(y, NDualArray{Element,Nw,ndims(y),Y}(y, y_partials))
end
function rrule!!(::CoDual{typeof(cumsum)}, x::CoDual{<:CuMaybeComplexArray}; kw...)
    px, dx = arrayify(x)
    y = cumsum(px; kw...)
    dy_out = zero(y)
    d = get(kw, :dims, 1)
    function cumsum_pb!!(::NoRData)
        dx .+= _scan_pullback(dy_out, d)
        return NoRData(), NoRData()
    end
    return CoDual(y, dy_out), cumsum_pb!!
end

# Rules for `cumprod(x)` on GPU arrays.
#
# y[k] = Πᵢ₌₁ᵏ x[i], so ∂y[k]/∂x[i] for i ≤ k is that prefix product with x[i] excluded.
# Excluding it by dividing y[k] by x[i] fails at a zero, exactly as it did for `prod`, and the
# same accounting fixes it — per prefix rather than per slice: a nonzero x[i] contributes only
# to prefixes that hold no zero at all, a zero x[i] only to prefixes where it is the only
# zero, and a prefix holding two zeros has no derivative anywhere. `nzeros` and `pnz` are the
# cumulative forms of the two reductions `_prod_exclusive` uses; `denom` is x with its zeros
# replaced by ones, so nothing ever divides by zero.
#
#   dy[k] = pnz[k] · (nzeros[k] == 0 ? cumsum(dx ./ denom)[k] :
#                     nzeros[k] == 1 ? cumsum(dx at the zeros)[k] : 0)
#   dx[i] += (Σ over k ≥ i of dy[k]·pnz[k], masked to the prefixes i contributes to) / denom[i]
function _cumprod_pieces(px, d)
    T = eltype(px)
    zero_at = iszero.(px)
    denom = ifelse.(zero_at, one(T), px)
    return zero_at, cumsum(zero_at; dims=d), cumprod(denom; dims=d), denom
end

@is_primitive(MinimalCtx, Tuple{typeof(cumprod),CuMaybeComplexArray})
function frule!!(
    ::Lifted{typeof(cumprod),Nw}, x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray}; kw...
) where {Nw}
    px = primal(x)
    d = get(kw, :dims, 1)
    y = cumprod(px; kw...)
    x_partials = Nfwd._lane_views(tangent(x))
    # The four pieces depend on the primal alone, so they are formed once for all lanes.
    zero_at, nzeros, pnz, denom = _cumprod_pieces(px, d)
    clean, only_zero = iszero.(nzeros), nzeros .== 1
    y_partials = ntuple(Val(Nw)) do k
        dx = x_partials[k]
        from_nonzeros = cumsum(dx ./ denom; dims=d)
        from_the_zero = cumsum(ifelse.(zero_at, dx, zero(eltype(px))); dims=d)
        contribution = ifelse.(
            clean, from_nonzeros, ifelse.(only_zero, from_the_zero, zero(eltype(px)))
        )
        return pnz .* contribution
    end
    Y = typeof(y)
    Element = eltype(y)
    return Lifted{Y,Nw}(y, NDualArray{Element,Nw,ndims(y),Y}(y, y_partials))
end
function rrule!!(::CoDual{typeof(cumprod)}, x::CoDual{<:CuMaybeComplexArray}; kw...)
    px, dx = arrayify(x)
    y = cumprod(px; kw...)
    dy_out = zero(y)
    d = get(kw, :dims, 1)
    # Pre-computed once at rule construction time: reused on every pullback call.
    zero_at, nzeros, pnz, denom = _cumprod_pieces(px, d)
    clean, only_zero = iszero.(nzeros), nzeros .== 1
    cpnz, cdenom = conj.(pnz), conj.(denom)
    function cumprod_pb!!(::NoRData)
        # Wirtinger chain rule: Δxᵢ = Σₖ≥ᵢ Δyₖ · conj(∂yₖ/∂xᵢ).  For real inputs conj is a
        # no-op.  A nonzero xᵢ reads the prefixes with no zero, a zero xᵢ those where it is
        # the only one, so the two masked reverse scans cover every case.
        weighted = dy_out .* cpnz
        dx .+=
            ifelse.(
                zero_at,
                _scan_pullback(weighted .* only_zero, d),
                _scan_pullback(weighted .* clean, d),
            ) ./ cdenom
        return NoRData(), NoRData()
    end
    return CoDual(y, dy_out), cumprod_pb!!
end

# Mooncake lowers every keyword call to Core.kwcall, which the positional claims above do
# not cover, so `cumsum(x; dims=1)` decomposed onto the untraceable kernel instead.  Base
# requires `dims` for an array of more than one dimension, so those rules reached only
# vectors.  The keyword-free bodies already take the keywords, hence the forwarding.
for _fn in (:cumsum, :cumprod)
    @eval @is_primitive(
        MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof($_fn),CuMaybeComplexArray}
    )
    @eval function frule!!(
        ::Lifted{typeof(Core.kwcall),Nw},
        kw::Lifted{<:NamedTuple},
        f::Lifted{typeof($_fn),Nw},
        x::Lifted{<:CuMaybeComplexArray},
    ) where {Nw}
        return frule!!(f, x; primal(kw)...)
    end
    @eval function rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        kw::CoDual{<:NamedTuple},
        f::CoDual{typeof($_fn)},
        x::CoDual{<:CuMaybeComplexArray},
    )
        kw_rdata = zero_rdata(primal(kw))
        y, pb = rrule!!(f, x; primal(kw)...)
        cum_kw_pb!!(dy) = (NoRData(), kw_rdata, pb(dy)...)
        return y, cum_kw_pb!!
    end
end

# CUDA scans a `dims`-less `accumulate` in linear order — `reshape(accumulate(op, A[:]),
# size(A))` — rather than column by column, so an N-d array's adjoint is the vectorised one.
# cumsum needs no such split: Base rejects it without `dims` past one dimension.
_scan_jvp(dx, ::Nothing) = reshape(cumsum(vec(dx)), size(dx))
_scan_jvp(dx, d) = cumsum(dx; dims=d)
# `reverse` sizes its grid from the element count and refuses an empty one, while `cumsum`
# over an empty array is fine — so an empty input reached the pullback and only the pullback.
# Its adjoint is empty whatever the scan was, including a 0x3 reduced along its non-empty
# dimension.
function _scan_pullback(dy, ::Nothing)
    isempty(dy) ? dy : reshape(reverse(cumsum(reverse(vec(dy)))), size(dy))
end
function _scan_pullback(dy, d)
    # Scanning along a dimension the array does not have is the identity — the primal and the
    # JVP both accept it — so the adjoint is the identity too.  `reverse` would refuse that
    # dimension, which is what used to make the pullback the only mode to reject it.
    (isempty(dy) || d > ndims(dy)) && return dy
    return reverse(cumsum(reverse(dy; dims=d); dims=d); dims=d)
end
# Same boundary seen from `init`: on that identity path CUDA's scan copies the input and
# returns before applying `init`, so the primal does not depend on it and neither may the
# derivative.
_scan_applies_init(px, ::Nothing) = true
_scan_applies_init(px, d) = d <= ndims(px)

# Rules for `accumulate(+, x)` — identical to cumsum but via the accumulate interface.
# Other operators are not supported and throw an informative error (catch-all below).
@is_primitive(MinimalCtx, Tuple{typeof(accumulate),typeof(+),CuMaybeComplexArray})
function frule!!(
    ::Lifted{typeof(accumulate),Nw},
    ::Lifted{typeof(+),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray};
    kw...,
) where {Nw}
    px = primal(x)
    y = accumulate(+, px; kw...)
    d = get(kw, :dims, nothing)
    x_partials = Nfwd._lane_views(tangent(x))
    y_partials = ntuple(k -> _scan_jvp(x_partials[k], d), Val(Nw))
    Y = typeof(y)
    Element = eltype(y)
    return Lifted{Y,Nw}(y, NDualArray{Element,Nw,ndims(y),Y}(y, y_partials))
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
    d = get(kw, :dims, nothing)
    function accumulate_plus_pb!!(::NoRData)
        dx .+= _scan_pullback(dy_out, d)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy_out), accumulate_plus_pb!!
end
@is_primitive(MinimalCtx, Tuple{typeof(accumulate),Any,CuArray})
function frule!!(
    ::Lifted{typeof(accumulate),Nw}, op::Lifted, x::Lifted{<:CuArray}; kwargs...
) where {Nw}
    return _throw_gpu_argument_error(
        "Mooncake: accumulate on CuArray supports only op=+ over a float or complex " *
        "array; got op=$(primal(op)) over $(typeof(primal(x))). " *
        _UNIMPL_MSG,
    )
end
function rrule!!(::CoDual{typeof(accumulate)}, op::CoDual, x::CoDual{<:CuArray}; kwargs...)
    return _throw_gpu_argument_error(
        "Mooncake: accumulate on CuArray supports only op=+ over a float or complex " *
        "array; got op=$(primal(op)) over $(typeof(primal(x))). " *
        _UNIMPL_MSG,
    )
end
# The Core.kwcall spellings of both, `accumulate(+, x; dims=1)` and its error arm.  These do
# not forward to the rules above because `accumulate` takes an `init` that cumsum does not:
# it adds to every output element, so it collects the whole cotangent, and it widens the
# output eltype, which the tangent has to follow.
@is_primitive(
    MinimalCtx,
    Tuple{typeof(Core.kwcall),NamedTuple,typeof(accumulate),typeof(+),CuMaybeComplexArray},
)
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple},
    ::Lifted{typeof(accumulate),Nw},
    ::Lifted{typeof(+),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw},
) where {Nw}
    pkw = primal(kw)
    px, x_partials = arrayify(x)
    d = get(pkw, :dims, nothing)
    y = accumulate(+, px; pkw...)
    applies = _scan_applies_init(px, d)
    y_partials = ntuple(Val(Nw)) do k
        dy = eltype(y).(_scan_jvp(x_partials[k], d))
        return _kw_init_jvp(dy, _kw_init_tangent(tangent(kw, k)), applies)
    end
    Y = typeof(y)
    return Lifted{Y,Nw}(y, NDualArray{eltype(Y),Nw,ndims(y),Y}(y, y_partials))
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(accumulate)},
    ::CoDual{typeof(+)},
    x::CoDual{<:CuMaybeComplexArray},
)
    pkw = primal(kw)
    kw_rdata = zero_rdata(pkw)
    px, dx = arrayify(x)
    y = accumulate(+, px; pkw...)
    dy_out = zero(y)
    d = get(pkw, :dims, nothing)
    function accumulate_plus_kw_pb!!(::NoRData)
        dx .+= _scan_pullback(dy_out, d)
        dkw = _kw_init_rdata(kw_rdata, dy_out, _scan_applies_init(px, d))
        return NoRData(), dkw, NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy_out), accumulate_plus_kw_pb!!
end
@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(accumulate),Any,CuArray}
)
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    ::Lifted{<:NamedTuple},
    f::Lifted{typeof(accumulate),Nw},
    op::Lifted,
    x::Lifted{<:CuArray},
) where {Nw}
    return frule!!(f, op, x)
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    ::CoDual{<:NamedTuple},
    f::CoDual{typeof(accumulate)},
    op::CoDual,
    x::CoDual{<:CuArray},
)
    return rrule!!(f, op, x)
end

# Rule for `sum(x)` — widened from CuFloatArray to also cover complex CuArrays.
# See also `src/rules/performance_patches`.
@is_primitive(DefaultCtx, Tuple{typeof(sum),CuMaybeComplexArray})
function frule!!(
    ::Lifted{typeof(sum),Nw}, x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray}
) where {Nw}
    px = primal(x)
    y = sum(px)
    x_partials = Nfwd._lane_views(tangent(x))
    # `isempty` routes past the batched path: gemv_batched! with a zero-length operand quick-returns
    # before its `beta=0` overwrite, leaving the output buffers uninitialised — so sum over an empty
    # array (derivative 0) must use the per-lane reduction, which is 0 for every lane.
    if Nw == 1 || isempty(px)
        dy_lanes = ntuple(k -> sum(x_partials[k]), Val(Nw))
    else
        # Batch the N per-lane reductions into one gemv_batched!: sum(xₖ) = onesᵀ·xₖ ('T' transposes
        # the n×1 ones column into the 1×n contracting row; ones is real and gemv never conjugates its
        # vector operand, so the partials pass through unchanged), then one concatenated readback.
        T = eltype(px)
        onev = reshape(fill!(similar(px, T, length(px)), one(T)), :, 1)
        xvs = [reshape(xp, :) for xp in x_partials]
        outs = [similar(px, T, 1) for _ in 1:Nw]
        cuBLAS.gemv_batched!('T', one(T), fill(onev, Nw), xvs, zero(T), outs)
        host = Array(reduce(vcat, outs))
        dy_lanes = ntuple(k -> host[k], Val(Nw))
    end
    return Lifted{typeof(y),Nw}(y, _wrap_scalar_v_lanes(y, dy_lanes))
end
function rrule!!(::CoDual{typeof(sum)}, x::CoDual{<:CuMaybeComplexArray})
    _, dx = arrayify(x)
    function sum_pb!!(dz)
        dx .+= dz
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(primal(x))), sum_pb!!
end

# Reductions and orderings over an index or mask array have no derivative: the array has no
# tangent, and the result is an Integer or another such array.  Without a rule they
# decompose onto one of GPUArrays' untraceable paths, which one depending on spelling and
# eltype.
#
# sum and prod can use @zero_derivative; for the other five it generates
# `CoDual{<:typeof(f)}`, equal in extent to their catch-all error rules' `CoDual{typeof(f)}`
# but not comparable with it, so the two would be ambiguous.  Spelling the signature the
# same way lets the narrower array type decide, leaving those rules to report float arrays.
@zero_derivative MinimalCtx Tuple{typeof(sum),CuNonDiffArray}
@zero_derivative MinimalCtx Tuple{typeof(prod),CuNonDiffArray}
for _fn in (:maximum, :minimum, :diff, :sort)
    @eval @is_primitive(MinimalCtx, Tuple{typeof($_fn),CuNonDiffArray})
    @eval frule!!(f::Lifted{typeof($_fn),Nw}, x::Lifted{<:CuNonDiffArray}) where {Nw} = zero_derivative(
        f, x
    )
    @eval rrule!!(f::CoDual{typeof($_fn)}, x::CoDual{<:CuNonDiffArray}) = zero_adjoint(f, x)
end

# The scans over such an array are zero-derivative for the same reason, whatever the operator
# accumulates: the array carries no derivative, so nothing built from it does either.  Without
# these, `x[cumsum(idx)]` decomposed onto the untraceable scan kernel, and `cumprod(idx)` —
# which lowers to `accumulate(*, …)` — reported the accumulate rule's operator restriction as
# though the operator were the problem.  cumsum and cumprod have no catch-all error rules to
# be ambiguous with, so they can use the macro; accumulate does, hence the manual spelling.
@zero_derivative MinimalCtx Tuple{typeof(cumsum),CuNonDiffArray}
@zero_derivative MinimalCtx Tuple{typeof(cumprod),CuNonDiffArray}
@zero_derivative MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(cumsum),CuNonDiffArray
}
@zero_derivative MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(cumprod),CuNonDiffArray
}
@is_primitive(MinimalCtx, Tuple{typeof(accumulate),Any,CuNonDiffArray})
function frule!!(
    f::Lifted{typeof(accumulate),Nw}, op::Lifted, x::Lifted{<:CuNonDiffArray}
) where {Nw}
    zero_derivative(f, op, x)
end
function rrule!!(f::CoDual{typeof(accumulate)}, op::CoDual, x::CoDual{<:CuNonDiffArray})
    zero_adjoint(f, op, x)
end
@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(accumulate),Any,CuNonDiffArray}
)
# The array carries no derivative, but a float `init` widens the output to a float array and
# adds to every element of it, so it collects the whole cotangent — the same accounting the
# differentiable accumulate rules do, including the identity path where the scan dimension
# exceeds the array's and `init` never reaches the result.
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple},
    ::Lifted{typeof(accumulate),Nw},
    op::Lifted,
    x::Lifted{<:CuNonDiffArray},
) where {Nw}
    pkw = primal(kw)
    px = primal(x)
    y = accumulate(primal(op), px; pkw...)
    # Without a float `init` the output is another index array carrying no derivative.
    dual_type(Val(Nw), typeof(y)) === NoDual && return Lifted{typeof(y),Nw}(y, NoDual())
    applies = _scan_applies_init(px, get(pkw, :dims, nothing))
    Y = typeof(y)
    y_partials = ntuple(
        k -> _kw_init_jvp(zero(y), _kw_init_tangent(tangent(kw, k)), applies), Val(Nw)
    )
    return Lifted{Y,Nw}(y, NDualArray{eltype(Y),Nw,ndims(y),Y}(y, y_partials))
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(accumulate)},
    op::CoDual,
    x::CoDual{<:CuNonDiffArray},
)
    pkw = primal(kw)
    px = primal(x)
    kw_rdata = zero_rdata(pkw)
    y = accumulate(primal(op), px; pkw...)
    applies = _scan_applies_init(px, get(pkw, :dims, nothing))
    out = zero_fcodual(y)
    dy_out = tangent(out)
    function accumulate_nodiff_kw_pb!!(::Any)
        dkw = _kw_init_rdata(kw_rdata, dy_out, applies)
        return NoRData(), dkw, NoRData(), NoRData(), NoRData()
    end
    return out, accumulate_nodiff_kw_pb!!
end

# `count` returns an Integer for every array, float included, so it never has a derivative.
# It reaches GPUArrays through `mapreduce(pred, add_sum, A; init=0)`, a Core.kwcall the
# positional mapreduce catch-all does not claim, so claim `count` itself.
@zero_derivative MinimalCtx Tuple{typeof(count),CuArray}
@zero_derivative MinimalCtx Tuple{typeof(count),Any,CuArray}

# The positional spellings above take no `init` and are zero-derivative outright.  The keyword
# ones accept one, and GPUArrays folds it as it does `sum`'s: `count(>(0.0), x; init=1.0)`
# returns 98.0 where the count is 3.  Same identity check, so the wrong value is refused
# rather than its derivative silently zeroed.
for _n_args in (0, 1)
    _pred_d = _n_args == 1 ? (:(pred::Lifted),) : ()
    _pred_c = _n_args == 1 ? (:(pred::CoDual),) : ()
    _pred_v = _n_args == 1 ? (:(primal(pred)),) : ()
    _slot = _n_args == 1 ? (:Any,) : ()
    @eval @is_primitive(
        MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(count),$(_slot...),CuArray}
    )
    @eval function frule!!(
        ::Lifted{typeof(Core.kwcall),Nw},
        kw::Lifted{<:NamedTuple},
        ::Lifted{typeof(count),Nw},
        $(_pred_d...),
        x::Lifted{<:CuArray},
    ) where {Nw}
        pkw = primal(kw)
        _check_reduction_identity(count, pkw)
        for k in 1:Nw
            _check_reduction_init(tangent(kw, k))
        end
        y = count($(_pred_v...), primal(x); pkw...)
        return zero_lifted(Val(Nw), y)
    end
    @eval function rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        kw::CoDual{<:NamedTuple},
        ::CoDual{typeof(count)},
        $(_pred_c...),
        x::CoDual{<:CuArray},
    )
        pkw = primal(kw)
        _check_reduction_identity(count, pkw)
        # The keyword slot owes `zero_rdata(pkw)`, not `NoRData()`: a float `init` makes the
        # count a float, and its rdata is then a NamedTuple the caller increments into.
        kw_rdata = zero_rdata(pkw)
        y = count($(_pred_v...), primal(x); pkw...)
        function count_kw_pb!!(::Any)
            return NoRData(), kw_rdata, $(fill(:(NoRData()), _n_args + 2)...)
        end
        return zero_fcodual(y), count_kw_pb!!
    end
end

# GPUArrays folds `init` into a backend-defined number of partial reductions, so for the
# non-idempotent ops it changes the value itself: sum(CuArray([1, 2, 3]); init=1.0) is 101.0,
# not 7.0. Neither that value nor a derivative through it is defined, so both modes refuse a
# non-identity `init` — a check on the value, which reverse mode can see too, unlike the
# tangent check below. max and min fold exactly, so a non-identity `init` is the point of
# passing one to them and they need no guard; diff/sort/sortperm take no `init` at all.
_reduction_identity(::typeof(sum)) = 0
_reduction_identity(::typeof(count)) = 0
_reduction_identity(::typeof(prod)) = 1
_reduction_identity(_) = nothing
function _check_reduction_identity(f, pkw)
    identity = _reduction_identity(f)
    (identity === nothing || !haskey(pkw, :init) || pkw.init == identity) && return nothing
    return _throw_gpu_argument_error(
        "Mooncake: $f over CuArray was given init=$(pkw.init), which is not its identity " *
        "($identity). GPUArrays folds `init` into a backend-defined number of partial " *
        "reductions, so the result itself is undefined — sum(CuArray([1, 2, 3]); init=1.0) " *
        "returns 101.0, not 7.0. Pass the identity, or add `init` to the array instead. " *
        _UNIMPL_MSG,
    )
end

function _check_reduction_init(dkw)
    dinit = dkw isa NamedTuple ? get(dkw, :init, NoTangent()) : NoTangent()
    (dinit isa NoTangent || iszero(dinit)) && return nothing
    return _throw_gpu_argument_error(
        "Mooncake: keyword reductions over CuArray treat `init` as a constant, but it " *
        "received a nonzero tangent. Differentiating with respect to `init` is not " *
        "supported. " *
        _UNIMPL_MSG,
    )
end

for _fn in (:sum, :prod, :diff, :sort)
    @eval @is_primitive(
        MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof($_fn),CuNonDiffArray}
    )
    @eval function frule!!(
        ::Lifted{typeof(Core.kwcall),Nw},
        kw::Lifted{<:NamedTuple},
        ::Lifted{typeof($_fn),Nw},
        x::Lifted{<:CuNonDiffArray},
    ) where {Nw}
        _check_reduction_identity($_fn, primal(kw))
        for k in 1:Nw
            _check_reduction_init(tangent(kw, k))
        end
        y = $_fn(primal(x); primal(kw)...)
        return zero_lifted(Val(Nw), y)
    end
    @eval function rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        kw::CoDual{<:NamedTuple},
        ::CoDual{typeof($_fn)},
        x::CoDual{<:CuNonDiffArray},
    )
        pkw = primal(kw)
        _check_reduction_identity($_fn, pkw)
        kw_rdata = zero_rdata(pkw)
        y = $_fn(primal(x); pkw...)
        # `::Any`: the output is an Integer without `init` and a float with one, so the
        # incoming rdata is NoRData or a number depending on the call.
        nodiff_reduction_kw_pb!!(::Any) = (NoRData(), kw_rdata, NoRData(), NoRData())
        return zero_fcodual(y), nodiff_reduction_kw_pb!!
    end
end

# maximum/minimum over an index or mask array: the array carries no derivative, but a float
# `init` competes with its elements and takes the whole derivative of every slice it beats.
# The array's own max, which decides that, needs a second reduction, so plain `dims` calls
# keep the zero-derivative path.
for (_fn, _beat) in ((:maximum, :<), (:minimum, :>))
    @eval @is_primitive(
        MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof($_fn),CuNonDiffArray}
    )
    @eval function frule!!(
        ::Lifted{typeof(Core.kwcall),Nw},
        kw::Lifted{<:NamedTuple},
        ::Lifted{typeof($_fn),Nw},
        x::Lifted{<:CuNonDiffArray},
    ) where {Nw}
        pkw = primal(kw)
        px = primal(x)
        y = $_fn(px; pkw...)
        # Without a float `init` the result is another index value carrying no derivative.
        dual_type(Val(Nw), typeof(y)) === NoDual && return Lifted{typeof(y),Nw}(y, NoDual())
        won = if _minmax_kw_is_plain(pkw)
            true
        else
            .!broadcast($_beat, $_fn(px; dims=get(pkw, :dims, :)), y)
        end
        lanes = ntuple(
            k -> _kw_init_jvp(zero(y), _kw_init_tangent(tangent(kw, k)), .!won), Val(Nw)
        )
        return Lifted{typeof(y),Nw}(y, _wrap_v_lanes(y, lanes))
    end
    @eval function rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        kw::CoDual{<:NamedTuple},
        ::CoDual{typeof($_fn)},
        x::CoDual{<:CuNonDiffArray},
    )
        pkw = primal(kw)
        kw_rdata = zero_rdata(pkw)
        px = primal(x)
        out = zero_fcodual($_fn(px; pkw...))
        won = if _minmax_kw_is_plain(pkw)
            true
        else
            .!broadcast($_beat, $_fn(px; dims=get(pkw, :dims, :)), primal(out))
        end
        # `::Any`: a `dims` reduction carries its cotangent in the output's fdata, a scalar
        # one in the incoming rdata.
        function nodiff_minmax_kw_pb!!(dy::Any)
            dkw = _kw_init_rdata(kw_rdata, dy isa NoRData ? tangent(out) : dy, .!won)
            return NoRData(), dkw, NoRData(), NoRData()
        end
        return out, nodiff_minmax_kw_pb!!
    end
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
    ::Lifted{typeof(unsafe_copyto!),Nw},
    dest::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    doffs::Lifted{<:Integer},
    # Covers both a GPU src and a host-Array src (cross-device copy) — the body dispatches the inner
    # `unsafe_copyto!` on the runtime element/device types, so one method serves both @is_primitives.
    src::Lifted{<:Union{CuMaybeComplexArray,Array},Nw,<:NDualArray},
    soffs::Lifted{<:Integer},
    n::Lifted{<:Integer},
) where {Nw}
    doffs_v, soffs_v, n_v = primal(doffs), primal(soffs), primal(n)
    unsafe_copyto!(primal(dest), doffs_v, primal(src), soffs_v, n_v)
    dest_partials = Nfwd._lane_views(tangent(dest))
    src_partials = Nfwd._lane_views(tangent(src))
    # A device src's lanes are contiguous (lane-major block); a host src's lanes are strided
    # (element-major block) — materialize those so the cross-device copy sees contiguous memory.
    src_is_device = primal(src) isa CuArray
    @inbounds for lane in 1:Nw
        src_lane = src_is_device ? src_partials[lane] : collect(src_partials[lane])
        unsafe_copyto!(dest_partials[lane], doffs_v, src_lane, soffs_v, n_v)
    end
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
# (host-Array-src forward mode is served by the `unsafe_copyto!` frule above.)
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

# `unsafe_copyto!` and `fill!` into a device array whose elements carry no derivative
# information (index arrays, masks).  The rules above all require a tangent array, so
# these spellings had no rule at all and died in the same try/catch and fill kernel.
# There is no gradient to propagate, but the pullback must still undo the mutation, or
# re-running the rule would see a different array.
#
# CuNonDiffArray is disjoint from CuMaybeComplexArray and CuMaybeWrappedArray, so these
# claims are unambiguous with the rules above and cover exactly what they can serve.
#
# Note the asymmetry in `src`: a CPU Array{Int} has fdata Vector{NoTangent}, and only the
# CuArray side collapses to NoTangent, so `src`'s tangent is left unconstrained.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(unsafe_copyto!),
        <:CuNonDiffArray,
        Integer,
        <:Union{Array,CuArray},
        Integer,
        Integer,
    },
)
function frule!!(
    ::Lifted{typeof(unsafe_copyto!),Nw},
    dest::Lifted{<:CuNonDiffArray,Nw,NoDual},
    doffs::Lifted{<:Integer,Nw,NoDual},
    src::Lifted{<:Union{Array,CuArray}},
    soffs::Lifted{<:Integer,Nw,NoDual},
    n::Lifted{<:Integer,Nw,NoDual},
) where {Nw}
    unsafe_copyto!(primal(dest), primal(doffs), primal(src), primal(soffs), primal(n))
    return dest
end
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)},
    dest::CoDual{<:CuNonDiffArray},
    doffs::CoDual{<:Integer,NoFData},
    src::CoDual{<:Union{Array,CuArray}},
    soffs::CoDual{<:Integer,NoFData},
    n::CoDual{<:Integer,NoFData},
)
    pdest = primal(dest)
    doffs_v, n_v = primal(doffs), primal(n)
    dest_range = doffs_v:(doffs_v + n_v - 1)
    saved = copy(view(pdest, dest_range))
    unsafe_copyto!(pdest, doffs_v, primal(src), primal(soffs), n_v)
    function unsafe_copyto!_nodiff_pb!!(::NoRData)
        copyto!(view(pdest, dest_range), saved)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return dest, unsafe_copyto!_nodiff_pb!!
end

@is_primitive(MinimalCtx, Tuple{typeof(fill!),<:CuNonDiffArray,Any})
function frule!!(
    ::Lifted{typeof(fill!),Nw}, a::Lifted{<:CuNonDiffArray,Nw,NoDual}, x::Lifted
) where {Nw}
    fill!(primal(a), primal(x))
    return a
end
function rrule!!(::CoDual{typeof(fill!)}, a::CoDual{<:CuNonDiffArray}, x::CoDual)
    pa = primal(a)
    old = copy(pa)
    fill!(pa, primal(x))
    function fill!_nodiff_pb!!(::NoRData)
        copyto!(pa, old)
        # `a` has no tangent, so nothing reaches `x`: a typed zero, not NoRData, unless
        # `x` is itself non-differentiable.
        dx = if tangent_type(typeof(primal(x))) == NoTangent
            NoRData()
        else
            zero_rdata(primal(x))
        end
        return NoRData(), NoRData(), dx
    end
    return a, fill!_nodiff_pb!!
end

# unsafe_free! releases GPU memory early (normally handled by GC finalizer).
# It is a pure side-effect with no mathematical output — gradient is zero.
#
# Only forward mode may free anything: there the tangent travels with the primal and dies with
# it.  Reverse mode frees neither, because the reverse sweep runs long after the forward sweep
# reaches this call and may reference both.  The fdata is the accumulator earlier pullbacks
# write into.  The primal is no safer: pullbacks close over it routinely — this extension's own
# gather keeps the index array to scatter with, and the non-differentiable `unsafe_copyto!`
# rule keeps its destination to restore — so `CuArray([1, 2, 3])` freed after use took the
# reverse pass through `view` on released memory.  Reverse therefore gives up the saving
# entirely; it is a memory hint, and doing nothing is the only sound reading of it here.
@is_primitive MinimalCtx Tuple{typeof(unsafe_free!),CuArray}
function frule!!(
    ::Lifted{typeof(unsafe_free!),Nw}, x::Lifted{<:CuArray,Nw,<:NDualArray}
) where {Nw}
    unsafe_free!(primal(x))
    # The N lanes share one backing block; free it once (freeing a per-lane view is invalid).
    unsafe_free!(getfield(tangent(x), :partials_block))
    return Lifted{Nothing,Nw}(nothing, NoDual())
end
# The claim covers index and mask arrays too, whose forward V is `NoDual`: free the primal
# only, there being no per-lane storage to release.
function frule!!(
    ::Lifted{typeof(unsafe_free!),Nw}, x::Lifted{<:CuArray,Nw,NoDual}
) where {Nw}
    unsafe_free!(primal(x))
    return Lifted{Nothing,Nw}(nothing, NoDual())
end
function rrule!!(::CoDual{typeof(unsafe_free!)}, ::CoDual{<:CuArray})
    return CoDual(nothing, NoFData()), _nopb(Val(2))
end

# Core.finalizer(f, x) registers f as a GC finalizer for x. This is a pure side-effect
# (no mathematical output) encountered inside CuArray constructors (e.g. view/derive).
# The primal registration must happen; the gradient is zero.
@is_primitive MinimalCtx Tuple{typeof(Core.finalizer),Any,Any}
function frule!!(::Lifted{typeof(Core.finalizer),Nw}, f::Lifted, x::Lifted) where {Nw}
    Core.finalizer(primal(f), primal(x))
    return Lifted{Nothing,Nw}(nothing, NoDual())
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
function frule!!(::Lifted{typeof(hasfieldcount),Nw}, T::Lifted{<:Type}) where {Nw}
    return Lifted{Bool,Nw}(hasfieldcount(primal(T)), NoDual())
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
function frule!!(
    ::Lifted{typeof(fill!),Nw}, a::Lifted{<:CuMaybeWrappedArray,Nw}, x::Lifted
) where {Nw}
    # `arrayify` handles a dense CuArray and Adjoint/Transpose/SubArray wrappers alike: `pa` is
    # the destination (filled through the wrapper) and each `a_partials[lane]` is the same wrapper
    # shape over lane `k`'s partials, so `fill!` writes the constant into exactly the region the
    # primal touches — matching the reverse rrule, which also goes through `arrayify`.
    pa, a_partials = arrayify(a)
    fill!(pa, primal(x))
    Eout = eltype(a_partials[1])
    if tangent(x) isa NoDual
        for partial in a_partials
            fill!(partial, zero(Eout))
        end
    else
        # Per-lane scalar tangent via the canonical accessor, which handles a real `NDual`
        # and a complex `Complex{NDual}` alike — the raw `.partials` field exists only on `NDual`.
        @inbounds for lane in 1:Nw
            fill!(a_partials[lane], Eout(tangent(x, lane)))
        end
    end
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
            # A real `x` filling a complex array is owed only dL/dRe, so the sum is
            # projected onto x's own field before it is narrowed to x's type — narrowing a
            # complex sum to a real would throw instead.
            rdata_type(typeof(primal(x)))(_project_cotangent(primal(x), sum(da)))
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
# `Transpose{T,<:CuArray}` has canonical V `ImmutableDual{@NamedTuple{parent::NDualArray}}`;
# sum the parent's per-lane partials.
function frule!!(
    ::Lifted{typeof(sum),Nw},
    x::Lifted{<:Transpose{T,<:CuMaybeComplexArray},Nw,<:ImmutableDual},
) where {Nw,T<:CuFloatOrComplex}
    y = sum(primal(x))
    parent_partials = Nfwd._lane_views(tangent(x).value.parent)
    dy_lanes = ntuple(k -> sum(parent_partials[k]), Val(Nw))
    return Lifted{typeof(y),Nw}(y, _wrap_scalar_v_lanes(y, dy_lanes))
end
function frule!!(
    ::Lifted{typeof(sum),Nw},
    x::Lifted{<:Adjoint{T,<:CuMaybeComplexArray},Nw,<:ImmutableDual},
) where {Nw,T<:CuFloatOrComplex}
    y = sum(primal(x))
    parent_partials = Nfwd._lane_views(tangent(x).value.parent)
    # Adjoint applies elementwise conj — sum then conjugate.
    dy_lanes = ntuple(k -> conj(sum(parent_partials[k])), Val(Nw))
    return Lifted{typeof(y),Nw}(y, _wrap_scalar_v_lanes(y, dy_lanes))
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
# Limitation: the NDual pass threads partials through the CuArray *elements* and the scalar
# arguments only.  Anything a function carries itself — a closed-over scalar or array, a
# callable struct's field — is invisible to the kernel, so its gradient would come back an
# exact zero.  Both the mapped reductions and the broadcast rules refuse that instead.
#
# Keyed on `tangent_type`, not `rdata_type`: a captured CuArray has no rdata at all yet is
# every bit as differentiable, and `rdata_type` applied to a primal type throws inside
# `fields_type` — which is what used to reach the user in place of this message.
function _throw_gpu_unthreaded(T, spelling, noun)
    # A float range is the one refusal here with an easy remedy, and the one users are most
    # likely to hit: the range itself is usually constant, but it may be built from
    # differentiated endpoints, and nothing at this point can tell the two apart.
    hint = if T <: AbstractRange
        "Materialise the range first — `x .* CuArray(collect(r))` — or, if its endpoints " *
        "carry no derivative, use an integer range, which is threaded as a constant. "
    else
        "Pass captured state as an argument instead — `((t, c) -> c * t).(x, a)` " *
        "differentiates correctly — or write a rule for the enclosing operation. "
    end
    return _throw_gpu_argument_error(
        "Mooncake: $spelling over CuArray does not support $noun of type $T. Partials are " *
        "threaded through GPU array elements and real or complex float scalars only, so " *
        "this one's gradient would silently be zero. " *
        hint *
        _UNIMPL_MSG,
    )
end
# Does a tangent type bottom out in anything a kernel would have to thread?  NoTangent does
# not, and a struct or Ref tangent does only if one of its fields does — `x .^ 7` lowers to
# `literal_pow.(Ref(^), x, Ref(Val(7)))`, whose Ref tangents are MutableTangents over
# NoTangent and must pass.  Unrecognised tangent types count as carrying data: refusing a
# case that turns out to be inert is a visible error, while passing one that is not is a
# silent zero.
_carries_tangent(::Type{NoTangent}) = false
_carries_tangent(::Type{Mooncake.PossiblyUninitTangent{T}}) where {T} = _carries_tangent(T)
function _carries_tangent(::Type{T}) where {T}
    T <: Union{Tangent,Mooncake.MutableTangent} || return true
    return any(_carries_tangent, fieldtypes(Mooncake.fields_type(T)))
end
_carries_differentiable_state(x) = _carries_tangent(tangent_type(typeof(x)))

function _check_gpu_captured_state(f, spelling)
    _carries_differentiable_state(f) || return nothing
    return _throw_gpu_unthreaded(typeof(f), spelling, "a function carrying its own state")
end
_check_gpu_sum_f(f) = _check_gpu_captured_state(f, "sum(f, x) / mapreduce")

# A leaf the kernel can thread: a GPU array (wrappers included) or a float/complex scalar.
# Anything else that carries a tangent — a Ref over a float, a struct, a host array — would be
# dropped in silence by _leaf_effective_tangent's catch-all.
_gpu_threads_leaf(::CuMaybeWrappedArray) = true
_gpu_threads_leaf(::CuFloatOrComplex) = true
_gpu_threads_leaf(x) = !_carries_differentiable_state(x)

function _gpu_sum_f_rrule(f, x, pkw=NamedTuple())
    _check_gpu_sum_f(f)
    flat_px = parent(primal(x))
    flat_dx = _fields(tangent(x)).parent
    out = _gpu_broadcast_dual(f, flat_px)
    decoded = _gpu_decode_ndual_output(Val(:sum), out, pkw)
    if tangent_type(typeof(decoded.primal_out)) === NoTangent
        function sum_f_nondiff_pb!!(::NoRData)
            return NoRData(), NoRData(), NoRData()
        end
        return zero_fcodual(decoded.primal_out), sum_f_nondiff_pb!!
    elseif get(pkw, :dims, :) isa Colon
        function sum_f_scalar_pb!!(dy)
            decoded.is_diff && _gpu_reduced_pullback!(flat_px, flat_dx, out, dy)
            return NoRData(), NoRData(), NoRData()
        end
        return zero_fcodual(decoded.primal_out), sum_f_scalar_pb!!
    else
        dy_out = zero_tangent(decoded.primal_out)
        function sum_f_array_pb!!(::NoRData)
            decoded.is_diff && _gpu_reduced_pullback!(flat_px, flat_dx, out, dy_out)
            return NoRData(), NoRData(), NoRData()
        end
        return CoDual(decoded.primal_out, dy_out), sum_f_array_pb!!
    end
end

@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,CuFloatArray})
@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,<:Adjoint{<:IEEEFloat,<:CuFloatArray}})
@is_primitive(MinimalCtx, Tuple{typeof(sum),Any,<:Transpose{<:IEEEFloat,<:CuFloatArray}})

# `abs2` has a simple analytic derivative, so avoid the generic NDual reduction's temporary
# arrays and partial-extraction kernels. This is the loss reduction used by Flux's model tests.
function frule!!(
    ::Lifted{typeof(sum),Nw}, ::Lifted{typeof(abs2),Nw}, x::Lifted{<:CuGpuSumFArray,Nw}
) where {Nw}
    px, dx = arrayify(x)
    y = sum(abs2, px)
    return Lifted{typeof(y),Nw}(
        y, _wrap_scalar_v_lanes(y, ntuple(k -> 2 * real(dot(px, dx[k])), Val(Nw)))
    )
end
function rrule!!(::CoDual{typeof(sum)}, ::CoDual{typeof(abs2)}, x::CoDual{<:CuGpuSumFArray})
    px, dx = arrayify(x)
    function sum_abs2_pb!!(dy)
        dx .+= (2 * dy) .* px
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(sum(abs2, px)), sum_abs2_pb!!
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
# Width-`Nw` forward rule for `sum(f, x)` on real/complex CuArrays.
# Shared width-N `sum(f, x)` forward body: one dual broadcast computes f and df/dx for every
# element, then each lane reuses `out` for a cheap reduction against that lane's input tangent.
# `flat_px`/`x_partials` are extracted per V shape by the callers (dense NDualArray vs the
# Transpose/Adjoint ImmutableDual parent). A non-differentiable mapping result (e.g. a Bool/Int-valued
# `f`) yields a `NoDual` V — a non-float `primal_out` must NOT go through `_wrap_scalar_v_lanes`
# (float-only). Mirrors the zero-derivative reverse rrule.
@inline function _gpu_sum_f_lifted(::Val{Nw}, pf, flat_px, x_partials) where {Nw}
    _check_gpu_sum_f(pf)
    out = _gpu_broadcast_dual(pf, flat_px)
    decoded = _gpu_decode_ndual_output(Val(:sum), out, (flat_px,))
    P_out = typeof(decoded.primal_out)
    # `is_diff` reports that the kernel's elements carried no partials, NOT that the result is
    # non-differentiable — a mapped `f` that strips the dual lands here with a differentiable
    # `P_out`. `zero_dual` gives the canonical V either way, `NoDual` included.
    decoded.is_diff || return Lifted{P_out,Nw}(
        decoded.primal_out, Mooncake.zero_dual(Val(Nw), decoded.primal_out)
    )
    dy_lanes = ntuple(
        k -> _gpu_accumulate_reduced_jvp(
            out, (flat_px,), (x_partials[k],), decoded.primal_out
        ),
        Val(Nw),
    )
    return Lifted{P_out,Nw}(
        decoded.primal_out, _wrap_scalar_v_lanes(decoded.primal_out, dy_lanes)
    )
end

function frule!!(
    ::Lifted{typeof(sum),Nw}, f::Lifted, x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray}
) where {Nw}
    return _gpu_sum_f_lifted(Val(Nw), primal(f), primal(x), Nfwd._lane_views(tangent(x)))
end

# Wrap a scalar primal `y` with per-lane partials `dy_lanes` into the canonical
# V, picking by scalar shape: `NDual` for real, `Complex{NDual}` for complex —
# so the invalid `NDual{Complex{R},N}` (NDual.T must be IEEEFloat) is never built.
@inline function _wrap_scalar_v_lanes(y::T, dy_lanes::NTuple{N,T}) where {T<:IEEEFloat,N}
    return NDual{T,N}(y, dy_lanes)
end
@inline function _wrap_scalar_v_lanes(
    y::Complex{R}, dy_lanes::NTuple{N,Complex{R}}
) where {R<:IEEEFloat,N}
    re = NDual{R,N}(real(y), ntuple(k -> real(dy_lanes[k]), Val(N)))
    im = NDual{R,N}(imag(y), ntuple(k -> imag(dy_lanes[k]), Val(N)))
    return Complex(re, im)
end
# A `dims` reduction returns an array and a Colon one a scalar, and the keyword rules below
# produce both from one body, so the V they build has to follow the result.
@inline _wrap_v_lanes(y::Number, dy_lanes) = _wrap_scalar_v_lanes(y, dy_lanes)
@inline function _wrap_v_lanes(y::AbstractArray, dy_lanes::NTuple{N,Any}) where {N}
    return NDualArray{eltype(y),N,ndims(y),typeof(y)}(y, dy_lanes)
end

# Transpose/Adjoint of CuFloatArray — V is ImmutableDual{@NamedTuple{parent::NDualArray}}.
function frule!!(
    ::Lifted{typeof(sum),Nw},
    f::Lifted,
    x::Lifted{
        <:Union{Adjoint{<:IEEEFloat,<:CuFloatArray},Transpose{<:IEEEFloat,<:CuFloatArray}},
        Nw,
        <:ImmutableDual,
    },
) where {Nw}
    return _gpu_sum_f_lifted(
        Val(Nw), primal(f), parent(primal(x)), Nfwd._lane_views(tangent(x).value.parent)
    )
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
        ::Lifted{typeof(mapreduce),Nw},
        f::Lifted,
        ::Lifted{typeof($_op)},
        x::Lifted{<:CuMaybeComplexArray},
    ) where {Nw}
        return frule!!(zero_lifted(Val(Nw), sum), f, x)
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
        ::Lifted{typeof(reduce),Nw},
        ::Lifted{typeof($_op)},
        x::Lifted{<:CuMaybeComplexArray},
    ) where {Nw}
        return frule!!(zero_lifted(Val(Nw), $_fn), x)
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

# The Core.kwcall spellings.  Mooncake lowers every keyword call to Core.kwcall, which the
# positional claims above do not cover, so `reduce(+, x; dims=1)` and friends escaped to
# GPUArrays' untraceable reduction kernel.  `reduce` forwards to the sum/prod *keyword* rules,
# picking up `dims`, `init` and the identity check with them.
for (_op, _fn) in ((:(+), :sum), (:(Base.:*), :prod))
    @eval @is_primitive(
        MinimalCtx,
        Tuple{
            typeof(Core.kwcall),NamedTuple,typeof(reduce),typeof($_op),CuMaybeComplexArray
        },
    )
    @eval function frule!!(
        kc::Lifted{typeof(Core.kwcall),Nw},
        kw::Lifted{<:NamedTuple},
        ::Lifted{typeof(reduce),Nw},
        ::Lifted{typeof($_op)},
        x::Lifted{<:CuMaybeComplexArray},
    ) where {Nw}
        return frule!!(kc, kw, zero_lifted(Val(Nw), $_fn), x)
    end
    @eval function rrule!!(
        kc::CoDual{typeof(Core.kwcall)},
        kw::CoDual{<:NamedTuple},
        ::CoDual{typeof(reduce)},
        ::CoDual{typeof($_op)},
        x::CoDual{<:CuMaybeComplexArray},
    )
        y, pb!! = rrule!!(kc, kw, zero_fcodual($_fn), x)
        function reduce_kw_pb!!(dy)
            _, r_kw, _, r_x = pb!!(dy)     # delegate pullback: (kwcall, kw, fn, x)
            return NoRData(), r_kw, NoRData(), NoRData(), r_x
        end
        return y, reduce_kw_pb!!
    end
end

for _op in (:(+), :(Base.add_sum))
    @eval @is_primitive(
        MinimalCtx,
        Tuple{
            typeof(Core.kwcall),
            NamedTuple,
            typeof(mapreduce),
            Any,
            typeof($_op),
            CuMaybeComplexArray,
        },
    )
    @eval function frule!!(
        ::Lifted{typeof(Core.kwcall),Nw},
        kw::Lifted{<:NamedTuple},
        ::Lifted{typeof(mapreduce),Nw},
        f::Lifted,
        ::Lifted{typeof($_op)},
        x::Lifted{<:CuMaybeComplexArray},
    ) where {Nw}
        pkw = primal(kw)
        _check_reduction_identity(sum, pkw)
        for k in 1:Nw
            _check_reduction_init(tangent(kw, k))
        end
        _mapreduce_kw_is_plain(pkw) || return _throw_gpu_mapreduce_dims()
        out = frule!!(zero_lifted(Val(Nw), sum), f, x)
        _check_mapreduce_init_type(pkw, primal(out))
        return out
    end
    @eval function rrule!!(
        kc::CoDual{typeof(Core.kwcall)},
        kw::CoDual{<:NamedTuple},
        ::CoDual{typeof(mapreduce)},
        f::CoDual,
        ::CoDual{typeof($_op)},
        x::CoDual{<:CuMaybeComplexArray},
    )
        y, pb!! = rrule!!(kc, kw, zero_fcodual(sum), f, x)
        function mapreduce_kw_pb!!(dy)
            _, r_kw, _, r_f, r_x = pb!!(dy)
            return NoRData(), r_kw, NoRData(), r_f, NoRData(), r_x
        end
        return y, mapreduce_kw_pb!!
    end
end

# Catch-all rules for unsupported operators — give a clear error rather than letting
# Mooncake attempt to trace into an opaque CUDA reduction kernel.
@is_primitive(MinimalCtx, Tuple{typeof(mapreduce),Any,Any,CuArray})
function frule!!(
    ::Lifted{typeof(mapreduce),Nw}, f::Lifted, op::Lifted, x::Lifted{<:CuArray}
) where {Nw}
    return _throw_gpu_argument_error(
        "Mooncake: mapreduce on CuArray only supports op=+ or op=Base.add_sum over " *
        "float or complex arrays; got op=$(primal(op)) over " *
        "$(eltype(primal(x))). " *
        _UNIMPL_MSG,
    )
end
function rrule!!(::CoDual{typeof(mapreduce)}, f::CoDual, op::CoDual, x::CoDual{<:CuArray})
    return _throw_gpu_argument_error(
        "Mooncake: mapreduce on CuArray only supports op=+ or op=Base.add_sum over " *
        "float or complex arrays; got op=$(primal(op)) over " *
        "$(eltype(primal(x))). " *
        _UNIMPL_MSG,
    )
end

@is_primitive(MinimalCtx, Tuple{typeof(reduce),Any,CuArray})
function frule!!(::Lifted{typeof(reduce),Nw}, op::Lifted, x::Lifted{<:CuArray}) where {Nw}
    _throw_gpu_argument_error(
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

@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(mapreduce),Any,Any,CuArray}
)
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    ::Lifted{<:NamedTuple},
    f::Lifted{typeof(mapreduce),Nw},
    mf::Lifted,
    op::Lifted,
    x::Lifted{<:CuArray},
) where {Nw}
    return frule!!(f, mf, op, x)
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    ::CoDual{<:NamedTuple},
    f::CoDual{typeof(mapreduce)},
    mf::CoDual,
    op::CoDual,
    x::CoDual{<:CuArray},
)
    return rrule!!(f, mf, op, x)
end
@is_primitive(MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(reduce),Any,CuArray})
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    ::Lifted{<:NamedTuple},
    f::Lifted{typeof(reduce),Nw},
    op::Lifted,
    x::Lifted{<:CuArray},
) where {Nw}
    return frule!!(f, op, x)
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    ::CoDual{<:NamedTuple},
    f::CoDual{typeof(reduce)},
    op::CoDual,
    x::CoDual{<:CuArray},
)
    return rrule!!(f, op, x)
end

# repeat on GPU arrays, both the `counts...` and the `inner=`/`outer=` spellings.
# Both launch a kernel Mooncake cannot trace, and the keyword form needs its own claim
# because the positional one does not cover Core.kwcall.
#
# repeat places the copy of x[i] for inner offset k and outer offset o at
# k + (i-1)*I + (o-1)*I*S along each dimension, which is exactly column-major
# (inner, size, outer), so reshaping into those triples and summing the inner and outer
# axes needs no scalar indexing.  ChainRules instead loops over
# `pairs(IndexCartesian(), dY)` for the keyword form, which cannot run on a device.
#
# `counts` may be shorter than ndims(x) (padded with 1) or longer (the result gains
# dimensions), so sizes are padded to ndims of the output and the result reshaped back.
function _repeat_reduce(
    dY, S::NTuple{N,Int}, inner::NTuple{N,Int}, outer::NTuple{N,Int}
) where {N}
    triples = ntuple(3N) do d
        dim, which = fldmod1(d, 3)
        which == 1 && return inner[dim]
        return which == 2 ? S[dim] : outer[dim]
    end
    reduced_axes = ntuple(2N) do d
        dim, which = fldmod1(d, 2)
        which == 1 ? 3dim - 2 : 3dim
    end
    return reshape(sum(reshape(dY, triples); dims=reduced_axes), S)
end

_repeat_pad(t, ::Val{N}) where {N} = ntuple(d -> d <= length(t) ? Int(t[d]) : 1, N)
# `inner`/`outer` default to `nothing` in Base.repeat, so callers may pass it explicitly.
_repeat_pad(::Nothing, v::Val) = _repeat_pad((), v)

@is_primitive(MinimalCtx, Tuple{typeof(repeat),CuMaybeComplexArray,Vararg{Integer}})
function frule!!(
    ::Lifted{typeof(repeat),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw},
    counts::Vararg{Lifted{<:Integer}},
) where {Nw}
    px, x_partials = arrayify(x)
    c = map(primal, counts)
    y = repeat(px, c...)
    y_partials = ntuple(k -> repeat(x_partials[k], c...), Val(Nw))
    Y = typeof(y)
    return Lifted{Y,Nw}(y, NDualArray{eltype(Y),Nw,ndims(y),Y}(y, y_partials))
end
function rrule!!(
    ::CoDual{typeof(repeat)},
    x::CoDual{<:CuMaybeComplexArray},
    counts::Vararg{CoDual{<:Integer}},
)
    px, dx = arrayify(x)
    c = map(primal, counts)
    y = repeat(px, c...)
    N = ndims(y)
    S = _repeat_pad(size(px), Val(N))
    outer = _repeat_pad(c, Val(N))
    inner = ntuple(_ -> 1, Val(N))
    dy = zero(y)
    function repeat_pb!!(::NoRData)
        dx .+= reshape(_repeat_reduce(dy, S, inner, outer), size(dx))
        return ntuple(_ -> NoRData(), length(counts) + 2)
    end
    return CoDual(y, dy), repeat_pb!!
end

@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(repeat),CuMaybeComplexArray}
)
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple},
    ::Lifted{typeof(repeat),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw},
) where {Nw}
    pkw = primal(kw)
    px, x_partials = arrayify(x)
    y = repeat(px; pkw...)
    y_partials = ntuple(k -> repeat(x_partials[k]; pkw...), Val(Nw))
    Y = typeof(y)
    return Lifted{Y,Nw}(y, NDualArray{eltype(Y),Nw,ndims(y),Y}(y, y_partials))
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(repeat)},
    x::CoDual{<:CuMaybeComplexArray},
)
    pkw = primal(kw)
    px, dx = arrayify(x)
    y = repeat(px; pkw...)
    N = ndims(y)
    S = _repeat_pad(size(px), Val(N))
    inner = _repeat_pad(get(pkw, :inner, ()), Val(N))
    outer = _repeat_pad(get(pkw, :outer, ()), Val(N))
    dy = zero(y)
    function repeat_kw_pb!!(::NoRData)
        dx .+= reshape(_repeat_reduce(dy, S, inner, outer), size(dx))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), repeat_kw_pb!!
end

# Rule for keyword `sum(x; dims, init)`. The bare-sum primitive above does not
# claim the Core.kwcall spelling, so Base lowers it onto GPUArrays'
# mapreducedim! and finally `cufunction`, whose `@lock` try/finally kills
# reverse-mode tracing (#1273).
#
# `init` is treated as a non-differentiated constant: Julia requires it to be a
# neutral element, and GPUArrays folds it into a backend-defined number of
# partial reductions, so a derivative through it is not well-defined; the frule
# rejects a nonzero `init` tangent instead. Reverse mode gets no such signal, so
# `dx` is correct and `init`'s component is silently zero. `init` also sets the
# output eltype (GPUArrays takes it from typeof(init), so init=0.0 on a Float32
# array gives a Float64 result), hence the output-typed zero seeding the tangent
# reduction.
@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(sum),CuMaybeComplexArray}
)
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple},
    ::Lifted{typeof(sum),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw},
) where {Nw}
    pkw = primal(kw)
    _check_reduction_identity(sum, pkw)
    for k in 1:Nw
        _check_reduction_init(tangent(kw, k))
    end
    px, x_partials = arrayify(x)
    y = sum(px; pkw...)
    dims = get(pkw, :dims, :)
    lanes = ntuple(k -> sum(x_partials[k]; dims=dims, init=zero(eltype(y))), Val(Nw))
    return Lifted{typeof(y),Nw}(y, _wrap_v_lanes(y, lanes))
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(sum)},
    x::CoDual{<:CuMaybeComplexArray},
)
    pkw = primal(kw)
    _check_reduction_identity(sum, pkw)
    kw_rdata = zero_rdata(pkw)
    px, dx = arrayify(x)
    y = sum(px; pkw...)
    if get(pkw, :dims, :) isa Colon
        function sum_kw_scalar_pb!!(dy)
            dx .+= dy
            return NoRData(), kw_rdata, NoRData(), NoRData()
        end
        return CoDual(y, NoFData()), sum_kw_scalar_pb!!
    end
    dy_out = zero(y)
    function sum_kw_array_pb!!(::NoRData)
        dx .+= dy_out
        return NoRData(), kw_rdata, NoRData(), NoRData()
    end
    return CoDual(y, dy_out), sum_kw_array_pb!!
end

@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(sum),Any,CuMaybeComplexArray},
)
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple,Nw},
    ::Lifted{typeof(sum),Nw},
    f::Lifted,
    x::Lifted{<:CuMaybeComplexArray,Nw},
) where {Nw}
    pkw = primal(kw)
    _check_reduction_identity(sum, pkw)
    for k in 1:Nw
        _check_reduction_init(tangent(kw, k))
    end
    # `dims` is not yet threaded through the lifted reduction path; refuse it rather than
    # silently reducing over everything.
    _mapreduce_kw_is_plain(pkw) || return _throw_gpu_mapreduce_dims()
    return _gpu_sum_f_lifted(Val(Nw), primal(f), primal(x), Nfwd._lane_views(tangent(x)))
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(sum)},
    f::CoDual,
    x::CoDual{<:CuMaybeComplexArray},
)
    pkw = primal(kw)
    _check_reduction_identity(sum, pkw)
    kw_rdata = zero_rdata(pkw)
    y, pb!! = _gpu_sum_f_rrule(primal(f), x, pkw)
    function sum_f_kw_pb!!(dy)
        _, r_f, r_x = pb!!(dy)
        return NoRData(), kw_rdata, NoRData(), r_f, r_x
    end
    return y, sum_f_kw_pb!!
end

@is_primitive(MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(sum),Any,CuArray})
function frule!!(
    ::Lifted{typeof(Core.kwcall),Nw},
    ::Lifted{<:NamedTuple},
    ::Lifted{typeof(sum),Nw},
    ::Lifted,
    ::Lifted{<:CuArray},
) where {Nw}
    return _throw_gpu_argument_error(
        "Mooncake: sum(f, x; ...) on CuArray only supports float or complex arrays; got " *
        "$(eltype(primal(x))). " *
        _UNIMPL_MSG,
    )
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    ::CoDual{<:NamedTuple},
    ::CoDual{typeof(sum)},
    ::CoDual,
    x::CoDual{<:CuArray},
)
    return _throw_gpu_argument_error(
        "Mooncake: sum(f, x; ...) on CuArray only supports float or complex arrays; got " *
        "$(eltype(primal(x))). " *
        _UNIMPL_MSG,
    )
end

# vcat / hcat / cat on CuMaybeWrappedArray (see its definition above for what that
# includes and why).
# frule:    tangent of concatenation = concatenation of tangents (concat is linear).
# pullback: selectdim returns a view (no allocation per slice); running-offset loop
#           avoids pre-allocating an offsets array.
# A real variable's cotangent is real even where the operation that consumed it produced a
# complex output: Mooncake's convention carries dL/dRe in the real field, and the imaginary
# part of a real leaf's contribution belongs to no variable.  Concatenating or casting a real
# array alongside a complex one is where that shows up, and accumulating unprojected would ask
# the device to convert a Complex to a Float — a kernel-side exception, not a clean error.
@inline _project_cotangent(dst, contrib) = contrib
@inline _project_cotangent(dst::AbstractArray{<:Real}, contrib) = real.(contrib)
@inline _project_cotangent(::Real, contrib) = real(contrib)

@inline function _cu_concat_pb!(fdatas, dy_out, dim::Integer)
    offset = 0
    for i in eachindex(fdatas)
        n = size(fdatas[i], dim)
        fdatas[i] .+= _project_cotangent(
            fdatas[i], selectdim(dy_out, dim, (offset + 1):(offset + n))
        )
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
            fi .+= _project_cotangent(fi, view(dy_out, ranges...))
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
_unwrap_cat_dim(d::AbstractVector{<:Integer}) = Tuple(d)
function _unwrap_cat_dim(d)
    return throw(
        ArgumentError(
            "Mooncake: cat requires dims to be an Integer, Val{N}, or a Tuple or vector " *
            "of Integers; got dims=$(d).",
        ),
    )
end

@is_primitive(
    MinimalCtx, Tuple{typeof(vcat),CuMaybeWrappedArray,Vararg{CuMaybeWrappedArray}}
)
function frule!!(
    ::Lifted{typeof(vcat),Nw}, args::Lifted{<:CuMaybeWrappedArray}...
) where {Nw}
    # vcat is linear: concat the primals, and concat each lane's partials the same way. `arrayify`
    # canonicalises each argument's primal and its per-lane partials through any wrapper (mirroring
    # the reverse rrule below); the dense result gives a plain `NDualArray` V.
    pairs = map(arrayify, args)
    y = vcat(map(first, pairs)...)
    y_partials = ntuple(k -> vcat(map(p -> p[2][k], pairs)...), Val(Nw))
    Y = typeof(y)
    return Lifted{Y,Nw}(y, NDualArray{eltype(y),Nw,ndims(y),Y}(y, y_partials))
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
function frule!!(
    ::Lifted{typeof(hcat),Nw}, args::Lifted{<:CuMaybeWrappedArray}...
) where {Nw}
    pairs = map(arrayify, args)
    y = hcat(map(first, pairs)...)
    y_partials = ntuple(k -> hcat(map(p -> p[2][k], pairs)...), Val(Nw))
    Y = typeof(y)
    return Lifted{Y,Nw}(y, NDualArray{eltype(y),Nw,ndims(y),Y}(y, y_partials))
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
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple},
    ::Lifted{typeof(cat)},
    args::Lifted{<:CuMaybeWrappedArray}...,
) where {Nw}
    pkw = primal(kw)
    pairs = map(arrayify, args)
    y = cat(map(first, pairs)...; pkw...)
    y_partials = ntuple(k -> cat(map(p -> p[2][k], pairs)...; pkw...), Val(Nw))
    Y = typeof(y)
    return Lifted{Y,Nw}(y, NDualArray{eltype(y),Nw,ndims(y),Y}(y, y_partials))
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
    @eval function frule!!(
        ::Lifted{typeof($_fn),Nw}, ::Lifted{<:Union{AbstractArray,Number}}...
    ) where {Nw}
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
    ::Lifted{typeof(Core.kwcall),Nw},
    ::Lifted{<:NamedTuple},
    ::Lifted{typeof(cat)},
    ::Lifted{<:Union{AbstractArray,Number}}...,
) where {Nw}
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

# Avoid descending through the mutating rules below when `*` allocated its output fresh.
for BDim in (2, 1)
    @eval begin
        @is_primitive MinimalCtx Tuple{
            typeof(*),CuArray{P,2},CuArray{P,$BDim}
        } where {P<:CuFloatOrComplex}
        function frule!!(
            ::Lifted{typeof(*),Nw},
            A::Lifted{<:CuArray{P,2},Nw},
            B::Lifted{<:CuArray{P,$BDim},Nw},
        ) where {P<:CuFloatOrComplex,Nw}
            pA, dA = arrayify(A)
            pB, dB = arrayify(B)
            C = pA * pB
            out = zero_lifted(Val(Nw), C)
            _, dC = arrayify(out)
            # `dA_k * B + A * dB_k`, per lane; the primal product is computed once.
            for k in 1:Nw
                mul!(dC[k], dA[k], pB, one(eltype(C)), zero(eltype(C)))
                mul!(dC[k], pA, dB[k], one(eltype(C)), one(eltype(C)))
            end
            return out
        end
        function rrule!!(
            ::CoDual{typeof(*)}, A::CoDual{<:CuArray{P,2}}, B::CoDual{<:CuArray{P,$BDim}}
        ) where {P<:CuFloatOrComplex}
            pA, dA = arrayify(A)
            pB, dB = arrayify(B)
            C = pA * pB
            dC = zero(C)
            function gpu_mul_pb!!(::NoRData)
                mul!(dA, dC, adjoint(pB), one(eltype(C)), one(eltype(C)))
                mul!(dB, adjoint(pA), dC, one(eltype(C)), one(eltype(C)))
                return NoRData(), NoRData(), NoRData()
            end
            return CoDual(C, dC), gpu_mul_pb!!
        end
    end
end

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

# Batching convention (GPU): a width-N frule that would issue one cuBLAS call per lane instead
# batches into a single `gemm_batched!` / `gemv_batched!` — the batched API takes the N separate
# partial arrays (no gather) and collapses N launches to one (~3x measured). The shared operand
# repeats across lanes via `fill`; `Nw == 1` keeps the direct single-call path.
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
    ::Lifted{typeof(LinearAlgebra.generic_matmatmul!),Nw},
    C::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    tA::Lifted{Char},
    tB::Lifted{Char},
    A::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    B::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
) where {Nw}
    pC = primal(C)
    pA = primal(A)
    pB = primal(B)
    tAv = primal(tA)
    tBv = primal(tB)
    T = eltype(pA)
    _check_complex_transpose_flag(T, tAv, tBv)
    _1 = one(T)
    _0 = zero(T)
    cuBLAS.gemm!(tAv, tBv, _1, pA, pB, _0, pC)
    C_partials = Nfwd._lane_views(tangent(C))
    A_partials = Nfwd._lane_views(tangent(A))
    B_partials = Nfwd._lane_views(tangent(B))
    if Nw == 1
        dC = C_partials[1]
        cuBLAS.gemm!(tAv, tBv, _1, A_partials[1], pB, _0, dC)
        cuBLAS.gemm!(tAv, tBv, _1, pA, B_partials[1], _1, dC)
    else
        # Batch the 2N per-lane gemms into 2 gemm_batched! (see the batching convention above).
        dCs = collect(C_partials)
        cuBLAS.gemm_batched!(tAv, tBv, _1, collect(A_partials), fill(pB, Nw), _0, dCs)
        cuBLAS.gemm_batched!(tAv, tBv, _1, fill(pA, Nw), collect(B_partials), _1, dCs)
    end
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
# alpha / beta are differentiated.  They are usually `true`/`false` (from `MulAddMul`), which
# carry no derivative and cost nothing here, but a caller may pass floats — `mul!(C, A, B, α,
# β)` — and their derivatives are simple: ⟨op_A(A)·op_B(B), dC⟩ and ⟨C_old, dC⟩.

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
    ::Lifted{typeof(LinearAlgebra.generic_matmatmul!),Nw},
    C::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    tA::Lifted{Char},
    tB::Lifted{Char},
    A::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    B::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    alpha::Lifted{<:Number},
    beta::Lifted{<:Number},
) where {Nw}
    pC = primal(C)
    pA = primal(A)
    pB = primal(B)
    tAv = primal(tA)
    tBv = primal(tB)
    T = eltype(pA)
    _check_complex_transpose_flag(T, tAv, tBv)
    _α = T(primal(alpha))
    _β = T(primal(beta))
    _1 = one(T)
    # tangent: dCₖ := α*(op_A(dAₖ)*op_B(pB) + op_A(pA)*op_B(dBₖ)) + β*dCₖ + dαₖ*op_A(A)*op_B(B)
    #                 + dβₖ*C_old.  Every lane runs before the primal so that the dβ term
    # still sees the old C, which the primal overwrites.
    C_partials = Nfwd._lane_views(tangent(C))
    A_partials = Nfwd._lane_views(tangent(A))
    B_partials = Nfwd._lane_views(tangent(B))
    if Nw == 1
        dC = C_partials[1]
        cuBLAS.gemm!(tAv, tBv, _α, A_partials[1], pB, _β, dC)
        cuBLAS.gemm!(tAv, tBv, _α, pA, B_partials[1], _1, dC)
    else
        # Batch the N per-lane JVP gemms into two `gemm_batched!` launches (2 vs 2N): the shared
        # operand (pB, then pA) repeats across lanes, so cuBLAS applies it to all N partials in one
        # kernel — launch-overhead-bound N gemms collapse to one (measured ~2x on GPU).
        dCs = collect(C_partials)
        cuBLAS.gemm_batched!(tAv, tBv, _α, collect(A_partials), fill(pB, Nw), _β, dCs)
        cuBLAS.gemm_batched!(tAv, tBv, _α, fill(pA, Nw), collect(B_partials), _1, dCs)
    end
    for k in 1:Nw
        _geam_scalar_jvp!(C_partials[k], tangent(beta, k), 'N', pC)
    end
    _blas_product_jvp!(
        C_partials,
        ntuple(k -> tangent(alpha, k), Val(Nw)),
        _geam_op(tAv, pA),
        _geam_op(tBv, pB),
    )
    # primal: C := α*op_A(A)*op_B(B) + β*C
    cuBLAS.gemm!(tAv, tBv, _α, pA, pB, _β, pC)
    return C
end
function rrule!!(
    ::CoDual{typeof(LinearAlgebra.generic_matmatmul!)},
    C::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    tA::CoDual{Char,NoFData},
    tB::CoDual{Char,NoFData},
    A::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    B::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    alpha::CoDual{<:Number},
    beta::CoDual{<:Number},
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
        # Adjoint of C = α*op_A(A)*op_B(B) + β*C_old requires conj(α) and conj(β), except
        # against a transposed operand: adjoint(α*adjoint(A)) folds the two conjugations
        # together, leaving α itself.  For real scalars conj is identity either way.
        _cα = conj(_α)
        _cβ = conj(_β)
        if tAv == 'N'
            cuBLAS.gemm!('N', tBv == 'N' ? 'C' : 'N', _cα, dC, pB, _1, dA) # dA += conj(α)*dC*op_B(B)^H
        else
            cuBLAS.gemm!(tBv, 'C', _α, pB, dC, _1, dA)                      # dA += α*op_B(B)*dC^H
        end
        if tBv == 'N'
            cuBLAS.gemm!(tAv == 'N' ? 'C' : 'N', 'N', _cα, pA, dC, _1, dB) # dB += conj(α)*op_A(A)^H*dC
        else
            cuBLAS.gemm!('C', tAv, _α, dC, pA, _1, dB)                      # dB += α*dC^H*op_A(A)
        end
        # Both scalar cotangents read the incoming dC, before β rescales it below.
        dα = _blas_product_rdata(primal(alpha), _geam_op(tAv, pA), _geam_op(tBv, pB), dC)
        dβ = _blas_scalar_rdata(primal(beta), pC_copy, dC)
        copyto!(pC, pC_copy)
        dC .*= _cβ  # gradient w.r.t. C_old: ΔC_old = conj(β) * ΔC_new
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dα, dβ
    end
    return C, generic_matmatmul!_7arg_pb!!
end

# cuBLAS.geam!: C := alpha*op_a(A) + beta*op_b(B).  This is the foreign-call boundary
# reached by `A + B` and `A - B` on CuMatrix, which cuBLAS generates for all nine
# Transpose/Adjoint combinations (cuBLAS/src/linalg.jl).  Tracing further reaches the
# ccall, whose scalars are passed as CuRef{T} — a primitive type with no tangent_type,
# which is the error users actually see; defining that tangent_type would only move the
# failure to the ccall itself.
function _geam_op(t::Char, M)
    t == 'N' && return M
    return t == 'T' ? transpose(M) : adjoint(M)
end

# alpha and beta are differentiated: the generated `+`/`-` pass `one(T)`, a float literal,
# so they cannot be typed as NoTangent the way the gemm! rules type theirs, and reverse
# mode has no way to tell that literal from a scalar the caller wants a gradient for.
_geam_scalar_jvp!(dC, ::NoTangent, ::Char, X) = dC
function _geam_scalar_jvp!(dC, ds::Number, t::Char, X)
    iszero(ds) && return dC
    return dC .+= convert(eltype(dC), ds) .* _geam_op(t, X)
end

# A scalar factor s multiplying a matrix X into the output takes the real inner product
# ⟨X, dC⟩ = sum(conj(X) .* dC), projected onto s's own type: a real s over a complex array
# keeps the real part.  A Bool or Integer s has no derivative; the assertion makes a scalar
# type that does carry one an error, not a silent zero.
_blas_scalar_rdata(s::IEEEFloat, X, dC) = oftype(s, real(sum(conj.(X) .* dC)))
_blas_scalar_rdata(s::Complex{<:IEEEFloat}, X, dC) = oftype(s, sum(conj.(X) .* dC))
_blas_scalar_rdata(s::Number, X, dC) = zero_rdata(s)::NoRData

# The same, where X is a product the rule never formed: `mul!(C, A, B)` passes Bool scalars,
# whose rdata is NoRData, and dispatch then skips the matmul rather than computing it to
# throw it away.  The JVP counterpart skips it for a zero tangent, which is what the float
# literal in a keyword-free `mul!` carries.
_blas_product_rdata(s::Number, X1, X2, dC) = zero_rdata(s)::NoRData
function _blas_product_rdata(s::CuFloatOrComplex, X1, X2, dC)
    return _blas_scalar_rdata(s, X1 * X2, dC)
end
# The product does not vary by lane, so it is formed once for the whole chunk — and only when
# some lane's scalar actually carries a tangent.
function _blas_product_jvp!(dCs::NTuple{N,Any}, ds::NTuple{N,Any}, X1, X2) where {N}
    any(d -> d isa Number && !iszero(d), ds) || return dCs
    X = X1 * X2
    for k in 1:N
        _geam_scalar_jvp!(dCs[k], ds[k], 'N', X)
    end
    return dCs
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(cuBLAS.geam!),
        Char,
        Char,
        Number,
        CuMaybeComplexArray,
        Number,
        CuMaybeComplexArray,
        CuMaybeComplexArray,
    },
)
function frule!!(
    ::Lifted{typeof(cuBLAS.geam!),Nw},
    ta::Lifted{Char},
    tb::Lifted{Char},
    alpha::Lifted{<:Number},
    A::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    beta::Lifted{<:Number},
    B::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    C::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
) where {Nw}
    pA, pB, pC = primal(A), primal(B), primal(C)
    A_partials = Nfwd._lane_views(tangent(A))
    B_partials = Nfwd._lane_views(tangent(B))
    C_partials = Nfwd._lane_views(tangent(C))
    T = eltype(pA)
    tav, tbv = primal(ta), primal(tb)
    _a, _b = T(primal(alpha)), T(primal(beta))
    # Every lane's tangent runs first: cuBLAS geam runs in place for C === A or C === B, so
    # the primal call would otherwise overwrite the pA/pB that the scalar terms read.
    for k in 1:Nw
        cuBLAS.geam!(tav, tbv, _a, A_partials[k], _b, B_partials[k], C_partials[k])
        _geam_scalar_jvp!(C_partials[k], tangent(alpha, k), tav, pA)
        _geam_scalar_jvp!(C_partials[k], tangent(beta, k), tbv, pB)
    end
    cuBLAS.geam!(tav, tbv, _a, pA, _b, pB, pC)
    return C
end
function rrule!!(
    ::CoDual{typeof(cuBLAS.geam!)},
    ta::CoDual{Char,NoFData},
    tb::CoDual{Char,NoFData},
    alpha::CoDual{<:Number},
    A::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    beta::CoDual{<:Number},
    B::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
    C::CoDual{<:CuMaybeComplexArray,<:CuMaybeComplexArray},
)
    pA, dA = matrixify(A)
    pB, dB = matrixify(B)
    pC, dC = matrixify(C)
    T = eltype(pA)
    tav, tbv = primal(ta), primal(tb)
    _a, _b = T(primal(alpha)), T(primal(beta))
    pC_copy = copy(pC)
    cuBLAS.geam!(tav, tbv, _a, pA, _b, pB, pC)
    function geam!_pb!!(::NoRData)
        # C may be A or B (cuBLAS geam runs in place for C === A or C === B), so restore the
        # primal and read dC out before touching pA/pB or dA/dB, which alias them.
        copyto!(pC, pC_copy)
        dC_copy = copy(dC)
        # geam has no C_old term, so C's cotangent is consumed rather than scaled.
        fill!(dC, zero(T))
        # 'C' takes alpha unconjugated: adjoint(alpha * adjoint(A)) folds the two
        # conjugations together.  Real scalars are unaffected either way.
        dA .+= (tav == 'C' ? _a : conj(_a)) .* _geam_op(tav, dC_copy)
        dB .+= (tbv == 'C' ? _b : conj(_b)) .* _geam_op(tbv, dC_copy)
        return NoRData(),
        NoRData(),
        NoRData(),
        _blas_scalar_rdata(primal(alpha), _geam_op(tav, pA), dC_copy),
        NoRData(),
        _blas_scalar_rdata(primal(beta), _geam_op(tbv, pB), dC_copy),
        NoRData(),
        NoRData()
    end
    return C, geam!_pb!!
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
#   tA='N': dA += conj(alpha) * ȳ * B^H  (outer product via gemm!('N','C'))
#   tA≠'N': dA += alpha * B * ȳ^H        (roles swapped; adjoint(alpha*adjoint(A)) folds
#                                         the two conjugations together)
#   tA='N': dB += conj(alpha) * A^H * ȳ  (gemv!('C'))
#   tA≠'N': dB += conj(alpha) * A   * ȳ  (gemv!('N'), since op(A)^H = A)
#   dY_old  = conj(beta) * ȳ             (pass-through scaled by beta)
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
    ::Lifted{typeof(LinearAlgebra.generic_matvecmul!),Nw},
    Y::Lifted{<:CuMaybeComplexVec,Nw,<:NDualArray},
    tA::Lifted{<:AbstractChar},
    A::Lifted{<:CuMaybeComplexMat,Nw,<:NDualArray},
    B::Lifted{<:CuMaybeComplexVec,Nw,<:NDualArray},
    alpha::Lifted{<:Number},
    beta::Lifted{<:Number},
) where {Nw}
    pY = primal(Y)
    pA = primal(A)
    pB = primal(B)
    tAv = primal(tA)
    av = primal(alpha)
    bv = primal(beta)
    T = eltype(pA)
    _check_gemv_eltypes(T, eltype(pB))
    _check_complex_matvecmul_transpose(T, tAv)
    _1 = one(T)
    Y_partials = Nfwd._lane_views(tangent(Y))
    A_partials = Nfwd._lane_views(tangent(A))
    B_partials = Nfwd._lane_views(tangent(B))
    # tangent (product rule): dYₖ = av*op(dAₖ)*pB + av*op(pA)*dBₖ + bv*dYₖ + davₖ*op(pA)*pB
    #                               + dbvₖ*Y_old.  The dbv term needs the old Y, so the primal
    # runs last.
    if Nw == 1
        dY = Y_partials[1]
        cuBLAS.gemv!(tAv, av, A_partials[1], pB, bv, dY)
        cuBLAS.gemv!(tAv, av, pA, B_partials[1], _1, dY)
    else
        # Batch the 2N per-lane gemvs into 2 gemv_batched! (see the batching convention above).
        dYs = collect(Y_partials)
        cuBLAS.gemv_batched!(tAv, av, collect(A_partials), fill(pB, Nw), bv, dYs)
        cuBLAS.gemv_batched!(tAv, av, fill(pA, Nw), collect(B_partials), _1, dYs)
    end
    for k in 1:Nw
        _geam_scalar_jvp!(Y_partials[k], tangent(beta, k), 'N', pY)
    end
    _blas_product_jvp!(
        Y_partials, ntuple(k -> tangent(alpha, k), Val(Nw)), _geam_op(tAv, pA), pB
    )
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
    alpha::CoDual{<:Number},
    beta::CoDual{<:Number},
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
            cuBLAS.gemm!('N', 'C', conj(av), dY_mat, pB_mat, _1, dA) # dA += conj(av)*ȳ*B^H
        else
            cuBLAS.gemm!('N', 'C', av, pB_mat, dY_mat, _1, dA) # dA += av * B * ȳ^H
        end
        # dB update: gemv with Hermitian conjugate of op(A)
        if tAv == 'N'
            cuBLAS.gemv!('C', conj(av), pA, dY, _1, dB) # dB += conj(av) * A^H * ȳ
        else
            cuBLAS.gemv!('N', conj(av), pA, dY, _1, dB) # dB += conj(av)*A*ȳ (op(A)^H = A)
        end
        # Both scalar cotangents read the incoming dY, before beta rescales it below.
        dav = _blas_product_rdata(av, _geam_op(tAv, pA), pB, dY)
        dbv = _blas_scalar_rdata(bv, pY_copy, dY)
        # Y tangent passes through scaled by beta
        dY .*= conj(bv)
        copyto!(pY, pY_copy)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dav, dbv
    end
    return Y, generic_matvecmul!_pb!!
end
# The tangent of Array{T} is Array{T} (fdata, accumulated in-place).
# The tangent of CuArray{T} is CuArray{T} (fdata, accumulated in-place).
#
# `cu` is a host-to-device transfer for a host argument and a copy for a device one, and the
# claim admits both, so the cotangent has to travel back to whichever side the argument's
# fdata lives on.  Pulling it to the host unconditionally asked a device buffer to accumulate
# a host array, which is a kernel-compilation failure rather than a scalar-indexing error.
_cu_pullback_like(::AbstractArray, dy) = Array(dy)
_cu_pullback_like(::CuArray, dy) = dy

# Claimed for the two shapes whose tangent is the array itself. An `Adjoint`, `Transpose` or
# `Diagonal` argument carries an `FData{@NamedTuple{parent::…}}` (reverse) or the
# corresponding struct lift (forward) instead, which the body would treat as a bare array;
# those spellings decompose through the wrapper's own rules.
const _CuTransferable = Union{Array{<:CuFloatOrComplex},CuArray{<:CuFloatOrComplex}}
@is_primitive(MinimalCtx, Tuple{typeof(cu),_CuTransferable})
# `cu` the primal and each lane's partial to the device.
function frule!!(
    ::Lifted{typeof(cu),Nw}, x::Lifted{<:_CuTransferable,Nw,<:NDualArray}
) where {Nw}
    y = cu(primal(x))
    x_partials = Nfwd._lane_views(tangent(x))
    y_partials = ntuple(k -> cu(x_partials[k]), Val(Nw))
    Y = typeof(y)
    Element = eltype(y)
    return Lifted{Y,Nw}(y, NDualArray{Element,Nw,ndims(y),Y}(y, y_partials))
end
function rrule!!(::CoDual{typeof(cu)}, x::CoDual{<:_CuTransferable})
    dx = tangent(x)
    dy_gpu = cu(zero(primal(x)))  # output fdata, accumulated into by downstream
    function cu_pb!!(::NoRData)
        dx .+= _cu_pullback_like(dx, dy_gpu)
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
    ::Lifted{Type{Array{T,D}},Nw}, x::Lifted{<:CuArray{T,D},Nw,<:NDualArray}
) where {T<:CuFloatOrComplex,D,Nw}
    y = Array(primal(x))
    x_partials = Nfwd._lane_views(tangent(x))
    y_partials = ntuple(k -> Array(x_partials[k]), Val(Nw))
    Y = typeof(y)
    return Lifted{Y,Nw}(y, NDualArray{T,Nw,D,Y}(y, y_partials))
end
function rrule!!(
    ::CoDual{Type{Array{T,N}}}, x::CoDual{<:CuArray{T,N}}
) where {T<:CuFloatOrComplex,N}
    dx = tangent(x)
    dy_cpu = Array(zero(primal(x)))  # output fdata, accumulated into by downstream
    function array_pb!!(::NoRData)
        # `cu` is an adaptor, not a transfer: it narrows Float64 to Float32 and ComplexF64
        # to ComplexF32. The primal is an exact same-eltype copy, so its adjoint has to be
        # one too; `similar(dx)` also matches dx's memory kind.
        dx .+= copyto!(similar(dx), dy_cpu)
        return NoRData(), NoRData()
    end
    return CoDual(Array(primal(x)), dy_cpu), array_pb!!
end

# Rule for `Diagonal(v::CuMaybeComplexArray)` — construction of a GPU diagonal matrix.
# Diagonal is a thin wrapper: its only differentiable field is `.diag`.
# frule:    d(Diagonal(v)) = Diagonal(dv)
# pullback: dv += diag(dD)  (i.e. extract the diagonal from the output cotangent)
# Vectors only: `Diagonal(A::CuMatrix)` is `Diagonal(diag(A))`, whose `.diag` is a length-n
# vector, while these rules put the argument's own tangent in that slot and the pullback
# returns nothing on the assumption `.diag === v`. The matrix spelling decomposes through
# `diag`, which carries the gradient back to the right elements.
@is_primitive(MinimalCtx, Tuple{Type{<:Diagonal},CuMaybeComplexVec})
# Diagonal is a non-mutable struct; per the structural lift, its V is
# `ImmutableDual{@NamedTuple{diag::Vdiag}}` where `Vdiag` is the input's
# NDualArray V.
function frule!!(
    ::Lifted{<:Type{<:Diagonal},Nw}, v::Lifted{<:CuMaybeComplexVec,Nw,Vdiag}
) where {Nw,Vdiag<:NDualArray}
    y = Diagonal(primal(v))
    return Lifted{typeof(y),Nw}(y, ImmutableDual((; diag=tangent(v))))
end
function rrule!!(::CoDual{<:Type{<:Diagonal}}, v::CoDual{<:CuMaybeComplexVec})
    pv, dv = arrayify(v)
    # `Diagonal(v).diag === v`, so the aliasing invariant — primal(a) === primal(b) implies
    # fdata(a) === fdata(b) — requires the output's `.diag` fdata to *be* v's.  Accumulating a
    # separate zero(pv) into dv at the end of the pullback instead credits v with the
    # wrapper's cotangent as of the wrong moment, which a mutation of v in between makes
    # visible.  Sharing the buffer leaves the pullback nothing to do: downstream rules
    # accumulate into dv directly, and the frule already aliases the same way.
    # fdata_type(Diagonal{T, CuArray{T,1}}) = FData{(; diag::CuArray{T,1})}
    diagonal_pb!!(::NoRData) = (NoRData(), NoRData())
    return CoDual(Diagonal(pv), FData((; diag=dv))), diagonal_pb!!
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
# Cost: one fused GPU kernel evaluating f with one NDual slot per real operand and two
# per complex operand. Wider elements increase per-thread arithmetic and register use.
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
function frule!!(::Lifted{typeof(_gpu_broadcast_dual)}, ::Vararg{Lifted})
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

@inline function _gpu_leaf_slot_meta(pa, offset)
    dof = Nfwd._nfwd_input_dof(_gpu_rep_element(pa))
    return (; dof, slot1=offset + 1, slot2=offset + 2, is_scalar=pa isa CuFloatOrComplex)
end

@inline function _gpu_decode_ndual_output(
    ::Val{:broadcast}, out, ; extract_primal::Bool=true
)
    is_diff = Nfwd._nfwd_dual_has_partials(eltype(out))
    primal_out = if extract_primal
        is_diff ? broadcast(Nfwd._nfwd_dual_value, out) : out
    else
        nothing
    end
    return (; is_diff, primal_out)
end

@inline function _gpu_write_broadcast_primal!(dest, out, is_diff::Bool)
    if is_diff
        broadcast!(Nfwd._nfwd_dual_value, dest, out)
    elseif out isa AbstractArray
        copyto!(dest, out)
    else
        # `x .= 1` seeds no dual, so the kernel hands back the scalar itself; `copyto!`
        # would iterate it and set one element by scalar indexing.
        fill!(dest, out)
    end
    return dest
end

@inline function _gpu_decode_ndual_output(
    ::Val{:sum}, out, flat_pargs; extract_partials::Bool=false, pkw=NamedTuple()
)
    decoded = _gpu_decode_ndual_meta(out, flat_pargs; extract_partials)
    # Differentiable output: reduce the NDual `.value`s with a matching-typed `init`. Non-differentiable
    # output (a non-`NDual` element type, e.g. a `Bool` from a predicate `f`): reduce the array
    # directly and let the result type promote naturally (`sum(::Bool array)::Int`). Forcing the
    # NDual-derived `init` there mismatched the promoted accumulator and crashed the GPU reduction.
    # `pkw` carries the caller's reduction keywords, `dims` among them.
    primal_out = if decoded.is_diff
        reduction_kw = merge((; init=zero(Nfwd._nfwd_dual_primal_type(eltype(out)))), pkw)
        sum(Nfwd._nfwd_dual_value, out; reduction_kw...)
    else
        sum(out; pkw...)
    end
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
# `map(>(0.5f0), x)` reaches the kernel as a `Fix2` carrying its threshold. The comparison
# still decides the result, which is a Bool, so the capture's derivative is genuinely zero and
# the state guard below has nothing to protect.
@inline _gpu_bcast_has_nondiff_result(f::Base.Fix2) = _gpu_bcast_has_nondiff_result(f.f)
@inline _gpu_bcast_has_nondiff_result(::Any) = false
@inline _gpu_is_simple_cast_broadcast(::Any) = false
@inline function _gpu_is_simple_cast_broadcast(bc::Broadcasted)
    return bc.f isa Type{<:CuFloatOrComplex} &&
           length(bc.args) == 1 &&
           !(first(bc.args) isa Broadcasted)
end

# `Float64.(Float32.(x))` — a cast whose argument is itself a cast. `_premat_nondiff_args`
# materializes the inner one first, which makes the outer a simple cast and collapses the
# whole chain to a plain leaf, so the tangent that has to be reattached belongs to the array
# at the bottom rather than to the outer cast's immediate argument.
@inline _gpu_is_cast_chain(::Any) = false
@inline function _gpu_is_cast_chain(bc::Broadcasted)
    bc.f isa Type{<:CuFloatOrComplex} || return false
    length(bc.args) == 1 || return false
    a = first(bc.args)
    return !(a isa Broadcasted) || _gpu_is_cast_chain(a)
end
@inline _gpu_cast_chain_leaf(a, td) = (a, td)
@inline _gpu_cast_chain_leaf(bc::Broadcasted, td) = _gpu_cast_chain_leaf(
    first(bc.args), first(_fields(td).args)
)

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
# A non-contiguous view stays a SubArray (a contiguous one collapses to a plain CuArray);
# `arrayify` rebuilds its tangent as a SubArray over the parent's tangent, same indices.
@inline _leaf_effective_tangent(pa::SubArray{P,N,A}, t) where {P<:CuFloatOrComplex,N,A<:CuMaybeComplexArray} = arrayify(
    pa, t
)[2]
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
# A real->complex cast leaf gets a complex contribution back; project before narrowing, or
# the eltype conversion is Float32(::ComplexF32).  Widening casts (Float64.(x32)) are real
# throughout and unaffected.
@inline _gpu_cast_back_like(pa::AbstractArray, contrib) = eltype(pa).(
    _project_cotangent(pa, contrib)
)
@inline _gpu_cast_back_like(pa::CuFloatOrComplex, contrib) = convert(
    typeof(pa), _project_cotangent(pa, contrib)
)

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
@inline function _leaf_accum_fdata!(
    pa::SubArray{P,N,A}, fd, contrib
) where {P<:CuFloatOrComplex,N,A<:CuMaybeComplexArray}
    _, dpa = arrayify(pa, fd)
    dpa .+= _unbroadcast(contrib, size(pa))
    return dpa
end
@inline function _leaf_accum_fdata!(_, diff::_GpuBroadcastCastDiff, contrib)
    return _leaf_accum_fdata!(
        diff.primal_arg, diff.diff_arg, _gpu_cast_back_like(diff.primal_arg, contrib)
    )
end
@inline _leaf_accum_fdata!(_, _, _) = nothing  # non-differentiable

# A same-shaped broadcast contribution can be accumulated without first materializing a
# temporary array. Return whether the leaf supports this fused path.
@inline function _leaf_accum_broadcast!(
    ::CuMaybeComplexArray, fd::CuArray, contrib::Broadcasted
)
    fd .+= contrib
    return true
end
@inline function _leaf_accum_broadcast!(
    pa::SubArray{P,N,A}, fd, contrib::Broadcasted
) where {P<:CuFloatOrComplex,N,A<:CuMaybeComplexArray}
    _, dpa = arrayify(pa, fd)
    dpa .+= contrib
    return true
end
@inline _leaf_accum_broadcast!(_, _, _) = false

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
    # Casting is the identity up to rounding, so the whole chain contributes the bottom
    # leaf's tangent under the outermost cast; the value itself comes from the prepared
    # (already fully cast) leaf.
    p, t = _gpu_cast_chain_leaf(first(bc.args), first(_fields(td).args))
    return _gpu_broadcast_cast_diff(bc.f, p, t)
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
        elseif _gpu_is_cast_chain(a1_primal)
            _gpu_cast_diff_arg(a1_primal, td1)
        else
            NoTangent()
        end
        return (a1_prepared, rest_ps...), (diff, rest_ts...)
    else
        return (a1_prepared, rest_ps...), (td1, rest_ts...)
    end
end

# The functions are checked before flattening, so the type named in the error is the one the
# caller wrote rather than a Broadcast-internal composition wrapper.
_check_gpu_bcast_captures(::Tuple{}) = nothing
function _check_gpu_bcast_captures(args::Tuple)
    a = first(args)
    if a isa Broadcasted
        _check_gpu_bcast_captures(a)
    elseif !_gpu_threads_leaf(a)
        _throw_gpu_unthreaded(typeof(a), "broadcasting", "a differentiable argument")
    end
    return _check_gpu_bcast_captures(Base.tail(args))
end
function _check_gpu_bcast_captures(bc::Broadcasted)
    # A function whose result carries no derivative cannot silently zero one, so its own
    # state is not the hazard this guard exists for.
    _gpu_bcast_has_nondiff_result(bc.f) || _check_gpu_captured_state(bc.f, "broadcasting")
    return _check_gpu_bcast_captures(bc.args)
end

# A `Type` used as a broadcast function is a `DataType`, which is not a bitstype and so
# cannot be captured by a GPU kernel. `flatten` either hands it over raw, which fails to
# compile, or folds it into a closure whose element type inference is version-dependent —
# Julia 1.10 gives up and infers `Any` where 1.12 does not. Swapping in a singleton that
# carries the target type as a parameter removes the question: it is a bitstype either way,
# and it composes into whatever `flatten` builds. The tree is rebuilt before flattening and
# only the functions change, so the leaves walk still pairs it with the original arg by arg.
struct _CastTo{T} end
@inline (::_CastTo{T})(x) where {T} = T(x)
# Converting a dual has no method of its own, so a cast reaching the kernel over live
# partials infers `Union{}` and the launch is refused. The cast is the identity up to
# rounding, so it applies to the partials exactly as it does to the value.
@inline function (::_CastTo{T})(x::Nfwd.NDual{V,N}) where {T<:Real,V,N}
    return Nfwd.NDual{T,N}(T(x.value), map(T, x.partials))
end
# A complex leaf reaches the kernel as `Complex{<:NDual}`, one dual per degree of freedom
# rather than one dual holding a complex, so a complex target converts the two parts. Casting
# a real leaf to a complex one leaves its derivative wholly in the real part.
@inline (::_CastTo{T})(x::Complex{<:Nfwd.NDual}) where {T<:Complex} = Complex(
    _CastTo{real(T)}()(real(x)), _CastTo{real(T)}()(imag(x))
)
@inline function (::_CastTo{T})(x::Nfwd.NDual) where {T<:Complex}
    re = _CastTo{real(T)}()(x)
    return Complex(re, zero(re))
end
# Only the outermost function needs this. A cast nested inside the tree is materialized by
# `_premat_nondiff_args` before the kernel ever sees it; the one at the top is the only one
# that survives, because nothing materializes the tree's own root.
function _desugar_casts(bc::Broadcasted{S}) where {S}
    bc.f isa Type ? Broadcasted{S}(_CastTo{bc.f}(), bc.args, bc.axes) : bc
end

function _prepare_gpu_broadcast(bc_primal, tangent_or_fdata)
    _check_gpu_bcast_captures(bc_primal)
    bc_prepared = _premat_nondiff_args(bc_primal, tangent_or_fdata)
    flat_bc = Base.Broadcast.flatten(_desugar_casts(bc_prepared))
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

function _gpu_collect_scalar_map_args(args::Tuple, scalar_index::Ref{Int})
    return ntuple(length(args)) do i
        a = args[i]
        if a isa Broadcasted
            _gpu_collect_scalar_map_args(a.args, scalar_index)
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
            # Fuse the (lane-independent) partial extraction with the seed multiply and the
            # accumulate: a dot-call stays lazy, so `dy .+=` runs one in-place kernel with no
            # per-lane intermediate array (an eager `broadcast(f, dual_out)` would allocate one).
            if meta.dof == 1
                dy .+= Nfwd._nfwd_dual_partial.(dual_out, meta.slot1) .* t_eff
            elseif meta.dof == 2
                dy .+= Nfwd._nfwd_dual_partial.(dual_out, meta.slot1) .* real.(t_eff)
                dy .+= Nfwd._nfwd_dual_partial.(dual_out, meta.slot2) .* imag.(t_eff)
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
            # Fuse map into the reduction: `mapreduce` over the two arrays computes and sums in one
            # pass, so no per-lane intermediate array materialises (an eager `broadcast` would).
            if meta.dof == 1
                dy += mapreduce(
                    (o, tt) -> Nfwd._nfwd_dual_partial(o, meta.slot1) * tt,
                    +,
                    out,
                    t_eff,
                )
            elseif meta.dof == 2
                dy += mapreduce(
                    (o, tt) ->
                        Nfwd._nfwd_dual_partial(o, meta.slot1) * real(tt) +
                        Nfwd._nfwd_dual_partial(o, meta.slot2) * imag(tt),
                    +,
                    out,
                    t_eff,
                )
            end
        end,
    )
    return dy
end

# `dims` reductions cannot use the fused accumulator above: the map has to materialise before
# a dimensional `sum` can slice it. Single leaf, which is all a dimensional reduction admits.
function _gpu_reduced_jvp(out, px, dx, y, dims)
    meta = _gpu_leaf_slot_meta(px, 0)
    if meta.dof == 1
        return sum(
            broadcast((o, t) -> Nfwd._nfwd_dual_partial(o, meta.slot1) * t, out, dx);
            dims,
            init=zero(eltype(y)),
        )
    elseif meta.dof == 2
        return sum(
            broadcast(
                (o, t) ->
                    Nfwd._nfwd_dual_partial(o, meta.slot1) * real(t) +
                    Nfwd._nfwd_dual_partial(o, meta.slot2) * imag(t),
                out,
                dx,
            );
            dims,
            init=zero(eltype(y)),
        )
    end
    return zero_tangent(y)
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
    flat_pargs, flat_fdatas, dual_out, dy_out, bc_primal, scalar_map, scalar_count
)
    scalar_grads = isnothing(scalar_map) ? nothing : Vector{Any}(undef, scalar_count)
    scalar_index = 1
    offset = 0
    for (pa, fd) in zip(flat_pargs, flat_fdatas)
        meta = _gpu_leaf_slot_meta(pa, offset)
        if meta.dof == 1
            contrib = Base.broadcasted(
                (o, d) -> real(conj(d) * Nfwd._nfwd_dual_partial(o, meta.slot1)),
                dual_out,
                dy_out,
            )
            if meta.is_scalar
                # The partials carry whatever type the broadcast promoted to, which is the
                # array's eltype whenever it is the wider one, but a leaf's rdata follows
                # the leaf — the same `oftype` the BLAS scalars take.
                (scalar_grads::Vector{Any})[scalar_index] = oftype(pa, sum(contrib))
                scalar_index += 1
            else
                if !(size(pa) == size(dy_out) && _leaf_accum_broadcast!(pa, fd, contrib))
                    _leaf_accum_fdata!(pa, fd, Base.Broadcast.materialize(contrib))
                end
            end
        elseif meta.dof == 2
            contrib = Base.broadcasted(
                (o, d) -> complex(
                    real(conj(d) * Nfwd._nfwd_dual_partial(o, meta.slot1)),
                    real(conj(d) * Nfwd._nfwd_dual_partial(o, meta.slot2)),
                ),
                dual_out,
                dy_out,
            )
            if meta.is_scalar
                # The partials carry whatever type the broadcast promoted to, which is the
                # array's eltype whenever it is the wider one, but a leaf's rdata follows
                # the leaf — the same `oftype` the BLAS scalars take.
                (scalar_grads::Vector{Any})[scalar_index] = oftype(pa, sum(contrib))
                scalar_index += 1
            else
                if !(size(pa) == size(dy_out) && _leaf_accum_broadcast!(pa, fd, contrib))
                    _leaf_accum_fdata!(pa, fd, Base.Broadcast.materialize(contrib))
                end
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

function _gpu_reduced_pullback!(px, dx, dual_out, dy)
    meta = _gpu_leaf_slot_meta(px, 0)
    if meta.dof == 1
        contrib = broadcast(
            (o, d) -> real(conj(d) * Nfwd._nfwd_dual_partial(o, meta.slot1)), dual_out, dy
        )
        _leaf_accum_fdata!(px, dx, contrib)
    elseif meta.dof == 2
        contrib = broadcast(
            (o, d) -> complex(
                real(conj(d) * Nfwd._nfwd_dual_partial(o, meta.slot1)),
                real(conj(d) * Nfwd._nfwd_dual_partial(o, meta.slot2)),
            ),
            dual_out,
            dy,
        )
        _leaf_accum_fdata!(px, dx, contrib)
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

# Per-lane tangent extraction for the canonical forward V shapes that appear as
# `Broadcasted.args` entries, used by the `materialize` / `materialize!` frules
# below to reconstruct a legacy reverse-mode-shaped Broadcasted tangent (which
# the existing `_prepare_gpu_broadcast` / `_gpu_bcast_leaves` helpers consume).
# `lane` selects the chunk slot, so each lane reuses the single dual-broadcast
# kernel for an independent JVP.
@inline _bc_tangent(::Union{Mooncake.NoDual,Mooncake.NoTangent}, _, _) = NoTangent()
@inline _bc_tangent(::Tuple{<:Union{Mooncake.NoDual,Mooncake.NoTangent}}, _, _) = NoTangent()
@inline _bc_tangent(v::Nfwd.NDual, _, lane) = v.partials[lane]
@inline _bc_tangent(v::Nfwd.NDualArray, _, lane) = Nfwd.tangent_view(v, lane)
@inline function _bc_tangent(v::Complex{Nfwd.NDual{R,N}}, _, lane) where {R,N}
    return Complex(real(v).partials[lane], imag(v).partials[lane])
end
@inline function _bc_tangent(v::ImmutableDual, p::Broadcasted, lane)
    nt = v.value
    targs = ntuple(length(p.args)) do i
        _bc_tangent(nt.args[i], p.args[i], lane)
    end
    return Tangent((; style=NoTangent(), f=NoTangent(), args=targs, axes=NoTangent()))
end
# Wrapper-arg fall-throughs: Transpose/Adjoint primals with parent NDualArray V.
@inline function _bc_tangent(v::ImmutableDual, p::Union{Transpose,Adjoint}, lane)
    parent_tangent = _bc_tangent(v.value.parent, parent(p), lane)
    return Tangent((; parent=parent_tangent))
end
# Non-contiguous SubArray leaf (a contiguous view collapses to a plain CuArray). `copy`
# materialises the parent's lane partial as a plain device array — `_leaf_effective_tangent`
# feeds `.parent` to `arrayify(::SubArray, …)`, which requires a `CuArray` there and re-applies
# the primal's indices; the lazy block-row view is a `SubArray`, which that `arrayify` rejects.
@inline function _bc_tangent(v::ImmutableDual, p::SubArray, lane)
    parent_tangent = copy(_bc_tangent(v.value.parent, parent(p), lane))
    return Tangent((; parent=parent_tangent))
end
# Generic Ref/struct primal with ImmutableDual or MutableDual V — the Broadcast `args`
# may include e.g. `RefValue{F}` (non-diff inner), which carries no tangent. A struct V
# with any differentiable field reaching this fall-through has no lane extraction, so
# fail loudly rather than silently zeroing its derivative.
@inline function _bc_tangent(v::Union{ImmutableDual,MutableDual}, p, _)
    _bc_tangent_free(v) || throw(
        ArgumentError(
            "broadcast argument of type $(typeof(p)) carries a struct V with " *
            "differentiable fields ($(typeof(v))); per-lane extraction is not " *
            "implemented for this shape, so its derivative would be silently dropped.",
        ),
    )
    return NoTangent()
end
_bc_tangent_free(::Mooncake.NoDual) = true
_bc_tangent_free(v::Union{ImmutableDual,MutableDual}) = all(_bc_tangent_free, v.value)
_bc_tangent_free(v::Union{Tuple,NamedTuple}) = all(_bc_tangent_free, v)
function _bc_tangent_free(v::Mooncake.PossiblyUninitTangent)
    return !Mooncake.is_init(v) || _bc_tangent_free(Mooncake.val(v))
end
_bc_tangent_free(::Any) = false

function frule!!(
    ::Lifted{typeof(Base.Broadcast.materialize),Nw},
    bc::Lifted{<:Broadcasted{<:CuArrayStyle},Nw},
) where {Nw}
    bc_primal = primal(bc)
    # Refuse an unthreadable leaf before `_bc_tangent` reaches it: as an argument to
    # `_prepare_gpu_broadcast` it runs before that function's own guard, and a leaf shape it
    # has no method for would report a MethodError in place of the intended message.
    _check_gpu_bcast_captures(bc_primal)
    bc_V = tangent(bc)
    # `out`, `decoded`, `flat_bc`, and `flat_pargs` are primal-only (lane-independent),
    # so run the single dual-broadcast kernel once and reuse it for every lane's JVP
    # (including lane 1's flattened tangents, captured here).
    bc_prepared, flat_bc, flat_pargs, flat_ts_1 = _prepare_gpu_broadcast(
        bc_primal, _bc_tangent(bc_V, bc_primal, 1)
    )
    out = _gpu_broadcast_dual(flat_bc.f, flat_pargs...)
    decoded = _gpu_decode_ndual_output(Val(:broadcast), out)

    # `is_diff` says the kernel's element type carried no partials, which is not the same as
    # the output being non-differentiable: a float array built only from index arrays, say
    # `cnt ./ 2`, has a forward V even though nothing flows into it. Bool output from a
    # comparison is the case this branch was written for, and there the V really is `NoDual`.
    if !decoded.is_diff
        return Lifted{typeof(out),Nw}(out, Mooncake.zero_dual(Val(Nw), out))
    end
    dy_lanes = ntuple(Val(Nw)) do k
        flat_ts_k = if k == 1
            flat_ts_1
        else
            _gpu_bcast_leaves(bc_prepared, bc_primal, _bc_tangent(bc_V, bc_primal, k))[2]
        end
        _gpu_accumulate_jvp!(zero(decoded.primal_out), flat_pargs, flat_ts_k, out)
    end
    A = typeof(decoded.primal_out)
    T = eltype(A)
    D = ndims(A)
    return Lifted{A,Nw}(
        decoded.primal_out, NDualArray{T,Nw,D,A}(decoded.primal_out, dy_lanes)
    )
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
    decoded = _gpu_decode_ndual_output(Val(:broadcast), out)

    # As in the frule: a float output with no differentiable leaf still needs an fdata of
    # its declared type, or the caller is handed a NoFData where it expects a CuArray.
    if !decoded.is_diff
        fd = tangent_type(typeof(out)) === NoTangent ? NoFData() : zero_tangent(out)
        return CoDual(out, fd), NoPullback(mat_fn, bc)
    end

    dy_out = zero(decoded.primal_out)  # accumulated into by the downstream reverse pass

    function materialize_pb!!(::NoRData)
        r_bc = _gpu_accum_pullback!(
            flat_pargs, flat_fdatas, out, dy_out, bc_primal, scalar_map, scalar_count
        )
        return NoRData(), r_bc
    end

    return CoDual(decoded.primal_out, dy_out), materialize_pb!!
end

# Julia 1.10 can expose the `copy` inside `materialize` as the call boundary.  Claim that
# equivalent boundary rather than tracing through GPUArrays' KernelAbstractions launch.
@static if VERSION < v"1.11-"
    @is_primitive MinimalCtx Tuple{typeof(copy),<:Broadcasted{<:CuArrayStyle}}
    function frule!!(
        ::Lifted{typeof(copy),Nw}, bc::Lifted{<:Broadcasted{<:CuArrayStyle},Nw}
    ) where {Nw}
        return frule!!(zero_lifted(Val(Nw), Base.Broadcast.materialize), bc)
    end
    function rrule!!(::CoDual{typeof(copy)}, bc::CoDual{<:Broadcasted{<:CuArrayStyle}})
        return rrule!!(CoDual(Base.Broadcast.materialize, NoFData()), bc)
    end
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
#
# `x .= 0f0` is claimed here too, through `DefaultArrayStyle{0}`: the two-argument
# materialize! is handed the right-hand side's style alone, and the destination is combined
# in only inside the three-argument method below this claim, so a scalar right-hand side
# never reaches `CuArrayStyle`. Its args are all scalars, so the kernel hands back a scalar
# NDual, which the primal write and the JVP accumulation each broadcast over `dest`. A
# `DefaultArrayStyle{N}` for N > 0 means host arrays are involved and stays out.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(Base.Broadcast.materialize!),P,<:Broadcasted{<:_GpuMaterializeStyle}
    } where {P<:CuMaybeWrappedArray},
)
function frule!!(
    ::Lifted{typeof(Base.Broadcast.materialize!),Nw},
    dest::Lifted{P,Nw},
    bc::Lifted{<:Broadcasted{<:_GpuMaterializeStyle},Nw},
) where {P<:CuMaybeWrappedArray,Nw}
    bc_primal = primal(bc)
    _check_gpu_bcast_captures(bc_primal)
    bc_V = tangent(bc)
    # Primal-only prep + single kernel, reused across lanes (see the `materialize` frule).
    bc_prepared, flat_bc, flat_pargs, flat_ts_1 = _prepare_gpu_broadcast(
        bc_primal, _bc_tangent(bc_V, bc_primal, 1)
    )
    dual_out = _gpu_broadcast_dual(flat_bc.f, flat_pargs...)
    # `arrayify` covers a bare CuArray and the Adjoint/Transpose/SubArray wrappers the claim
    # admits alike; each lane view writes through to the slot's own partials.
    pout, dest_partials = arrayify(dest)
    decoded = _gpu_decode_ndual_output(
        Val(:broadcast), dual_out, flat_pargs; extract_primal=false
    )
    _gpu_write_broadcast_primal!(pout, dual_out, decoded.is_diff)
    if !decoded.is_diff
        for lane in 1:Nw
            fill!(dest_partials[lane], zero(eltype(dest_partials[lane])))
        end
        return dest
    end
    # Aliasing-safe accumulator reused across lanes: `dest` may appear in `bc.args`, so we
    # accumulate into a buffer distinct from `dest_partials[lane]` and copy in only at lane end.
    tmp = similar(pout)
    for lane in 1:Nw
        flat_ts = if lane == 1
            flat_ts_1
        else
            _gpu_bcast_leaves(bc_prepared, bc_primal, _bc_tangent(bc_V, bc_primal, lane))[2]
        end
        fill!(tmp, zero(eltype(tmp)))
        _gpu_accumulate_jvp!(tmp, flat_pargs, flat_ts, dual_out)
        copyto!(dest_partials[lane], tmp)
    end
    return dest
end
function rrule!!(
    ::CoDual{typeof(Base.Broadcast.materialize!),NoFData},
    dest::CoDual{P},
    bc::CoDual{<:Broadcasted{<:_GpuMaterializeStyle}},
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
    decoded = _gpu_decode_ndual_output(Val(:broadcast), dual_out; extract_primal=false)

    # Write primal result in-place into dest.
    _gpu_write_broadcast_primal!(pout, dual_out, decoded.is_diff)

    # Non-differentiable output (e.g. Bool arrays): no gradient to propagate.
    # Check eltype(dual_out) (NDual elements), NOT eltype(pout) (plain floats after
    # shared NDual-value extraction): eltype(pout) is always IEEEFloat for CuMaybeComplexArray.
    if !decoded.is_diff
        function materialize!_nodiff_pb!!(::NoRData)
            # dest was overwritten by a result that does not depend on its old value, so the
            # cotangent standing in dest's fdata belongs to that result and is consumed here
            # — there is nothing to redistribute it to.  Leaving it would hand it back to
            # dest's own history, as if the broadcast had passed dest through.  The
            # differentiable branch below consumes it the same way, after distributing it.
            fill!(dout, zero(eltype(dout)))
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
            flat_pargs, flat_fdatas, dual_out, g, bc_primal, scalar_map, scalar_count
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
function frule!!(
    ::Lifted{typeof(permutedims),Nw}, x::Lifted{<:CuMaybeWrappedArray}, perm::Lifted
) where {Nw}
    px, x_partials = arrayify(x)
    pperm = primal(perm)
    y = permutedims(px, pperm)
    y_partials = ntuple(k -> permutedims(x_partials[k], pperm), Val(Nw))
    Y = typeof(y)
    return Lifted{Y,Nw}(y, NDualArray{eltype(y),Nw,ndims(y),Y}(y, y_partials))
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
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple},
    ::Lifted{typeof(varm),Nw},
    x::Lifted{<:CuFloatArray,Nw,<:NDualArray},
    m::Lifted{<:CuFloatArray,Nw,<:NDualArray},
) where {Nw}
    pkw = primal(kw)
    px, x_partials = arrayify(x)
    pm, m_partials = arrayify(m)
    σ² = varm(px, pm; pkw...)
    # `dims` has no default in GPUArrays' method, so the call above has already thrown
    # unless it is present; `corrected` does default to true there.
    _raw_dims = pkw.dims
    corrected = get(pkw, :corrected, true)
    T = eltype(px)
    diff = px .- pm
    # As in the primal: λ-prescale before reducing (a raw product sum could overflow
    # Float16), with λ inverted in Float64 then converted. `one(T) / n` would convert
    # n to T first, turning n > 65504 into Inf16 and silently zeroing every Float16
    # gradient.
    if _raw_dims isa Colon
        λ = convert(T, inv(length(px) - Int(corrected)))
        dσ²_lanes = ntuple(
            k -> sum((2λ) .* diff .* (x_partials[k] .- m_partials[k])), Val(Nw)
        )
        return Lifted{typeof(σ²),Nw}(σ², _wrap_scalar_v_lanes(σ², dσ²_lanes))
    end
    # The denominator mirrors GPUArrays' _mean_denom: a repeated dim (dims=(1,1))
    # counts once, and `init=1` keeps an empty dims=() from throwing on the empty
    # prod. sum(...; dims) already deduplicates repeated dims on its own.
    _dims = _raw_dims isa Integer ? (_raw_dims,) : _raw_dims
    n = prod(d -> size(px, d), unique(_dims); init=1)
    λ = convert(T, inv(n - Int(corrected)))
    y_partials = ntuple(
        k -> sum((2λ) .* diff .* (x_partials[k] .- m_partials[k]); dims=_dims), Val(Nw)
    )
    Y = typeof(σ²)
    return Lifted{Y,Nw}(σ², NDualArray{eltype(σ²),Nw,ndims(σ²),Y}(σ², y_partials))
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
# derivative. Dividing each summand first would be better conditioned — it returns
# 46677.3 where summing first overflows Float16 to Inf — but only here, so the two
# backends would then disagree. At Float16 n > 65504 both orderings give NaN, since
# the summand overflows before the Inf16-promoted denominator can zero the quotient.
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
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple},
    ::Lifted{typeof(varm),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    m::Lifted{<:CuFloatOrComplex,Nw},
) where {Nw}
    pkw = primal(kw)
    px, x_partials = arrayify(x)
    pm = primal(m)
    # Statistics' scalar-m method has no dims kwarg, so a stray dims=1 throws here
    # exactly as it does without AD.
    σ² = varm(px, pm; pkw...)
    corrected = get(pkw, :corrected, true)
    dσ²_lanes = ntuple(
        k -> _varm_scalar_colon_tangent(px, x_partials[k], pm, tangent(m, k), corrected),
        Val(Nw),
    )
    return Lifted{typeof(σ²),Nw}(σ², _wrap_scalar_v_lanes(σ², dσ²_lanes))
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
    ::Lifted{typeof(varm),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
    m::Lifted{<:CuFloatOrComplex,Nw},
) where {Nw}
    px, x_partials = arrayify(x)
    pm = primal(m)
    σ² = varm(px, pm)
    dσ²_lanes = ntuple(
        k -> _varm_scalar_colon_tangent(px, x_partials[k], pm, tangent(m, k), true), Val(Nw)
    )
    return Lifted{typeof(σ²),Nw}(σ², _wrap_scalar_v_lanes(σ², dσ²_lanes))
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
    ::Lifted{typeof(Core.kwcall),Nw},
    kw::Lifted{<:NamedTuple},
    ::Lifted{typeof(mean),Nw},
    x::Lifted{<:CuMaybeComplexArray,Nw,<:NDualArray},
) where {Nw}
    pkw = primal(kw)
    px, x_partials = arrayify(x)
    μ = mean(px; pkw...)
    raw_dims = get(pkw, :dims, :)
    if raw_dims isa Colon
        # The primal's Colon branch is sum(A)/length(A); dividing after summing keeps
        # the gradient identical to the traced bare-mean decomposition (at Float16
        # n > 65504 both give exactly zero, where a λ-prescale would not). n == 0:
        # the primal is a constant NaN, so the JVP is the zero map, consistent with
        # the rrule — 0/0 would manufacture a NaN tangent.
        n = length(px)
        if n == 0 || Nw == 1
            dμ_lanes = ntuple(k -> n == 0 ? zero(μ) : sum(x_partials[k]) / n, Val(Nw))
        else
            # sum(xpₖ)/n batched: sum(xpₖ) = onesᵀ·xpₖ via gemv_batched! (ones shared, no gather; 'T'
            # transposes the n×1 ones column into the 1×n contracting row — ones is real and gemv never
            # conjugates its vector operand, so the partials pass through unchanged), one readback.
            T = eltype(px)
            onev = reshape(fill!(similar(px, T, n), one(T)), :, 1)
            xvs = [reshape(xp, :) for xp in x_partials]
            outs = [similar(px, T, 1) for _ in 1:Nw]
            cuBLAS.gemv_batched!('T', one(T), fill(onev, Nw), xvs, zero(T), outs)
            host = Array(reduce(vcat, outs))
            dμ_lanes = ntuple(k -> host[k] / n, Val(Nw))
        end
        return Lifted{typeof(μ),Nw}(μ, _wrap_scalar_v_lanes(μ, dμ_lanes))
    end
    _dims = raw_dims isa Integer ? (raw_dims,) : raw_dims
    n = prod(d -> size(px, d), unique(_dims); init=1)
    λ = eltype(px)(inv(n))
    y_partials = ntuple(k -> sum(λ .* x_partials[k]; dims=_dims), Val(Nw))
    Y = typeof(μ)
    return Lifted{Y,Nw}(μ, NDualArray{eltype(μ),Nw,ndims(μ),Y}(μ, y_partials))
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
