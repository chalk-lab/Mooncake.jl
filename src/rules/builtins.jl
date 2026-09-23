
#
# Core.Builtin -- these are "primitive" functions which must have rrules because no IR
# is available.
#
# There is a finite number of these functions.
# Any built-ins which don't have rules defined are left as comments with their names
# in this block of code
# As of version 1.9.2 of Julia, there are exactly 139 examples of `Core.Builtin`s.
#

@is_primitive MinimalCtx Tuple{Core.Builtin,Vararg}

struct MissingRuleForBuiltinException <: Exception
    msg::String
end

function rrule!!(f::CoDual{<:Core.Builtin}, args...)
    T_args = map(typeof ∘ primal, args)
    throw(
        MissingRuleForBuiltinException(
            "All built-in functions are primitives by default, as they do not have any Julia " *
            "code to recurse into. This means that they must all have methods of `rrule!!` " *
            "written for them by hand. " *
            "The built-in $(primal(f)) has been called with arguments with types $T_args, " *
            "but there is no specialised method of `rrule!!` for this built-in and these " *
            "types. In order to fix this problem, you will either need to modify your code " *
            "to avoid hitting this built-in function, or implement a method of `rrule!!` " *
            "which is specialised to this case. " *
            "Either way, please consider commenting on " *
            "https://github.com/chalk-lab/Mooncake.jl/issues/208/ so that the issue can be " *
            "fixed more widely.\n" *
            "For reproducibility, note that the full signature is:\n" *
            "$(typeof((f, args...)))",
        ),
    )
end

function Base.showerror(io::IO, err::MissingRuleForBuiltinException)
    return _print_boxed_error(io, split(err.msg, '\n'))
end

"""
    module IntrinsicsWrappers

The purpose of this `module` is to associate to each function in `Core.Intrinsics` a regular
Julia function.

To understand the rationale for this observe that, unlike regular Julia functions, each
`Core.IntrinsicFunction` in `Core.Intrinsics` does _not_ have its own type. Rather, they
are instances of `Core.IntrinsicFunction`. To see this, observe that
```jldoctest
julia> typeof(Core.Intrinsics.add_float)
Core.IntrinsicFunction

julia> typeof(Core.Intrinsics.sub_float)
Core.IntrinsicFunction
```

While we could simply write a rule for `Core.IntrinsicFunction`, this would (naively) lead
to a large list of conditionals of the form
```julia
if f === Core.Intrinsics.add_float
    # return add_float and its pullback
elseif f === Core.Intrinsics.sub_float
    # return add_float and its pullback
elseif
    ...
end
```
which has the potential to cause quite substantial type instabilities.
(This might not be true anymore -- see extended help for more context).

Instead, we map each `Core.IntrinsicFunction` to one of the regular Julia functions in
`Mooncake.IntrinsicsWrappers`, to which we can dispatch in the usual way.

# Extended Help

It is possible that owing to improvements in constant propagation in the Julia compiler in
version 1.10, we actually _could_ get away with just writing a single method of `rrule!!` to
handle all intrinsics, so this dispatch-based mechanism might be unnecessary. Someone should
investigate this. Discussed at https://github.com/chalk-lab/Mooncake.jl/issues/387 .
"""
module IntrinsicsWrappers

using Base: IEEEFloat
using Core: Intrinsics
using Mooncake
import ..Mooncake:
    rrule!!,
    frule!!,
    CoDual,
    VoidPtrTangent,
    Lifted,
    NDual,
    NoDual,
    primal,
    tangent,
    zero_tangent,
    NoPullback,
    tangent_type,
    increment!!,
    @is_primitive,
    MinimalCtx,
    _is_primitive,
    NoFData,
    zero_rdata,
    NoRData,
    tuple_map,
    fdata,
    NoRData,
    rdata,
    increment_rdata!!,
    zero_fcodual,
    zero_dual,
    NoTangent,
    Mode,
    extract,
    nan_tangent_guard,
    NDualArray,
    NDualBlock,
    NDualEltype,
    _scalar_ndual

struct MissingIntrinsicWrapperException <: Exception
    msg::String
end

function translate(f)
    msg =
        "Unable to translate the intrinsic $f into a regular Julia function. " *
        "Please see github.com/chalk-lab/Mooncake.jl/issues/208 for more discussion."
    throw(MissingIntrinsicWrapperException(msg))
end

# Note: performance is not considered _at_ _all_ in this implementation.
function rrule!!(f::CoDual{<:Core.IntrinsicFunction}, args...)
    return rrule!!(CoDual(translate(Val(primal(f))), tangent(f)), args...)
end

macro intrinsic(name)
    expr = quote
        $name(x...) = Intrinsics.$name(x...)
        function _is_primitive(
            ::Type{MinimalCtx}, ::Type{<:Mode}, ::Type{<:Tuple{typeof($name),Vararg}}
        )
            return true
        end
        translate(::Val{Intrinsics.$name}) = $name
    end
    return esc(expr)
end

macro inactive_intrinsic(name)
    expr = quote
        $name(x...) = Intrinsics.$name(x...)
        function _is_primitive(
            ::Type{MinimalCtx}, ::Type{<:Mode}, ::Type{<:Tuple{typeof($name),Vararg}}
        )
            return true
        end
        translate(::Val{Intrinsics.$name}) = $name
        function rrule!!(f::CoDual{typeof($name)}, args::Vararg{Any,N}) where {N}
            return Mooncake.zero_adjoint(f, args...)
        end
        function frule!!(
            f::Mooncake.Lifted{typeof($name)}, args::Vararg{Mooncake.Lifted,M}
        ) where {M}
            return Mooncake.zero_derivative(f, args...)
        end
    end
    return esc(expr)
end

@intrinsic abs_float
function frule!!(
    ::Lifted{typeof(abs_float),N}, x::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(abs_float(primal(x)), sign(primal(x)) * tangent(x))
end
function rrule!!(::CoDual{typeof(abs_float)}, x)
    abs_float_pullback!!(dy) = NoRData(), sign(primal(x)) * dy
    y = abs_float(primal(x))
    return CoDual(y, NoFData()), abs_float_pullback!!
end

@intrinsic add_float
function frule!!(
    ::Lifted{typeof(add_float),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(add_float(primal(a), primal(b)), tangent(a) + tangent(b))
end
function rrule!!(::CoDual{typeof(add_float)}, a, b)
    add_float_pb!!(c̄) = NoRData(), c̄, c̄
    c = add_float(primal(a), primal(b))
    return CoDual(c, NoFData()), add_float_pb!!
end

@intrinsic add_float_fast
function frule!!(
    ::Lifted{typeof(add_float_fast),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(add_float_fast(primal(a), primal(b)), tangent(a) + tangent(b))
end
function rrule!!(::CoDual{typeof(add_float_fast)}, a, b)
    add_float_fast_pb!!(c̄) = NoRData(), c̄, c̄
    c = add_float_fast(primal(a), primal(b))
    return CoDual(c, NoFData()), add_float_fast_pb!!
end

@inactive_intrinsic add_int

@intrinsic add_ptr
function rrule!!(::CoDual{typeof(add_ptr)}, a, b)
    throw(error("add_ptr intrinsic hit. This should never happen. Please open an issue"))
end

@inactive_intrinsic and_int
@inactive_intrinsic ashr_int

# unsafe_wrap() gives an array view for the memory pointed by p.
# Tangent propagation happens through memory aliasing rather than explicit
# computation in the pullback. Downstream rules write directly into 
# the tangent memory pointed to by tangent_arr.
@is_primitive MinimalCtx Tuple{typeof(unsafe_wrap),<:Type{<:Array},Ptr,Any}
# V for `Ptr{T}` is `NTuple{Nw, Ptr{T}}`; wrap each lane's per-lane Ptr
# into the corresponding lane of the canonical NDualArray V.
function frule!!(
    ::Lifted{typeof(unsafe_wrap),Nw},
    ::Lifted{<:Type{<:Array},Nw},
    p::Lifted{Ptr{T},Nw,NTuple{Nw,Ptr{T}}},
    dims::Lifted,
) where {Nw,T<:NDualEltype}
    _dims = primal(dims)
    primal_arr = unsafe_wrap(Array, primal(p), _dims)
    p_partials = tangent(p)
    D = ndims(primal_arr)
    # Block-backed pointers address contiguous lanes of one element-major column.
    # Re-wrap lane 1 as the block's flat parent to preserve tangent aliasing; packing
    # separate wraps into a fresh block would sever it. Reject the non-NULL
    # `uninit_*` placeholder first: it aliases the primal's own bytes.
    for k in 1:Nw
        _check_fwd_tangent_ptr_addressable(primal(p), p_partials[k])
    end
    all(k -> p_partials[k] == p_partials[1] + (k - 1) * sizeof(T), 1:Nw) || throw(
        ArgumentError(
            "Forward-mode `unsafe_wrap` requires the lifted pointer's lane pointers to be " *
            "the contiguous lanes of one element-major block column. Wrapping separated " *
            "per-lane buffers cannot preserve tangent aliasing.",
        ),
    )
    bd = (Nw, (_dims isa Integer ? (_dims,) : _dims)...)
    block = NDualBlock(unsafe_wrap(Array, p_partials[1], prod(bd)), bd)
    return Lifted{Array{T,D},Nw}(
        primal_arr, NDualArray{T,Nw,D,Array{T,D}}(primal_arr, block)
    )
end
# Pointer elements use `Array{NTuple{Nw,Ptr{R}},D}`, not NDualArray.
# Raw lanes have dense element strides, so only width one aliases their storage as V.
# Wider chunks require a pointer representation carrying the tangent stride.
function frule!!(
    ::Lifted{typeof(unsafe_wrap),Nw},
    ::Lifted{<:Type{<:Array},Nw},
    p::Lifted{Ptr{Ptr{R}},Nw,NTuple{Nw,Ptr{Ptr{R}}}},
    dims::Lifted,
) where {Nw,R<:NDualEltype}
    _dims = primal(dims)
    primal_arr = unsafe_wrap(Array, primal(p), _dims)
    p_partials = tangent(p)
    D = ndims(primal_arr)
    Nw == 1 || throw(
        ArgumentError(
            "Forward-mode `unsafe_wrap` of pointer elements cannot preserve tangent " *
            "aliasing at chunk width $Nw > 1. Use chunk width one or reverse mode.",
        ),
    )
    _check_fwd_tangent_ptr_addressable(primal(p), p_partials[1])
    vptr = Ptr{NTuple{Nw,Ptr{R}}}(p_partials[1])
    v = unsafe_wrap(Array, vptr, _dims)
    return Lifted{Array{Ptr{R},D},Nw}(primal_arr, v)
end

# Non-differentiable pointees use NoDual, but the wrapped array needs its canonical V.
# This fallback keeps method coverage aligned with the primitive's unrestricted Ptr.
function frule!!(
    ::Lifted{typeof(unsafe_wrap),Nw},
    ::Lifted{<:Type{<:Array},Nw},
    p::Lifted{Ptr{T},Nw,NoDual},
    dims::Lifted,
) where {Nw,T}
    # A manually constructed NoDual slot can still have a differentiable pointee.
    # Refuse it before wrapping an array without derivative storage.
    tangent_type(T) === NoTangent || throw(
        ArgumentError(
            "unsafe_wrap of a `Ptr{$T}` whose forward V is `NoDual` even though `$T` is " *
            "differentiable: the wrapped array would carry no tangent and its derivative " *
            "would be read from unrelated memory.",
        ),
    )
    arr = unsafe_wrap(Array, primal(p), primal(dims))
    return Lifted{typeof(arr),Nw}(arr, zero_dual(Val(Nw), arr))
end

# Other differentiable pointees have no supported array V. The primitive covers
# every Ptr, so refuse explicitly rather than falling through to a MethodError.
function frule!!(
    ::Lifted{typeof(unsafe_wrap),Nw},
    ::Lifted{<:Type{<:Array},Nw},
    ::Lifted{Ptr{T},Nw,<:NTuple{Nw,Ptr}},
    ::Lifted,
) where {Nw,T}
    throw(
        ArgumentError(
            "unsafe_wrap of a differentiable `Ptr{$T}` whose element is neither a scalar " *
            "float/complex nor a pointer-to-scalar: the array-of-duals element V is unsupported.",
        ),
    )
end

function rrule!!(
    ::CoDual{typeof(unsafe_wrap)},
    ::CoDual{<:Type{<:Array}},
    p::CoDual{<:Ptr{T}},
    dims::CoDual,
) where {T}
    # Check before wrapping: an Array over a NULL or placeholder tangent pointer
    # cannot be distinguished from valid storage by downstream consumers.
    _check_tangent_ptr(primal(p), tangent(p))
    primal_arr = unsafe_wrap(Array, primal(p), primal(dims))
    tangent_arr = unsafe_wrap(Array, tangent(p), primal(dims))
    function unsafe_wrap_pullback!!(::NoRData)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(primal_arr, tangent_arr), unsafe_wrap_pullback!!
end

# atomic_fence
# atomic_pointermodify
# atomic_pointerreplace

# Atomic analogue of `pointerref`/`pointerset` below; keep the pullbacks in sync.
@intrinsic atomic_pointerref
# As in pointerref, _scalar_ndual packs real or complex lanes into canonical V.
function frule!!(
    ::Lifted{typeof(atomic_pointerref),Nw},
    x::Lifted{Ptr{T},Nw,NTuple{Nw,Ptr{T}}},
    order::Lifted,
) where {Nw,T<:NDualEltype}
    _order = primal(order)
    a = atomic_pointerref(primal(x), _order)
    x_partials = tangent(x)
    @inbounds for lane in 1:Nw
        _check_fwd_tangent_ptr_addressable(primal(x), x_partials[lane])
    end
    da_lanes = ntuple(lane -> atomic_pointerref(x_partials[lane], _order), Val(Nw))
    return Lifted{T,Nw}(a, _scalar_ndual(a, da_lanes))
end
# Non-differentiable pointer (V === NoDual): the loaded value carries no derivative.
function frule!!(
    ::Lifted{typeof(atomic_pointerref),Nw}, x::Lifted{Ptr{T},Nw,NoDual}, order::Lifted
) where {Nw,T}
    _check_nodual_diff_ptr(T)
    a = atomic_pointerref(primal(x), primal(order))
    return Lifted{typeof(a),Nw}(a, NoDual())
end
# Incoherent lanes may load non-differentiable elements, but cannot recover a
# differentiable value from an interleaved object tangent (see pointerref).
function frule!!(
    ::Lifted{typeof(atomic_pointerref),Nw},
    x::Lifted{Ptr{T},Nw,<:NTuple{Nw,Ptr}},
    order::Lifted,
) where {Nw,T}
    tangent_type(T) === NoTangent || throw(
        ArgumentError(
            "Forward-mode AD cannot take a raw atomic load (`atomic_pointerref`) of a " *
            "differentiable `Ptr{$T}` whose per-lane tangent is not the canonical " *
            "`NTuple{$Nw,Ptr{$T}}` per-lane-partial shape. Use reverse mode, or hold the value " *
            "in a `Ref`/`Array` whose forward tangent keeps a parallel partials buffer.",
        ),
    )
    a = atomic_pointerref(primal(x), primal(order))
    return Lifted{typeof(a),Nw}(a, NoDual())
end
function rrule!!(::CoDual{typeof(atomic_pointerref)}, x, order)
    _x = primal(x)
    _order = primal(order)
    dx = tangent(x)
    _check_tangent_ptr(_x, dx)
    # Tangent bookkeeping uses :monotonic: a load-only primal ordering (e.g. :acquire) would
    # throw ConcurrencyViolationError if reused for the pullback's store.
    a = CoDual(atomic_pointerref(_x, _order), fdata(atomic_pointerref(dx, :monotonic)))
    if Mooncake.rdata_type(tangent_type(Mooncake._typeof(primal(a)))) == NoRData
        return a, NoPullback((NoRData(), NoRData(), NoRData()))
    else
        function atomic_pointerref_pullback!!(da)
            atomic_pointerset(
                dx, increment_rdata!!(atomic_pointerref(dx, :monotonic), da), :monotonic
            )
            return NoRData(), NoRData(), NoRData()
        end
        return a, atomic_pointerref_pullback!!
    end
end

@intrinsic atomic_pointerset
# Exact Ptr{T} lanes admit scalar and coherent pointer stores; element-wise
# Ptr{S} lanes need the guard below. Non-differentiable elements write only the primal.
function frule!!(
    ::Lifted{typeof(atomic_pointerset),Nw},
    p::Lifted{Ptr{T},Nw,NTuple{Nw,Ptr{T}}},
    x::Lifted,
    order::Lifted,
) where {Nw,T}
    _order = primal(order)
    atomic_pointerset(primal(p), primal(x), _order)
    if tangent_type(T) !== NoTangent
        p_partials = tangent(p)
        @inbounds for lane in 1:Nw
            _check_fwd_tangent_ptr_addressable(primal(p), p_partials[lane])
        end
        @inbounds for lane in 1:Nw
            atomic_pointerset(p_partials[lane], tangent(x, lane), _order)
        end
    end
    return p
end
# Non-differentiable pointer (V === NoDual): store the primal; no tangent to write.
function frule!!(
    ::Lifted{typeof(atomic_pointerset),Nw},
    p::Lifted{Ptr{T},Nw,NoDual},
    x::Lifted,
    order::Lifted,
) where {Nw,T}
    _check_nodual_diff_ptr(T)
    atomic_pointerset(primal(p), primal(x), primal(order))
    return p
end
# As in pointerset, an element-wise tangent layout cannot accept a bare lane store.
function frule!!(
    ::Lifted{typeof(atomic_pointerset),Nw},
    p::Lifted{Ptr{T},Nw,<:NTuple{Nw,Ptr}},
    x::Lifted,
    order::Lifted,
) where {Nw,T}
    tangent_type(T) === NoTangent || throw(
        ArgumentError(
            "atomic_pointerset into a differentiable `Ptr{$T}` with an element-wise " *
            "array-of-duals per-lane V; the array-of-pointers store is unsupported.",
        ),
    )
    atomic_pointerset(primal(p), primal(x), primal(order))
    return p
end
function rrule!!(::CoDual{typeof(atomic_pointerset)}, p::CoDual{<:Ptr}, x::CoDual, order)
    _p = primal(p)
    _order = primal(order)
    _check_tangent_ptr(primal(p), tangent(p))
    # Bookkeeping loads/stores use :monotonic: a store-only primal ordering (e.g. :release)
    # would throw ConcurrencyViolationError if reused for these save/restore loads.
    old_value = atomic_pointerref(_p, :monotonic)
    old_tangent = atomic_pointerref(tangent(p), :monotonic)
    dp = tangent(p)
    function atomic_pointerset_pullback!!(::NoRData)
        dx_r = atomic_pointerref(dp, :monotonic)
        atomic_pointerset(_p, old_value, :monotonic)
        atomic_pointerset(dp, old_tangent, :monotonic)
        return NoRData(), NoRData(), rdata(dx_r), NoRData()
    end

    atomic_pointerset(_p, primal(x), _order)
    # zero_tangent(primal(x), tangent(x)) is used to correctly handle
    # Ptr types, whose tangent is purely fdata (a Ptr) with NoRData.
    atomic_pointerset(dp, zero_tangent(primal(x), tangent(x)), :monotonic)
    return p, atomic_pointerset_pullback!!
end

# atomic_pointerswap

# Int/UInt -> Ptr has no shadow pointer to recover; both modes share this refusal.
const _INT2PTR_ERR_MSG =
    "It is not permissible to bitcast from an Int/UInt type to a Ptr type during AD, as " *
    "this risks giving the wrong answer, or causing Julia to segfault. " *
    "If this call to bitcast appears as part of the implementation of a " *
    "differentiable function, you should write a rule for this function, or modify " *
    "its implementation to avoid the bitcast."

const _PLACEHOLDER_TANGENT_PTR_MSG =
    "Cannot differentiate a load or store through a `Ptr` whose tangent is the placeholder that " *
    "the `uninit_*` convention builds from the pointer's own address. There is no derivative " *
    "buffer behind it, so writing a derivative through it would land in the primal buffer. " *
    "This arises when a bare `Ptr` reaches AD as a differentiable input; differentiate the " *
    "underlying array instead, so a real tangent buffer exists."

const _NULL_TANGENT_PTR_MSG =
    "Cannot differentiate a load or store through a `Ptr` with no tangent storage behind it. " *
    "The pointer derives from a buffer whose element type is non-differentiable (a " *
    "`Vector{UInt8}`, say), so no derivative buffer exists to read or write and its tangent " *
    "pointer is NULL. Reinterpreting such a buffer as a differentiable element type under AD " *
    "is not supported; allocate it with the differentiable element type instead."

# NULL is safe only for zero-size tangent elements. Every consumer, including
# container constructors such as unsafe_wrap, must reject it otherwise.
# Representing absent storage with its own type would enforce this by dispatch,
# but requires changing every rule accepting a Ptr tangent.
@inline function _check_tangent_ptr(x, dx)
    if dx isa Ptr && _elements_occupy_storage(eltype(dx))
        iszero(UInt(dx)) && throw(ArgumentError(_NULL_TANGENT_PTR_MSG))
        # The non-NULL uninit_* placeholder aliases the primal's own bytes.
        x isa Ptr &&
            UInt(dx) == UInt(x) &&
            throw(ArgumentError(_PLACEHOLDER_TANGENT_PTR_MSG))
    end
    return nothing
end

# Reference elements occupy pointer-sized slots. sizeof is only safe for isbits
# elements: abstract types and even concrete String are unsized.
@inline _elements_occupy_storage(::Type{E}) where {E} = !isbitstype(E) || sizeof(E) > 0

# The uninit_* placeholder aliases the primal address. Reject it with the same
# forward error as NoDual; a correct derivative needs separate partial storage.
@inline function _check_fwd_tangent_ptr_addressable(p::Ptr, dp::Ptr)
    if IntrinsicsWrappers._elements_occupy_storage(eltype(dp)) && UInt(dp) == UInt(p)
        throw(ArgumentError(IntrinsicsWrappers._NODUAL_DIFF_PTR_MSG))
    end
    return nothing
end

const _NODUAL_DIFF_PTR_MSG =
    "Forward-mode AD cannot load from or store to a `Ptr` whose pointee is a differentiable " *
    "scalar but whose forward representation is `NoDual`: there are no per-lane partial " *
    "pointers behind it, so the derivative cannot be carried. This typically arises from " *
    "reinterpreting a non-differentiable buffer (a `Vector{UInt8}`, say) as a differentiable " *
    "element type; allocate the buffer with that element type instead."

# The NoDual fallback permits UInt64 and Ptr{Float64} pointees; a
# scalar differentiable pointee needs per-lane pointers. Do not replace missing
# storage with a zero dual: that would silently discard a derivative.
@inline function _check_nodual_diff_ptr(::Type{T}) where {T}
    T <: NDualEltype && throw(ArgumentError(_NODUAL_DIFF_PTR_MSG))
    return nothing
end

@intrinsic bitcast
function frule!!(::Lifted{typeof(bitcast),Nw}, ::Lifted{Type{T},Nw}, x::Lifted) where {Nw,T}
    if T <: IEEEFloat
        msg =
            "It is not permissible to bitcast to a differentiable type during AD, as " *
            "this risks dropping tangents, and therefore risks silently giving the wrong " *
            "answer. If this call to bitcast appears as part of the implementation of a " *
            "differentiable function, you should write a rule for this function, or modify " *
            "its implementation to avoid the bitcast."
        throw(ArgumentError(msg))
    end
    # Int/UInt -> Ptr cannot supply a shadow pointer, just as in reverse mode.
    T <: Ptr && primal(x) isa Union{Int,UInt} && throw(ArgumentError(_INT2PTR_ERR_MSG))
    v = bitcast(T, primal(x))
    # Non-Ptr or NoDual-V bitcast: no forward derivative to carry.
    return Lifted{typeof(v),Nw}(v, NoDual())
end
# Re-type lane pointers only when their stride survives; otherwise retain the
# incoherent V so downstream consumers refuse it. Element-wise duals may be wider
# than primals, and pointer_from_objref can supply zero-size tangent elements.
# Restrict to Ptr targets so differentiable scalar targets still hit the refusal above.
@inline function frule!!(
    ::Lifted{typeof(bitcast),Nw}, ::Lifted{Type{T},Nw}, x::Lifted{P,Nw,<:NTuple{Nw,<:Ptr}}
) where {Nw,T<:Ptr,P<:Ptr}
    tx = tangent(x)
    lanes = ntuple(Val(Nw)) do k
        p = tx[k]
        _retyping_keeps_stride(typeof(p), T) ? bitcast(T, p) : p
    end
    return Lifted{T,Nw}(bitcast(T, primal(x)), lanes)
end
# Nothing erases the element type in the pointer(::Array) chain and may re-type
# freely. NoTangent is a known zero-size sentinel, so must not re-type even to Nothing
# (which would launder it into any type). Abstract elements have no known size.
@inline function _retyping_keeps_stride(::Type{Ptr{A}}, ::Type{Ptr{B}}) where {A,B}
    A === NoTangent && return false
    A === Nothing && return true
    (isconcretetype(A) && isconcretetype(B)) || return false
    return sizeof(A) == sizeof(B)
end

# VoidPtrTangent preserves the erased element type so a Cvoid hop gets the same
# verdict as direct re-typing; checking primal types alone cannot recover it.
@inline function _retype_tangent_ptr(::Type{Ptr{Nothing}}, ::Type{Ptr{A}}, dx) where {A}
    return VoidPtrTangent(bitcast(Ptr{Nothing}, dx), tangent_type(A))
end
@inline function _retype_tangent_ptr(::Type{Ptr{Nothing}}, ::Type{Ptr{Nothing}}, dx)
    return dx
end
@inline function _retype_tangent_ptr(::Type{Ptr{B}}, ::Type{Ptr{Nothing}}, dx) where {B}
    TB = tangent_type(B)
    _tangent_retyping_verdict(
        dx.elt, TB, " (the element type was erased through a `Ptr{Cvoid}`)"
    )
    return bitcast(Ptr{TB}, dx.p)
end
@inline function _retype_tangent_ptr(::Type{Ptr{B}}, ::Type{Ptr{A}}, dx) where {A,B}
    return bitcast(Ptr{tangent_type(B)}, dx)
end

# Share the verdict between direct re-typing and recovery from Cvoid.
# Equal-sized reference and inline elements are not interchangeable: writing a
# Float64 over a Vector{Float64} reference would corrupt the GC pointer.
@inline function _tangent_retyping_verdict(
    @nospecialize(TA), @nospecialize(TB), whence::String
)
    TA === Nothing && return nothing            # a tangent OBJECT, checked where it was created
    TA === TB && return nothing
    isbitstype(TB) && sizeof(TB) == 0 && return nothing   # asks nothing of the buffer
    isbitstype(TA) && isbitstype(TB) && sizeof(TA) == sizeof(TB) && return nothing
    why = if TA === NoTangent
        "there is no tangent storage behind this address at all, so a `$TB` load or store through " *
        "the re-typed pointer would touch memory no tangent buffer owns"
    elseif !isbitstype(TA)
        "the tangent storage behind this address holds `$TA` REFERENCES, not inline values, so a " *
        "`$TB` load or store through the re-typed pointer would read or overwrite a pointer the " *
        "garbage collector owns"
    else
        "the tangent storage behind this address is laid out in `$TA` elements, so a `$TB` load " *
        "or store through the re-typed pointer would straddle two of them and corrupt both"
    end
    # The `Ptr{Cvoid}` round trip is recoverable — re-type between the real element types and the
    # check can see them. A direct mismatch is not, so do not tell that caller to do what they did.
    advice = if isempty(whence)
        "These element types genuinely differ, so there is no sound re-typing between them."
    else
        "Re-type directly between the element types you differentiate through."
    end
    throw(
        ArgumentError(
            "Cannot re-type a tangent pointer to `Ptr{$TB}` during AD$whence: $why. $advice"
        ),
    )
end

@inline function _check_tangent_retyping_fits(::Type{Ptr{A}}, ::Type{Ptr{B}}) where {A,B}
    A === Nothing && return nothing
    return _tangent_retyping_verdict(tangent_type(A), tangent_type(B), "")
end
function rrule!!(f::CoDual{typeof(bitcast)}, t::CoDual{Type{T}}, x) where {T}
    if T <: IEEEFloat
        msg =
            "It is not permissible to bitcast to a differentiable type during AD, as " *
            "this risks dropping tangents, and therefore risks silently giving the wrong " *
            "answer. If this call to bitcast appears as part of the implementation of a " *
            "differentiable function, you should write a rule for this function, or modify " *
            "its implementation to avoid the bitcast."
        throw(ArgumentError(msg))
    end
    _x = primal(x)
    v = bitcast(T, _x)
    if T <: Ptr && _x isa Ptr
        _check_tangent_retyping_fits(typeof(_x), T)
        dv = _retype_tangent_ptr(T, typeof(_x), tangent(x))
    elseif T <: Ptr && _x isa Union{Int,UInt}
        throw(ArgumentError(_INT2PTR_ERR_MSG))
    else
        dv = NoFData()
    end
    return CoDual(v, dv), NoPullback(f, t, x)
end

@inactive_intrinsic bswap_int
@inactive_intrinsic ceil_llvm

"""
    __cglobal(::Val{s}, x::Vararg{Any, N}) where {s, N}

Replacement for `Core.Intrinsics.cglobal`. `cglobal` is different from the other intrinsics
in that the name `cglobal` is reserved by the language (try creating a variable called
`cglobal` -- Julia will not let you). Additionally, it requires that its first argument,
the specification of the name of the C cglobal variable that this intrinsic returns a
pointer to, is known statically. In this regard it is like foreigncalls.

As a consequence, it requires special handling. The name is converted into a `Val` so that
it is available statically, and the function into which `cglobal` calls are converted is
named `Mooncake.IntrinsicsWrappers.__cglobal`, rather than
`Mooncake.IntrinsicsWrappers.cglobal`.

If you examine the code associated with `Mooncake.intrinsic_to_function`, you will see that
special handling of `cglobal` is used.
"""
__cglobal(::Val{s}, x::Vararg{Any,N}) where {s,N} = cglobal(s, x...)

translate(::Val{Intrinsics.cglobal}) = __cglobal
function Mooncake._is_primitive(
    ::Type{MinimalCtx}, ::Type{<:Mode}, ::Type{<:Tuple{typeof(__cglobal),Vararg}}
)
    return true
end
function frule!!(::Lifted{typeof(__cglobal),Nw}, args::Vararg{Lifted,M}) where {Nw,M}
    y = __cglobal(tuple_map(primal, args)...)
    return Lifted{typeof(y),Nw}(y, NoDual())
end
function rrule!!(f::CoDual{typeof(__cglobal)}, args...)
    return Mooncake.uninit_fcodual(__cglobal(map(primal, args)...)), NoPullback(f, args...)
end

@inactive_intrinsic checked_sadd_int
@inactive_intrinsic checked_sdiv_int
@inactive_intrinsic checked_smul_int
@inactive_intrinsic checked_srem_int
@inactive_intrinsic checked_ssub_int
@inactive_intrinsic checked_uadd_int
@inactive_intrinsic checked_udiv_int
@inactive_intrinsic checked_umul_int
@inactive_intrinsic checked_urem_int
@inactive_intrinsic checked_usub_int

@intrinsic copysign_float
function frule!!(
    ::Lifted{typeof(copysign_float),N}, x::Lifted{T,N,NDual{T,N}}, y::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    z = copysign_float(primal(x), primal(y))
    # flipsign(sign(x), y) handles y=0; scale partials only to preserve V.value == z.
    s = flipsign(sign(primal(x)), primal(y))
    return Lifted{T,N}(z, NDual{T,N}(z, s .* tangent(x).partials))
end
function rrule!!(::CoDual{typeof(copysign_float)}, x, y)
    _x = primal(x)
    _y = primal(y)
    # d copysign(x,y)/dx = flipsign(sign(x), y) (correct at y=0); derivative w.r.t. y is zero.
    copysign_float_pullback!!(dz) = NoRData(), dz * flipsign(sign(_x), _y), zero_rdata(_y)
    z = copysign_float(_x, _y)
    return CoDual(z, NoFData()), copysign_float_pullback!!
end

@inactive_intrinsic ctlz_int
@inactive_intrinsic ctpop_int
@inactive_intrinsic cttz_int

@intrinsic div_float
function frule!!(
    ::Lifted{typeof(div_float),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(div_float(primal(a), primal(b)), tangent(a) / tangent(b))
end
function rrule!!(::CoDual{typeof(div_float)}, a, b)
    _a = primal(a)
    _b = primal(b)
    _y = div_float(_a, _b)
    # Reuse the quotient: squaring b can overflow or underflow when a and b are
    # both large or tiny, losing an otherwise representable derivative.
    div_float_pullback!!(dy) = NoRData(), div_float(dy, _b), -dy * _y / _b
    return CoDual(_y, NoFData()), div_float_pullback!!
end

@intrinsic div_float_fast
function frule!!(
    ::Lifted{typeof(div_float_fast),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(div_float_fast(primal(a), primal(b)), tangent(a) / tangent(b))
end
function rrule!!(::CoDual{typeof(div_float_fast)}, a, b)
    _a = primal(a)
    _b = primal(b)
    _y = div_float_fast(_a, _b)
    function div_float_pullback!!(dy)
        return NoRData(), div_float_fast(dy, _b), -dy * div_float_fast(_y, _b)
    end
    return CoDual(_y, NoFData()), div_float_pullback!!
end

@inactive_intrinsic eq_float
@inactive_intrinsic eq_float_fast
@inactive_intrinsic eq_int
@inactive_intrinsic flipsign_int
@inactive_intrinsic floor_llvm

@intrinsic fma_float
function frule!!(
    ::Lifted{typeof(fma_float),N},
    x::Lifted{T,N,NDual{T,N}},
    y::Lifted{T,N,NDual{T,N}},
    z::Lifted{T,N,NDual{T,N}},
) where {N,T<:IEEEFloat}
    # Use fused fma to preserve single rounding and the inner-value invariant.
    dy = fma(tangent(x), tangent(y), tangent(z))
    return Lifted{T,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(fma_float)}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    fma_float_pullback!!(da) = NoRData(), da * _y, da * _x, da
    return CoDual(fma_float(_x, _y, primal(z)), NoFData()), fma_float_pullback!!
end

@intrinsic fpext
function frule!!(
    ::Lifted{typeof(fpext),N}, ::Lifted{Type{Pext},N}, x::Lifted{P,N,NDual{P,N}}
) where {N,Pext<:IEEEFloat,P<:IEEEFloat}
    # NDual{Pext,N}(::NDual{P,N}) is the cross-precision constructor.
    return Lifted{Pext,N}(fpext(Pext, primal(x)), NDual{Pext,N}(tangent(x)))
end
function rrule!!(
    ::CoDual{typeof(fpext)}, ::CoDual{Type{Pext}}, x::CoDual{P}
) where {Pext<:IEEEFloat,P<:IEEEFloat}
    fpext_adjoint!!(dy::Pext) = NoRData(), NoRData(), fptrunc(P, dy)
    return zero_fcodual(fpext(Pext, primal(x))), fpext_adjoint!!
end

@inactive_intrinsic fpiseq
@inactive_intrinsic fptosi
@inactive_intrinsic fptoui

@intrinsic fptrunc
function frule!!(
    ::Lifted{typeof(fptrunc),N}, ::Lifted{Type{Ptrunc},N}, x::Lifted{P,N,NDual{P,N}}
) where {N,Ptrunc<:IEEEFloat,P<:IEEEFloat}
    return Lifted{Ptrunc,N}(fptrunc(Ptrunc, primal(x)), NDual{Ptrunc,N}(tangent(x)))
end
function rrule!!(
    ::CoDual{typeof(fptrunc)}, ::CoDual{Type{Ptrunc}}, x::CoDual{P}
) where {Ptrunc<:IEEEFloat,P<:IEEEFloat}
    fptrunc_adjoint!!(dy::Ptrunc) = NoRData(), NoRData(), convert(P, dy)
    return zero_fcodual(fptrunc(Ptrunc, primal(x))), fptrunc_adjoint!!
end

@inactive_intrinsic have_fma
@inactive_intrinsic le_float
@inactive_intrinsic le_float_fast

# llvmcall -- interesting and not implementable at the minute

@inactive_intrinsic lshr_int
@inactive_intrinsic lt_float
@inactive_intrinsic lt_float_fast

@static if VERSION >= v"1.12.0-rc2"
    @intrinsic max_float
    function frule!!(
        ::Lifted{typeof(max_float),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
    ) where {N,T<:IEEEFloat}
        p = max_float(primal(a), primal(b))
        # The selector must handle NaN and signed zero, where ordering is insufficient.
        return Lifted{T,N}(
            p,
            NDual{T,N}(
                p,
                ifelse(
                    Mooncake.Nfwd._ndual_pick_max(primal(a), primal(b)),
                    tangent(a).partials,
                    tangent(b).partials,
                ),
            ),
        )
    end
    function rrule!!(
        ::CoDual{typeof(max_float)}, a::CoDual{P}, b::CoDual{P}
    ) where {P<:Base.IEEEFloat}
        _a = primal(a)
        _b = primal(b)
        x = max_float(_a, _b)
        # Credit the returned operand for NaN; isequal ties select the second operand.
        tmp = isequal(x, _a) & !isequal(x, _b)
        function max_float_adjoint(dx)
            da = ifelse(tmp, dx, zero(P))
            db = ifelse(tmp, zero(P), dx)
            return NoRData(), da, db
        end
        return zero_fcodual(x), max_float_adjoint
    end

    @intrinsic max_float_fast
    function frule!!(
        ::Lifted{typeof(max_float_fast),N},
        a::Lifted{T,N,NDual{T,N}},
        b::Lifted{T,N,NDual{T,N}},
    ) where {N,T<:IEEEFloat}
        p = max_float_fast(primal(a), primal(b))
        # Match FastMath's comparison: signed-zero ties have unspecified primal sign,
        # so isequal cannot select reliably. Either partial is a valid subgradient at
        # a numerical tie; NaN is outside FastMath's contract.
        return Lifted{T,N}(
            p,
            NDual{T,N}(
                p, ifelse(primal(a) > primal(b), tangent(a).partials, tangent(b).partials)
            ),
        )
    end
    function rrule!!(
        ::CoDual{typeof(max_float_fast)}, a::CoDual{P}, b::CoDual{P}
    ) where {P<:Base.IEEEFloat}
        _a = primal(a)
        _b = primal(b)
        tmp = _a > _b
        x = max_float_fast(_a, _b)
        function max_float_fast_adjoint(dx)
            da = ifelse(tmp, dx, zero(P))
            db = ifelse(tmp, zero(P), dx)
            return NoRData(), da, db
        end
        return zero_fcodual(x), max_float_fast_adjoint
    end

    @intrinsic min_float
    function frule!!(
        ::Lifted{typeof(min_float),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
    ) where {N,T<:IEEEFloat}
        p = min_float(primal(a), primal(b))
        # The selector must handle NaN and signed zero, where ordering is insufficient.
        return Lifted{T,N}(
            p,
            NDual{T,N}(
                p,
                ifelse(
                    Mooncake.Nfwd._ndual_pick_min(primal(a), primal(b)),
                    tangent(a).partials,
                    tangent(b).partials,
                ),
            ),
        )
    end
    function rrule!!(
        ::CoDual{typeof(min_float)}, a::CoDual{P}, b::CoDual{P}
    ) where {P<:Base.IEEEFloat}
        _a = primal(a)
        _b = primal(b)
        x = min_float(_a, _b)
        # Credit the returned operand for NaN; isequal ties select the first operand.
        tmp = isequal(x, _a) | !isequal(x, _b)
        function min_float_adjoint(dx)
            da = ifelse(tmp, dx, zero(P))
            db = ifelse(tmp, zero(P), dx)
            return NoRData(), da, db
        end
        return zero_fcodual(x), min_float_adjoint
    end

    @intrinsic min_float_fast
    function frule!!(
        ::Lifted{typeof(min_float_fast),N},
        a::Lifted{T,N,NDual{T,N}},
        b::Lifted{T,N,NDual{T,N}},
    ) where {N,T<:IEEEFloat}
        p = min_float_fast(primal(a), primal(b))
        # A bare comparison, mirroring `Base.FastMath.min_fast(::NDual)`; see `max_float_fast`.
        return Lifted{T,N}(
            p,
            NDual{T,N}(
                p, ifelse(primal(a) < primal(b), tangent(a).partials, tangent(b).partials)
            ),
        )
    end
    function rrule!!(
        ::CoDual{typeof(min_float_fast)}, a::CoDual{P}, b::CoDual{P}
    ) where {P<:Base.IEEEFloat}
        _a = primal(a)
        _b = primal(b)
        tmp = _a < _b
        x = min_float_fast(_a, _b)
        function min_float_fast_adjoint(dx)
            da = ifelse(tmp, dx, zero(P))
            db = ifelse(tmp, zero(P), dx)
            return NoRData(), da, db
        end
        return zero_fcodual(x), min_float_fast_adjoint
    end
end

@intrinsic mul_float
function frule!!(
    ::Lifted{typeof(mul_float),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(mul_float(primal(a), primal(b)), tangent(a) * tangent(b))
end
function rrule!!(::CoDual{typeof(mul_float)}, a, b)
    _a = primal(a)
    _b = primal(b)
    mul_float_pb!!(dc) = NoRData(), dc * _b, _a * dc
    return CoDual(mul_float(_a, _b), NoFData()), mul_float_pb!!
end

@intrinsic mul_float_fast
function frule!!(
    ::Lifted{typeof(mul_float_fast),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(mul_float_fast(primal(a), primal(b)), tangent(a) * tangent(b))
end
function rrule!!(::CoDual{typeof(mul_float_fast)}, a, b)
    _a = primal(a)
    _b = primal(b)
    mul_float_fast_pb!!(dc) = NoRData(), dc * _b, _a * dc
    return CoDual(mul_float_fast(_a, _b), NoFData()), mul_float_fast_pb!!
end

@inactive_intrinsic mul_int

@intrinsic muladd_float
function frule!!(
    ::Lifted{typeof(muladd_float),N},
    x::Lifted{T,N,NDual{T,N}},
    y::Lifted{T,N,NDual{T,N}},
    z::Lifted{T,N,NDual{T,N}},
) where {N,T<:IEEEFloat}
    # Use muladd's possibly fused rounding to preserve the inner-value invariant.
    dy = muladd(tangent(x), tangent(y), tangent(z))
    return Lifted{T,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(muladd_float)}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    _z = primal(z)
    muladd_float_pullback!!(da) = NoRData(), da * _y, da * _x, da
    return CoDual(muladd_float(_x, _y, _z), NoFData()), muladd_float_pullback!!
end

@inactive_intrinsic ne_float
@inactive_intrinsic ne_float_fast
@inactive_intrinsic ne_int

@intrinsic neg_float
function frule!!(
    ::Lifted{typeof(neg_float),N}, x::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(neg_float(primal(x)), -tangent(x))
end
function rrule!!(::CoDual{typeof(neg_float)}, x)
    _x = primal(x)
    neg_float_pullback!!(dy) = NoRData(), -dy
    return CoDual(neg_float(_x), NoFData()), neg_float_pullback!!
end

@intrinsic neg_float_fast
function frule!!(
    ::Lifted{typeof(neg_float_fast),N}, x::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(neg_float_fast(primal(x)), -tangent(x))
end
function rrule!!(::CoDual{typeof(neg_float_fast)}, x)
    _x = primal(x)
    neg_float_fast_pullback!!(dy) = NoRData(), -dy
    return CoDual(neg_float_fast(_x), NoFData()), neg_float_fast_pullback!!
end

@inactive_intrinsic neg_int
@inactive_intrinsic not_int
@inactive_intrinsic or_int

@intrinsic pointerref
# _scalar_ndual builds NDual for real elements and Complex{NDual} for complex ones.
function frule!!(
    ::Lifted{typeof(pointerref),Nw},
    x::Lifted{Ptr{T},Nw,NTuple{Nw,Ptr{T}}},
    y::Lifted,
    z::Lifted,
) where {Nw,T<:NDualEltype}
    _y = primal(y)
    _z = primal(z)
    a = pointerref(primal(x), _y, _z)
    x_partials = tangent(x)
    # Reject uninit_* placeholders before reading derivatives from the primal bytes.
    # Check every lane before any derivative store can leave a buffer half-written.
    @inbounds for lane in 1:Nw
        _check_fwd_tangent_ptr_addressable(primal(x), x_partials[lane])
    end
    da_lanes = ntuple(lane -> pointerref(x_partials[lane], _y, _z), Val(Nw))
    return Lifted{T,Nw}(a, _scalar_ndual(a, da_lanes))
end
# Pointees in NDualEltype need partial pointers even when presented with NoDual.
function frule!!(
    ::Lifted{typeof(pointerref),Nw}, x::Lifted{Ptr{T},Nw,NoDual}, y::Lifted, z::Lifted
) where {Nw,T}
    _check_nodual_diff_ptr(T)
    a = pointerref(primal(x), primal(y), primal(z))
    return Lifted{typeof(a),Nw}(a, NoDual())
end
# unsafe_convert/bitcast can give non-differentiable pointers incoherent lanes;
# their loads may collapse to NoDual. Differentiable loads must refuse this layout,
# including pointer elements and pointer_from_objref on general mutable structs.
# MutableDual interleaves values and partials, so a same-offset raw load cannot
# recover a derivative. Supporting it needs an opt-in primal-shaped partials shadow,
# as NDualRef/NDualArray provide; making that the default regresses chunked struct
# math. A shortcut through the interleaving would hardcode NDual's layout.
function frule!!(
    ::Lifted{typeof(pointerref),Nw},
    x::Lifted{Ptr{T},Nw,<:NTuple{Nw,Ptr}},
    y::Lifted,
    z::Lifted,
) where {Nw,T}
    tangent_type(T) === NoTangent || throw(
        ArgumentError(
            "Forward-mode AD cannot take a raw scalar load (`pointerref`/`unsafe_load`) of a " *
            "differentiable `Ptr{$T}` whose per-lane tangent is not the canonical " *
            "`NTuple{$Nw,Ptr{$T}}` per-lane-partial shape. This typically arises from " *
            "`pointerref`/`unsafe_load` through `pointer_from_objref` of a mutable struct: its " *
            "forward tangent interleaves the value and partials in one object (no separate parallel " *
            "partials storage at the object's address), so the load cannot recover the derivative. " *
            "Use reverse mode, hold the value in a `Ref` or `Array` (whose forward tangents keep a " *
            "parallel partials buffer), or write a custom forward tangent for the struct.",
        ),
    )
    a = pointerref(primal(x), primal(y), primal(z))
    return Lifted{typeof(a),Nw}(a, NoDual())
end
function rrule!!(::CoDual{typeof(pointerref)}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    _z = primal(z)
    dx = tangent(x)
    _check_tangent_ptr(_x, dx)
    a = CoDual(pointerref(_x, _y, _z), fdata(pointerref(dx, _y, _z)))
    if Mooncake.rdata_type(tangent_type(Mooncake._typeof(primal(a)))) == NoRData
        return a, NoPullback((NoRData(), NoRData(), NoRData(), NoRData()))
    else
        function pointerref_pullback!!(da)
            pointerset(dx, increment_rdata!!(pointerref(dx, _y, _z), da), _y, _z)
            return NoRData(), NoRData(), NoRData(), NoRData()
        end
        return a, pointerref_pullback!!
    end
end

@intrinsic pointerset
# Exact Ptr{T} lanes admit scalar and coherent pointer stores; element-wise
# Ptr{S} lanes need the guard below. Non-differentiable elements write only the primal.
function frule!!(
    ::Lifted{typeof(pointerset),Nw},
    p::Lifted{Ptr{T},Nw,NTuple{Nw,Ptr{T}}},
    x::Lifted,
    idx::Lifted,
    z::Lifted,
) where {Nw,T}
    _idx = primal(idx)
    _z = primal(z)
    pointerset(primal(p), primal(x), _idx, _z)
    if tangent_type(T) !== NoTangent
        p_partials = tangent(p)
        @inbounds for lane in 1:Nw
            _check_fwd_tangent_ptr_addressable(primal(p), p_partials[lane])
        end
        @inbounds for lane in 1:Nw
            pointerset(p_partials[lane], tangent(x, lane), _idx, _z)
        end
    end
    return p
end
# Non-differentiable pointer (V === NoDual): store the primal; no tangent to write.
function frule!!(
    ::Lifted{typeof(pointerset),Nw},
    p::Lifted{Ptr{T},Nw,NoDual},
    x::Lifted,
    idx::Lifted,
    z::Lifted,
) where {Nw,T}
    _check_nodual_diff_ptr(T)
    pointerset(primal(p), primal(x), primal(idx), primal(z))
    return p
end
# Element-wise pointer tangents store tuples of pointers, not bare lane pointers.
# A bare store would corrupt the stride. The coherent-lane method is more specific.
function frule!!(
    ::Lifted{typeof(pointerset),Nw},
    p::Lifted{Ptr{T},Nw,<:NTuple{Nw,Ptr}},
    x::Lifted,
    idx::Lifted,
    z::Lifted,
) where {Nw,T}
    tangent_type(T) === NoTangent || throw(
        ArgumentError(
            "pointerset into a differentiable `Ptr{$T}` with an element-wise array-of-duals per-lane V; " *
            "the array-of-pointers store is unsupported.",
        ),
    )
    pointerset(primal(p), primal(x), primal(idx), primal(z))
    return p
end
function rrule!!(::CoDual{typeof(pointerset)}, p, x, idx, z)
    _p = primal(p)
    _idx = primal(idx)
    _z = primal(z)
    _check_tangent_ptr(primal(p), tangent(p))
    old_value = pointerref(_p, _idx, _z)
    old_tangent = pointerref(tangent(p), _idx, _z)
    dp = tangent(p)
    function pointerset_pullback!!(::NoRData)
        dx_r = pointerref(dp, _idx, _z)
        pointerset(_p, old_value, _idx, _z)
        pointerset(dp, old_tangent, _idx, _z)
        return NoRData(), NoRData(), rdata(dx_r), NoRData(), NoRData()
    end

    pointerset(_p, primal(x), _idx, _z)
    # zero_tangent(primal(x), tangent(x)) is used to correctly handle
    # Ptr types, whose tangent is purely fdata (a Ptr) with NoRData.
    pointerset(dp, zero_tangent(primal(x), tangent(x)), _idx, _z)
    return p, pointerset_pullback!!
end

@inactive_intrinsic rint_llvm
@inactive_intrinsic sdiv_int
@inactive_intrinsic sext_int
@inactive_intrinsic shl_int
@inactive_intrinsic sitofp
@inactive_intrinsic sle_int
@inactive_intrinsic slt_int

@intrinsic sqrt_llvm
@intrinsic sqrt_llvm_fast
function frule!!(
    f::Lifted{F,Nw}, x::Lifted{T,Nw,NDual{T,Nw}}
) where {F<:Union{typeof(sqrt_llvm),typeof(sqrt_llvm_fast)},Nw,T<:IEEEFloat}
    # Use the intrinsic: Base.sqrt throws for negative inputs instead of returning NaN.
    y = primal(f)(primal(x))
    dy = NDual{T,Nw}(y, Mooncake._fwd_guarded_scale(tangent(x).partials, inv(2 * y)))
    return Lifted{T,Nw}(y, dy)
end
function rrule!!(::CoDual{typeof(sqrt_llvm)}, x::CoDual{P}) where {P}
    _y = sqrt_llvm(primal(x))
    function llvm_sqrt_pullback!!(dy)
        dx = nan_tangent_guard(dy, dy / (2 * _y))
        return NoRData(), dx
    end
    return CoDual(_y, NoFData()), llvm_sqrt_pullback!!
end

function rrule!!(::CoDual{typeof(sqrt_llvm_fast)}, x::CoDual{P}) where {P}
    _y = sqrt_llvm_fast(primal(x))
    function llvm_sqrt_fast_pullback!!(dy)
        dx = nan_tangent_guard(dy, dy / (2 * _y))
        return NoRData(), dx
    end
    return CoDual(_y, NoFData()), llvm_sqrt_fast_pullback!!
end

@inactive_intrinsic srem_int

@intrinsic sub_float
function frule!!(
    ::Lifted{typeof(sub_float),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(sub_float(primal(a), primal(b)), tangent(a) - tangent(b))
end
function rrule!!(::CoDual{typeof(sub_float)}, a, b)
    _a = primal(a)
    _b = primal(b)
    sub_float_pullback!!(dc) = NoRData(), dc, -dc
    return CoDual(sub_float(_a, _b), NoFData()), sub_float_pullback!!
end

@intrinsic sub_float_fast
function frule!!(
    ::Lifted{typeof(sub_float_fast),N}, a::Lifted{T,N,NDual{T,N}}, b::Lifted{T,N,NDual{T,N}}
) where {N,T<:IEEEFloat}
    return Lifted{T,N}(sub_float_fast(primal(a), primal(b)), tangent(a) - tangent(b))
end
function rrule!!(::CoDual{typeof(sub_float_fast)}, a, b)
    _a = primal(a)
    _b = primal(b)
    sub_float_fast_pullback!!(dc) = NoRData(), dc, -dc
    return CoDual(sub_float_fast(_a, _b), NoFData()), sub_float_fast_pullback!!
end

@inactive_intrinsic sub_int

@intrinsic sub_ptr
function rrule!!(::CoDual{typeof(sub_ptr)}, a, b)
    throw(error("sub_ptr intrinsic hit. This should never happen. Please open an issue"))
end

@inactive_intrinsic trunc_int
@inactive_intrinsic trunc_llvm
@inactive_intrinsic udiv_int
@inactive_intrinsic uitofp
@inactive_intrinsic ule_int
@inactive_intrinsic ult_int
@inactive_intrinsic urem_int
@inactive_intrinsic xor_int
@inactive_intrinsic zext_int

# This intrinsic was removed in 1.11 as part of the Array implementation refactor.
@static if VERSION < v"1.11.0-rc4"
    @inactive_intrinsic arraylen
end

end # IntrinsicsWrappers

@zero_derivative MinimalCtx Tuple{typeof(<:),Any,Any}
@zero_derivative MinimalCtx Tuple{typeof(===),Any,Any}

# Core._abstracttype

#
# Core._apply_iterate
#
# We don't differentiate `Core._apply_iterate`. Instead, we differentiate
# _apply_iterate_equivalent instead, having replaced all calls to _apply_iterate with it as
# a pre-processing step.

# A function with the same semantics as `Core._apply_iterate`, but which is differentiable.
function _apply_iterate_equivalent(itr, f::F, args::Vararg{Any,N}) where {F,N}
    vec_args = reduce(vcat, map(collect, args))
    tuple_args = __vec_to_tuple(vec_args)
    return tuple_splat(f, tuple_args)
end

# A primitive used to avoid exposing `_apply_iterate_equivalent` to `Core._apply_iterate`.
__vec_to_tuple(v::Vector) = Tuple(v)

@is_primitive MinimalCtx Tuple{typeof(__vec_to_tuple),Vector}
# Tuple accepts both NDualArray and element-wise Vector Vs; __vec_to_tuple
# itself accepts only Vector. The V bound excludes NoDual.
@inline function frule!!(
    ::Lifted{typeof(__vec_to_tuple),Nw}, v::Lifted{<:Vector,Nw,<:AbstractVector}
) where {Nw}
    x = __vec_to_tuple(primal(v))
    # An all-non-differentiable splat needs whole NoDual, not Tuple{NoDual,...}.
    dual_type(Val(Nw), typeof(x)) === NoDual && return Lifted{typeof(x),Nw}(x, NoDual())
    return Lifted{typeof(x),Nw}(x, Tuple(tangent(v)))
end

function rrule!!(::CoDual{typeof(__vec_to_tuple)}, v::CoDual{<:Vector})
    dv = tangent(v)
    y = CoDual(Tuple(primal(v)), fdata(Tuple(dv)))
    function vec_to_tuple_pb!!(dy::Union{Tuple,NoRData})
        if dy isa Tuple
            for n in eachindex(dy)
                dv[n] = increment_rdata!!(dv[n], dy[n])
            end
        end
        return NoRData(), NoRData()
    end
    return y, vec_to_tuple_pb!!
end

# Core._apply_pure
# Core._call_in_world
# Core._call_in_world_total
# Core._call_latest

# Doesn't do anything differentiable.
@zero_adjoint MinimalCtx Tuple{typeof(Core._compute_sparams),Vararg}

# Core._equiv_typedef
# Core._expr
# Core._primitivetype
# Core._setsuper!
# Core._structtype

# SimpleVector mirrors reverse's Vector{Any}, holding each element's canonical V.
# Even an all-non-differentiable svec needs Vector{Any}, not whole NoDual, to match
# the svec/_svec_ref rules and forward-over-reverse return annotations.
@foldable @inline dual_type(::Val{N}, ::Type{Core.SimpleVector}) where {N} = Vector{Any}
function frule!!(
    ::Lifted{typeof(Core._svec_ref),Nw}, v::Lifted{Core.SimpleVector}, _ind::Lifted{Int}
) where {Nw}
    ind = primal(_ind)
    pv = Core._svec_ref(primal(v), ind)
    return Lifted{typeof(pv),Nw}(pv, tangent(v)[ind])
end
function rrule!!(
    f::CoDual{typeof(Core._svec_ref)}, _v::CoDual{Core.SimpleVector}, _ind::CoDual{Int}
)
    ind = primal(_ind)
    v, dv = extract(_v)
    pv = Core._svec_ref(v, ind)
    tv = getindex(dv, ind)
    return _svec_ref_rrule(f, _v, _ind, pv, tv)
end

# Function barrier to limit runtime dispatch
function _svec_ref_rrule(f, _v, _ind, pv, tv)
    ind = primal(_ind)
    a = CoDual(pv, fdata(tv))
    if rdata_type(tangent_type(_typeof(pv))) == NoRData
        return a, NoPullback(f, _v, _ind)
    else
        function _svec_ref_pullback!!(da)
            dv = tangent(_v)
            setindex!(dv, increment_rdata!!(getindex(dv, ind), da), ind)
            return NoRData(), NoRData(), NoRData()
        end
        return a, _svec_ref_pullback!!
    end
end

# The output `SimpleVector`'s forward V is the per-element `Vector{Any}` of each arg's V, so a
# differentiable element (e.g. a float read back out by `_svec_ref`) keeps its derivative.
function frule!!(f::Lifted{typeof(svec),Nw}, args::Vararg{Lifted,M}) where {Nw,M}
    primal_output = svec(tuple_map(primal, args)...)
    return Lifted{Core.SimpleVector,Nw}(primal_output, Any[tangent(a) for a in args])
end

# Forward seed/lift/accessor/unlift for the `Vector{Any}` V (per-element forward V), mirroring
# the reverse `Vector{Any}` machinery. Each element recurses through its own V.
for f in (:_zero_dual_internal, :_uninit_dual_internal)
    @eval function $f(::Val{N}, v::Core.SimpleVector, c::MaybeCache) where {N}
        return Any[$f(Val(N), v[i], c) for i in 1:length(v)]
    end
end
function _randn_dual_internal(
    ::Val{N}, rng::AbstractRNG, v::Core.SimpleVector, c::MaybeCache
) where {N}
    return Any[_randn_dual_internal(Val(N), rng, v[i], c) for i in 1:length(v)]
end
# Cache-free factories: without these, the generic fieldcount-0 fallback returns
# `NTuple{N, Vector{Any}}` for a `SimpleVector`, mismatching `dual_type === Vector{Any}`.
for f in (:zero_dual, :uninit_dual)
    @eval function $f(::Val{N}, v::Core.SimpleVector) where {N}
        return Any[$f(Val(N), v[i]) for i in 1:length(v)]
    end
end
function randn_dual(::Val{N}, rng::AbstractRNG, v::Core.SimpleVector) where {N}
    return Any[randn_dual(Val(N), rng, v[i]) for i in 1:length(v)]
end
lift(v::Core.SimpleVector, ẋ::Vector{Any}) = lift(v, ẋ, nothing)
function lift(v::Core.SimpleVector, ẋ::Vector{Any}, c::Union{Nothing,IdDict})
    # Share a cache across elements so aliased arrays share one V.
    d = c === nothing ? IdDict() : c
    return Lifted{Core.SimpleVector,1}(
        v, Any[tangent(lift(v[i], ẋ[i], d)) for i in 1:length(v)]
    )
end
function _materialise_lane(
    x::Lifted{Core.SimpleVector,N,Vector{Any}}, lane::Integer, cache::IdDict
) where {N}
    p = primal(x)
    v = tangent(x)
    return Any[
        _materialise_lane(Lifted{typeof(p[i]),N}(p[i], v[i]), lane, cache) for
        i in 1:length(p)
    ]
end

function rrule!!(f::CoDual{typeof(svec)}, args::Vararg{Any,N}) where {N}
    primal_output = svec(map(primal, args)...)
    # Tangent type for `SimpleVector` is `Vector{Any}`
    tangent_output = collect(
        Any,
        map(args) do x
            return tangent(x.dx, zero_rdata(x.x))
        end,
    )
    function svec_pullback!!(::NoRData)
        return NoRData(), map(rdata, tangent_output)...
    end
    return CoDual(primal_output, tangent_output), svec_pullback!!
end

@static if VERSION > v"1.12-"
    function frule!!(f::Lifted{typeof(Core._svec_len)}, v::Lifted)
        return Mooncake.zero_derivative(f, v)
    end
    function rrule!!(f::CoDual{typeof(Core._svec_len)}, v)
        return zero_fcodual(Core._svec_len(primal(v))), NoPullback(f, v)
    end
end

# Core._typebody!
function frule!!(::Lifted{typeof(Core._typevar),Nw}, args::Vararg{Lifted,M}) where {Nw,M}
    y = Core._typevar(tuple_map(primal, args)...)
    return Lifted{typeof(y),Nw}(y, NoDual())
end
function rrule!!(f::CoDual{typeof(Core._typevar)}, args...)
    return zero_fcodual(Core._typevar(map(primal, args)...)), NoPullback(f, args...)
end

function frule!!(::Lifted{typeof(Core.apply_type),Nw}, args::Vararg{Lifted,M}) where {Nw,M}
    y = Core.apply_type(tuple_map(primal, args)...)
    return Lifted{typeof(y),Nw}(y, NoDual())
end
function rrule!!(f::CoDual{typeof(Core.apply_type)}, args...)
    T = Core.apply_type(tuple_map(primal, args)...)
    return CoDual{_typeof(T),NoFData}(T, NoFData()), NoPullback(f, args...)
end

function frule!!(
    ::Lifted{typeof(compilerbarrier),Nw}, setting::Lifted{Symbol,Nw}, v::Lifted{P,Nw}
) where {Nw,P}
    s = primal(setting)
    return Lifted{P,Nw}(compilerbarrier(s, primal(v)), compilerbarrier(s, tangent(v)))
end
function rrule!!(::CoDual{typeof(compilerbarrier)}, setting::CoDual{Symbol}, val::CoDual)
    compilerbarrier_pb(dout) = NoRData(), NoRData(), dout
    return compilerbarrier(setting.x, val), compilerbarrier_pb
end

# Core.donotdelete
# Core.finalizer
# Core.get_binding_type

# Both branches arrive evaluated; selecting a slot supports arbitrary branch
# types and preserves stability when they share a type.
@inline function frule!!(
    ::Lifted{typeof(Core.ifelse),Nw}, cond::Lifted{Bool,Nw}, a::Lifted, b::Lifted
) where {Nw}
    return primal(cond) ? a : b
end
function rrule!!(f::CoDual{typeof(Core.ifelse)}, cond, a::A, b::B) where {A,B}
    _cond = primal(cond)
    p_a = primal(a)
    p_b = primal(b)
    pb!! =
        if rdata_type(tangent_type(A)) == NoRData && rdata_type(tangent_type(B)) == NoRData
            NoPullback(f, cond, a, b)
        else
            lazy_da = lazy_zero_rdata(p_a)
            lazy_db = lazy_zero_rdata(p_b)
            function ifelse_pullback!!(dc)
                da = ifelse(_cond, dc, instantiate(lazy_da))
                db = ifelse(_cond, instantiate(lazy_db), dc)
                return NoRData(), NoRData(), da, db
            end
        end

    # Select whole slots to preserve the two-element Union{A,B}; combining
    # union-typed parts introduces four pairings and widens inference to bare CoDual.
    return (_cond ? a : b), pb!!
end

@zero_derivative MinimalCtx Tuple{typeof(Core.sizeof),Any}

# Core.svec

@zero_derivative MinimalCtx Tuple{typeof(applicable),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(fieldtype),Vararg}

const StandardTangentType = Union{Tuple,NamedTuple,Tangent,MutableTangent,NoTangent}
const StandardFDataType = Union{Tuple,NamedTuple,FData,MutableTangent,NoFData}

# Keep runtime-name getfield available on 1.10 for forward-over-reverse HVPs.
# _get_lifted_field handles tuples, named tuples and structs.
function frule!!(::Lifted{typeof(getfield),Nw}, x::Lifted, name::Lifted) where {Nw}
    # Val(runtime_name) would force runtime dispatch through an abstract Lifted slot.
    _name = primal(name)
    y = getfield(primal(x), _name)
    P = _typeof(primal(x))
    if tangent_type(P) == NoTangent
        # Build canonical V even for a SimpleVector field of a NoTangent parent.
        # TODO(#1295): the fresh partials do not alias the field's seeded storage.
        return uninit_lifted(Val(Nw), y)
    else
        V_i = _get_lifted_field(tangent(x), _name)
        _check_lifted_field_ptr_lanes(V_i, Val(Nw))
        return Lifted{typeof(y),Nw}(y, V_i)
    end
end
function frule!!(
    ::Lifted{typeof(getfield),Nw}, x::Lifted, name::Lifted, inbounds::Lifted
) where {Nw}
    _name = primal(name)
    _inbounds = primal(inbounds)
    y = getfield(primal(x), _name, _inbounds)
    P = _typeof(primal(x))
    if tangent_type(P) == NoTangent
        # See the 2-arg `getfield` frule: canonical zero V (handles a `SimpleVector` field
        # whose V is `Vector{Any}`, not `NoDual`), and its TODO(#1295).
        return uninit_lifted(Val(Nw), y)
    else
        V_i = _get_lifted_field(tangent(x), _name)
        _check_lifted_field_ptr_lanes(V_i, Val(Nw))
        return Lifted{typeof(y),Nw}(y, V_i)
    end
end
# NDualRef has no _get_lifted_field method: rebuild its real/complex scalar V
# from parallel partials, as lgetfield does. Its only field is :x (index 1).
function frule!!(
    ::Lifted{typeof(getfield),Nw}, x::Lifted{<:Base.RefValue{P},Nw,<:NDualRef}, name::Lifted
) where {Nw,P<:NDualEltype}
    v = getfield(primal(x), primal(name))
    return Lifted{P,Nw}(v, _scalar_ndual(v, tangent(x).partials[]))
end
function frule!!(
    ::Lifted{typeof(getfield),Nw},
    x::Lifted{<:Base.RefValue{P},Nw,<:NDualRef},
    name::Lifted,
    inbounds::Lifted,
) where {Nw,P<:NDualEltype}
    v = getfield(primal(x), primal(name), primal(inbounds))
    return Lifted{P,Nw}(v, _scalar_ndual(v, tangent(x).partials[]))
end
function rrule!!(
    f::CoDual{typeof(getfield)}, x::CoDual{P,<:StandardFDataType}, name::CoDual
) where {P}
    if tangent_type(P) == NoTangent
        # TODO(#1295): a `NoTangent` parent can hold a differentiable field, so this mints fresh
        # fdata that does not alias the storage the pass already seeded, and drops a contribution.
        y = uninit_fcodual(getfield(primal(x), primal(name)))
        return y, NoPullback(f, x, name)
    elseif !ismutabletype(P)
        # Immutable structs can update the selected field directly without going through lgetfield.
        dx_r = lazy_zero_rdata(primal(x))
        _name = primal(name)
        function immutable_lgetfield_pb!!(dy)
            return NoRData(), increment_field!!(instantiate(dx_r), dy, _name), NoRData()
        end
        yp = getfield(primal(x), _name)
        y = CoDual(yp, _get_fdata_field(primal(x), tangent(x), _name))
        return y, immutable_lgetfield_pb!!
    else
        return rrule!!(uninit_fcodual(lgetfield), x, uninit_fcodual(Val(primal(name))))
    end
end

function rrule!!(
    f::CoDual{typeof(getfield)}, x::CoDual{P,F}, name::CoDual, order::CoDual
) where {P,F<:StandardFDataType}
    if tangent_type(P) == NoTangent
        # TODO(#1295): a `NoTangent` parent can hold a differentiable field, so this mints fresh
        # fdata that does not alias the storage the pass already seeded, and drops a contribution.
        y = uninit_fcodual(getfield(primal(x), primal(name)))
        return y, NoPullback(f, x, name, order)
    elseif !ismutabletype(P)
        # The ordered immutable case can use the same direct field update path.
        dx_r = lazy_zero_rdata(primal(x))
        _name = primal(name)
        function immutable_lgetfield_pb!!(dy)
            tmp = increment_field!!(instantiate(dx_r), dy, _name)
            return NoRData(), tmp, NoRData(), NoRData()
        end
        yp = getfield(primal(x), _name, primal(order))
        y = CoDual(yp, _get_fdata_field(primal(x), tangent(x), _name))
        return y, immutable_lgetfield_pb!!
    else
        literal_name = uninit_fcodual(Val(primal(name)))
        literal_order = uninit_fcodual(Val(primal(order)))
        return rrule!!(uninit_fcodual(lgetfield), x, literal_name, literal_order)
    end
end

# # Highly specialised rrule to handle tuples of DataTypes.
# function rrule!!(::CoDual{typeof(getfield)}, value::CoDual{P}, name::CoDual) where {P<:NTuple{<:Any, DataType}}
#     pb!! = NoPullback((NoRData(), NoRData(), NoRData(), NoRData()))
#     y = CoDual{DataType, NoFData}(getfield(primal(value), primal(name)), NoFData())
#     return y, pb!!
# end
# function rrule!!(::CoDual{typeof(getfield)}, value::CoDual{P}, name::CoDual, order::CoDual) where {P<:NTuple{<:Any, DataType}}
#     pb!! = NoPullback((NoRData(), NoRData(), NoRData(), NoRData()))
#     y = CoDual{DataType, NoFData}(getfield(primal(value), primal(name), primal(order)), NoFData())
#     return y, pb!!
# end

@zero_derivative MinimalCtx Tuple{typeof(getglobal),Any,Any}

# invoke

@zero_derivative MinimalCtx Tuple{typeof(isa),Any,Any}
@zero_derivative MinimalCtx Tuple{typeof(isdefined),Vararg}

# modifyfield!

@zero_derivative MinimalCtx Tuple{typeof(nfields),Any}

# replacefield!

function frule!!(
    ::Lifted{typeof(setfield!),Nw}, value::Lifted, name::Lifted, x::Lifted
) where {Nw}
    nm = primal(name)
    setfield!(primal(value), nm, primal(x))
    # Normalise an integer field index to its symbol name for the symbol-keyed `MutableDual` V
    # backing `NamedTuple` (the primal `setfield!` above already accepts the integer index).
    sym = nm isa Integer ? fieldname(typeof(primal(value)), nm) : nm
    _setfield_tangent!(tangent(value), sym, tangent(x))
    return x
end
# Array and RefValue Vs are immutable wrappers, so _setfield_tangent! cannot
# update them. Delegate to lsetfield!'s partials/memref-aliasing path, normalising
# field indices to symbols (Array :ref/:size, RefValue :x).
@inline function frule!!(
    ::Lifted{typeof(setfield!),Nw},
    value::Lifted{<:Union{Array,Base.RefValue}},
    name::Lifted,
    x::Lifted,
) where {Nw}
    nm = primal(name)
    sym = nm isa Integer ? fieldname(typeof(primal(value)), nm) : nm
    return frule!!(
        zero_lifted(Val(Nw), lsetfield!), value, zero_lifted(Val(Nw), Val(sym)), x
    )
end
# MutableDual updates its backing NamedTuple, not its own wrapper properties.
# NoDual/NoTangent need no write; other Vs delegate through setproperty!.
@inline _setfield_tangent!(::Union{NoDual,NoTangent}, _, _) = nothing
@inline function _setfield_tangent!(tv::MutableDual, nm, vx)
    nt = getfield(tv, :fields)
    v_i = _coerce_backing_field(fieldtype(typeof(nt), nm), vx)
    # setfield! does not convert, and NamedTuple is invariant: merge narrows
    # abstract fields, so convert back to the stored NamedTuple type.
    setfield!(tv, :fields, convert(typeof(nt), merge(nt, NamedTuple{(nm,)}((v_i,)))))
    return nothing
end
@inline _setfield_tangent!(tv, nm, vx) = (setproperty!(tv, nm, vx); nothing)
function rrule!!(::CoDual{typeof(setfield!)}, value::CoDual, name::CoDual, x::CoDual)
    literal_name = uninit_fcodual(Val(primal(name)))
    return rrule!!(uninit_fcodual(lsetfield!), value, literal_name, x)
end

# swapfield!

function frule!!(::Lifted{typeof(throw),Nw}, args::Vararg{Lifted,M}) where {Nw,M}
    throw(tuple_map(primal, args)...)
end
function rrule!!(::CoDual{typeof(throw)}, args::CoDual...)
    throw(map(primal, args)...), _ -> (NoRData(), map(_ -> NoRData(), args)...)
end

# Only defined in v1.12+
@static if isdefined(Core, :throw_methoderror)
    frule!!(::Lifted{typeof(Core.throw_methoderror),Nw}, args::Vararg{Lifted,M}) where {Nw,M} = Core.throw_methoderror(
        tuple_map(primal, args)...
    )
    function rrule!!(::CoDual{typeof(Core.throw_methoderror)}, args::CoDual...)
        return (
            Core.throw_methoderror(map(primal, args)...),
            _ -> (NoRData(), map(_ -> NoRData(), args)...),
        )
    end
end

function frule!!(
    ::Lifted{typeof(Core.throw_inexacterror),Nw}, args::Vararg{Lifted,M}
) where {Nw,M}
    return Core.throw_inexacterror(tuple_map(primal, args)...)
end
function rrule!!(::CoDual{typeof(Core.throw_inexacterror)}, args::CoDual...)
    return (
        Core.throw_inexacterror(map(primal, args)...),
        _ -> (NoRData(), map(_ -> NoRData(), args)...),
    )
end

struct TuplePullback{N} end

@inline (::TuplePullback{N})(dy::Tuple) where {N} = NoRData(), dy...

@inline function (::TuplePullback{N})(::NoRData) where {N}
    return NoRData(), ntuple(_ -> NoRData(), N)...
end

@inline tuple_pullback(dy) = NoRData(), dy...

@inline tuple_pullback(dy::NoRData) = NoRData()

function frule!!(f::Lifted{typeof(tuple),Nw}, args::Vararg{Lifted,M}) where {Nw,M}
    primal_output = tuple(tuple_map(primal, args)...)
    # typeof preserves a tuple value's DataType elements; _typeof sharpens them
    # to Type{X}, producing a slot type the value cannot inhabit.
    P_out = typeof(primal_output)
    if dual_type(Val(Nw), _typeof(primal_output)) === NoDual
        return Lifted{P_out,Nw}(primal_output, NoDual())
    else
        return Lifted{P_out,Nw}(primal_output, tuple_map(tangent, args))
    end
end

function rrule!!(f::CoDual{typeof(tuple)}, args::Vararg{Any,N}) where {N}
    primal_output = tuple(map(primal, args)...)
    if tangent_type(_typeof(primal_output)) == NoTangent
        return zero_fcodual(primal_output), NoPullback(f, args...)
    else
        if fdata_type(tangent_type(_typeof(primal_output))) == NoFData
            return zero_fcodual(primal_output), TuplePullback{N}()
        else
            return CoDual(primal_output, tuple(map(tangent, args)...)), TuplePullback{N}()
        end
    end
end

function frule!!(
    ::Lifted{typeof(typeassert),Nw}, x::Lifted{P,Nw}, type::Lifted
) where {Nw,P}
    return Lifted{P,Nw}(typeassert(primal(x), primal(type)), tangent(x))
end
function rrule!!(::CoDual{typeof(typeassert)}, x::CoDual, type::CoDual)
    typeassert_pullback(dy) = NoRData(), dy, NoRData()
    return CoDual(typeassert(primal(x), primal(type)), tangent(x)), typeassert_pullback
end

@zero_derivative MinimalCtx Tuple{typeof(typeof),Any}

function __pointers_to_pointers()
    # Pointer to pointer.
    c_1 = [5.0]
    c_2 = [3.0, 4.0]
    c = [pointer(c_1), pointer(c_2)]

    c_new_val = [6.0, 5.0, 4.0]
    cs = (c_1, c_2, c, c_new_val)

    # Tangents of pointers to pointers.
    dc_1 = copy(c_1)
    dc_2 = copy(c_2)
    dc = [pointer(dc_1), pointer(dc_2)]
    dc_new_val = randn(3)
    dcs = (dc_1, dc_2, dc, dc_new_val)
    return cs, dcs
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:builtins})
    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    _a = Vector{Vector{Float64}}(undef, 3)
    _a[1] = [5.4, 4.23, -0.1, 2.1]

    x = randn(5)
    p = pointer(x)
    dx = randn(5)
    dp = pointer(dx)

    y = [1, 2, 3]
    q = pointer(y)
    dy = zero_tangent(y)
    dq = pointer(dy)

    cs, dcs = __pointers_to_pointers()
    (c_1, c_2, c, c_new_val) = cs
    (dc_1, dc_2, dc, dc_new_val) = dcs

    # Slightly wider range for builtins whose performance is known not to be great.
    _range = (lb=1e-3, ub=200.0)
    memory = Any[_x, _dx, _a, x, p, dx, dp, y, q, dy, dq, cs..., dcs...]

    test_cases = Any[

        # Core.Intrinsics:
        (false, :stability, nothing, IntrinsicsWrappers.abs_float, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.abs_float, 5.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.add_float, 4.0, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.add_float, 4.0f0, 5.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.add_float_fast, 4.0, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.add_float_fast, 4.0f0, 5.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.add_int, 1, 2),
        (false, :stability, nothing, IntrinsicsWrappers.and_int, 2, 3),
        (
            false,
            :stability,
            nothing,
            IntrinsicsWrappers.ashr_int,
            123456,
            0x0000000000000020,
        ),
        # atomic_fence -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointermodify -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerreplace -- NEEDS IMPLEMENTING AND TESTING
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.atomic_pointerref,
            CoDual(p, dp),
            :monotonic,
        ),
        (
            true,
            :stability,
            # Reverse-only: raw atomic pointer-element loads lack coherent forward storage.
            (skip_forward=true,),
            IntrinsicsWrappers.atomic_pointerref,
            CoDual(pointer(c), pointer(dc)),
            :monotonic,
        ),
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.atomic_pointerref,
            CoDual(q, dq),
            :monotonic,
        ),
        # Load-only ordering: the pullback's tangent store must not reuse it.
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.atomic_pointerref,
            CoDual(p, dp),
            :acquire,
        ),
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.atomic_pointerset,
            CoDual(p, dp),
            1.0,
            :monotonic,
        ),
        # Store-only ordering: the rule's save/restore loads must not reuse it.
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.atomic_pointerset,
            CoDual(p, dp),
            1.0,
            :release,
        ),
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.atomic_pointerset,
            CoDual(pointer(c), pointer(dc)),
            CoDual(pointer(c_new_val), pointer(dc_new_val)),
            :monotonic,
        ),
        # atomic_pointerswap -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, IntrinsicsWrappers.bitcast, Int64, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.bswap_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.ceil_llvm, 4.1),
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.__cglobal,
            Val{:jl_uv_stdout}(),
            Ptr{Cvoid},
        ),
        (false, :stability, nothing, IntrinsicsWrappers.checked_sadd_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_sdiv_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_smul_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_srem_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_ssub_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_uadd_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_udiv_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_umul_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_urem_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_usub_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, 5.0, -3.0),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, -5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, -5.0, -3.0),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, 5.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, 5.0f0, -3.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, -5.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, -5.0f0, -3.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.ctlz_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.ctpop_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.cttz_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.div_float, 5.0, 3.0),
        (false, :stability, nothing, IntrinsicsWrappers.div_float_fast, 5.0, 3.0),
        (false, :stability, nothing, IntrinsicsWrappers.div_float, 5.0f0, 3.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.div_float_fast, 5.0f0, 3.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float, 4.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float, 5.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float, 4.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float_fast, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float_fast, 4.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float_fast, 5.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float_fast, 4.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.eq_int, 4, 4),
        (false, :stability, nothing, IntrinsicsWrappers.flipsign_int, 4, -3),
        (false, :stability, nothing, IntrinsicsWrappers.floor_llvm, 4.1),
        (false, :stability, nothing, IntrinsicsWrappers.fma_float, 5.0, 4.0, 3.0),
        (false, :stability, nothing, IntrinsicsWrappers.fma_float, 5.0f0, 4.0f0, 3.0f0),
        # Validate cross-precision derivatives with FD, retaining stability/allocation checks.
        (false, :stability_and_allocs, nothing, IntrinsicsWrappers.fpext, Float64, 5.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.fpiseq, 4.1, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.fpiseq, 4.0f1, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.fptosi, UInt32, 4.1),
        (false, :stability, nothing, IntrinsicsWrappers.fptoui, Int32, 4.1),
        (false, :stability, nothing, IntrinsicsWrappers.fptrunc, Float32, 5.0),
        (true, :stability, nothing, IntrinsicsWrappers.have_fma, Float64),
        (false, :stability, nothing, IntrinsicsWrappers.le_float, 4.1, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.le_float, 4.0f1, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.le_float_fast, 4.1, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.le_float_fast, 4.0f1, 4.0f0),
        # llvm_call -- NEEDS IMPLEMENTING AND TESTING
        (
            false,
            :stability,
            nothing,
            IntrinsicsWrappers.lshr_int,
            1308622848,
            0x0000000000000018,
        ),
        (false, :stability, nothing, IntrinsicsWrappers.lt_float, 4.1, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.lt_float, 4.0f1, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.lt_float_fast, 4.1, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.lt_float_fast, 4.0f1, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.mul_float, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.mul_float, 5.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.mul_float_fast, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.mul_float_fast, 5.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.mul_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.muladd_float, 5.0, 4.0, 3.0),
        (false, :stability, nothing, IntrinsicsWrappers.muladd_float, 5.0f0, 4.0f0, 3.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.ne_float, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.ne_float, 5.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.ne_float_fast, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.ne_float_fast, 5.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.ne_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.ne_int, 5, 5),
        (false, :stability, nothing, IntrinsicsWrappers.neg_float, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.neg_float, 5.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.neg_float_fast, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.neg_float_fast, 5.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.neg_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.not_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.or_int, 5, 5),
        (true, :stability, nothing, IntrinsicsWrappers.pointerref, CoDual(p, dp), 2, 1),
        (true, :stability, nothing, IntrinsicsWrappers.pointerref, CoDual(q, dq), 2, 1),
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.pointerset,
            CoDual(p, dp),
            5.0,
            2,
            1,
        ),
        (true, :stability, nothing, IntrinsicsWrappers.pointerset, CoDual(q, dq), 1, 2, 1),
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.pointerset,
            CoDual(pointer(c), pointer(dc)),
            CoDual(pointer(c_new_val), pointer(dc_new_val)),
            1,
            1,
        ),
        # rem_float -- untested and unimplemented because seemingly unused on master
        # rem_float_fast -- untested and unimplemented because seemingly unused on master
        (false, :stability, nothing, IntrinsicsWrappers.rint_llvm, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.sdiv_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.sext_int, Int64, Int32(1308622848)),
        (
            false,
            :stability,
            nothing,
            IntrinsicsWrappers.shl_int,
            1308622848,
            0xffffffffffffffe8,
        ),
        (false, :stability, nothing, IntrinsicsWrappers.sitofp, Float64, 0),
        (false, :stability, nothing, IntrinsicsWrappers.sle_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.slt_int, 4, 5),
        (false, :stability, nothing, IntrinsicsWrappers.sqrt_llvm, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.sqrt_llvm, 5.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.sqrt_llvm_fast, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.sqrt_llvm_fast, 5.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.srem_int, 4, 1),
        (false, :stability, nothing, IntrinsicsWrappers.sub_float, 4.0, 1.0),
        (false, :stability, nothing, IntrinsicsWrappers.sub_float, 4.0f0, 1.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.sub_float_fast, 4.0, 1.0),
        (false, :stability, nothing, IntrinsicsWrappers.sub_float_fast, 4.0f0, 1.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.sub_int, 4, 1),
        (false, :stability, nothing, IntrinsicsWrappers.trunc_int, UInt8, 78),
        (false, :stability, nothing, IntrinsicsWrappers.trunc_llvm, 5.1),
        (false, :stability, nothing, IntrinsicsWrappers.udiv_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.uitofp, Float16, 4),
        (false, :stability, nothing, IntrinsicsWrappers.ule_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.ult_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.urem_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.xor_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.zext_int, Int64, 0xffffffff),

        # Non-intrinsic built-ins:
        # Core._abstracttype -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, nothing, __vec_to_tuple, [1.0]),
        (false, :none, nothing, __vec_to_tuple, Any[1.0]),
        (false, :none, nothing, __vec_to_tuple, Any[[1.0]]),
        (false, :none, nothing, __vec_to_tuple, [1]),
        # Core._apply_pure -- NEEDS IMPLEMENTING AND TESTING
        # Core._call_in_world -- NEEDS IMPLEMENTING AND TESTING
        # Core._call_in_world_total -- NEEDS IMPLEMENTING AND TESTING
        # Core._call_latest -- NEEDS IMPLEMENTING AND TESTING
        # Core._compute_sparams -- CONSIDER TESTING
        # Core._equiv_typedef -- NEEDS IMPLEMENTING AND TESTING
        # Core._expr -- NEEDS IMPLEMENTING AND TESTING
        # Core._primitivetype -- NEEDS IMPLEMENTING AND TESTING
        # Core._setsuper! -- NEEDS IMPLEMENTING AND TESTING
        # Core._structtype -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, _range, Core._svec_ref, svec(5, 4), 2),
        (false, :none, _range, Core._svec_ref, svec(5, 4.0), 2),
        (false, :none, _range, Core._svec_ref, svec(5, randn(rng_ctor(1234), 2, 3)), 2),
        (false, :none, (lb=1e-3, ub=500.0), Core.svec, 5, 4.0, randn(rng_ctor(1234), 2, 3)),
        # check svec with no arguments
        (false, :none, _range, Core.svec),
        # check svec with an argument that has both fdata and rdata
        (
            false,
            :none,
            (lb=1e-3, ub=500.0),
            Core.svec,
            (5, 4.0, randn(rng_ctor(1234), 2, 3)),
        ),
        # Core._typebody! -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, <:, Float64, Int),
        (false, :stability, nothing, <:, Any, Float64),
        (false, :stability, nothing, <:, Float64, Any),
        (false, :stability, nothing, ===, 5.0, 4.0),
        (false, :stability, nothing, ===, 5.0, randn(5)),
        (false, :stability, nothing, ===, randn(5), randn(3)),
        (false, :stability, nothing, ===, 5.0, 5.0),
        (true, :stability, nothing, Core._typevar, :T, Union{}, Any),
        (false, :none, _range, Core.apply_type, Vector, Float64),
        (false, :none, _range, Core.apply_type, Array, Float64, 2),
        (false, :none, (lb=1e-3, ub=100), compilerbarrier, :type, 5.0),
        # Core.const_arrayref -- NEEDS IMPLEMENTING AND TESTING
        # Core.donotdelete -- NEEDS IMPLEMENTING AND TESTING
        # Core.finalizer -- NEEDS IMPLEMENTING AND TESTING
        # Core.get_binding_type -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, nothing, Core.ifelse, true, randn(5), 1),
        (false, :none, nothing, Core.ifelse, false, randn(5), 2),
        (false, :stability, nothing, Core.ifelse, true, 5, 4),
        (false, :stability, nothing, Core.ifelse, false, true, false),
        (false, :stability, nothing, Core.ifelse, false, 1.0, 2.0),
        (false, :stability, nothing, Core.ifelse, true, 1.0, 2.0),
        (false, :stability, nothing, Core.ifelse, false, randn(5), randn(3)),
        (false, :stability, nothing, Core.ifelse, true, randn(5), randn(3)),
        # Core.set_binding_type! -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, Core.sizeof, Float64),
        (false, :stability, nothing, Core.sizeof, randn(5)),
        (false, :stability, nothing, applicable, sin, Float64),
        (false, :stability, nothing, applicable, sin, Type),
        (false, :stability, nothing, applicable, +, Type, Float64),
        (false, :stability, nothing, applicable, +, Float64, Float64),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, StructFoo, :a),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, StructFoo, :b),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, MutableFoo, :a),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, MutableFoo, :b),
        # These primals are tiny builtins, so keep some ratio headroom for timing noise.
        (true, :none, (lb=1e-3, ub=350), getfield, StructFoo(5.0), :a),
        (false, :none, (lb=1e-3, ub=350), getfield, StructFoo(5.0, randn(5)), :a),
        (false, :none, (lb=1e-3, ub=350), getfield, StructFoo(5.0, randn(5)), :b),
        # Tuple-valued type fields need typeof, not _typeof's uninhabitable sharpening.
        (false, :none, nothing, getfield, Pair(1.0, (Float64, Int)), :second),
        (false, :none, nothing, getfield, Pair(1.0, (Float64, Int)), :second, true),
        # Integer field lookup still merits a slightly wider bound than symbol lookup.
        (true, :none, (lb=1e-3, ub=500), getfield, StructFoo(5.0), 1),
        (false, :none, (lb=1e-3, ub=500), getfield, StructFoo(5.0, randn(5)), 1),
        (false, :none, (lb=1e-3, ub=500), getfield, StructFoo(5.0, randn(5)), 2),
        (true, :none, _range, getfield, MutableFoo(5.0), :a),
        (false, :none, _range, getfield, MutableFoo(5.0, randn(5)), :b),
        (false, :stability_and_allocs, nothing, getfield, UnitRange{Int}(5:9), :start),
        (false, :stability_and_allocs, nothing, getfield, UnitRange{Int}(5:9), :stop),
        (false, :stability_and_allocs, nothing, getfield, (5.0,), 1),
        (false, :stability_and_allocs, nothing, getfield, (5.0, 4.0), 1),
        (false, :stability_and_allocs, nothing, getfield, (5.0,), 1, false),
        (false, :stability_and_allocs, nothing, getfield, (5.0, 4.0), 1, false),
        (false, :stability_and_allocs, nothing, getfield, (1,), 1, false),
        (false, :stability_and_allocs, nothing, getfield, (1, 2), 1),
        (false, :stability_and_allocs, nothing, getfield, (a=5, b=4), 1),
        (false, :stability_and_allocs, nothing, getfield, (a=5, b=4), 2),
        # getfield on Tuple{Type{T},...} with integer index: the primal is trivial but the
        # rule triggers type-system dispatch, making the ratio large. Loose bounds are intentional.
        (false, :none, (lb=1e-3, ub=200), getfield, (Float64, Float64), 1),
        (false, :none, (lb=1e-3, ub=250), getfield, (Float64, Float64), 2, false),
        # The reverse oracle must convert NoTangent to NoFData via to_fwds.
        # This row checks that conversion, although the derivative is zero.
        (
            false,
            :none,
            (
                oracle=(value=1, deriv=(NoRData(), NoRData(), NoRData())),
                output_tangent=NoTangent(),
                mode=ReverseMode,
            ),
            getfield,
            (1, 2),
            1,
        ),
        (false, :none, _range, getfield, (a=5.0, b=4), 1),
        (false, :none, _range, getfield, (a=5.0, b=4), 2),
        (false, :none, _range, getfield, UInt8, :name),
        (false, :none, _range, getfield, UInt8, :super),
        (true, :none, _range, getfield, UInt8, :layout),
        (false, :none, _range, getfield, UInt8, :hash),
        (false, :none, _range, getfield, UInt8, :flags),
        # getglobal requires compositional testing, because you can't deepcopy a module
        # invoke -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, isa, 5.0, Float64),
        (false, :stability, nothing, isa, 1, Float64),
        (false, :stability, nothing, isdefined, MutableFoo(5.0, randn(5)), :sim),
        (false, :stability, nothing, isdefined, MutableFoo(5.0, randn(5)), :a),
        # modifyfield! -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, nfields, MutableFoo),
        (false, :stability, nothing, nfields, StructFoo),
        # replacefield! -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, _range, setfield!, MutableFoo(5.0, randn(5)), :a, 4.0),
        (false, :none, nothing, setfield!, MutableFoo(5.0, randn(5)), :b, randn(5)),
        (false, :none, _range, setfield!, MutableFoo(5.0, randn(5)), 1, 4.0),
        (false, :none, _range, setfield!, MutableFoo(5.0, randn(5)), 2, randn(5)),
        (false, :none, _range, setfield!, NonDifferentiableFoo(5, false), 1, 4),
        (false, :none, _range, setfield!, NonDifferentiableFoo(5, true), 2, false),
        # runtime-name setfield! on a Ref (V is NDualRef): delegates to the lsetfield! frule.
        (false, :none, _range, setfield!, Ref(5.0), :x, 4.0),
        (false, :none, _range, setfield!, Ref(5.0), 1, 4.0),
        # runtime-name getfield on a Ref (V is NDualRef) — the read counterpart; rebuilds the
        # scalar V via _scalar_ndual. Real + complex element, by name and by index.
        (false, :none, _range, getfield, Ref(5.0), :x),
        (false, :none, _range, getfield, Ref(5.0), 1),
        (false, :none, _range, getfield, Ref(5.0), :x, false),
        (false, :none, _range, getfield, Ref(1.0 + 2.0im), :x),
        # swapfield! -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability_and_allocs, nothing, tuple, 5.0, 4.0),
        (false, :stability_and_allocs, nothing, tuple, randn(5), 5.0),
        (false, :stability_and_allocs, nothing, tuple, randn(5), randn(4)),
        (false, :stability_and_allocs, nothing, tuple, 5.0, randn(1)),
        (false, :stability_and_allocs, nothing, tuple),
        (false, :stability_and_allocs, nothing, tuple, 1),
        (false, :stability_and_allocs, nothing, tuple, 1, 5),
        (false, :stability_and_allocs, nothing, tuple, 1.0, (5,)),
        (false, :stability, nothing, typeassert, 5.0, Float64),
        (false, :stability, nothing, typeassert, randn(5), Vector{Float64}),
        (false, :stability, nothing, typeof, 5.0),
        (false, :stability, nothing, typeof, randn(5)),
        (true, :stability, nothing, unsafe_wrap, Array, CoDual(p, dp), 1),
        (true, :stability, nothing, unsafe_wrap, Vector{Float64}, CoDual(p, dp), 1),
        (
            false,
            :none,
            (
                throws=(ArgumentError, "cannot preserve tangent aliasing"),
                mode=ForwardMode,
                chunk_size=2,
            ),
            unsafe_wrap,
            Array,
            Lifted{typeof(pointer(c)),2}(
                pointer(c), (pointer(dc), pointer(dc) + sizeof(Ptr{Float64}))
            ),
            1,
        ),
        (true, :stability, nothing, unsafe_wrap, Array, CoDual(q, dq), 3),
    ]

    if VERSION > v"1.12-"
        fs = [
            IntrinsicsWrappers.min_float,
            IntrinsicsWrappers.min_float_fast,
            IntrinsicsWrappers.max_float,
            IntrinsicsWrappers.max_float_fast,
        ]
        for P in [Float32, Float64], f in fs
            push!(test_cases, (false, :stability_and_allocs, nothing, f, P(5.0), P(4.0)))
            push!(test_cases, (false, :stability_and_allocs, nothing, f, P(2.0), P(3.1)))
        end
    end

    # Cancellation distinguishes fused rounding only with an exact value oracle.
    # Pin d/da with CoDual seeds. Forward-only: this checks the inner-value invariant;
    # ordinary rows above cover reverse.
    let a = 1.0 + 2.0^-27, b = 1.0 + 2.0^-27, z = -((1.0 + 2.0^-27) * (1.0 + 2.0^-27))
        for f in (IntrinsicsWrappers.fma_float, IntrinsicsWrappers.muladd_float)
            push!(
                test_cases,
                (
                    false,
                    :none,
                    (oracle=(value=fma(a, b, z), deriv=b), skip_reverse=true),
                    f,
                    CoDual(a, 1.0),
                    CoDual(b, 0.0),
                    CoDual(z, 0.0),
                ),
            )
        end
    end

    # Unlike Base.sqrt, the intrinsics return NaN for negative inputs. Pin the oracle:
    # finite differences cannot resolve NaN values or the singular derivative at zero.
    # Zero input seeds also check that inactive lanes stay zero in both chunk widths.
    for P in (Float16, Float32, Float64),
        f in (IntrinsicsWrappers.sqrt_llvm, IntrinsicsWrappers.sqrt_llvm_fast),
        x in (P(-1), P(0))

        y = x < 0 ? P(NaN) : P(0)
        seed = x < 0 ? P(1) : P(0)
        rvs = (NoRData(), x < 0 ? P(NaN) : P(0))
        opts = (oracle=(value=y, deriv=(fwd=P(0), rvs=rvs)), output_tangent=seed)
        push!(test_cases, (false, :stability, opts, f, CoDual(x, P(0))))
    end

    # NaN selection must carry the returned operand's partials, not an ordering result.
    @static if VERSION >= v"1.12.0-rc2"
        for (f, tie_deriv) in
            ((IntrinsicsWrappers.max_float, 1.0), (IntrinsicsWrappers.min_float, 3.0))
            for (av, bv, want) in ((NaN, 1.0, 1.0), (1.0, NaN, 2.0))
                push!(
                    test_cases,
                    (
                        false,
                        :none,
                        (oracle=(value=NaN, deriv=want), skip_reverse=true),
                        f,
                        CoDual(av, 1.0),
                        CoDual(bv, 2.0),
                    ),
                )
            end
            # Non-NaN: the value tracks the selected operand and its partial comes with it.
            push!(
                test_cases,
                (
                    false,
                    :none,
                    (oracle=(value=f(2.0, 1.0), deriv=tie_deriv), skip_reverse=true),
                    f,
                    CoDual(2.0, 1.0),
                    CoDual(1.0, 3.0),
                ),
            )
        end
        # Reverse must credit the same NaN operand. Plain primals need only an output seed.
        for (f, want) in (
            (IntrinsicsWrappers.max_float, (NoRData(), 1.0, 0.0)),
            (IntrinsicsWrappers.min_float, (NoRData(), 1.0, 0.0)),
        )
            push!(
                test_cases,
                (
                    false,
                    :none,
                    (oracle=(value=NaN, deriv=want), output_tangent=1.0, mode=ReverseMode),
                    f,
                    NaN,
                    1.0,
                ),
            )
        end
        # min_float_fast ties select the second operand, unlike _ndual_pick_min.
        # Within FastMath's contract (NaN excluded), max selections agree on ties.
        push!(
            test_cases,
            (
                false,
                :none,
                (oracle=(value=1.0, deriv=2.0), skip_reverse=true),
                IntrinsicsWrappers.min_float_fast,
                CoDual(1.0, 1.0),
                CoDual(1.0, 2.0),
            ),
        )
    end
    throwing_rows, throwing_memory = _builtins_throwing_rows()
    test_cases = vcat(Any[test_cases...], Any[_throwing_row(c) for c in throwing_rows])
    memory = vcat(Any[memory...], throwing_memory)
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:builtins})
    cs, dcs = __pointers_to_pointers()
    (c_1, c_2, c, c_new_val) = cs
    (dc_1, dc_2, dc, dc_new_val) = dcs

    function f_pointerset(x)
        c_1 = Ref(x)
        c_2 = Ref(x * 2.0)
        p = Ref(Base.unsafe_convert(Ptr{Float64}, c_1))
        GC.@preserve c_1 c_2 p begin
            pointerset(
                Base.unsafe_convert(Ptr{Ptr{Float64}}, p),
                Base.unsafe_convert(Ptr{Float64}, c_2),
                1,
                1,
            )
            unsafe_load(p[])
        end
    end

    function f_atomic_pointerset(x)
        c_1 = Ref(x)
        c_2 = Ref(x * 2.0)
        p = Ref(Base.unsafe_convert(Ptr{Float64}, c_1))
        GC.@preserve c_1 c_2 p begin
            Core.Intrinsics.atomic_pointerset(
                Base.unsafe_convert(Ptr{Ptr{Float64}}, p),
                Base.unsafe_convert(Ptr{Float64}, c_2),
                :monotonic,
            )
            unsafe_load(p[])
        end
    end

    # A Cvoid hop must still allow narrowing to zero-tangent elements.
    # Interface-only: the result steps with the bit pattern, invalidating FD.
    function narrow_through_cvoid(b::Vector{Float64}, x::Float64)
        return GC.@preserve b x * Float64(unsafe_load(Ptr{UInt8}(Ptr{Cvoid}(pointer(b)))))
    end
    # Own-address reads are refused in forward mode but supported in reverse.
    # Shifting an erased pointer must preserve its erased tangent element type.
    ref_objref_roundtrip(x) = pointerref(
        bitcast(Ptr{Float64}, pointer_from_objref(Ref(x))), 1, 1
    )
    function shift_through_cvoid(b::Vector{Float64}, x::Float64)
        return GC.@preserve b x * unsafe_load(Ptr{Float64}(Ptr{Cvoid}(pointer(b)) + 8))
    end
    # Boxed fields use reference slots, not sizeof(fieldtype). Only compare the address:
    # returning it would differ across runs over equal inputs.
    function objref_boxed_field(r::Base.RefValue{Any}, x::Float64)
        return GC.@preserve r (pointer_from_objref(r) === C_NULL ? 0.0 : x)
    end

    test_cases = Any[
        (true, :none, nothing, narrow_through_cvoid, [1.0, 2.0], 2.0),
        # `skip_chunked`: takes a raw pointer to a float array (see the `unsafe_copyto_tester`
        # rows in `foreigncall.jl`); the lane stride leaves no buffer a pointer can address.
        (false, :none, (skip_chunked=true,), shift_through_cvoid, [1.0, 2.0], 2.0),
        (false, :none, nothing, objref_boxed_field, Base.RefValue{Any}(1.0), 2.0),
        (false, :none, nothing, _apply_iterate_equivalent, Base.iterate, *, 5.0, 4.0),
        (false, :none, nothing, _apply_iterate_equivalent, Base.iterate, *, (5.0, 4.0)),
        (false, :none, nothing, _apply_iterate_equivalent, Base.iterate, *, [5.0, 4.0]),
        (false, :none, nothing, _apply_iterate_equivalent, Base.iterate, *, [5.0], (4.0,)),
        (false, :none, nothing, _apply_iterate_equivalent, Base.iterate, *, 3, (4.0,)),
        (
            # 33 arguments is the critical length at which splatting gives up on inferring,
            # and backs off to `Core._apply_iterate`. It's important to check this in order
            # to verify that we don't wind up in an infinite recursion.
            false,
            :none,
            nothing,
            _apply_iterate_equivalent,
            Base.iterate,
            +,
            randn(33),
        ),
        (
            # Check that Core._apply_iterate gets lifted to _apply_iterate_equivalent.
            false,
            :none,
            nothing,
            x -> +(x...),
            randn(33),
        ),
        # Forward refuses own-address reads because the optimiser can elide the primal
        # Ref store. Reverse remains supported and needs a separate correctness row.
        (
            false,
            :none,
            (throws=(ArgumentError, "invisible to the optimiser"), mode=ForwardMode),
            ref_objref_roundtrip,
            5.0,
        ),
        (false, :none, (mode=ReverseMode,), ref_objref_roundtrip, 5.0),
        (
            # skip_chunked: raw pointers cannot address stride-N lanes of a float array's
            # partials block (as in foreigncall.jl's unsafe_copyto_tester).
            false,
            :none,
            (skip_chunked=true,),
            (v, x) -> (pointerset(pointer(x), v, 2, 1); x),
            3.0,
            randn(5),
        ),
        (
            false,
            :none,
            nothing,
            x -> (pointerset(pointer(x), UInt8(3), 2, 1); x),
            rand(UInt8, 5),
        ),
        # Reverse-only: pointer stores into pointer arrays have an element-wise
        # forward layout that cannot accept bare lane pointers.
        (
            true,
            :none,
            (skip_forward=true,),
            (x, v) ->
                unsafe_wrap(Array, pointerset(pointer(x), pointer(v), 1, 1), length(x)),
            CoDual(c, dc),
            CoDual(c_new_val, dc_new_val),
        ),
        (
            true,
            :none,
            (skip_forward=true,),
            (x, v) -> unsafe_wrap(
                Array,
                Core.Intrinsics.atomic_pointerset(pointer(x), pointer(v), :monotonic),
                length(x),
            ),
            CoDual(c, dc),
            CoDual(c_new_val, dc_new_val),
        ),
        # Reverse-only for the same pointer-store layout restriction above.
        # test/rules/builtins.jl also checks explicit value_and_gradient!! results.
        (true, :none, (skip_forward=true,), f_pointerset, CoDual(3.0, 1.0)),
        (true, :none, (skip_forward=true,), f_atomic_pointerset, CoDual(3.0, 1.0)),
        (
            false,
            :none,
            nothing,
            x -> (GC.@preserve x Tuple(unsafe_wrap(Array, pointer(x), length(x)))),
            [1, 2],
        ),
        (false, :none, nothing, getindex, randn(5), [1, 1]),
        (false, :none, nothing, getindex, randn(5), [1, 2, 2]),
        (false, :none, nothing, setindex!, randn(5), [4.0, 5.0], [1, 1]),
        (false, :none, nothing, setindex!, randn(5), [4.0, 5.0, 6.0], [1, 2, 2]),
    ]
    # Repeated arguments must share one tangent to accumulate the full 2x gradient.
    let x = randn(rng_ctor(1), 3)
        push!(test_cases, (false, :none, nothing, (a, b) -> sum(a .* b), x, x))
    end
    return test_cases, Any[]
end

function _builtins_throwing_rows()
    # atomic_pointerset through a differentiable element with an element-wise (incoherent)
    # per-lane V must hit the loud guard, mirroring pointerset.
    xv = [1.0]
    ptr = pointer(xv)
    pslot = Lifted{Ptr{Float64},1}(ptr, (Ptr{Tuple{Float64}}(UInt(ptr)),))
    # NoDual on a scalar differentiable pointee means missing partial storage.
    ndslot = Lifted{Ptr{Float64},1,NoDual}(ptr, NoDual())
    # A non-NULL uninit_* placeholder aliases primal bytes; supply it explicitly
    # because forward Ptr seeding cannot express this slot.
    phslot = Lifted{Ptr{Float64},1}(ptr, (ptr,))
    rethrows_its_argument(e) = throw(e)
    # Other differentiable pointees match the primitive but no coherent wrap rule;
    # require a deliberate error, not a MethodError.
    incoherent_slot = let S = Tuple{Float64,Float64}, N = 2
        v = ntuple(_ -> Ptr{Mooncake.tangent_type(S)}(0), N)
        Lifted{Ptr{S},N,typeof(v)}(Ptr{S}(0), v)
    end
    # Retyped bytes have no tangent storage. Vector keeps these tests valid on 1.10.
    function unsafe_wrap_retyped_bytes(b, x)
        return GC.@preserve b begin
            x * unsafe_wrap(Array{Float64,1}, Ptr{Float64}(pointer(b)), 1)[1]
        end
    end
    # Reverse seeds a bare Ptr with the uninit_* placeholder automatically.
    load_through_bare_ptr(q) = unsafe_load(q) * 2.0
    function atomic_load_retyped_bytes(b, x)
        return GC.@preserve b begin
            x * unsafe_load(Ptr{Float64}(pointer(b)), :monotonic)
        end
    end
    # A Cvoid hop must not launder widening across tangent element boundaries.
    function laundered_retyped_load(b, x)
        return GC.@preserve b x * unsafe_load(Ptr{Float64}(Ptr{Cvoid}(pointer(b))))
    end
    # Equal byte widths also cannot justify re-typing reference slots as inline values.
    cases = Any[
        (
            ArgumentError,
            IntrinsicsWrappers.pointerref,
            (ndslot, 1, 1),
            (; mode=ForwardMode),
        ),
        (
            (ArgumentError, "cannot load from or store to"),
            unsafe_wrap,
            (Array{Float64,1}, phslot, 1),
            (; mode=ForwardMode),
        ),
        (
            ArgumentError,
            IntrinsicsWrappers.atomic_pointerref,
            (ndslot, :monotonic),
            (; mode=ForwardMode),
        ),
        (
            ArgumentError,
            IntrinsicsWrappers.pointerset,
            (ndslot, 2.0, 1, 1),
            (; mode=ForwardMode),
        ),
        (
            ArgumentError,
            IntrinsicsWrappers.atomic_pointerset,
            (ndslot, 2.0, :monotonic),
            (; mode=ForwardMode),
        ),
        # The same four guards must reject non-NULL placeholders as well as NoDual.
        (
            (ArgumentError, "cannot load from or store to"),
            IntrinsicsWrappers.pointerref,
            (phslot, 1, 1),
            (; mode=ForwardMode),
        ),
        (
            (ArgumentError, "cannot load from or store to"),
            IntrinsicsWrappers.atomic_pointerref,
            (phslot, :monotonic),
            (; mode=ForwardMode),
        ),
        (
            (ArgumentError, "cannot load from or store to"),
            IntrinsicsWrappers.pointerset,
            (phslot, 2.0, 1, 1),
            (; mode=ForwardMode),
        ),
        (
            (ArgumentError, "cannot load from or store to"),
            IntrinsicsWrappers.atomic_pointerset,
            (phslot, 2.0, :monotonic),
            (; mode=ForwardMode),
        ),
        (
            (ArgumentError, "placeholder"),
            load_through_bare_ptr,
            (ptr,),
            (; mode=ReverseMode),
        ),
        # Int/UInt -> Ptr must refuse in both modes: there is no derivative storage.
        (ArgumentError, IntrinsicsWrappers.bitcast, (Ptr{Float64}, UInt(pointer(xv))), (;)),
        (
            ArgumentError,
            IntrinsicsWrappers.atomic_pointerset,
            (pslot, 2.0, :monotonic),
            (; mode=ForwardMode),
        ),
        # Forward `throw` rule must re-raise (the reverse rule is covered by the `throw` rrule cases).
        (ArgumentError, throw, (ArgumentError("hello"),), (;)),
        (AssertionError, throw, (AssertionError("hello"),), (;)),
        # A DERIVED rule must re-raise what the primal threw, not just the primitive.
        (ArgumentError, rethrows_its_argument, (ArgumentError("hello"),), (;)),
        (AssertionError, rethrows_its_argument, (AssertionError("hmmm"),), (;)),
        # `bitcast` to a differentiable type, and an integer reinterpreted as a pointer: neither
        # has a tangent that means anything, so both must refuse.
        (ArgumentError, IntrinsicsWrappers.bitcast, (Float64, 5), (;)),
        (ArgumentError, IntrinsicsWrappers.bitcast, (Ptr{Float64}, 5), (;)),
    ]
    # Refuse at re-typing on every version, including 1.10 where NoTangent storage
    # has a real address rather than the Memory NULL sentinel.
    push!(
        cases,
        (
            (ArgumentError, "no tangent storage"),
            atomic_load_retyped_bytes,
            (zeros(UInt8, 8), 2.0),
            (; mode=ReverseMode),
        ),
        (
            (ArgumentError, "the element type was erased"),
            laundered_retyped_load,
            (Float32[1, 2, 3, 4], 2.0),
            (; mode=ReverseMode),
        ),
        (
            (ArgumentError, "REFERENCES, not inline values"),
            laundered_retyped_load,
            ([[1.0], [2.0]], 2.0),
            (; mode=ReverseMode),
        ),
        # The same re-typing reached through `unsafe_wrap` rather than a load: wrapping it handed a
        # container over unusable storage to the next consumer.
        (
            (ArgumentError, "no tangent storage"),
            unsafe_wrap_retyped_bytes,
            (zeros(UInt8, 8), 2.0),
            (; mode=ReverseMode),
        ),
    )
    push!(
        cases,
        (
            ArgumentError,
            unsafe_wrap,
            (Array, incoherent_slot, (2,)),
            (; mode=ForwardMode, chunk_size=2),
        ),
    )
    return cases, Any[xv]
end
