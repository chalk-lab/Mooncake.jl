
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
    NDualEltype,
    _scalar_ndual

using Core.Intrinsics: atomic_pointerref

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
    partials = ntuple(lane -> unsafe_wrap(Array, p_partials[lane], _dims), Val(Nw))
    D = ndims(primal_arr)
    return Lifted{Array{T,D},Nw}(
        primal_arr, NDualArray{T,Nw,D,Array{T,D}}(primal_arr, partials)
    )
end
# Pointer-to-pointer: wrapping a `Ptr{Ptr{R}}` yields an `Array{Ptr{R}}` whose elements are
# themselves differentiable pointers, so the canonical V is the element-wise
# `Array{NTuple{Nw,Ptr{R}}, D}` (each element holds that pointer's Nw per-lane shadow Ptrs),
# not the `NDualArray` parallel-arrays form above (which only applies to `NDualEltype` elements). Wrap each
# lane's shadow pointer into its own array, then interleave element-wise into the V.
function frule!!(
    ::Lifted{typeof(unsafe_wrap),Nw},
    ::Lifted{<:Type{<:Array},Nw},
    p::Lifted{Ptr{Ptr{R}},Nw,NTuple{Nw,Ptr{Ptr{R}}}},
    dims::Lifted,
) where {Nw,R<:NDualEltype}
    _dims = primal(dims)
    primal_arr = unsafe_wrap(Array, primal(p), _dims)
    p_partials = tangent(p)
    lane_arrays = ntuple(lane -> unsafe_wrap(Array, p_partials[lane], _dims), Val(Nw))
    D = ndims(primal_arr)
    v = similar(primal_arr, NTuple{Nw,Ptr{R}})
    @inbounds for i in eachindex(primal_arr, v)
        v[i] = ntuple(lane -> lane_arrays[lane][i], Val(Nw))
    end
    return Lifted{Array{Ptr{R},D},Nw}(primal_arr, v)
end

# Non-differentiable pointer element: `dual_type(Ptr{T}) === NoDual` when `T` is non-differentiable
# (e.g. `Ptr{UInt8}` from String/IO wrapping, `Ptr{Int}`), so the lifted pointer's V is `NoDual` and
# the wrapped array's element type is non-differentiable too — its canonical V is `NoDual`. The broad
# `@is_primitive` covers every `Ptr` and the reverse rule handles all `T`, so without this the forward
# rule's method coverage is narrower than its `@is_primitive` (a MethodError at call time). Mirrors the
# `NoDual` fallbacks on the sibling pointer rules (pointerref/pointerset/unsafe_copyto!).
function frule!!(
    ::Lifted{typeof(unsafe_wrap),Nw},
    ::Lifted{<:Type{<:Array},Nw},
    p::Lifted{<:Ptr,Nw,NoDual},
    dims::Lifted,
) where {Nw}
    arr = unsafe_wrap(Array, primal(p), primal(dims))
    return Lifted{typeof(arr),Nw}(arr, NoDual())
end

function rrule!!(
    ::CoDual{typeof(unsafe_wrap)},
    ::CoDual{<:Type{<:Array}},
    p::CoDual{<:Ptr{T}},
    dims::CoDual,
) where {T}
    primal_arr = unsafe_wrap(Array, primal(p), primal(dims))
    tangent_arr = unsafe_wrap(Array, tangent(p), primal(dims))
    function unsafe_wrap_pullback!!(::NoRData)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(primal_arr, tangent_arr), unsafe_wrap_pullback!!
end

# atomic_fence
# atomic_pointermodify
# atomic_pointerref
# atomic_pointerreplace

@intrinsic atomic_pointerset
# The V is exactly `NTuple{Nw,Ptr{T}}` (partial element `=== Ptr{T}`, since `tangent_type`
# is the identity on the leaf float/`Ptr` element types reaching here), so the per-lane
# `atomic_pointerset(partial::Ptr{T}, tangent::T, …)` typechecks for float scalars and a
# coherent `Ptr{Ptr{Float64}}` alike — and the element-wise `Ptr{S≠T}` shape is excluded. A
# non-differentiable element (incoherent per-lane V, e.g. `Ptr{UInt8}`) writes only the
# primal; `tangent_type(T)` folds at specialisation so the branch is compile-time.
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
            atomic_pointerset(p_partials[lane], tangent(x, lane), _order)
        end
    end
    return p
end
# Non-differentiable pointer (V === NoDual): store the primal; no tangent to write.
function frule!!(
    ::Lifted{typeof(atomic_pointerset),Nw},
    p::Lifted{<:Ptr,Nw,NoDual},
    x::Lifted,
    order::Lifted,
) where {Nw}
    atomic_pointerset(primal(p), primal(x), primal(order))
    return p
end
# Element-wise per-lane V (`NTuple{Nw,Ptr{S}}` with `S !== Ptr{T}`): see the matching
# `pointerset` guard — the array-of-pointers store is unsupported, so fail loudly for a
# differentiable element rather than raise a raw `MethodError`.
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
    old_value = atomic_pointerref(_p, _order)
    old_tangent = atomic_pointerref(tangent(p), _order)
    dp = tangent(p)
    function atomic_pointerset_pullback!!(::NoRData)
        dx_r = atomic_pointerref(dp, _order)
        atomic_pointerset(_p, old_value, _order)
        atomic_pointerset(dp, old_tangent, _order)
        return NoRData(), NoRData(), rdata(dx_r), NoRData()
    end

    atomic_pointerset(_p, primal(x), _order)
    # zero_tangent(primal(x), tangent(x)) is used to correctly handle
    # Ptr types, whose tangent is purely fdata (a Ptr) with NoRData.
    atomic_pointerset(dp, zero_tangent(primal(x), tangent(x)), _order)
    return p, atomic_pointerset_pullback!!
end

# atomic_pointerswap

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
    v = bitcast(T, primal(x))
    # Non-Ptr or NoDual-V bitcast: no forward derivative to carry.
    return Lifted{typeof(v),Nw}(v, NoDual())
end
# Ptr→Ptr bitcast of a per-lane `NTuple{Nw,Ptr}` V: re-type each lane's pointer.
# (`T` is already the full target `Ptr` type, e.g. `Ptr{ComplexF64}` — not wrapped
# again.) Constrained to `T<:Ptr`: a bitcast to a differentiable type still falls to
# the generic frule above, which throws (it must not be silently bypassed here).
# Two lane element types keep their type instead of re-typing to the target:
#  - an element-wise dual-element lane (`Ptr{NDualArray}`) addresses a tangent buffer whose element stride
#    differs from the re-typed primal pointer's, so a downstream `unsafe_copyto!` must copy it;
#  - an objref tangent-object address (`Ptr{tangent_type(Nothing)}`, from `pointer_from_objref`)
#    must NOT be re-typed into a fake-coherent `Ptr{<:IEEEFloat}` per-lane-partial pointer — kept
#    distinct, a subsequent `pointerref` correctly hits the loud incoherent-V throw rather than
#    reading bytes off the interleaved `MutableDual`.
# Other (raw/scalar) lanes (`Ptr{Nothing}`, `Ptr{<:IEEEFloat}`) re-type to the target.
@inline function frule!!(
    ::Lifted{typeof(bitcast),Nw}, ::Lifted{Type{T},Nw}, x::Lifted{P,Nw,<:NTuple{Nw,<:Ptr}}
) where {Nw,T<:Ptr,P<:Ptr}
    lanes = ntuple(Val(Nw)) do k
        p = tangent(x)[k]
        if eltype(typeof(p)) <: NDualArray || p isa Ptr{tangent_type(Nothing)}
            p
        else
            bitcast(T, p)
        end
    end
    return Lifted{T,Nw}(bitcast(T, primal(x)), lanes)
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
        dv = bitcast(Ptr{tangent_type(eltype(T))}, tangent(x))
    elseif T <: Ptr && _x isa Union{Int,UInt}
        int2ptr_err_msg =
            "It is not permissible to bitcast from an Int/UInt type to a Ptr type during AD, as " *
            "this risks giving the wrong answer, or causing Julia to segfault. " *
            "If this call to bitcast appears as part of the implementation of a " *
            "differentiable function, you should write a rule for this function, or modify " *
            "its implementation to avoid the bitcast."
        throw(ArgumentError(int2ptr_err_msg))
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
    # d copysign(x,y)/dx = sign(x) * sign(y). Scale only the partials and set V.value to `z`.
    # A naive `sign(y) * tangent(x)` both drops the sign(x) factor (wrong derivative for x<0)
    # and scales the inner NDual's `.value` to `sign(y) * x_p` (≠ z), breaking V.value === primal.
    s = sign(primal(x)) * sign(primal(y))
    return Lifted{T,N}(z, NDual{T,N}(z, s .* tangent(x).partials))
end
function rrule!!(::CoDual{typeof(copysign_float)}, x, y)
    _x = primal(x)
    _y = primal(y)
    # d copysign(x,y)/dx = sign(x) * sign(y); the derivative w.r.t. y is zero.
    copysign_float_pullback!!(dz) = NoRData(), dz * sign(_x) * sign(_y), zero_rdata(_y)
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
    div_float_pullback!!(dy) = NoRData(), div_float(dy, _b), -dy * _a / _b^2
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
        return NoRData(), div_float_fast(dy, _b), -dy * div_float_fast(_a, _b^2)
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
    return Lifted{T,N}(
        fma_float(primal(x), primal(y), primal(z)), tangent(x) * tangent(y) + tangent(z)
    )
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
        return Lifted{T,N}(p, ifelse(primal(a) > primal(b), tangent(a), tangent(b)))
    end
    function rrule!!(
        ::CoDual{typeof(max_float)}, a::CoDual{P}, b::CoDual{P}
    ) where {P<:Base.IEEEFloat}
        _a = primal(a)
        _b = primal(b)
        tmp = _a > _b
        x = max_float(_a, _b)
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
        return Lifted{T,N}(p, ifelse(primal(a) > primal(b), tangent(a), tangent(b)))
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
        return Lifted{T,N}(p, ifelse(primal(a) < primal(b), tangent(a), tangent(b)))
    end
    function rrule!!(
        ::CoDual{typeof(min_float)}, a::CoDual{P}, b::CoDual{P}
    ) where {P<:Base.IEEEFloat}
        _a = primal(a)
        _b = primal(b)
        tmp = _a < _b
        x = min_float(_a, _b)
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
        return Lifted{T,N}(p, ifelse(primal(a) < primal(b), tangent(a), tangent(b)))
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
    return Lifted{T,N}(
        muladd_float(primal(x), primal(y), primal(z)), tangent(x) * tangent(y) + tangent(z)
    )
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
# Load scalar via primal Ptr; load each lane's tangent scalar via that lane's partial Ptr; pack into
# the canonical inner V — `NDual` for a real element, `Complex{NDual}` for a complex one (the inner
# representation of a complex scalar is `Complex{NDual}`, never `NDual{Complex}`), via `_scalar_ndual`.
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
    da_lanes = ntuple(lane -> pointerref(x_partials[lane], _y, _z), Val(Nw))
    return Lifted{T,Nw}(a, _scalar_ndual(a, da_lanes))
end
# Non-differentiable pointer (V === NoDual): the element type is not an `NDualEltype`
# (e.g. `Ptr{UInt64}`, `Ptr{Ptr{Float64}}`), so the loaded value carries no derivative.
function frule!!(
    ::Lifted{typeof(pointerref),Nw}, x::Lifted{<:Ptr,Nw,NoDual}, y::Lifted, z::Lifted
) where {Nw}
    a = pointerref(primal(x), primal(y), primal(z))
    return Lifted{typeof(a),Nw}(a, NoDual())
end
# A non-differentiable `Ptr` can carry an incoherent per-lane `NTuple{Nw,Ptr}` V (its canonical
# V is `NoDual`) when it is produced by an upstream `unsafe_convert`/`bitcast` chain — e.g.
# `_getindex_ra` reading a byte out of a reinterpreted integer array does `Ptr{UInt8}(unsafe_convert(
# Ref{UInt32}, …))`. The load of a non-differentiable element carries no derivative, so collapse
# to `NoDual`. The `T <: NDualEltype` frule above serves the scalar differentiable case (it packs
# the canonical inner V from per-lane partial pointers). Two differentiable cases reach here with an
# incoherent V and fail loudly: a non-scalar element (e.g. `Ptr{Ptr{Float64}}`, whose load is a
# `Ptr{Float64}` with a per-lane-pointer V, not a scalar dual); and a raw scalar load through
# `pointer_from_objref` of a general mutable struct, whose objref tangent-address lanes
# (`Ptr{tangent_type(Nothing)}`) are not the canonical per-lane-partial shape (see the
# `pointer_from_objref` rule). `Ref` itself is handled correctly — its forward tangent (`NDualRef`)
# keeps a parallel partials buffer with primal-identical layout, so its branch above packs a dual.
#
# To LIFT the general mutable-struct case (make a forward raw scalar load correct, matching reverse):
# the struct's forward tangent must keep its per-lane partials in a parallel buffer with
# primal-identical layout (as `NDualRef` does for `Ref` and `NDualArray` for `Array`), so a
# same-offset pointer lands the partials. Today a mutable struct's tangent is a `MutableDual` that
# interleaves the value and partials in one object, with no parallel partials buffer to point at.
# The principled lift is a per-struct primal-shaped partials shadow (correct by construction); best
# done opt-in, since making it the default mutable-struct tangent regresses chunked struct-field math.
# A pointer shortcut that reads through the interleaving is rejected: it hardcodes the `NDual` layout.
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
# The V is exactly `NTuple{Nw,Ptr{T}}` (partial element `=== Ptr{T}`, since `tangent_type`
# is the identity on the leaf float/`Ptr` element types reaching here), so the per-lane
# `pointerset(partial::Ptr{T}, tangent::T, …)` typechecks for float scalars and a coherent
# `Ptr{Ptr{Float64}}` alike — and the element-wise `Ptr{S≠T}` shape is excluded. A non-differentiable
# element (incoherent per-lane V, e.g. `Ptr{UInt8}`) writes only the primal; `tangent_type(T)`
# folds at specialisation so the branch is compile-time.
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
            pointerset(p_partials[lane], tangent(x, lane), _idx, _z)
        end
    end
    return p
end
# Non-differentiable pointer (V === NoDual): store the primal; no tangent to write.
function frule!!(
    ::Lifted{typeof(pointerset),Nw},
    p::Lifted{<:Ptr,Nw,NoDual},
    x::Lifted,
    idx::Lifted,
    z::Lifted,
) where {Nw}
    pointerset(primal(p), primal(x), primal(idx), primal(z))
    return p
end
# An element-wise per-lane V (`NTuple{Nw,Ptr{S}}` with partial element `S !== Ptr{T}`) reaches
# here when the destination is an array of differentiable pointers (e.g.
# `pointer(::Vector{Ptr{Float64}})`, whose tangent buffer holds `Tuple{Ptr{Float64}}`
# elements, not bare `Ptr{Float64}`). Writing a bare lane tangent through that pointer
# would corrupt the element-wise stride, so fail loudly — the array-of-pointers store is
# unsupported. The parallel-arrays `NTuple{Nw,Ptr{T}}` frule above is strictly more specific.
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
function frule!!(
    ::Lifted{typeof(sqrt_llvm),Nw}, x::Lifted{T,Nw,NDual{T,Nw}}
) where {Nw,T<:IEEEFloat}
    # NDual.sqrt overload (Nfwd.jl) applies _pt_guarded_scale — the NDual
    # analogue of nan_tangent_guard — so the singular `sqrt(0)` case has
    # zeroed partials instead of NaN.
    return Lifted{T,Nw}(sqrt_llvm(primal(x)), sqrt(tangent(x)))
end
function rrule!!(::CoDual{typeof(sqrt_llvm)}, x::CoDual{P}) where {P}
    _y = sqrt_llvm(primal(x))
    function llvm_sqrt_pullback!!(dy)
        dx = nan_tangent_guard(dy, dy / (2 * _y))
        return NoRData(), dx
    end
    return CoDual(_y, NoFData()), llvm_sqrt_pullback!!
end

@intrinsic sqrt_llvm_fast
function frule!!(
    ::Lifted{typeof(sqrt_llvm_fast),Nw}, x::Lifted{T,Nw,NDual{T,Nw}}
) where {Nw,T<:IEEEFloat}
    return Lifted{T,Nw}(sqrt_llvm_fast(primal(x)), sqrt(tangent(x)))
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
# The tangent V is either the parallel-arrays `NDualArray` (for `Vector{<:IEEEFloat}`) or a
# plain `Vector` of per-element Vs (element-wise); both are `AbstractVector`, so `Tuple`
# iterates either into the per-element tangent tuple. (`__vec_to_tuple` itself is
# the `::Vector`-only primal helper and does not accept an NDualArray.) The
# `<:AbstractVector` V bound also excludes non-differentiable (`NoDual`) vectors.
@inline function frule!!(
    ::Lifted{typeof(__vec_to_tuple),Nw}, v::Lifted{<:Vector,Nw,<:AbstractVector}
) where {Nw}
    x = __vec_to_tuple(primal(v))
    # An all-non-differentiable splat (e.g. a permutation `Vector{Int}`, whose V is
    # `Vector{NoDual}`) yields a tuple with `dual_type === NoDual`; build whole `NoDual`
    # to match, not the element-wise `Tuple{NoDual,…}` the consumer slot would reject.
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

# `Core.SimpleVector`'s forward V is an array-of-structures `Vector{Any}` holding each
# element's forward V (`NoDual` for the usual non-differentiable elements — DataType, Symbol,
# … — or a real inner V like `NDual`/`NDualArray` for a differentiable element), mirroring the
# reverse `tangent_type(SimpleVector) === Vector{Any}`. Keeping the V coherent with what the
# `svec` / `_svec_ref` frules build avoids the OpaqueClosure return typeassert-reject in
# forward-over-reverse, where `svec` sparams (all non-differentiable) flow through rule
# construction as an all-`NoDual` `Vector{Any}`.
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
function tangent(x::Lifted{Core.SimpleVector,N,Vector{Any}}, lane::Integer) where {N}
    p = primal(x)
    v = tangent(x)
    return Any[tangent(Lifted{typeof(p[i]),N}(p[i], v[i]), lane) for i in 1:length(p)]
end
function lift(v::Core.SimpleVector, ẋ::Vector{Any})
    return Lifted{Core.SimpleVector,1}(
        v, Any[tangent(lift(v[i], ẋ[i])) for i in 1:length(v)]
    )
end
function _unlift_seed(x::Lifted{Core.SimpleVector,1,Vector{Any}}, cache::IdDict)
    p = primal(x)
    v = tangent(x)
    return Any[_unlift_seed(Lifted{typeof(p[i]),1}(p[i], v[i]), cache) for i in 1:length(p)]
end
@inline unlift(x::Lifted{Core.SimpleVector,1,Vector{Any}}) = (
    primal(x), _unlift_seed(x, IdDict{Any,Any}())
)

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
    ::Lifted{typeof(compilerbarrier),Nw}, setting::Lifted{Symbol,Nw}, v::Lifted{P,Nw,V}
) where {Nw,P,V}
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

# `Core.ifelse` is a non-short-circuiting scalar select; both branches arrive as
# already-evaluated slots, so the JVP is just the selected slot. This covers any
# branch types (matching the reverse rrule's `a::A, b::B` breadth) and stays
# type-stable when the branches share a type.
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

    # It's a good idea to split up applying ifelse to the primal and tangent. This is
    # because if you push a `CoDual` through ifelse, it _forces_ the construction of the
    # CoDual. Conversely, if you pass through the primal and tangents separately, the
    # compiler will often be able to avoid constructing the CoDual at all by inlining lots
    # of stuff away.
    return CoDual(ifelse(_cond, p_a, p_b), ifelse(_cond, tangent(a), tangent(b))), pb!!
end

@zero_derivative MinimalCtx Tuple{typeof(Core.sizeof),Any}

# Core.svec

@zero_derivative MinimalCtx Tuple{typeof(applicable),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(fieldtype),Vararg}

const StandardTangentType = Union{Tuple,NamedTuple,Tangent,MutableTangent,NoTangent}
const StandardFDataType = Union{Tuple,NamedTuple,FData,MutableTangent,NoFData}

# 2-arg `getfield(x, name)`: delegate to `lgetfield`, whose generic Lifted body
# (`_get_lifted_field` in misc.jl) covers tuples, named-tuples, and structs. Kept
# here rather than memory.jl so it is available on Julia 1.10 (array_legacy path),
# where the forward-over-reverse HVP public interface needs it.
function frule!!(::Lifted{typeof(getfield),Nw}, x::Lifted, name::Lifted) where {Nw}
    # Extract the field directly rather than routing through `lgetfield(x, Val(primal(name)))`:
    # `Val(runtime_name)` is type-unstable (the parameter is a runtime value), so the routed
    # form constructed an abstract-`P` `Lifted{Val{...}}` and ran the `lgetfield` frule via
    # runtime dispatch. Mirrors the 3-arg `getfield` frule below.
    _name = primal(name)
    y = getfield(primal(x), _name)
    P = _typeof(primal(x))
    if tangent_type(P) == NoTangent
        # A non-differentiable parent yields a non-differentiable field, but its forward V is
        # the field's *canonical* zero V — `NoDual` for the usual scalars, yet `Vector{Any}`
        # for a `SimpleVector` field (e.g. `getfield(::DataType, :parameters)`). Blanket
        # `NoDual()` here produced a non-canonical `Lifted{SimpleVector,…,NoDual}` that the svec
        # consumers reject. `uninit_lifted` builds the canonical slot (mirrors the reverse
        # `uninit_fcodual` used by the corresponding `rrule!!`).
        return uninit_lifted(Val(Nw), y)
    else
        return Lifted{_typeof(y),Nw}(y, _get_lifted_field(tangent(x), _name))
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
        # whose V is `Vector{Any}`, not `NoDual`).
        return uninit_lifted(Val(Nw), y)
    else
        return Lifted{_typeof(y),Nw}(y, _get_lifted_field(tangent(x), _name))
    end
end
# `Ref{P<:NDualEltype}` (`NDualRef` V): the generic `_get_lifted_field` path above has no
# `NDualRef` method, so rebuild the scalar inner V from the parallel partials buffer, mirroring the
# literal-name `lgetfield` Ref branch in misc.jl (the read counterpart of the `setfield!` Ref frule).
# Runtime name is `:x` (or its index `1`), the Ref's only field. Covers real and complex elements.
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

# TODO: remove once no remaining callers depend on the older homogeneous-immutable
# getfield fast-path selection.
@generated is_homogeneous_and_immutable(::P) where {P<:Tuple} = allequal(fieldtypes(P))

@inline is_homogeneous_and_immutable(p::NamedTuple) = is_homogeneous_and_immutable(Tuple(p))
is_homogeneous_and_immutable(::Any) = false

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
# Array `.ref`/`.size` mutation (e.g. resize) via the runtime-name `setfield!` can't use
# the `_setfield_tangent!` path — the parallel-arrays `NDualArray` V is immutable. Delegate to the
# `lsetfield!` array frule (which updates the V via the partials/memref-aliasing), with the
# positional field index normalised to the symbol it dispatches on (`1`→`:ref`, `2`→`:size`).
@inline function frule!!(
    ::Lifted{typeof(setfield!),Nw}, value::Lifted{<:Array}, name::Lifted, x::Lifted
) where {Nw}
    nm = primal(name)
    sym = nm isa Integer ? fieldname(typeof(primal(value)), nm) : nm
    return frule!!(
        zero_lifted(Val(Nw), lsetfield!), value, zero_lifted(Val(Nw), Val(sym)), x
    )
end
# A `RefValue`'s V is `NDualRef` (an immutable wrapper over per-lane partials, no `:x` field), so
# the `_setfield_tangent!` path's generic `setproperty!` fallback would throw. Delegate to the Ref
# `lsetfield!` frule (which writes the partials), like the Array branch — `RefValue`'s only field
# is `:x` (index 1).
@inline function frule!!(
    ::Lifted{typeof(setfield!),Nw}, value::Lifted{<:Base.RefValue}, name::Lifted, x::Lifted
) where {Nw}
    nm = primal(name)
    sym = nm isa Integer ? fieldname(typeof(primal(value)), nm) : nm
    return frule!!(
        zero_lifted(Val(Nw), lsetfield!), value, zero_lifted(Val(Nw), Val(sym)), x
    )
end
# A `MutableDual` struct V stores fields in its backing `value` NamedTuple (the
# same path `lsetfield!` takes), so merge there — `setproperty!` on the
# `MutableDual` itself would hit its single `value` field. A non-diff V (`NoDual`)
# has nothing to write; any other V (e.g. a `MutableDualTangentView`) routes
# through `setproperty!`, which delegates to the parent.
@inline _setfield_tangent!(::Union{NoDual,NoTangent}, _, _) = nothing
@inline function _setfield_tangent!(tv::MutableDual, nm, vx)
    nt = getfield(tv, :value)
    v_i = _coerce_backing_field(fieldtype(typeof(nt), nm), vx)
    # `convert` to the stored NamedTuple type: `setfield!` is strict (no implicit convert) and
    # `NamedTuple` is invariant in its `Tuple` parameter, so for a struct with an abstract field
    # (e.g. `Foo.x::Real` -> dual field `@NamedTuple{x}`, x::Any) the `merge` narrows to
    # `@NamedTuple{x::NDual}`, which is NOT `isa @NamedTuple{x}` — a bare `setfield!` would throw.
    # `convert` rebuilds it at the field's (possibly abstract) element type.
    setfield!(tv, :value, convert(typeof(nt), merge(nt, NamedTuple{(nm,)}((v_i,)))))
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
    # Derive the slot `P` from the value's own type, not `_typeof`: `_typeof`
    # sharpens a `Type`-valued element to `Type{X}`, but a tuple *value* always
    # types that slot as `DataType` — so the sharpened tuple type is unsatisfiable
    # by any value. Mirrors reverse-mode `tuple`'s `zero_fcodual(primal_output)`.
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
    ::Lifted{typeof(typeassert),Nw}, x::Lifted{P,Nw,V}, type::Lifted
) where {Nw,P,V}
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
        # atomic_pointerref -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerreplace -- NEEDS IMPLEMENTING AND TESTING
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.atomic_pointerset,
            CoDual(p, dp),
            1.0,
            :monotonic,
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
        (true, :stability_and_allocs, nothing, IntrinsicsWrappers.fpext, Float64, 5.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.fpiseq, 4.1, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.fpiseq, 4.0f1, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.fptosi, UInt32, 4.1),
        (false, :stability, nothing, IntrinsicsWrappers.fptoui, Int32, 4.1),
        (true, :stability, nothing, IntrinsicsWrappers.fptrunc, Float32, 5.0),
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

    test_cases = Any[
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
        (
            false,
            :none,
            nothing,
            (function (x)
                rx = Ref(x)
                return pointerref(bitcast(Ptr{Float64}, pointer_from_objref(rx)), 1, 1)
            end),
            5.0,
        ),
        (
            false,
            :none,
            nothing,
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
        # Reverse only: a `pointerset`/`atomic_pointerset` into a `Vector{Ptr{Float64}}` stores a
        # differentiable pointer into an array of differentiable pointers — its forward per-lane
        # tangent is an array-of-structs of pointers, which the forward `pointerset` rule rejects
        # loudly (same limitation class as `f_pointerset`). Reverse mode is correct.
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
        # Reverse only: a differentiable pointer-to-pointer raw store (`pointerset`/
        # `atomic_pointerset` into a `Ptr{Ptr{Float64}}`) cannot be done in forward mode — the
        # destination's per-lane tangent is an array-of-structs of pointers, so the forward rule
        # fails loudly (it silently produced a wrong derivative before the loud guard). Reverse mode
        # is correct (see the explicit value_and_gradient!! testset in test/rules/builtins.jl).
        (true, :none, (skip_forward=true,), f_pointerset, CoDual(3.0, 1.0)),
        (true, :none, (skip_forward=true,), f_atomic_pointerset, CoDual(3.0, 1.0)),
        (false, :none, nothing, getindex, randn(5), [1, 1]),
        (false, :none, nothing, getindex, randn(5), [1, 2, 2]),
        (false, :none, nothing, setindex!, randn(5), [4.0, 5.0], [1, 1]),
        (false, :none, nothing, setindex!, randn(5), [4.0, 5.0, 6.0], [1, 2, 2]),
    ]
    return test_cases, Any[]
end

function throwing_rule_test_cases(::Val{:builtins})
    # atomic_pointerset through a differentiable element with an element-wise (incoherent)
    # per-lane V must hit the loud guard, mirroring pointerset.
    xv = [1.0]
    ptr = pointer(xv)
    pslot = Lifted{Ptr{Float64},1}(ptr, (Ptr{Tuple{Float64}}(UInt(ptr)),))
    cases = Any[(
        ArgumentError,
        IntrinsicsWrappers.atomic_pointerset,
        (pslot, zero_lifted(Val(1), 2.0), zero_lifted(Val(1), :monotonic)),
    )]
    return cases, Any[xv]
end
