
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
    print(io, "MissingRuleForBuiltinException: ")
    return println(io, err.msg)
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
    Dual,
    primal,
    tangent,
    zero_tangent,
    NoPullback,
    tangent_type,
    increment!!,
    @is_primitive,
    MinimalCtx,
    is_primitive,
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
    Mode

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
        function is_primitive(
            ::Type{MinimalCtx}, ::Type{<:Mode}, ::Type{<:Tuple{typeof($name),Vararg}}
        )
            true
        end
        translate(::Val{Intrinsics.$name}) = $name
    end
    return esc(expr)
end

macro inactive_intrinsic(name)
    expr = quote
        $name(x...) = Intrinsics.$name(x...)
        function is_primitive(
            ::Type{MinimalCtx}, ::Type{<:Mode}, ::Type{<:Tuple{typeof($name),Vararg}}
        )
            true
        end
        translate(::Val{Intrinsics.$name}) = $name
        function rrule!!(f::CoDual{typeof($name)}, args::Vararg{Any,N}) where {N}
            return Mooncake.zero_adjoint(f, args...)
        end
        function frule!!(f::Dual{typeof($name)}, args::Vararg{Dual,N}) where {N}
            f_primal = primal(f)
            args_primal = map(primal, args)
            return zero_dual(f_primal(args_primal...))
        end
    end
    return esc(expr)
end

@intrinsic abs_float
function frule!!(::Dual{typeof(abs_float)}, x)
    return Dual(abs_float(primal(x)), sign(primal(x)) * tangent(x))
end
function rrule!!(::CoDual{typeof(abs_float)}, x)
    abs_float_pullback!!(dy) = NoRData(), sign(primal(x)) * dy
    y = abs_float(primal(x))
    return CoDual(y, NoFData()), abs_float_pullback!!
end

@intrinsic add_float
function frule!!(::Dual{typeof(add_float)}, a, b)
    return Dual(add_float(primal(a), primal(b)), add_float(tangent(a), tangent(b)))
end
function rrule!!(::CoDual{typeof(add_float)}, a, b)
    add_float_pb!!(c̄) = NoRData(), c̄, c̄
    c = add_float(primal(a), primal(b))
    return CoDual(c, NoFData()), add_float_pb!!
end

@intrinsic add_float_fast
function frule!!(::Dual{typeof(add_float_fast)}, a, b)
    c = add_float_fast(primal(a), primal(b))
    dc = add_float_fast(tangent(a), tangent(b))
    return Dual(c, dc)
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

# atomic_fence
# atomic_pointermodify
# atomic_pointerref
# atomic_pointerreplace

@intrinsic atomic_pointerset
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
    atomic_pointerset(dp, zero_tangent(primal(x)), _order)
    return p, atomic_pointerset_pullback!!
end

# atomic_pointerswap

@intrinsic bitcast
function frule!!(f::Dual{typeof(bitcast)}, t::Dual{Type{T}}, x) where {T}
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
    else
        dv = NoTangent()
    end
    return Dual(v, dv)
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
            "It is not permissible to bitcast to an Int/UInt type to a Ptr type during AD, as " *
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
function Mooncake.is_primitive(
    ::Type{MinimalCtx}, ::Type{<:Mode}, ::Type{<:Tuple{typeof(__cglobal),Vararg}}
)
    return true
end
function frule!!(::Dual{typeof(__cglobal)}, args...)
    return Mooncake.uninit_dual(__cglobal(map(primal, args)...))
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
function frule!!(::Dual{typeof(copysign_float)}, x, y)
    z = copysign_float(primal(x), primal(y))
    dz = sign(primal(y)) * tangent(x)
    return Dual(z, dz)
end
function rrule!!(::CoDual{typeof(copysign_float)}, x, y)
    _x = primal(x)
    _y = primal(y)
    copysign_float_pullback!!(dz) = NoRData(), dz * sign(_y), zero_rdata(_y)
    z = copysign_float(_x, _y)
    return CoDual(z, NoFData()), copysign_float_pullback!!
end

@inactive_intrinsic ctlz_int
@inactive_intrinsic ctpop_int
@inactive_intrinsic cttz_int

@intrinsic div_float
function frule!!(::Dual{typeof(div_float)}, a, b)
    c = div_float(primal(a), primal(b))
    da = tangent(a)
    db = tangent(b)
    dc = div_float(da, primal(b)) - div_float(primal(a) * db, primal(b)^2)
    return Dual(c, dc)
end
function rrule!!(::CoDual{typeof(div_float)}, a, b)
    _a = primal(a)
    _b = primal(b)
    _y = div_float(_a, _b)
    div_float_pullback!!(dy) = NoRData(), div_float(dy, _b), -dy * _a / _b^2
    return CoDual(_y, NoFData()), div_float_pullback!!
end

@intrinsic div_float_fast
function frule!!(::Dual{typeof(div_float_fast)}, a, b)
    c = div_float_fast(primal(a), primal(b))
    da = tangent(a)
    db = tangent(b)
    dc = div_float_fast(da, primal(b)) - div_float_fast(primal(a) * db, primal(b)^2)
    return Dual(c, dc)
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
function frule!!(::Dual{typeof(fma_float)}, x, y, z)
    a = fma_float(primal(x), primal(y), primal(z))
    da = fma_float(tangent(x), primal(y), fma_float(primal(x), tangent(y), tangent(z)))
    return Dual(a, da)
end
function rrule!!(::CoDual{typeof(fma_float)}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    fma_float_pullback!!(da) = NoRData(), da * _y, da * _x, da
    return CoDual(fma_float(_x, _y, primal(z)), NoFData()), fma_float_pullback!!
end

@intrinsic fpext
function frule!!(
    ::Dual{typeof(fpext)}, ::Dual{Type{Pext}}, x::Dual{P}
) where {Pext<:IEEEFloat,P<:IEEEFloat}
    return Dual(fpext(Pext, primal(x)), fpext(Pext, tangent(x)))
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
    ::Dual{typeof(fptrunc)}, ::Dual{Type{Ptrunc}}, x::Dual{P}
) where {Ptrunc<:IEEEFloat,P<:IEEEFloat}
    return Dual(fptrunc(Ptrunc, primal(x)), fptrunc(Ptrunc, tangent(x)))
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

@intrinsic mul_float
function frule!!(::Dual{typeof(mul_float)}, a, b)
    p = mul_float(primal(a), primal(b))
    dp = add_float(mul_float(primal(a), tangent(b)), mul_float(primal(b), tangent(a)))
    return Dual(p, dp)
end
function rrule!!(::CoDual{typeof(mul_float)}, a, b)
    _a = primal(a)
    _b = primal(b)
    mul_float_pb!!(dc) = NoRData(), dc * _b, _a * dc
    return CoDual(mul_float(_a, _b), NoFData()), mul_float_pb!!
end

@intrinsic mul_float_fast
function frule!!(::Dual{typeof(mul_float_fast)}, a, b)
    c = mul_float_fast(primal(a), primal(b))
    dc = mul_float_fast(primal(a), tangent(b)) + mul_float_fast(tangent(a), primal(b))
    return Dual(c, dc)
end
function rrule!!(::CoDual{typeof(mul_float_fast)}, a, b)
    _a = primal(a)
    _b = primal(b)
    mul_float_fast_pb!!(dc) = NoRData(), dc * _b, _a * dc
    return CoDual(mul_float_fast(_a, _b), NoFData()), mul_float_fast_pb!!
end

@inactive_intrinsic mul_int

@intrinsic muladd_float
function frule!!(::Dual{typeof(muladd_float)}, x, y, z)
    a = muladd_float(primal(x), primal(y), primal(z))
    dz = tangent(z)
    da = muladd_float(tangent(x), primal(y), muladd_float(primal(x), tangent(y), dz))
    return Dual(a, da)
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
frule!!(::Dual{typeof(neg_float)}, x) = Dual(neg_float(primal(x)), neg_float(tangent(x)))
function rrule!!(::CoDual{typeof(neg_float)}, x)
    _x = primal(x)
    neg_float_pullback!!(dy) = NoRData(), -dy
    return CoDual(neg_float(_x), NoFData()), neg_float_pullback!!
end

@intrinsic neg_float_fast
function frule!!(::Dual{typeof(neg_float_fast)}, x)
    return Dual(neg_float_fast(primal(x)), neg_float_fast(tangent(x)))
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
function frule!!(::Dual{typeof(pointerref)}, x, y, z)
    a = pointerref(primal(x), primal(y), primal(z))
    da = pointerref(tangent(x), primal(y), primal(z))
    return Dual(a, da)
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
function frule!!(::Dual{typeof(pointerset)}, p, x, idx, z)
    pointerset(primal(p), primal(x), primal(idx), primal(z))
    pointerset(tangent(p), tangent(x), primal(idx), primal(z))
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
    pointerset(dp, zero_tangent(primal(x)), _idx, _z)
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
function frule!!(::Dual{typeof(sqrt_llvm)}, x)
    y = sqrt_llvm(primal(x))
    dy = tangent(x) / (2 * y)
    return Dual(y, dy)
end
function rrule!!(::CoDual{typeof(sqrt_llvm)}, x)
    _x = primal(x)
    _y = sqrt_llvm(primal(x))
    llvm_sqrt_pullback!!(dy) = NoRData(), dy / (2 * _y)
    return CoDual(_y, NoFData()), llvm_sqrt_pullback!!
end

@intrinsic sqrt_llvm_fast
function frule!!(::Dual{typeof(sqrt_llvm_fast)}, x)
    y = sqrt_llvm_fast(primal(x))
    dy = tangent(x) / (2 * y)
    return Dual(y, dy)
end
function rrule!!(::CoDual{typeof(sqrt_llvm_fast)}, x)
    _y = sqrt_llvm_fast(primal(x))
    llvm_sqrt_fast_pullback!!(dy) = NoRData(), dy / (2 * _y)
    return CoDual(_y, NoFData()), llvm_sqrt_fast_pullback!!
end

@inactive_intrinsic srem_int

@intrinsic sub_float
function frule!!(::Dual{typeof(sub_float)}, a, b)
    c = sub_float(primal(a), primal(b))
    dc = sub_float(tangent(a), tangent(b))
    return Dual(c, dc)
end
function rrule!!(::CoDual{typeof(sub_float)}, a, b)
    _a = primal(a)
    _b = primal(b)
    sub_float_pullback!!(dc) = NoRData(), dc, -dc
    return CoDual(sub_float(_a, _b), NoFData()), sub_float_pullback!!
end

@intrinsic sub_float_fast
function frule!!(::Dual{typeof(sub_float_fast)}, a, b)
    c = sub_float_fast(primal(a), primal(b))
    dc = sub_float_fast(tangent(a), tangent(b))
    return Dual(c, dc)
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
function frule!!(::Dual{typeof(__vec_to_tuple)}, v::Dual{<:Vector})
    x = __vec_to_tuple(primal(v))
    if tangent_type(_typeof(x)) == NoTangent
        return zero_dual(x)
    else
        return Dual(x, __vec_to_tuple(tangent(v)))
    end
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

# Verify that there thing at the index is non-differentiable. Otherwise error.
function frule!!(
    ::Dual{typeof(Core._svec_ref)}, v::Dual{Core.SimpleVector}, _ind::Dual{Int}
)
    ind = primal(_ind)
    pv = Core._svec_ref(primal(v), ind)
    tv = getindex(tangent(v), ind)
    isa(tv, NoTangent) || error("expected non-differentiable thing in SimpleVector")
    return Dual(pv, tv)
end
function rrule!!(
    f::CoDual{typeof(Core._svec_ref)}, v::CoDual{Core.SimpleVector}, _ind::CoDual{Int}
)
    ind = primal(_ind)
    pv = Core._svec_ref(primal(v), ind)
    tv = getindex(tangent(v), ind)
    isa(tv, NoTangent) || error("expected non-differentiable thing in SimpleVector")
    return zero_fcodual(pv), NoPullback(f, v, _ind)
end

# Core._typebody!
function frule!!(::Dual{typeof(Core._typevar)}, args...)
    return zero_dual(Core._typevar(map(primal, args)...))
end
function rrule!!(f::CoDual{typeof(Core._typevar)}, args...)
    return zero_fcodual(Core._typevar(map(primal, args)...)), NoPullback(f, args...)
end

function frule!!(::Dual{typeof(Core.apply_type)}, args...)
    return zero_dual(Core.apply_type(map(primal, args)...))
end
function rrule!!(f::CoDual{typeof(Core.apply_type)}, args...)
    T = Core.apply_type(tuple_map(primal, args)...)
    return CoDual{_typeof(T),NoFData}(T, NoFData()), NoPullback(f, args...)
end

function frule!!(::Dual{typeof(compilerbarrier)}, setting::Dual{Symbol}, v::Dual)
    return Dual(
        compilerbarrier(primal(setting), primal(v)),
        compilerbarrier(primal(setting), tangent(v)),
    )
end
function rrule!!(::CoDual{typeof(compilerbarrier)}, setting::CoDual{Symbol}, val::CoDual)
    compilerbarrier_pb(dout) = NoRData(), NoRData(), dout
    return compilerbarrier(setting.x, val), compilerbarrier_pb
end

# Core.donotdelete
# Core.finalizer
# Core.get_binding_type

function frule!!(::Dual{typeof(Core.ifelse)}, cond::Dual{Bool}, a::Dual, b::Dual)
    _cond = primal(cond)
    return Dual(ifelse(_cond, primal(a), primal(b)), ifelse(_cond, tangent(a), tangent(b)))
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

function frule!!(
    ::Dual{typeof(getfield)}, x::Dual{P,<:StandardTangentType}, name::Dual
) where {P}
    _name = primal(name)
    if tangent_type(P) == NoTangent
        return uninit_dual(getfield(primal(x), _name))
    else
        return Dual(getfield(primal(x), _name), _get_tangent_field(tangent(x), _name))
    end
end
function frule!!(
    ::Dual{typeof(getfield)}, x::Dual{P,<:StandardTangentType}, name::Dual, inbounds::Dual
) where {P}
    _name = primal(name)
    _inbounds = primal(inbounds)
    if tangent_type(P) == NoTangent
        return uninit_dual(getfield(primal(x), _name, _inbounds))
    else
        y = getfield(primal(x), _name, _inbounds)
        dy = _get_tangent_field(tangent(x), _name, _inbounds)
        return Dual(y, dy)
    end
end
function rrule!!(
    f::CoDual{typeof(getfield)}, x::CoDual{P,<:StandardFDataType}, name::CoDual
) where {P}
    if tangent_type(P) == NoTangent
        y = uninit_fcodual(getfield(primal(x), primal(name)))
        return y, NoPullback(f, x, name)
    elseif is_homogeneous_and_immutable(primal(x))
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
    elseif is_homogeneous_and_immutable(primal(x))
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

function frule!!(::Dual{typeof(setfield!)}, value::Dual, name::Dual, x::Dual)
    literal_name = zero_dual(Val(primal(name)))
    return frule!!(zero_dual(lsetfield!), value, literal_name, x)
end
function rrule!!(::CoDual{typeof(setfield!)}, value::CoDual, name::CoDual, x::CoDual)
    literal_name = uninit_fcodual(Val(primal(name)))
    return rrule!!(uninit_fcodual(lsetfield!), value, literal_name, x)
end

# swapfield!

frule!!(::Dual{typeof(throw)}, args::Dual...) = throw(map(primal, args)...)
rrule!!(::CoDual{typeof(throw)}, args::CoDual...) = throw(map(primal, args)...)

struct TuplePullback{N} end

@inline (::TuplePullback{N})(dy::Tuple) where {N} = NoRData(), dy...

@inline function (::TuplePullback{N})(::NoRData) where {N}
    return NoRData(), ntuple(_ -> NoRData(), N)...
end

@inline tuple_pullback(dy) = NoRData(), dy...

@inline tuple_pullback(dy::NoRData) = NoRData()

function frule!!(f::Dual{typeof(tuple)}, args::Vararg{Any,N}) where {N}
    primal_output = tuple(map(primal, args)...)
    if tangent_type(_typeof(primal_output)) == NoTangent
        return zero_dual(primal_output)
    else
        return Dual(primal_output, tuple(map(tangent, args)...))
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

function frule!!(::Dual{typeof(typeassert)}, x::Dual, type::Dual)
    return Dual(typeassert(primal(x), primal(type)), tangent(x))
end
function rrule!!(::CoDual{typeof(typeassert)}, x::CoDual, type::CoDual)
    typeassert_pullback(dy) = NoRData(), dy, NoRData()
    return CoDual(typeassert(primal(x), primal(type)), tangent(x)), typeassert_pullback
end

@zero_derivative MinimalCtx Tuple{typeof(typeof),Any}

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:builtins})
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

    # Slightly wider range for builtins whose performance is known not to be great.
    _range = (lb=1e-3, ub=200.0)
    memory = Any[_x, _dx, _a, x, p, dx, dp, y, q, dy, dq]

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
        # (
        #     true,
        #     :none,
        #     nothing,
        #     IntrinsicsWrappers.atomic_pointerset,
        #     CoDual(p, dp),
        #     1.0,
        #     :monotonic,
        # ),
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
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, 5.0f0, 4.0f0),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, 5.0f0, -3.0f0),
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
        # rem_float -- untested and unimplemented because seemingly unused on master
        # rem_float_fast -- untested and unimplemented because seemingly unused on master
        (false, :stability, nothing, IntrinsicsWrappers.rint_llvm, 5),
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
        (false, :none, nothing, compilerbarrier, :type, 5.0),
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
        # Core.svec -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, applicable, sin, Float64),
        (false, :stability, nothing, applicable, sin, Type),
        (false, :stability, nothing, applicable, +, Type, Float64),
        (false, :stability, nothing, applicable, +, Float64, Float64),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, StructFoo, :a),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, StructFoo, :b),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, MutableFoo, :a),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, MutableFoo, :b),
        (true, :none, _range, getfield, StructFoo(5.0), :a),
        (false, :none, _range, getfield, StructFoo(5.0, randn(5)), :a),
        (false, :none, _range, getfield, StructFoo(5.0, randn(5)), :b),
        (true, :none, _range, getfield, StructFoo(5.0), 1),
        (false, :none, _range, getfield, StructFoo(5.0, randn(5)), 1),
        (false, :none, _range, getfield, StructFoo(5.0, randn(5)), 2),
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
        (false, :none, nothing, getfield, (Float64, Float64), 1),
        (false, :none, nothing, getfield, (Float64, Float64), 2, false),
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
    ]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:builtins})
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
        (false, :none, nothing, getindex, randn(5), [1, 1]),
        (false, :none, nothing, getindex, randn(5), [1, 2, 2]),
        (false, :none, nothing, setindex!, randn(5), [4.0, 5.0], [1, 1]),
        (false, :none, nothing, setindex!, randn(5), [4.0, 5.0, 6.0], [1, 2, 2]),
    ]
    return test_cases, Any[]
end
