module Mooncake

const CC = Core.Compiler

using ADTypes,
    ChainRules, DiffRules, ExprTools, InteractiveUtils, LinearAlgebra, MistyClosures, Random

# There are many clashing names, so we will always qualify uses of names from CRC.
import ChainRulesCore as CRC

using Base:
    IEEEFloat,
    unsafe_convert,
    unsafe_pointer_to_objref,
    pointer_from_objref,
    arrayref,
    arrayset,
    TwicePrecision,
    twiceprecision
using Base.Experimental: @opaque
using Base.Iterators: product
using Core:
    Intrinsics,
    bitcast,
    SimpleVector,
    svec,
    ReturnNode,
    GotoNode,
    GotoIfNot,
    PhiNode,
    PiNode,
    SSAValue,
    Argument,
    OpaqueClosure,
    compilerbarrier
using Core.Compiler: IRCode, NewInstruction
using Core.Intrinsics: pointerref, pointerset
using LinearAlgebra.BLAS: @blasfunc, BlasInt, trsm!, BlasFloat
using LinearAlgebra.LAPACK: getrf!, getrs!, getri!, trtrs!, potrf!, potrs!
using FunctionWrappers: FunctionWrapper
using DispatchDoctor: @stable, @unstable

# Needs to be defined before various other things.
function _foreigncall_ end

"""
    rrule!!(f::CoDual, x::CoDual...)

Performs the forwards-pass of AD. The `tangent` field of `f` and each `x` should contain the
forwards tangent data (fdata) associated to each corresponding `primal` field.

Returns a 2-tuple.
The first element, `y`, is a `CoDual` whose `primal` field is the value associated to
running `f.primal(map(x -> x.primal, x)...)`, and whose `tangent` field is its associated
`fdata`.
The second element contains the pullback, which runs the reverse-pass. It maps from
the rdata associated to `y` to the rdata associated to `f` and each `x`.

```jldoctest
using Mooncake: zero_fcodual, CoDual, NoFData, rrule!!
y, pb!! = rrule!!(zero_fcodual(sin), CoDual(5.0, NoFData()))
pb!!(1.0)

# output

(NoRData(), 0.28366218546322625)
```
"""
function rrule!! end

"""
    build_primitive_rrule(sig::Type{<:Tuple})

Construct an rrule for signature `sig`. For this function to be called in `build_rrule`, you
must also ensure that `is_primitive(context_type, sig)` is `true`. The callable returned by
this must obey the rrule interface, but there are no restrictions on the type of callable
itself. For example, you might return a callable `struct`. By default, this function returns
`rrule!!` so, most of the time, you should just implement a method of `rrule!!`.

# Extended Help

The purpose of this function is to permit computation at rule construction time, which can
be re-used at runtime. For example, you might wish to derive some information from `sig`
which you use at runtime (e.g. the fdata type of one of the arguments). While constant
propagation will often optimise this kind of computation away, it will sometimes fail to do
so in hard-to-predict circumstances. Consequently, if you need certain computations not to
happen at runtime in order to guarantee good performance, you might wish to e.g. emit a
callable `struct` with type parameters which are the result of this computation. In this
context, the motivation for using this function is the same as that of using staged
programming (e.g. via `@generated` functions) more generally.
"""
build_primitive_rrule(::Type{<:Tuple}) = rrule!!

#! format: off
@stable default_mode = "disable" default_union_limit = 2 begin
include("utils.jl")
include("tangents.jl")
include("fwds_rvs_data.jl")
include("codual.jl")
include("debug_mode.jl")
include("stack.jl")

@unstable begin
include(joinpath("interpreter", "bbcode.jl"))
using .BasicBlockCode

include(joinpath("interpreter", "contexts.jl"))
include(joinpath("interpreter", "abstract_interpretation.jl"))
include(joinpath("interpreter", "patch_for_319.jl"))
include(joinpath("interpreter", "ir_utils.jl"))
include(joinpath("interpreter", "ir_normalisation.jl"))
include(joinpath("interpreter", "zero_like_rdata.jl"))
include(joinpath("interpreter", "s2s_reverse_mode_ad.jl"))
end

include("tools_for_rules.jl")
@unstable include("test_utils.jl")
@unstable include("test_resources.jl")

include(joinpath("rrules", "avoiding_non_differentiable_code.jl"))
include(joinpath("rrules", "blas.jl"))
include(joinpath("rrules", "builtins.jl"))
include(joinpath("rrules", "dispatch_doctor.jl"))
include(joinpath("rrules", "fastmath.jl"))
include(joinpath("rrules", "foreigncall.jl"))
include(joinpath("rrules", "function_wrappers.jl"))
include(joinpath("rrules", "iddict.jl"))
include(joinpath("rrules", "lapack.jl"))
include(joinpath("rrules", "linear_algebra.jl"))
include(joinpath("rrules", "low_level_maths.jl"))
include(joinpath("rrules", "misc.jl"))
include(joinpath("rrules", "new.jl"))
include(joinpath("rrules", "random.jl"))
include(joinpath("rrules", "tasks.jl"))
include(joinpath("rrules", "twice_precision.jl"))
@static if VERSION >= v"1.11-rc4"
    include(joinpath("rrules", "memory.jl"))
else
    include(joinpath("rrules", "array_legacy.jl"))
end
include(joinpath("rrules", "performance_patches.jl"))

include("interface.jl")
include("config.jl")
include("developer_tools.jl")

# Public, not exported
include("public.jl")
end
#! format: on

@public Config, value_and_pullback!!, prepare_pullback_cache

# Public, exported
export value_and_gradient!!, prepare_gradient_cache

end
