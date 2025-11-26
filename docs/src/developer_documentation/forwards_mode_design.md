# Forwards-Mode Design

**Disclaimer**: this document refers to an as-yet-unimplemented forwards-mode AD. This will disclaimer will be removed once it has been implemented.

The purpose of this document is to explain how forwards-mode AD in Mooncake.jl is implemented.
It should do so to a sufficient level of depth to enable the interested reader to read to the forwards-mode AD code in Mooncake.jl and understand what is going on.

This document
1. specifies the semantics of a "rule" for forwards-mode AD,
1. specifies how to implement rules by-hand for primitives,
1. specifies how to derive rules from `IRCode` algorithmically in general,
1. discusses batched forwards-mode,
1. discusses some notable technical differences between our forwards-mode AD implementation details and reverse-mode AD implementation details, and
1. concludes with a brief comparison with ForwardDiff.jl.

## Forwards-Rule Interface

Loosely, a rule for a function simultaneously
1. performs the same computation as the original function, and
1. computes the Frechet derivative.

This is best explained through a worked example.
Consider a function call
```julia
z = f(x, y)
```
where `f` itself may contain data / state which is modified by executing `f`.
`rule_for_f` is _some_ callable which claims to be a forwards-rule for `f`.
For `rule_for_f` to be a valid forwards-rule for `f`, it must be applicable to `Dual`s as follows:
```julia
z_dz = rule_for_f(Dual(f, df), Dual(x, dx), Dual(y, dy))::Dual
```
where:
1. `rule_for_f` is a callable. It might be written by-hand, or derived algorithmically.
1. `df`, `dx`, and `dy` are tangents for `f`, `x`, and `y` respectively. Before executing `rule_for_f`, they are inputs to the derivative of `(f, x, y)`. After executing they are outputs of this derivative.
1. `z_dz` is a `Dual` containing the primal and the component of the derivative of `(f, x, y)` to `(df, dx, dy)` associated to `z`.
1. running `rule_for_f` leaves `f`, `x`, and `y` in the same state that running `f` does.

We refer readers to [Algorithmic Differentiation](@ref) to explain what we mean when we talk about the "derivative" above.
We also discussed some worked examples shortly.

Note that `rule_for_f` is an as-yet-unspecified callable which we introduced purely to specify the interface that a forwards-rule must satisfy.
In [Hand-Written Rules](@ref) and [Derived Rules](@ref) below, we introduce two concrete ways to produce rules for `f`.

### Tangent Types

We will use the type system documented in [Representing Gradients](@ref).
This means that every primal type has a unique tangent type.
Moreover, if a `Dual` is defined as follows:
```julia
struct Dual{P, T}
    primal::P
    tangent::T
end
```
it must always hold that `T = tangent_type(P)`.


### Testing

Suppose that we have (somehow) produced a supposed forwards-rule. To check that it is correctly implemented, we must ensure that
1. all primal state after running the rule is approximately the same as all primal state after running the primal, and
2. the inner product between all tangents (both output and input) and a random tangent vector after running the rule is approximately the same as the estimate of the same quantity produced by finite differencing or reverse-mode AD.

We already have the functionality to do this in a very general way (see [`Mooncake.TestUtils.test_rule`](@ref)).

## Hand-Written Rules

Hand-written rules are implemented by writing methods of two functions: `is_primitive` and `frule!!`.

### `is_primitive`

`is_primitive(::Type{<:Union{MinimalForwardsCtx, DefaultForwardsCtx}}, signature::Type{<:Tuple}, world)` must return `true` if AD must attempt to differentiate a call by passing the arguments to `frule!!`, and `false` otherwise.
The [`Mooncake.@is_primitive`](@ref) macro must be used to extend to create new primitives.

### `frule!!`

Methods of `frule!!` do the actual differentiation, and must satisfy the [Forwards-Rule Interface](@ref) discussed above.

In what follows, we will refer to `frule!!`s for signatures.
For example, the `frule!!` for signature `Tuple{typeof(sin), Float64}` is the rule which would differentiate calls like `sin(5.0)`.

#### Simple Scalar Function

Recall that for ``y = \sin(x)`` we have that ``\dot{y} = \cos(x) \dot{x}``.
So the `frule!!` for signature `Tuple{typeof(sin), Float64}` is:
```julia
function frule!!(::Dual{typeof(sin)}, x::Dual{Float64})
    return Dual(sin(x.primal), cos(x.primal) * x.tangent)
end
```

#### Pre-allocated Matrix-Matrix Multiply

Recall that for ``Z = X Y`` we have that ``\dot{Z} = X \dot{Y} + \dot{X} Y``.
So the `frule!!` for signature `Tuple{typeof(mul!), Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}` is:
```julia
function frule!!(
    ::Dual{typeof(LinearAlgebra.mul!)}, Z::Dual{P}, X::Dual{P}, Y::Dual{P}
) where {P<:Matrix{Float64}}

    # Primal computation.
    mul!(Z.primal, X.primal, Y.primal)

    # Overwrite tangent of `z` to contain propagated tangent.
    mul!(Z.tangent, X.primal, Y.tangent)

    # Add the result of x.tangent * y.primal to `z.tangent`.
    mul!(Z.tangent, X.tangent, Y.primal, 0.0, 1.0) 
    return Z
end
```
(In practice we would probably implement a rule for a lower-level function like `LinearAlgebra.BLAS.gemm!`, rather than `mul!`).


## Derived Rules

This is the "automatic" / "algorithmic" bit of AD!
This is the second way of producing concrete callable objects which satisfy the [Forwards-Rule Interface](@ref) discussed above.
The object which we will ultimately construct is an instance `Mooncake.DerivedFRule`.

#### Worked Example: Julia Function

Before explaining how derived rules are produced algorithmically, we explain by way of example what a derived rule should look like if we work things through by hand.

A derived rule for a function such as
```julia
function f(x)
    y = g(x)
    z = h(x, y)
    return z
end
```
should be something of the form
```julia
function rule_for_f(::Dual{typeof(f)}, x::Dual)
    y = rule_for_g(zero_dual(g), x)
    z = rule_for_h(zero_dual(h), x, y)
    return z
end
```
Observe that the transformation is simply
1. replace all variables with `Dual` variables,
1. replace all constants (e.g. `g` and `h`) with constant `Dual`s,
1. replace all calls with calls to rules.

In general, all control flow should be identical between primal and rule.

#### Worked Example: IRCode

The above example is expressed in terms of Julia code, but we will be operating on Julia `Compiler.IRCode`, so it is helpful to consider how the above example translates into this form.
If we call `f` on a `Float64`, and suppose that `g` and `h` both return `Float64`s, the primal `Compiler.IRCode` will look something like the following:
```julia
julia> Base.code_ircode_by_type(Tuple{typeof(f), Float64})
1-element Vector{Any}:
2 1 ─ %1 = invoke Main.g(_2::Float64)::Float64
3 │   %2 = invoke Main.h(_2::Float64, %1::Float64)::Float64
4 └──      return %2
   => Float64
```
Recall that `_2` is the second argument, in this case a `Float64`, and `%1` and `%2` are `SSAValue`s.
Roughly speaking, the forwards-mode IR for the (ficiticious) function `rule_for_f` should look something like:
```julia
julia> Base.code_ircode_by_type(Tuple{typeof(rule_for_f), Dual{typeof(f), NoTangent}, Dual{Float64, Float64}})
1-element Vector{Any}:
2 1 ─ %1 = invoke rule_for_g($(Dual(Main.g, NoTangent())), _3::Dual{Float64, Float64})::Dual{Float64, Float64}
3 │   %2 = invoke rule_for_h($(Dual(Main.h, NoTangent())), _3::Dual{Float64, Float64}, %1::Dual{Float64, Float64})::Dual{Float64, Float64}
4 └──      return %2
   => Dual{Float64, Float64}
```
Observe that:
1. All `Argument`s have been incremented by `1`. i.e. `_2` has been replaced with `_3`. This corresponds to the fact that the arguments to the rule have all been shuffled along by one, and the rule itself is now the first argument.
1. Everything has been turned into a `Dual`.
1. Constants such as `Dual(Main.g, NoTangent())` appear directly in the code (here as `QuoteNode`s).

(In practice it might be that we actually construct the `Dual`ed constants on the lines immediately preceding a call and rely on the compiler to optimise them back into the call directly).

Here, as before, we have not specified exactly what `rule_for_f`, `rule_for_g`, and `rule_for_h` are.
This is intentional -- they are just callables satisfying the [Forwards-Rule Interface](@ref).
In the following we show how to derive `rule_for_f`, and show how `rule_for_g` and `rule_for_h` might be methods of `Mooncake.frule!!`, or themselves derived rules.

#### Rule Derivation Outline

Equipped with some intuition about what a derived rule ought to look like, we examine how we go about producing it algorithmically.

Rule derivation is implemented via the function `Mooncake.build_frule`.
This function accepts as arguments a context and a signature / `Base.MethodInstance` / `MistyClosure` and, roughly speaking, does the following:
1. Look up the optimised `Compiler.IRCode`.
1. Apply a series of standardising transformations to the `IRCode`.
1. Transform each statement according to a set of rules to produce a new `IRCode`.
1. Apply standard Julia optimisations to this new `IRCode`.
1. Put this code inside a `MistyClosure` in order to produce an executable object.
1. Wrap this `MistyClosure` in a `DerivedFRule` to handle various bits of book-keeping around varargs.


In order:

#### Looking up the `Compiler.IRCode`.

This is done using `Mooncake.lookup_ir`.
This function has methods with will return the `IRCode` associated to:
1. signatures (e.g. `Tuple{typeof(f), Float64}`)
1. `Base.MethodInstances` (relevant for `:invoke` expressions -- see [Statement Transformation](@ref) below)
1. `MistyClosures.MistyClosure` objects, which is essential when computing higher order derivatives and Hessians by applying Mooncake.jl to itself.

#### Standardisation

We apply the following transformations to the Julia IR.
They can all be found in `ir_normalisation.jl`:

1. [`Mooncake.foreigncall_to_call`](@ref): convert `Expr(:foreigncall, ...)` expressions into `Expr(:call, Mooncake._foreigncall_, ...)` expressions.
1. [`Mooncake.new_to_call`](@ref): convert `Expr(:new, ...)` expressions to `Expr(:call, Mooncake._new_, ...)` expressions.
1. [`Mooncake.splatnew_to_call`](@ref): convert `Expr(:splatnew, ...)` expressions to `Expr(:call, Mooncake._splat_new_...)` expressions.
1. [`Mooncake.intrinsic_to_function`](@ref): convert `Expr(:call, ::IntrinsicFunction, ...)` to calls to the corresponding function in [Mooncake.IntrinsicsWrappers](@ref).

The purpose of converting `Expr(:foreigncall...)`, `Expr(:new, ...)` and `Expr(:splatnew, ...)` into `Expr(:call, ...)`s is to enable us to differentiate such expressions by adding methods to `frule!!(::Dual{typeof(Mooncake._foreigncall_)})`, `frule!!(::Dual{typeof(Mooncake._new_)})`, and `frule!!(::Dual{typeof(Mooncake._splat_new_)})`, in exactly the same way that we would for any other regular Julia function.

The purpose of translating `Expr(:call, ::IntrinsicFunction, ...)` is to do with type stability -- see the docstring for the [Mooncake.IntrinsicsWrappers](@ref) module for more info.


#### Statement Transformation

Each statment which can appear in the Julia IR is transformed by a method of `Mooncake.make_fwds_ad_stmts`.
Consequently, this transformation phase simply corresponds to iterating through all of the expressions in the `IRCode`, applying `Mooncake.make_fwd_ad_stmts` to each to produce new `IRCode`.
To understand how to modify `IRCode` and insert new instructions, see [Oxinabox's Gist](https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802).

We provide here a high-level summary of the transformations for the most important Julia IR statements, and refer readers to the methods of `Mooncake.make_fwds_ad_stmts` for the definitive explanation of what transformation is applied, and the rationale for applying it.
In particular there are quite a number more statements which can appear in Julia IR than those listed here and, for those we do list here, there are typically a few edge cases left out.

**`Expr(:invoke, method_instance, f, x...)` and `Expr(:call, f, x...)`**

`:call` expressions correspond to _dynamic_ dispatch, while `:invoke` expressions correspond to _static_ dispatch.
That is, if you see an `:invoke` expression, you know for sure that the compiler knows enough information about the types of `f` and `x` to prove exactly which specialisation of which method to call.
This specialisation is `method_instance`.
This typically happens when the compiler is able to prove the types of `f` and `x`.
Conversely, a `:call` expression typically occurs when the compiler has not been able to deduce the exact types of `f` and `x`, and therefore not been able to figure out what to call.
It therefore has to wait until runtime to figure out what to call, resulting in dynamic dispatch.

As we saw earlier, the idea is to translate these kinds of expressions into something vaguely along the lines of
```julia
Expr(:call, rule_for_f, f, x...)
```
There are three cases to consider, in order of preference:

Primitives:

If `is_primitive` returns `true` when applied to the signature constructed from the static types of `f` and `x`, then we simply replace the expression with `Expr(:call, frule!!, f, x...)`, regardless whether we have an `:invoke` or `:call` expression.
(Due to the [Standardisation](@ref) steps, it regularly happens that we see `:call` expressions in which we actually do know enough type information to do this, e.g. for `Mooncake._new_` `:call` expressions).

Static Dispatch:

In the case of `:invoke` nodes we know for sure at rule compilation time what `rule_for_f` must be.
We derive a rule for the call by passing `method_instance` to `Mooncake.build_frule`.
(In practice, we might do this lazily, but while retaining enough information to maintain type stability. See the `Mooncake.LazyDerivedRule` for how this is handled in reverse-mode).

Dynamic Dispatch:

If we have a `:call` expression and are not able to prove that `is_primitive` will return `true`, we must defer dispatch until runtime.
We do this by replacing the `:call` expression with a call to a `DynamicFRule`, which simply constructs (or retrieves from a cache) the rule at runtime.
Reverse-mode utilises a similar strategy via `Mooncake.DynamicDerivedRule`.



The above was written in terms of `f` and `x`.
In practice, of course, we encounter various kinds of constants (e.g. `Base.sin`), `Argument`s (e.g. `_3`), and `Core.SSAValue`s (e.g. `%5`).
The translation rules for these are:
1. constants are turned into constant duals in which the tangent is zero,
1. `Arguments` are incremented by `1`.
1. `SSAValue`s are left as-is.

**`Core.GotoNode`s**

These remain entirely unchanged.

**`Core.GotoIfNot`**

These require minor modification.
Suppose that a `Core.GotoIfNot` of the form `Core.GotoIfNot(%5, 4)` is encountered in the primal.
Since `%5` will be a `Dual` in the derived rule, we must pull out the `primal` field, and pass that to the conditional instead.
Therefore, these statments get lowered to two lines in the derived rule.
For example, `Core.GotoIfNot(%5, 4)` would be translated to:
```julia
%n = getfield(%5, :primal)
Core.GotoIfNot(%n, 4)
```

**`Core.PhiNode`**

`Core.PhiNode` looks something like the following in the general case:
```julia
φ (#1 => %3, #2 => _2, #3 => 4, #4 => #undef)
```
They map from a collection of basic block numbers (`#1`, `#2`, etc) to values.
The values can be `Core.Argument`s, `Core.SSAValue`s, constants (literals and `QuoteNode`s), or undefined.

`Core.PhiNode`s in the primal are mapped to `Core.PhiNode`s in the rule.
They contain exactly the same basic block numbers, and apply the following translation rules to the values:
1. `Core.SSAValue`s are unchanged.
1. `Core.Argument`s are incremented by `1` (as always).
1. constants are translated into constant duals.
1. undefined values remain undefined.

So the above example would be translated into something like
```julia
φ (#1 => %3, #2 => _3, #3 => $(CoDual(4, NoTangent())), #4 => #undef)
```

#### Optimisation

The IR generated in the previous step will typically be uninferred, and suboptimal in a variety of ways.
We fix this up by running inference and optimisation on the generated `IRCode`.
This is implemented by [`Mooncake.optimise_ir!`](@ref).

#### Put IRCode in MistyClosure

Now that we have an optimised `IRCode` object, we need to turn it into something that can actually be run.
This can, in general, be straightforwardly achieved by putting it inside a `Core.OpaqueClosure`.
This works, but `Core.OpaqueClosure`s have the disadvantage that once you've constructed a `Core.OpaqueClosure` using an `IRCode`, it is not possible to get it back out.
Consequently, we use `MistyClosure`s, in order to keep the `IRCode` readily accessible if we want to access it later.

#### Put the MistyClosure in a DerivedFRule

See the implementation of `DerivedRule` (used in reverse-mode) for more context on this.
_This_ is the "rule" that users get.

## Batch Mode

So far, we have assumed that we would only apply forwards-mode to a single tangent vector at a time.
However, in practice, it is typically best to pass a collection of tangents through at a time.

In order to do this, all of the transformation code listed above can remain the same, we will just need to devise a system of "batched tangents".
Then, instead of propagating a "primal-tangent" pairs via `Dual`s, we propagate primal-tangent_batch pairs (perhaps also via `Dual`s).

## Forwards vs Reverse Implementation

The implementation of forwards-mode AD is quite dramatically simpler than that of reverse-mode AD.
Some notable technical differences include:
1. forwards-mode AD only makes use of the tangent system, whereas reverse-mode also makes use of the fdata / rdata system.
1. forwards-mode AD comprises only line-by-line transformations of the `IRCode`. In particular, it does not require the insertion of additional basic blocks, nor the modification of the successors / predecessors of any given basic block. Consequently, there is no need to make use of the `BBCode` infrastructure built up for reverse-mode AD -- everything can be straightforwardly done at the `Compiler.IRCode` level.

## Comparison with ForwardDiff.jl

With reference to [the limitations of ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/#Limitations-of-ForwardDiff), there are a few noteworthy differences between ForwardDiff.jl and this implementation:
1. `:foreigncall`s pose much less of a problem for Mooncake's forward-mode than for ForwardDiff.jl, because we can write a rule for any method of any function. In essence, you can only (reliably) write rules for ForwardDiff.jl via dispatch on `ForwardDiff.Dual`.
1. the target function can be of any arity in Mooncake.jl, but must be unary in ForwardDiff.jl.
1. there are no limitations on the argument type constraints that Mooncake.jl can handle, while ForwardDiff.jl requires that argument type constraints be `<:Real` or arrays of `<:Real`.
1. No special storage types are required with Mooncake.jl, while ForwardDiff.jl requires that any container you write to is able to contain `ForwardDiff.Dual`s.
