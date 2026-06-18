# Forwards-Mode Design

The purpose of this document is to explain how forwards-mode AD in Mooncake.jl is implemented.
It should do so to a sufficient level of depth to enable the interested reader to read the forwards-mode AD code in Mooncake.jl and understand what is going on.

This document
1. specifies the semantics of a "rule" for forwards-mode AD,
1. specifies how to implement rules by-hand for primitives,
1. specifies how to derive rules from `IRCode` algorithmically in general,
1. discusses batched ("chunked") forwards-mode,
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
For `rule_for_f` to be a valid forwards-rule for `f`, it must be applicable to `Lifted`s as follows:
```julia
z_dz = rule_for_f(lift(f, df), lift(x, dx), lift(y, dy))::Lifted
```
where:
1. `rule_for_f` is a callable. It might be written by-hand, or derived algorithmically.
1. `df`, `dx`, and `dy` are tangents for `f`, `x`, and `y` respectively. They are the inputs to the derivative of `(f, x, y)`; the tangent carried by `z_dz` is the output.
1. `z_dz` is a `Lifted` containing the primal `z` and the component of the derivative of `(f, x, y)` in the direction `(df, dx, dy)` associated with `z`.
1. running `rule_for_f` leaves `f`, `x`, and `y` in the same state that running `f` does.

We refer readers to [Algorithmic Differentiation](@ref) to explain what we mean when we talk about the "derivative" above.

Note that `rule_for_f` is an as-yet-unspecified callable which we introduced purely to specify the interface that a forwards-rule must satisfy.
In [Hand-Written Rules](@ref) and [Derived Rules](@ref) below, we introduce two concrete ways to produce rules for `f`.

### The `Lifted` slot and the forward value `V`

A forwards-mode argument is a [`Mooncake.Lifted`](@ref) slot:
```julia
struct Lifted{P, N, V}
    primal::P
    value::V
end
```
`P` is the primal type, `N` is the *chunk width* (the number of derivative directions propagated at once — `N == 1` for a standard single-direction rule, `N > 1` for batched mode), and `V` is the *forward value*: the representation of `N` tangents of `P` packed together.

For a concrete `P` it must always hold that `V === dual_type(Val(N), P)`.
`dual_type` plays the same role for forwards-mode that [`tangent_type`](@ref) plays for reverse-mode, and it is *recursively coherent* with it: where the reverse representation of a component is `tangent_type(component)`, the forward representation is `dual_type(Val(N), component)`, mirroring each other shape-for-shape. Concretely:

- a real scalar `P` → `NDual{P, N}` (a value plus an `NTuple{N}` of per-lane partials);
- an array `Array{T, D}` → `NDualArray{T, N, D, ...}`, a *parallel-arrays* wrapper holding the primal array and `N` partial arrays separately (each a genuine `Array{T, D}` usable directly in a `ccall`);
- a struct → `ImmutableDual` / `MutableDual` wrapping the per-field forward values;
- tuples / named-tuples → element-wise recursion;
- a non-differentiable `P` (integers, `Symbol`, `Module`, types, …) → [`Mooncake.NoDual`](@ref), the forwards-mode analogue of reverse-mode's `NoTangent`.

Rules read and write slots through the accessors rather than touching the fields directly: `primal(slot)`, `tangent(slot)` (the whole `V`), and `tangent(slot, lane)` (the `lane`-th direction's partial). This keeps rule bodies independent of the inner-`V` shape — see [Hand-Written Rules](@ref).

### Testing

Suppose that we have (somehow) produced a supposed forwards-rule. To check that it is correctly implemented, we must ensure that
1. all primal state after running the rule is approximately the same as all primal state after running the primal, and
2. the inner product between all tangents (both output and input) and a random tangent vector after running the rule is approximately the same as the estimate of the same quantity produced by finite differencing or reverse-mode AD.

We already have the functionality to do this in a very general way (see [`Mooncake.TestUtils.test_rule`](@ref)), which additionally checks the chunk widths `N = 1, 2, 3` and the canonical-`V` coherence invariant.

## Hand-Written Rules

Hand-written rules are implemented by writing methods of two functions: `is_primitive` and `frule!!`.

### `is_primitive`

`is_primitive(context, ::Type{ForwardMode}, signature::Type{<:Tuple}, world)` must return `true` if AD must attempt to differentiate a call by passing the arguments to `frule!!`, and `false` otherwise.
`context` is a context type such as `Mooncake.MinimalCtx` or `Mooncake.DefaultCtx` — the *same* contexts used for reverse-mode; the mode is selected by the `ForwardMode` argument.
The [`Mooncake.@is_primitive`](@ref) macro is used to declare new primitives, e.g.
```julia
@is_primitive MinimalCtx ForwardMode Tuple{typeof(sin), Float64}
```
The declaration must stay in lockstep with the `frule!!` method coverage: a broader `@is_primitive` than the rule's methods fails only at call time with a `MethodError`.

### `frule!!`

Methods of `frule!!` do the actual differentiation, and must satisfy the [Forwards-Rule Interface](@ref) discussed above.
A `frule!!` must return the canonical `dual_type(Val(N), typeof(result))` shape (use `zero_dual` / `zero_lifted` for a zero derivative); it must never pair a `NoDual` with a differentiable result, nor double-wrap a `Lifted`.

In what follows, we will refer to `frule!!`s for signatures.
For example, the `frule!!` for signature `Tuple{typeof(sin), Float64}` is the rule which would differentiate calls like `sin(5.0)`.

#### Simple Scalar Function

Recall that for ``y = \sin(x)`` we have that ``\dot{y} = \cos(x) \dot{x}``.
The scalar `NDual` overloads compute the value and its partials in a single call (here `sin` of an `NDual` returns an `NDual` whose `.value` is `sin(x)` and whose partials carry ``\cos(x)\dot{x}``).
So the `frule!!` for signature `Tuple{typeof(sin), Float64}` is, for any chunk width `N`:
```julia
function frule!!(::Lifted{typeof(sin), N}, x::Lifted{Float64, N, NDual{Float64, N}}) where {N}
    dy = sin(tangent(x))                 # one call: value AND all N lane partials
    return Lifted{Float64, N}(dy.value, dy)   # read the primal back from dy (inner-value invariant)
end
```
Note that the primal `sin(x)` is read out of `dy.value` rather than recomputed — `f(tangent(x))` has already evaluated it, so a separate `sin(primal(x))` would run a transcendental twice. This "do not recompute the primal" discipline is the dominant cost in element-wise forward AD.

#### Pre-allocated Matrix-Matrix Multiply

Recall that for ``Z = X Y`` we have that ``\dot{Z} = X \dot{Y} + \dot{X} Y``.
Because the forward value of an array is the parallel-arrays `NDualArray` (primal and each lane's partials are genuine `Array`s), we can apply the primal `mul!` once and then a per-lane `mul!` over the partial arrays:
```julia
function frule!!(
    ::Lifted{typeof(mul!), N},
    Z::Lifted{P, N, <:NDualArray}, X::Lifted{P, N, <:NDualArray}, Y::Lifted{P, N, <:NDualArray},
) where {N, P<:Matrix{Float64}}
    mul!(primal(Z), primal(X), primal(Y))            # primal update, once
    for k in 1:N
        dZ, dX, dY = tangent(Z, k), tangent(X, k), tangent(Y, k)
        mul!(dZ, primal(X), dY)                       # X * Ẏ
        mul!(dZ, dX, primal(Y), 1.0, 1.0)            # + Ẋ * Y
    end
    return Z
end
```
The in-place primal update is hoisted out of the per-lane loop: repeating it would corrupt the shared primal seen by later lanes.
(In practice we would implement a rule for a lower-level function like `LinearAlgebra.BLAS.gemm!`, rather than `mul!`.)


## Derived Rules

This is the "automatic" / "algorithmic" bit of AD!
This is the second way of producing concrete callable objects which satisfy the [Forwards-Rule Interface](@ref) discussed above.
The object which we will ultimately construct is an instance of `Mooncake.DerivedFRule`.

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
function rule_for_f(::Lifted{typeof(f)}, x::Lifted)
    y = rule_for_g(zero_lifted(Val(1), g), x)
    z = rule_for_h(zero_lifted(Val(1), h), x, y)
    return z
end
```
Observe that the transformation is simply
1. replace all variables with `Lifted` variables,
1. replace all constants (e.g. `g` and `h`) with constant `Lifted`s (with zero tangent),
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
Roughly speaking, the forwards-mode IR for the (ficticious) function `rule_for_f` should look something like:
```julia
julia> Base.code_ircode_by_type(Tuple{typeof(rule_for_f), Lifted{typeof(f), 1, NoDual}, Lifted{Float64, 1, NDual{Float64, 1}}})
1-element Vector{Any}:
2 1 ─ %1 = invoke rule_for_g($(zero_lifted(Val(1), Main.g)), _3::Lifted{Float64, 1, NDual{Float64, 1}})::Lifted{Float64, 1, NDual{Float64, 1}}
3 │   %2 = invoke rule_for_h($(zero_lifted(Val(1), Main.h)), _3::Lifted{Float64, 1, NDual{Float64, 1}}, %1::Lifted{Float64, 1, NDual{Float64, 1}})::Lifted{Float64, 1, NDual{Float64, 1}}
4 └──      return %2
   => Lifted{Float64, 1, NDual{Float64, 1}}
```
Observe that:
1. All `Argument`s have been incremented by `1`. i.e. `_2` has been replaced with `_3`. This corresponds to the fact that the arguments to the rule have all been shuffled along by one, and the rule itself is now the first argument.
1. Everything has been turned into a `Lifted`.
1. Constants such as `g` appear as constant `Lifted`s with a zero tangent (`zero_lifted(Val(1), Main.g)`).

Here, as before, we have not specified exactly what `rule_for_f`, `rule_for_g`, and `rule_for_h` are.
This is intentional -- they are just callables satisfying the [Forwards-Rule Interface](@ref).
In the following we show how to derive `rule_for_f`, and show how `rule_for_g` and `rule_for_h` might be methods of `Mooncake.frule!!`, or themselves derived rules.

#### Rule Derivation Outline

Equipped with some intuition about what a derived rule ought to look like, we examine how we go about producing it algorithmically.

Rule derivation is implemented via the function `Mooncake.build_frule`.
This function accepts a context and a signature / `Base.MethodInstance` / `MistyClosure` (plus a `chunk_size` keyword, see [Batch Mode](@ref)) and, roughly speaking, does the following:
1. Look up the optimised `Compiler.IRCode`.
1. Apply a series of standardising transformations to the `IRCode`.
1. Transform each statement according to a set of rules to produce a new `IRCode`.
1. Apply standard Julia optimisations to this new `IRCode`.
1. Put this code inside a `MistyClosure` in order to produce an executable object.
1. Wrap this `MistyClosure` in a `DerivedFRule` to handle various bits of book-keeping around varargs.


In order:

#### Looking up the `Compiler.IRCode`.

This is done using `Mooncake.lookup_ir`.
This function has methods which return the `IRCode` associated to:
1. signatures (e.g. `Tuple{typeof(f), Float64}`)
1. `Base.MethodInstance`s (relevant for `:invoke` expressions -- see [Statement Transformation](@ref) below)
1. `MistyClosures.MistyClosure` objects, which is essential when computing higher order derivatives and Hessians by applying Mooncake.jl to itself.

#### [Standardisation](@id standardisation)

We apply the following transformations to the Julia IR.
They can all be found in `ir_normalisation.jl`:

1. [`Mooncake.foreigncall_to_call`](@ref): convert `Expr(:foreigncall, ...)` expressions into `Expr(:call, Mooncake._foreigncall_, ...)` expressions.
1. [`Mooncake.new_to_call`](@ref): convert `Expr(:new, ...)` expressions to `Expr(:call, Mooncake._new_, ...)` expressions.
1. [`Mooncake.splatnew_to_call`](@ref): convert `Expr(:splatnew, ...)` expressions to `Expr(:call, Mooncake._splat_new_...)` expressions.
1. [`Mooncake.intrinsic_to_function`](@ref): convert `Expr(:call, ::IntrinsicFunction, ...)` to calls to the corresponding function in [Mooncake.IntrinsicsWrappers](@ref).

The purpose of converting `Expr(:foreigncall...)`, `Expr(:new, ...)` and `Expr(:splatnew, ...)` into `Expr(:call, ...)`s is to enable us to differentiate such expressions by adding methods to `frule!!(::Lifted{typeof(Mooncake._foreigncall_)})`, `frule!!(::Lifted{typeof(Mooncake._new_)})`, and `frule!!(::Lifted{typeof(Mooncake._splat_new_)})`, in exactly the same way that we would for any other regular Julia function.

The purpose of translating `Expr(:call, ::IntrinsicFunction, ...)` is to do with type stability -- see the docstring for the [Mooncake.IntrinsicsWrappers](@ref) module for more info.


#### Statement Transformation

Each statement which can appear in the Julia IR is transformed by a method of `Mooncake.modify_fwd_ad_stmts!`.
Consequently, this transformation phase simply corresponds to iterating through all of the expressions in the `IRCode`, applying `Mooncake.modify_fwd_ad_stmts!` to each, modifying the `IRCode` in place.
To understand how to modify `IRCode` and insert new instructions, see [Oxinabox's Gist](https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802).

We provide here a high-level summary of the transformations for the most important Julia IR statements, and refer readers to the methods of `Mooncake.modify_fwd_ad_stmts!` for the definitive explanation of what transformation is applied, and the rationale for applying it.
In particular there are quite a number more statements which can appear in Julia IR than those listed here and, for those we do list here, there are typically a few edge cases left out.

**`Expr(:invoke, method_instance, f, x...)` and `Expr(:call, f, x...)`**

`:call` expressions correspond to _dynamic_ dispatch, while `:invoke` expressions correspond to _static_ dispatch.
That is, if you see an `:invoke` expression, you know for sure that the compiler knows enough information about the types of `f` and `x` to prove exactly which specialisation of which method to call.
This specialisation is `method_instance`.
Conversely, a `:call` expression typically occurs when the compiler has not been able to deduce the exact types of `f` and `x`, and therefore has to wait until runtime to figure out what to call, resulting in dynamic dispatch.

As we saw earlier, the idea is to translate these kinds of expressions into something vaguely along the lines of
```julia
Expr(:call, rule_for_f, f, x...)
```
There are three cases to consider, in order of preference:

Primitives:

If `is_primitive` returns `true` when applied to the signature constructed from the static types of `f` and `x`, then we simply replace the expression with `Expr(:call, frule!!, f, x...)`, regardless whether we have an `:invoke` or `:call` expression.
(Due to the [Standardisation](@ref standardisation) steps, it regularly happens that we see `:call` expressions in which we actually do know enough type information to do this, e.g. for `Mooncake._new_` `:call` expressions).

Static Dispatch:

In the case of `:invoke` nodes we know for sure at rule compilation time what `rule_for_f` must be.
We derive a rule for the call by passing `method_instance` to `Mooncake.build_frule`.
(In practice, we might do this lazily, but while retaining enough information to maintain type stability, mirroring the `Mooncake.LazyDerivedRule` used in reverse-mode).

Dynamic Dispatch:

If we have a `:call` expression and are not able to prove that `is_primitive` will return `true`, we must defer dispatch until runtime.
We do this by replacing the `:call` expression with a call to a `Mooncake.DynamicFRule`, which simply constructs (or retrieves from a cache) the rule at runtime.
Reverse-mode utilises a similar strategy via `Mooncake.DynamicDerivedRule`.



The above was written in terms of `f` and `x`.
In practice, of course, we encounter various kinds of constants (e.g. `Base.sin`), `Argument`s (e.g. `_3`), and `Core.SSAValue`s (e.g. `%5`).
The translation rules for these are:
1. constants are turned into constant `Lifted`s in which the tangent is zero (via `zero_lifted(Val(N), v)`),
1. `Argument`s are incremented by `1`,
1. `SSAValue`s are left as-is.

**`Core.GotoNode`s**

These remain entirely unchanged.

**`Core.GotoIfNot`**

These require minor modification.
Suppose that a `Core.GotoIfNot` of the form `Core.GotoIfNot(%5, 4)` is encountered in the primal.
Since `%5` will be a `Lifted` in the derived rule, we must pull out its primal field and pass that to the conditional instead.
Therefore, these statements get lowered to two lines in the derived rule. For example, `Core.GotoIfNot(%5, 4)` becomes a call to `Mooncake._primal` on the (incremented) condition, immediately followed by the `GotoIfNot` on the result:
```julia
%n = Mooncake._primal(%5)
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
1. constants are translated into constant `Lifted`s (with a zero tangent, via `zero_lifted`).
1. undefined values remain undefined.

The node's declared type is also widened to `lifted_type(Val(N), ...)`. So the above example would be translated into something like
```julia
φ (#1 => %3, #2 => _3, #3 => $(zero_lifted(Val(1), 4)), #4 => #undef)
```
The related `PiNode`, `UpsilonNode`, and `PhiCNode` statements are handled analogously (increment `Argument`s, `zero_lifted` constants, widen the declared type with `lifted_type`).

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

So far, we have mostly assumed that we would only apply forwards-mode to a single tangent vector at a time (chunk width `N == 1`).
However, in practice, it is typically best to pass a collection of tangents through at a time — for example, computing a gradient or a Jacobian via forwards-mode requires one derivative direction per input degree of freedom, and propagating them in chunks amortises the cost of the primal computation.

No separate "batched" transformation is needed: the forward value `V = dual_type(Val(N), P)` is *already* parameterised by the chunk width `N`. Every statement transformation above threads the width (`Val(info.width)`) into the constant `zero_lifted`s and the `lifted_type` annotations, and the hand-written `frule!!`s loop over the `N` lanes (`tangent(slot, k)`) as shown in the matrix-multiply example. A width-`N` rule is obtained simply by passing `chunk_size = N` to `Mooncake.build_frule`; the `NDual` / `NDualArray` forward values then carry `N` partials per primal instead of one.

## Forwards vs Reverse Implementation

The implementation of forwards-mode AD is quite dramatically simpler than that of reverse-mode AD.
Some notable technical differences include:
1. forwards-mode AD only makes use of the (forward) tangent system — the `Lifted` slot and its `dual_type` forward value — whereas reverse-mode also makes use of the fdata / rdata system.
1. forwards-mode AD comprises only line-by-line transformations of the `IRCode`. In particular, it does not require the insertion of additional basic blocks, nor the modification of the successors / predecessors of any given basic block (the `GotoIfNot` rewrite inserts a node within the same block). Consequently, there is no need to make use of the `BBCode` infrastructure built up for reverse-mode AD -- everything can be straightforwardly done at the `Compiler.IRCode` level.

## Comparison with ForwardDiff.jl

With reference to [the limitations of ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/#Limitations-of-ForwardDiff), there are a few noteworthy differences between ForwardDiff.jl and this implementation:
1. `:foreigncall`s pose much less of a problem for Mooncake's forward-mode than for ForwardDiff.jl, because we can write a rule for any method of any function. In essence, you can only (reliably) write rules for ForwardDiff.jl via dispatch on `ForwardDiff.Dual`.
1. the target function can be of any arity in Mooncake.jl, but must be unary in ForwardDiff.jl.
1. there are no limitations on the argument type constraints that Mooncake.jl can handle, while ForwardDiff.jl requires that argument type constraints be `<:Real` or arrays of `<:Real`.
1. No special storage types are required with Mooncake.jl, while ForwardDiff.jl requires that any container you write to is able to contain `ForwardDiff.Dual`s. (Mooncake's array forward value, `NDualArray`, keeps the partials in *separate* native arrays rather than interleaving them into the primal container.)
