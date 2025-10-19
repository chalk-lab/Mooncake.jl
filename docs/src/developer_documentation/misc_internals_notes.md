# Misc. Internals Notes

This document contains an assortment of notes on some implementation details in Mooncake.jl.
It is occassionally helpful to have them here for reference, but they are typically not essential reading unless working on the specific parts of Mooncake.jl to which they pertain.

## `tangent_type` and friends

Last checked: 21/01/2025, Julia v1.10.7 / v1.11.2, Mooncake 0.4.

### Background

Mooncake.jl makes extensive use of `@generated` functions to ensure that its `tangent_type` function (among others) is both type-stable, and constant folds.
I recently changed how `tangent_type` is implemented in Mooncake.jl to ensure that the implementations respect some specific limitations of generated functions.
Here I outline the overall problem, the mistake the previous implementation made, and how [the recent changes](https://github.com/chalk-lab/Mooncake.jl/pull/426) fix it.

### `tangent_type`

`tangent_type` is a regular Julia function, which given a "primal" type returns another type, the tangent type. It is side-effect free, and its return value is determined entirely by the _type_ of its argument. This means it should be possible to constant-fold. For example, consider the following definitions:
```julia
tangent_type(::Type{Float64}) = Float64
tangent_type(::Type{P}) where {P<:Tuple} = Tuple{map(tangent_type, fieldtypes(P))...}
```
If we inspect the `IRCode` associated to this for `Float64`, we see that everything is as expected -- the function literally just returns `Float64`:
```julia
julia> Base.code_ircode(tangent_type, (Type{Float64}, ))[1]
 1 ─     return Main.Float64
  => Type{Float64}
```
A simple `Tuple` type will also have this property:
```julia
julia> Base.code_ircode(tangent_type, (Type{Tuple{Float64}}, ))[1]
 1 ─     return Tuple{Float64}
  => Type{Tuple{Float64}}
```
However, for even slightly more complicated types, things fall over:
```julia
julia> Base.code_ircode(tangent_type, (Type{Tuple{Tuple{Float64}}}, ))[1]
1 1 ─ %1 = Main.tangent_type::Core.Const(tangent_type)
  │   %2 = invoke %1(Tuple{Float64}::Type{Tuple{Float64}})::Type{<:Tuple}
  │   %3 = Core.apply_type(Tuple, %2)::Type{<:Tuple}
  └──      return %3
   => Type{<:Tuple}
```
This is just one specific example, but it is really very straightforward to find others, necessitating a hunt for a more robust way of implementing tangent_type.

### Bad Generated Function Implementation

You might think to instead implement `tangent_type` for `Tuple`s as follows:
```julia
bad_tangent_type(::Type{Float64}) = Float64
@generated function bad_tangent_type(::Type{P}) where {P<:Tuple}
    return Tuple{map(bad_tangent_type, fieldtypes(P))...}
end
bad_tangent_type(::Type{Float32}) = Float32
```
Since the generated function literally just returns the type that we want, it will definitely constant-fold:
```julia
julia> Base.code_ircode(bad_tangent_type, (Type{Tuple{Tuple{Float64}}}, ))[1]
1 1 ─     return Tuple{Tuple{Float64}}
   => Type{Tuple{Tuple{Float64}}}
```
However, this implementation has a crucial flaw: we rely on the definition of `bad_tangent_type` in the body of the `@generated` method of `bad_tangent_type`. This means that if we e.g. add methods to `bad_tangent_type` after the initial definition, they won't show up. For example, in the above block, we defined the method of `bad_tangent_type` for `Float32` _after_ that of `Tuple`s. This results in the following error when we call `bad_tangent_type(Tuple{Float32})`:
```julia
julia> bad_tangent_type(Tuple{Float32})
ERROR: MethodError: no method matching bad_tangent_type(::Type{Float32})
The applicable method may be too new: running in world age 26713, while current world is 26714.

Closest candidates are:
  bad_tangent_type(::Type{Float32}) (method too new to be called from this world context.)
   @ Main REPL[10]:1
  bad_tangent_type(::Type{Float64})
   @ Main REPL[8]:1
  bad_tangent_type(::Type{P}) where P<:Tuple
   @ Main REPL[9]:1

Stacktrace:
 [1] map(f::typeof(bad_tangent_type), t::Tuple{DataType})
   @ Base ./tuple.jl:355
 [2] #s1#1
   @ ./REPL[9]:2 [inlined]
 [3] var"#s1#1"(P::Any, ::Any, ::Any)
   @ Main ./none:0
 [4] (::Core.GeneratedFunctionStub)(::UInt64, ::LineNumberNode, ::Any, ::Vararg{Any})
   @ Core ./boot.jl:707
 [5] top-level scope
   @ REPL[12]:1
```

This behaviour of `@generated` functions is discussed in the [Julia docs](https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions) -- I would recommend reading this bit of the docs if you've not previously, as the explanation is quite clear.

### Good Generated Function Implementation

`@generated` functions can still come to our rescue though. A better implementation is as follows:
```julia
good_tangent_type(::Type{Float64}) = Float64
@generated function good_tangent_type(::Type{P}) where {P<:Tuple}
    exprs = map(p -> :(good_tangent_type($p)), fieldtypes(P))
    return Expr(:curly, :Tuple, exprs...)
end
good_tangent_type(::Type{Float32}) = Float32
```
This leads to generated code which constant-folds / infers correctly:
```julia
julia> Base.code_ircode(good_tangent_type, (Type{Tuple{Tuple{Float64}}}, ))[1]
1 1 ─     return Tuple{Tuple{Float64}}
   => Type{Tuple{Tuple{Float64}}}
```
I believe this works better because the recursion doesn't happen through another function, but appears directly in the function body. This is right at the edge of my understanding of Julia's compiler heuristics surrounding recursion though, so I might be mistaken.

It also behaves correctly under the addition of new methods of `good_tangent_type`, because `good_tangent_type` only appears in the expression returned by the generated function, not the body of the generated function itself:
```julia
julia> good_tangent_type(Tuple{Float32})
Tuple{Float32}
```

### Effects Etc

This implementation is nearly sufficient to guarantee correct performance in all situations.
However, in some cases it is possible that even this implementation will fall over. Annoyingly I've not managed to produce a MWE that is even vaguely minimal in order to support this example, so you'll just have to believe me.

Based on all of the examples that I have seen thus far, it appears to be true that if you just _tell_ the compiler that
1. for the same inputs, the function always returns the same outputs, and
2. the function has no side-effects, so can be removed,
everything will always constant fold nicely.
This can be achieved by using the `Base.@assume_effects` macro in your method definitions, with the effects `:consistent` and `:removable`.


## How Recursion Is Handled

Last checked: 09/02/2025, Julia v1.10.8 / v1.11.3, Mooncake 0.4.82.

Mooncake handles recursive function calls by delaying code generation for generic function calls until the first time that they are actually run.
The docstring below contains a thorough explanation:

```@docs; canonical=false
Mooncake.LazyDerivedRule
```


