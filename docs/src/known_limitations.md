# Known Limitations

Mooncake.jl has a number of known qualitative limitations, which we document here.

## Mutation of Global Variables

```@meta
DocTestSetup = quote
    using Mooncake
    using Mooncake: NoTangent, build_rrule
end
```

While great care is taken in this package to prevent silent errors, this is one edge case that we have yet to provide a satisfactory solution for.
Consider a function of the form:
```jldoctest bad_globalref
julia> const x = Ref(1.0);

julia> function foo(y::Float64)
           x[] = y
           return x[]
       end
foo (generic function with 1 method)
```
`x` is a global variable (if you refer to it in your code, it appears as a `GlobalRef` in the AST or lowered code).
For some technical reasons that are beyond the scope of this section, this package cannot propagate gradient information through `x`.
`foo` is the identity function, so it should have gradient `1.0`.
However, if you differentiate this example, you'll see:
```jldoctest bad_globalref
julia> rule = Mooncake.build_rrule(foo, 2.0);

julia> Mooncake.value_and_gradient!!(rule, foo, 2.0)
(2.0, (NoTangent(), 0.0))
```
Observe that while it has correctly computed the identity function, the gradient is zero.

The takehome: do not attempt to differentiate functions which modify global state. Uses of globals which does not involve mutating them is fine though.

## Circular References

To a large extent, Mooncake.jl does not presently support circular references in an automatic fashion.
It is generally possible to hand-write solutions, so we explain some of the problems here, and the general approach to resolving them.

### Tangent Types

_**The Problem**_

Suppose that you have a type such as:
```julia
mutable struct A
    x::Float64
    a::A
    function A(x::Float64)
        a = new(x)
        a.a = a
        return a
    end
end
```

This is a fairly canonical example of a self-referential type.
There are a couple of things which will not work with it out-of-the-box.
`tangent_type(A)` will produce a stack overflow error.
To see this, note that it will in effect try to produce a tangent of type `Tangent{Tuple{tangent_type(A)}}` -- the circular dependency on the `tangent_type` function causes real problems here.

_**The Solution**_

In order to resolve this, you need to produce a tangent type by hand.
You might go with something like
```julia
mutable struct TangentForA
    x::Float64 # tangent type for Float64 is Float64
    a::TangentForA
    function TangentForA(x::Float64)
        a = new(x)
        a.a = a
        return a
    end
end
```
The point here is that you can manually resolve the circular dependency using a data structure which mimics the primal type.
You will, however, need to implement similar methods for `zero_tangent`, `randn_tangent`, etc, and presumably need to implement additional `getfield` and `setfield` rules which are specific to this type.

### Circular References in General

_**The Problem**_

Consider a type of the form
```julia
mutable struct Foo
    x
    Foo() = new()
end
```
In this instance, `tangent_type` will work fine because `Foo` does not directly reference itself in its definition.
Moreover, general uses of `Foo` will be fine.

However, it's possible to construct an instance of `Foo` with a circular reference:
```julia
f = Foo()
f.x = f
```
This is actually fine provided we never attempt to call `zero_tangent` / `randn_tangent` / similar functionality on `f` once we've set its `x` field to itself.
If we attempt to call such a function, we'll find ourselves with a stack overflow.

_**The Solution**_
This is a little tricker to handle.
You could specialise `zero_tangent` etc for `Foo`, but this is something of a pain.
Fortunately, it seems to be incredibly rare that this is ever a problem in practice.
If we gain evidence that this _is_ often a problem in practice, we'll look into supporting `zero_tangent` etc automatically for this case.


## Tangent Generation and Pointers

_**The Problem**_


In many use cases, a pointer provides the address of the start of a block of memory which has been allocated to e.g. store an array.
However, we cannot get any of this context from the pointer itself -- by just looking at a pointer, I cannot know whether its purpose is to refer to the start of a large block of memory, some proportion of the way through a block of memory, or even to keep track of a single address.

Recall that the tangent to a pointer is another pointer:
```jldoctest
julia> Mooncake.tangent_type(Ptr{Float64})
Ptr{Float64}
```
Plainly I cannot implement a method of `zero_tangent` for `Ptr{Float64}` because I don't know how much memory to allocate.

This is, however, fine if a pointer appears half way through a function, having been derived from another data structure. e.g.
```jldoctest
function foo(x::Vector{Float64})
    p = pointer(x, 2)
    return unsafe_load(p)
end

rule = build_rrule(Tuple{typeof(foo), Vector{Float64}})
Mooncake.value_and_gradient!!(rule, foo, [5.0, 4.0])

# output
(4.0, (NoTangent(), [0.0, 1.0]))
```

_**The Solution**_

This is only really a problem for tangent / fdata / rdata generation functionality, such as `zero_tangent`.
As a work-around, AD testing functionality permits users to pass in `CoDual`s.
So if you are testing something involving a pointer, you will need to construct its tangent yourself, and pass a `CoDual` to e.g. `Mooncake.TestUtils.test_rule`.

While pointers tend to be a low-level implementation detail in Julia code, you could in principle actually be interested in differentiating a function of a pointer.
In this case, you will not be able to use `Mooncake.value_and_gradient!!` as this requires the use of `zero_tangent`.
Instead, you will need to use lower-level (internal) functionality, such as `Mooncake.__value_and_gradient!!`, or use the rule interface directly.

Honestly, your best bet is just to avoid differentiating functions whose arguments are pointers if you can.

```@meta
DocTestSetup = nothing
```
