# Mooncake.jl's Rule System

Mooncake.jl's approach to AD is recursive.
It has a single specification for _what_ it means to differentiate a Julia callable, and basically two approaches to achieving this.
This section of the documentation explains the former.

We take an iterative approach to this explanation, starting at a high-level and adding more depth as we go.

# 10,000 Foot View

A rule `r(f, x)` for a `function` `f(x)` "does reverse mode AD", and executes in two phases, known as the forwards pass and the reverse pass.
In the forwards pass a rule executes the original `function`, and does some additional book-keeping in preparation for the reverse pass.
On the reverse pass it undoes the computation from the forwards pass, "backpropagates" the gradient w.r.t. the output of the original function by applying the adjoint of the derivative of the original `function` to it, and writes the results of this computation to the correct places.

A precise mathematical model for the original function is therefore entirely crucial to this discussion, as it is needed to understand what the adjoint of its derivative is.

# A Model For A Julia Function

Since Julia permits the in-place modification / mutation of many data structures, we cannot make a naive translation between a Julia function and a mathematical object.
Rather, we will have to model the state of the arguments to a function both before and after execution.
Moreover, since a function can allocate new memory as part of execution and return it to the calling scope, we must track that too.

### Consider Only Externally-Visible Effects Of Function Evaluation

We wish to treat a given `function` as a black box -- we care about _what_ a `function` does, not _how_ it does it -- so we consider only the externally-visible results of executing it.
There are two ways in which changes can be made externally visible.

_**Return Value**_

(This point hardly requires explanation, but for the sake of completeness we do so anyway.)

The most obvious way in which a result can be made visible outside of a `function` is via its return value.
For example, letting `bar(x) = sin(x)`, consider the function
```julia
function foo(x)
    y = bar(x)
    z = bar(y)
    return z
end
```
The communication between the two invocations of `bar` happen via the value it `return`s.


_**Modification of arguments**_

In contrast to the above, changes made by one `function` can be made available to another implicitly if it modifies the values of its arguments, even if it doesn't return anything.
For example, consider:
```julia
function bar(x::Vector{Float64})
    x .*= 2
    return nothing
end

function foo(x::Vector{Float64})
    bar(x)
    bar(x)
    return x
end
```
The second call to `bar` in `foo` sees the changes made to `x` by the first call to `bar`, despite not being explicitly returned.

_**No Global Mutable State**_

`function`s can in principle also communicate via `global` mutable state.
We make the decision to _not_ support this.

For example, we assume `function`s of the following form cannot be encountered:
```julia
const a = randn(10)

function bar(x)
    a .+= x
    return nothing
end

function foo(x)
    bar(x)
    return a
end
```
In this example, `a` is modified by `bar`, the effect of which is visible to `foo`.

For a variety of reasons this is very awkward to handle well.
Since it's largely considered poor practice anyway, we explicitly outlaw this mode of communication between `function`s.
See [Why Support Closures But Not Mutable Globals](@ref) for more info.

Note that this does not preclude the use of closed-over values or callable `struct`s.
For example, something like
```julia
function foo(x)
    function bar(y)
        x .+= y
        return nothing
    end
    return bar(x)
end
```
is perfectly fine.

### The Model

It is helpful to have a concrete example which uses both of the permissible methods to make results externally visible.
To this end, consider the following `function`:
```julia
function f(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}, s::Ref{Vector{Float64}})
    z .*= y .* x
    s[] = 2z
    return sum(z)
end
```
We draw your attention to a variety of features of this `function`:
1. `z` is mutated,
2. `s` is mutated to reference freshly allocated memory,
3. the value previously pointed to by `s` is unmodified, and
4. we allocate a new value and return it (albeit, it is probably allocated on the stack).

The model we adopt for any Julia `function` `f` is a function ``f : \mathcal{X} \to \mathcal{X} \times \mathcal{A}`` where ``\mathcal{X}`` is the real finite Hilbert space associated to the arguments to `f` prior to execution, and ``\mathcal{A}`` is the real finite Hilbert space associated to any newly allocated data during execution which is externally visible after execution -- any newly allocated data which is not made visible is of no concern.

In this example, ``\mathcal{X} = \RR^D \times \RR^D \times \RR^D \times \RR^S`` where ``D`` is the length of `x` / `y` / `z`, and ``S`` the length of `s[]` prior to running `f`.
``\mathcal{A} = \RR^D \times \RR``, where the ``\RR^D`` component corresponds to the freshly allocated memory that `s` references, and ``\RR`` to the return value.
Observe that we model `Float64`s as elements of ``\RR``, `Vector{Float64}`s as elements of ``\RR^D`` (for some value of ``D``), and `Ref`s with whatever the model for their contents is.
The keen-eyed reader will note that these choices abstract away several details which could conceivably be included in the model.
In particular, `Vector{Float64}` is implemented via a memory buffer, a pointer to the start of this buffer, and an integer which indicates the length of this buffer -- none of these details are exposed in the model.

In this example, some of the memory allocated during execution is made externally visible by modifying one of the arguments, not just via the return value.

The argument to ``f`` is the arguments to `f` _before_ execution, and the output is the 2-tuple comprising the same arguments _after_ execution and the values associated to any newly allocated / created data.
Crucially, observe that we distinguish between the state of the arguments before and after execution.

For our example, the exact form of ``f`` is
```math
f((x, y, z, s)) = ((x, y, x \odot y, s), (2 x \odot y, \sum_{d=1}^D x \odot y))
```
Observe that ``f`` behaves a little like a transition operator, in the that the first element of the tuple returned is the updated state of the arguments.

This model is good enough for the vast majority of functions.
Unfortunately it isn't sufficient to describe a `function` when arguments alias each other (e.g. consider the way in which this particular model is wrong if `y` aliases `z`).
Fortunately this is only a problem in a small fraction of all cases of aliasing, so we defer discussion of this until later on.

Consider now how this approach can be used to model several additional Julia functions, and to obtain their derivatives and adjoints.

_**`sin(x::Float64)`**_

``\mathcal{X} = \RR``, ``\mathcal{A} = \RR``, ``f(x) = (x, \sin(x))``.

Thus the derivative is ``D f [x] (\dot{x}) = (\dot{x}, \cos(x) \dot{x})``, and its adjoint is ``D f [x]^\ast (\bar{y}) = \bar{y}_x + \bar{y}_a \cos(x)``, where ``\bar{y} = (\bar{y}_x, \bar{y}_a)``.

Observe that this result is slightly different to the last example we saw involving `sin`.

_**AD With Mutable Data**_

Consider again
```julia
function f!(x::Vector{Float64})
    x .*= x
    return sum(x)
end
```
Our framework is able to accomodate this function, and has essentially the same solution as the last time we saw this example:
```math
f(x) = (x \odot x, \sum_{n=1}^N x_n^2)
```

_**Non-Mutating Functions**_

A very interesting class of functions are those which do not modify their arguments.
These are interesting because they are common, and are all that many AD frameworks like ChainRules.jl / Zygote.jl support -- by considering this class of functions, we highlight some key similarities between these distinct rule systems.

As always we can model these kinds of `function`s with a function ``f : \mathcal{X} \to \mathcal{X} \times \mathcal{A}``, but we additionally have that ``f`` must have the form
```math
f(x) = (x, \varphi(x))
```
for some function ``\varphi : \mathcal{X} \to \mathcal{A}``.
The derivative is
```math
D f [x] (\dot{x}) = (\dot{x}, D \varphi [x](\dot{x})).
```
Consider the usual inner product to derive the adjoint:
```math
\begin{align}
    \langle \bar{y}, D f [x] (\dot{x}) \rangle &= \langle (\bar{y}_1, \bar{y}_2), (\dot{x}, D \varphi [x](\dot{x})) \rangle \nonumber \\
        &= \langle \bar{y}_1, \dot{x} \rangle + \langle \bar{y}_2, D \varphi [x](\dot{x}) \rangle \nonumber \\
        &= \langle \bar{y}_1, \dot{x} \rangle + \langle D \varphi [x]^\ast (\bar{y}_2), \dot{x} \rangle \nonumber \quad \text{(by definition of the adjoint)} \\
        &= \langle \bar{y}_1 + D \varphi [x]^\ast (\bar{y}_2), \dot{x} \rangle. \nonumber
\end{align}
```
So the adjoint of the derivative is
```math
D f [x]^\ast (\bar{y}) =  \bar{y}_1 + D \varphi [x]^\ast (\bar{y}_2).
```
We see the correct thing to do is to increment the gradient of the output -- ``\bar{y}_1`` -- by the result of applying the adjoint of the derivative of ``\varphi`` to ``\bar{y}_2``.
In a `ChainRules.rrule` the ``\bar{y}_1`` term is always zero, but the ``D \varphi [x]^\ast (\bar{y}_2)`` term is essentially the same.



# The Rule Interface (Round 1)

Having explained in principle what it is that a rule must do, we now take a first look at the interface we use to achieve this.
A rule for a `function` `foo` with signature
```julia
Tuple{typeof(foo), Float64} -> Float64
```
must have signature
```julia
Tuple{Trule, CoDual{typeof(foo), NoFData}, CoDual{Float64, NoFData}} ->
    Tuple{CoDual{Float64, NoFData}, Trvs_pass}
```
For example, if we call `foo(5.0)`, its rules would be called as `rule(CoDual(foo, NoFData()), CoDual(5.0, NoFData()))`.
The precise definition and role of `NoFData` will be explained shortly, but the general scheme is that to a rule for `foo` you must pass `foo` itself, its arguments, and some additional data for book-keeping.
`foo` and each of its arguments are paired with this additional book-keeping data via the `CoDual` type.

The rule returns another `CoDual` (it propagates book-keeping information forwards), along with a `function` which runs the reverse pass.

In a little more depth:


_**Notation: primal**_

Throughout the rest of this document, we will refer to the `function` being differentiated as the "primal" computation, and its arguments as the "primal" arguments.

### Forwards Pass

_**Inputs**_

Each piece of each input to the primal is paired with shadow data, if it has a fixed address.
For example, a `Vector{Float64}` argument is paired with another `Vector{Float64}`.
The adjoint of `f` is accumulated into this shadow vector on the reverse pass.
However, a `Float64` argument gets paired with `NoFData()`, since it is a bits type and therefore has no fixed address.

_**Outputs**_

A rule must return a `Tuple` of two things.
The first thing must be a `CoDual` containing the output of the primal computation and its shadow memory (if it has any).
The second must be a function which runs the reverse pass of AD -- this will usually be a closure of some kind.

_**Functionality**_

A rule must
1. ensure that the state of the primal components of all inputs / the output are as they would have been had the primal computation been run (up to differences due to finite precision arithmetic),
2. propagate / construct the shadow memory associated to the output (initialised to zero), and
3. construct the function to run the reverse pass -- typically this will involve storing some quantities computed during the forwards pass.

### Reverse Pass

The second element of the output of a rule is a function which runs the reverse pass.

_**Inputs**_

The "rdata" associated to the output of the primal.

_**Outputs**_

The "rdata" associated to the inputs of the primal.

_**Functionality**_

1. undo changes made to primal state on the forwards pass.
2. apply adjoint of derivative of primal operation, putting the results in the correct place.

This description should leave you with (at least) a couple of questions.
What is "rdata", and what is "the correct place" to put the results of applying the adjoint of the derivative?
In order to address these, we need to discuss the types that Mooncake.jl uses to represent the results of AD, and to propagate results backwards on the reverse pass.





# Representing Gradients

We refer to both inputs and outputs of derivatives ``D f [x] : \mathcal{X} \to \mathcal{Y}`` as _tangents_, e.g. ``\dot{x}`` or ``\dot{y}``.
Conversely, we refer to both inputs and outputs to the adjoint of this derivative ``D f [x]^\ast : \mathcal{Y} \to \mathcal{X}`` as _gradients_, e.g. ``\bar{y}`` and ``\bar{x}``.

Note, however, that the sets involved are the same whether dealing with a derivative or its adjoint.
Consequently, we use the same type to represent both.

_**Representing Gradients**_

This package assigns to each type in Julia a unique `tangent_type`, the purpose of which is to contain the gradients computed during reverse mode AD.
The extended docstring for [`tangent_type`](@ref) provides the best introduction to the types which are used to represent tangents / gradients.

```@docs; canonical=false
Mooncake.tangent_type(P)
```



_**FData and RData**_

While tangents are the things used to represent gradients and are what high-level interfaces will return, they are not what gets propagated forwards and backwards by rules during AD.

Rather, during AD, Mooncake.jl makes a fundamental distinction between data which is identified by its address in memory (`Array`s, `mutable struct`s, etc), and data which is identified by its value (is-bits types such as `Float64`, `Int`, and `struct`s thereof).
In particular, memory which is identified by its address gets assigned a unique location in memory in which its gradient lives (that this "unique gradient address" system is essential will become apparent when we discuss aliasing later on).
Conversely, the gradient w.r.t. a value type resides in another value type.

The following docstring provides the best in-depth explanation.

```@docs; canonical=false
Mooncake.fdata_type(T)
```

_**More Info**_

See [Tangents](@ref) for complete information on what you must do if you wish to implement your own tangent type.
(In the vast majority of cases this is unnecessary).

_**CoDuals**_

CoDuals are simply used to bundle together a primal and an associated fdata, depending upon context.
Occassionally, they are used to pair together a primal and a tangent.

_**A quick aside: Non-Differentiable Data**_

In the introduction to algorithmic differentiation, we assumed that the domain / range of function are the same as that of its derivative.
Unfortunately, this story is only partly true.
Matters are complicated by the fact that not all data types in Julia can reasonably be thought of as forming a Hilbert space.
e.g. the `String` type.

Consequently we introduce the special type `NoTangent`, instances of which can be thought of as representing the set containing only a ``0`` tangent.
Morally speaking, for any non-differentiable data `x`, `x + NoTangent() == x`.

Other than non-differentiable data, the model of data in Julia as living in a real-valued finite dimensional Hilbert space is quite reasonable.
Therefore, we hope readers will forgive us for largely ignoring the distinction between the domain and range of a function and that of its derivative in mathematical discussions, while simultaneously drawing a distinction when discussing code.

TODO: update this to cast e.g. each possible `String` as its own vector space containing only the `0` element.
This works, even if it seems a little contrived.

# The Rule Interface (Round 2)

Now that you've seen what data structures are used to represent gradients, we can describe in more depth the detail of how fdata and rdata are used to propagate gradients backwards on the reverse pass.

```@meta
DocTestSetup = quote
    using Mooncake
    using Mooncake: CoDual, NoFData, NoRData
    import Mooncake: rrule!!
end
```

Consider the `function`
```jldoctest foo-doctest
julia> foo(x::Tuple{Float64, Vector{Float64}}) = x[1] + sum(x[2])
foo (generic function with 1 method)
```
The fdata for `x` is a `Tuple{NoFData, Vector{Float64}}`, and its rdata is a `Tuple{Float64, NoRData}`.
The function returns a `Float64`, which has no fdata, and whose rdata is `Float64`.
So on the forwards pass there is really nothing that needs to happen with the fdata for `x`.

Under the framework introduced above, the model for this `function` is
```math
f(x) = (x, x_1 + \sum_{n=1}^N (x_2)_n)
```
where the vector in the second element of `x` is of length ``N``.
Now, following our usual steps, the derivative is
```math
D f [x](\dot{x}) = (\dot{x}, \dot{x}_1 + \sum_{n=1}^N (\dot{x}_2)_n)
```
A gradient for this is a tuple ``(\bar{y}_x, \bar{y}_a)`` where ``\bar{y}_a \in \RR`` and ``\bar{y}_x \in \RR \times \RR^N``.
A quick derivation will show that the adjoint is
```math
D f [x]^\ast(\bar{y}) = ((\bar{y}_x)_1 + \bar{y}_a, (\bar{y}_x)_2 + \bar{y}_a \mathbf{1})
```
where ``\mathbf{1}`` is the vector of length ``N`` in which each element is equal to ``1``.
(Observe that this agrees with the result we derived earlier for functions which don't mutate their arguments).

Now that we know what the adjoint is, we'll write down the `rrule!!`, and then explain what is going on in terms of the adjoint.
This hand-written implementation is to aid your understanding -- Mooncake.jl should be relied upon to generate this code automatically in practice.

```jldoctest foo-doctest
julia> function rrule!!(::CoDual{typeof(foo)}, x::CoDual{Tuple{Float64, Vector{Float64}}})
           dx_fdata = x.dx
           function dfoo_adjoint(dy::Float64)
               dx_fdata[2] .+= dy
               dx_1_rdata = dy
               dx_rdata = (dx_1_rdata, NoRData())
               return NoRData(), dx_rdata
           end
           x_p = x.x
           return CoDual(x_p[1] + sum(x_p[2]), NoFData()), dfoo_adjoint
       end;

```
where `dy` is the rdata for the output to `foo`.
The `rrule!!` can be called with the appropriate `CoDual`s:
```jldoctest foo-doctest
julia> out, pb!! = rrule!!(CoDual(foo, NoFData()), CoDual((5.0, [1.0, 2.0]), (NoFData(), [0.0, 0.0])))
(CoDual{Float64, NoFData}(8.0, NoFData()), var"#dfoo_adjoint#1"{Tuple{NoFData, Vector{Float64}}}((NoFData(), [0.0, 0.0])))
```
and the pullback with appropriate rdata:
```jldoctest foo-doctest
julia> pb!!(1.0)
(NoRData(), (1.0, NoRData()))
```

```@meta
DocTestSetup = nothing
```

Observe that the forwards pass:
1. computes the result of the initial function, and
2. pulls out the fdata for the `Vector{Float64}` component of the argument.

As promised, the forwards pass really has nothing to do with the adjoint.
It's just book-keeping and running the primal computation.

The reverse pass:
1. increments each element of `dx_fdata[2]` by `dy` -- this corresponds to ``(\bar{y}_x)_2 + \bar{y}_a \mathbf{1}`` in the adjoint,
2. sets `dx_1_rdata` to `dy` -- this corresponds ``(\bar{y}_x)_1 + \bar{y}_a`` subject to the constraint that ``(\bar{y}_x)_1 = 0``,
3. constructs the rdata for `x` -- this is essentially just book-keeping.

Each of these items serve to demonstrate more general points.
The first that, upon entry into the reverse pass, all fdata values correspond to gradients for the arguments / output of `f` "upon exit" (for the components of these which are identified by their address), and once the reverse-pass finishes running, they must contain the gradients w.r.t. the arguments of `f` "upon entry".

The second that we always assume that the components of ``\bar{y}_x`` which are identified by their value have zero-rdata.

The third is that the components of the arguments of `f` which are identified by their value must have rdata passed back explicitly by a rule, while the components of the arguments to `f` which are identified by their address get their gradients propagated back implicitly (i.e. via the in-place modification of fdata).

_**Reminder**_: the first element of the tuple returned by `dfoo_adjoint` is the rdata associated to `foo` itself, hence it is `NoRData`.

# Testing

Mooncake.jl has an almost entirely automated system for testing rules -- [`Mooncake.TestUtils.test_rule`](@ref).
You should absolutely make use of these when writing rules. This function can test both forward and backward modes,
or you can specify a single mode to test. `test_rule` will also optionally test the performance of your rules.
See the docstring for `test_rule` for more information.

# Summary

In this section we have covered the rule system.
Every callable object / function in the Julia language is differentiated using rules with this interface, whether they be hand-written `rrule!!`s, or rules derived by Mooncake.jl.

At this point you should be equipped with enough information to understand what a rule in Mooncake.jl does, and how you can write your own ones.
Later sections will explain how Mooncake.jl goes about deriving rules itself in a recursive manner, and introduce you to some of the internals.








# Asides

### Why Uniqueness of Type For Tangents / FData / RData?

Why does Mooncake.jl insist that each primal type `P` be paired with a single tangent type `T`, as opposed to being more permissive.
There are a few notable reasons:
1. To provide a precise interface. Rules pass fdata around on the forwards pass and rdata on the reverse pass -- being able to make strong assumptions about the type of the fdata / rdata given the primal type makes implementing rules much easier in practice.
1. Conditional type stability. We wish to have a high degree of confidence that if the primal code is type-stable, then the AD code will also be. It is straightforward to construct type stable primal codes which have type-unstable forwards and reverse passes if you permit there to be more than one fdata / rdata type for a given primal. So while uniqueness is certainly not sufficient on its own to guarantee conditional type stability, it is probably necessary in general.
1. Test-case generation and coverage. There being a unique tangent / fdata / rdata type for each primal makes being confident that a given rule is being tested thoroughly much easier. For a given primal, rather than there being many possible input / output types to consider, there is just one.

This topic, in particular what goes wrong with permissive tangent type systems like those employed by ChainRules, deserves a more thorough treatment -- hopefully someone will write something more expansive on this topic at some point.


### Why Support Closures But Not Mutable Globals

First consider why closures are straightforward to support.
Look at the type of the closure produced by `foo`:
```jldoctest
function foo(x)
    function bar(y)
        x .+= y
        return nothing
    end
    return bar
end
bar = foo(randn(5))
typeof(bar)

# output
var"#bar#1"{Vector{Float64}}
```
Observe that the `Vector{Float64}` that we passed to `foo`, and closed over in `bar`, is present in the type.
This alludes to the fact that closures are basically just callable `struct`s whose fields are the closed-over variables.
Since the function itself is an argument to its rule, everything enters the rule for `bar` via its arguments, and the rule system developed in this document applies straightforwardly.

On the other hand, globals do not appear in the functions that they are a part of.
For example,
```jldoctest
const a = randn(10)

function g(x)
    a .+= x
    return nothing
end

typeof(g)

# output
typeof(g) (singleton type of function g, subtype of Function)
```
Neither the value nor type of `a` are present in `g`.
Since `a` doesn't enter `g` via its arguments, it is unclear how it should be handled in general.
