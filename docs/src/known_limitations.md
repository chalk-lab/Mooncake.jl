# Known Limitations

Mooncake.jl has a number of known qualitative limitations, which we document here.

## Coverage of Julia Syntax and Standard Library

While `Mooncake.jl` should now work on a very large subset of the language, there remain things that you should expect not to work. A non-exhaustive list of things to bear in mind includes:
1. It is always necessary to produce hand-written rules for `ccall`s (and, more generally, foreigncall nodes). We have rules for many `ccall`s, but not all. If you encounter a foreigncall without a hand-written rule, you should get an informative error message which tells you what is going on and how to deal with it.
1. Builtins which require rules. The vast majority of them have rules now, but some don't. You should get a sensible error if you encounter a primitive without a rule.
1. Element access and stores through `reinterpret` between `T` and `Complex{T}` (`T <: IEEEFloat`) are supported for ordinary Array/Memory storage and views, preserving aliases with the parent. Other reinterpretations and pointer re-typing are restricted, including through `Ptr{Cvoid}`, to the same primal element type when derivative storage is required; write a rule for the enclosing function otherwise. Reads through pointers narrowed to non-differentiable elements remain admitted, but stores and nonempty copies through such pointers over differentiated storage raise `ArgumentError`.
1. Nonzero pointer byte shifts raise `ArgumentError` when the tangent storage stride differs from the primal element stride, including through erased or narrowed pointers. Use typed element indexing where supported, or write a rule for the enclosing function. Zero shifts remain supported. Reverse pointer `unsafe_copyto!` also refuses differing strides; forward copies and array/MemoryRef copies use their own element strides. Forward pointers from derivative-carrying `Memory` retain their source element type: re-typing to a different differentiable element type is refused. Numeric Memory permits same-type derivative access at width one; widths greater than one refuse derivative loads and stores because lanes are interleaved. Address-only casts and shifts over storage with no tangent are supported except for forward transport of a struct pointer through a non-inlined call, which can raise `TypeError`; Symbol-selected transport can also terminate Julia before returning a derivative. Narrowed stores, nonempty copies, and `unsafe_wrap` over differentiated storage raise `ArgumentError`.
1. Anything involving tasks / threading -- we have no thread safety guarantees and, at the time of writing, I'm not entirely sure what error you will find if you attempt to AD through code which uses Julia's task / thread system. The same applies to distributed computing. These limitations ought to be possible to resolve.
1. Prepared caches refuse `Dict`s whose values can carry derivatives, including those nested in `f`, struct fields, tuples, named tuples, array element types or `Ref`s; pass values in a `Vector`/`NamedTuple`, build the `Dict` inside the function, or use rule-taking forms (abstract or `Any`-typed storage is inspected by runtime type at preparation only, so preserve those types on reuse).
1. The rule-taking forms support `Dict` and `IdDict` tangent conversion by key, including nested native tangent leaves; prepared caches support this conversion for `IdDict`. Direction keys must match the call-time primal; table capacity and insertion order do not affect the tangent. Native `Dict` tangents are keyed dictionaries. Dictionary keys carry no derivatives. Insertion and lookup accept differentiable keys with zero key derivatives. Exposing a stored differentiable key (pair/key iteration, `getkey`, or `pop!` on a set on Julia 1.11+) is unsupported and raises an error. Value iteration does not expose keys. `replace`/`replace!` on a `Dict` or `Set`, and random sampling from those containers, remain unsupported. Sampling Dict key/value iterators also reaches unsupported table internals on Julia 1.11+ (Julia 1.10 can sample keys, but Base has no value-iterator sampler); direct differentiation through `Dict` fields or unsupported hash-table internals raises an error. Friendly forward directions for other `AbstractDict` primals are refused; pass a native tangent through a cache prepared with `Config(; friendly_tangents=false)`.
1. An `IdDict` argument's key objects must be the prepare-time ones; prepare a new cache after replacing them.
1. HVPs through a captured or constructed `FunctionWrapper` are unsupported. On Julia 1.10/1.11, captured wrappers may fail only after a long compilation or may not terminate in practice.

BLAS/LAPACK matrix rules require a column stride at least the row count for multi-row matrices, and a positive column stride for one-row matrices with multiple columns; they preserve `gemv!`'s own stride normalisation, empty-output calls, and `lacpy!` single-column and identity copies.
LAPACK operands must also have unit first stride where the routine requires it, including matrix and vector right-hand sides of `trtrs!` when they have more than one row; offset and padded storage satisfying these requirements remains supported.

## Resizing shared arrays on Julia 1.10

On Julia 1.10, resizing a vector whose tangent storage is shared with another array
header throws an `ArgumentError`. Copy the array before resizing it. Resizing can
otherwise detach tangent storage and silently lose gradients, including on prepared
cache reuse. Non-resizing operations on aliases remain supported.

This applies to `push!`, `append!`, `resize!`, `pushfirst!`, `prepend!`, `insert!`,
`pop!`, `popfirst!`, `deleteat!`, `splice!`, `empty!`, and reallocating `sizehint!`
calls. The checks cover both differentiation modes at the legacy `_growbeg!`,
`_growend!`, `_growat!`, `_deletebeg!`, `_deleteend!`, `_deleteat!`, and `sizehint!`
rules. Arrays on Julia 1.11 and later are unaffected.

## Overlapping BLAS and LAPACK Operands

Mutating BLAS matrix products and triangular operations, and LAPACK solves, require
an output that does not overlap a separate input operand. Mooncake checks this in
both modes and throws `ArgumentError` before modifying the operands. Copy the input
when calling these operations with overlapping views or with the same array in
both positions. The usual in-place right-hand side of a triangular operation is
supported when it is disjoint from the coefficient matrix.

`axpby!(a, x, b, x)` and `LAPACK.lacpy!(A, A, uplo)` support identical source and
destination objects. Real `axpy!` also supports identical source and destination
memory walks. Other overlapping source/destination patterns are refused; complex
`axpy!` refuses even identical walks because its BLAS kernel can overwrite a
component before reading it. These checks do not change the general boundary for
aliasing established only through raw pointers.

## Numerical Workarounds and Singularities

Numerical workarounds can preserve function values while producing incorrect derivatives under automatic differentiation.
For example, `LogExpFunctions.xlogy` implements:

```julia
function xlogy(x::Number, y::Number)
    result = x * log(y)
    return iszero(x) && !isnan(y) ? zero(result) : result
end
```

At `(0, 2)`, differentiating the selected branch gives zero, but the partial derivative with respect to `x` is `log(2)`.
Special functions with parameter-specific formulas or termination conditions can present similar problems.
These limitations are not specific to Mooncake.
Custom differentiation rules may be needed; see [Defining Rules](@ref).

## `try`/`catch`/`finally` Blocks

Mooncake.jl does not support differentiating through `try`/`catch` or `try`/`finally` blocks
in **reverse mode**. Attempting to do so will produce an `UnhandledLanguageFeatureException`
with a message explaining the cause. Forward mode supports `try`/`catch`, `try`/`finally` and
`rethrow()`, except that binding the exception (`catch e`) is refused.

**The fix** is to replace `try`/`catch` blocks with explicit conditional checks where possible.
For example:

```julia
# Does not work with Mooncake
function safe_log(x)
    try
        return log(x)
    catch e
        return -Inf
    end
end

# Works with Mooncake
function safe_log(x)
    x <= 0 && return -Inf
    return log(x)
end
```

If the `try`/`catch` block cannot be avoided (e.g. it lives inside a third-party library), the
alternative is to provide a custom `rrule!!` for the function that contains the `try`/`catch`
block, so that Mooncake never needs to differentiate through the block itself.
See [Defining Rules](@ref) for guidance on writing custom rules.

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

The takeaway: do not attempt to differentiate functions which modify global state.
When reading a non-`const` global through a `GlobalRef` in the lowered code, both modes require its value to remain `===` or `==` to its value when the rule was built; otherwise, the call raises an error (explicit `getglobal` calls bypass this check).
Rebinding to an equal value is accepted, but the same object must not also be an argument.

Forward mode reads current values and replaced storage inside a global between calls (`R[] = [7.0]`), including changes of the slot's value type and scalars replaced inside an immutable aggregate (`R[] = (7.0, 7.0)`). Reverse mode supports changes that remain compatible with its cached tangent representation; replacement with another type can raise an exception. A global array grown beyond its cached derivative storage, whether a numeric array's partials block or an element-wise array's tangent array (`Vector{Vector{Float64}}`, `Vector{Any}`), is refused in forward mode with an `ArgumentError`; this also applies when the array is returned as a Jacobian output. Shrinking is supported, except that on Julia 1.11 `reshape` of a vector whose capacity exceeds its length (any grown or shrunk vector, global or not) takes a `jl_genericmemory_slice` foreign call that has no rule and raises `MissingForeigncallRuleError`. The CUDA extension's block readers do not perform the growth check. Storage replaced inside an immutable aggregate held by a global (`R[] = ([7.0],)`) is read correctly in forward mode.

### Passing a global as an argument

Reading a global is fine only while that object is not *also* an argument. The global's derivative
storage is built once when the rule is built, so it shares nothing with the argument's, and the
contribution through the global is written somewhere the caller never sees:

```julia
const G = [1.0, 2.0]
f(x) = sum(x .* G)
value_and_gradient!!(build_rrule(f, G), f, G)   # gradient [1.0, 2.0]; the truth is [2.0, 4.0]
```

The value returned is correct, which makes this easy to miss. Both modes refuse such a call with an
`ArgumentError` rather than returning the wrong derivative. Pass a copy, or read the value through
an argument instead of a global.

The check is by object identity, so it covers what can actually be shared: arrays, mutable structs,
and aggregates containing them. A differentiable *immutable* constant that is also passed as an
argument — `const C = ("a", 1.0)` — is not caught, and neither is a global read behind a call whose
own arguments do not include the aliased object (`f(x) = x[1] * get_G()[1]`), since the guard
compares a rule's constants against that rule's own arguments.

Aliasing between *arguments* is a different matter and is supported: two arguments over one array
share derivative storage, so both positions report the one accumulated gradient.

### Conflicting forward tangents for one shared storage

Positions that share one storage share one tangent, so supplying different tangents for them is
ill-posed: only one direction can be carried. `value_and_derivative!!` refuses it with an
`ArgumentError` — through a prepared cache under either `friendly_tangents` setting, and against a
bare rule. Pass the same tangent object at every position the shared storage occupies.

The refusal is by object identity, so it does not see sharing that identity cannot express: a
`reshape` or a `view` of an argument, or an `Array` beside its backing `Memory`. Those still answer
silently, carrying whichever tangent the lift reached first. `f(a, b) = sum(a) + sum(b)` run at
`(a, da)` and `(reshape(a, 3, 1), db)` reports `sum(da)` twice rather than `sum(da) + sum(db)`.

### Forward gradients of arguments that share one storage

Forward mode gives each argument its own tangent storage, so two arguments over one array are two
independent directions rather than one shared derivative, and `value_and_gradient!!` — whose
gradient is assembled from one standard-basis dimension range per input — cannot represent that. It
refuses with an `ArgumentError` rather than returning a gradient scaled by the number of positions
the storage occupies. Repeated leaves *within* one argument are supported and agree with reverse
mode: `f(t) = sum(t[1]) + sum(t[2])` at `t = (a, a)` gives `([2, 2, 2], [2, 2, 2])` in both modes,
because one `lift` threads a single aliasing cache through the argument's leaves. Sharing that is
not object identity — a `reshape`, a `view`, an `Array` beside its backing `Memory` — is refused
wherever it appears, including within one argument.

Reverse mode has no such restriction: aliased arguments share one cotangent buffer, so every
position reports the one accumulated gradient. The forward Jacobian is also unaffected — it
differentiates a single argument with `f` held fixed, so one dimension range covers every position, and
an aliased capture (`f = x -> x .* sum(q)` called at `x === q`) gives the full Jacobian.

## Restoration of arguments a rule mutated

Forward input snapshots retain the original mutable objects as well as their contents.
If `f` rebinds a field of a mutable argument (`s.v = 2 .* s.v` rather than `s.v .= 2 .* s.v`),
restoration puts the original object back in the field and restores its contents.

The forward entry points (`value_and_derivative!!`, `value_and_jacobian!!` and the forward
`value_and_gradient!!`) snapshot the arguments and restore them afterwards, including when the rule
raises. RNG state is deliberately excluded from restoration: random draws leave the RNG advanced,
including on failure. Random operations execute afresh in each gradient or Jacobian chunk.
Reverse mode restores mutations on the pullback instead, so an exception during the forward
sweep leaves them in place: the
arguments of a `value_and_pullback!!` or `value_and_gradient!!` call that threw are whatever the
rule left behind. Copy them if you intend to retry.

## Mutable aliases involving `NoTangent` parents or globals

Mooncake may silently return incorrect derivatives when the same mutable storage is differentiated directly and also reachable through a `NoTangent` parent. Reverse and `frule!!`-based forward modes are affected. See [issue #1295](https://github.com/chalk-lab/Mooncake.jl/issues/1295). The same aliasing through a global is now refused rather than silent.

A `NoTangent` parent:

```jldoctest opaque-alias
julia> struct Box
           x::Vector{Float64}
       end

julia> Mooncake.tangent_type(::Type{Box}) = NoTangent;

julia> function f(state)
           box, x = state
           box.x[1] *= 2
           return x[1]
       end;

julia> x = [3.0]; state = (Box(x), x);

julia> rule = Mooncake.build_rrule(f, state);

julia> Mooncake.value_and_gradient!!(rule, f, state)
(6.0, (NoTangent(), (NoTangent(), [1.0])))
```

`state` exposes one vector as `x` and `box.x`. Mooncake differentiates `x`, but reading through the `NoTangent` `Box` creates separate derivative storage. Mutation through `box.x` updates only that storage, so Mooncake returns `[1.0]`. This program should be rejected.

A global, which is refused:

```jldoctest global-alias
julia> const X = [3.0];

julia> function g(x)
           X[1] *= 2
           return x[1]
       end;

julia> rule = Mooncake.build_rrule(g, X);

julia> Mooncake.value_and_gradient!!(rule, g, X)
ERROR: ArgumentError: An argument is the same object as a constant or global read inside the function being differentiated (a Vector{Float64}). Their derivative storage is separate — the constant's is created once when the rule is built — so the contribution through the constant would be silently dropped and the derivative returned would be wrong. Pass a copy of the argument, or read the value through an argument instead of a global.
[...]
```

`X` and `x` are the same vector, and their derivative storage is separate, so the contribution
through `X` would be dropped. Mooncake used to return `[1.0]` silently; it now raises. See
[Passing a global as an argument](@ref) for the guard.

Nesting escapes either way — matching is by identity at the root: `const T = (A,)` read at
`x === A`, and `f(t) = sum(t[1] .* A)` called at `t = (A,)`, are both silently wrong.

## Passing Differentiable Data as a Type

Credit goes to Guillaume Dalle for noticing this limitation.

This is an example of a known silent correctness issue.
```jldoctest
julia> struct T{x} end

julia> @noinline getparam(::T{x}) where {x} = x
getparam (generic function with 1 method)

julia> mysquare(x) = getparam(T{x}())^2
mysquare (generic function with 1 method)

julia> cache = Mooncake.prepare_derivative_cache(mysquare, 3.0);

julia> Mooncake.value_and_derivative!!(cache, (mysquare, Mooncake.NoTangent()), (3.0, 1.0))
(9.0, 0.0)
```
As you can see, the tangent is `0.0` rather than `6.0`.

However, we view this as a pathological use of Julia's language features, and believe it is unlikely to cause trouble in practice.
If you encounter a practical situation in which it is very important that this example work correctly, please open an issue.

## Primitives and Overlays

[`@mooncake_overlay`](@ref Mooncake.@mooncake_overlay) and [`@is_primitive`](@ref Mooncake.@is_primitive) both let you change what Mooncake sees when it differentiates a function, but they hook in at different layers and do not compose.
`@mooncake_overlay` swaps in a replacement body Mooncake walks through; `@is_primitive` stops Mooncake walking and dispatches to a hand-written rule instead.
Marking a signature as a primitive shadows any overlay reachable through it — the rule fires, the overlay does not, and Mooncake infers return types from the original method, not from the overlay.

This causes two failure modes:

1. **An overlay that changes the return type breaks the rule's return-type contract.**
   Mooncake infers the return type from the original method, so a manual rule that returns a value of the overlaid type produces a `CoDual` that disagrees with that inferred type, triggering a runtime `TypeError`.
   When the original and overlaid methods return the same type, the runtime check is trivially satisfied and the bug surfaces only as an incorrect gradient.

2. **An overlay inside a primitive's body never runs.**
   The rule replaces the body and runs the primal through ordinary Julia dispatch, which ignores overlays.
   An overlay introduced to work around unsupported code reached from inside the primitive is silently inert — move it to a signature Mooncake still differentiates, or fold the workaround into the rule.

The example below covers both cases.
Case (1) overlays the primitive itself; case (2) overlays a helper called from the primitive's body.
Each identity check asks whether Mooncake's inferred return type for the caller is `A` (the original) rather than `B` (what the overlay would produce):

```jldoctest overlay-primitive-doctest
julia> struct A end

julia> struct B end

julia> direct_primitive(::A) = A()
direct_primitive (generic function with 1 method)

julia> Mooncake.@mooncake_overlay direct_primitive(::A) = B();

julia> Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(direct_primitive), A}

julia> helper(::A) = A()
helper (generic function with 1 method)

julia> Mooncake.@mooncake_overlay helper(::A) = B();

julia> indirect_primitive(x::A) = helper(x)
indirect_primitive (generic function with 1 method)

julia> Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(indirect_primitive), A}

julia> caller_direct(x::A) = direct_primitive(x)
caller_direct (generic function with 1 method)

julia> caller_indirect(x::A) = indirect_primitive(x)
caller_indirect (generic function with 1 method)

julia> interp = Mooncake.MooncakeInterpreter(Mooncake.DefaultCtx, Mooncake.ReverseMode);

julia> Base.code_ircode_by_type(Tuple{typeof(caller_direct), A}; interp)[1][2] === A
true

julia> Base.code_ircode_by_type(Tuple{typeof(caller_indirect), A}; interp)[1][2] === A
true
```

Mooncake infers `A` for both callers, even though the overlays would produce a `B` — so any rule written against the overlay's return type fails the runtime check.
Apply only one of `@mooncake_overlay` or `@is_primitive` to a given signature, and make sure no overlay you rely on sits behind a primitive's rule.

## Explicit `invoke` of a primitive

An explicit `invoke` encountered during differentiation bypasses Mooncake's protection against inlining primitives.
Inlining can silently discard the custom rule; surviving calls can use the rule for ordinary dispatch, changing the value and derivative ([#1300](https://github.com/chalk-lab/Mooncake.jl/issues/1300)).

```julia
f(x::Real) = sin(x)
f(x::Float64) = cos(x)
g(x::Float64) = invoke(f, Tuple{Real}, x)
```

With a rule for `f(::Float64)`, `g` may incorrectly use that rule although its primal computes `sin(x)`.

This can occur when differentiated code explicitly invokes a generic fallback despite a rule for the specialised call.
It can also arise indirectly: [`Base.Math.@horner`](https://github.com/JuliaLang/julia/blob/v1.12.7/base/math.jl#L176-L179) expands to `invoke(evalpoly, Tuple{Any, Tuple}, ...)`.
That becomes problematic if a matching `evalpoly` rule is added; the macro alone does not trigger the limitation.
Without a matching rule, Mooncake differentiates the invoked method normally.

Simply replacing `invoke` with ordinary dispatch can change the primal's behaviour.
Wrap the `invoke` in a function with its own rule.
Calls inside an existing primitive's body are unaffected.

## Differentiating CUDA Kernels

Mooncake.jl supports differentiation of CUDA kernels in general, provided a suitable rule exists. However, it does not support kernels that surface as foreign calls, such as those generated via KernelAbstractions.jl (see [issue #648](https://github.com/chalk-lab/Mooncake.jl/issues/648) and [issue #835](https://github.com/chalk-lab/Mooncake.jl/issues/835)). Support for these foreign-call kernels is outside the scope of the project and is considered a non-goal.

Users who need to differentiate through these code paths may do so by providing a custom rule, potentially generated with the assistance of another automatic differentiation tool (cf. [this comment](https://github.com/chalk-lab/Mooncake.jl/issues/648#issuecomment-3058010288)).

Not every array operation on a `CuArray` has a rule yet. `maximum`, `minimum`, `diff` and `sort` do
not, in either their plain or their `f`-mapped form, and the higher-order reductions carry a rule
only for the operators they were written for — `reduce` for `+` and `*`, `mapreduce` for `+`, and
`accumulate` for `+`. Anything outside those sets is registered as a primitive whose rule raises an
`ArgumentError` naming the operation, so you get a clear failure at the call rather than a wrong
derivative or an obscure error from inside a kernel. Reductions over an array whose element type is
non-differentiable are unaffected: those correctly give a zero derivative.

Forward mode over `NNlib.gather` on a GPU array refuses for a related reason: the traced kernel
launch takes the process down with no catchable exception, so the rule raises instead. The
limitation is specific to that combination. Reverse mode over the same signature works, and so
does forward mode on a CPU array.

Second-order AD (HVP / Hessian, via forward-over-reverse) is more restricted on CUDA: it works for array-level operations whose rules do not launch a custom per-element kernel (e.g. `sum(x)`, `dot`, matrix multiplication), but operations that map a Julia function over array elements inside a GPU kernel (broadcasting, `sum(f, x)`-style reductions) cannot yet be differentiated at second order. These raise a clear `ArgumentError` rather than silently returning wrong derivatives. Gradients and JVPs are unaffected.

## Differentiating SIMD Code

When the primal code admits SIMD (Single Instruction, Multiple Data) optimisations by the LLVM compiler, reverse-mode automatic differentiation in Mooncake can inhibit LLVM's ability to apply the same optimisations. This occurs because the transformations introduced during differentiation may obscure the structural patterns that LLVM relies on for vectorisation.

Consequently, code that performs well in its primal form may suffer substantial slowdowns after differentiation. In performance-critical regions, this can often be mitigated by providing hand-written derivative rules (`rrule!!`s, see [Defining Rules](@ref)). Such rules allow explicit control over the structure of the adjoint computation and can be used to preserve SIMD-friendly loop forms.

If you observe unexpected performance regressions in differentiated code that is known to vectorise effectively in its primal form, consider implementing custom rules for the relevant operations. For example, a customised performance-oriented rule is added for `kron` in Mooncake ([PR #886](https://github.com/chalk-lab/Mooncake.jl/pull/886)) to address the loss of SIMD vectorisation that arises when differentiating the SIMD-friendly primal implementation of `kron` in Julia's standard library ([LinearAlgebra implementation](https://github.com/JuliaLang/LinearAlgebra.jl/blob/b1f48a442ba5f41479d4fb6f3e931b1ac2f21059/src/dense.jl#L499-L531)).

## Differentiating Through Another Automatic Differentiation Library

Mooncake.jl is not meant to differentiate *through* another automatic differentiation library.
If the function you hand to Mooncake itself calls a second autograd tool — for example, by building and replaying a tape — you are asking Mooncake to differentiate that tool's internals rather than your underlying mathematical function.
This is outside Mooncake's scope and is not officially supported. See [issue #1209](https://github.com/chalk-lab/Mooncake.jl/issues/1209) for an example.

If part of your computation genuinely needs another AD tool, wrap that part in a custom rule, so Mooncake treats it as a primitive and never looks inside; see [Defining Rules](@ref).

## Circular References in Type Declarations

Mooncake.jl's default `tangent_type` implementation cannot support types which refer to themselves either directly or indirectly in their definition.
Below is an example in which a type refers to itself directly in its definition.

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
An example implementation of this is provided [here](developer_documentation/custom_tangent_type.md).

## Tangent Generation and Pointers

Reverse mode refuses raw pointer loads and stores that copy mutable
structs with inline differentiable fields, including `unsafe_load` /
`unsafe_store!` and atomic variants: cotangents of these copies cannot
be propagated back to their sources. Copies whose differentiable data
is held only in referenced children share those children's tangents
and remain supported. The guard uses primal field types conservatively;
a custom tangent does not override an inline primal float field.
Boxed reference slots and isbits loads are unaffected.
To recover an object from `pointer_from_objref`, use
`unsafe_pointer_to_objref`; casts through boxed or mutable pointer
types are supported. Non-differentiable casts retain pointer provenance;
recovery with a NULL tangent address is refused.
Casts to inline targets must match the payload element type, so
reading individual float fields of a multi-field object is refused.

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

Forward mode supports ordinary and atomic stores through `pointer(::Array{Any})`
at every chunk width: each boxed shadow slot receives the complete forward value.
Boxed stores and copies retain the storage owner and apply a GC write barrier to
both primal and shadow writes, including reverse-mode restoration. Stores through
hand-built `Ptr{Any}` values without a known boxed shadow and owner are refused
with an `ArgumentError` before mutation when the pointer carries tangent storage,
even for non-differentiable values. Forward stores with no tangent storage are allowed.
Address-only round trips from `Ptr{Any}` through
`Ptr{Cvoid}` or `Ptr{UInt8}` preserve unknown shadow provenance and are allowed.
Forward copies between boxed pointers require known boxed shadows. Copies through
pointers narrowed or erased from boxed slots are refused when they move data,
even with unknown shadows. Zero-byte copies and copies onto the same pointer
are allowed.
Boxed pointer loads remain unsupported in forward mode.

While pointers tend to be a low-level implementation detail in Julia code, you could in principle actually be interested in differentiating a function of a pointer.
In this case, you will not be able to use `Mooncake.value_and_gradient!!` as this requires the use of `zero_tangent`.
Instead, you will need to use lower-level (internal) functionality, such as `Mooncake.__value_and_gradient!!`, or use the rule interface directly.

Honestly, your best bet is just to avoid differentiating functions whose arguments are pointers if you can.

### Raw pointers into a nested array, at chunk width above one

Forward mode stores an array's `N` lane partials in one element-major block, so a single lane is a
strided view rather than a dense buffer. That is fine for a flat array — `pointer` and
`unsafe_copyto!` on a `Vector{Float64}` work at any chunk width — but an array *of arrays* has no
dense per-lane buffer for a raw pointer to address:

```julia
f(x, y, n) = (unsafe_copyto!(pointer(x), pointer(y), n); sum(sum, x))
x = [randn(3) for _ in 1:5]
y = [randn(4) for _ in 1:6]
# chunk width 1: fine. Above 1: ArgumentError naming the width.
```

The rule refuses rather than dropping the derivative. Differentiate that call at chunk width 1, or
use reverse mode, which is unaffected.

Wrapping a pointer-to-pointer buffer with `unsafe_wrap` preserves its shadow pointer storage at
chunk width one. Wider chunks raise `ArgumentError`: the wrapped array interleaves its shadow
pointers by element, whereas each raw pointer lane addresses a dense buffer. Supporting both
layouts requires a pointer representation that carries the tangent stride; use chunk width one
or reverse mode.

### Re-typing a pointer through `Ptr{Cvoid}`

A tangent pointer carries its element type, and that is what lets Mooncake check that a re-typing is
sound: re-typing `Ptr{Float32}` to `Ptr{Float64}` is refused, because a load or store through the
result would address eight bytes per element in a buffer laid out in four-byte ones.

Erasing the element type loses the information that check needs, because Mooncake gives an erased
pointer and a pointer with no tangent storage at all the same representation — both are
`Ptr{Nothing}`, and `fdata_type` pins a pointer field's fdata to `Ptr`.

The erasure itself is allowed. What is refused is re-typing to an element type the underlying
tangent storage cannot hold, and that is checked when the pointer is widened back:

```julia
f(b::Vector{Float32}, x) = x * unsafe_load(Ptr{Float64}(Ptr{Cvoid}(pointer(b))))   # ArgumentError
```

Erasing and re-typing back to the *same* element type is fine, which is the common direction:
`pointer(::Array)` passes through a `Ptr{Cvoid}` intermediate and re-types back to the element type
a foreigncall needs.

The `Ptr{Cvoid}` round trip is not itself the problem. Going straight from `Ptr{Float32}` to
`Ptr{Float64}` is refused identically — a `Float64` load would straddle two `Float32` tangent
elements either way. The round trip only matters in that it hides the source type until the
widening, which is why the error for that path suggests re-typing directly instead.

```@meta
DocTestSetup = nothing
```

## Reusing prepared caches

Reuse must preserve the alias relationships from preparation. Per-call aliasing comparisons
run by default; set `Config(check_aliasing=false)` to skip their fixed-position comparisons
and, for structured forward gradient seeds, array-leaf traversal at zero per-call cost.
With the opt-out, a call with changed aliasing can return a wrong gradient without an error:
for `f(a, b, x) = (a[1] *= x; sum(b) * x + x * x)`, preparing with
`f(v, v, 1.5)` and calling with `f(a, b, 1.5)`, where `v`, `a` and `b` are distinct
arrays each equal to `[1.0, 2.0]`,
can return `7.5` for the derivative with respect to `x`, whose correct value is `6.0`.
This fixed-position mismatch is refused by default. Aliasing in array elements is not
checked per call yet. Shape and type checks and preparation-time refusals remain on, including
`Dict` admission and aliased storage across arguments in forward gradients and HVP/Hessian.

`prepare_gradient_cache` and `prepare_pullback_cache` record the alias relationships of
mutable inputs and storage carried by immutable values, including `MemoryRef`, tuples,
named tuples, closures and immutable array wrappers. Reuse must preserve
both object identity relationships and shared backing storage for leaf types that can carry
derivatives, including empty containers that the function can fill. Distinct empty arrays
count as independent storage only when their owning backing `Memory` has length zero
(Julia 1.11+). Arrays emptied with `empty!` can retain
nonempty backing storage, whose aliasing must still be preserved. Julia 1.10 uses
`Base.dataids` to identify storage. On Julia 1.11 and later,
relative offsets within each shared backing `Memory` group must also match. With the default
aliasing checks, a mismatch raises
`Mooncake.PreparedCacheError` before resetting tangent buffers or running the rule.
New objects with the same relationships are allowed; prepare a separate cache when
those relationships change. Forward caches using structured gradient seeds also check
the aliasing of differentiable array leaves inside struct fields before refreshing their buffers.

Reverse reuse checks derivative storage through tuples, named tuples, builtin arrays
and structural tangents in `f` and `x`: sizes, compatible slot types, defined primal
slots and `MemoryRef` offsets. Above the first mutable or array storage, immutable
slots may change type when their fdata type is unchanged; the same swap inside a
vector, `Ref` or mutable struct is refused. An undefined primal slot may reuse an
initialized tangent slot. Custom tangent interiors, including `IdDict` values and
`FunctionWrapper` captures, are not checked. Custom `AbstractArray` tangents have a
size check when the primal is also an `AbstractArray`. Callers must preserve aliasing
in the parts the alias guard does not inspect: it does not traverse mutable struct interiors,
abstractly typed fields, array elements, or other variable-length containers such as
dictionaries. It does not establish overlap introduced through raw pointers or
`unsafe_wrap`; on Julia 1.10, array storage aliasing is identified by `Base.dataids`.
Every forward prepared-cache entry point preserves the prepared storage and object aliasing
at fixed positions, including aliasing between the callable and its arguments, unless
`check_aliasing=false` opts out of those per-call comparisons.
With the default aliasing checks, preparing with distinct arrays and reusing with one
array at both positions raises
`Mooncake.PreparedCacheError`, even for caller-supplied native directions. Derivative
and Jacobian calls permit offset changes that preserve these aliasing relationships.
Unchanged aliasing remains supported for derivative and Jacobian calls; forward gradients
refuse derivative storage shared across arguments, including the callable.
The prepared aliasing check has the same boundary, including abstractly typed
fields and mutable functor interiors.
Passing this check does not establish that unchecked aliases are safe. Rebuild the cache
when their relationships change.

## Low-level BLAS increments

Low-level `BLAS.nrm2(n, x, incx)` differentiation requires `incx > 0` in both
modes. Zero- and negative-increment behavior varies between BLAS implementations.
Low-level `BLAS.scal!` differentiation requires a positive increment in both modes.
Low-level `BLAS.asum`, `BLAS.axpy!`, `BLAS.nrm2` and `BLAS.scal!`
differentiation requires the raw-memory walk to stay within the supplied operand's
elements in both modes, even if BLAS can read other elements of its parent.
Pass a view covering every visited address. `BLAS.dot`, `BLAS.dotc` and `BLAS.dotu`
also accept walks outside a view when every visited address stays in its backing
array. BLAS level-1 rules refuse `ReinterpretArray` operands in both modes,
including generic routes that reach these rules. Direct `LinearAlgebra.dot` between
vectors of the same element type supports operands reinterpreting `T` ↔ `Complex{T}`
(`T <: IEEEFloat`). High-level matrix multiplication on `reinterpret(Float64, z)`
and Cholesky on `Symmetric(reinterpret(Float64, z))` also differentiate correctly
through the element-access rules. These rules require the same `T`/`Complex{T}`
layout throughout the backing storage, including views, reshapes, and nested
reinterprets; an incompatible inner reinterpret throws `ArgumentError` naming the
access. Wrapped `Float16` and `ComplexF16` `dot` operands (including views of
reinterpreted vectors) work through generic element access. Wrapped `dot` operands that reach
BLAS (`Float32`, `Float64`, `ComplexF32`, `ComplexF64`), `axpy!`, and `scal!`
remain refused with `BLAS does not support operand`.
`reinterpret(reshape, …)` remains refused by the pointer re-typing, object-pointer,
or chunk-width guards; ordinary `reshape(reinterpret(T, x), …)` is supported.
A bare `Ptr` operand has no owning array, so its walk is not bounds-checked and
staying within the buffer is the caller's responsibility.
