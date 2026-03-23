# [MWE Patterns for Compiler Issues](@id compiler_mwe_patterns)

Templates for constructing minimal working examples that isolate Julia compiler
behavior from package-specific code. Each pattern targets a specific category
of compiler boundary issue.

## General principles

1. **Preserve the call shape.** The MWE must have the same argument structure
   (Vararg, Function type, `Type{T}`, etc.) that triggers the compiler behavior.
2. **Use `@noinline` barriers.** Prevent the compiler from seeing through the
   function you're testing, so the specialization/inlining decision is isolated.
3. **Create A/B pairs.** Always have two variants -- one that triggers the issue,
   one that doesn't -- differing by exactly one factor.
4. **Use `Base.specializations` to observe.** Don't rely on `@code_typed` alone --
   it can create a separate, fully-concrete specialization that hides the problem.

### Observing specializations

```julia
# List all specializations for a method
for s in Base.specializations(only(methods(my_function)))
    s === nothing && continue
    println(s.specTypes)
end
```

This is critical because `@code_typed` creates a new specialization on demand,
which may be more concrete than what runtime dispatch actually uses.

## Pattern 1: Specialization widening (Vararg + Function)

A Vararg method receiving a Function-typed argument. Julia may widen the
concrete function type to `Function` in the compilation signature.

```julia
# A: Vararg -- Function gets widened to `Function` in compilation sig
@noinline function via_vararg(args::Vararg{Any,N}) where {N}
    return args[1](args[2])
end

# B: Explicit args -- Function stays concrete
@noinline function via_explicit(f::F, x::X) where {F,X}
    return f(x)
end

function caller_vararg(f, x)
    s = 0.0
    for _ in 1:10_000
        s += via_vararg(f, x)
    end
    return s
end

function caller_explicit(f, x)
    s = 0.0
    for _ in 1:10_000
        s += via_explicit(f, x)
    end
    return s
end

# Warmup
caller_vararg(sin, 1.0)
caller_explicit(sin, 1.0)

# Observe: vararg has `Function` in specTypes, explicit has `typeof(sin)`
println("=== Vararg specializations ===")
for s in Base.specializations(only(methods(via_vararg)))
    s === nothing && continue
    println("  ", s.specTypes)
end

println("=== Explicit specializations ===")
for s in Base.specializations(only(methods(via_explicit)))
    s === nothing && continue
    println("  ", s.specTypes)
end
```

What to observe:
- `via_vararg` has specTypes with `Function` instead of `typeof(sin)`
- `caller_vararg`'s `@code_typed` shows `::Function` in the invoke
- Significant timing difference between the two

Variations:
- Replace `sin` with `Float64` (triggers `Type{T}` widening instead)
- Add `@inline` to `via_vararg` (bypasses specialization entirely)
- Use a callable struct instead of a function (not widened)

## Pattern 2: Specialization widening (Type{T})

A `Type` argument in a general-typed slot.

```julia
@noinline function with_type_vararg(args::Vararg{Any,N}) where {N}
    return args[1](args[2])
end

@noinline function with_type_explicit(::Type{T}, x) where {T}
    return T(x)
end

# A: Type{Float64} gets widened to Type in vararg
with_type_vararg(Float64, 1)
# B: Type{Float64} stays concrete
with_type_explicit(Float64, 1)

# Check specializations
for s in Base.specializations(only(methods(with_type_vararg)))
    s === nothing && continue
    println(s.specTypes)
end
```

## Pattern 3: Inlining failure

A small callee that should be inlined but isn't.

```julia
@noinline function tiny_callee(x)
    return x + 1.0
end

function caller_noinline(x)
    return tiny_callee(x) * 2.0
end

@inline function tiny_callee_inlined(x)
    return x + 1.0
end

function caller_inline(x)
    return tiny_callee_inlined(x) * 2.0
end

# A: has :invoke in optimized code
display(@code_typed optimize=true caller_noinline(1.0))
# B: callee is inlined away
display(@code_typed optimize=true caller_inline(1.0))
```

To investigate real inlining failures (where `@noinline` wasn't used but
inlining still didn't happen), count remaining `:invoke` statements:

```julia
ci = first(code_typed(my_function, my_argtypes; optimize=true))[1]
invokes = count(st -> st isa Expr && st.head === :invoke, ci.code)
println("Remaining invokes after optimization: $invokes")
```

## Pattern 4: SROA failure

A mutable struct allocation that should be eliminated but isn't. Use **mutable**
structs and escape through multiple paths to reliably trigger SROA failure:

```julia
mutable struct MutPair
    x::Float64
    y::Float64
end

# A: SROA succeeds -- struct never escapes
function sroa_success(a, b)
    p = MutPair(a, b)
    return p.x + p.y
end

# B: SROA fails -- struct escapes through conditional path
@noinline use_pair(p::MutPair) = p.x + p.y
@noinline modify_pair!(p::MutPair) = (p.x += 1.0; p)

function sroa_failure(a, b, flag)
    p = MutPair(a, b)
    if flag
        modify_pair!(p)
    end
    return use_pair(p)
end

# Check LLVM IR for allocation patterns
@code_llvm debuginfo=:none sroa_success(1.0, 2.0)
@code_llvm debuginfo=:none sroa_failure(1.0, 2.0, true)
```

!!! note
    The exact LLVM allocation function names change across Julia versions. Search
    for patterns: `gc_pool_alloc`, `gc_small_alloc`, `gc_alloc_obj`,
    `gc_alloc_bytes`. Do not hardcode a single function name.

## Pattern 5: Dynamic dispatch

A call site where inference can't determine the concrete type, forcing runtime
dispatch via `jl_apply_generic`.

```julia
# A: type-stable -- direct call
function stable_call(x::Float64)
    return sin(x)
end

# B: type-unstable with wide union -- forces dynamic dispatch
function wide_union_call(x, flag::Int)
    if flag == 1;     return x
    elseif flag == 2; return Int(round(x))
    elseif flag == 3; return string(x)
    else;             return Complex(x, x)
    end
end

function caller_of_unstable(x, flag)
    y = wide_union_call(x, flag)
    return sizeof(y)
end

display(@code_typed caller_of_unstable(1.0, 1))
@code_llvm debuginfo=:none caller_of_unstable(1.0, 1)
```

!!! note
    On modern Julia, small unions (2-3 types) are union-split and never hit
    `jl_apply_generic`. You need 4+ union members or `Any`-typed values to
    reliably see dynamic dispatch.

## Pattern 6: Boxing

A value type being passed through a type-erased boundary.

```julia
# A: concrete types -- no boxing
function no_boxing(x::Float64, y::Float64)
    return x + y
end

# B: Any-typed container forces boxing
function with_boxing(x::Float64)
    container = Any[x]
    return container[1]::Float64 + 1.0
end

@code_llvm debuginfo=:none no_boxing(1.0, 2.0)
@code_llvm debuginfo=:none with_boxing(1.0)
```

## Reduction technique

When reducing a package-specific issue to a Julia-only MWE:

1. **Identify the predicate.** What compiler behavior are you reproducing?
2. **Replace the package with a stub** that preserves the same signature shape:
   ```julia
   # If the original was: do_work(state, f, args..., options)
   function my_stub(state, f::F, x::Vararg{Any,N}) where {F,N}
       return f(x...)
   end
   ```
3. **Strip until it breaks.** Remove lines one at a time. If removing a line
   makes the predicate stop firing, that line is essential.
4. **Verify the A/B pair.** The "fixed" version should differ by exactly one
   factor.
