# `Lifted{P, N, V}` — Forward-Mode Slot Wrapper

This document describes `Lifted{P, N, V}`, the forward-mode slot wrapper used at OpaqueClosure boundaries in the primal-mode transform. It is the forward-mode counterpart to reverse mode's `CoDual{Tx, Tdx}`.

For background on primal-mode forward AD, see [Primal-Mode Design](@ref). For the chunked `NDual{T, N}` representation used as the inner shape for `IEEEFloat` primals, see [Scalar Rules via NDual](@ref).

## Motivation

`CoDual{Tx, Tdx}` parameterises on primal type `Tx` and tangent type `Tdx`. Its rule bodies write `CoDual(y, dy)` directly — there is one inner shape and `Tdx` absorbs all variability.

Forward mode has *several* inner shapes at every slot, depending on `(primal type, width)`:

| Primal type `P` | Width `N` | Inner dual `V` |
|---|---|---|
| `IEEEFloat` (e.g. `Float64`) | N | `NDual{T, N}` |
| `Complex{<:IEEEFloat}` | N | `Complex{NDual{T, N}}` |
| **Top-level** `Array{<:IEEEFloat, D}` (or `Complex{<:IEEEFloat}` elements) | N | `Array{NDual{T, N}, D}` |
| `Memory{<:IEEEFloat}` (1.11+) | N | `Memory{NDual{T, N}}` |
| `MemoryRef{<:IEEEFloat}` (1.11+) | N | `MemoryRef{NDual{T, N}}` |
| Concrete `Tuple{P1, ...}` | N | `Tuple{V_1, ...}` (element-wise recursive) |
| Concrete `NamedTuple{names, T}` | N | `NamedTuple{names, T'}` (element-wise recursive) |
| `Transpose{<:IEEEFloat, P_parent}` | N | `Transpose{NDual{T, N}, dual_type(Val(N), P_parent)}` |
| `Adjoint`, `SubArray`, `Diagonal`, `Symmetric`, `Hermitian`, `UpperTriangular`, `LowerTriangular`, `UnitUpperTriangular`, `UnitLowerTriangular`, `UpperHessenberg`, `ReshapedArray`, `ReinterpretArray` over IEEEFloat parents | N | Same template — element type lifted to `NDual{T, N}`, parent V from recursive `dual_type` |
| Concrete immutable struct with default `Tangent` tangent_type and all fields always initialised | N | `NamedTuple{fieldnames(P), Tuple{V_i, ...}}` (structural lift; recurses per-field) |
| Concrete mutable struct with at least one **top-level** `Array-of-IEEEFloat` field | N | `SplitDual{NamedTuple{fieldnames(P), Tuple{V_i, ...}}}` |
| **Nested arrays** (e.g. `Vector{Vector{Float64}}`), PossiblyUninit, custom tangent_type, non-canonical mutable, `NoTangent` primals, abstract, etc. | N | `Dual{P, tangent_type(Val(N), P)}` (parallel-Dual fallback) |

### Where the canonical NDual element form applies

The "Array{NDual{T,N}, D}" canonical form is reserved for **top-level**
IEEEFloat-element arrays (and their wrapper-views: Transpose, Symmetric,
etc.) and for **Tuple/NamedTuple** elements that themselves resolve to
NDual leaves. The reason is element-wise interleaving: an `Array{NDual{T,
N}}` lays `(value, partials)` in a single packed array, so in-place
mutations on the dual array propagate through the underlying buffer.
This requires *every element* to be a defined NDual.

Container types that allow undef elements at the Julia level cannot use
canonical NDual interleaving, because no NDual can be constructed for an
undef slot. **`Vector{Vector{Float64}}`** is the canonical example:
`setindex!(Vector{Vector{Float64}}(undef, 2), [1.0], 1)` leaves slot 2
unassigned, and any `map(zero_dual, ::Vector{Vector{Float64}})` errors
with `UndefRefError`. Nested arrays therefore use the parallel-Dual
fallback form, where the outer Vector stays in parallel storage and the
inner Vector elements (when defined) are themselves laid out as
`Vector{Float64}` tangents — no element-wise NDual recursion.

Concrete policy:

- `Array{IEEEFloat, D}` (top-level): canonical `Array{NDual{T, N}, D}`
- `Array{Complex{IEEEFloat}, D}` (top-level): canonical
  `Array{Complex{NDual{T, N}}, D}`
- `Tuple{T1, T2, ...}` / `NamedTuple{names, T}`: element-wise recursive
  — each element's `dual_type` is computed independently, and if an
  element is itself a top-level Array its inner form is canonical NDual
- `Array{Array{IEEEFloat, K}, D}` (nested): parallel-Dual fallback
  `Dual{Array{Array{T, K}, D}, NTangent{NTuple{N, Array{Array{T, K}, D}}}}`
- Same fallback applies to any other container that admits undef slots
  or whose element type lacks a canonical NDual form

The same constraint extends to SplitDual: a mutable struct qualifies
for `SplitDual{NamedTuple{...}}` only if it has a field whose type is
**top-level Array-of-IEEEFloat** (direct or via a nested struct
field). A field of type `Vector{Vector{Float64}}` does *not* count.

A 1-to-1 port of `CoDual`'s surface — having rule bodies write `Dual(y, dy)` — fails at the constructor: a rule body would have to *choose* which inner shape to build for each output slot. Every change to the inner-shape rules (e.g. flipping `IEEEFloat` from `Dual` to `NDual` at width 1) would ripple through every rule body.

`Lifted{P, N, V}` solves this by pushing the inner-shape choice down into the inner types' constructor methods, while making the slot's identity `(P, N)` dispatch-visible at the wrapper level.

## Three-layer type hierarchy

Each layer has a name-aligned type-level query:

```
Layer 1 — tangent_type:   P  →  T          bare tangent type
Layer 2 — dual_type:      P  →  V          bare inner dual type
Layer 3 — lifted_type:    P  →  L          wrapped slot type
```

`tangent_type` is the existing Layer 1 interface (see [Tangents](@ref)). The Lifted design adds `dual_type` (Layer 2) as the dispatch table mapping `(P, N)` to the chosen inner shape, and `lifted_type` (Layer 3) for the wrapped slot.

```julia
# Layer 2 — bare inner dual shape
dual_type(Val(0), P)                       = P
dual_type(Val(N), T<:IEEEFloat)            = NDual{T, N}
dual_type(Val(N), Complex{T})              = Complex{NDual{T, N}}
dual_type(Val(N), Array{T, D})             = Array{NDual{T, N}, D}                # T<:IEEEFloat (top-level only)
dual_type(Val(N), Wrapper{T, P_parent})    = Wrapper{NDual{T, N}, dual_type(Val(N), P_parent)}
                                            # Wrapper ∈ {Transpose, Adjoint, SubArray, Diagonal,
                                            # Symmetric, Hermitian, *Triangular, UpperHessenberg,
                                            # ReshapedArray, ReinterpretArray}; T<:IEEEFloat
dual_type(Val(N), Tuple{T1, T2, ...})      = Tuple{dual_type(Val(N), T1), ...}    # element-wise recursive
dual_type(Val(N), NamedTuple{names, T})    = NamedTuple{names, dual_type(Val(N), T)}
dual_type(Val(N), ImmutableStruct{T...})   = NamedTuple{fieldnames, Tuple{V_i, ...}}
                                            # structural lift; tangent_type(P) <: Tangent +
                                            # all fields always_initialised + lift-safe fields
dual_type(Val(N), MutableStruct{T...})     = SplitDual{NamedTuple{fieldnames, Tuple{V_i, ...}}}
                                            # if struct has a *top-level* Array-of-IEEEFloat field
                                            # (direct or via a nested struct, not nested-Array element)
dual_type(Val(N), P)                       = Dual{P, tangent_type(Val(N), P)}    # parallel-Dual fallback
                                            # for: nested arrays (e.g. Vector{Vector{Float64}}),
                                            # PossiblyUninit, custom tangent_type, abstract P,
                                            # NoTangent primals (String, Symbol, Nothing, etc.)

# Layer 3 — wrapped slot type
lifted_type(Val(0), P)                     = P                                  # primal passthrough
lifted_type(Val(N), P)                     = Lifted{P, N, dual_type(Val(N), P)}
```

**Element-wise NDual interleaving is reserved for top-level Array and
Tuple/NamedTuple elements**. The element type's `dual_type` is computed
recursively, but the canonical NDual interleaving (laying value+partials
in a single packed array) only fires at the *top level* of an
IEEEFloat-element container. Nested arrays like `Vector{Vector{Float64}}`
fall through to the parallel-Dual fallback because the canonical form
requires every element to be defined, which Julia's `Vector(undef, n)`
semantics doesn't guarantee.

`lifted_type` does *not* recurse into `Tuple` / `NamedTuple` primals — the element-wise structure lives in the `dual_type` layer (the `V` parameter). For `P = Tuple{Float64, Float64}` and `N = 2`:

```julia
lifted_type(Val(2), Tuple{Float64, Float64})
# = Lifted{Tuple{Float64, Float64}, 2, Tuple{NDual{Float64, 2}, NDual{Float64, 2}}}
```

The Tuple sits *inside* the wrapper, not outside it. There is exactly one outer `Lifted` per slot.

### Invariants

- `lifted_type(Val(N), P) === Lifted{P, N, dual_type(Val(N), P)}` for all `P` and `N >= 1`.
- `Lifted` never appears inside another `Lifted`'s `V`. Slot boundaries always look like `Lifted{P, N, V}` for a single `P`.
- `dual_type(Val(0), P) === P` and `lifted_type(Val(0), P) === P` — Layer 0 means "no AD"; both layers degenerate to bare `P`.

## The wrapper struct

```julia
struct Lifted{P, N, V}
    value::V
end
```

| Param | Meaning |
|---|---|
| `P` | The primal type (the type of `primal(d)`'s value). |
| `N` | The width — `1` for ordinary forward mode, `N >= 2` for chunked. |
| `V` | The inner dual shape. **Invariant**: `V === dual_type(Val(N), P)`. |

`V` appears in the struct definition only because Julia struct fields need explicit types; semantically, the wrapper's identity is the `(P, N)` pair. Two construction forms:

```julia
# 1-arg: wrap an already-built inner. V inferred from typeof(value).
Lifted{P, N}(value)

# 2-arg: build the inner via the inner type's own constructor — mirrors CoDual(x, dx).
Lifted{P, N}(primal, tangent)
```

## Rule-body convention

Rule bodies dispatch on `Lifted{P, N}` and build outputs via the 2-arg constructor. Width and primal type come from the where-clause; the inner shape `V` is invisible.

### Output `P` matches input `P`

```julia
function frule!!(::Lifted{typeof(abs_float), N},
                 x::Lifted{P, N}) where {N, P}
    px = primal(x)
    return Lifted{P, N}(abs_float(px), sign(px) * tangent(x))
end
```

### Output `P` differs from input `P` (e.g. `getfield`)

Use `typeof(value)` for the output's `P`:

```julia
function frule!!(::Lifted{typeof(getfield), N},
                 x::Lifted{P_in, N}, ::Val{name}) where {N, P_in, name}
    val  = getfield(primal(x), name)
    dval = getfield(tangent(x), name)
    return Lifted{typeof(val), N}(val, dval)
end
```

### In-place mutating rules

Mutate `primal(arg)` and `tangent(arg)` directly; return the unchanged arg:

```julia
function frule!!(::Lifted{typeof(setindex!), N},
                 x::Lifted{<:AbstractArray, N},
                 v::Lifted, i::Lifted) where {N}
    setindex!(primal(x), primal(v), primal(i))
    setindex!(tangent(x), tangent(v), primal(i))
    return x
end
```

### Tuple-primal rules — match `Lifted{<:Tuple, N}`

Concrete `Tuple` primals lift to a single outer `Lifted` whose inner is a `Tuple` of bare inner duals. Rule signatures dispatch on `Lifted{<:Tuple, N}` and reach the per-element duals via `tangent` / `primal`:

```julia
# tuple constructor — wrap a Tuple{Lifted, ...} of args into one outer Lifted{<:Tuple}.
function frule!!(::Lifted{typeof(tuple), N}, args::Vararg{Lifted, M}) where {N, M}
    P_out = Tuple{ntuple(i -> typeof(primal(args[i])), Val(M))...}
    inner = ntuple(i -> _unlift(args[i]), Val(M))
    return Lifted{P_out, N}(inner)
end
```

`_unlift(::Lifted)` returns the bare inner (the `V` field). Mark the rule as Lifted-aware so the IR-emit dispatches directly:

```julia
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(tuple), Vararg}}) = true
```

## Why the parameterisation works

### Slot identity is dispatch-visible

The rule body's signature names `(P, N)` directly:

```julia
frule!!(::Lifted{typeof(op), N}, x::Lifted{P, N}) where {N, P}
```

Helper dispatch can specialise on width and/or primal type:

```julia
helper(d::Lifted{P, 1})           where {P}    = ...   # width-1 fast path
helper(d::Lifted{P, N})           where {P, N} = ...   # generic chunked
helper(d::Lifted{<:IEEEFloat, N}) where {N}    = ...   # IEEEFloat-only
```

Inner-shape choice is hidden in the third parameter `V`, which the rule body never names.

### Inner-shape changes don't ripple

If `dual_type(Val(N), P)` changes for some `(N, P)` (for example, hypothetically flipping `dual_type(Val(1), T<:IEEEFloat)` from `NDual{T, 1}` to `Dual{T, T}`), only `dual_type` changes. Rule bodies dispatching on `Lifted{P, N}` are insensitive — they don't see `V`. The new inner type's constructor takes over inside the 2-arg `Lifted{P, N}` construction automatically.

### Tuple rule bodies collapse to a single branch

The "single outer `Lifted`" invariant eliminates the "chunked-NDual element tuple vs whole-Tuple legacy Dual" branching that exists in tuple rule bodies under the legacy bare `Dual` interface. Under `Lifted{P, N, V}`, tuple primal `P` has exactly one slot shape: `Lifted{P, N, Tuple{dual_type(Val(N), T1), ...}}`. There is no alternative form to choose between.

The concrete consequence is that the `tuple` rule collapses from three output shapes to one:

```julia
# Three structurally different output shapes (legacy):
function frule!!(::Dual{typeof(tuple)}, args::Vararg{Any, N}) where {N}
    if _has_ndual(args...)
        return tuple(args...)                        # bare element-wise lifted tuple
    end
    primal_output = tuple(map(primal, args)...)
    if tangent_type(_typeof(primal_output)) == NoTangent
        return zero_dual(primal_output)              # whole-Tuple Dual, NoTangent
    else
        return Dual(primal_output,                   # whole-Tuple Dual, Tuple tangent
                    tuple(map(tangent, args)...))
    end
end

# One branch (Lifted):
@inline function frule!!(::Lifted{typeof(tuple), N}, args::Vararg{Lifted, M}) where {N, M}
    P_out = Tuple{ntuple(i -> typeof(primal(args[i])), Val(M))...}
    return Lifted{P_out, N}(map(_unlift, args))
end
```

## Public API surface

```julia
# Type-level (3)
tangent_type(Val(N), P)                 # bare tangent type — Layer 1
dual_type(Val(N), P)                    # bare inner dual type — Layer 2
lifted_type(Val(N), P)                  # wrapped slot type — Layer 3

# Wrapper struct + constructors
struct Lifted{P, N, V}; value::V; end
Lifted{P, N}(inner)                     # 1-arg from inner; V inferred
Lifted{P, N}(primal, tangent)           # 2-arg, mirrors CoDual(x, dx)

# Layer-2 seed factories — return bare inner dual
zero_dual(::Val{N}, x)
uninit_dual(::Val{N}, x)
randn_dual(::Val{N}, rng, x)

# Layer-3 seed factories — return wrapped Lifted slot
zero_lifted(::Val{N}, x)
uninit_lifted(::Val{N}, x)
randn_lifted(::Val{N}, rng, x)

# Accessors
primal(d::Lifted)        = primal(d.value)
tangent(d::Lifted)       = tangent(d.value)
extract(d::Lifted)       = (primal(d), tangent(d))

# Wrap / unwrap mechanics
_lift(::Val{N}, ::Type{P}, inner)       # = Lifted{P, N}(inner)
_unlift(d::Lifted)                      # = d.value
```

## Comparison to `CoDual`

| | `CoDual{Tx, Tdx}` (reverse) | `Lifted{P, N, V}` (forward) |
|---|---|---|
| Type-level | `tangent_type`, `codual_type`, `fcodual_type` | `tangent_type`, `dual_type`, `lifted_type` |
| Inner shapes | one (`Tdx` absorbs all variability) | several (`Dual`, `NDual`, `Complex{<:NDual}`, `Array{<:NDual}`, …) |
| Width parameter | none | `N` (1 for ordinary, N>=2 for chunked) |
| Slot identity | `(Tx, Tdx)` | `(P, N)` |
| 2-arg construction | `CoDual(x, dx)` | `Lifted{P, N}(primal, tangent)` |
| Layer-2 factories | none | `zero_dual`, `uninit_dual`, `randn_dual` |
| Layer-3 factories | `zero_codual`, `uninit_codual`, `zero_fcodual`, `uninit_fcodual` | `zero_lifted`, `uninit_lifted`, `randn_lifted` |
| Tuple handling | one outer `CoDual{<:Tuple}` (inner tangent is `Tuple` of cotangents) | one outer `Lifted{<:Tuple}` (inner `V` is `Tuple` of bare inner duals) |

The forward-mode hierarchy has one extra layer (`dual_type`) and one extra factory family — both consequences of supporting multiple inner shapes. The remaining surface is parallel.
