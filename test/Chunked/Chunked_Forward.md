# Design Doc: Chunked Forward Mode AD via `ChunkedPrimitiveTangent`

## Summary

This PR introduces Chunked Forward Mode AD in a totally Modular way, without affecting, interfering with the current Core Library's Implementation & Performance. It only compliments it.
Specifically, we currently work with — `ChunkedPrimitiveTangent{T, N}` — and a corresponding
dispatch for `Mooncake.frule!!` that propagates **N tangent directions simultaneously** in a
single forward pass. This is the forward-mode analogue of batched reverse-mode AD, and is
particularly useful for functions `f: R^n -> R^m` where `m > n` (many outputs, few inputs).

---

## Motivation

Mooncake's current forward mode propagates **one tangent direction per call** to `frule!!`.
Writing all rules again for Chunked Tangent Types is a waste of time.
A step further is computing a full Jacobian for `f: R^n -> R^m` therefore requires `n` separate `frule!!` calls —
one per input dimension.

But best way ? 

The key observation is: if `f` is **Fréchet differentiable** at the primal point, its derivative
is a bounded linear map. This means we can:

1. Compute the `n` standard basis JVPs once (one `frule!!` call per input), assembling the full `m × n` Jacobian.
2. Apply that Jacobian as a **single linear map** to all `N` tangent directions in the chunk simultaneously.

This avoids re-running `frule!!` N times per input — instead we pay `O(n * cost(f))` regardless
of chunk size `N`, plus an `O(m * n * N)` matrix-vector product at the end.

---

## Background: Fréchet vs Gâteaux Differentiability

This distinction is critical to correctness.

- A function is **Gâteaux differentiable** at `x` if the directional derivative exists in every
  direction `v`: `lim_{t→0} [f(x + tv) - f(x)] / t`
- A function is **Fréchet differentiable** at `x` if there exists a **bounded linear map** `Df(x)`
  such that `f(x + h) = f(x) + Df(x)h + o(||h||)`. Fréchet implies the derivative is the *same*
  linear map in all directions.

**Why this matters for chunking:** The chunked approach applies a single precomputed Jacobian
`basis_jvps` linearly across all N directions. This is only valid if the Jacobian is the same in
all those directions — i.e., if `f` is Fréchet differentiable.

> **Note:** Mooncake's own forward mode documentation assumes Fréchet differentiability.
> See: https://chalk-lab.github.io/Mooncake.jl/dev/tutorial/#Mooncake.jl-Functions
> This prototype therefore inherits the same assumption.

### What about Gâteaux-only functions?

For functions that are Gâteaux but not Fréchet differentiable ("crumpled manifolds"), the linear
map assumption breaks down. The correct approach in this case is to fall back to **N separate
`frule!!` calls** — one per direction in the chunk. This is a TODO and is described below.

---

## Design

### `ChunkedPrimitiveTangent{T, N}`

```julia
struct ChunkedPrimitiveTangent{T<:PrimitiveTangents, N}
    values::NTuple{N, T}
end
```

Holds `N` tangent directions for a single scalar primal. Stored as `NTuple{N,T}`:

- **SoA (Struct-of-Arrays) layout** — contiguous per field, SIMD-friendly
- For `N ≤ 16`: lives entirely in registers, fully unrolled by the Julia compiler
- Satisfies Mooncake's tangent interface (`zero_tangent`, `tangent_type`)

`T` is constrained to `PrimitiveTangents = Union{Base.IEEEFloat, NoTangent}`, matching
Mooncake's primitive tangent types.

---

### `frule!!` dispatch

```
Mooncake.frule!!(f::Dual{F}, args::Dual{P, ChunkedPrimitiveTangent{T,N}}...) where {F,N,P<:Real,T}
```

**Current domain:** scalar (`P<:Real`) primal arguments only. (This is because currently, we only have written helper methods for `ChunkedPrimitiveTangent` input tangents.)
**Current range:** scalar, `AbstractArray`, `Tuple`, `NamedTuple` outputs all supported.

#### Algorithm

```
1. Build atomic_frule once:   atomic_frule = Mooncake.build_frule(f, args...)

2. Construct n standard basis dual sets:
      e_i = (0, ..., 1_i, ..., 0)  for i = 1..n

3. Evaluate atomic_frule(f, e_i...) for each i:
      - Collect primal output (check consistency across probes → mutation detection)
      - Collect JVP column: flatten_tangent(jvp_i) → Vector{T}

4. Assemble Jacobian:
      basis_jvps = hcat(columns...)   # m × n Matrix

5. Apply linear map across all N chunk directions:
      scaled_tangents = basis_jvps * dargs_chunks   # O(m * n * N)

6. Unflatten back into output structure:
      chunked_tangents = unflatten_tangent(chunks, primal_out)

7. Return Dual(primal_out, chunked_tangents)
```

#### Complexity

| | Cost |
|---|---|
| Rule compilation | `O(1)` — `build_frule` called once |
| Basis probes | `O(n * cost(f))` — n evaluations of the compiled rule |
| Linear map | `O(m * n * N)` — matrix × chunk multiplication |
| Space | `O(m * n + m * N)` — Jacobian + output chunks |

Where `n` = number of inputs, `m` = flattened output dimension, `N` = chunk size.

---

### Output Flattening / Unflattening

To support structured outputs (`Tuple`, `AbstractArray`, `NamedTuple`), tangents are flattened
to a `Vector{T}` for Jacobian assembly, then unflattened back to the original structure:

```
flatten_tangent:   output tangent  →  Vector{T}  (flat scalar sequence)
unflatten_tangent: Vector{CPT} + primal shape  →  original tangent structure
```

Supported output types:

| Type | Flatten | Unflatten |
|---|---|---|
| `Real` | `[t]` | scalar `ChunkedPrimitiveTangent` |
| `AbstractArray` | `vec(collect(t))` | `reshape` to `size(primal_ref)` |
| `Tuple` | `vcat` over elements | recurse per element |
| `NamedTuple` | `vcat` over values | recurse, restore key names |
| arbitrary struct | recurse over `fieldnames` | reconstruct via positional constructor |

---

### Mutation Detection

Since the Jacobian is assembled over `n` separate basis probes, the primal output must be
**identical across all probes**. If it changes, the function is mutating its own state between
calls, which invalidates the Jacobian. This is detected via `approx_equal(primal_acc, primal_n)`
and raises an informative error:

```
Mutation detected across basis probes.
Your function is mutating its state and has memory of its calls!
```

> **Note:** Argument mutation is a separate concern. Since the current domain is `P<:Real`,
> scalar arguments are immutable in Julia and this cannot occur. When array inputs are added
> (see TODOs), argument mutation that does not affect the primal output value should be
> explicitly allowed.

---

## Correctness

### Test Strategy

Each example is tested by comparing the chunked output against Mooncake's scalar `frule!!`
for **every direction** `dir ∈ 1:N` independently:

```julia
# Chunked frule called once
chunked_out = Mooncake.frule!!(zero_dual(f), Dual(a, da), Dual(b, db))

# Scalar frule called N times as ground truth
for dir in 1:N
    scalar_ref = scalar_rule(zero_dual(f), Dual(a, da_vals[dir]), Dual(b, db_vals[dir]))
    @test extract_direction(tangent(chunked_out), dir) ≈ tangent(scalar_ref)
end
```

All tangent values are randomly generated (`randn()`) with `Random.seed!(42)` for reproducibility.

### Test Cases

| Example | Signature | Tests |
|---|---|---|
| 1 | `f: R² → R` | scalar output, 2 inputs |
| 2 | `f: R² → R³` | vector output |
| 3 | `f: R² → R^(2×2)` | matrix output |
| 4 | `f: R³ → (R, R^(2×2), R²)` | mixed tuple output |

---

## Limitations & TODOs

### In scope for follow-up PRs

- [ ] **Vector/array inputs** — current dispatch constrains `P<:Real`. Supporting
  `P<:AbstractArray` requires iterating over elements within each argument to construct
  per-element basis probes, not just per-argument.

- [ ] **Gâteaux fallback** — for functions that are not Fréchet differentiable, fall back to
  N separate `frule!!` evaluations (one per chunk direction). This handles "crumpled manifolds"
  at the cost of `O(N * n * cost(f))` instead of `O(n * cost(f))`.

- [ ] **Randomized Linearity Test** (`check_linearity_residue`) — use the Schwartz-Zippel lemma
  to probabilistically verify Fréchet differentiability at the primal point before applying the
  linear map. Catches cases where the function is Gâteaux-only and would silently produce wrong
  gradients.

- [ ] **Mixed dispatch** — allow `frule!!` to handle mixed arguments where some carry
  `ChunkedPrimitiveTangent` and others carry scalar tangents or unequal chunk sizes.

- [ ] **Large N** (`N > 16`) — consider a `ChunkArrayTangent` backed by a `Matrix` rather
  than `NTuple` for chunk sizes that exceed register capacity.

- [ ] **`one()` helper for non-scalar tangent types** — generalise `atomic_unit_directions`
  construction to cover `NamedTuple` and array tangent types when array inputs are added.

### Known non-goals

- Custom structs with non-positional constructors are not supported by the generic
  `unflatten_tangent` fallback and require explicit dispatch.
- `AbstractArray` elements that are not `<:Real` (e.g. `Vector{Vector{Float64}}`) are not
  supported — elements must be flat scalars.

---

## Alternatives Considered

**Option A: Propagate all N directions directly through `frule!!`**  
Run `frule!!` N times, once per chunk direction. Correct for Gâteaux-only functions but scales
as `O(N * n * cost(f))` — loses the key efficiency gain for large chunks.

**Option B: This PR (basis probe + linear map)**  
Run `frule!!` n times (once per input), build the Jacobian, apply as a linear map. Scales as
`O(n * cost(f) + m*n*N)` — optimal for Fréchet differentiable functions with large chunks.

Option B is strictly better than Option A when `f` is Fréchet differentiable and `N > 1`, which
is the common case. Option A becomes necessary only for Gâteaux-only points and is preserved
as a TODO fallback.