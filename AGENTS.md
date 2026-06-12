# AGENTS.md

## Purpose

Mooncake.jl is a Julia-first automatic differentiation package. Priorities, in order:

- broad coverage of real Julia behaviour: mutation, dynamic control flow, foreign calls, intrinsics, arrays, structs, tasks, closures, package extensions
- correctness and testability before aggressive optimisation, verified empirically through wide test coverage and tangent-type design
- composability: rules must compose predictably across primitives, custom tangents, nested AD, and mixed-mode AD
- representation discipline: tangent/dual types canonical enough that invariants are easy to state, test, and preserve
- clear validity boundaries and strong diagnostics: unsupported cases fail loudly and locally, never silently produce wrong derivatives
- numerical robustness (e.g. removable singularities must not produce NaN/Inf)
- performance via hand-written low-level `rrule!!` / `frule!!`, strict tangent types, and cached prepare/run APIs

Target: correct by construction where possible, aggressively testable where not, and explicit about every place where semantics depend on a rule.

## Repository Layout

- `src/`: main package code (`src/interpreter/`: IR/interpreter machinery; `src/rules/`: differentiation rules)
- `ext/`: package extensions
- `test/`: core suite; `test/ext/` and `test/integration_testing/`: separate-environment suites
- `docs/src/`: user, conceptual, utility, and developer documentation

## Working Conventions

- Mirror the source/test layout: tests for `src/.../foo.jl` live at `test/.../foo.jl`. Shared test setup goes in `test/front_matter.jl`; group dispatch in `test/runtests.jl` (see it for how to run groups, including interactively).
- Write rules at the lowest practical level (often foreign-call boundaries, see `src/rules/blas.jl`) to minimise the rule count. Canonicalise array-like inputs at the rule boundary (`arrayify`) instead of proliferating specialised methods.
- Implement both `frule!!` and `rrule!!` for new primitives where possible. Use `@zero_derivative` for zero-derivative rules; check `src/rules/` for other convenience macros first.
- Every custom rule needs an `@is_primitive` declaration, and the declaration must stay in lockstep with the rule's method coverage: a broader `@is_primitive` than the rule methods fails only at call time with a `MethodError`. Prefer the narrowest signature that covers the intended cases — overly broad signatures shadow more-specific rules or create ambiguities.
- `@from_rrule` / `@from_chainrules` import ChainRules rules; restrict them to scalar/array cases with `IEEEFloat` or `Complex` element types, where tangent conversions round-trip.
- Tangent-type choice: `NoTangent` for non-differentiable types; `ZeroTangent` for differentiable types whose derivative is structurally zero in a given rule.
- In `@generated` functions of the `tangent_type` family (`tangent_type`, `build_fdata`, `dual_type`, …): all sub-function calls **and all throws** belong in the returned expression, never in the generator body. For fallbacks, use the deferred-error pattern (`msg = "..."; return :(error($msg))`): an expansion-time throw gets baked into callers' compiled IR (and cached past invalidation under `@foldable`), so a later, more-specific extension overload could never take effect.
- For a signature or cache key that must match what `rule(args...)` dispatches to, use `Base._stable_typeof` (gives `Type{T}` for type values). `Mooncake._typeof` additionally sharpens `Tuple`/`NamedTuple` elements to `Type{X}` — use it only where that per-element sharpening is wanted (e.g. `fcodual_type`); in a cache key it silently mismatches runtime dispatch.
- Aliasing invariants: in **reverse mode**, `primal(a) === primal(b)` implies `fdata(a) === fdata(b)` (aliased primals share fdata so cotangent accumulation lands in one storage; see the "Aliasing Invariant" section of `docs/src/understanding_mooncake/rule_system.md`; a rule that intentionally breaks this must not let the shared primal be mutated in-place while both `CoDual`s are live). In **forward mode** the contract is asymmetric: `primal(slot)` aliases the user's storage, but tangent storage is slot-local — two `Lifted` slots over one primal carry independent JVP directions, by design.
- `Ptr` has no ownable zero-derivative storage: `zero_tangent`/`zero_codual`/`zero_fcodual` fall back to the `uninit_*` bitcast convention (primal address reinterpreted as `Ptr{tangent_type(P)}`) — a type-correct placeholder that must never be dereferenced as a derivative.
- In in-place tangent-set rules (`arrayset`/`memoryset`), zero the destination with the two-argument `zero_tangent(primal(v), tangent(v))` so the zero matches the existing runtime tangent's structure, not just the primal's type.
- Reverse mode restores mutations on the pullback by default; stateful exceptions need explicit rules and focused tests.
- Only forward-over-reverse nested AD is tested; do not assume other higher-order combinations compose without verifying.
- The AD transform must preserve execution properties: zero-allocation primals yield zero-allocation pullbacks (otherwise small constant-factor), and type-stable primals yield type-stable pullbacks.
- Fix representation problems by making `CoDual`/`Lifted` types correct inside rules, not by normalising in the transform or public interfaces.
- Avoid `src/interpreter/` unless the task targets it. `Mooncake.primal_ir`/`dual_ir`/`fwd_ir`/`rvs_ir` are for inspection only — not semver-stable.
- Internal helpers may change freely; exported/public behaviour needs tests, docs, and clear error messages. Prepared caches are shape/type dependent — when cache construction changes, test reuse and failure modes.
- Write clear error messages, especially for malformed rules, unsupported cases, and rule-construction failures; prefer clear, concise names for variables, types, and methods.
- Investigate before editing: root-cause and verify the intended fix first; keep investigation notes in `temp/` (untracked scratch). Prefer targeted changes and minimal inline fixes over new helpers or broad refactors; run the `minimise` skill before committing.
- Run JuliaFormatter only from `test/integration_testing/format` (pins the CI version): `julia --project=test/integration_testing/format -e 'using JuliaFormatter; JuliaFormatter.format(".")'`.

## Forward-mode representation (Lifted / dual_type)

The canonical forward value of a primal `P` at chunk width `N` is `V = dual_type(Val(N), P)`. The legacy two-field `Dual{P,T}` is gone. A *lane* is one derivative slot in a width-`N` seed; use the term consistently.

- `Lifted{P,N,V}` is the slot wrapper (fields `primal::P, value::V`), parallel to `CoDual`. For concrete runtime values, `P` is concrete and `V === dual_type(Val(N), P)`; abstract slots use broad width-preserving annotations from `lifted_type`.
- **Recursive coherence**: for every accessible field/element of `P`, the reverse representation is `tangent_type(component)` and the forward one is `dual_type(Val(N), component)`, mirroring each other shape-for-shape:
  - structs → `Tangent`/`MutableTangent` ↔ `ImmutableDual`/`MutableDual` (single-field wrappers holding the per-field `NamedTuple`; the slot primal lives in `Lifted`, not in them);
  - arrays → `Array{tangent_type(T),D}` ↔ `NDualArray{T,N,D,A,W}`, the parallel-arrays wrapper: `primal::A` aliases user storage, `partials::NTuple{N,A}` is slot-local, `W` is the per-element dual eltype (`NDual{T,N}` / `Complex{NDual{T,N}}`);
  - tuples/named-tuples → element-wise recursion; wrapper types (Diagonal, Adjoint, SubArray, …) recurse through the parent.

  Both rule families rely on this. A non-coherent `dual_type` breaks `lgetfield` chains and silently corrupts forward AD on mutable structs with array fields (`docs/src/known_limitations.md`).
- `NDual` lives only inside `V`, never as a field of the user's primal. `V` is built from bare inner duals (`NDual`, `Complex{NDual}`, `NDualArray`, `NDualMemoryRef`, tuples/named-tuples of those, `Immutable`/`MutableDual`); `Lifted` wraps exactly once at the top — never nested. `Array{<:NDual}` is **not** part of the protocol (arrays use `NDualArray`); rule signatures must match only shapes `dual_type` returns. Keep NDual-specific rules in `src/rules/rules_via_nfwd.jl`.
- `frule!!` must return the canonical `dual_type(Val(N), typeof(result))` shape — `zero_dual(Val(N), result)` for zero derivatives; never double-wrap (nested `Lifted`, or `NoDual`/`NoTangent` paired with a differentiable value).
- **Inner-value invariant**: an inner `NDual`'s `.value` must equal the primal result; rules scale only `partials`. Violations are silent correctness bugs. For width-`N` in-place rules, apply the in-place primal update once, hoisted out of the per-lane loop — repeating it corrupts the shared primal and later lanes.
- Construct outputs with `Lifted{P,N}(primal, value_or_seed_tangent)`, not raw `NDual{T,N}(...)`. Abstract/nonconcrete `P` must sharpen through `typeof(primal)` and stay compatible with the abstract slot.
- Do not branch on inner-`V` shape in rule bodies: `dual_type` determines `V`; use `primal(slot)`, `tangent(slot)`, `tangent(slot, lane)` and construct the output. `_unlift`/`_lift` are for boundaries and centralized compatibility only, not per-primitive scaffolding.
- Mutable-struct lane tangents are `MutableDualTangentView` proxies (writes delegate to the parent `MutableDual`; internals are underscore-prefixed so user fields named `parent`/`primal`/`lane` resolve correctly). There is no supertype shared with reverse `MutableTangent`: code handling both must use property syntax, not type dispatch.
- `Lifted` is invariant in `P`: never annotate an IR join as `Lifted{Union{A,B},...}` when runtime values are `Lifted{A,...}`/`Lifted{B,...}` — use `Union{Lifted{A,...}, Lifted{B,...}}`, a broad `UnionAll`, or an unwrapped join, or downstream `PiNode`s/OpaqueClosures lower valid paths to `unreachable`.
- `dual_type(Val(N), Ptr{T}) === NTuple{N, Ptr{T}}`: per-lane tangent pointers, valid whenever a separate tangent buffer exists to point at. A `pointer_from_objref` → `pointerref` round-trip through a value's *own* address has no addressable tangent (the partial is interleaved inside the dual), so that lane is deliberately left incoherent and fails loudly rather than silently dropping the derivative. See the forward-mode pointer note in `docs/src/known_limitations.md`.

## Consistency

- Changing Julia version support touches `Project.toml`, `.github/workflows/CI.yml`, and `SUPPORT_POLICY.md` together.
- A rule that depends on an external package's internals needs a tightened `[compat]` bound.
- Keep source, test-group wiring (`test/runtests.jl`), and CI coverage in sync when adding rules or internals.

## Testing

- MWE first, then the smallest focused test group, then broader groups only if needed. Before adding a test, check the behaviour isn't already covered; extend existing cases over adding new ones, and prune additions.
- Canonical utilities: `TestUtils.test_rule` for rules; `TestUtils.test_tangent_splitting` on a concrete value (constructors in `src/test_resources.jl`) for tangent/fdata/rdata correctness; `TestUtils.test_data` for custom tangent types.
- Prefer registering new tests as cases for these utilities in `src/` — a `hand_written_rule_test_cases` entry (in the rule's `src/rules/*.jl` file) or a `src/test_resources.jl` constructor — over bespoke test code in `test/`: registered cases get the full battery (both modes, widths 1–3, stability/allocs flags) automatically. Hand-write a test only for what the registries cannot express (e.g. specific seed shapes, mutation-aliasing assertions, `@test_throws` on guards).
- `test_rule` exercises forward rules at chunk widths 1, 2, 3 by default (via `TestUtils.test_frule`): width-1 finite-difference correctness, plus chunked checks that the primal is unchanged, inner `.value` tracks the primal, and — for primitive rules with plain numeric-dual arguments — per-lane partials match the width-1 oracle. Gaps to know: the per-lane oracle skips struct-lift/`Dict`/closure/`Ref` V shapes; derived (`is_primitive=false`) rules skip chunked widths entirely; seedless cases (raw `Ptr`) opt out via `skip_chunked`; random seeds rarely hit numeric edge cases — exercise those by hand (e.g. `x < 0` for `copysign`/`powi`).
- Ensure supported primal types and their tangent types are exercised against the relevant rules, for compatibility and composability.
- Never disable tests or weaken performance assertions to get CI green; stop and ask first.
- Debug mode helps test malformed rules and diagnose failures: `docs/src/utilities/debug_mode.md`.
- For performance-sensitive rules, run the rule directly: `@allocated` (zero-alloc primal ⇒ zero-alloc AD) and `@code_warntype` (type stability).
- Bug fixes land with a focused regression test; compiler/world-age-dependent fixes get isolated direct tests.
- `friendly_tangents` can mislead for structured/wrapped types; inspect the raw tangent before concluding a bug.
- `src/test_resources.jl` is shared infrastructure feeding broad interpreter/rule tests via `TestResources.generate_test_functions()` — not dead code.
- Prefer version-specific manifests (`Manifest-v1.10.toml`, `Manifest-v1.12.toml`) when running multiple Julia versions locally.
- Extension/integration suites under `test/ext/` and `test/integration_testing/` run in their own environments and are part of the package contract: weakdep/extension changes usually need updates there even when core tests pass.

## Documentation

- `docs/make.jl` defines the Documenter build and navigation; top-level user pages include `index.md`, `tutorial.md`, and `interface.md`.
- Known unsupported/incomplete behaviour: `docs/src/known_limitations.md`. Conceptual material: `docs/src/understanding_mooncake/`. Utilities: `docs/src/utilities/`. Contributor material: `docs/src/developer_documentation/`.
- Defining/adapting rules: `docs/src/utilities/defining_rules.md` (its "Canonicalising Tangent Types" section covers `arrayify`/`matrixify`). Custom tangent types and recursive types: `docs/src/developer_documentation/custom_tangent_type.md`.
- Update docs when changing public APIs, developer tooling, or core internals.
