# AGENTS.md

## Purpose

Mooncake.jl is a Julia-first automatic differentiation package focused on:

- broad coverage of real Julia behaviour, especially mutation, dynamic control flow, foreign calls, intrinsics, arrays, structs, tasks, closures, and package-extension code
- correctness and testability before aggressive optimisation, verified empirically through wide test coverage and tangent-type design
- composability: rules should compose predictably across primitives, custom tangents, nested AD, and mixed-mode AD
- representation discipline: tangent and cotangent types should be canonical enough that invariants are easy to state, test, and preserve
- strong diagnostics: malformed rules, tangent mismatches, world-age/compiler issues, and mutation mistakes should be easy to surface and debug
- clear validity boundaries: unsupported cases should fail loudly and locally, not silently produce wrong derivatives
- numerical robustness, including removable-singularity cases that would otherwise produce NaNs/Infs
- performance via hand-written low-level `rrule!!` / `frule!!`, strict tangent and cotangent types, and cached prepare/run APIs

The overall target is: correct by construction where possible, aggressively testable where not, and explicit about every place where semantics depend on a rule.

## Repository Layout

- `src/`: main package code
- `src/interpreter/`: IR and interpreter machinery for forward and reverse mode
- `src/rules/`: primitive- and domain-specific differentiation rules
- `ext/`: package extensions
- `test/`: core test suite
- `test/ext/`: extension tests in separate environments
- `test/integration_testing/`: broader integration suites in separate environments
- `docs/src/`: user, conceptual, utility, and developer documentation

## Working Conventions

- Keep changes aligned with the existing source/test layout: tests for `src/.../foo.jl` usually live at `test/.../foo.jl`.
- Put shared test setup in `test/front_matter.jl`; test-group dispatch lives in `test/runtests.jl`.
- For complex rules, especially array-heavy rules, prefer canonicalising inputs at the rule boundary with utilities such as `arrayify` rather than proliferating specialised methods.
- Mooncake provides helpers for importing rules from ChainRules via `@from_rrule` / `@from_chainrules`, but use them conservatively. In practice, restrict to scalar and array-like cases whose element types are `IEEEFloat` or `Complex` numbers, for which tangent conversions are well-defined and round-trip correctly.
- World-age issues can arise when generated functions call back into Julia dispatch. `tangent_type` is a generated function, and `build_fdata` is an `@inline` entry point that dispatches into generated machinery (`_build_fdata_cartesian`); in any such `@generated` body all sub-function calls must be in the returned expression (runtime), not in the generator body (generation time). If you add or modify `tangent_type`, `build_fdata`, or a similar generated function, verify this.
- A subtle second-order form of the same trap: even with sub-calls in the returned expression, an *expansion-time throw* (e.g. `return error(...)` in the meta-function body) bakes the throw into the caller's compiled IR. With `@foldable`, the optimizer caches that throw and Julia does not invalidate it when a later, more-specific overload is added — so compiled OpaqueClosure bodies (e.g. from `build_frule`) keep the stale throw in any later world. The mitigation, used by `tangent_type` and `fdata_type` for their primitive-type fallback, is a *runtime* error expression: `msg = "..."; return :(error($msg))` — the throw then fires only if dispatch actually reaches the fallback at runtime, which a more-specific extension overload can override. Always use this deferred-error pattern when adding a new `@generated` `tangent_type`-like function.
- Avoid modifying `src/interpreter/` unless the task explicitly targets it. `Mooncake.primal_ir`, `Mooncake.dual_ir`, `Mooncake.fwd_ir`, and `Mooncake.rvs_ir` (see `docs/src/developer_documentation/developer_tools.md`) are available for inspection, but do not write rules or code that depends on their output — they are not semver-stable.
- Prefer writing rules at the lowest practical level, often around foreign-call boundaries (see `src/rules/blas.jl`), to reduce the total number of rules that need to be maintained.
- Implement both `frule!!` and `rrule!!` for new primitives where possible; rules that cover only one mode limit composability.
- Every custom rule must be accompanied by an `@is_primitive` declaration; without it, the AD will not dispatch to the rule.
- Use `@zero_derivative` for rules with a zero derivative rather than writing a manual rule. Check `src/rules/` for other convenience macros before writing a rule from scratch.
- When choosing a tangent type: use `NoTangent` for non-differentiable types (e.g. integers, booleans, symbols); use `ZeroTangent` when the type is differentiable, but the derivative is structurally zero in a given rule.
- Prefer the narrowest rule signature that covers the intended cases; overly broad signatures can silently shadow more specific rules or cause method ambiguity errors.
- Keep an `@is_primitive` declaration and its rule methods in lockstep: marking a signature primitive stops the AD recursing into it and commits to a rule existing, so an `@is_primitive` broader than its `rule!!` / `frule!!` method coverage fails only at *call time* with a `MethodError` (e.g. `@is_primitive` on `AbstractMatrix` but a rule only on `Matrix` breaks on `SparseMatrixCSC`, GPU arrays, or `Float32`).
- When building a signature or cache key that must match what `rule(args...)` actually dispatches to, use `Base._stable_typeof`, not `Mooncake._typeof`. The three type-of queries differ on type-valued arguments: `typeof(T)` gives `DataType`; `Base._stable_typeof(T)` gives `Type{T}`; and `Mooncake._typeof` additionally recurses element-wise into `Tuple`/`NamedTuple`, sharpening type-valued elements to `Type{X}`. That extra sharpening over-specifies a cache-key signature and silently mismatches the runtime dispatch type, violating an assertion (a hard-to-debug failure). Use `Mooncake._typeof` only where the sharp per-element type is genuinely wanted (e.g. `fcodual_type` per element). This is the reverse-mode twin of the forward-mode `Type{X}` over-sharpening that forces the `typeof(primal)` fallback in abstract/type-valued `Lifted` construction.
- Only forward-over-reverse nested AD is tested. Do not assume rules compose correctly under reverse-over-reverse or other higher-order combinations unless explicitly verified.
- Prefer clear Julia error messages, especially around malformed rules, unsupported cases, and rule-construction failures.
- Mooncake's AD transform should preserve core execution properties: if the primal has zero allocation, the pullback should also have zero allocation; otherwise, pullbacks should allocate only a small constant-factor times the primal allocation (`c *` primal allocation); and type-stable primals should yield type-stable pullbacks.
- Preserve the aliasing invariant in *reverse mode*: `primal(a) === primal(b)` implies `fdata(a) === fdata(b)` — aliased primals must share fdata so cotangent accumulation lands in one storage. Custom rules that intentionally break this must not allow the shared primal to be mutated in-place while both `CoDual`s are live. See the "Aliasing Invariant" subsection of `docs/src/understanding_mooncake/rule_system.md`. The *forward-mode* aliasing contract is asymmetric: `primal(slot) === user_primal` (primal aliases user storage) but tangent storage is slot-local — two `Lifted` slots over the same primal carry independent tangent directions (different JVPs), and that is the correct semantics for forward propagation.
- `Ptr` has no ownable zero-derivative storage, so `zero_tangent` / `zero_codual` / `zero_fcodual` fall back to the `uninit_*` *bitcast* convention (the primal address reinterpreted as `Ptr{tangent_type(P)}`) — a type-correct structural placeholder that must **not** be dereferenced as valid derivatives. Rules touching pointers must respect this.
- In in-place tangent-set rules (`arrayset` / `memoryset`), zero the destination with the two-argument `zero_tangent(primal(v), tangent(v))`, not the single-argument form — so the zero matches the *existing* runtime tangent's structure, not just the type inferred from the primal.
- In reverse mode, Mooncake usually restores mutations on the pullback; stateful exceptions need explicit rules and focused tests.
- Internal helper APIs may change freely, but exported and public behaviour should come with tests, documentation, and clear error messages.
- Prepared caches are shape/type dependent; when cache construction changes, test reuse semantics and failure modes.
- If you change public APIs, developer tooling, or core internals, update docs under `docs/src/` when needed.
- Prefer targeted changes and clear, concise names; avoid broad refactors unless the task explicitly requires restructuring.
- Prefer ensuring correct `CoDual` / `Lifted` types inside rules over adding normalization in the autograd transform or public interfaces.
- When investigating bugs, first understand and document the cause, verify the intended fix before editing source files, and keep the investigation note in `temp/`.
- When fixing bugs or performance issues (allocations, type instability), prefer minimal inline fixes over new helper functions; make multiple pruning passes before committing to arrive at the smallest correct diff. Use the `minimise` skill before committing.
- Always run JuliaFormatter from the `test/integration_testing/format` environment (e.g. `julia --project=test/integration_testing/format -e 'using JuliaFormatter; JuliaFormatter.format(".")'`); it pins the version CI checks against, so any other env can introduce or miss diffs.

## Forward-mode representation (Lifted / dual_type)

The canonical forward value of a primal `P` at width `N` is `V = dual_type(Val(N), P)`. The legacy two-field `Dual{P, T}` has been removed.

- `Lifted{P,N,V}` is the forward-mode slot wrapper, parallel to `CoDual{Tx,Tdx}`. It is a two-field struct (`primal::P, value::V`); `primal(d::Lifted) = d.primal` is an O(1) field load at the slot level. For concrete runtime values `P` is concrete and `V === dual_type(Val(N), P)`; abstract slots use broad, width-preserving annotations such as `lifted_type(Val(N), abstract_P)`, while runtime values stay concrete.
- `tangent_type` and `dual_type` must be recursively coherent. For every accessible field, element, or component of `P`, the tangent representation is `tangent_type(component_type)` and the forward representation is `dual_type(Val(N), component_type)`:
  - Struct lifts: `tangent_type` produces `Tangent{NamedTuple{names, Tuple{(tangent_type(field_type_i))...}}}` (immutable) or `MutableTangent{...}` (mutable); `dual_type` mirrors this with per-field `dual_type(Val(N), field_type_i)` wrapped in `ImmutableDual{NamedTuple{...}}` (immutable) or `MutableDual{NamedTuple{...}}` (mutable). These structural-lift wrappers are single-field — each holds only the recursive `NamedTuple` — and mirror reverse-mode `Tangent` / `MutableTangent`; slot-level primal lives in `Lifted`, not inside them.
  - Arrays: `tangent_type(Array{T,D}) === Array{tangent_type(T),D}` (or `Array{T,D}` for IEEEFloat); `dual_type(Val(N), Array{T,D}) === NDualArray{T, N, D, Array{T,D}, W}` — the SoA wrapper whose `primal::A` aliases user storage and whose `partials::NTuple{N, A}` is slot-local. The trailing `W` is the wrapped per-element dual eltype (e.g. `NDual{T,N}`, or `Complex{NDual{T,N}}` for complex arrays), filled in by the 4-parameter constructors.
  - Tuples / NamedTuples: element-wise recursion in both functions.
  - Wrapper types (Diagonal, Adjoint, SubArray, …): canonical V recurses through the parent.

  Reverse rules (`rrule!!`, `zero_tangent`, `randn_tangent`, `fdata`/`rdata`) and forward rules (`frule!!`) rely on this. A non-coherent `dual_type` breaks `lgetfield` chains and can silently corrupt forward AD on mutable structs with array fields (see `docs/src/known_limitations.md`).
- `NDual` appears only inside `V`, never as a field added to the user's primal struct. The inner `V` is built from bare inner duals — `NDual`, `Complex{NDual}`, `NDualArray`, `NDualMemoryRef`, tuples/named-tuples of those, `ImmutableDual`, or `MutableDual` — and `Lifted` wraps once at the top level (including for concrete `Tuple`/`NamedTuple` primals); the inner `V` never contains a nested `Lifted`. Rule signatures must not assume array elements are themselves dual numbers: `Array{<:NDual}` and similar element-wise (AoS) wrappings are not part of the `dual_type` protocol — arrays use the SoA `NDualArray`. Keep NDual-specific rules in `src/rules/rules_via_nfwd.jl`, and match only shapes returned by `dual_type`.
- `frule!!` must return the canonical `dual_type(Val(N), typeof(primal_result))` shape. Use `zero_dual(Val(N), result)` for zero derivatives. Do not double-wrap an NDual-bearing result (e.g. in a nested `Lifted`, or paired with `NoDual`/`NoTangent` when the value is differentiable); return the canonical `V` directly.
- The inner `NDual`'s `.value` must equal the primal result; a rule scales only the `partials`. Overwriting `.value` with a derivative-scaled quantity (e.g. `grad * tangent(x)`) is a silent correctness bug that `test_rule` does not catch. For width-`N` in-place rules, apply any in-place primal update *once*, hoisted out of the per-lane loop — repeating it per lane corrupts the shared primal and all later lanes.
- Construct rule outputs with `Lifted{P,N}(primal, value)` or `Lifted{P,N}(primal, seed_tangent)`, not direct `NDual{T,N}(...)` calls. Concrete `P` delegates to the inner construction path; abstract/nonconcrete `P` must sharpen through `typeof(primal)` and return a value compatible with the abstract slot, never `Any(primal, tangent)` or an exact abstract wrapper.
- Do not branch on inner-`V` shape inside Lifted-typed rule bodies. `dual_type(Val(N), P)` determines `V`; use `primal(slot)`, `tangent(slot)`, and `tangent(slot, lane)` — a *lane* is one derivative slot in a width-`N` forward seed; use this term consistently — then construct outputs with `Lifted{P_out,N}(...)`. Use `_unlift` / `_lift` only for centralized compatibility, boundary code, or operations that naturally return an already-canonical inner value; not as repeated per-primitive scaffolding.
- Forward-mode lane tangents for mutable struct slots are `MutableDualTangentView` proxies (immutable views that delegate `setproperty!` to the parent `MutableDual`). Rules dispatching on tangent shape should use the `AbstractMutableTangent` supertype to accept both reverse-mode owned `MutableTangent` and forward-mode views.
- In generated IR, never annotate a join as `Lifted{Union{A,B}, ...}` when runtime values are `Lifted{A, ...}` or `Lifted{B, ...}`; `Lifted` is invariant. Use `Union{Lifted{A, ...}, Lifted{B, ...}}`, a broad `UnionAll`, or an unwrapped join representation, otherwise downstream `PiNode`s / OpaqueClosures may trust an impossible type fact and lower a valid path to `unreachable`.

## Consistency

- When changing Julia version support, update `Project.toml`, `.github/workflows/CI.yml`, and `SUPPORT_POLICY.md` together.
- When a new rule depends on internals of an external package, tighten the corresponding `[compat]` bound in `Project.toml`.
- For new rules and internals, keep source, test-group wiring, and CI coverage in sync: add the matching test file, wire it into `test/runtests.jl` when applicable, and update CI if it deserves its own group.

## Testing

- Prefer constructing a minimal working example (MWE) first, then running the smallest focused test group that validates the fix, and only then broader test groups if needed.
- Before adding a new test or test helper, check whether the behaviour is already covered; prefer extending an existing case over introducing a new one, make multiple pruning passes, and keep additions minimal.
- Use the canonical test utilities: `Mooncake.TestUtils.test_rule` for new differentiation rules; `TestUtils.test_tangent_splitting` on a concrete value (add constructors to `src/test_resources.jl`) for tangent/fdata/rdata correctness rather than direct `@test tangent_type(...)` assertions; `TestUtils.test_data` for custom tangent type implementations.
- `test_rule` builds a **width-1** seed and checks only the outer slot (`primal` + `partials`); it does not exercise chunked widths and does not assert the inner `NDual.value`. Forward-rule bugs slip past it easily — verify every new `frule!!` at widths 1, 2, and 3, exercise numeric edge cases (e.g. `x < 0` for `copysign`/`powi`), and confirm the inner `.value` equals the primal result.
- Do not disable tests or weaken performance assertions just to get CI green; if that appears necessary, stop and ask for confirmation first.
- Ensure supported primal types and their tangent types are exercised against the relevant rules for compatibility and composability.
- Mooncake has a debug mode which is useful for testing malformed rules and diagnosing rule failures; see `docs/src/utilities/debug_mode.md`.
- For performance-sensitive rules, verify by running the `frule!!` or `rrule!!` directly and checking allocations and runtime against the primal. Use `@allocated` to ensure that zero-allocation primals still yield zero-allocation AD paths, and `@code_warntype` to check for type stability.
- Bug fixes should land with a focused regression test; if the fix depends on compiler or world-age behaviour, isolate it and test directly.
- `friendly_tangents` can display a misleading value for structured or wrapped types even when the underlying tangent data is correct. Do not treat a surprising `friendly_tangents` result as proof of a bug without also inspecting the raw tangent.
- `src/test_resources.jl` is shared test infrastructure, not dead code. It feeds broad interpreter/rule tests indirectly via `TestResources.generate_test_functions()`, so do not judge it by one-file-one-test symmetry.
- Treat `temp/` as local scratch space, preferably untracked. Put ad hoc experiments, scratch scripts, and debugging MWEs there rather than in source or test files.
- See `test/runtests.jl` for how to run tests (interactively or in groups).
- When running multiple Julia minor versions locally, prefer version-specific manifests such as `Manifest-v1.10.toml` and `Manifest-v1.12.toml` instead of re-resolving a shared `Manifest.toml`. Julia will pick the matching manifest automatically, which avoids cross-version resolver breakage.
- Extension and integration tests should generally be run from their own files/environments under `test/ext/` and `test/integration_testing/`. These are part of the package contract, not optional extras, so changes to weakdeps/extensions often need updates there even if core tests still pass.

## Documentation

- `docs/make.jl` defines the Documenter build and navigation structure.
- Main docs sections include top-level user pages such as `index.md`, `tutorial.md`, and `interface.md`.
- Known unsupported or incomplete behaviour is documented in `docs/src/known_limitations.md`.
- Conceptual material lives under `docs/src/understanding_mooncake/`.
- Utility docs live under `docs/src/utilities/`.
- Internal and contributor material lives under `docs/src/developer_documentation/`.
- For defining or adapting rules, see `docs/src/utilities/defining_rules.md`; for complex array-like rules, see its `Canonicalising Tangent Types` section for `arrayify`/`matrixify` guidance.
- For recursive types or custom tangent implementations, start with `docs/src/developer_documentation/custom_tangent_type.md`.
