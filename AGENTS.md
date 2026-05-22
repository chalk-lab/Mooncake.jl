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
- World-age issues can arise when generated functions call back into Julia dispatch. `tangent_type` and `build_fdata` are generated functions; all sub-function calls must be in the returned expression (runtime), not in the generator body (generation time). If you add or modify either function, verify this.
- Avoid modifying `src/interpreter/` unless the task explicitly targets it. `Mooncake.primal_ir`, `Mooncake.dual_ir`, `Mooncake.fwd_ir`, and `Mooncake.rvs_ir` (see `docs/src/developer_documentation/developer_tools.md`) are available for inspection, but do not write rules or code that depends on their output — they are not semver-stable.
- Prefer writing rules at the lowest practical level, often around foreign-call boundaries (see `src/rules/blas.jl`), to reduce the total number of rules that need to be maintained.
- Implement both `frule!!` and `rrule!!` for new primitives where possible; rules that cover only one mode limit composability.
- `frule!!` must return the canonical `dual_type(Val(N), typeof(primal_result))` shape. Use `zero_dual(width, result)` for zero derivatives. Do not wrap NDual-bearing results in `Dual(..., NoTangent())`; that forces downstream `Dual`-wrapped overloads.
- `Dual` always wraps the top-level value (e.g. `Dual{Vector{Float64}, Vector{Float64}}`); rule signatures and dispatch helpers must not assume that array elements are themselves `Dual`. `Array{<:Dual}` and similar element-wise wrappings are not part of the `dual_type` protocol and should not appear in normal AD flow — do not add rule overloads matching them.
- `NDual` appears only inside the canonical forward representation `V = dual_type(Val(N), P)`, never as fields added to the user's primal struct. Direct NDual-bearing shapes are `IEEEFloat`, `Complex{<:IEEEFloat}`, arrays/memory of those scalars, tuples/named-tuples of those shapes, and recursive structural `NamedTuple` / `SplitDual` forms produced from those fields. Keep NDual-specific rules in `src/rules/rules_via_nfwd.jl`, and match only shapes returned by `dual_type`.
- In chunked forward mode, use the term `lane` consistently. A lane is one derivative slot in a width-`N` forward seed; `tangent(x, lane)` extracts that slot's tangent.
- **Both `tangent_type` and `dual_type` must be recursively coherent.** For every accessible field, element, or component of `P`, the tangent representation is `tangent_type(component_type)` and the forward representation is `dual_type(Val(N), component_type)`. This applies to:
  - Struct lifts: `tangent_type` produces `Tangent{NamedTuple{names, Tuple{(tangent_type(field_type_i))...}}}` (immutable) or `MutableTangent{...}` (mutable); `dual_type`'s structural lift mirrors this with per-field `dual_type(Val(N), field_type_i)`.
  - Arrays: `tangent_type(Array{T,D}) === Array{tangent_type(T),D}` (or `Array{T,D}` for IEEEFloat); `dual_type(Val(N), Array{T,D}) === Array{dual_type(Val(N), T),D}` for IEEEFloat T (canonical NDual-element form recursing on the element).
  - Tuples / NamedTuples: element-wise recursion in both functions.
  - Wrapper types (Diagonal, Adjoint, SubArray, …): the canonical NDual-element form recurses through the parent.
  Reverse rules (`rrule!!`, `zero_tangent`, `randn_tangent`, `fdata`/`rdata`) and forward rules (`frule!!`, `V === dual_type(Val(N), P)`) rely on this. A non-coherent `dual_type` breaks `lgetfield` chains and can silently corrupt forward AD on mutable structs with array fields (see `docs/src/known_limitations.md`). For mutable structs whose array fields use NDual-element canonical Vs, the coherent shape is `SplitDual{NamedTuple{names, Tuple{(dual_type(Val(N), field_type_i))...}}}` with no separate primal-half storage; primal values live inside the field Vs, e.g. each NDual's `.value`.
- `Lifted{P,N,V}` is the forward-mode slot wrapper, parallel to `CoDual{Tx,Tdx}`. For concrete runtime values, `P` is concrete and `V === dual_type(Val(N), P)`. Abstract slots use broad, width-preserving annotations such as `lifted_type(Val(N), abstract_P)`, while runtime values stay concrete.
- Construct rule outputs with `Lifted{P,N}(primal, tangent)`, not direct `Dual(...)` or `NDual{T,N}(...)` calls. Concrete `P` delegates to the inner constructors; abstract/nonconcrete `P` must sharpen through `typeof(primal)` and return a value compatible with the abstract slot, never `Any(primal, tangent)` or an exact abstract wrapper.
- **Do not branch on inner-`V` shape inside Lifted-typed rule bodies.** `dual_type(Val(N), P)` determines `V`; rule bodies should use `primal(slot)`, `tangent(slot)`, and `tangent(slot, lane)`, then construct outputs with `Lifted{P_out,N}(...)`. Use `_unlift` / `_lift` only for centralized compatibility, boundary code, or operations that naturally return an already-canonical inner value; do not use them as repeated per-primitive scaffolding.
- `Lifted` wraps once at the top level, including for concrete `Tuple` and `NamedTuple` primals. The inner `V` contains bare inner duals (`NDual`, `Complex{NDual}`, arrays of `NDual`, `Dual`, tuple/named-tuple values, or `SplitDual`), never nested `Lifted`.
- In generated IR, never annotate a join as `Lifted{Union{A,B}, ...}` when runtime values are `Lifted{A, ...}` or `Lifted{B, ...}`; `Lifted` is invariant. Use `Union{Lifted{A, ...}, Lifted{B, ...}}`, a broad `UnionAll`, or an unwrapped join representation, otherwise downstream `PiNode`s / OpaqueClosures may trust an impossible type fact and lower a valid path to `unreachable`.
- Every custom rule must be accompanied by an `@is_primitive` declaration; without it, the AD will not dispatch to the rule.
- Use `@zero_derivative` for rules with a zero derivative rather than writing a manual rule. Check `src/rules/` for other convenience macros before writing a rule from scratch.
- When choosing a tangent type: use `NoTangent` for non-differentiable types (e.g. integers, booleans, symbols); use `ZeroTangent` when the type is differentiable, but the derivative is structurally zero in a given rule.
- Prefer the narrowest rule signature that covers the intended cases; overly broad signatures can silently shadow more specific rules or cause method ambiguity errors.
- Only forward-over-reverse nested AD is tested. Do not assume rules compose correctly under reverse-over-reverse or other higher-order combinations unless explicitly verified.
- Prefer clear Julia error messages, especially around malformed rules, unsupported cases, and rule-construction failures.
- Mooncake's AD transform should preserve core execution properties: if the primal has zero allocation, the pullback should also have zero allocation; otherwise, pullbacks should allocate only a small constant-factor times the primal allocation (`c *` primal allocation); and type-stable primals should yield type-stable pullbacks.
- Preserve the aliasing invariant (`primal(a) === primal(b)` implies `fdata(a) === fdata(b)`): aliased primals must share fdata. Custom rules that intentionally break this must not allow the shared primal to be mutated in-place while both `CoDual`s are live. See the "Aliasing Invariant" subsection of `docs/src/understanding_mooncake/rule_system.md`.
- In reverse mode, Mooncake usually restores mutations on the pullback; stateful exceptions need explicit rules and focused tests.
- Internal helper APIs may change freely, but exported and public behaviour should come with tests, documentation, and clear error messages.
- Prepared caches are shape/type dependent; when cache construction changes, test reuse semantics and failure modes.
- If you change public APIs, developer tooling, or core internals, update docs under `docs/src/` when needed.
- Prefer targeted changes over broad refactors unless the task explicitly requires restructuring.
- Prefer clear, concise names for variables, types, and methods.
- Prefer ensuring correct `CoDual` / `Lifted` types inside rules over adding normalization in the autograd transform or public interfaces.
- When investigating bugs, first understand and document the cause, verify the intended fix before editing source files, and keep the investigation note in `temp/`.
- When fixing bugs or performance issues (allocations, type instability), prefer minimal inline fixes over new helper functions; make multiple pruning passes before committing to arrive at the smallest correct diff. Use the `minimise` skill before committing.
- Always run JuliaFormatter from the `test/integration_testing/format` environment (e.g. `julia --project=test/integration_testing/format -e 'using JuliaFormatter; JuliaFormatter.format(".")'`); it pins the version CI checks against, so any other env can introduce or miss diffs.

## Consistency

- When changing Julia version support, update `Project.toml`, `.github/workflows/CI.yml`, and `SUPPORT_POLICY.md` together.
- When a new rule depends on internals of an external package, tighten the corresponding `[compat]` bound in `Project.toml`.
- For new rules and internals, keep source, test-group wiring, and CI coverage in sync: add the matching test file, wire it into `test/runtests.jl` when applicable, and update CI if it deserves its own group.

## Testing

- Prefer constructing a minimal working example (MWE) first, then running the smallest focused test group that validates the fix, and only then broader test groups if needed.
- Before adding a new test or test helper, check whether the behaviour is already covered; prefer extending an existing case over introducing a new one, make multiple pruning passes, and keep additions minimal.
- Use the canonical test utilities: `Mooncake.TestUtils.test_rule` for new differentiation rules; `TestUtils.test_tangent_splitting` on a concrete value (add constructors to `src/test_resources.jl`) for tangent/fdata/rdata correctness rather than direct `@test tangent_type(...)` assertions; `TestUtils.test_data` for custom tangent type implementations. Layer-2 and Layer-3 forward construction checks should mirror `test_tangent`: use `test_dual`, `test_dual_types`, `test_lifted`, and `test_lifted_types` over the same representative primal cases.
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
