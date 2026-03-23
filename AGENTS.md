# AGENTS.md

## Purpose

Mooncake.jl is a Julia-first automatic differentiation package focused on:

- broad coverage of real Julia behaviour, especially mutation, dynamic control flow,
  foreign calls, intrinsics, arrays, structs, tasks, closures, and package-extension
  code
- correctness and testability before aggressive optimization, verified empirically
  through wide test coverage and tangent-type design
- composability: rules should compose predictably across primitives, custom tangents,
  nested AD, and mixed-mode AD
- representation discipline: tangent and cotangent types should be canonical enough that
  invariants are easy to state, test, and preserve
- strong diagnostics: malformed rules, tangent mismatches, world-age/compiler issues,
  and mutation mistakes should be easy to surface and debug
- clear invalidity boundaries: unsupported cases should fail loudly and locally, not
  silently produce wrong derivatives
- numerical robustness, including removable-singularity cases that would otherwise
  produce NaNs/Infs
- performance via hand-written low-level `rrule!!` / `frule!!`, strict tangent and
  cotangent types, and cached prepare/run APIs

The overall target is: correct by construction where possible, aggressively testable where not, and explicit about every place semantics depend on a rule.

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

- Keep changes aligned with the existing source/test layout: tests for
  `src/.../foo.jl` usually live at `test/.../foo.jl`.
- Put shared test setup in `test/front_matter.jl`; test-group dispatch lives in
  `test/runtests.jl`.
- For complex rules, especially array-heavy rules, prefer canonicalising inputs at the
  rule boundary with utilities such as `arrayify` rather than proliferating specialized
  methods.
- Mooncake provides helpers for importing rules from ChainRules via `@from_rrule` /
  `@from_chainrules`, but use them conservatively. In practice, restrict to scalar and
  array-like cases whose element types are `IEEEFloat` or `Complex` numbers, where
  tangent conversions are well-defined and round-trip correctly.
- World-age issues can arise when generated functions call back into Julia dispatch.
  `tangent_type` and `build_fdata` are generated functions; all sub-function calls must
  be in the returned expression (runtime), not in the generator body (generation time).
  If you add or modify either function, verify this.
- Avoid modifying `src/interpreter/` unless the task explicitly targets it.
  `Mooncake.primal_ir`, `Mooncake.dual_ir`, `Mooncake.fwd_ir`, and `Mooncake.rvs_ir`
  (see `docs/src/developer_documentation/developer_tools.md`) are available for
  inspection, but do not write rules or code that depends on their output — they are not
  semver-stable.
- Prefer writing rules at the lowest practical level, often around foreign-call
  boundaries (see `src/rules/blas.jl`), to reduce the total number of rules that need to
  be maintained.
- Implement both `frule!!` and `rrule!!` for new primitives where possible; rules that
  cover only one mode limit composability.
- Every custom rule must be accompanied by an `@is_primitive` declaration; without it
  the AD will not dispatch to the rule.
- Use `@zero_derivative` for rules with a zero derivative rather than writing a manual
  rule. Check `src/rules/` for other convenience macros before writing a rule from
  scratch.
- When choosing a tangent type: use `NoTangent` for non-differentiable types (e.g.
  integers, booleans, symbols); use `ZeroTangent` when the type is differentiable but
  the derivative is structurally zero in a given rule.
- Prefer the narrowest rule signature that covers the intended cases; overly broad
  signatures can silently shadow more specific rules or cause method ambiguity errors.
- Only forward-over-reverse nested AD is tested. Do not assume rules compose correctly
  under reverse-over-reverse or other higher-order combinations unless explicitly verified.
- Prefer clear Julia error messages, especially around malformed rules, unsupported
  cases, and rule-construction failures.
- Mooncake's AD transform should preserve core execution properties: allocation-free
  primals should yield allocation-free pullbacks; otherwise pullbacks should add only a
  small constant-factor allocation overhead; and type-stable primals should yield
  type-stable pullbacks.
- Preserve semantics under mutation and aliasing, not just pure-function cases.
- In reverse mode, Mooncake usually restores mutations on the pullback; stateful
  exceptions need explicit rules and focused tests.
- Internal helper APIs may change freely, but exported and public behaviour should come
  with tests, documentation, and clear error messages.
- Prepared caches are shape/type dependent; when cache construction changes, test reuse
  semantics and failure modes.
- If you change public APIs, developer tooling, or core internals, update docs under
  `docs/src/` when needed.
- Prefer targeted changes over broad refactors unless the task explicitly requires
  restructuring.
- Prefer clear, concise names for variables, types, and methods.

## Consistency

- When changing Julia version support, update `Project.toml`,
  `.github/workflows/CI.yml`, and `SUPPORT_POLICY.md` together.
- For new rules and internals, keep source, test-group wiring, and CI coverage in sync:
  add the matching test file, wire it into `test/runtests.jl` when applicable, and
  update CI if it deserves its own group.

## Testing

- Run focused test groups during development instead of the full suite when possible.
- For new differentiation rules, prefer testing them with `Mooncake.TestUtils.test_rule`.
- Ensure supported primal types and their tangent types are exercised against the
  relevant rules for compatibility and composability.
- Mooncake has a debug mode which is useful for testing malformed rules and diagnosing
  rule failures; see `docs/src/utilities/debug_mode.md`.
- For performance-sensitive rules, verify by running the `frule!!` or `rrule!!`
  directly and checking allocations and runtime against the primal.
- Bug fixes in rules, the interpreter, or compiler interop should ideally land with a
  focused regression test.
- If a fix depends on compiler or world-age behaviour, isolate it and test it directly.
- `friendly_tangents` can display a misleading value for structured or wrapped types even
  when the underlying tangent data is correct. Do not treat a surprising `friendly_tangents`
  result as proof of a bug without also inspecting the raw tangent.
- `src/test_resources.jl` is shared test infrastructure, not dead code. It feeds broad
  interpreter/rule tests indirectly via `TestResources.generate_test_functions()`, so do
  not judge it by one-file-one-test symmetry.
- Put ad hoc experiments and scratch scripts in `temp/`, not in tracked source or test
  files.
- Typical command from the repo root:

```bash
julia --project=. -e 'import Pkg; Pkg.test(; test_args=ARGS)' -- rules/random
```

- For an interactive test loop, work from the top level of the `Mooncake.jl` checkout
  and use a local tooling environment such as `temp/testenv`:

```bash
mkdir -p temp/testenv
julia --project=temp/testenv -e 'using Pkg; Pkg.add("TestEnv"); Pkg.develop(path=".")'
julia --project=temp/testenv -e 'using TestEnv; TestEnv.activate(); include("test/front_matter.jl")'
```

Then include the specific test file you want.

- Extension and integration tests should generally be run from their own
  files/environments under `test/ext/` and `test/integration_testing/`. These are part
  of the package contract, not optional extras, so changes to weakdeps/extensions often
  need updates there even if core tests still pass.

## Documentation

- `docs/make.jl` defines the Documenter build and navigation structure.
- Main docs sections include top-level user pages such as `index.md`, `tutorial.md`,
  and `interface.md`.
- Known unsupported or incomplete behaviour is documented in
  `docs/src/known_limitations.md`.
- Conceptual material lives under `docs/src/understanding_mooncake/`.
- Utility docs live under `docs/src/utilities/`.
- Internal and contributor material lives under `docs/src/developer_documentation/`.
- For defining or adapting rules, see `docs/src/utilities/defining_rules.md`; for
  complex array-like rules, see its `Canonicalising Tangent Types` section for
  `arrayify`/`matrixify` guidance.
- For recursive types or custom tangent implementations, start with
  `docs/src/developer_documentation/custom_tangent_type.md`.
