# AGENTS.md

## Purpose

Mooncake.jl is a Julia-first automatic differentiation package focused on:

- broad coverage of real Julia behaviour, especially mutation, dynamic control flow,
  foreign calls, intrinsics, arrays, structs, tasks, closures, and package-extension
  code
- correctness and testability before aggressive optimization, verified empirically
  through wide test coverage and tangent-type design
- composability: rules should compose predictably across primitives, custom tangents, nested AD, and mixed-mode AD
- representation discipline: tangent and cotangent types should be canonical enough that invariants are easy to state, test, and preserve
- strong diagnostics: malformed rules, tangent mismatches, world-age/compiler issues,
  and mutation mistakes should be easy to surface and debug
- clear invalidity boundaries: unsupported cases should fail loudly and locally, not silently produce wrong derivatives
- numerical robustness, including removable-singularity cases that would otherwise
  produce NaNs/Infs
- performance via hand-written low-level `rrule!!` / `frule!!`, strict tangent and cotangent types, and cached prepare/run APIs

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

- Keep changes aligned with the existing source/test layout: tests for `src/.../foo.jl` usually live at `test/.../foo.jl`.
- Put shared test setup in `test/front_matter.jl`; test-group dispatch lives in `test/runtests.jl`.
- For complex rules, especially array-heavy rules, prefer canonicalising inputs at the rule boundary with utilities such as `arrayify` rather than proliferating specialized methods.
- Mooncake provides helpers for importing rules from ChainRules via `@from_rrule` / `@from_chainrules`, but use them conservatively. In practice, keep this to scalar and array-like cases where the tangent conversions are robust.
- Prefer writing rules at the lowest practical level, often around foreign-call
  boundaries (see `src/rules/blas.jl`), to reduce the total number of rules that need to
  be maintained.
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
  with tests, documentation, and good errors.
- Prepared caches are shape/type dependent; when cache construction changes, test reuse
  semantics and failure modes.
- If you change public APIs, developer tooling, or core internals, update docs under `docs/src/` when needed.
- Prefer targeted changes over broad refactors unless the task explicitly requires restructuring.

## Consistency

- When changing Julia version support, update `Project.toml`, `.github/workflows/CI.yml`, and `SUPPORT_POLICY.md` together.
- For new rules and internals, keep source, test-group wiring, and CI coverage in sync:
  add the matching test file, wire it into `test/runtests.jl` when applicable, and
  update CI if it deserves its own group.

## Testing

- Run focused test groups during development instead of the full suite when possible.
- For new differentiation rules, prefer testing them with `Mooncake.TestUtils.test_rule`.
- Ensure supported primal types and their tangent types are exercised against the relevant rules for compatibility and composability.
- Mooncake has a debug mode which is useful for testing malformed rules and diagnosing
  rule failures; see `docs/src/utilities/debug_mode.md`.
- Bug fixes in rules, the interpreter, or compiler interop should ideally land with a
  focused regression test.
- If a fix depends on compiler or world-age behaviour, isolate it and test it directly.
- Be careful with `friendly_tangents` for structured wrappers such as `Symmetric` and
  `Hermitian`: the displayed tangent can be misleading even when the underlying tangent
  data is correct.
- `src/test_resources.jl` is shared test infrastructure, not dead code. It feeds broad
  interpreter/rule tests indirectly via `TestResources.generate_test_functions()`, so do
  not judge it by one-file-one-test symmetry.
- Typical command from the repo root:

```bash
julia --project=. -e 'import Pkg; Pkg.test(; test_args=ARGS)' -- rules/random
```

- Use `TestEnv.jl` to activate the package test environment when you need test-only dependencies:

```bash
TEST_GROUP=rules/random julia --project=. -e 'using TestEnv; TestEnv.activate(); include("test/runtests.jl")'
```

- Extension and integration tests should generally be run from their own
  files/environments under `test/ext/` and `test/integration_testing/`. These are part
  of the package contract, not optional extras, so changes to weakdeps/extensions often
  need updates there even if core tests still pass.

## Documentation

- `docs/make.jl` defines the Documenter build and navigation structure.
- Main docs sections include top-level user pages such as `index.md`, `tutorial.md`, and `interface.md`.
- Known unsupported or incomplete behaviour is documented in `docs/src/known_limitations.md`.
- Conceptual material lives under `docs/src/understanding_mooncake/`.
- Utility docs live under `docs/src/utilities/`.
- Internal and contributor material lives under `docs/src/developer_documentation/`.
- For defining or adapting differentiation rules, start with `docs/src/utilities/defining_rules.md`.
- For complex array-like rules, see the `Canonicalising Tangent Types` section in `docs/src/utilities/defining_rules.md` for `arrayify`/`matrixify` guidance.
- For recursive types or custom tangent implementations, start with `docs/src/developer_documentation/custom_tangent_type.md`.
