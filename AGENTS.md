# AGENTS.md

## Purpose

Mooncake.jl is a Julia-first automatic differentiation package focused on:

- broad language coverage, especially mutation and dynamic control flow
- correctness and testability before aggressive optimization
- uniquely typed tangents for robust rule composition and testing
- performance via hand-written low-level `rrule!!` / `frule!!`, strict tangent and cotangent types, and cached prepare/run APIs

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

- Keep changes aligned with the existing source/test symmetry: tests for `src/.../foo.jl` usually live at `test/.../foo.jl`.
- Put shared test setup in `test/front_matter.jl`; test-group dispatch lives in `test/runtests.jl`.
- For complex rules, especially array-heavy rules, prefer canonicalising inputs at the rule boundary with utilities such as `arrayify` rather than proliferating specialized methods.
- Mooncake provides helpers for importing rules from ChainRules via `@from_rrule` / `@from_chainrules`, but use them conservatively. In practice, keep this to scalar and array-like cases where the tangent conversions are robust.
- If you change public APIs, developer tooling, or core internals, update docs under `docs/src/` when needed.
- Prefer targeted changes over broad refactors unless the task explicitly requires restructuring.

## Testing

- Run focused test groups during development instead of the full suite when possible.
- For new differentiation rules, prefer testing them with `Mooncake.TestUtils.test_rule`.
- Typical command from the repo root:

```bash
julia --project=. -e 'import Pkg; Pkg.test(; test_args=ARGS)' -- rules/random
```

- Extension and integration tests should generally be run from their own files/environments under `test/ext/` and `test/integration_testing/`.

## Documentation

- `docs/make.jl` defines the Documenter build and navigation structure.
- Main docs sections include top-level user pages such as `index.md`, `tutorial.md`, and `interface.md`.
- Conceptual material lives under `docs/src/understanding_mooncake/`.
- Utility docs live under `docs/src/utilities/`.
- Internal and contributor material lives under `docs/src/developer_documentation/`.
- For defining or adapting differentiation rules, start with `docs/src/utilities/defining_rules.md`.
- For complex array-like rules, see the `Canonicalising Tangent Types` section in `docs/src/utilities/defining_rules.md` for `arrayify`/`matrixify` guidance.
- For recursive types or custom tangent implementations, start with `docs/src/developer_documentation/custom_tangent_type.md`.
