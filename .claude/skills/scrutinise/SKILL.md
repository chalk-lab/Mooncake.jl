---
name: scrutinise
description: Scrutinise newly added or changed code on the current branch against main. Checks new types, methods, changed signatures, overloads, and helpers for necessity, correctness, clarity, consistency, robustness, and minimality. Reviews new tests for gap coverage, overlap with existing tests, minimality, and use of established testing patterns. Invoke with /scrutinise.
tools: Bash, Glob, Grep, Read, Edit, Write
---

# Scrutinise

Review all changes on the current branch relative to `main`. Cover source and tests separately, then simplify.

## Step 1: Gather the diff

```bash
git diff main...HEAD --name-only
git diff main...HEAD -- src/ ext/ test/
```

Read changed files in full before commenting.

## Step 2: Source

For every new type, method, changed signature, or overload:

- **Necessary?** Does existing infrastructure (`@zero_derivative`, `@from_rrule`, broader signatures) already cover this? Could an overload be eliminated by broadening an existing one?
- **Correct?** Tangent/cotangent types consistent with `tangent_type`? `@is_primitive` declared? For `rrule!!`: pullback restores mutations, aliasing handled. For `frule!!`: dual propagation correct, removable singularities handled.
- **Clear and consistent?** Names and structure match the surrounding file and `src/rules/`. `NoTangent`/`ZeroTangent` used correctly. Names should be Julian; don't redundantly encode input types (scalar, vector, array, complex) that the signature already makes clear, but prefixes/suffixes for disambiguation or clarity are fine.
- **Audit globals.** List every new global variable, method, and type. State the purpose of each. Evaluate whether each is truly needed or could be inlined, merged, or removed.
- **Robust?** Edge cases (empty arrays, zero-size structs, complex types) handled or explicitly excluded. Fails loudly on unsupported inputs.
- **Minimal?** No dead branches, unused arguments, or speculative generalisations.

For every new helper: does it genuinely aid readability or reduce duplication, or can it be inlined? Does it belong in `src/utils.jl` or is it rule-local?

## Step 3: Tests

For every new or changed test:

- **Real gap?** Would removing it leave a regression undetected, or is it duplicating interpreter-level coverage via `TestResources.generate_test_functions()`?
- **No overlap?** Check the corresponding test file and `test/front_matter.jl` for existing tests on the same rule/type.
- **Minimal?** Smallest example that exercises the gap; no redundant argument combinations.
- **Right pattern?** Rules → `test_rule`. Tangents → `test_tangent` / `test_tangent_type_and_tglob_type_agree`. Duals → `test_dual` / `test_fdata` / `test_rdata`. Allocations → `count_allocs`. Malformed rules → `DebugMode`. Flag any test reimplementing logic already in the test utilities.

## Step 4: Output

Findings grouped by file, labelled:

- **Unnecessary** / **Incorrect** / **Unclear** / **Inconsistent** / **Non-minimal** / **Fragile**
- **Test: redundant** / **Test: missing pattern** / **Test: weak gap**

No issues in a section → write "No issues." Do not suggest additions beyond what the diff introduces.

## Step 5: Simplify

Invoke the `simplify` skill to apply code-quality and reuse fixes to the changed files.
