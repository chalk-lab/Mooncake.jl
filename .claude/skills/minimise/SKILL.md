---
name: minimise
description: Prune a bug fix or new tests down to the smallest correct diff through multiple elimination passes. Use before committing any fix or test addition.
---

# Minimise

The goal is to remove every line that is not strictly required for correctness,
then verify the result still passes the relevant tests.

## Process

Repeat the following until no further reductions are possible:

1. **Read the diff.** Run `git diff HEAD` (or `git diff --cached` if staged) and
   read every changed file in full.

2. **Challenge each change.** For every changed line ask:
   - Would removing this line cause a test to fail or a bug to reappear?
   - Is this a cleanup, rename, refactor, or comment that is not load-bearing?
   - For new tests: does an existing test already cover this behaviour?
     If so, drop the new test entirely.

3. **Remove non-essential changes.** Delete anything that does not answer
   "yes" to the first question above. Prefer shrinking an existing case over
   adding a new one.

4. **Run the minimal test group.** Use the smallest focused test group that
   exercises the changed code (see `test/runtests.jl` for group names).
   Confirm all tests pass before continuing.

5. **Repeat** from step 1 until a full pass produces no further removals.

## Heuristics

- A one-line fix is better than a five-line fix.
- A new test case added to an existing `@testset` is better than a new `@testset`.
- A new value constructor in `src/test_resources.jl` should be the minimum needed
  to instantiate the type under test; no extra fields or variants.
- Comments and blank lines added alongside a fix are not load-bearing; remove them
  unless they explain something non-obvious.
- Helper functions introduced solely for the fix are a red flag; inline them.

## When to stop

Stop when every remaining line answers "yes" to: *if I remove this, the targeted
bug reappears or the targeted test fails*. At that point report the final diff and
suggest committing.
