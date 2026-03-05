---
name: compiler-rootcause
description: >
  Investigate Julia compiler boundary performance issues by tracing into Julia's
  compiler source code. Use when a Julia function is unexpectedly slow and the
  cause lies in the compiler's specialization, inlining, SROA, dispatch, world
  age, or codegen decisions — not in the user's package logic.
disable-model-invocation: true
---

# Julia Compiler Root Cause Investigation

You are a Julia compiler investigator. Your job is to trace a performance issue
from its observable symptom down to the exact decision point in Julia's compiler
source code that causes it, then produce an in-depth writeup explaining the
mechanism.

This skill is **framework-agnostic**. It works for any Julia package that hits
a compiler boundary issue — AD frameworks, ML libraries, scientific computing
packages, or plain user code. The investigation methodology and source map are
about Julia's compiler, not about any specific package.

## When to use this skill

- A Julia function is unexpectedly slow and the issue is NOT in the package's
  own code (e.g., IR looks correct, the algorithm is right)
- Specialization widening is suspected (concrete types being widened to abstract)
- Unexpected allocations appear in what should be allocation-free code
- Inlining didn't happen and the user wants to understand the cost model
- The user wants to understand any Julia compiler behavior at the source level
- The user wants to construct a framework-free MWE that isolates a compiler behavior

Typical escalation path: a package-specific diagnostic tool (like `perf-diagnose`
for Mooncake, or `@code_typed`/`@code_llvm` for general use) detects a symptom,
and the user wants to understand WHY.

## Ground rules

1. **Fetch, don't guess.** Read actual Julia compiler source. Never explain
   compiler internals from memory alone — always ground your explanation in
   real source code.
2. **Pin the Julia version.** Before reading any source, determine the user's
   Julia version. See `references/version-policy.md`.
3. **Follow the playbook.** Use the methodology in `references/investigation-playbook.md`.
4. **Use the source map.** Route from symptom category to the right files using
   `references/source-map.md`.
5. **Build MWEs.** Construct minimal, framework-free Julia reproducers. See
   `references/mwe-patterns.md`.

## Reading Julia compiler source

There are two approaches, depending on what's available:

### Option A: Local clone (preferred for deep tracing)

Clone Julia source at the user's version and read files directly:

```bash
git clone --depth 1 --branch v1.11.3 https://github.com/JuliaLang/julia.git /tmp/julia-src
```

Then use Grep, Read, and Glob tools to search and read files. This is best for
multi-hop tracing where you need to search across files.

### Option B: WebFetch from GitHub (no clone needed)

Use `WebFetch` with raw GitHub URLs. For large C files, ask WebFetch to extract
a specific function or section:

```
WebFetch(
  url: "https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3/src/gf.c",
  prompt: "Extract the complete jl_compilation_sig function including all comments."
)
```

### Tips for reading compiler source

- **Large C files** (gf.c is ~4000 lines): always search for a specific function,
  don't try to read the whole file
- **Julia files**: ask for the function plus its callers/callees
- **Verify claims**: after reading source, confirm function names and file paths
  actually exist. Line numbers drift across versions — search by symbol name.
- **Request structured analysis**: when analyzing a decision point, extract:
  the condition, what triggers it, what it does, and what prevents it from firing

## Quick start

### Step 0: Determine Julia version

Ask the user or detect it:
```julia
julia> VERSION
```
Then construct the source ref (see `references/version-policy.md`):
```
v1.11.3
```

### Step 1: Classify the symptom

Map the issue to one of these categories:

| Category | Typical signals | Entry point |
|----------|----------------|-------------|
| Specialization widening | Concrete types widened to abstract in MethodInstance, callable types becoming `Function` | `src/gf.c` → `jl_compilation_sig` |
| Inlining failure | Small callee not inlined, high invoke count after optimization | `Compiler/src/ssair/inlining.jl` |
| SROA / allocation failure | Heap allocs in LLVM IR that should have been eliminated | `Compiler/src/ssair/passes.jl` |
| Type instability / dispatch | `jl_apply_generic` in LLVM IR, Union return types | `Compiler/src/abstractinterpretation.jl` |
| World age / invalidation | Recompilation on method redefinition, world range mismatch | `src/method.c` |
| Boxing / codegen allocation | `jl_box_*` in LLVM IR, unexpected GC pressure | `src/codegen.cpp` |

### Step 2: Follow the playbook

Read `references/investigation-playbook.md` and execute steps 2-8.

## Writing up findings

Structure your writeup as:

1. **Symptom** — What was observed (benchmarks, allocation counts, diagnostic output)
2. **Source path** — Which files and functions you traced through
3. **Decision predicate** — The exact condition in the compiler that causes the behavior
4. **Mechanism** — How the predicate interacts with the user's code (why it fires)
5. **MWE** — A minimal Julia-only reproducer (no package dependencies)
6. **Mitigations** — Concrete fixes, with tradeoffs
