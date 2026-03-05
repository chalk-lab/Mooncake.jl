# Investigation Playbook

An 8-step methodology for tracing a Julia compiler boundary performance issue
from symptom to root cause. Each step produces a concrete artifact.

This playbook is framework-agnostic — it works for any Julia package or user
code that hits a compiler boundary issue.


## Step 1: Classify the symptom

**Input:** Diagnostic output (e.g., `@code_typed`, `@code_llvm`, allocation
counts, benchmark results), or a package-specific diagnostic tool's report.

**Action:** Map the issue to one or more categories from the source map. Start
with the category that has the strongest signal. Compiler boundary issues are
often **cross-category** — e.g., inference imprecision leads to an inlining
failure, which causes an SROA failure, which results in allocations. If the
initial category doesn't explain everything, revisit and check adjacent categories.

**Artifact:** A one-line classification, e.g.:
> "Specialization widening: `typeof(sum)` widened to `Function` in Vararg slot 4"

**Decision table:**

| Signal | Category |
|--------|----------|
| Concrete callable type widened to `Function` or `Any` in MethodInstance | Specialization widening |
| `Type{T}` widened to `Type` in MethodInstance | Specialization widening |
| Small callee survives optimization, invoke count stays high | Inlining failure |
| Heap allocs in `@code_llvm` but not expected from `@code_typed` | SROA failure |
| `jl_apply_generic` in `@code_llvm`, Union return types in `@code_typed` | Type instability |
| Recompilation on method redefinition, world range mismatch | World age |
| `jl_box_*` in `@code_llvm` without corresponding Julia-level allocation | Boxing |


## Step 2: Pin the Julia version

**Input:** The user's `VERSION` or environment.

**Action:** Construct the source reference. See `version-policy.md`.

**Artifact:** A version ref and base URL, e.g.:
```
REF=v1.11.3
BASE=https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3
```


## Step 3: Fast triage against documentation

**Action:** Before reading compiler source, check if this is a known, documented
pitfall.

**Fetch these based on category:**

| Category | Doc to check |
|----------|-------------|
| Specialization widening | `{BASE}/doc/src/manual/performance-tips.md` — search for "specializing" |
| Inlining | `{BASE}/doc/src/manual/performance-tips.md` — search for "@inline" |
| SROA / allocation | `{BASE}/doc/src/devdocs/EscapeAnalysis.md` |
| Type instability | `{BASE}/doc/src/manual/performance-tips.md` — search for "type stability" |
| World age | `{BASE}/doc/src/manual/methods.md` — search for "world age" |
| Boxing | `{BASE}/doc/src/manual/performance-tips.md` — search for "abstract containers" |

**Decision:** If docs fully explain the issue, summarize and stop. If not, proceed.


## Step 4: Fetch entry source files

**Action:** Look up the category in `source-map.md`. Read the primary files.

With a local clone, use Grep to find functions and Read to examine them.
With WebFetch, use raw GitHub URLs with targeted prompts:
```
WebFetch(
  url: "{BASE}/src/gf.c",
  prompt: "Extract the complete jl_compilation_sig function including all comments."
)
```

**Tips:**
- For large C files (gf.c is ~4000 lines), always ask for a specific function
- For Julia files, ask for the function plus its callers/callees
- Fetch 2-3 files max per round — you can always fetch more

**Artifact:** The relevant source code sections, with file paths noted.


## Step 5: Find the decision point

**Action:** In the fetched source, locate the exact predicate that controls the
behavior. This is the "decision point" — the `if` statement or condition that
determines whether widening/inlining/SROA happens.

**What to look for by category:**
- Specialization: `notcalled_func`, `nospecialize`, `very_general_type`, `iscalled`
- Inlining: cost comparison, effect requirements, `@noinline` propagation
- SROA: escape state check, `is_load_forwardable`
- Dispatch: `MethodMatchInfo` resolution, union splitting decisions

**Caution: `@code_typed` can be misleading.** Calling `@code_typed f(args...)`
creates a fresh, fully-concrete specialization on demand — one that runtime
dispatch may never use. If the issue is specialization widening, `@code_typed`
will show perfectly optimized code while the actual runtime path uses a widened
`MethodInstance`. Always cross-check with `Base.specializations(method)` to see
what the runtime actually compiled. See `mwe-patterns.md` Pattern 1.

**Artifact:** The exact condition (quote the source), e.g.:
```c
int notcalled_func = (i_arg > 0 && i_arg <= 8 &&
    !(definition->called & (1 << (i_arg - 1))) &&
    jl_subtype(elt, (jl_value_t*)jl_function_type));
```


## Step 6: Trace upstream and downstream

**Action:** From the decision point, trace in both directions until you reach
a boundary you can explain to the user:
- **Upstream:** What sets the inputs to this predicate? Follow the chain as far
  as needed — often the root cause is 2-3 hops upstream (e.g., lowering → slotflags
  → called bitmask → compilation sig).
- **Downstream:** What does the decision affect? Trace through to the observable
  consequence (widened MethodInstance, failed inlining, heap allocation).

Use Grep to search across the source tree for where inputs are set, e.g.:
```
Grep(pattern: "->called", path: "/tmp/julia-src/src/method.c")
```

**Artifact:** A chain showing: input source → decision point → output effect, e.g.:
```
scope_analysis.jl:685 (sets is_called when slot appears in call position)
  → eval.jl:238 (encodes to slotflags bit 6)
    → method.c:938-942 (reads slotflags → definition->called bitmask)
      → gf.c:1188 (i_arg collapses vararg positions to last parameter)
        → gf.c:1305-1314 (notcalled_func fires → widens to Function)
          → widened MethodInstance used at runtime
```


## Step 7: Construct MWE

**Action:** Build a minimal Julia-only reproducer that preserves the same compiler
predicate firing. See `mwe-patterns.md` for templates.

**Requirements for a good MWE:**
1. No package dependencies — pure Julia (maybe Chairmarks for benchmarking)
2. An A/B pair: one case where the predicate fires, one where it doesn't
3. Observable difference (timing, allocation count, `Base.specializations` output)
4. Removing any remaining line breaks the reproduction

**Artifact:** A self-contained Julia script with comments explaining what to observe.


## Step 8: Write the root cause report

**Action:** Synthesize steps 1-7 into a structured writeup.

**Template:**

```markdown
## Symptom

[What was observed — benchmarks, diagnostic output, user report]

## Classification

[Category and one-line summary]

## Source path

[The chain from Step 6, with file:line references and Julia version]

## Decision predicate

[The exact condition from Step 5, quoted from source]

## Mechanism

[How the predicate interacts with the user's code — why it fires in this case.
This is where you explain the gap between what the user expected and what the
compiler actually does.]

## Minimal reproducer

[The MWE from Step 7]

## Mitigations

[Concrete fixes, with tradeoffs for each:]
- Fix A: [description] — Pros: ... Cons: ...
- Fix B: [description] — Pros: ... Cons: ...

[If comparing multiple fixes, benchmark each one multiple times across separate
sessions. Initial measurements can be misleading — nanosecond-level differences
between mitigation strategies are often noise. Only report a speed difference
between fixes if it is reproducible across multiple independent runs.]

## References

- Julia version: [version]
- Julia source ref: [tag/commit]
- Files consulted: [list with URLs]
- Julia docs: [relevant doc sections]
- Related issues: [if any]
```
