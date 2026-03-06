# Version Policy

How to determine the correct Julia source version and construct GitHub URLs.

## Determining the Julia version

Ask the user or detect it in a Julia session:

```julia
julia> VERSION
v"1.11.3"

julia> Base.GIT_VERSION_INFO.commit_short
"d63adeda50"
```

## Constructing raw GitHub URLs

The base URL pattern:
```
https://raw.githubusercontent.com/JuliaLang/julia/{REF}/{PATH}
```

### Choosing `{REF}`

Use this priority order:

1. **Exact version tag:** `v1.11.3`
   ```
   https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3/src/gf.c
   ```
   This is the default. Most stable releases have a corresponding tag.

2. **Release branch:** `release-1.11`
   ```
   https://raw.githubusercontent.com/JuliaLang/julia/release-1.11/src/gf.c
   ```
   Use if the exact tag doesn't exist (e.g., for release candidates or nightlies).

3. **Exact commit:** `d63adeda50`
   ```
   https://raw.githubusercontent.com/JuliaLang/julia/d63adeda50/src/gf.c
   ```
   Use when the user reports a specific commit or when investigating whether a
   fix has landed.

4. **master branch:** `master`
   ```
   https://raw.githubusercontent.com/JuliaLang/julia/master/src/gf.c
   ```
   Use only for "is this fixed upstream?" checks. Always note that master may
   differ from the user's version.

### Common version → tag mapping

| Julia version | Tag | Notes |
|--------------|-----|-------|
| 1.10.x | `v1.10.x` | LTS |
| 1.11.x | `v1.11.x` | Previous stable |
| 1.12.x | `v1.12.x` | Current stable (2025-2026) |
| nightly | `master` | Unstable, changes daily |

### Important: Julia 1.12+ compiler restructuring

Starting with Julia 1.12, the compiler was extracted into a separate
`Compiler/` module. File paths changed:

| Before 1.12 | After 1.12 |
|-------------|------------|
| `base/compiler/abstractinterpretation.jl` | `Compiler/src/abstractinterpretation.jl` |
| `base/compiler/ssair/inlining.jl` | `Compiler/src/ssair/inlining.jl` |
| `base/compiler/ssair/passes.jl` | `Compiler/src/ssair/passes.jl` |
| `base/compiler/optimize.jl` | `Compiler/src/optimize.jl` |
| `base/compiler/typeinfer.jl` | `Compiler/src/typeinfer.jl` |
| `base/compiler/types.jl` | `Compiler/src/types.jl` |

C files (`src/gf.c`, `src/method.c`, etc.) did not move.

When the user is on Julia < 1.12, use `base/compiler/` paths.
When on Julia >= 1.12, use `Compiler/src/` paths.

### JuliaLowering (1.12+ only)

Julia 1.12 introduced `JuliaLowering/` as a subproject within the Julia tree.
This contains the new Julia-native lowering pass (replacing the Scheme-based
`src/julia-syntax.scm`). Files like `JuliaLowering/src/scope_analysis.jl` and
`JuliaLowering/src/eval.jl` are only available on Julia 1.12+.

On Julia < 1.12, the equivalent lowering logic is in `src/julia-syntax.scm`
(Scheme) and `src/ast.c`. Tracing `slotflags` and `called` on older versions
requires reading the Scheme lowering code, which is significantly harder.

### LLVM allocation function names

The exact names of allocation functions in LLVM IR change across Julia versions:
- `jl_gc_pool_alloc` (older)
- `ijl_gc_pool_alloc` (internal variant)
- `ijl_gc_small_alloc` (Julia 1.12+)
- `julia.gc_alloc_obj` (intrinsic)
- `julia.gc_alloc_bytes` (intrinsic)

Do not hardcode a single name. Search for patterns: `gc_pool_alloc`,
`gc_small_alloc`, `gc_alloc_obj`, `gc_alloc_bytes`.

## Recording provenance

Every root cause report should include the exact ref used:

```markdown
## References

- Julia version: v1.11.3
- Source ref: `v1.11.3` (tag)
- Files consulted:
  - https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3/src/gf.c
  - https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3/src/method.c
```

This ensures reproducibility — if someone revisits the analysis later, they
can fetch the exact same source.

## Verifying a tag exists

If unsure whether a tag exists, try fetching a known small file:
```
WebFetch(
  url: "https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3/VERSION",
  prompt: "What is the content of this file?"
)
```
If this returns a 404 or redirect, try the release branch instead.
