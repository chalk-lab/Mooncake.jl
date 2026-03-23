# [Version Handling for Compiler Investigation](@id compiler_version_policy)

When investigating Julia compiler behavior, it is important to pin the exact
Julia version so that source references are reproducible.

## Determining the Julia version

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

Use this priority order for `{REF}`:

1. **Exact version tag:** `v1.11.3` (preferred, most stable)
2. **Release branch:** `release-1.11` (for RCs or nightlies without a tag)
3. **Exact commit:** `d63adeda50` (for investigating specific fixes)
4. **`master`:** only for "is this fixed upstream?" checks

## Julia 1.12+ compiler restructuring

Starting with Julia 1.12, the compiler was extracted into a separate `Compiler/`
module. File paths changed:

| Before 1.12 | After 1.12 |
|-------------|------------|
| `base/compiler/abstractinterpretation.jl` | `Compiler/src/abstractinterpretation.jl` |
| `base/compiler/ssair/inlining.jl` | `Compiler/src/ssair/inlining.jl` |
| `base/compiler/ssair/passes.jl` | `Compiler/src/ssair/passes.jl` |
| `base/compiler/optimize.jl` | `Compiler/src/optimize.jl` |
| `base/compiler/typeinfer.jl` | `Compiler/src/typeinfer.jl` |
| `base/compiler/types.jl` | `Compiler/src/types.jl` |

C files (`src/gf.c`, `src/method.c`, etc.) did not move.

## JuliaLowering (1.12+ only)

Julia 1.12 introduced `JuliaLowering/` as a subproject containing the new
Julia-native lowering pass (replacing the Scheme-based `src/julia-syntax.scm`).
On Julia < 1.12, the equivalent lowering logic is in `src/julia-syntax.scm`
(Scheme) and `src/ast.c`.

## LLVM allocation function names

The exact names of allocation functions in LLVM IR vary across Julia versions:

- `jl_gc_pool_alloc` (older versions)
- `ijl_gc_pool_alloc` (internal variant)
- `ijl_gc_small_alloc` (Julia 1.12+)
- `julia.gc_alloc_obj` (intrinsic)
- `julia.gc_alloc_bytes` (intrinsic)

Do not hardcode a single name. Search for patterns: `gc_pool_alloc`,
`gc_small_alloc`, `gc_alloc_obj`, `gc_alloc_bytes`.

## Recording provenance

Every root cause report should include the exact ref used:

```
## References

- Julia version: v1.11.3
- Source ref: v1.11.3 (tag)
- Files consulted:
  - https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3/src/gf.c
  - https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3/src/method.c
```
