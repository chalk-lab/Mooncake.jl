# Advanced Debugging

This guide covers debugging techniques for issues that go beyond Mooncake's
rule system -- problems at the boundary between Mooncake's generated code and
Julia's compiler, runtime, or type system.

For rule-level debugging (wrong tangent types, segfaults), see
[Debug Mode](@ref) and [Debugging and MWEs](@ref).

## IR inspection

Mooncake's AD pipeline transforms IR through several stages. The `SkillUtils`
module in `src/skill_utils.jl` provides IR inspection utilities to view each
stage and diff consecutive transformations. These are
internal developer tools accessible via the `Mooncake.SkillUtils` prefix.

```@setup advanced_debugging
using Mooncake: Mooncake
using Mooncake.SkillUtils: inspect_ir, show_ir, show_stage, show_diff, show_world_info
```

```@example advanced_debugging
# Inspect all stages of the reverse-mode pipeline
ins = inspect_ir(sin, 1.0)
show_ir(ins)                              # all stages
```

```@example advanced_debugging
show_stage(ins, :raw)                     # one stage
```

```@example advanced_debugging
show_diff(ins; from=:raw, to=:normalized) # diff between stages
```

```@example advanced_debugging
# Forward mode
ins = inspect_ir(sin, 1.0; mode=:forward)

# World age info (useful for debugging stale code)
show_world_info(ins)
```

### Reverse mode stages

`:raw` → `:normalized` → `:bbcode` → `:fwd_ir` / `:rvs_ir` → `:optimized_fwd` / `:optimized_rvs`

### Forward mode stages

`:raw` → `:normalized` → `:bbcode` → `:dual_ir` → `:optimized`

When something looks wrong in generated code, diff consecutive stages to find
which transformation introduced the issue.

## Allocations

Unexpected allocations are the most common performance issue. Diagnostic
approach:

1. **`@code_typed optimize=true`** -- check return types are concrete (no
   `Union` or `Any`).
2. **`@code_llvm debuginfo=:none`** -- search for `gc_pool_alloc`,
   `gc_alloc_obj`, `jl_box_*`, or `jl_apply_generic`.
3. **`Base.specializations`** -- verify whether the runtime MethodInstance
   matches the concrete types you expect (see warning below).

!!! warning
    `@code_typed f(args...)` creates a fresh, fully-concrete specialization on
    demand. If the issue is specialization widening, it will show optimized code
    while the runtime path uses a widened MethodInstance. Cross-check with
    `Base.specializations(method)`.

## World age

World age issues arise when compiled code references methods from a different
world than expected. Symptoms include `MethodError` at runtime or stale
compiled code not picking up new method definitions.

In Mooncake, this most commonly affects:

- **`DerivedRule` / `DerivedFRule`**: rules compiled at one world may become
  stale if methods they depend on are redefined.
- **`LazyDerivedRule` / `DynamicDerivedRule`**: these handle world age
  transitions by recompiling on demand.

To debug, inspect the world age of generated code:

```@example advanced_debugging
ins = inspect_ir(sin, 1.0)
show_world_info(ins)
```

This reports the world at which each stage was compiled and flags mismatches.
