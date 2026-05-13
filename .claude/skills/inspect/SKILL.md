---
name: inspect
description: Inspect the AD pipeline IR for a Julia function at each Mooncake compilation stage.
---

# Inspect

Inspect IR transformations in Mooncake's AD pipeline for a given function.

## Setup

```julia
using Mooncake, Mooncake.SkillUtils
```

## Gathering user intent

Ask the user:

1. **Function and arguments** — e.g. `sin, 1.0` or a custom function
2. **Mode** — reverse (default) or forward
3. **What to view** — all stages, a specific stage, a diff between two stages, or world age info

Do not assume — ask the user to pick.

## Pipeline stages

### Reverse mode (default)
| Stage | Symbol | Description |
|-------|--------|-------------|
| Raw IR | `:raw` | optimised, type-inferred SSAIR from Julia's compiler |
| Normalised | `:normalized` | after Mooncake's normalisation passes |
| BBCode | `:bbcode` | BBCode representation with stable IDs |
| Forward IR | `:fwd_ir` | generated forward-pass IR |
| Reverse IR | `:rvs_ir` | generated pullback IR |
| Optimised Forward | `:optimized_fwd` | forward pass after optimisation |
| Optimised Reverse | `:optimized_rvs` | pullback after optimisation |

### Forward mode
| Stage | Symbol | Description |
|-------|--------|-------------|
| Raw IR | `:raw` | optimised, type-inferred SSAIR from Julia's compiler |
| Normalised | `:normalized` | after Mooncake's normalisation passes |
| BBCode | `:bbcode` | inspection-only — forward mode does not use BBCode internally |
| Dual IR | `:dual_ir` | generated dual-number IR |
| Optimised | `:optimized` | after optimisation passes |

## Commands

```julia
# Full inspection
ins = inspect_ir(f, args...; mode=:reverse)  # or mode=:forward

# View stages
show_ir(ins)                          # all stages
show_stage(ins, :raw)                 # one stage

# Diffs between stages
show_diff(ins; from=:raw, to=:normalized)
show_all_diffs(ins)

# World age debugging
show_world_info(ins)

# Write everything to files
write_ir(ins, "/tmp/ir_output")

# Shorthand helpers
ins = inspect_fwd(f, args...)         # forward mode
ins = inspect_rvs(f, args...)         # reverse mode
ins = quick_inspect(f, args...)       # inspect + display immediately

# Options
inspect_ir(f, args...;
    mode       = :reverse,
    optimize   = true,
    do_inline  = true,
    debug_mode = false,
)
```

## Presenting results

- Run commands via Bash and present IR in fenced code blocks.
- When showing diffs, explain what changed and why the transformation matters.
- If errors occur, check that Mooncake is loaded and the function signature is valid.

## Limitations

Inspects Mooncake's internal AD pipeline only. For allocation, world-age, or compiler-boundary debugging, see `docs/src/developer_documentation/advanced_debugging.md`.
