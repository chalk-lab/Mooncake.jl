---
name: ir-inspect
description: Inspect Mooncake.jl IR transformations at each stage of the AD pipeline. Use when the user wants to view, debug, or understand IR at any compilation stage.
disable-model-invocation: true
---

# Mooncake IR Inspection

You are helping the user inspect IR (Intermediate Representation) transformations in Mooncake.jl's automatic differentiation pipeline.

## Setup

The IR inspection script lives at `.claude/skills/ir-inspect/scripts/ir_inspect.jl`.

Start a Julia session (or reuse an existing one) and load the script:

```julia
using Revise
includet(".claude/skills/ir-inspect/scripts/ir_inspect.jl")
```

## Gathering user intent

Ask the user what they want to inspect. Offer these choices:

1. **Function to inspect** — ask which function and arguments (e.g. `sin, 1.0` or a custom function)
2. **Mode** — reverse mode (default) or forward mode
3. **What to view**:
   - All stages at once
   - A specific stage
   - A diff between two stages
   - World age info
   - CFG as DOT/Graphviz

Do not assume — ask the user to pick.

## Pipeline stages

### Reverse mode stages (default)
| Stage | Symbol | Description |
|-------|--------|-------------|
| Raw IR | `:raw` | Original IR from Julia's compiler |
| Normalized | `:normalized` | After Mooncake's normalization passes |
| BBCode | `:bbcode` | BBCode representation with stable IDs |
| Forward IR | `:fwd_ir` | Generated forward-pass IR |
| Reverse IR | `:rvs_ir` | Generated pullback (reverse-pass) IR |
| Optimized Forward | `:optimized_fwd` | Forward pass after optimization |
| Optimized Reverse | `:optimized_rvs` | Pullback after optimization |

### Forward mode stages
| Stage | Symbol | Description |
|-------|--------|-------------|
| Raw IR | `:raw` | Original IR from Julia's compiler |
| Normalized | `:normalized` | After Mooncake's normalization passes |
| BBCode | `:bbcode` | BBCode representation with stable IDs |
| Dual IR | `:dual_ir` | Generated dual-number IR |
| Optimized | `:optimized` | After optimization passes |

## Commands reference

```julia
# Full inspection
ins = inspect_ir(f, args...; mode=:reverse)  # or mode=:forward

# View stages
show_ir(ins)                          # all stages
show_stage(ins, :raw)                 # one stage
show_stage(ins, :normalized)

# Diffs between stages
show_diff(ins; from=:raw, to=:normalized)
show_all_diffs(ins)

# World age debugging
show_world_info(ins)

# CFG as DOT string
println(cfg_dot(ins, :bbcode))

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

## How to present results

- Run the Julia commands via Bash and capture output.
- Present the IR text in fenced code blocks with context about what each stage represents.
- When showing diffs, highlight what changed and explain why the transformation matters.
- If the user asks for a CFG, return the DOT string and suggest how to render it (e.g. `dot -Tpng cfg.dot -o cfg.png` or paste into a Graphviz viewer).
- If errors occur, check that Mooncake is loaded and the function signature is valid.

## Limitations

This skill inspects Mooncake's **internal AD pipeline** — the IR it generates for forward/reverse passes. It does **not** detect performance issues at the Julia compiler boundary (specialization widening, SROA/allocation failures, inlining failures). If the IR looks correct but the function is still slow, use the **perf-diagnose** skill instead (`.claude/skills/perf-diagnose/`).
