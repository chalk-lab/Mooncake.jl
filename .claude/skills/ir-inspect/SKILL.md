---
name: ir-inspect
description: Inspect Mooncake.jl IR transformations at each stage of the AD pipeline. Use when the user wants to view, debug, or understand IR at any compilation stage.
disable-model-invocation: true
---

# Mooncake IR Inspection

You are helping the user inspect IR (Intermediate Representation) transformations in Mooncake.jl's automatic differentiation pipeline.

## Setup

The IR inspection functions are part of Mooncake.jl. Start a Julia session and load the package:

```julia
using Mooncake
```

All functions below are accessed via the `Mooncake` module (e.g. `Mooncake.inspect_ir`).

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
using Mooncake

# Full inspection
ins = Mooncake.inspect_ir(f, args...; mode=:reverse)  # or mode=:forward

# View stages
Mooncake.show_ir(ins)                          # all stages
Mooncake.show_stage(ins, :raw)                 # one stage

# Diffs between stages
Mooncake.show_diff(ins; from=:raw, to=:normalized)
Mooncake.show_all_diffs(ins)

# World age debugging
Mooncake.show_world_info(ins)

# CFG as DOT string
println(Mooncake.cfg_dot(ins, :bbcode))

# Write everything to files
Mooncake.write_ir(ins, "/tmp/ir_output")

# Shorthand helpers
ins = Mooncake.inspect_fwd(f, args...)         # forward mode
ins = Mooncake.inspect_rvs(f, args...)         # reverse mode
ins = Mooncake.quick_inspect(f, args...)       # inspect + display immediately

# Options
Mooncake.inspect_ir(f, args...;
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

This skill inspects Mooncake's **internal AD pipeline** — the IR it generates for forward/reverse passes. It does **not** detect performance issues at the Julia compiler boundary (specialization widening, SROA/allocation failures, inlining failures). If the IR looks correct but the function is still slow, use the **perf-diagnose** skill instead.
