# Primitives and Overlays

[`Mooncake.@is_primitive`](@ref) and [`Mooncake.@mooncake_overlay`](@ref) are the two main ways a user can intervene in how Mooncake differentiates a function. They act at different layers:

- `@mooncake_overlay` changes **method lookup under Mooncake's interpreter**. It gives Mooncake a different primal body to see.
- `@is_primitive` changes **what Mooncake differentiates**. It declares a boundary where Mooncake stops differentiating the primal body and uses a hand-written rule instead.

This page explains each macro on its own terms, describes how Mooncake uses them during type inference, and then derives which compositions are supported and which are not. Implementation-level details are tucked into collapsible blocks for readers who want to follow the mechanism.

## Primitives

To differentiate a function, Mooncake runs a *compilation step*: it walks the function's inferred source IR statement by statement, rewriting each into the forward- and reverse-pass code that will execute at differentiation time. The output is a derived rule — a callable that runs the program forward while tracking the data needed to compute the gradient on the reverse pass.

[`Mooncake.@is_primitive`](@ref) declares a function as a stopping point for that rewriting. At a matching call site:

- The body is not walked: Mooncake leaves the call statement in the transformed IR as-is.
- At runtime, the registered [`Mooncake.rrule!!`](@ref) (or [`Mooncake.frule!!`](@ref)) is dispatched in place of the primal call.

The rule, not the body, produces the value at this call site; what the surrounding AD code sees is a `CoDual` whose primal type matches the primitive's return type. Examples on this page use reverse mode (`rrule!!`) for concreteness; the same machinery applies to forward mode (`frule!!`).

!!! details "Mechanism"
    Mooncake's `AbstractInterpreter` override of `abstract_call_gf_by_type` (in [`src/interpreter/abstract_interpretation.jl`](https://github.com/chalk-lab/Mooncake.jl/blob/main/src/interpreter/abstract_interpretation.jl)) checks each call site against the primitive table via `any_matches_primitive`. When a match is detected, the resulting `CallMeta` is wrapped in a `NoInlineCallInfo`, which Mooncake's inlining policy (`inlining_policy` pre-1.12, `src_inlining_policy` from 1.12) then refuses to inline. The primitive call therefore survives inlining, and the rule-dispatch code is emitted at that statement instead of inlined primal code.

## Overlays

See [Simplifying Code via Overlays](@ref) in the Defining Rules guide for the `@mooncake_overlay` docstring and a user-facing introduction; this section covers the inference-level picture.

`@mooncake_overlay` registers an additional method for a function in a private method table, `Mooncake.mooncake_method_table`. Only Mooncake's interpreter consults this table; plain Julia dispatch and `Core.Compiler.NativeInterpreter` do not.

For example:

```julia
# Imagine `slow_or_unsupported` hits something Mooncake doesn't handle
# (a foreign call, a `try`/`catch`, ...) or handles only inefficiently.
f(x) = slow_or_unsupported(x)

# An AD-friendly body that returns the same value and the same type.
Mooncake.@mooncake_overlay f(x) = ad_friendly_alternative(x)
```

When `MooncakeInterpreter` infers a call to `f`, method lookup goes through `OverlayMethodTable` and resolves to the overlay (`ad_friendly_alternative(x)`). Plain Julia dispatch is unchanged — code calling `f` outside `MooncakeInterpreter` still executes the original (`slow_or_unsupported(x)`). Inside `MooncakeInterpreter`, the *primal* of `f` is the overlay body whenever a matching overlay exists, so the inferred source IR — and any AD rewriting subsequently applied to it — sees the overlay body, not the original.

The intended use is to substitute a body Mooncake can't differentiate (e.g. a foreign call, or a construct that hits a known limitation) — or one Mooncake can differentiate but only inefficiently — with an equivalent body that AD handles better. Mooncake doesn't verify equivalence; the author is responsible for ensuring the overlay returns the same value and the same type as the original, so that differentiating the overlay yields a derivative of the original semantics.

!!! details "Mechanism"
    `mooncake_method_table` is a `Core.MethodTable` created by [`Base.Experimental.@MethodTable`](https://docs.julialang.org/en/v1/base/base/#Base.Experimental.@MethodTable). `@mooncake_overlay` is essentially a thin wrapper around [`Base.Experimental.@overlay`](https://docs.julialang.org/en/v1/base/base/#Base.Experimental.@overlay): both rewrite the method definition's call head into an `Expr(:overlay, mt, name)`, which the frontend registers into `mt` (visible as the resulting `Method`'s `external_mt` field) rather than the global method table.

    The lookup that makes overlays "win" is `Core.Compiler.OverlayMethodTable` (defined in [`Compiler/src/methodtable.jl`](https://github.com/JuliaLang/julia/blob/master/Compiler/src/methodtable.jl)). `CC.method_table(::MooncakeInterpreter)` returns one constructed over `mooncake_method_table`, and during inference every method lookup goes through it: if `mooncake_method_table` has a matching method that fully covers the signature, it wins; otherwise lookup falls back to the global table.

## Type inference

Mooncake's IR transformation is driven by inferred type information. Three places matter, broadly in this order:

1. **Source-IR inference.** The function being differentiated is inferred via `MooncakeInterpreter`. This produces the IR that the AD transformation rewrites.
2. **Per-call `CallMeta`.** At each call statement during the source-IR walk, Mooncake needs the return type, effects, and call info. Julia's `abstract_call_gf_by_type` (in `Compiler/src/abstractinterpretation.jl`) is the per-call-site inference entry point that produces this; Mooncake overrides it for `MooncakeInterpreter` to insert primitive/overlay handling before the recursive per-match step. See [Inference at primitive call sites](@ref) for the primitive case.
3. **Rule-type inference.** Later, during AD IR construction, Mooncake calls `Core.Compiler.return_type` with the default interpreter — for example when emitting a `pullback_type` lookup — to learn the type the rule itself returns.

The key asymmetry to internalise: **Mooncake's source-function inference (1) is overlay-aware via `OverlayMethodTable`; `NativeInterpreter`, used at primitive boundaries during (2), is not.**

### Inference at primitive call sites

At every call site in the source IR, Mooncake needs a return type — downstream code is typed against it. At a primitive call site, the surrounding code still wants the primal's return type; the rule is an *implementation* keyed to that type, not a *source* for it. So inference asks the primal what it returns. The rule itself is not consulted at this stage.

The natural choice — recursing into the body with `MooncakeInterpreter` — is expensive and unnecessary. It is expensive because `MooncakeInterpreter` re-fires its primitive/overlay check at every nested call site and uses its own inference cache separate from Julia's global one, so each function Mooncake differentiates triggers a fresh walk of its transitive call tree (see [PR #1115](https://github.com/chalk-lab/Mooncake.jl/pull/1115) for a SciML-shaped case where this explodes into a silent compile-time hang). It is unnecessary because the body isn't being rewritten into AD code, only inferred for its return type — any interpreter that produces a correct `CallMeta` is sufficient. Mooncake therefore delegates to `NativeInterpreter` at primitive boundaries, bounding the recursion at each one.

The wrinkle: `NativeInterpreter` is overlay-blind. Any overlay that would affect the primitive — directly, or indirectly via a call inside its body — is invisible to inference at the primitive boundary.

## Composition

The supported and unsupported combinations follow directly from the layers each macro touches.

### Overlay only, no primitive

Mooncake's interpreter sees the overlay's body in place of the original. AD differentiates the overlay body. Fully supported; this is the canonical use of `@mooncake_overlay`.

### Primitive only, no overlay

The rule replaces the primal at runtime. Inference at the call site asks `NativeInterpreter` for the *original* body's `CallMeta`. Fully supported, on the standard contract that the rule's primal return type matches the original's.

### Direct overlay on a primitive signature

This is not a recommended pattern — choose one of `@mooncake_overlay` or `@is_primitive` on a given signature, not both. Mooncake currently supports it as a special case ([#1170](https://github.com/chalk-lab/Mooncake.jl/pull/1170)): when both apply, the rule still fires at runtime, and Mooncake routes call-site inference through the overlay-aware default path so the inferred return type matches what the rule actually returns rather than what the original primal would have returned. When the overlay's return type happens to equal the original's, the routing change is harmless; when it differs, it is what keeps inference and the rule coherent.

!!! details "Mechanism"
    `any_matches_overlay` (in [`src/interpreter/abstract_interpretation.jl`](https://github.com/chalk-lab/Mooncake.jl/blob/main/src/interpreter/abstract_interpretation.jl)) walks the applicable methods returned by `find_method_matches` and checks `method.external_mt === mooncake_method_table` on each. The check is per-method and signature-aware: an overlay registered for `f(::Float64)` is invisible at a call site that dispatches to a different method (e.g. `f(::Int)`). It also does not try to detect whether the overlay actually *changes* the return type — any applicable overlay triggers the same path, even when the overlay-aware and overlay-blind inference paths would agree. When the check returns true, `abstract_call_gf_by_type` takes the `@invoke` branch — i.e. it defers to the default `abstract_call_gf_by_type` *with `MooncakeInterpreter` still as the interpreter*, so method lookup inside that call still goes through `OverlayMethodTable` and resolves to the overlay's body. The `NativeInterpreter` fast path is reserved for primitives whose applicable methods have no overlay.

### Primitive called from inside an overlay's body — supported

Unlike the previous section, the two macros sit on *different* functions here: an overlay replaces one function's body so that it bottoms out on a *separate* function carrying a hand-written rule. This is the ordinary, supported flow — Mooncake walks the overlay body for AD, and the primitive call inside it is handled by the same machinery as any other primitive call (primitive detection, `NativeInterpreter` for the `CallMeta`, rule dispatch at runtime). No special arrangement is needed; this is in fact the most common reason to write an overlay.

```julia
my_primitive(x::Float64) = 2x
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(my_primitive),Float64}
function Mooncake.rrule!!(::CoDual{typeof(my_primitive)}, x::CoDual{Float64})
    pb(dy) = NoRData(), 2dy
    return Mooncake.zero_fcodual(2 * Mooncake.primal(x)), pb
end

# `original_f` has some body Mooncake handles awkwardly. Overlay redirects through
# `my_primitive`, whose rule supplies the derivative.
original_f(x::Float64) = unsupported_or_expensive(x)
Mooncake.@mooncake_overlay original_f(x::Float64) = my_primitive(x)
```

Differentiating any caller of `original_f` walks the overlay's body, hits `my_primitive`, dispatches its `rrule!!`, and computes the gradient via the rule's adjoint. The standard contract applies as anywhere else: the primitive's rule must return a `CoDual` whose primal type matches Mooncake's inferred return type at the call site.

### Overlay on a non-primitive called from inside a primitive's body — not supported

If a primitive's return type depends on an overlay applied to a function it calls internally, AD silently uses the un-overlaid return type. The failure surfaces in one of two shapes, depending on whether the affected return value is a singleton:

- **Singleton return (the SciMLBase `Originator` shape in [#1169](https://github.com/chalk-lab/Mooncake.jl/issues/1169)).** Inference produces a `CC.Const(value)` from the *original* body, and Julia constant-folds the primitive call to that literal *before* AD construction sees it. No `rrule!!` call is emitted, no typeassert fires, and downstream code is compiled against the wrong singleton type — a silent wrong gradient.
- **Non-singleton return.** No const-folding, but inferred return type and rule output still disagree: downstream dispatch is keyed to the inferred (un-overlaid) type while the rule returns the overlaid type. This is the case the runtime `Core.typeassert` Mooncake emits at primitive call sites (in [`src/interpreter/reverse_mode.jl`](https://github.com/chalk-lab/Mooncake.jl/blob/main/src/interpreter/reverse_mode.jl)) normally catches at the rule-output boundary, surfacing as a `TypeError` rather than a silent wrong gradient. In the singleton case the typeassert is *not* a safety net — by the time it would have run, the primitive call has already been replaced by a literal, so no typeassert is emitted.

The mechanism is the same in both shapes: although the primitive's body is not *differentiated*, it is still *inferred* — `NativeInterpreter` walks it to produce the primitive's `CallMeta`. Because `NativeInterpreter` does not consult `mooncake_method_table`, any overlay on a nested call within the body is invisible to that walk. Inference of the primitive's return type therefore sees the original definitions of its nested calls, not the overlays.

Example:

```julia
helper(::A) = A()
Mooncake.@mooncake_overlay helper(::A) = B()

primitive_wrapper(x::A) = helper(x)
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(primitive_wrapper), A}
```

From any caller, inference at the `primitive_wrapper` call site reports the return type as `A`, even though the overlay would return `B` if honoured. The nested overlay is invisible only at this boundary — `MooncakeInterpreter` walking the body directly would resolve it. The break is concrete: Mooncake takes the `NativeInterpreter` fast path for the primitive's `CallMeta`, and `NativeInterpreter`'s method lookup uses the standard method table, so the overlay on `helper` registered in `mooncake_method_table` simply isn't seen.

!!! details "How a singleton call gets folded away"
    1. `NativeInterpreter` (overlay-blind) infers the primitive call as `Const(original_value)` — the singleton instance from the *original* body.
    2. [`widen_rettype_callmeta`](@ref Mooncake.widen_rettype_callmeta) exists to prevent `Const` from causing primitive calls to fold away, but it has a documented carve-out: if every runtime argument at the call site is also `Const`, folding is treated as safe (the `sin(1.0)`-with-a-literal case). A zero-runtime-argument primitive trivially satisfies this; many SciML-style overlays do too.
    3. Const propagation in subsequent compiler passes replaces the primitive call with the literal value — the *original*'s singleton, not the overlay's.
    4. By the time AD construction processes the IR, there is no primitive call site at this location, only a constant. No `rrule!!` call is emitted; no `Core.typeassert` is emitted; no runtime check fires.
    5. Downstream code is compiled against the inferred (wrong) singleton type and picks rules keyed to it. The runtime never has the opportunity to course-correct.

This is by design: Mooncake treats primitives as sealed boundaries and does not walk into a primitive's body to discover what overlays might affect it. The fix in [#1170](https://github.com/chalk-lab/Mooncake.jl/pull/1170) extends overlay-awareness only to the *primitive's own signature* — the boundary inference is already looking at. For overlays reachable from inside a primitive's body, the rule and Mooncake's inferred type may diverge, and keeping them coherent is the rule author's responsibility.
