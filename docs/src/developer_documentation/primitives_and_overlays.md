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

The rule, not the body, produces the value at this call site; its return type — typically `Tuple{CoDual{B,F}, Pullback}` — is what the surrounding AD code sees.

!!! details "Mechanism"
    Mooncake's `AbstractInterpreter` override of `abstract_call_gf_by_type` (in `src/interpreter/abstract_interpretation.jl`) checks each call site against the primitive table via `any_matches_primitive`. When a match is detected, the resulting `CallMeta` is wrapped in a `NoInlineCallInfo`, which Mooncake's `inlining_policy` / `src_inlining_policy` then refuses to inline. As a result, the primitive call survives into the IR that AD construction sees, and the rule-dispatch code is emitted at that statement instead of inlined primal code.

## Overlays

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

Mooncake's IR transformation is driven by inferred type information. Three places matter, and they fire in this order:

1. **Source-IR inference.** The function being differentiated is inferred via `MooncakeInterpreter`. This produces the IR that the AD transformation rewrites.
2. **Per-call `CallMeta`.** At each call statement during the source-IR walk, Mooncake needs the return type, effects, and call info. `abstract_call_gf_by_type` produces this `CallMeta`. Primitive call sites are handled specially here — see [Inference at primitive call sites](@ref).

   !!! details "What `abstract_call_gf_by_type` does"
       This is Julia's central per-call-site inference entry point, in `Compiler/src/abstractinterpretation.jl`. Given a function value, the call's argument info / `atype`, the current inference state, and a cap on how many methods to consider, it returns a `Future{CallMeta}` with the inferred return type, exception type, effects, and call-site info.

       At a high level it does three things: (a) **method lookup** via `find_method_matches`, finding all method candidates whose signatures intersect the call's `atype`; (b) **per-match abstract interpretation**, calling `abstract_call_method` to recursively infer each candidate's body and optionally running constant propagation; (c) **aggregation**, joining each candidate's return type, exception type, and effects over the IPO lattice, recording inference edges, and producing a final `CallMeta`. The per-match loop is cooperatively pausable — if a sub-inference is in flight, the work is rescheduled — which is how stackless inference on Julia 1.12+ works.

       Mooncake overrides this function for `MooncakeInterpreter` and inserts its primitive / overlay logic *before* the recursive per-match step: if the call site is a primitive, we don't want `abstract_call_method` recursion at all, just a `CallMeta`. See [Inference at primitive call sites](@ref) for why this matters and how Mooncake produces the `CallMeta` without recursion.
3. **Rule-type inference.** Later, during AD IR construction, Mooncake calls `Core.Compiler.return_type` with the default interpreter — for example when emitting a `pullback_type` lookup — to learn the type the rule itself returns.

The key asymmetry to internalise: **Mooncake's source-function inference is overlay-aware via `OverlayMethodTable`; `NativeInterpreter`, used at primitive boundaries, is not.**

### Inference at primitive call sites

At every call site in the source IR, Mooncake needs a return type — downstream code is typed against it. At a primitive call site this is no different: the surrounding code wants the primal's return type, and the rule is an *implementation* keyed to that type, not a *source* for it. So inference asks the primal what it returns; the rule isn't consulted at this stage.

That leaves the question of *how* to obtain the primal `CallMeta`. Why not just use `MooncakeInterpreter` for this inference too? It is, after all, the AD-aware interpreter we are already running on the source IR — recursing into a primitive's body with it is the natural choice.

The problem is what that recursion costs. `MooncakeInterpreter`'s `abstract_call_gf_by_type` override re-passes itself into the recursive walk: every call site visited during inference of the body re-enters the override, runs the primitive check, and re-enters Julia's recursive inference under `MooncakeInterpreter` again. `MooncakeInterpreter` also uses its own inference caches separate from Julia's global cache, so the re-walk does not reuse Julia's already-warm results — every function Mooncake differentiates triggers a fresh walk of its transitive call tree.

For most call trees this is fine. Some real-world ones are not — see [PR #1115](https://github.com/chalk-lab/Mooncake.jl/pull/1115) for a SciML-shaped case where this recursion explodes into a silent compile-time hang.

The license to do something different comes from observing what the primitive's body is actually used for at this point in AD. The body is *not* going to be rewritten into AD-generated code: at a primitive call site, the registered rule replaces the body at runtime, and the AD transformation emits a rule dispatch rather than walking the body's statements. The body's contents therefore do not need to be inspected by an AD-aware interpreter; only its return type needs to flow into the surrounding primal IR's `CallMeta`. Any interpreter that produces a correct `CallMeta` for the boundary is sufficient.

Mooncake therefore asks `NativeInterpreter` for the `CallMeta` and stops. Standard inference still walks the body to compute the type, but it does so against Julia's global cache and without re-firing Mooncake's primitive-detection machinery at every nested call site. The recursion that would otherwise cascade through `solve`'s call tree under `MooncakeInterpreter` is bounded at each primitive boundary.

The wrinkle: `NativeInterpreter` is overlay-blind. Any overlay that would affect the primitive — directly, or indirectly via a call inside its body — is invisible to inference at the primitive boundary.

## Composition

The supported and unsupported combinations follow directly from the layers each macro touches.

### Overlay only, no primitive

Mooncake's interpreter sees the overlay's body in place of the original. AD differentiates the overlay body. Fully supported; this is the canonical use of `@mooncake_overlay`.

### Primitive only, no overlay

The rule replaces the primal at runtime. Inference at the call site asks `NativeInterpreter` for the *original* body's `CallMeta`. Fully supported, on the standard contract that the rule's primal return type matches the original's.

### Overlay and primitive on the *same* signature — supported corner case

From the previous discussion: as long as the overlay returns the same type as the original, `NativeInterpreter`'s overlay-blindness doesn't matter — inference and the rule agree on the type at the call site either way. The corner case is when the overlay *changes* the return type: inference would see the original's type, the rule would produce the overlay's, and downstream code would be typed against the wrong one.

Mooncake detects this configuration and routes inference through the overlay-aware default path, so the inferred return type at the call site matches the overlay's, not the original's. At runtime, the registered `rrule!!` still fires.

In effect: the rule produces the value and the adjoint; the overlay's only job is to align inference's view of the return type with what the rule actually returns. This matters when the rule returns a value of a different type from the original primal and downstream code dispatches on that type. Most users should not need this pattern — prefer to express the change as either an overlay or a primitive, not both — but it is supported.

!!! details "Mechanism"
    `any_matches_overlay` (in `src/interpreter/abstract_interpretation.jl`) walks the applicable methods and checks `method.external_mt === mooncake_method_table`. When that returns true, `abstract_call_gf_by_type` takes the `@invoke` branch — i.e. it defers to the default `abstract_call_gf_by_type` *with `MooncakeInterpreter` still as the interpreter*, so method lookup inside that call still goes through `OverlayMethodTable` and resolves to the overlay's body. The `NativeInterpreter` fast path is reserved for primitives whose applicable methods have no overlay.

### Primitive called from inside an overlay's body — supported

An overlay's body may itself call a registered primitive. This is the ordinary, supported flow: Mooncake walks the overlay body for AD, and any primitive call inside it is handled by the same machinery that handles primitive calls anywhere else (primitive detection, `NativeInterpreter` for the `CallMeta`, rule dispatch at runtime). No special arrangement is needed; this is in fact the most common reason to write an overlay — substituting an AD-unfriendly body with one that bottoms out on a hand-written rule.

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

Differentiating any caller of `original_f` walks the overlay's body, hits `my_primitive`, dispatches its `rrule!!`, and computes the gradient via the rule's adjoint. The [drift](@ref "Drift between rules and overlays") hazard applies as anywhere else: the primitive's rule must agree with its inferred primal return type.

### Overlay on a non-primitive called from inside a primitive's body — not supported

Although the primitive's body is not *differentiated*, it is still *inferred* — `NativeInterpreter` walks it to produce the primitive's `CallMeta`. Because `NativeInterpreter` does not consult `mooncake_method_table`, any overlay on a nested call within the body is invisible to that walk. Inference of the primitive's return type therefore sees the original definitions of its nested calls, not the overlays.

Example:

```julia
helper(::A) = A()
Mooncake.@mooncake_overlay helper(::A) = B()

primitive_wrapper(x::A) = helper(x)
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(primitive_wrapper), A}
```

Call sites of `primitive_wrapper` will infer the return type as `A`, even though the overlay would return `B` if it were honoured.

!!! details "Mechanism — why inference sees `A`"
    Walking the layers for a call `primitive_wrapper(a::A)`:

    1. The outer function containing this call is inferred under `MooncakeInterpreter`. At the `primitive_wrapper(a)` statement, `abstract_call_gf_by_type` is called with `MooncakeInterpreter`.
    2. Applicable-method lookup uses `OverlayMethodTable`, but `primitive_wrapper` itself has no overlay. The match resolves to the ordinary `primitive_wrapper(::A)`.
    3. `any_matches_primitive` returns true. `any_matches_overlay` returns false (the method's `external_mt` is unset).
    4. The branch added in #1170 takes the `NativeInterpreter` fast path and asks for the `CallMeta` of `primitive_wrapper(::A)` under `NativeInterpreter`.
    5. `NativeInterpreter` infers `primitive_wrapper`'s body. At the `helper(x)` statement inside that body, its method lookup uses the standard method table — `mooncake_method_table` is invisible to it — so it resolves to `helper(::A) = A()` and infers the return as `A`.
    6. The inferred return type of `primitive_wrapper(::A)` therefore propagates as `A` back to the outer caller, even though under Mooncake the overlay would have made it `B`.

    The break is at step (5): the right layer (Mooncake) knows about the overlay, but it has delegated this lookup to a layer that does not.

**The wrong-gradient mechanism.** The reported cases of [#1169](https://github.com/chalk-lab/Mooncake.jl/issues/1169) — including the SciMLBase `Originator` shape — involve primitives whose return types are singletons. For these, inference produces not just a type but a `CC.Const(value)`. The consequence is that the rule never gets a chance to fire: Julia constant-folds the call to the literal value before AD construction sees it.

Walked out:

1. `NativeInterpreter` (overlay-blind) infers the primitive call as `Const(original_value)` — the singleton instance from the *original* body.
2. `widen_rettype_callmeta` exists to prevent `Const` from causing primitive calls to fold away, but it has a documented carve-out: if every runtime argument at the call site is also `Const`, folding is treated as safe (the `sin(1.0)`-with-a-literal case). A zero-runtime-argument primitive trivially satisfies this; many SciML-style overlays do too.
3. Const propagation in subsequent compiler passes replaces the primitive call with the literal value — the *original*'s singleton, not the overlay's.
4. By the time AD construction processes the IR, there is no primitive call site at this location, only a constant. No `rrule!!` call is emitted; no `Core.typeassert` is emitted; no runtime check fires.
5. Downstream code is compiled against the inferred (wrong) singleton type and picks rules keyed to it. The runtime never has the opportunity to course-correct.

The result is a silent wrong gradient: not because the rule produced the wrong value, but because the rule was never called. The typeassert that Mooncake emits at primitive call sites (in `src/interpreter/reverse_mode.jl`) is not the safety net here — by the time it would have run, the call has already been replaced by a literal.

For overlays that don't yield a `Const` (e.g. a primitive whose return is concrete but not a singleton), the failure mode is different: either inference and the rule happen to agree on the type and there is no problem, or they disagree and the typeassert traps with a `TypeError` — loud rather than silent. The dangerous combination is overlay + singleton return, which is exactly the shape both #1169's MWE and the SciMLBase usage take.

This is by design: Mooncake treats primitives as sealed boundaries and does not walk into a primitive's body to discover what overlays might affect it. The fix in [#1170](https://github.com/chalk-lab/Mooncake.jl/pull/1170) extends overlay-awareness only to the *primitive's own signature* — the boundary inference is already looking at. For overlays reachable from inside a primitive's body, the rule and Mooncake's inferred type may diverge, and keeping them coherent is the rule author's responsibility (see [Drift between rules and overlays](@ref) for the contract).

An alternative approach in [PR #1168](https://github.com/chalk-lab/Mooncake.jl/pull/1168) instead walks into primitive bodies with overlay-aware method lookup (via a wrapper around `NativeInterpreter` that uses `mooncake_method_table`); it would fix the inside-body case as well. It was not adopted in #1170 — the sealed-boundary policy makes for a smaller and more focused fix.

Workaround when you do encounter this shape: lift the overlay to the level the user actually calls. Either remove the primitive declaration on the wrapper and let AD differentiate it, or register the desired behaviour as a primitive on the outer function.

### Drift between rules and overlays

A rule's return type is hand-written and fixed when `rrule!!` is authored. If an overlay is introduced later — directly on the primitive, or on a function called in its body — that changes the primal type Mooncake's inference produces, the rule and the inference can disagree.

The invariant the author must maintain is:

```text
inferred primal return type at the call site
    ==
primal type inside the CoDual returned by the rule
```

When the types are concrete and non-singleton, the typeassert Mooncake emits on the rule's primal output (in `src/interpreter/reverse_mode.jl`) catches violations at runtime. For example:

```julia
struct DriftOld; v::Float64 end
struct DriftNew; v::Float64 end

f(x::Float64) = DriftOld(x)

# Overlay introduced later, changing the return type.
Mooncake.@mooncake_overlay f(x::Float64) = DriftNew(x)

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(f), Float64}

# Stale rule, authored before the overlay was added: still returns DriftOld(...).
function Mooncake.rrule!!(g::CoDual{typeof(f)}, x::CoDual{Float64})
    pb(dy) = NoRData(), NoRData(), dy
    return zero_fcodual(DriftOld(primal(x))), pb
end
```

Differentiating any caller of `f` then traps with:

```text
TypeError: in typeassert, expected CoDual{DriftNew, NoFData},
           got a value of type CoDual{DriftOld, NoFData}
```

When the inferred return is a singleton (`CC.Const`), the primitive call is liable to be const-folded to the literal value before the rule fires, so the typeassert is bypassed. The runtime then follows the inferred-type path — which is what the overlay would have produced — and the rule's stale return type ends up irrelevant in practice. Convenient, but it's coincidence: don't rely on it.

#1170 makes inference at primitive boundaries overlay-aware so the left-hand side reflects what the overlay-modified primal would actually return. Keeping the right-hand side in sync — adjusting the rule when the overlay changes the type — is the author's responsibility.
