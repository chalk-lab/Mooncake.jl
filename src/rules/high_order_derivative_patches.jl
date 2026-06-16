@zero_derivative MinimalCtx Tuple{typeof(get_interpreter),Type{<:Mode}}
@zero_derivative MinimalCtx Tuple{
    typeof(build_rrule_checks),MooncakeInterpreter,Any,Bool,Bool
}
@zero_derivative MinimalCtx Tuple{typeof(is_primitive),Type,Type{<:Mode},Type,UInt}

@is_primitive MinimalCtx Tuple{
    typeof(build_derived_rrule),MooncakeInterpreter{C},Any,Any,Bool
} where {C}

# This file defines three forward-over-reverse (FoR) types organised into two layers.
# Big-picture context: reverse mode has `DerivedRule` / `LazyDerivedRule` /
# `DynamicDerivedRule` (`src/interpreter/reverse_mode.jl`); forward mode has
# `DerivedFRule` / `LazyFRule` / `DynamicFRule` (`src/interpreter/forward_mode.jl`).
# The FoR triplet below mirrors them — with the layer caveat described at the end
# of this block.
#
# Value layer — `DerivedFoRRule{D}` (defined below, near `compile_for_rule`): a FoR
# rule (or its absence) held as a value and reused across many calls. `D <: Dual`
# carries a pre-built `Dual(rule, rule_tangent)` for HVP / Hessian; `D === Nothing`
# is the primitive-passthrough sentinel. Reuse is safe because the rule's Stacks
# self-reset within each forward+reverse pass.
#
# Frule layer — `LazyFoRRule` / `DynamicFoRRule` (this section): callable caches
# dispatched as the frule for `build_derived_rrule`. Each call returns a fresh
# `Dual(rule, rule_tangent)`, with Stacks cloned via `_for_rule_cached_dual` —
# defensive against nested AD where the rule may be re-entered before the
# previous call's Stacks have self-reset. `build_primitive_frule` selects between
# the two via `__build_primitive_frule` (@generated):
#
#   • Concrete Trule → LazyFoRRule{Trule,Tfwd,Trvs}: fully-typed single-slot cache.
#     Zero virtual dispatch on cache hits. Safe because each instance lives at exactly
#     one call site in the compiled IR, so only one inner signature ever reaches it.
#
#   • Non-concrete Trule (Trule = Any) → DynamicFoRRule: Dict-keyed cache. Arises
#     when build_rrule's @nospecialize sig_or_mi causes the forward-mode compiler to
#     see SMI=Any/S=Any, yielding Trule=Any. The single frule call site is then
#     shared by multiple LazyDerivedRule instances (different inner functions), so a
#     single-slot cache would serve the wrong rule — see DynamicFoRRule for key design.
#
# The trio mirrors reverse-mode `DerivedRule` / `LazyDerivedRule` / `DynamicDerivedRule`
# at a slight angle: reverse mode's three types are all *rule-callable*
# (`(rule)(args...)` with primal args, returning AD output). None of the three
# FoR types here is rule-callable in that sense:
#   • `DerivedFoRRule` is a *carrier* of a pre-built `Dual(rule, rule_tangent)`;
#     callers extract via `get_inner_rrule`, they do not invoke it directly.
#   • `LazyFoRRule` / `DynamicFoRRule` are *constructors* — their call shape is
#     `(::Dual{typeof(build_derived_rrule)}, …)`, returning a fresh Dual rule
#     per invocation (the frule for `build_derived_rrule`, not for user code).
# Mooncake has no top-level "FoR rule" type — the user entry point
# `value_and_hvp!!` dispatches through a regular `DerivedFRule`. The shared
# `FoRRule` suffix is a family label (= "lives in the FoR machinery"), not a
# contract suffix.
#
# TODO: rename to make the role honest — e.g. carrier → `FoRRuleDual` /
# `RRuleDualBox`, constructors → `Lazy` / `DynamicBuildRRuleFRule`. Kept as-is
# for now to minimise diff while wiring HVP through `DerivedFoRRule`.
mutable struct LazyFoRRule{Trule,Tfwd,Trvs}
    rule::Trule
    fwd_dual_callable::Tfwd
    rvs_dual_callable::Trvs
    LazyFoRRule{Trule,Tfwd,Trvs}() where {Trule,Tfwd,Trvs} = new()
end

# Dict-keyed cache for the non-concrete (Any) case of __build_primitive_frule.
# Cache key is (sig, debug_mode):
#   - sig        distinguishes inner functions sharing the @nospecialize call site.
#                We intentionally do not key on sig_or_mi: the compiled DerivedRule is a
#                function of the signature-level IR selected here, and each reachable
#                MethodInstance at this call site currently has a unique sig. If that
#                assumption ever breaks (two MethodInstances with the same sig but different
#                IR), we would silently serve the wrong cached rule, producing incorrect
#                derivatives. A future fix would be to key on sig_or_mi instead.
#   - debug_mode is included because DebugRRule and plain DerivedRule have different
#     field layouts; serving one to a caller expecting the other causes FieldError on the
#     `new_rule.rule` access in _for_rule_cached_dual's debug branch.
# Not thread-safe: the Dict is mutated without a lock (same caveat as LazyFoRRule's
# bare field assignment).
mutable struct DynamicFoRRule
    cache::Dict{Tuple{Any,Bool},Tuple{Any,Any,Any}}  # (sig, debug_mode) => (rule, fwd_dc, rvs_dc)
    DynamicFoRRule() = new(Dict{Tuple{Any,Bool},Tuple{Any,Any,Any}}())
end

@generated function __build_primitive_frule(
    sig::Type{<:Tuple{typeof(build_derived_rrule),MooncakeInterpreter{C},SMI,S,Bool}}
) where {C,SMI,S}
    Trule = Core.Compiler.return_type(
        build_derived_rrule, Tuple{MooncakeInterpreter{C},SMI,S,Bool}
    )
    # build_derived_rrule is called inside build_rrule with @nospecialize sig_or_mi, so
    # the forward-mode compiler sees SMI=Any/S=Any here, causing inference to return Any
    # for Trule. Guard against this: fieldtype(Any, :fwds_oc) would throw FieldError.
    # Use DynamicFoRRule (dict-keyed cache) rather than LazyFoRRule{Any,Any,Any}: the
    # shared call site in build_rrule's frule may be reached with different inner
    # signatures (e.g. collect vs num_to_vec when _build_rule! is called for multiple
    # LazyDerivedRule instances), so a single-slot cache is incorrect.
    if !isconcretetype(Trule)
        return :(DynamicFoRRule())
    end
    # Extract DerivedRule from the DebugRRule wrapper (if present) to access
    # the forward and reverse closure field types.
    # build_derived_rrule always returns a DerivedRule (or DebugRRule{DerivedRule{...}}),
    # so inner always has :fwds_oc and :pb_oc_ref. Guard against any other inner type
    # (e.g. a primitive wrapped in DebugRRule) that would throw FieldError here.
    inner = Trule <: DebugRRule ? fieldtype(Trule, :rule) : Trule
    if !hasfield(inner, :fwds_oc) || !hasfield(inner, :pb_oc_ref)
        return :(DynamicFoRRule())
    end
    fwds_oc_T = fieldtype(inner, :fwds_oc)
    rvs_oc_T = fieldtype(fieldtype(inner, :pb_oc_ref), :x)
    interp_fwd_T = MooncakeInterpreter{C,ForwardMode}
    Tfwd = Core.Compiler.return_type(build_frule, Tuple{interp_fwd_T,fwds_oc_T})
    Trvs = Core.Compiler.return_type(build_frule, Tuple{interp_fwd_T,rvs_oc_T})
    # Fall back to DynamicFoRRule if inference cannot pin down the dual callable types;
    # LazyFoRRule{Trule,Any,Any} would defeat its own purpose (typed single-slot cache).
    if !isconcretetype(Tfwd) || !isconcretetype(Trvs)
        return :(DynamicFoRRule())
    end
    return :(LazyFoRRule{$Trule,$Tfwd,$Trvs}())
end

function build_primitive_frule(
    sig::Type{<:Tuple{typeof(build_derived_rrule),MooncakeInterpreter{C},SMI,S,Bool}}
) where {C,SMI,S}
    return __build_primitive_frule(sig)
end

# Forward-over-reverse seed for a reverse rule (or one of its closures). The
# forward V must carry the forward dual-callables — which only `_dual_mc` /
# `zero_tangent` can build — and its `fwds_oc` / `pb_oc` must share their comms
# captures. Both fall out of lifting the reverse zero tangent: `zero_tangent`
# already builds the dual-callables into its `MistyClosureTangent`s, and `lift`'s
# aliasing cache shares the captures. The generic structural seed factory cannot
# do either (it would try to construct the `Any`-typed `MistyClosureTangent`
# backing from scratch), so route these types through `lift`.
@inline function zero_dual(::Val{1}, x::Union{DerivedRule,MistyClosure})
    return tangent(lift(x, zero_tangent(x), IdDict()))
end

# A `MooncakeInterpreter` is compiler state used to *build* rules (its caches map
# `MethodInstance`s to compiled code); no user derivative flows through it. Its
# reverse `tangent_type` is nonetheless a deep struct over `IdDict`/`Memory{Any}`
# caches. In forward mode it is non-differentiable: its V is `NoDual`. This lets
# `_build_rule!` (lazy rule construction inside a forward pass, forward-over-
# reverse) seed the interpreter slot without lifting the entire compiler-cache
# structure.
@foldable @inline dual_type(::Val{N}, ::Type{<:MooncakeInterpreter}) where {N} = NoDual
@inline zero_dual(::Val{N}, ::MooncakeInterpreter) where {N} = NoDual()
@inline uninit_dual(::Val{N}, ::MooncakeInterpreter) where {N} = NoDual()
# A `MooncakeInterpreter` is non-diff, so its only tangent kinds are `NoTangent`/`NoDual`. Match those
# two *specifically* (not `::Any`) so we dominate the generic `lift(x, ::NoTangent)`/`lift(x::P, ::NoDual)`
# in the 1st arg rather than being ambiguous (an `::Any` 2nd arg ties with those 2nd-arg-specific generics).
@inline lift(x::MooncakeInterpreter, ::NoTangent) = Lifted{typeof(x),1,NoDual}(x, NoDual())
@inline lift(x::MooncakeInterpreter, ::NoDual) = Lifted{typeof(x),1,NoDual}(x, NoDual())

# LazyFoRRule / DynamicFoRRule are frules for build_derived_rrule:
#
#   build_derived_rrule : (interp, sig_or_mi, sig, debug_mode) → rrule
#   LazyFoRRule         : (Lifted(build_derived_rrule, ·), Lifted(interp, ·), ...) → Lifted(rrule, t_rule)
#                         where t_rule = J_{build_derived_rrule} · (t_interp, ...)
#
# _for_rule_cached_dual and _compile_for_rule are shared helpers used by both.

# Cache-hit helper: given a previously compiled (rule, fwd_dc, rvs_dc), return a
# Lifted(rule, rule_tangent) with fresh empty Stacks for this call.
#
# Stack aliasing invariant: fwd_oc and rvs_oc share the same comms Stack objects from
# shared_data (fwd_oc.captures[i] === rvs_oc.captures[i]).  Their tangent Stacks must
# also be aliased: the fwds tangent pass writes to comms tangent Stacks and the rvs
# tangent pass reads from the same objects.  zero_tangent uses an IdDict internally, so
# calling it jointly on both captures tuples ensures
# captures_tangent[1][i] === captures_tangent[2][i] for aliased primal objects.
# _copy(Stack{T}) resets each primal Stack to empty; regenerating captures_tangent from
# the fresh primal keeps tangent Stacks size-consistent.
function _for_rule_cached_dual(rule, fwd_dc, rvs_dc, debug_mode::Bool)
    new_rule = _copy(rule)
    inner_rule = debug_mode ? new_rule.rule : new_rule
    captures_tangent = zero_tangent((
        inner_rule.fwds_oc.oc.captures, inner_rule.pb_oc_ref[].oc.captures
    ))
    inner_tangent = Tangent((;
        fwds_oc=MistyClosureTangent(captures_tangent[1], _copy(fwd_dc)),
        pb_oc_ref=MutableTangent((;
            x=PossiblyUninitTangent(MistyClosureTangent(captures_tangent[2], _copy(rvs_dc)))
        )),
        nargs=NoTangent(),
    ))
    rule_tangent = debug_mode ? Tangent((; rule=inner_tangent)) : inner_tangent
    # Lift the (reverse-shaped) rule_tangent into the canonical forward V: `lift`
    # carries each `MistyClosureTangent`'s dual-callable and, via its aliasing
    # cache, gives the rule's `fwds_oc` / `pb_oc` one shared forward captures slot.
    return lift(new_rule, rule_tangent, IdDict())
end

# First-call compilation helper: build a DerivedRule (+ dual callables + tangent) for
# (interp, sig_or_mi, sig, debug_mode). Returns (rule, fwd_dc, rvs_dc, rule_tangent).
function _compile_for_rule(
    interp::MooncakeInterpreter{C}, sig_or_mi, sig, debug_mode::Bool
) where {C}
    @nospecialize sig_or_mi sig

    # Derive unoptimized forwards- and reverse-pass IR.
    dri = generate_ir(interp, sig_or_mi; debug_mode, do_optimize=false)

    # Optimize and build the primal DerivedRule.
    raw_rule = let
        optimized_fwd_ir = optimise_ir!(CC.copy(dri.fwd_ir))
        optimized_rvs_ir = optimise_ir!(CC.copy(dri.rvs_ir))
        fwd_oc = misty_closure(dri.fwd_ret_type, optimized_fwd_ir, dri.shared_data...)
        rvs_oc = misty_closure(dri.rvs_ret_type, optimized_rvs_ir, dri.shared_data...)
        nargs = num_args(dri.info)
        sig_flat = flatten_va_sig(sig, dri.isva, nargs)
        DerivedRule(sig_flat, fwd_oc, Ref(rvs_oc), dri.isva, Val(nargs))
    end

    # Build forward-mode dual callables for the fwd and rvs passes.
    # Use a forward-mode interpreter to block inlining of frules during optimisation.
    #
    # Aliasing: fwd_oc and rvs_oc share the comms Stack objects from dri.shared_data
    # (fwd_oc.oc.captures[i] === rvs_oc.oc.captures[i] for shared slots).  Calling
    # zero_tangent jointly on (fwd_oc.oc.captures, rvs_oc.oc.captures) preserves this
    # aliasing in the returned captures_tangent, so the tangent Stacks written by the
    # forward-tangent pass are the same objects read by the reverse-tangent pass.
    # NOTE: fwd_dc and rvs_dc returned here alias with the tangent embedded in
    # raw_rule_tangent (fwds_oc / pb_oc_ref fields). Callers that cache (rule, fwd_dc,
    # rvs_dc) and later call _for_rule_cached_dual must use _copy to get fresh Stacks
    # and a new independent tangent — do not reuse these objects directly.
    fwd_dc, rvs_dc, raw_rule_tangent = let
        interp_forward = MooncakeInterpreter(C, ForwardMode; world=interp.world)
        optimized_fwd_ir = optimise_ir!(dri.fwd_ir; interp=interp_forward)
        optimized_rvs_ir = optimise_ir!(dri.rvs_ir; interp=interp_forward)
        fwd_oc = misty_closure(dri.fwd_ret_type, optimized_fwd_ir, dri.shared_data...)
        rvs_oc = misty_closure(dri.rvs_ret_type, optimized_rvs_ir, dri.shared_data...)
        captures_tangent = zero_tangent((fwd_oc.oc.captures, rvs_oc.oc.captures))
        fwd_dc = build_frule(interp_forward, fwd_oc; skip_world_age_check=true, debug_mode)
        rvs_dc = build_frule(interp_forward, rvs_oc; skip_world_age_check=true, debug_mode)
        tangent = Tangent((;
            fwds_oc=MistyClosureTangent(captures_tangent[1], fwd_dc),
            pb_oc_ref=MutableTangent((;
                x=PossiblyUninitTangent(MistyClosureTangent(captures_tangent[2], rvs_dc))
            )),
            nargs=NoTangent(),
        ))
        fwd_dc, rvs_dc, tangent
    end

    rule = debug_mode ? DebugRRule(raw_rule) : raw_rule
    rule_tangent = debug_mode ? Tangent((; rule=raw_rule_tangent)) : raw_rule_tangent
    return rule, fwd_dc, rvs_dc, rule_tangent
end

function (cache::LazyFoRRule{Trule,Tfwd,Trvs})(
    ::Lifted{typeof(build_derived_rrule)},
    _interp::Lifted{<:MooncakeInterpreter{C}},
    _sig_or_mi::Lifted,
    _sig::Lifted,
    _debug_mode::Lifted{Bool},
) where {Trule,Tfwd,Trvs,C}
    @nospecialize _sig_or_mi _sig

    debug_mode = primal(_debug_mode)

    # Cache hit: reuse compiled artifacts with fresh empty Stacks. sig is not
    # re-checked because each LazyFoRRule lives at exactly one call site in the
    # compiled IR (inside a fixed-grad_f closure), so the inner signature is
    # invariant for its lifetime. debug_mode is checked below because the cached rule
    # layout differs between DebugRRule and plain DerivedRule.
    if isdefined(cache, :rule)
        if debug_mode != (cache.rule isa DebugRRule)
            error(
                "LazyFoRRule cache hit with debug_mode=$debug_mode but cached rule is " *
                "$(typeof(cache.rule)); debug_mode must be consistent across calls.",
            )
        end
        return _for_rule_cached_dual(
            cache.rule, cache.fwd_dual_callable, cache.rvs_dual_callable, debug_mode
        )
    end

    # First call: compile, populate the single-slot cache, return.
    rule, fwd_dc, rvs_dc, rule_tangent = _compile_for_rule(
        primal(_interp), primal(_sig_or_mi), primal(_sig), debug_mode
    )
    cache.rule = rule
    cache.fwd_dual_callable = fwd_dc
    cache.rvs_dual_callable = rvs_dc
    return lift(rule, rule_tangent, IdDict())
end

function (cache::DynamicFoRRule)(
    ::Lifted{typeof(build_derived_rrule)},
    _interp::Lifted{<:MooncakeInterpreter{C}},
    _sig_or_mi::Lifted,
    _sig::Lifted,
    _debug_mode::Lifted{Bool},
) where {C}
    @nospecialize _sig_or_mi _sig

    debug_mode = primal(_debug_mode)

    # Key on (sig, debug_mode): sig distinguishes inner functions sharing this call
    # site, while sig_or_mi is intentionally omitted because the compiled rule is
    # determined by the signature-level IR selected here and each relevant
    # MethodInstance currently has a unique sig. debug_mode is included because
    # DebugRRule and DerivedRule have different field layouts — serving one to a
    # caller expecting the other causes FieldError.
    dict_key = (primal(_sig), debug_mode)

    entry = get(cache.cache, dict_key, nothing)
    if entry !== nothing
        rule, fwd_dc, rvs_dc = entry
        return _for_rule_cached_dual(rule, fwd_dc, rvs_dc, debug_mode)
    end

    # First call for this (sig, debug_mode): compile, cache, return.
    rule, fwd_dc, rvs_dc, rule_tangent = _compile_for_rule(
        primal(_interp), primal(_sig_or_mi), primal(_sig), debug_mode
    )
    cache.cache[dict_key] = (rule, fwd_dc, rvs_dc)
    return lift(rule, rule_tangent, IdDict())
end

function rrule!!(
    ::CoDual{typeof(build_derived_rrule)},
    _interp::CoDual{<:MooncakeInterpreter},
    _sig_or_mi::CoDual,
    _sig::CoDual,
    _debug_mode::CoDual{Bool},
)
    throw(
        ArgumentError(
            "Reverse-over-reverse differentiation is not supported. " *
            "Encountered attempt to differentiate build_derived_rrule in reverse mode.",
        ),
    )
end

# Wrapper around a pre-built `Dual(rule, rule_tangent)` so forward AD sees the
# rule with its forward-mode-compiled dual callables instead of `zero_tangent`
# re-deriving them over the reverse-mode-optimised primal IR (which would
# reintroduce the bug `_compile_for_rule` exists to avoid). `tangent_type` is
# `NoTangent`; the tangent rides inside the cached `Dual`. The type parameter
# `D` discriminates: `D <: Dual` for derived rrules, `D === Nothing` for
# primitives — callers branch via `for_rule isa DerivedFoRRule{Nothing}`.
# Stale-rule caveat: the cached `Dual` is pinned to world age at prep; rebuild
# the `HVPCache` if methods change after prep.
struct DerivedFoRRule{D}
    rule_dual::D
end
function compile_for_rule(f, x...; debug_mode::Bool=false)
    sig = _typeof((f, x...))
    interp = get_interpreter(ReverseMode)
    if is_primitive(DefaultCtx, ReverseMode, sig, interp.world)
        return DerivedFoRRule{Nothing}(nothing)
    end
    rule, _, _, rule_tangent = _compile_for_rule(interp, sig, sig, debug_mode)
    return DerivedFoRRule(Dual(rule, rule_tangent))
end
tangent_type(::Type{<:DerivedFoRRule}) = NoTangent

get_inner_rrule(r::DerivedFoRRule{<:Dual}) = primal(r.rule_dual)
@is_primitive MinimalCtx Tuple{typeof(get_inner_rrule),<:DerivedFoRRule{<:Dual}}
function frule!!(::Dual{typeof(get_inner_rrule)}, r::Dual{<:DerivedFoRRule{<:Dual}})
    return primal(r).rule_dual
end
function rrule!!(::CoDual{typeof(get_inner_rrule)}, ::CoDual{<:DerivedFoRRule{<:Dual}})
    throw(
        ArgumentError(
            "DerivedFoRRule is forward-over-reverse only; reverse-mode " *
            "differentiation through it is not supported.",
        ),
    )
end

# TODO: This is a workaround for forward-over-reverse. Primitives in reverse mode can get
# inlined when building the forward rule, exposing internal ccalls that lack an frule!!.
# For example, `dataids` is a reverse-mode primitive, but inlining it exposes
# `jl_genericmemory_owner`. The proper fix is to prevent primitive inlining during
# forward-over-reverse by forwarding `inlining_policy` through `BugPatchInterpreter` to
# `MooncakeInterpreter` during `optimise_ir!`, but this causes allocation regressions.
# See https://github.com/chalk-lab/Mooncake.jl/pull/878 for details.
# TODO: can be removed once we improve the performance of differentiating through building
# rules, such that the DI test will pass with no inner prep without this workaround.
@static if VERSION >= v"1.11-"
    function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{:jl_genericmemory_owner},Nw},
        ::Lifted{Val{Any},Nw},
        ::Lifted{Tuple{Val{Any}},Nw},
        ::Lifted{Val{0},Nw},
        ::Lifted{Val{:ccall},Nw},
        a::Lifted{<:Memory},
    ) where {Nw}
        y = ccall(:jl_genericmemory_owner, Any, (Any,), primal(a))
        return Lifted{typeof(y),Nw}(y, NoDual())
    end
    function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{:jl_genericmemory_owner}},
        ::CoDual{Val{Any}},
        ::CoDual{Tuple{Val{Any}}},
        ::CoDual{Val{0}},
        ::CoDual{Val{:ccall}},
        a::CoDual{<:Memory},
    )
        y = zero_fcodual(ccall(:jl_genericmemory_owner, Any, (Any,), primal(a)))
        return y, NoPullback(ntuple(_ -> NoRData(), 7))
    end
end

# This rule is potentially unnecessary if fixes are made elsewhere,
# but currently fixes differentiating through zero_tangent_internal for Arrays.
@zero_derivative MinimalCtx Tuple{typeof(zero_tangent),Any}

@static if VERSION < v"1.11-"
    function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{:jl_alloc_array_1d},Nw},
        ::Lifted{Val{Vector{P}},Nw},
        ::Lifted{Tuple{Val{Any},Val{Int}},Nw},
        ::Lifted{Val{0},Nw},
        ::Lifted{Val{:ccall},Nw},
        ::Lifted{Type{Vector{P}},Nw},
        n::Lifted{Int},
        args::Vararg{Lifted,M},
    ) where {Nw,P,M}
        _n = primal(n)
        y = ccall(:jl_alloc_array_1d, Vector{P}, (Any, Int), Vector{P}, _n)
        return Lifted{Vector{P},Nw}(y, uninit_dual(Val(Nw), y))
    end
    function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{:jl_alloc_array_2d},Nw},
        ::Lifted{Val{Matrix{P}},Nw},
        ::Lifted{Tuple{Val{Any},Val{Int},Val{Int}},Nw},
        ::Lifted{Val{0},Nw},
        ::Lifted{Val{:ccall},Nw},
        ::Lifted{Type{Matrix{P}},Nw},
        m::Lifted{Int},
        n::Lifted{Int},
        args::Vararg{Lifted,M},
    ) where {Nw,P,M}
        _m, _n = primal(m), primal(n)
        y = ccall(:jl_alloc_array_2d, Matrix{P}, (Any, Int, Int), Matrix{P}, _m, _n)
        return Lifted{Matrix{P},Nw}(y, uninit_dual(Val(Nw), y))
    end
    function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{:jl_alloc_array_3d},Nw},
        ::Lifted{Val{Array{P,3}},Nw},
        ::Lifted{Tuple{Val{Any},Val{Int},Val{Int},Val{Int}},Nw},
        ::Lifted{Val{0},Nw},
        ::Lifted{Val{:ccall},Nw},
        ::Lifted{Type{Array{P,3}},Nw},
        l::Lifted{Int},
        m::Lifted{Int},
        n::Lifted{Int},
        args::Vararg{Lifted,M},
    ) where {Nw,P,M}
        _l, _m, _n = primal(l), primal(m), primal(n)
        y = ccall(
            :jl_alloc_array_3d, Array{P,3}, (Any, Int, Int, Int), Array{P,3}, _l, _m, _n
        )
        return Lifted{Array{P,3},Nw}(y, uninit_dual(Val(Nw), y))
    end
end
