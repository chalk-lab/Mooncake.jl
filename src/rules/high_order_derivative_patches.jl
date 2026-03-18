@zero_derivative MinimalCtx Tuple{typeof(get_interpreter),Type{<:Mode}}
@zero_derivative MinimalCtx Tuple{
    typeof(build_rrule_checks),MooncakeInterpreter,Any,Bool,Bool
}
@zero_derivative MinimalCtx Tuple{typeof(is_primitive),Type,Type{<:Mode},Type,UInt}

@is_primitive MinimalCtx Tuple{
    typeof(build_derived_rrule),MooncakeInterpreter{C},Any,Any,Bool
} where {C}

# LazyFoRRule and DynamicFoRRule are the frule for `build_derived_rrule` in
# forward-over-reverse mode.  In HVPCache, grad_f calls
# prepare_gradient_cache → build_rrule → build_derived_rrule on every
# value_and_hvp!! call, so caching is essential: the first call compiles the
# inner DerivedRule and dual callables; subsequent calls reuse them via _copy (cheap).
#
# build_primitive_frule selects between the two via __build_primitive_frule (@generated):
#
#   • Concrete Trule → LazyFoRRule{Trule,Tfwd,Trvs}: fully-typed single-slot cache.
#     Zero virtual dispatch on cache hits. Safe because each instance lives at exactly
#     one call site in the compiled IR (not safe_for_literal), so only one inner
#     signature ever reaches it.
#
#   • Non-concrete Trule (Trule = Any) → DynamicFoRRule: Dict-keyed cache. Arises
#     when build_rrule's @nospecialize sig_or_mi causes the forward-mode compiler to
#     see SMI=Any/S=Any, yielding Trule=Any.  The single frule call site is then
#     shared by multiple LazyDerivedRule instances (different inner functions), so a
#     single-slot cache would serve the wrong rule — see DynamicFoRRule for key design.
mutable struct LazyFoRRule{Trule,Tfwd,Trvs}
    rule::Trule
    fwd_dual_callable::Tfwd
    rvs_dual_callable::Trvs
    LazyFoRRule{Trule,Tfwd,Trvs}() where {Trule,Tfwd,Trvs} = new()
end

# Dict-keyed cache for the non-concrete (Any) case of __build_primitive_frule.
# Cache key is (sig, debug_mode):
#   - sig        distinguishes inner functions sharing the @nospecialize call site
#   - debug_mode is included because DebugRRule and plain DerivedRule have different
#     field layouts; mixing them in _for_rule_cached_dual causes FieldError on the
#     `new_rule.rule` access in the debug branch.
# Not thread-safe: the Dict is mutated without a lock (same caveat as LazyFoRRule's
# bare field assignment).
mutable struct DynamicFoRRule
    cache::Dict{Any,Tuple{Any,Any,Any}}  # (sig, debug_mode) => (rule, fwd_dc, rvs_dc)
    DynamicFoRRule() = new(Dict{Any,Tuple{Any,Any,Any}}())
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
    inner = Trule <: DebugRRule ? fieldtype(Trule, :rule) : Trule
    fwds_oc_T = fieldtype(inner, :fwds_oc)
    rvs_oc_T = fieldtype(fieldtype(inner, :pb_oc_ref), :x)
    interp_fwd_T = MooncakeInterpreter{C,ForwardMode}
    Tfwd = Core.Compiler.return_type(build_frule, Tuple{interp_fwd_T,fwds_oc_T})
    Trvs = Core.Compiler.return_type(build_frule, Tuple{interp_fwd_T,rvs_oc_T})
    return :(LazyFoRRule{$Trule,$Tfwd,$Trvs}())
end

function build_primitive_frule(
    sig::Type{<:Tuple{typeof(build_derived_rrule),MooncakeInterpreter{C},SMI,S,Bool}}
) where {C,SMI,S}
    return __build_primitive_frule(sig)
end

# LazyFoRRule / DynamicFoRRule are frules for build_derived_rrule:
#
#   build_derived_rrule : (interp, sig_or_mi, sig, debug_mode) → rrule
#   LazyFoRRule         : (Dual(build_derived_rrule, ·), Dual(interp, ·), ...) → Dual(rrule, t_rule)
#                         where t_rule = J_{build_derived_rrule} · (t_interp, ...)
#
# _for_rule_cached_dual and _compile_for_rule are shared helpers used by both.

# Cache-hit helper: given a previously compiled (rule, fwd_dc, rvs_dc), return
# Dual(rule, rule_tangent) with fresh empty Stacks for this call.
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
    return Dual(new_rule, rule_tangent)
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

@static if VERSION < v"1.11-"
    # On Julia 1.10, we encounter a segfault when an OpaqueClosure/MistyClosure is
    # called with a specialised signature whose declared return type disagrees with
    # what the IR actually returns (JuliaLang/julia#51016).
    # The MistyClosure dual callables produced by _compile_for_rule can hit this if
    # called with wrong tangent types.
    #
    # The fix: @generated lets us check tangent types at compile time.  For a bad
    # specialisation we return :(error(...)), so the compiler never generates the
    # MistyClosure call for those types.  For well-typed calls the normal body is emitted.
    #
    # NOTE: @generated alone (without the type check) does NOT prevent the segfault —
    # returning an unconditional quote generates the same code as a plain function.
    # The early-return on mismatch is the critical part.
    @generated function (cache::LazyFoRRule{Trule,Tfwd,Trvs})(
        _bdr::Dual{typeof(build_derived_rrule)},
        _interp::Dual{<:MooncakeInterpreter{C}},
        _sig_or_mi::Dual,
        _sig::Dual,
        _debug_mode::Dual{Bool},
    ) where {Trule,Tfwd,Trvs,C}
        all_dts = (_bdr, _interp, _sig_or_mi, _sig, _debug_mode)
        for dt in all_dts
            P = dt.parameters[1]
            T = dt.parameters[2]
            if T !== tangent_type(P)
                msg = "Error in inputs to rule with input types $(Tuple{all_dts...})"
                return :(error($msg))
            end
        end
        return quote
            debug_mode = primal(_debug_mode)
            if isdefined(cache, :rule)
                return _for_rule_cached_dual(
                    cache.rule, cache.fwd_dual_callable, cache.rvs_dual_callable, debug_mode
                )
            end
            rule, fwd_dc, rvs_dc, rule_tangent = _compile_for_rule(
                primal(_interp), primal(_sig_or_mi), primal(_sig), debug_mode
            )
            cache.rule = rule
            cache.fwd_dual_callable = fwd_dc
            cache.rvs_dual_callable = rvs_dc
            return Dual(rule, rule_tangent)
        end
    end

    @generated function (cache::DynamicFoRRule)(
        _bdr::Dual{typeof(build_derived_rrule)},
        _interp::Dual{<:MooncakeInterpreter{C}},
        _sig_or_mi::Dual,
        _sig::Dual,
        _debug_mode::Dual{Bool},
    ) where {C}
        all_dts = (_bdr, _interp, _sig_or_mi, _sig, _debug_mode)
        for dt in all_dts
            P = dt.parameters[1]
            T = dt.parameters[2]
            if T !== tangent_type(P)
                msg = "Error in inputs to rule with input types $(Tuple{all_dts...})"
                return :(error($msg))
            end
        end
        return quote
            debug_mode = primal(_debug_mode)
            dict_key = (primal(_sig), debug_mode)
            entry = get(cache.cache, dict_key, nothing)
            if entry !== nothing
                rule, fwd_dc, rvs_dc = entry
                return _for_rule_cached_dual(rule, fwd_dc, rvs_dc, debug_mode)
            end
            rule, fwd_dc, rvs_dc, rule_tangent = _compile_for_rule(
                primal(_interp), primal(_sig_or_mi), primal(_sig), debug_mode
            )
            cache.cache[dict_key] = (rule, fwd_dc, rvs_dc)
            return Dual(rule, rule_tangent)
        end
    end
else
    function (cache::LazyFoRRule{Trule,Tfwd,Trvs})(
        ::Dual{typeof(build_derived_rrule)},
        _interp::Dual{<:MooncakeInterpreter{C}},
        _sig_or_mi::Dual,
        _sig::Dual,
        _debug_mode::Dual{Bool},
    ) where {Trule,Tfwd,Trvs,C}
        @nospecialize _sig_or_mi _sig

        debug_mode = primal(_debug_mode)

        # Cache hit: reuse compiled artifacts with fresh empty Stacks. Neither sig nor
        # debug_mode is re-checked: each LazyFoRRule lives at exactly one call site in the
        # compiled IR (inside a fixed-grad_f closure), so both are invariant for its lifetime.
        if isdefined(cache, :rule)
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
        return Dual(rule, rule_tangent)
    end

    function (cache::DynamicFoRRule)(
        ::Dual{typeof(build_derived_rrule)},
        _interp::Dual{<:MooncakeInterpreter{C}},
        _sig_or_mi::Dual,
        _sig::Dual,
        _debug_mode::Dual{Bool},
    ) where {C}
        @nospecialize _sig_or_mi _sig

        debug_mode = primal(_debug_mode)

        # Key on (sig, debug_mode): sig distinguishes inner functions sharing this call site;
        # debug_mode is included because DebugRRule and DerivedRule have different field
        # layouts — serving one to a caller expecting the other causes FieldError.
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
        return Dual(rule, rule_tangent)
    end
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
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_genericmemory_owner}},
        ::Dual{Val{Any}},
        ::Dual{Tuple{Val{Any}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        a::Dual{<:Memory},
    )
        return zero_dual(ccall(:jl_genericmemory_owner, Any, (Any,), primal(a)))
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
    @generated function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_alloc_array_1d}},
        ::Dual{Val{Vector{P}}},
        ::Dual{Tuple{Val{Any},Val{Int}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        ::Dual{Type{Vector{P}}},
        n::Dual{Int},
        args::Vararg{Dual},
    ) where {P}
        T = tangent_type(P)
        return quote
            _n = primal(n)
            y = ccall(:jl_alloc_array_1d, Vector{$P}, (Any, Int), Vector{$P}, _n)
            dy = ccall(:jl_alloc_array_1d, Vector{$T}, (Any, Int), Vector{$T}, _n)
            return Dual(y, dy)
        end
    end
    @generated function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_alloc_array_2d}},
        ::Dual{Val{Matrix{P}}},
        ::Dual{Tuple{Val{Any},Val{Int},Val{Int}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        ::Dual{Type{Matrix{P}}},
        m::Dual{Int},
        n::Dual{Int},
        args::Vararg{Dual},
    ) where {P}
        T = tangent_type(P)
        return quote
            _m, _n = primal(m), primal(n)
            y = ccall(:jl_alloc_array_2d, Matrix{$P}, (Any, Int, Int), Matrix{$P}, _m, _n)
            dy = ccall(:jl_alloc_array_2d, Matrix{$T}, (Any, Int, Int), Matrix{$T}, _m, _n)
            return Dual(y, dy)
        end
    end
    @generated function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_alloc_array_3d}},
        ::Dual{Val{Array{P,3}}},
        ::Dual{Tuple{Val{Any},Val{Int},Val{Int},Val{Int}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        ::Dual{Type{Array{P,3}}},
        l::Dual{Int},
        m::Dual{Int},
        n::Dual{Int},
        args::Vararg{Dual},
    ) where {P}
        T = tangent_type(P)
        return quote
            _l, _m, _n = primal(l), primal(m), primal(n)
            y = ccall(
                :jl_alloc_array_3d,
                Array{$P,3},
                (Any, Int, Int, Int),
                Array{$P,3},
                _l,
                _m,
                _n,
            )
            dy = ccall(
                :jl_alloc_array_3d,
                Array{$T,3},
                (Any, Int, Int, Int),
                Array{$T,3},
                _l,
                _m,
                _n,
            )
            return Dual(y, dy)
        end
    end
end
