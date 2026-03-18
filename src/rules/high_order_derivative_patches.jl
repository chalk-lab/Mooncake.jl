@zero_derivative MinimalCtx Tuple{typeof(get_interpreter),Type{<:Mode}}
@zero_derivative MinimalCtx Tuple{
    typeof(build_rrule_checks),MooncakeInterpreter,Any,Bool,Bool
}
@zero_derivative MinimalCtx Tuple{typeof(is_primitive),Type,Type{<:Mode},Type,UInt}

@is_primitive MinimalCtx Tuple{
    typeof(build_derived_rrule),MooncakeInterpreter{C},Any,Any,Bool
} where {C}

# LazyFoRRule is the frule for `build_derived_rrule` in forward-over-reverse mode.
# In HVPCache, grad_f calls prepare_gradient_cache → build_rrule → build_derived_rrule
# on every value_and_hvp!! call, so caching is essential: the first call compiles the
# inner DerivedRule and dual callables; subsequent calls reuse them via _copy (cheap).
#
# build_primitive_frule returns a LazyFoRRule as the frule. LazyFoRRule is a callable struct
# rather than a plain function so it can cache compiled artifacts.
# __build_primitive_frule is @generated: it uses Core.Compiler.return_type to infer
# Trule/Tfwd/Trvs so the LazyFoRRule is fully typed at construction. The three artifact
# fields are uninitialized until the first call, at which point they are populated and
# reused on every subsequent call with no virtual dispatch. Each LazyFoRRule instance is
# captured at exactly one call site in the compiler IR (LazyFoRRule is not safe_for_literal),
# so at most one initialization ever occurs.
mutable struct LazyFoRRule{Trule,Tfwd,Trvs}
    rule::Trule
    fwd_dual_callable::Tfwd
    rvs_dual_callable::Trvs
    LazyFoRRule{Trule,Tfwd,Trvs}() where {Trule,Tfwd,Trvs} = new()
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
    # LazyFoRRule{Any,Any,Any} is functionally correct (first-call path compiles at runtime)
    # but type-unstable.
    if !isconcretetype(Trule)
        return :(LazyFoRRule{Any,Any,Any}())
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

# LazyFoRRule is the frule for build_derived_rrule. The primal inside the returned Dual is a
# DerivedRule (an rrule); LazyFoRRule itself is the frule that differentiates *through*
# the rrule construction:
#
#   build_derived_rrule : (interp, sig_or_mi, sig, debug_mode) → rrule
#   LazyFoRRule         : (Dual(build_derived_rrule, ·), Dual(interp, t_interp), ...) → Dual(rrule, t_rule)
#                         where t_rule = J_{build_derived_rrule} · (t_interp, ...)
function (cache::LazyFoRRule{Trule,Tfwd,Trvs})(
    ::Dual{typeof(build_derived_rrule)},
    _interp::Dual{<:MooncakeInterpreter{C}},
    _sig_or_mi::Dual,
    _sig::Dual,
    _debug_mode::Dual{Bool},
) where {Trule,Tfwd,Trvs,C}
    @nospecialize _sig_or_mi _sig

    debug_mode = primal(_debug_mode)

    # Cache hit: reuse compiled rule + dual callables. Neither `debug_mode` nor the
    # signature (`sig_or_mi`) is re-checked: build_primitive_frule places one LazyFoRRule
    # per call site in the compiled IR, and the call site is inside the closure for a
    # fixed `grad_f`, so both the config and the signature of the inner function are
    # invariant across value_and_hvp!! calls for the lifetime of this closure.
    #
    # LazyFoRRule is shared across value_and_hvp!! calls (like LazyDerivedRule), but
    # unlike LazyDerivedRule it cannot guarantee its Stacks are balanced on return, so
    # each call gets fresh empty Stacks via _copy (compiled code is shared, not copied)
    # and fresh zero tangent Stacks via zero_tangent.
    if isdefined(cache, :rule)
        new_rule = _copy(cache.rule)
        # _copy(Stack{T}) resets each primal Stack to empty. Regenerate captures_tangent
        # from the fresh primal so tangent Stacks are empty and size-consistent.
        # fwd_oc and rvs_oc share the same comms Stack objects from shared_data
        # (fwd_oc.captures[i] === rvs_oc.captures[i]), so their tangents must also be
        # aliased: the fwds tangent pass writes to comms tangent Stacks and the rvs
        # tangent pass reads from the same objects. zero_tangent uses an IdDict
        # internally, so calling it jointly on both captures tuples ensures
        # captures_tangent[1][i] === captures_tangent[2][i] for aliased primal objects.
        inner_rule = debug_mode ? new_rule.rule : new_rule
        captures_tangent = zero_tangent((
            inner_rule.fwds_oc.oc.captures, inner_rule.pb_oc_ref[].oc.captures
        ))
        inner_tangent = Tangent((;
            fwds_oc=MistyClosureTangent(
                captures_tangent[1], _copy(cache.fwd_dual_callable)
            ),
            pb_oc_ref=MutableTangent((;
                x=PossiblyUninitTangent(
                    MistyClosureTangent(captures_tangent[2], _copy(cache.rvs_dual_callable))
                )
            )),
            nargs=NoTangent(),
        ))
        rule_tangent = debug_mode ? Tangent((; rule=inner_tangent)) : inner_tangent
        return Dual(new_rule, rule_tangent)
    end

    # First call: compile the rule and its dual callables, then populate the cache.
    interp = primal(_interp)
    sig_or_mi = primal(_sig_or_mi)
    sig = primal(_sig)

    # Derive unoptimized forwards- and reverse-pass IR.
    dri = generate_ir(interp, sig_or_mi; debug_mode, do_optimize=false)

    # Optimize and build the primal DerivedRule.
    raw_rule = let
        optimized_fwd_ir = optimise_ir!(CC.copy(dri.fwd_ir))
        optimized_rvs_ir = optimise_ir!(CC.copy(dri.rvs_ir))
        fwd_oc = misty_closure(dri.fwd_ret_type, optimized_fwd_ir, dri.shared_data...)
        rvs_oc = misty_closure(dri.rvs_ret_type, optimized_rvs_ir, dri.shared_data...)

        nargs = num_args(dri.info)
        sig = flatten_va_sig(sig, dri.isva, nargs)
        DerivedRule(sig, fwd_oc, Ref(rvs_oc), dri.isva, Val(nargs))
    end

    # Build forward-mode dual callables for the fwd and rvs passes.
    fwd_dual_callable, rvs_dual_callable, raw_rule_tangent = let
        # Use a forward-mode interpreter to block inlining of frules during optimisation.
        interp_forward = MooncakeInterpreter(C, ForwardMode; world=interp.world)

        optimized_fwd_ir = optimise_ir!(dri.fwd_ir; interp=interp_forward)
        optimized_rvs_ir = optimise_ir!(dri.rvs_ir; interp=interp_forward)
        fwd_oc = misty_closure(dri.fwd_ret_type, optimized_fwd_ir, dri.shared_data...)
        rvs_oc = misty_closure(dri.rvs_ret_type, optimized_rvs_ir, dri.shared_data...)

        # fwd_oc and rvs_oc share the same comms Stack objects from shared_data
        # (fwd_oc.captures[i] === rvs_oc.captures[i]), so their tangents must also be
        # aliased: the fwds tangent pass writes to comms tangent Stacks and the rvs
        # tangent pass reads from the same objects. zero_tangent uses an IdDict
        # internally, so calling it jointly on both captures tuples ensures
        # captures_tangent[1][i] === captures_tangent[2][i] for aliased primal objects.
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
    cache.rule = rule
    cache.fwd_dual_callable = fwd_dual_callable
    cache.rvs_dual_callable = rvs_dual_callable
    return Dual(rule, rule_tangent)
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
