@zero_derivative MinimalCtx Tuple{typeof(get_interpreter),Type{<:Mode}}
@zero_derivative MinimalCtx Tuple{
    typeof(build_rrule_checks),MooncakeInterpreter,Any,Bool,Bool
}
@zero_derivative MinimalCtx Tuple{typeof(is_primitive),Type,Type{<:Mode},Type,UInt}

@is_primitive MinimalCtx Tuple{
    typeof(build_derived_rrule),MooncakeInterpreter{C},Any,Any,Bool
} where {C}

# LazyFoRRule and DynamicFoRRule are the frule for `build_derived_rrule` in
# forward-over-reverse mode.  build_primitive_frule selects between the two via
# __build_primitive_frule (@generated):
#   • Concrete Trule → LazyFoRRule{Trule,Tfwd,Trvs}: fully-typed single-slot cache.
#   • Non-concrete Trule → DynamicFoRRule: Dict-keyed cache.
mutable struct LazyFoRRule{Trule,Tfwd,Trvs}
    rule::Trule
    fwd_dual_callable::Tfwd
    rvs_dual_callable::Trvs
    LazyFoRRule{Trule,Tfwd,Trvs}() where {Trule,Tfwd,Trvs} = new()
end

mutable struct DynamicFoRRule
    cache::Dict{Tuple{Any,Bool},Tuple{Any,Any,Any}}
    DynamicFoRRule() = new(Dict{Tuple{Any,Bool},Tuple{Any,Any,Any}}())
end

@generated function __build_primitive_frule(
    sig::Type{<:Tuple{typeof(build_derived_rrule),MooncakeInterpreter{C},SMI,S,Bool}}
) where {C,SMI,S}
    Trule = Core.Compiler.return_type(
        build_derived_rrule, Tuple{MooncakeInterpreter{C},SMI,S,Bool}
    )
    if !isconcretetype(Trule)
        return :(DynamicFoRRule())
    end
    inner = Trule <: DebugRRule ? fieldtype(Trule, :rule) : Trule
    if !hasfield(inner, :fwds_oc) || !hasfield(inner, :pb_oc_ref)
        return :(DynamicFoRRule())
    end
    fwds_oc_T = fieldtype(inner, :fwds_oc)
    rvs_oc_T = fieldtype(fieldtype(inner, :pb_oc_ref), :x)
    interp_fwd_T = MooncakeInterpreter{C,ForwardMode}
    Tfwd = Core.Compiler.return_type(build_frule, Tuple{interp_fwd_T,fwds_oc_T})
    Trvs = Core.Compiler.return_type(build_frule, Tuple{interp_fwd_T,rvs_oc_T})
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

# Cache-hit helper: reuse compiled (rule, fwd_dc, rvs_dc) with fresh Stacks.
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

# First-call compilation helper.
function _compile_for_rule(
    interp::MooncakeInterpreter{C}, sig_or_mi, sig, debug_mode::Bool
) where {C}
    @nospecialize sig_or_mi sig

    dri = generate_ir(interp, sig_or_mi; debug_mode, do_optimize=false)

    raw_rule = let
        optimized_fwd_ir = optimise_ir!(CC.copy(dri.fwd_ir))
        optimized_rvs_ir = optimise_ir!(CC.copy(dri.rvs_ir))
        fwd_oc = misty_closure(dri.fwd_ret_type, optimized_fwd_ir, dri.shared_data...)
        rvs_oc = misty_closure(dri.rvs_ret_type, optimized_rvs_ir, dri.shared_data...)
        nargs = num_args(dri.info)
        sig_flat = flatten_va_sig(sig, dri.isva, nargs)
        DerivedRule(sig_flat, fwd_oc, Ref(rvs_oc), dri.isva, Val(nargs))
    end

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
    ::Dual{typeof(build_derived_rrule)},
    _interp::Dual{<:MooncakeInterpreter{C}},
    _sig_or_mi::Dual,
    _sig::Dual,
    _debug_mode::Dual{Bool},
) where {Trule,Tfwd,Trvs,C}
    @nospecialize _sig_or_mi _sig
    debug_mode = primal(_debug_mode)
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
