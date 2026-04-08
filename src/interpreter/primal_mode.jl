#
# Design note for `build_primal`.
#
# `build_primal` provides an IR-lifted forward path. It:
# 1. resolves and normalises primal IR for `f`
# 2. rebuilds `f` as a primal/lifted executable rather than re-running Julia dispatch on
#    AD wrapper types
# 3. recursively uses `build_primal(g, ...)` for resolved non-primitive callees so
#    dispatch is preserved through the resolved call graph
# 4. keeps primitive / intrinsic / builtin leaves on explicit rule paths
# 5. keeps an explicit fallback only for genuinely unresolved dynamic calls
#
# The recursive caching / cycle-breaking part is still the same problem current forward
# mode solves via `LazyPrimal`, `DynamicPrimal`, and `DerivedPrimal`; the remaining design
# work is mainly the lifted execution / representation choice, not inventing a separate
# recursion strategy.
#
# Because Mooncake's normalised IR lowers constructors to `__new__`, lifted `__new__`
# rules may be enough to handle concrete struct construction without explicit lifted
# constructor methods. That could support a fully lifted forward path whose width-aware
# dual type is selected by `dual_type(::Val{N}, P)` and whose tangent storage is selected by
# `tangent_type(::Val{N}, P)`. The default pairing is `Dual{P,tangent_type(Val(N), P)}`,
# but other overloads may choose `NDual` or another width-aware dual type. For nfwd in
# particular, direct public support is still narrower than the internal packed tangent
# machinery: today its direct inputs are IEEE-float scalars, complex IEEE-float scalars,
# and dense arrays with those element types, while direct outputs may also be tuples
# thereof, even though the internal packing code can represent broader tangent trees.
# That could potentially share machinery with broader transforms such as batched /
# vmap-style derived overloads. Candidate width-aware tangent layouts for that path
# include:
# - today's lane-outer `NTangent`, i.e. a tuple of ordinary tangents
# - a packed `NTangent` with lane-inner scalar leaves closer to NDual
# - overloading `tangent_type(::Val{N}, ...)` only for `Array{<:IEEEFloat}` /
#   `Array{<:Complex{<:IEEEFloat}}` so array leaves use NDual-like packed storage
# - overloading `tangent_type(::Val{N}, ...)` only for those arrays so tangents use one
#   extra lane dimension instead
#
# A natural next step is to make forward mode generic over a width-aware dual-type
# protocol, rather than assuming concrete `Dual{P,T}` everywhere. A minimal proposed
# protocol would be:
# - `primal(x)`
# - `tangent(x)`
# - `dual_type(::Val{N}, ::Type{P})`
# - construction via `dual_type(Val(N), P)(x, dx)`, which may yield `Dual`, `NDual`, or
#   another width-aware dual type
# - width-aware zero/uninit constructors, e.g. `zero_dual(::Val{N}, x)` and
#   `uninit_dual(::Val{N}, x)`
#
# That keeps width-aware dual-type choice type-driven and inference-friendly while
# separating dual-type selection from tangent-storage selection. The type-stability
# constraint is that `dual_type(Val(N), P)` must remain foldable to a concrete dual type
# for concrete `P`; runtime dual-type switching would immediately weaken inference through
# the transformed IR.
#
# The current `nfwd.jl` / `NfwdMooncake.jl` split is also intended to be transitional. A
# plausible cleanup is to keep only `nfwd.jl`, with the higher-level Mooncake integration
# logic folded into that file once the lifted-forward path and width-aware dual-type
# interfaces settle.

# Check if a type contains Union{} (bottom type) anywhere in its structure.
# This can happen with unreachable code or failed type inference.
@inline contains_bottom_type(T) = _contains_bottom_type(T, Base.IdSet{Any}())

function _contains_bottom_type(T, seen::Base.IdSet{Any})
    T === Union{} && return true
    if T isa Union
        return _contains_bottom_type(T.a, seen) || _contains_bottom_type(T.b, seen)
    elseif T isa TypeVar
        T in seen && return false
        push!(seen, T)
        return _contains_bottom_type(T.ub, seen)
    elseif T isa UnionAll
        T in seen && return false
        push!(seen, T)
        return _contains_bottom_type(T.body, seen)
    elseif T isa DataType
        T in seen && return false
        push!(seen, T)
        for p in T.parameters
            _contains_bottom_type(p, seen) && return true
        end
        return false
    else
        return false
    end
end

struct IRfwdMode{N} end

function _fwd_dual_type end
function _fwd_zero_dual end
function _fwd_uninit_dual end

@inline build_primitive_frule(
    ::Any, interp, sig; debug_mode=false, silence_debug_messages=true
) = nothing

function _fwd_dual_type(::IRfwdMode{N}, ::Type{P}) where {N,P}
    return dual_type(Val(N), P)
end

function _fwd_zero_dual(::IRfwdMode{N}, x) where {N}
    return zero_dual(Val(N), x)
end

function _fwd_uninit_dual(::IRfwdMode{N}, x) where {N}
    return uninit_dual(Val(N), x)
end

function build_frule(
    args...;
    chunk_size=nothing,
    debug_mode=false,
    silence_debug_messages=true,
    skip_world_age_check=false,
)
    primals = map(x -> x isa Dual ? primal(x) : x, args)
    sig = _typeof(primals)
    interp = get_interpreter(ForwardMode)
    mode = IRfwdMode{something(chunk_size, 1)}()
    return build_frule(
        interp,
        sig;
        call_target=first(primals),
        debug_mode,
        silence_debug_messages,
        skip_world_age_check,
        tangent_mode=mode,
    )
end

"""
    build_frule(
        interp::MooncakeInterpreter{C},
        sig_or_mi;
        call_target=nothing,
        debug_mode=false,
        silence_debug_messages=true,
        skip_world_age_check=false,
        tangent_mode=nothing,
    ) where {C}

Returns a callable which performs forward-mode AD for `sig_or_mi`.

`tangent_mode` selects the width-aware dual type and primitive-lowering mode used to derive
the rule. The default is the width-1 IR-lifted primal path. Use `NDualMode{N}()`
explicitly for direct nfwd rules.
"""
function build_frule(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    call_target=nothing,
    debug_mode=false,
    silence_debug_messages=true,
    skip_world_age_check=false,
    tangent_mode=nothing,
) where {C}
    @nospecialize sig_or_mi

    tangent_mode === nothing && (tangent_mode = IRfwdMode{1}())

    if !skip_world_age_check && Base.get_world_counter() > interp.world
        throw(
            ArgumentError(
                "World age associated to interp is behind current world age. Please " *
                "create a new interpreter for the current world age.",
            ),
        )
    end

    if !silence_debug_messages && debug_mode
        @info "Compiling frule for $sig_or_mi in debug mode. Disable for best performance."
    end

    sig = _get_sig(sig_or_mi)
    if tangent_mode isa getfield(getfield(@__MODULE__, :Nfwd), :NDualMode)
        if is_primitive(C, ForwardMode, sig, interp.world)
            rule = build_primitive_frule(
                tangent_mode, interp, sig; debug_mode, silence_debug_messages
            )
            isnothing(rule) && (rule = build_primitive_frule(sig))
            return debug_mode ? DebugFRule(rule) : rule
        end
        rule = NfwdMooncake.build_nfwd_frule(
            sig;
            chunk_size=typeof(tangent_mode).parameters[1],
            debug_mode=false,
            silence_debug_messages,
        )
        return debug_mode ? DebugFRule(rule) : rule
    end

    if is_primitive(C, ForwardMode, sig, interp.world)
        rule = build_primitive_frule(
            tangent_mode, interp, sig; debug_mode, silence_debug_messages
        )
        isnothing(rule) && (rule = build_primitive_frule(sig))
        return debug_mode ? DebugFRule(rule) : rule
    end

    return build_primal(
        interp, sig_or_mi; call_target, debug_mode, skip_world_age_check=true, tangent_mode
    )
end

"""
    build_primal(
        interp::MooncakeInterpreter{C},
        sig_or_mi;
        skip_world_age_check=false,
        do_inline=true,
    ) where {C}

Build a callable from resolved, normalised primal IR. The result can be called directly on
ordinary primal inputs, or on forward `Dual` inputs as an IR-lifted alternative to a
derived forward rule.
"""
function build_primal(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    call_target=nothing,
    debug_mode=false,
    skip_world_age_check=false,
    do_inline=true,
    tangent_mode=IRfwdMode{1}(),
) where {C}
    @nospecialize sig_or_mi

    if !skip_world_age_check && Base.get_world_counter() > interp.world
        throw(
            ArgumentError(
                "World age associated to interp is behind current world age. Please " *
                "create a new interpreter for the current world age.",
            ),
        )
    end

    lock(MOONCAKE_INFERENCE_LOCK)
    try
        oc_cache_key = ClosureCacheKey(
            interp.world, (sig_or_mi, :primal, do_inline, tangent_mode)
        )
        if haskey(interp.oc_cache, oc_cache_key)
            cached = _copy(interp.oc_cache[oc_cache_key])
            derived = DerivedPrimal(
                cached.primal_oc,
                cached.lifted_oc,
                cached.key,
                call_target,
                cached.world,
                cached.tangent_mode,
            )
            return debug_mode ? DebugPrimal(derived) : derived
        end

        primal_ir, info, captures = generate_primal_ir(interp, sig_or_mi; do_inline)
        primal_oc = misty_closure(
            info.ret_type, primal_ir, captures...; isva=info.isva, do_compile=true
        )
        lifted_ir, lifted_captures, dual_ret_type = generate_lifted_primal_ir(
            interp, sig_or_mi; do_inline, tangent_mode
        )
        lifted_oc = misty_closure(
            dual_ret_type, lifted_ir, lifted_captures...; isva=info.isva, do_compile=true
        )
        derived = DerivedPrimal(
            primal_oc, lifted_oc, sig_or_mi, nothing, interp.world, tangent_mode
        )
        interp.oc_cache[oc_cache_key] = derived
        derived = DerivedPrimal(
            primal_oc, lifted_oc, sig_or_mi, call_target, interp.world, tangent_mode
        )
        return debug_mode ? DebugPrimal(derived) : derived
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

function build_primal(
    args...;
    debug_mode=false,
    skip_world_age_check=false,
    do_inline=true,
    tangent_mode=IRfwdMode{1}(),
)
    primals = map(x -> x isa Dual ? primal(x) : x, args)
    sig = _typeof(primals)
    interp = get_interpreter(ForwardMode)
    return build_primal(
        interp,
        sig;
        call_target=first(primals),
        debug_mode,
        skip_world_age_check,
        do_inline,
        tangent_mode,
    )
end

struct PrimalIRInfo
    isva::Bool
    nargs::Int
    ret_type::Type
end

struct DualRuleInfo
    isva::Bool
    nargs::Int
    dual_ret_type::Type
end

struct PrimalInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
end

struct DualInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
    is_used::Vector{Bool}
    debug_mode::Bool
    tangent_mode
end

struct DerivedPrimal{Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tmode}
    primal_oc::Tprimal_oc
    lifted_oc::Tlifted_oc
    key::Tkey
    call_target::Tcall_target
    world::UInt
    tangent_mode::Tmode
end

const DerivedFRule = DerivedPrimal

"""
    DebugPrimal(primal)

Construct a callable equivalent to `primal` but with additional type checking on lifted
`Dual` calls. Ordinary primal calls are forwarded unchanged.
"""
struct DebugPrimal{Tprimal}
    primal::Tprimal
end

_copy(x::P) where {P<:DebugPrimal} = P(_copy(x.primal))

@inline (primal::DebugPrimal)(x::Vararg{Any,N}) where {N} = primal.primal(x...)

@noinline function (primal::DebugPrimal)(x::Vararg{Dual,N}) where {N}
    verify_args(primal.primal, x)
    verify_dual_inputs(x)
    y = primal.primal(x...)
    verify_dual_output(x, y)
    return y
end

@noinline function (primal::DebugPrimal)(f::Dual, x::Vararg{Dual,N}) where {N}
    args = (f, x...)
    verify_args(primal.primal, args)
    verify_dual_inputs(args)
    y = primal.primal(args...)
    verify_dual_output(args, y)
    return y
end

struct LazyPrimal{K,Tcache}
    key::K
    world::UInt
    primal_ref::Tcache
    tangent_mode
end

function LazyPrimal(key, world::UInt, tangent_mode=IRfwdMode{1}())
    LazyPrimal{typeof(key),Base.RefValue{Any}}(key, world, Ref{Any}(), tangent_mode)
end

struct DynamicPrimal{V}
    cache::V
    world::UInt
    tangent_mode
end

function DynamicPrimal(world::UInt, tangent_mode=IRfwdMode{1}())
    DynamicPrimal(Dict{Any,Any}(), world, tangent_mode)
end

function generate_primal_ir(
    interp::MooncakeInterpreter, sig_or_mi; do_inline=true
)::Tuple{IRCode,PrimalIRInfo,Tuple}
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    nargs = length(primal_ir.argtypes)
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)
    lifted_ir = CC.copy(primal_ir)
    pushfirst!(lifted_ir.argtypes, Any)
    captures = Any[]
    info = PrimalInfo(primal_ir, interp)
    for (n, inst) in enumerate(lifted_ir.stmts)
        modify_primal_stmts!(stmt(inst), lifted_ir, SSAValue(n), captures, info)
    end
    lifted_ir = CC.compact!(lifted_ir)
    captures_tuple = (captures...,)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)
    lifted_ir = optimise_ir!(lifted_ir; do_inline)
    return (
        lifted_ir, PrimalIRInfo(isva, nargs, compute_ir_rettype(lifted_ir)), captures_tuple
    )
end

@inline get_capture(captures::T, n::Int) where {T} = captures[n]

"""
    const_dual!(captures::Vector{Any}, stmt)::Union{Dual,Int}

Build a width-aware dual type from `stmt`, with zero / uninitialised tangent. If the
resulting dual is a bits type, return it directly. Otherwise push it into `captures` and
return its location.
"""
function const_dual!(
    captures::Vector{Any}, stmt, tangent_mode=IRfwdMode{1}()
)::Union{Dual,Int}
    v = get_const_primal_value(stmt)
    x = _fwd_uninit_dual(tangent_mode, v)
    if safe_for_literal(v)
        return x
    else
        push!(captures, x)
        return length(captures)
    end
end

const ATTACH_AFTER = true
const ATTACH_BEFORE = false

modify_fwd_ad_stmts!(::Nothing, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo) = nothing

modify_fwd_ad_stmts!(::GotoNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo) = nothing

function modify_fwd_ad_stmts!(
    stmt::GotoIfNot, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    replace_call!(dual_ir, ssa, Expr(:call, _primal, inc_args(stmt).cond))
    new_gotoifnot_inst = new_inst(Core.GotoIfNot(ssa, stmt.dest))
    CC.insert_node!(dual_ir, ssa, new_gotoifnot_inst, ATTACH_AFTER)
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::GlobalRef, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if isconst(stmt)
        d = const_dual!(captures, stmt, info.tangent_mode)
        if d isa Int
            replace_call!(dual_ir, ssa, Expr(:call, get_capture, Argument(1), d))
        else
            replace_call!(dual_ir, ssa, Expr(:call, identity, d))
        end
    else
        new_ssa = CC.insert_node!(dual_ir, ssa, new_inst(stmt), ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, Expr(:call, _fwd_zero_dual, info.tangent_mode, new_ssa))
    end

    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::ReturnNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    isdefined(stmt, :val) || return nothing
    if stmt.val isa Union{Argument,SSAValue}
        replace_call!(dual_ir, ssa, ReturnNode(__inc(stmt.val)))
        return nothing
    end
    d = const_dual!(captures, stmt.val, info.tangent_mode)
    if d isa Int
        get_dual = Expr(:call, get_capture, Argument(1), d)
        get_dual_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_dual), ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, ReturnNode(get_dual_ssa))
    else
        replace_call!(dual_ir, ssa, ReturnNode(d))
    end
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = _fwd_uninit_dual(
            info.tangent_mode, get_const_primal_value(stmt.values[n])
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        _fwd_dual_type(info.tangent_mode, CC.widenconst(get_ir(dual_ir, ssa, :type))),
    )
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PiNode, dual_ir::IRCode, ssa::SSAValue, ::Vector{Any}, info::DualInfo
)
    stmt == PiNode(nothing, Union{}) && return replace_call!(dual_ir, ssa, nothing)
    if stmt.val isa Union{Argument,SSAValue}
        v = __inc(stmt.val)
        primal_ssa = CC.insert_node!(
            dual_ir,
            ssa,
            new_inst(Expr(:call, getfield, v, QuoteNode(:primal))),
            ATTACH_BEFORE,
        )
        tangent_ssa = CC.insert_node!(
            dual_ir,
            ssa,
            new_inst(Expr(:call, getfield, v, QuoteNode(:tangent))),
            ATTACH_BEFORE,
        )
        refined_primal_ssa = CC.insert_node!(
            dual_ir,
            ssa,
            new_inst(PiNode(primal_ssa, CC.widenconst(stmt.typ))),
            ATTACH_BEFORE,
        )
        replace_call!(dual_ir, ssa, Expr(:call, Dual, refined_primal_ssa, tangent_ssa))
    else
        v = _fwd_uninit_dual(info.tangent_mode, get_const_primal_value(stmt.val))
        replace_call!(
            dual_ir,
            ssa,
            PiNode(v, _fwd_dual_type(info.tangent_mode, CC.widenconst(stmt.typ))),
        )
    end
    set_ir!(dual_ir, ssa, :type, _fwd_dual_type(info.tangent_mode, CC.widenconst(stmt.typ)))
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::UpsilonNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if !(stmt.val isa Union{Argument,SSAValue})
        stmt = UpsilonNode(
            _fwd_uninit_dual(info.tangent_mode, get_const_primal_value(stmt.val))
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        _fwd_dual_type(info.tangent_mode, CC.widenconst(get_ir(dual_ir, ssa, :type))),
    )
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiCNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = _fwd_uninit_dual(
            info.tangent_mode, get_const_primal_value(stmt.values[n])
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        _fwd_dual_type(info.tangent_mode, CC.widenconst(get_ir(dual_ir, ssa, :type))),
    )
    return nothing
end

@static if isdefined(Core, :EnterNode)
    function modify_fwd_ad_stmts!(
        ::Core.EnterNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo
    )
        return nothing
    end
end

__get_primal(x::Dual) = primal(x)

function modify_fwd_ad_stmts!(
    stmt::Expr, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        sig_types = map(raw_args) do x
            t = CC.widenconst(get_forward_primal_type(info.primal_ir, x))
            return contains_bottom_type(t) ? Any : t
        end
        sig = Tuple{sig_types...}
        args = map(__inc, raw_args)

        if !info.is_used[ssa.id] && get_const_primal_value(args[1]) == getfield
            fwds = new_inst(Expr(:call, __fwds_pass_no_ad!, args...))
            replace_call!(dual_ir, ssa, fwds)
            return nothing
        end

        if all(T -> tangent_type(T) === NoTangent, @view(sig_types[2:end])) &&
            !is_primitive(context_type(info.interp), ForwardMode, sig, info.interp.world)
            primal_ssa = CC.insert_node!(
                dual_ir,
                ssa,
                new_inst(Expr(:call, __fwds_pass_no_ad!, args...)),
                ATTACH_BEFORE,
            )
            replace_call!(
                dual_ir, ssa, Expr(:call, _fwd_zero_dual, info.tangent_mode, primal_ssa)
            )
            return nothing
        end

        dual_args = map(args) do arg
            arg isa Union{Argument,SSAValue} && return arg
            return _fwd_uninit_dual(info.tangent_mode, get_const_primal_value(arg))
        end

        if is_primitive(context_type(info.interp), ForwardMode, sig, info.interp.world)
            rule = build_primitive_frule(
                info.tangent_mode, info.interp, sig; debug_mode=info.debug_mode
            )
            isnothing(rule) && (rule = build_primitive_frule(sig))
            if safe_for_literal(rule)
                replace_call!(dual_ir, ssa, Expr(:call, rule, dual_args...))
            else
                push!(captures, rule)
                get_rule = Expr(:call, get_capture, Argument(1), length(captures))
                rule_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
                replace_call!(dual_ir, ssa, Expr(:call, rule_ssa, dual_args...))
            end
        else
            push!(
                captures,
                if isexpr(stmt, :invoke)
                    LazyPrimal(get_mi(stmt.args[1]), info.interp.world, info.tangent_mode)
                else
                    DynamicPrimal(info.interp.world, info.tangent_mode)
                end,
            )
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            replace_call!(dual_ir, ssa, Expr(:call, rule_ssa, dual_args...))
        end
    elseif isexpr(stmt, :boundscheck)
        inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        bc_ssa = CC.insert_node!(dual_ir, ssa, inst, ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, Expr(:call, _fwd_zero_dual, info.tangent_mode, bc_ssa))
    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(dual_ir, ssa, nothing)
    elseif Meta.isexpr(stmt, :copyast)
        new_copyast_inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        new_copyast_ssa = CC.insert_node!(dual_ir, ssa, new_copyast_inst, ATTACH_BEFORE)
        replace_call!(
            dual_ir, ssa, Expr(:call, _fwd_zero_dual, info.tangent_mode, new_copyast_ssa)
        )
    elseif Meta.isexpr(stmt, :loopinfo)
    elseif isexpr(stmt, :throw_undef_if_not)
        primal_cond = Expr(:call, _primal, inc_args(stmt).args[2])
        replace_call!(dual_ir, ssa, primal_cond)
        new_undef_inst = new_inst(Expr(:throw_undef_if_not, stmt.args[1], ssa))
        CC.insert_node!(dual_ir, ssa, new_undef_inst, ATTACH_AFTER)
    elseif isexpr(stmt, :enter) || isexpr(stmt, :leave) || isexpr(stmt, :pop_exception)
    else
        msg = "Expressions of type `:$(stmt.head)` are not yet supported in forward mode"
        throw(ArgumentError(msg))
    end
    return nothing
end

get_forward_primal_type(ir::CC.IRCode, a::Argument) = ir.argtypes[a.n]
get_forward_primal_type(ir::CC.IRCode, ssa::SSAValue) = get_ir(ir, ssa, :type)
get_forward_primal_type(::CC.IRCode, x::QuoteNode) = _typeof(x.value)
get_forward_primal_type(::CC.IRCode, x) = _typeof(x)
function get_forward_primal_type(::CC.IRCode, x::GlobalRef)
    return isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.binding.ty
end
function get_forward_primal_type(::CC.IRCode, x::Expr)
    x.head === :boundscheck && return Bool
    return error("Unrecognised expression $x found in argument slot.")
end

function modify_primal_stmts!(
    stmt, primal_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::PrimalInfo
)
    set_stmt!(primal_ir, ssa, inc_args(stmt))
end

function modify_primal_stmts!(
    stmt::Expr, primal_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::PrimalInfo
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        sig_types = map(raw_args) do x
            t = CC.widenconst(get_forward_primal_type(info.primal_ir, x))
            contains_bottom_type(t) ? Any : t
        end
        sig = Tuple{sig_types...}
        interp = info.interp
        if is_primitive(context_type(interp), ForwardMode, sig, interp.world)
            set_stmt!(primal_ir, ssa, inc_args(stmt))
        else
            push!(
                captures,
                if isexpr(stmt, :invoke)
                    LazyPrimal(get_mi(stmt.args[1]), interp.world)
                else
                    DynamicPrimal(interp.world, IRfwdMode{1}())
                end,
            )
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(primal_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            args = map(__inc, raw_args)
            replace_call!(primal_ir, ssa, Expr(:call, rule_ssa, args...))
        end
    else
        set_stmt!(primal_ir, ssa, inc_args(stmt))
    end
    return nothing
end

@inline function (primal::DerivedPrimal)(args::Vararg{Any,N}) where {N}
    return if isnothing(primal.call_target)
        primal.primal_oc(args...)
    else
        primal.primal_oc(primal.call_target, args...)
    end
end

@inline function (primal::DerivedPrimal{Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tmode})(
    x::Dual, args::Vararg{Dual,N}
) where {Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tmode,N}
    isnothing(primal.call_target) && throw(
        ArgumentError(
            "DerivedPrimal without a stored call target cannot be called on Dual arguments directly. Use LazyPrimal or DynamicPrimal with the primal callable as the first argument.",
        ),
    )
    return primal.lifted_oc(
        _fwd_zero_dual(primal.tangent_mode, primal.call_target), x, args...
    )
end

@inline function (primal::DerivedPrimal{Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tmode})(
    ::Dual, x::Dual, args::Vararg{Dual,N}
) where {Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tmode,N}
    isnothing(primal.call_target) && throw(
        ArgumentError(
            "DerivedPrimal without a stored call target cannot be called on Dual arguments directly. Use LazyPrimal or DynamicPrimal with the primal callable as the first argument.",
        ),
    )
    return primal.lifted_oc(
        _fwd_zero_dual(primal.tangent_mode, primal.call_target), x, args...
    )
end

@inline function (primal::DerivedPrimal{Tprimal_oc,Tlifted_oc,Tkey,Nothing,Tmode})(
    f::Dual, x::Dual, args::Vararg{Dual,N}
) where {Tprimal_oc,Tlifted_oc,Tkey,Tmode,N}
    return primal.lifted_oc(f, x, args...)
end

@static if VERSION < v"1.11-"
    @inline function __call_rule(
        primal::DerivedPrimal{
            Tprimal_oc,MistyClosure{OpaqueClosure{A,R}},Tkey,Tcall_target,Tmode
        },
        args,
    ) where {Tprimal_oc,A,R,Tkey,Tcall_target,Tmode}
        return primal(args...)::R
    end

    @inline __call_rule(primal::DebugPrimal, args) = primal(args...)
    @inline __call_rule(primal::LazyPrimal, args) = primal(args...)
    @inline __call_rule(primal::DynamicPrimal, args) = primal(args...)
end

function _copy(x::P) where {P<:DerivedPrimal}
    return P(
        replace_captures(x.primal_oc, _copy(x.primal_oc.oc.captures)),
        replace_captures(x.lifted_oc, _copy(x.lifted_oc.oc.captures)),
        x.key,
        x.call_target,
        x.world,
        x.tangent_mode,
    )
end

function verify_args(primal::DerivedPrimal, x)
    sig = primal.key isa Core.MethodInstance ? primal.key.specTypes : primal.key
    sig isa Type{<:Tuple} || return nothing
    Tx = if isnothing(primal.call_target)
        Tuple{map(_typeof ∘ Mooncake.primal, x)...}
    else
        Tuple{_typeof(primal.call_target),map(_typeof ∘ Mooncake.primal, x)...}
    end
    Tx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tx do not subtype primal signature, $sig"))
end

function __verify_sig(primal::DerivedPrimal, fx::Tuple)
    sig = primal.key isa Core.MethodInstance ? primal.key.specTypes : primal.key
    sig isa Type{<:Tuple} || return nothing
    Tx = Tuple{map(_typeof ∘ Mooncake.primal, fx)...}
    Tx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tx do not subtype primal signature, $sig"))
end

__verify_sig(primal::DebugPrimal, fx::Tuple) = __verify_sig(primal.primal, fx)

@inline function __value_and_gradient!!(
    rule::Union{DerivedPrimal,DebugPrimal,LazyPrimal,DynamicPrimal}, fx::Vararg{CoDual,N}
) where {N}
    return _irfwd_value_and_gradient!!(rule, 1, fx...)
end

@inline function __value_and_pullback!!(
    rule::Union{DerivedPrimal,DebugPrimal,LazyPrimal,DynamicPrimal},
    ȳ::T,
    fx::Vararg{CoDual,N};
    y_cache=nothing,
) where {T<:IEEEFloat,N}
    y, gradient = _irfwd_value_and_gradient!!(rule, 1, fx...)
    value = if y_cache === nothing
        _copy_output(y)
    else
        _copy_to_output!!(y_cache, y)
    end
    return value, tuple_map(g -> g isa NoTangent ? g : _scale(ȳ, g), gradient)
end

@inline function (primal::LazyPrimal)(args::Vararg{Any,N}) where {N}
    if !isassigned(primal.primal_ref)
        interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=primal.world)
        primal.primal_ref[] = build_primal(
            interp, primal.key; skip_world_age_check=true, tangent_mode=primal.tangent_mode
        )
    end
    return primal.primal_ref[](args...)
end

@inline function (primal::LazyPrimal)(f::Dual, x::Dual, args::Vararg{Dual,N}) where {N}
    if !isassigned(primal.primal_ref)
        interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=primal.world)
        primal.primal_ref[] = build_primal(
            interp, primal.key; skip_world_age_check=true, tangent_mode=primal.tangent_mode
        )
    end
    return primal.primal_ref[](f, x, args...)
end

_copy(x::P) where {P<:LazyPrimal} = P(x.key, x.world, Ref{Any}(), x.tangent_mode)

@inline function (dynamic::DynamicPrimal)(f, args::Vararg{Any,N}) where {N}
    sig = Tuple{map(x -> x isa Dual ? _typeof(primal(x)) : _typeof(x), (f, args...))...}
    rule = get(dynamic.cache, sig, nothing)
    if rule === nothing
        interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=dynamic.world)
        rule = build_primal(
            interp, sig; skip_world_age_check=true, tangent_mode=dynamic.tangent_mode
        )
        dynamic.cache[sig] = rule
    end
    return rule(f, args...)
end

@inline function (dynamic::DynamicPrimal)(f::Dual, x::Dual, args::Vararg{Dual,N}) where {N}
    sig = Tuple{map(x -> x isa Dual ? _typeof(primal(x)) : _typeof(x), (f, x, args...))...}
    interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=dynamic.world)
    rule = get(dynamic.cache, sig, nothing)
    if rule === nothing
        rule = build_primal(
            interp, sig; skip_world_age_check=true, tangent_mode=dynamic.tangent_mode
        )
        dynamic.cache[sig] = rule
    end
    return rule(f, x, args...)
end

_copy(x::P) where {P<:DynamicPrimal} = P(Dict{Any,Any}(), x.world, x.tangent_mode)

function dual_ret_type(primal_ir::IRCode, tangent_mode=IRfwdMode{1}())
    return _fwd_dual_type(tangent_mode, compute_ir_rettype(primal_ir))
end

function generate_dual_ir(
    interp::MooncakeInterpreter,
    sig_or_mi;
    debug_mode=false,
    do_inline=true,
    tangent_mode=IRfwdMode{1}(),
)
    lifted_ir, captures, dual_ret = generate_lifted_primal_ir(
        interp, sig_or_mi; do_inline, tangent_mode
    )
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)
    return lifted_ir, captures, DualRuleInfo(isva, length(primal_ir.argtypes), dual_ret)
end

#
# nfwd-backed primitive lowering for IRfwdMode
#
# `IRfwdMode` keeps derived code on ordinary primal values plus `NTangent` lanes, but some
# primitive call sites reached through rule dispatch can be lowered to nfwd's NDual-backed
# rules when the primitive signature and inferred output type are structurally supported,
# and the corresponding NDual-lifted primitive dispatch actually exists.

@inline _nfwd_supported_primal_type(::Type{<:IEEEFloat}) = true
@inline _nfwd_supported_primal_type(::Type{<:Complex{<:IEEEFloat}}) = true
@inline _nfwd_supported_primal_type(::Type{<:Array{T}}) where {T} = Nfwd._nfwd_is_supported_scalar(
    T
)
@inline _nfwd_supported_primal_type(::TypeVar) = false
@inline _nfwd_supported_primal_type(::Core.TypeofVararg) = false
@inline function _irfwd_supported_primitive_arg_type(T::Type)
    return _nfwd_supported_primal_type(T) ||
           (!Base.has_free_typevars(T) && isconcretetype(T) && tangent_type(T) == NoTangent)
end
@generated function _nfwd_supported_primal_type(::Type{T}) where {T<:Tuple}
    checks = map(p -> _nfwd_supported_primal_type(p), T.parameters)
    return all(checks) ? :(true) : :(false)
end
@inline _nfwd_supported_primal_type(::Type) = false

function _nfwd_supports_primitive_sig(sig::Type{<:Tuple})
    return all(_irfwd_supported_primitive_arg_type, Base.tail(Tuple(sig.parameters)))
end

@inline _nfwd_supported_primitive_output_type(::Type{<:IEEEFloat}) = true
@inline _nfwd_supported_primitive_output_type(::Type{<:Complex{<:IEEEFloat}}) = true
@inline _nfwd_supported_primitive_output_type(::Type{<:Array{T}}) where {T} = Nfwd._nfwd_is_supported_scalar(
    T
)
@inline _nfwd_supported_primitive_output_type(::Type{Union{}}) = false
@generated function _nfwd_supported_primitive_output_type(::Type{T}) where {T<:Tuple}
    checks = map(
        p -> tangent_type(p) != NoTangent && _nfwd_supported_primitive_output_type(p),
        T.parameters,
    )
    return all(checks) ? :(true) : :(false)
end
@inline _nfwd_supported_primitive_output_type(::Type) = false

@generated function _nfwd_supports_primitive_output_sig(sig::Type{<:Tuple})
    f = sig.parameters[1]
    arg_types = Tuple{sig.parameters[2:end]...}
    R = Core.Compiler.return_type(f, arg_types)
    return if tangent_type(R) != NoTangent && _nfwd_supported_primitive_output_type(R)
        :(true)
    else
        :(false)
    end
end

function build_primitive_frule(
    tangent_mode, interp, sig::Type{<:Tuple}; debug_mode=false, silence_debug_messages=true
)
    tangent_mode isa IRfwdMode ||
        tangent_mode isa getfield(getfield(@__MODULE__, :Nfwd), :NDualMode) ||
        return nothing
    N = typeof(tangent_mode).parameters[1]
    _nfwd_supports_primitive_sig(sig) || return nothing
    _nfwd_supports_primitive_output_sig(sig) || return nothing
    lifted_sig = Tuple{
        sig.parameters[1],map(Base.tail(Tuple(sig.parameters))) do T
            if tangent_type(T) == NoTangent
                T
            else
                NfwdMooncake._nfwd_pack_tangent_storage_type(Val(N), T)
            end
        end...
    }
    min = Base.RefValue{UInt}(typemin(UInt))
    max = Base.RefValue{UInt}(typemax(UInt))
    ms = Base._methods_by_ftype(
        lifted_sig, nothing, 1, interp.world, true, min, max, Ptr{Int32}(C_NULL)
    )
    (ms isa Vector && !isempty(ms)) || return nothing
    return NfwdMooncake.NfwdRule{sig,N}()
end

function generate_lifted_primal_ir(
    interp::MooncakeInterpreter, sig_or_mi; do_inline=true, tangent_mode=IRfwdMode{1}()
)
    seed_id!()
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)
    lifted_ir = CC.copy(primal_ir)
    for (a, P) in enumerate(primal_ir.argtypes)
        lifted_ir.argtypes[a] = _fwd_dual_type(tangent_mode, CC.widenconst(P))
    end
    pushfirst!(lifted_ir.argtypes, Any)
    captures = Any[]
    info = DualInfo(
        primal_ir,
        interp,
        characterised_used_ssas(stmt(primal_ir.stmts)),
        false,
        tangent_mode,
    )
    for (n, inst) in enumerate(lifted_ir.stmts)
        modify_lifted_primal_stmts!(stmt(inst), lifted_ir, SSAValue(n), captures, info)
    end
    lifted_ir = CC.compact!(lifted_ir)
    CC.verify_ir(lifted_ir)
    captures_tuple = (captures...,)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)
    lifted_ir = optimise_ir!(lifted_ir; do_inline)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)
    for (a, P) in enumerate(primal_ir.argtypes)
        lifted_ir.argtypes[a + 1] = _fwd_dual_type(tangent_mode, CC.widenconst(P))
    end
    return lifted_ir, captures_tuple, dual_ret_type(primal_ir, tangent_mode)
end

function modify_lifted_primal_stmts!(stmt, lifted_ir, ssa, captures, info)
    modify_fwd_ad_stmts!(stmt, lifted_ir, ssa, captures, info)
end
