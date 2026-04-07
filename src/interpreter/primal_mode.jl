"""
    build_primal(
        interp::MooncakeInterpreter{C},
        sig_or_mi;
        skip_world_age_check=false,
        do_inline=true,
    ) where {C}

Build a callable from resolved, normalised primal IR.
"""
function build_primal(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    call_target=nothing,
    debug_mode=false,
    skip_world_age_check=false,
    do_inline=true,
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
        oc_cache_key = ClosureCacheKey(interp.world, (sig_or_mi, :primal, do_inline))
        if haskey(interp.oc_cache, oc_cache_key)
            cached = _copy(interp.oc_cache[oc_cache_key])
            Tcached = typeof(cached)
            derived = DerivedPrimal{
                typeof(cached.primal_oc),
                typeof(cached.lifted_oc),
                typeof(cached.key),
                typeof(call_target),
                Tcached.parameters[5],
                Tcached.parameters[6],
            }(
                cached.primal_oc, cached.lifted_oc, cached.key, call_target, cached.world
            )
            return debug_mode ? DebugPrimal(derived) : derived
        end

        primal_ir, info, captures = generate_primal_ir(interp, sig_or_mi; do_inline)
        primal_oc = misty_closure(
            info.ret_type, primal_ir, captures...; isva=info.isva, do_compile=true
        )
        lifted_ir, lifted_captures, dual_ret_type = generate_lifted_primal_ir(
            interp, sig_or_mi; do_inline
        )
        lifted_oc = misty_closure(
            dual_ret_type, lifted_ir, lifted_captures...; isva=info.isva, do_compile=true
        )
        derived = DerivedPrimal{
            typeof(primal_oc),
            typeof(lifted_oc),
            typeof(sig_or_mi),
            Nothing,
            info.isva,
            info.nargs,
        }(
            primal_oc, lifted_oc, sig_or_mi, nothing, interp.world
        )
        interp.oc_cache[oc_cache_key] = derived
        derived = DerivedPrimal{
            typeof(primal_oc),
            typeof(lifted_oc),
            typeof(sig_or_mi),
            typeof(call_target),
            info.isva,
            info.nargs,
        }(
            primal_oc, lifted_oc, sig_or_mi, call_target, interp.world
        )
        return debug_mode ? DebugPrimal(derived) : derived
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

function build_primal(args...; debug_mode=false, skip_world_age_check=false, do_inline=true)
    primals = map(x -> x isa Dual ? primal(x) : x, args)
    sig = _typeof(primals)
    interp = get_interpreter(ForwardMode)
    return build_primal(
        interp, sig; call_target=first(primals), debug_mode, skip_world_age_check, do_inline
    )
end

struct PrimalIRInfo
    isva::Bool
    nargs::Int
    ret_type::Type
end

struct PrimalInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
end

struct DerivedPrimal{Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,isva,nargs}
    primal_oc::Tprimal_oc
    lifted_oc::Tlifted_oc
    key::Tkey
    call_target::Tcall_target
    world::UInt
end

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
end

function LazyPrimal(key, world::UInt)
    LazyPrimal{typeof(key),Base.RefValue{Any}}(key, world, Ref{Any}())
end

struct DynamicPrimal{V}
    cache::V
    world::UInt
end

DynamicPrimal(world::UInt) = DynamicPrimal(Dict{Any,Any}(), world)

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
                    DynamicPrimal(interp.world)
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

@inline function (primal::DerivedPrimal{Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,isva,nargs})(
    x::Dual, args::Vararg{Dual,N}
) where {Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,isva,nargs,N}
    isnothing(primal.call_target) && throw(
        ArgumentError(
            "DerivedPrimal without a stored call target cannot be called on Dual arguments directly. Use LazyPrimal or DynamicPrimal with the primal callable as the first argument.",
        ),
    )
    return primal.lifted_oc(zero_dual(primal.call_target), x, args...)
end

@inline function (primal::DerivedPrimal{Tprimal_oc,Tlifted_oc,Tkey,Nothing,isva,nargs})(
    f::Dual, x::Dual, args::Vararg{Dual,N}
) where {Tprimal_oc,Tlifted_oc,Tkey,isva,nargs,N}
    return primal.lifted_oc(f, x, args...)
end

function _copy(x::P) where {P<:DerivedPrimal}
    return P(
        replace_captures(x.primal_oc, _copy(x.primal_oc.oc.captures)),
        replace_captures(x.lifted_oc, _copy(x.lifted_oc.oc.captures)),
        x.key,
        x.call_target,
        x.world,
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

@inline function (primal::LazyPrimal)(args::Vararg{Any,N}) where {N}
    if !isassigned(primal.primal_ref)
        interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=primal.world)
        primal.primal_ref[] = build_primal(interp, primal.key; skip_world_age_check=true)
    end
    return primal.primal_ref[](args...)
end

@inline function (primal::LazyPrimal)(args::Vararg{Dual,N}) where {N}
    throw(
        ArgumentError(
            "LazyPrimal dual calls require the primal callable as the first argument."
        ),
    )
end

@inline function (primal::LazyPrimal)(f::Dual, x::Dual, args::Vararg{Dual,N}) where {N}
    if !isassigned(primal.primal_ref)
        interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=primal.world)
        primal.primal_ref[] = build_primal(interp, primal.key; skip_world_age_check=true)
    end
    return primal.primal_ref[](f, x, args...)
end

_copy(x::P) where {P<:LazyPrimal} = P(x.key, x.world, Ref{Any}())

@inline function (dynamic::DynamicPrimal)(f, args::Vararg{Any,N}) where {N}
    sig = Tuple{map(x -> x isa Dual ? _typeof(primal(x)) : _typeof(x), (f, args...))...}
    rule = get(dynamic.cache, sig, nothing)
    if rule === nothing
        interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=dynamic.world)
        rule = build_primal(interp, sig; skip_world_age_check=true)
        dynamic.cache[sig] = rule
    end
    return rule(f, args...)
end

@inline function (dynamic::DynamicPrimal)(f::Dual, x::Dual, args::Vararg{Dual,N}) where {N}
    sig = Tuple{map(x -> x isa Dual ? _typeof(primal(x)) : _typeof(x), (f, x, args...))...}
    interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=dynamic.world)
    rule = get(dynamic.cache, sig, nothing)
    if rule === nothing
        rule = build_primal(interp, sig; skip_world_age_check=true)
        dynamic.cache[sig] = rule
    end
    return rule(f, x, args...)
end

_copy(x::P) where {P<:DynamicPrimal} = P(Dict{Any,Any}(), x.world)

function generate_lifted_primal_ir(interp::MooncakeInterpreter, sig_or_mi; do_inline=true)
    seed_id!()
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)
    lifted_ir = CC.copy(primal_ir)
    for (a, P) in enumerate(primal_ir.argtypes)
        lifted_ir.argtypes[a] = _fwd_dual_type(IRfwdMode{1}(), CC.widenconst(P))
    end
    pushfirst!(lifted_ir.argtypes, Any)
    captures = Any[]
    info = DualInfo(
        primal_ir,
        interp,
        characterised_used_ssas(stmt(primal_ir.stmts)),
        false,
        IRfwdMode{1}(),
    )
    for (n, inst) in enumerate(lifted_ir.stmts)
        modify_lifted_primal_stmts!(stmt(inst), lifted_ir, SSAValue(n), captures, info)
    end
    lifted_ir = CC.compact!(lifted_ir)
    CC.verify_ir(lifted_ir)
    captures_tuple = (captures...,)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)
    lifted_ir = optimise_ir!(lifted_ir; do_inline)
    return lifted_ir, captures_tuple, dual_ret_type(primal_ir, IRfwdMode{1}())
end

function modify_lifted_primal_stmts!(stmt, lifted_ir, ssa, captures, info)
    modify_fwd_ad_stmts!(stmt, lifted_ir, ssa, captures, info)
end

function modify_lifted_primal_stmts!(
    stmt::Expr, lifted_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if !(isexpr(stmt, :invoke) || isexpr(stmt, :call))
        return modify_fwd_ad_stmts!(stmt, lifted_ir, ssa, captures, info)
    end

    raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
    sig_types = map(raw_args) do x
        t = CC.widenconst(get_forward_primal_type(info.primal_ir, x))
        contains_bottom_type(t) ? Any : t
    end
    sig = Tuple{sig_types...}
    args = map(__inc, raw_args)
    if !info.is_used[ssa.id] && get_const_primal_value(args[1]) == getfield
        replace_call!(lifted_ir, ssa, new_inst(Expr(:call, __fwds_pass_no_ad!, args...)))
        return nothing
    end

    dual_args = map(args) do arg
        arg isa Union{Argument,SSAValue} && return arg
        return _fwd_uninit_dual(info.tangent_mode, get_const_primal_value(arg))
    end

    if is_primitive(context_type(info.interp), ForwardMode, sig, info.interp.world)
        rule = build_primitive_frule(info.tangent_mode, info.interp, sig)
        isnothing(rule) && (rule = build_primitive_frule(sig))
        if safe_for_literal(rule)
            replace_call!(lifted_ir, ssa, Expr(:call, rule, dual_args...))
        else
            push!(captures, rule)
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            replace_call!(lifted_ir, ssa, Expr(:call, rule_ssa, dual_args...))
        end
    else
        push!(
            captures,
            if isexpr(stmt, :invoke)
                LazyPrimal(get_mi(stmt.args[1]), info.interp.world)
            else
                DynamicPrimal(info.interp.world)
            end,
        )
        get_rule = Expr(:call, get_capture, Argument(1), length(captures))
        rule_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
        replace_call!(lifted_ir, ssa, Expr(:call, rule_ssa, dual_args...))
    end
    return nothing
end
