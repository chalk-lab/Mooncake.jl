function build_frule(args...; debug_mode=false, silence_debug_messages=true)
    sig = _typeof(TestUtils.__get_primals(args))
    interp = get_interpreter(ForwardMode)
    return build_frule(interp, sig; debug_mode, silence_debug_messages)
end

struct DualRuleInfo
    isva::Bool
    nargs::Int
    dual_ret_type::Type
end

function build_frule(
    interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode=false, silence_debug_messages=true
) where {C}
    @nospecialize sig_or_mi

    # To avoid segfaults, ensure that we bail out if the interpreter's world age is greater
    # than the current world age.
    if Base.get_world_counter() > interp.world
        throw(
            ArgumentError(
                "World age associated to interp is behind current world age. Please " *
                "create a new interpreter for the current world age.",
            ),
        )
    end

    # If we're compiling in debug mode, let the user know by default.
    if !silence_debug_messages && debug_mode
        @info "Compiling frule for $sig_or_mi in debug mode. Disable for best performance."
    end

    # If we have a hand-coded rule, just use that.
    sig = _get_sig(sig_or_mi)
    is_primitive(C, ForwardMode, sig) && return (debug_mode ? DebugFRule(frule!!) : frule!!)

    # We don't have a hand-coded rule, so derive one.
    lock(MOONCAKE_INFERENCE_LOCK)
    try
        # If we've already derived the OpaqueClosures and info, do not re-derive, just
        # create a copy and pass in new shared data.
        oc_cache_key = ClosureCacheKey(interp.world, (sig_or_mi, debug_mode, :forward))
        if haskey(interp.oc_cache, oc_cache_key)
            return interp.oc_cache[oc_cache_key]
        else
            # Derive forward-pass IR, and shove in a `MistyClosure`.
            dual_ir, captures, info = generate_dual_ir(interp, sig_or_mi; debug_mode)
            dual_oc = misty_closure(
                info.dual_ret_type, dual_ir, captures...; do_compile=true
            )
            raw_rule = DerivedFRule{sig,typeof(dual_oc),info.isva,info.nargs}(dual_oc)
            rule = debug_mode ? DebugFRule(raw_rule) : raw_rule
            interp.oc_cache[oc_cache_key] = rule
            return rule
        end
    catch e
        rethrow(e)
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

struct DerivedFRule{primal_sig,Tfwd_oc,isva,nargs}
    fwd_oc::Tfwd_oc
end

@inline function (fwd::DerivedFRule{P,sig,isva,nargs})(
    args::Vararg{Dual,N}
) where {P,sig,N,isva,nargs}
    return fwd.fwd_oc(__unflatten_dual_varargs(isva, args, Val(nargs))...)
end

function _copy(x::P) where {P<:DerivedFRule}
    return P(replace_captures(x.fwd_oc, _copy(x.fwd_oc.oc.captures)))
end

"""
    __unflatten_dual_varargs(isva::Bool, args, ::Val{nargs}) where {nargs}

If isva and nargs=2, then inputs `(Dual(5.0, 0.0), Dual(4.0, 0.0), Dual(3.0, 0.0))`
are transformed into `(Dual(5.0, 0.0), Dual((5.0, 4.0), (0.0, 0.0)))`.
"""
function __unflatten_dual_varargs(isva::Bool, args, ::Val{nargs}) where {nargs}
    isva || return args
    group_primal = map(primal, args[nargs:end])
    if tangent_type(_typeof(group_primal)) == NoTangent
        grouped_args = zero_dual(group_primal)
    else
        grouped_args = Dual(group_primal, map(tangent, args[nargs:end]))
    end
    return (args[1:(nargs - 1)]..., grouped_args)
end

struct DualInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
    is_used::Vector{Bool}
    debug_mode::Bool
end

function generate_dual_ir(
    interp::MooncakeInterpreter, sig_or_mi; debug_mode=false, do_inline=true
)
    # Reset id count. This ensures that the IDs generated are the same each time this
    # function runs.
    seed_id!()

    # Grab code associated to the primal.
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    nargs = length(primal_ir.argtypes)

    # Normalise the IR.
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)

    # Keep a copy of the primal IR with the insertions
    dual_ir = CC.copy(primal_ir)

    # Modify dual argument types:
    # - add one for the captures in the first position, with placeholder type for now
    # - convert the rest to dual types
    for (a, P) in enumerate(primal_ir.argtypes)
        dual_ir.argtypes[a] = dual_type(CC.widenconst(P))
    end
    pushfirst!(dual_ir.argtypes, Any)

    # Data structure into which we can push any data which is to live in the captures field
    # of the OpaqueClosure used to implement this rule. The index at which a piece of data
    # lives in this data structure is equal to the index of the captures field of the
    # OpaqueClosure in which it will live. To write code which retrieves items from the
    # captures data structure, make use of `get_capture`.
    captures = Any[]

    is_used = characterised_used_ssas(stmt(primal_ir.stmts))
    info = DualInfo(primal_ir, interp, is_used, debug_mode)
    for (n, inst) in enumerate(dual_ir.stmts)
        ssa = SSAValue(n)
        modify_fwd_ad_stmts!(stmt(inst), dual_ir, ssa, captures, info)
    end

    # Process new nodes etc.
    dual_ir = CC.compact!(dual_ir)

    CC.verify_ir(dual_ir)

    # Now that the captured values are known, replace the placeholder value given for the
    # first argument type with the actual type.
    captures_tuple = (captures...,)
    dual_ir.argtypes[1] = _typeof(captures_tuple)

    # Optimize dual IR
    dual_ir_opt = optimise_ir!(dual_ir; do_inline)
    return dual_ir_opt, captures_tuple, DualRuleInfo(isva, nargs, dual_ret_type(primal_ir))
end

@inline get_capture(captures::T, n::Int) where {T} = captures[n]

"""
    const_dual!(captures::Vector{Any}, stmt)::Union{Dual,Int}

Build a `Dual` from `stmt`, with zero / uninitialised tangent. If the resulting `Dual` is
a bits type, then it is returned. If it is not, then the `Dual` is put into captures,
and its location in `captures` returned.

Whether or not the value is a literal, or an index into the captures, can be determined from
the return type.
"""
function const_dual!(captures::Vector{Any}, stmt)::Union{Dual,Int}
    v = get_const_primal_value(stmt)
    x = uninit_dual(v)
    if safe_for_literal(v)
        return x
    else
        push!(captures, x)
        return length(captures)
    end
end

## Modification of IR nodes

const ATTACH_AFTER = true
const ATTACH_BEFORE = false

modify_fwd_ad_stmts!(::Nothing, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo) = nothing

modify_fwd_ad_stmts!(::GotoNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo) = nothing

function modify_fwd_ad_stmts!(
    stmt::GotoIfNot, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    # replace GotoIfNot with the call to primal
    Mooncake.replace_call!(dual_ir, ssa, Expr(:call, _primal, inc_args(stmt).cond))

    # reinsert the GotoIfNot right after the call to primal
    new_gotoifnot_inst = new_inst(Core.GotoIfNot(ssa, stmt.dest))
    CC.insert_node!(dual_ir, ssa, new_gotoifnot_inst, ATTACH_AFTER)
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::GlobalRef, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::DualInfo
)
    if isconst(stmt)
        d = const_dual!(captures, stmt)
        if d isa Int
            Mooncake.replace_call!(dual_ir, ssa, Expr(:call, get_capture, Argument(1), d))
        else
            Mooncake.replace_call!(dual_ir, ssa, Expr(:call, identity, d))
        end
    else
        new_ssa = CC.insert_node!(dual_ir, ssa, new_inst(stmt), ATTACH_BEFORE)
        zero_dual_call = Expr(:call, Mooncake.zero_dual, new_ssa)
        Mooncake.replace_call!(dual_ir, ssa, zero_dual_call)
    end

    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::ReturnNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::DualInfo
)
    # undefined `val` field means that stmt is unreachable.
    isdefined(stmt, :val) || return nothing

    # stmt is an Argument, then already a dual, and must just be incremented.
    if stmt.val isa Union{Argument,SSAValue}
        Mooncake.replace_call!(dual_ir, ssa, ReturnNode(__inc(stmt.val)))
        return nothing
    end

    # stmt is a const, so we have to turn it into a dual.
    dual_stmt = ReturnNode(const_dual!(captures, stmt.val))
    Mooncake.replace_call!(dual_ir, ssa, dual_stmt)
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::DualInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = uninit_dual(get_const_primal_value(stmt.values[n]))
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(dual_ir, ssa, :type, dual_type(CC.widenconst(get_ir(dual_ir, ssa, :type))))
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PiNode, dual_ir::IRCode, ssa::SSAValue, ::Vector{Any}, ::DualInfo
)
    if stmt.val isa Union{Argument,SSAValue}
        v = __inc(stmt.val)
    else
        v = uninit_dual(get_const_primal_value(stmt.val))
    end
    replace_call!(dual_ir, ssa, PiNode(v, dual_type(CC.widenconst(stmt.typ))))
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::UpsilonNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::DualInfo
)
    if !(stmt.val isa Union{Argument,SSAValue})
        stmt = UpsilonNode(uninit_dual(get_const_primal_value(stmt.val)))
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(dual_ir, ssa, :type, dual_type(CC.widenconst(get_ir(dual_ir, ssa, :type))))
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiCNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::DualInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = uninit_dual(get_const_primal_value(stmt.values[n]))
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(dual_ir, ssa, :type, dual_type(CC.widenconst(get_ir(dual_ir, ssa, :type))))
    return nothing
end

@static if isdefined(Core, :EnterNode)
    function modify_fwd_ad_stmts!(
        ::Core.EnterNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo
    )
        return nothing
    end
end

## Modification of IR nodes - expressions

__get_primal(x::Dual) = primal(x)

function modify_fwd_ad_stmts!(
    stmt::Expr, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        sig_types = map(raw_args) do x
            return CC.widenconst(get_forward_primal_type(info.primal_ir, x))
        end
        sig = Tuple{sig_types...}
        mi = isexpr(stmt, :invoke) ? stmt.args[1] : missing
        args = map(__inc, raw_args)

        # Special case: if the result of a call to getfield is un-used, then leave the
        # primal statment alone (just increment arguments as usual). This was causing
        # performance problems in a couple of situations where the field being requested is
        # not known at compile time. `getfield` cannot be dead-code eliminated, because it
        # can throw an error if the requested field does not exist. Everything _other_ than
        # the boundscheck is eliminated in LLVM codegen, so it's important that AD doesn't
        # get in the way of this.
        #
        # This might need to be generalised to more things than just `getfield`, but at the
        # time of writing this comment, it's unclear whether or not this is the case.
        if !info.is_used[ssa.id] && get_const_primal_value(args[1]) == getfield
            fwds = new_inst(Expr(:call, __fwds_pass_no_ad!, args...))
            replace_call!(dual_ir, ssa, fwds)
            return nothing
        end

        # Dual-ise arguments.
        dual_args = map(args) do arg
            arg isa Union{Argument,SSAValue} && return arg
            return uninit_dual(get_const_primal_value(arg))
        end

        if is_primitive(context_type(info.interp), ForwardMode, sig)
            replace_call!(dual_ir, ssa, Expr(:call, frule!!, dual_args...))
        else
            dm = info.debug_mode
            push!(captures, isexpr(stmt, :invoke) ? LazyFRule(mi, dm) : DynamicFRule(dm))
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            replace_call!(dual_ir, ssa, Expr(:call, rule_ssa, dual_args...))
        end
    elseif isexpr(stmt, :boundscheck)
        # Keep the boundscheck, but put it in a Dual.
        inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        bc_ssa = CC.insert_node!(dual_ir, ssa, inst, ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, Expr(:call, zero_dual, bc_ssa))
    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(dual_ir, ssa, nothing)
    elseif Meta.isexpr(stmt, :copyast)
        new_copyast_inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        new_copyast_ssa = CC.insert_node!(dual_ir, ssa, new_copyast_inst, ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, Expr(:call, zero_dual, new_copyast_ssa))
    elseif Meta.isexpr(stmt, :loopinfo)
        # Leave this node alone.
    elseif isexpr(stmt, :throw_undef_if_not)
        # args[1] is a Symbol, args[2] is the condition which must be primalized
        primal_cond = Expr(:call, _primal, inc_args(stmt).args[2])
        replace_call!(dual_ir, ssa, primal_cond)
        new_undef_inst = new_inst(Expr(:throw_undef_if_not, stmt.args[1], ssa))
        CC.insert_node!(dual_ir, ssa, new_undef_inst, ATTACH_AFTER)
    elseif isexpr(stmt, :enter)
        # Leave this node alone
    elseif isexpr(stmt, :leave)
        # Leave this node alone
    elseif isexpr(stmt, :pop_exception)
        # Leave this node alone
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

mutable struct LazyFRule{primal_sig,Trule}
    debug_mode::Bool
    mi::Core.MethodInstance
    rule::Trule
    function LazyFRule(mi::Core.MethodInstance, debug_mode::Bool)
        interp = get_interpreter(ForwardMode)
        return new{mi.specTypes,frule_type(interp, mi;debug_mode)}(debug_mode, mi)
    end
    function LazyFRule{Tprimal_sig,Trule}(
        mi::Core.MethodInstance, debug_mode::Bool
    ) where {Tprimal_sig,Trule}
        return new{Tprimal_sig,Trule}(debug_mode, mi)
    end
end

_copy(x::P) where {P<:LazyFRule} = P(x.mi, x.debug_mode)

@inline function (rule::LazyFRule)(args::Vararg{Any,N}) where {N}
    return isdefined(rule, :rule) ? rule.rule(args...) : _build_rule!(rule, args)
end

@noinline function _build_rule!(rule::LazyFRule{sig,Trule}, args) where {sig,Trule}
    interp = get_interpreter(ForwardMode)
    rule.rule = build_frule(interp, rule.mi; debug_mode=rule.debug_mode)
    return rule.rule(args...)
end

function dual_ret_type(primal_ir::IRCode)
    return dual_type(Base.Experimental.compute_ir_rettype(primal_ir))
end

function frule_type(
    interp::MooncakeInterpreter{C}, mi::CC.MethodInstance; debug_mode
) where {C}
    primal_sig = _get_sig(mi)
    if is_primitive(C, ForwardMode, primal_sig)
        return debug_mode ? DebugFRule{typeof(frule!!)} : typeof(frule!!)
    end
    ir, _ = lookup_ir(interp, mi)
    nargs = length(ir.argtypes)
    isva, _ = is_vararg_and_sparam_names(mi)
    arg_types = map(CC.widenconst, ir.argtypes)
    dual_args_type = Tuple{map(dual_type, arg_types)...}
    closure_type = RuleMC{dual_args_type,dual_ret_type(ir)}
    Tderived_rule = DerivedFRule{primal_sig,closure_type,isva,nargs}
    return debug_mode ? DebugFRule{Tderived_rule} : Tderived_rule
end

struct DynamicFRule{V}
    cache::V
    debug_mode::Bool
end

DynamicFRule(debug_mode::Bool) = DynamicFRule(Dict{Any,Any}(), debug_mode)

_copy(x::P) where {P<:DynamicFRule} = P(Dict{Any,Any}(), x.debug_mode)

function (dynamic_rule::DynamicFRule)(args::Vararg{Dual,N}) where {N}
    sig = Tuple{map(_typeof ∘ primal, args)...}
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        interp = get_interpreter(ForwardMode)
        rule = build_frule(interp, sig; debug_mode=dynamic_rule.debug_mode)
        dynamic_rule.cache[sig] = rule
    end
    return rule(args...)
end
