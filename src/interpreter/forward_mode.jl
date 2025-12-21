struct DualRuleInfo
    isva::Bool
    nargs::Int
    dual_ret_type::Type
end
function build_forward_data(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    debug_mode=false,
    silence_debug_messages=true,
    world=Base.get_world_counter(),
) where {C}
    @nospecialize sig_or_mi

    # To avoid segfaults, ensure that we bail out if the interpreter's world age is greater
    # than the current world age.
    if world > interp.world
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
    if is_primitive(C, ForwardMode, sig, interp.world)
        return (; primitive=true, dual_ir=nothing, captures=nothing, info=nothing)
    end

    # We don't have a hand-coded rule, so derive one.
    lock(MOONCAKE_INFERENCE_LOCK)
    try
        # If we've already derived the IR and info, do not re-derive, just
        # create a copy and pass in new shared data.
        oc_cache_key = ClosureCacheKey(
            interp.world, (sig_or_mi, debug_mode, :forward, :data)
        )
        if haskey(interp.oc_cache, oc_cache_key)
            return interp.oc_cache[oc_cache_key]
        else
            dual_ir, captures, info = generate_dual_ir(interp, sig_or_mi; debug_mode, world)
            interp.oc_cache[oc_cache_key] = (dual_ir, captures, info)
            return (; primitive=false, dual_ir, captures, info)
        end
    catch e
        rethrow(e)
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

function generate_dual_ir(
    interp::MooncakeInterpreter, sig_or_mi; debug_mode=false, do_inline=true, world
)
    # Reset id count. This ensures that the IDs generated are the same each time this
    # function runs.
    seed_id!()

    # Grab code associated to the primal.
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    nargs = length(primal_ir.argtypes)

    # Normalise the IR.
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi; world)
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
        modify_fwd_ad_stmts!(stmt(inst), dual_ir, ssa, captures, info; world)
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

_primal_type(::Type{Dual{p,d}}) where {p,d} = p

function generated_frule_body end
function GeneratedFRule_body end

struct GeneratedFRule{Captures<:Tuple}
    captures::Captures
end

function refresh_generated_frule()
    @eval begin
        function generated_frule!!(args...)
            $(Expr(:meta, :generated_only))
            $(Expr(:meta, :generated, generated_frule_body))
        end
        function (::GeneratedFRule)(f::F, args...) where {F}
            $(Expr(:meta, :generated_only))
            $(Expr(:meta, :generated, GeneratedFRule_body))
        end
    end
end
#handy util for when you're changing the insides of the generated functions
refresh_generated_frule()

# # Copy forward rule with recursively copied captures
function _copy(x::GeneratedFRule)
    return GeneratedFRule(_copy(x.captures))
end

# This is the generated function body of generated_frule!!(args...)
function generated_frule_body(world::UInt, lnn, this, args)
    sig = Tuple{_primal_type.(args)...}
    interp = MooncakeInterpreter(ForwardMode; world)
    (; primitive, captures) = build_forward_data(interp, sig; world)
    if primitive
        ex = :(frule!!(args...))
    else
        ex = :(GeneratedFRule($captures)(args...))
    end
    ci = expr_to_codeinfo(@__MODULE__(), [Symbol("#self#"), :args], [], (), ex, true)
    # Attached edges from MethodInstrances of `f` to to this CodeInfo.
    # This should make it so that adding methods to `f` will
    # triggers recompilation, fixing the #265 equivalent for generated functions.
    matches = Base._methods_by_ftype(sig, -1, world)
    ci.edges = Core.svec(Base.specialize_method(only(matches)))
    return ci
end

# This is the generated function body of (::GeneratedFRule)(args...)
function GeneratedFRule_body(world::UInt, lnn, F, this, f, args)
    sig = Tuple{_primal_type(f),_primal_type.(args)...}
    interp = MooncakeInterpreter(ForwardMode; world)
    (; primitive, dual_ir) = build_forward_data(interp, sig; world)

    ci = irc_to_codeinfo(dual_ir)
    # Remove the type info so that it can be returned from the generated function
    ci.ssavaluetypes = length(ci.ssavaluetypes)

    # Attached edges from MethodInstrances of `f` to to this CodeInfo.
    # This should make it so that adding methods to `f` will
    # triggers recompilation of this generated function.
    matches = Base._methods_by_ftype(sig, -1, world)
    ci.edges = Core.svec(Base.specialize_method(only(matches)))
    return ci
end

function expr_to_codeinfo(m::Module, argnames, spnames, sp, e::Expr, isva)
    # This trick comes from https://github.com/NHDaly/StagedFunctions.jl/commit/22fc72740093892baa442850a1fd61d9cd61b4cd (but has been since modified)
    lam = Expr(
        :lambda,
        argnames,
        Expr(Symbol("scope-block"), Expr(:block, Expr(:return, Expr(:block, e)))),
    )
    ex = if spnames === nothing || isempty(spnames)
        lam
    else
        Expr(Symbol("with-static-parameters"), lam, spnames...)
    end

    # Get the code-info for the generator body in order to use it for generating a dummy
    # code info object.
    ci = if VERSION < v"1.12-"
        ccall(
            :jl_expand_and_resolve,
            Any,
            (Any, Any, Core.SimpleVector),
            ex,
            m,
            Core.svec(sp...),
        )
    else
        Base.generated_body_to_codeinfo(ex, @__MODULE__(), isva)
    end
    @assert ci isa Core.CodeInfo "Failed to create a CodeInfo from the given expression. This might mean it contains a closure or comprehension?\n Offending expression: $e"
    ci
end

function irc_to_codeinfo(
    ir::IRCode,
    @nospecialize env...;
    isva::Bool=false,
    slotnames::Union{Nothing,Vector{Symbol}}=nothing,
    kwargs...,
)
    CC = Core.Compiler
    # NOTE: we need ir.argtypes[1] == typeof(env)
    ir = copy(ir)
    nargtypes = length(ir.argtypes)
    nargs = nargtypes-1
    sig = CC.compute_oc_signature(ir, nargs, isva)
    rt = CC.compute_ir_rettype(ir)
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    if slotnames === nothing
        src.slotnames = fill(:none, nargtypes)
    else
        length(slotnames) == nargtypes || error("mismatched `argtypes` and `slotnames`")
        src.slotnames = slotnames
    end
    src.slotflags = fill(zero(UInt8), nargtypes)
    src.slottypes = copy(ir.argtypes)
    src.isva = isva
    src.nargs = UInt(nargtypes)
    src = CC.ir_to_codeinf!(src, ir)
    src.rettype = rt
    src
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

@inline get_capture(captures::T, n::Int) where {T} = captures[n]
@inline get_capture(fr::GeneratedFRule{T}, n::Int) where {T} = get_capture(fr.captures, n)

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

function modify_fwd_ad_stmts!(
    ::Nothing, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo; world
)
    nothing
end

function modify_fwd_ad_stmts!(
    ::GotoNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo; world
)
    nothing
end

function modify_fwd_ad_stmts!(
    stmt::GotoIfNot,
    dual_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::DualInfo;
    world,
)
    # replace GotoIfNot with the call to primal
    Mooncake.replace_call!(dual_ir, ssa, Expr(:call, _primal, inc_args(stmt).cond))

    # reinsert the GotoIfNot right after the call to primal
    new_gotoifnot_inst = new_inst(Core.GotoIfNot(ssa, stmt.dest))
    CC.insert_node!(dual_ir, ssa, new_gotoifnot_inst, ATTACH_AFTER)
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::GlobalRef,
    dual_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    ::DualInfo;
    world,
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
    stmt::ReturnNode,
    dual_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    ::DualInfo;
    world,
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
    stmt::PhiNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::DualInfo; world
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
    stmt::PiNode, dual_ir::IRCode, ssa::SSAValue, ::Vector{Any}, ::DualInfo; world
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
    stmt::UpsilonNode,
    dual_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    ::DualInfo;
    world,
)
    if !(stmt.val isa Union{Argument,SSAValue})
        stmt = UpsilonNode(uninit_dual(get_const_primal_value(stmt.val)))
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(dual_ir, ssa, :type, dual_type(CC.widenconst(get_ir(dual_ir, ssa, :type))))
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiCNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::DualInfo; world
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
    stmt::Expr, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo; world
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        sig_types = map(raw_args) do x
            return CC.widenconst(get_forward_primal_type(info.primal_ir, x))
        end
        sig = Tuple{sig_types...}
        mi = isexpr(stmt, :invoke) ? get_mi(stmt.args[1]) : missing
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
            if arg isa Union{Argument,SSAValue}
                return arg
            elseif arg isa GlobalRef && !isconst(arg)
                arg_ssa = CC.insert_node!(
                    dual_ir, ssa, new_inst(Expr(:call, uninit_dual, arg)), ATTACH_BEFORE
                )
                return arg_ssa
            else
                return uninit_dual(get_const_primal_value(arg))
            end
        end

        interp = info.interp
        if is_primitive(context_type(interp), ForwardMode, sig, interp.world)
            replace_call!(dual_ir, ssa, Expr(:call, frule!!, dual_args...))
        else
            dm = info.debug_mode
            # TODO debug mode?
            replace_call!(dual_ir, ssa, Expr(:call, generated_frule!!, dual_args...))
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

@noinline invokelatest_generated_frule!!(args...) = invokelatest(generated_frule!!, args...)

get_forward_primal_type(ir::CC.IRCode, a::Argument) = ir.argtypes[a.n]
get_forward_primal_type(ir::CC.IRCode, ssa::SSAValue) = get_ir(ir, ssa, :type)
get_forward_primal_type(::CC.IRCode, x::QuoteNode) = _typeof(x.value)
get_forward_primal_type(::CC.IRCode, x) = _typeof(x)
function get_forward_primal_type(::CC.IRCode, x::GlobalRef)
    @static if VERSION > v"1.12-"
        return if isconst(x)
            _typeof(getglobal(x.mod, x.name))
        else
            x.binding.partitions.restriction
        end
    else
        return isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.ty
    end
end
function get_forward_primal_type(::CC.IRCode, x::Expr)
    x.head === :boundscheck && return Bool
    return error("Unrecognised expression $x found in argument slot.")
end

function dual_ret_type(primal_ir::IRCode)
    return dual_type(compute_ir_rettype(primal_ir))
end
