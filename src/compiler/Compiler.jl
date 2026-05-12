module Compiler

using Core: Argument, GlobalRef, OpaqueClosure, PhiNode, SSAValue
using Core.Compiler: IRCode, NewInstruction
using MistyClosures: MistyClosure

using ..BasicBlockCode: BBCode, __line_numbers_to_block_numbers!
using ..Mooncake:
    BugPatchInterpreter,
    MooncakeCache,
    MooncakeInterpreter,
    NoInlineCallInfo,
    is_primitive,
    _typeof

const CC = Core.Compiler

instruction_statement_field() = @static VERSION < v"1.11" ? :inst : :stmt

"""
    statement_stream(ir::IRCode)

Return the compiler instruction stream for typed IR.
"""
statement_stream(ir::IRCode) = ir.stmts
statement_count(ir::IRCode) = length(statement_stream(ir))
control_flow_graph(ir::IRCode) = ir.cfg
static_parameter_states(ir::IRCode) = ir.sptypes

function static_parameter_map(ir::IRCode, spnames::Vector{Symbol})
    return Dict{Symbol,CC.VarState}(zip(spnames, static_parameter_states(ir)))
end

"""
    statements(ir::IRCode)
    statements(stream::CC.InstructionStream)

Return the statement vector for typed IR. This is a semantic operation over
IR statements; the Julia-version-specific field name is private here.
"""
statements(ir::IRCode) = statements(statement_stream(ir))
function statements(stream::CC.InstructionStream)
    CC.getfield(stream, instruction_statement_field())
end
statement_types(ir::IRCode) = statement_stream(ir).type
statement_type(ir::IRCode, n::Integer) = statement_types(ir)[n]

"""
    statement(inst::CC.Instruction)

Return the AST/IR node held by one compiler instruction.
"""
statement(inst::CC.Instruction) = CC.getindex(inst, instruction_statement_field())

argument_types(ir::IRCode) = ir.argtypes
argument_types(ir::BBCode) = ir.argtypes
argument_type(ir::IRCode, i::Integer) = argument_types(ir)[i]
argument_count(ir::IRCode) = length(argument_types(ir))
argument_count(ir::BBCode) = length(argument_types(ir))

function set_argument_type!(ir::IRCode, i::Integer, @nospecialize(T))
    argument_types(ir)[i] = T
    return ir
end

function prepend_argument_type!(ir::IRCode, @nospecialize(T))
    pushfirst!(argument_types(ir), T)
    return ir
end

function map_argument_types!(f, ir::IRCode)
    argtypes = argument_types(ir)
    for i in eachindex(argtypes)
        argtypes[i] = f(argtypes[i])
    end
    return ir
end

function instruction_stream_from_statements(insts::Vector{Any})
    n = length(insts)
    @static if VERSION > v"1.12-"
        lineinfo = Int32[]
        for _ in 1:n
            push!(lineinfo, 1, 0, 0)
        end
    else
        lineinfo = ones(Int32, n)
    end
    return CC.InstructionStream(
        insts,
        Any[Any for _ in 1:n],
        CC.CallInfo[CC.NoCallInfo() for _ in 1:n],
        lineinfo,
        fill(CC.IR_FLAG_REFINED, n),
    )
end

function ircode(
    insts::Vector{Any}, argtypes::Vector{Any}, sptypes::Vector{CC.VarState}=CC.VarState[]
)
    cfg = CC.compute_basic_blocks(insts)
    insts = __line_numbers_to_block_numbers!(insts, cfg)
    stmts = instruction_stream_from_statements(insts)
    @static if VERSION > v"1.12-"
        debug_info = CC.DebugInfoStream(nothing, CC.DebugInfo(:Mooncake), length(insts))
    else
        debug_info = [
            CC.LineInfoNode(
                parentmodule(@__MODULE__), :ircode, :Compiler, Int32(1), Int32(0)
            ),
        ]
    end
    meta = Expr[]
    return CC.IRCode(stmts, cfg, debug_info, argtypes, meta, CC.VarState[])
end

@static if VERSION > v"1.12-"
    function with_argument_types(ir::BBCode, argtypes::AbstractVector)
        argtypes = Any[argtypes...]
        return BBCode(
            ir.blocks, argtypes, ir.sptypes, ir.debuginfo, ir.meta, ir.valid_worlds
        )
    end
else
    function with_argument_types(ir::BBCode, argtypes::AbstractVector)
        argtypes = Any[argtypes...]
        return BBCode(ir.blocks, argtypes, ir.sptypes, ir.linetable, ir.meta)
    end
end

instruction(ir::IRCode, ssa::SSAValue) = CC.getindex(ir, ssa)
statement_type(ir::IRCode, ssa::SSAValue) = CC.getindex(instruction(ir, ssa), :type)

function set_statement!(ir::IRCode, ssa::SSAValue, @nospecialize(stmt))
    return CC.setindex!(instruction(ir, ssa), stmt, instruction_statement_field())
end

function set_statement_type!(ir::IRCode, n::Integer, @nospecialize(T))
    statement_types(ir)[n] = T
    return ir
end

function set_statement_type!(ir::IRCode, ssa::SSAValue, @nospecialize(T))
    return CC.setindex!(instruction(ir, ssa), T, :type)
end

function replace_statement!(
    ir::IRCode,
    ssa::SSAValue,
    @nospecialize(stmt);
    type=Any,
    info=CC.NoCallInfo(),
    flag=CC.IR_FLAG_REFINED,
)
    inst = instruction(ir, ssa)
    CC.setindex!(inst, stmt, instruction_statement_field())
    CC.setindex!(inst, type, :type)
    CC.setindex!(inst, info, :info)
    CC.setindex!(inst, flag, :flag)
    return ir
end

function insert_before!(ir::IRCode, ssa::SSAValue, inst::NewInstruction)
    CC.insert_node!(ir, ssa, inst, false)
end

function insert_after!(ir::IRCode, ssa::SSAValue, inst::NewInstruction)
    CC.insert_node!(ir, ssa, inst, true)
end

compact!(ir::IRCode) = CC.compact!(ir)
verify(ir::IRCode) = (CC.verify_ir(ir); ir)

function widen_const(@nospecialize(T))
    return CC.widenconst(T)
end

function block_for_statement(ir::IRCode, stmt_index::Integer)
    cfg = control_flow_graph(ir)
    next_block = findfirst(i -> i > stmt_index, cfg.index)
    return next_block === nothing ? length(cfg.blocks) : next_block
end

block_successors(ir::IRCode, block::Integer) = control_flow_graph(ir).blocks[block].succs
block_predecessors(ir::IRCode, block::Integer) = control_flow_graph(ir).blocks[block].preds

# On Julia 1.10, passing findfirst's Union{Int,Nothing} return directly to deleteat!
# introduces a union-split that propagates through generated AD IR and causes spurious
# allocation regressions. The !== nothing guard narrows to Int before the call.
@static if VERSION < v"1.11-"
    function _delete_item!(v, item)
        i = findfirst(==(item), v)
        i !== nothing && deleteat!(v, i)
        return v
    end
else
    _delete_item!(v, item) = deleteat!(v, findfirst(==(item), v))
end

"""
    remove_control_flow_edge!(ir::IRCode, from::Int, to::Int)

Remove a CFG edge and the matching incoming values from leading PhiNodes in the
destination block.
"""
function remove_control_flow_edge!(ir::IRCode, from::Int, to::Int)
    cfg = control_flow_graph(ir)

    _delete_item!(cfg.blocks[from].succs, to)

    to_block = cfg.blocks[to]
    _delete_item!(to_block.preds, from)

    stmts = statements(ir)
    for n in to_block.stmts
        stmt = stmts[n]
        if stmt isa PhiNode
            edge_index = findfirst(i::Int32 -> i == from, stmt.edges)
            edge_index === nothing && continue
            deleteat!(stmt.edges, edge_index)
            deleteat!(stmt.values, edge_index)
        else
            break
        end
    end
    return ir
end

function verify_debug_info(ir::IRCode)
    @static if VERSION > v"1.12-"
        CC.verify_linetable(ir.debuginfo, length(ir.stmts), true)
    else
        CC.verify_linetable(ir.linetable, true)
    end
    return ir
end

function method_instance(ci::Core.CodeInstance)
    @static if isdefined(CC, :get_ci_mi)
        return CC.get_ci_mi(ci)
    else
        return ci.def
    end
end
method_instance(mi::Core.MethodInstance) = mi

function method_instance_return_type(mi::Core.MethodInstance)
    return isdefined(mi, :cache) ? mi.cache.rettype : CC.return_type(mi.specTypes)
end

function method_instance_return_type(ci_or_mi)
    return method_instance_return_type(method_instance(ci_or_mi))
end

function top_level_method_instance(ir::IRCode, _module::Module)
    mi = ccall(:jl_new_method_instance_uninit, Ref{Core.MethodInstance}, ())
    mi.specTypes = Tuple{map(CC.widenconst, argument_types(ir))...}
    mi.def = _module
    return mi
end

function inference_world(interp::CC.AbstractInterpreter)
    @static if VERSION < v"1.11.0"
        return CC.get_world_counter(interp)
    else
        return CC.get_inference_world(interp)
    end
end

interpreter_world(interp::MooncakeInterpreter) = interp.world
inference_parameters(interp::MooncakeInterpreter) = interp.inf_params
optimization_parameters(interp::MooncakeInterpreter) = interp.opt_params
inference_cache(interp::MooncakeInterpreter) = interp.inf_cache
function code_cache_view(interp::MooncakeInterpreter)
    CC.WorldView(interp.code_cache, CC.WorldRange(interpreter_world(interp)))
end
cache_owner(::MooncakeInterpreter) = nothing

function overlay_method_table(interp::MooncakeInterpreter, table)
    CC.OverlayMethodTable(interpreter_world(interp), table)
end

function code_cache_get(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance, default)
    get(wvc.cache.dict, mi, default)
end
function code_cache_getindex(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance)
    getindex(wvc.cache.dict, mi)
end
function code_cache_haskey(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance)
    haskey(wvc.cache.dict, mi)
end
function code_cache_setindex!(
    wvc::CC.WorldView{MooncakeCache}, ci::Core.CodeInstance, mi::Core.MethodInstance
)
    return setindex!(wvc.cache.dict, ci, mi)
end

function call_matches(
    interp::MooncakeInterpreter,
    argtypes::Vector{Any},
    @nospecialize(atype),
    max_methods::Int,
)
    @static if VERSION < v"1.12-"
        lattice = CC.typeinf_lattice(interp)
        return CC.find_matching_methods(
            lattice,
            argtypes,
            atype,
            CC.method_table(interp),
            inference_parameters(interp).max_union_splitting,
            max_methods,
        )
    else
        return CC.find_method_matches(interp, argtypes, atype; max_methods)
    end
end

@static if VERSION < v"1.12-"
    method_match_signature(applicable_match) = applicable_match.spec_types
else
    method_match_signature(applicable_match) = applicable_match.match.spec_types
end

function any_primitive_match(applicable, ::Type{C}, ::Type{M}, world::UInt) where {C,M}
    for app in applicable
        if is_primitive(C, M, method_match_signature(app), world)
            return true
        end
    end
    return false
end

function has_primitive_match(matches, ::Type{C}, ::Type{M}, world::UInt) where {C,M}
    matches isa CC.FailedMethodMatch && return false
    return any_primitive_match(matches.applicable, C, M, world)
end

"""
    widen_primitive_rettype(call, argtypes)

Prevent compiler folding from erasing primitive calls whose return type inferred
to `Const` only because of partially constant arguments.
"""
function widen_primitive_rettype(call::CC.CallMeta, argtypes::Vector{Any})
    has_nonconst_runtime_arg = any(i -> !(argtypes[i] isa CC.Const), 2:length(argtypes))
    rt = call.rt isa CC.Const && has_nonconst_runtime_arg ? CC.widenconst(call.rt) : call.rt

    @static if VERSION >= v"1.11-"
        return CC.CallMeta(rt, call.exct, call.effects, call.info)
    else
        return CC.CallMeta(rt, call.effects, call.info)
    end
end

function block_inlining(call::CC.CallMeta, @nospecialize(atype))
    info = NoInlineCallInfo(call.info, atype)
    @static if VERSION >= v"1.11-"
        return CC.CallMeta(call.rt, call.exct, call.effects, info)
    else
        return CC.CallMeta(call.rt, call.effects, info)
    end
end

should_inline_call(::CC.CallInfo) = true
should_inline_call(::NoInlineCallInfo) = false

function noinline_primitive_callmeta(
    call::CC.CallMeta, argtypes::Vector{Any}, @nospecialize(atype)
)
    return block_inlining(widen_primitive_rettype(call, argtypes), atype)
end

function primitive_call_override(
    interp::MooncakeInterpreter{C,M},
    @nospecialize(f),
    arginfo::CC.ArgInfo,
    si::CC.StmtInfo,
    @nospecialize(atype),
    sv::CC.AbsIntState,
    max_methods::Int,
) where {C,M}
    argtypes = arginfo.argtypes
    matches = call_matches(interp, argtypes, atype, max_methods)
    has_primitive_match(matches, C, M, interpreter_world(interp)) || return nothing

    native_interp = CC.NativeInterpreter(interpreter_world(interp))
    ret = CC.abstract_call_gf_by_type(native_interp, f, arginfo, si, atype, sv, max_methods)

    @static if VERSION < v"1.12-"
        return noinline_primitive_callmeta(ret::CC.CallMeta, argtypes, atype)
    else
        return CC.Future{CC.CallMeta}(ret::CC.Future, interp, sv) do call, _interp, _sv
            return noinline_primitive_callmeta(call, argtypes, atype)
        end
    end
end

function run_constant_propagation!(
    ir::IRCode, interp::CC.AbstractInterpreter, mi::Core.MethodInstance
)
    @static if VERSION >= v"1.12-"
        nargs = argument_count(ir) - 1
        isva = false
        propagate_inbounds = true
        spec_info = CC.SpecInfo(nargs, isva, propagate_inbounds, nothing)
        max_world = min_world = world = inference_world(interp)
        irsv = CC.IRInterpretationState(
            interp, spec_info, ir, mi, argument_types(ir), world, min_world, max_world
        )
        CC.ir_abstract_constant_propagation(interp, irsv)
    else
        method_info = CC.MethodInfo(true, nothing)#=propagate_inbounds=#
        min_world = world = inference_world(interp)
        max_world = Base.get_world_counter()
        irsv = CC.IRInterpretationState(
            interp, method_info, ir, mi, argument_types(ir), world, min_world, max_world
        )
        CC._ir_abstract_constant_propagation(interp, irsv)
    end
    return ir
end

function infer_ir!(ir::IRCode)
    run_constant_propagation!(
        ir, CC.NativeInterpreter(), top_level_method_instance(ir, parentmodule(@__MODULE__))
    )
end

function strip_coverage_effects!(ir::IRCode)
    for n in eachindex(statements(ir))
        if Meta.isexpr(statements(ir)[n], :code_coverage_effect)
            statements(ir)[n] = nothing
        end
    end
    return ir
end

function run_adce!(ir::IRCode, inline_state)
    @static if VERSION < v"1.11-"
        return CC.adce_pass!(ir, inline_state)
    else
        ir, _ = CC.adce_pass!(ir, inline_state)
        return ir
    end
end

function optimize_ir!(ir::IRCode; show_ir=false, do_inline=true, interp=nothing)
    if show_ir
        println("Pre-optimization")
        display(ir)
        println()
    end
    verify(ir)
    ir = strip_coverage_effects!(ir)
    ir = compact!(ir)
    if isnothing(interp)
        local_interp = infer_interp = BugPatchInterpreter()
    else
        local_interp = interp
        infer_interp = BugPatchInterpreter()
    end
    mi = top_level_method_instance(ir, parentmodule(@__MODULE__))
    ir = run_constant_propagation!(ir, infer_interp, mi)
    if show_ir
        println("Post-inference")
        display(ir)
        println()
    end
    inline_state = CC.InliningState(local_interp)
    verify(ir)
    if do_inline
        ir = CC.ssa_inlining_pass!(ir, inline_state, true)#=propagate_inbounds=#
        ir = compact!(ir)
    end
    ir = strip_coverage_effects!(ir)
    ir = CC.sroa_pass!(ir, inline_state)
    ir = run_adce!(ir, inline_state)
    ir = compact!(ir)
    verify_debug_info(ir)
    if show_ir
        println("Post-optimization")
        display(ir)
        println()
    end
    return ir
end

method_matches(x::CC.MethodLookupResult) = x.matches
method_matches(x::Vector{Any}) = x

function infer_ir_for_match(
    interp::CC.AbstractInterpreter,
    match::Core.MethodMatch,
    target::Type{<:Tuple};
    optimize_until=nothing,
)
    @static if VERSION < v"1.11-"
        meth = Base.func_for_method_checked(match.method, target, match.sparams)
        return CC.typeinf_ircode(
            interp, meth, match.spec_types, match.sparams, optimize_until
        )
    else
        return Core.Compiler.typeinf_ircode(interp, match, optimize_until)
    end
end

function infer_ir(interp::CC.AbstractInterpreter, tt::Type{<:Tuple}; optimize_until=nothing)
    matches = CC.findall(tt, CC.method_table(interp))
    asts = []
    for match in method_matches(matches.matches)
        match = match::Core.MethodMatch
        code, ty = infer_ir_for_match(interp, match, tt; optimize_until)
        push!(asts, code === nothing ? match.method => Any : code => ty)
    end
    if isempty(asts)
        msg =
            "No methods found for signature: $tt.\n" *
            "\n" *
            "This is often caused by accidentally trying to get Mooncake.jl to " *
            "differentiate a call (directly or indirectly) which does not exist. For " *
            "example, defining\n" *
            "\n" *
            "f(x::Float64) = sin(x)\n" *
            "build_rrule(Tuple{typeof(f), Int})\n" *
            "\n" *
            "would cause this error, because there are no methods of `f` which accept " *
            "an `Int` argument."
        throw(ArgumentError(msg))
    elseif length(asts) > 1
        throw(ArgumentError("More than one method found for signature $tt."))
    end
    return only(asts)
end

function infer_ir(
    interp::CC.AbstractInterpreter, mi::Core.MethodInstance; optimize_until=nothing
)
    return CC.typeinf_ircode(interp, mi.def, mi.specTypes, mi.sparam_vals, optimize_until)
end

function infer_ir(::CC.AbstractInterpreter, mc::MistyClosure; optimize_until=nothing)
    return mc.ir[], opaque_closure_return_type(mc.oc)
end

opaque_closure_return_type(::Core.OpaqueClosure{A,B}) where {A,B} = B

@static if VERSION > v"1.12-"
    valid_world_range(ir::IRCode) = ir.valid_worlds
    max_valid_world(ir::IRCode) = UInt(CC.max_world(valid_world_range(ir)))
    valid_worlds_as_unit_range(ir::IRCode) =
        UInt(CC.min_world(valid_world_range(ir))):UInt(CC.max_world(valid_world_range(ir)))

    function valid_at_world(ir::IRCode, world::UInt)
        return world in valid_world_range(ir)
    end

    function restrict_to_world(ir::IRCode, world::UInt)
        if !valid_at_world(ir, world)
            error("World $world is not valid for this IRCode: $(valid_world_range(ir)).")
        end
        return CC.IRCode(
            ir.stmts,
            ir.cfg,
            ir.debuginfo,
            ir.argtypes,
            ir.meta,
            ir.sptypes,
            CC.WorldRange(world, world),
        )
    end

    function resolve_globalref_if_unbound_in_world_range(ir::IRCode, ref::GlobalRef)
        if ref.mod === Core || ref.mod === Base
            return ref
        end
        (valid_worlds, alldef) = CC.scan_leaf_partitions(
            nothing,
            ref,
            CC.WorldWithRange(CC.min_world(valid_world_range(ir)), valid_world_range(ir)),
        ) do _, _, bpart
            CC.is_defined_const_binding(CC.binding_kind(bpart))
        end
        if !alldef ||
            CC.max_world(valid_worlds) < CC.max_world(valid_world_range(ir)) ||
            CC.min_world(valid_worlds) > CC.min_world(valid_world_range(ir))
            return isdefined(ref.mod, ref.name) ? getglobal(ref.mod, ref.name) : ref
        end
        return ref
    end
else
    valid_at_world(::IRCode, ::UInt) = true
    restrict_to_world(ir::IRCode, ::UInt) = ir

    resolve_globalref_if_unbound_in_world_range(::IRCode, ref::GlobalRef) = ref
end

@static if VERSION > v"1.12-"
    inferred_return_type(ir::IRCode) = CC.compute_ir_rettype(ir)
    opaque_closure_signature(ir::IRCode, nargs, isva) = CC.compute_oc_signature(
        ir, nargs, isva
    )
else
    inferred_return_type(ir::IRCode) = Base.Experimental.compute_ir_rettype(ir)
    opaque_closure_signature(ir::IRCode, nargs, isva) = Base.Experimental.compute_oc_signature(
        ir, nargs, isva
    )
end

function codeinfo_from_ir(ir::IRCode; nargs, slottypes, isva::Bool=false)
    src = ccall(:jl_new_code_info_uninit, Ref{CC.CodeInfo}, ())
    src.slotnames = [Symbol(:_, i) for i in 1:length(slottypes)]
    src.slotflags = fill(zero(UInt8), length(slottypes))
    src.slottypes = copy(slottypes)
    @static if VERSION > v"1.12-"
        ir.debuginfo.def === nothing &&
            (ir.debuginfo.def = :var"generated IR for OpaqueClosure")
        src.min_world = ir.valid_worlds.min_world
        src.max_world = ir.valid_worlds.max_world
        src.isva = isva
        src.nargs = length(slottypes)
    end
    src = CC.ir_to_codeinf!(src, ir)
    return src
end

function opaque_closure_from_ir(
    ret_type::Type,
    ir::IRCode,
    @nospecialize(env...);
    isva::Bool=false,
    do_compile::Bool=true,
)
    ir = CC.copy(ir)
    set_argument_type!(ir, 1, _typeof(env))
    nargtypes = argument_count(ir)
    nargs = nargtypes - 1
    sig = opaque_closure_signature(ir, nargs, isva)
    src = codeinfo_from_ir(ir; nargs, slottypes=argument_types(ir), isva)
    src.rettype = ret_type
    return Base.Experimental.generate_opaque_closure(
        sig, Union{}, ret_type, src, nargs, isva, env...; do_compile
    )::Core.OpaqueClosure{sig,ret_type}
end

end
