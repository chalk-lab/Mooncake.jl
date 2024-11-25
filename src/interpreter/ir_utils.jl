"""
    const InstVector = Vector{NewInstruction}

Note: the `CC.NewInstruction` type is used to represent instructions because it has the
correct fields. While it is only used to represent new instrucdtions in `Core.Compiler`, it
is used to represent all instructions in `BBCode`.
"""
const InstVector = Vector{NewInstruction}

"""
    stmt(ir::CC.InstructionStream)

Get the field containing the instructions in `ir`. This changed name in 1.11 from `inst` to
`stmt`.
"""
stmt(ir::CC.InstructionStream) = @static VERSION < v"1.11.0-rc4" ? ir.inst : ir.stmt

"""
    ircode(
        inst::Vector{Any},
        argtypes::Vector{Any},
        sptypes::Vector{CC.VarState}=CC.VarState[],
    ) -> IRCode

Constructs an instance of an `IRCode`. This is useful for constructing test cases with known
properties.

No optimisations or type inference are performed on the resulting `IRCode`, so that
the `IRCode` contains exactly what is intended by the caller. Please make use of
`infer_types!` if you require the types to be inferred.

Edges in `PhiNode`s, `GotoIfNot`s, and `GotoNode`s found in `inst` must refer to lines (as
in `CodeInfo`). In the `IRCode` returned by this function, these line references are
translated into block references.
"""
function ircode(
    insts::Vector{Any}, argtypes::Vector{Any}, sptypes::Vector{CC.VarState}=CC.VarState[]
)
    cfg = CC.compute_basic_blocks(insts)
    insts = __line_numbers_to_block_numbers!(insts, cfg)
    stmts = __insts_to_instruction_stream(insts)
    linetable = [CC.LineInfoNode(Mooncake, :ircode, :ir_utils, Int32(1), Int32(0))]
    meta = Expr[]
    return CC.IRCode(stmts, cfg, linetable, argtypes, meta, CC.VarState[])
end

"""
    __line_numbers_to_block_numbers!(insts::Vector{Any}, cfg::CC.CFG)

Converts any edges in `GotoNode`s, `GotoIfNot`s, `PhiNode`s, and `:enter` expressions which
refer to line numbers into references to block numbers. The `cfg` provides the information
required to perform this conversion.

For context, `CodeInfo` objects have references to line numbers, while `IRCode` uses
block numbers.

This code is copied over directly from the body of `Core.Compiler.inflate_ir!`.
"""
function __line_numbers_to_block_numbers!(insts::Vector{Any}, cfg::CC.CFG)
    for i in eachindex(insts)
        stmt = insts[i]
        if isa(stmt, GotoNode)
            insts[i] = GotoNode(CC.block_for_inst(cfg, stmt.label))
        elseif isa(stmt, GotoIfNot)
            insts[i] = GotoIfNot(stmt.cond, CC.block_for_inst(cfg, stmt.dest))
        elseif isa(stmt, PhiNode)
            insts[i] = PhiNode(
                Int32[CC.block_for_inst(cfg, Int(edge)) for edge in stmt.edges], stmt.values
            )
        elseif Meta.isexpr(stmt, :enter)
            stmt.args[1] = CC.block_for_inst(cfg, stmt.args[1]::Int)
            insts[i] = stmt
        end
    end
    return insts
end

"""
    _instructions_to_blocks(insts::InstVector, cfg::CC.CFG)::InstVector

Pulls out the instructions from `insts`, and calls `__line_numbers_to_block_numbers!`.
"""
function _lines_to_blocks(insts::InstVector, cfg::CC.CFG)::InstVector
    stmts = __line_numbers_to_block_numbers!(Any[x.stmt for x in insts], cfg)
    return map((inst, stmt) -> NewInstruction(inst; stmt), insts, stmts)
end

"""
    __insts_to_instruction_stream(insts::Vector{Any})

Produces an instruction stream whose
- `stmt` (v1.11 and up) / `inst` (v1.10) field is `insts`,
- `type` field is all `Any`,
- `info` field is all `Core.Compiler.NoCallInfo`,
- `line` field is all `Int32(1)`, and
- `flag` field is all `Core.Compiler.IR_FLAG_REFINED`.

As such, if you wish to ensure that your `IRCode` prints nicely, you should ensure that its
linetable field has at least one element.
"""
function __insts_to_instruction_stream(insts::Vector{Any})
    return CC.InstructionStream(
        insts,
        fill(Any, length(insts)),
        fill(CC.NoCallInfo(), length(insts)),
        fill(Int32(1), length(insts)),
        fill(CC.IR_FLAG_REFINED, length(insts)),
    )
end

"""
    infer_ir!(ir::IRCode) -> IRCode

Runs type inference on `ir`, which mutates `ir`, and returns it.

Note: the compiler will not infer the types of anything where the corrsponding element of
`ir.stmts.flag` is not set to `Core.Compiler.IR_FLAG_REFINED`. Nor will it attempt to refine
the type of the value returned by a `:invoke` expressions. Consequently, if you find that
the types in your IR are not being refined, you may wish to check that neither of these
things are happening.
"""
function infer_ir!(ir::IRCode)
    return __infer_ir!(ir, CC.NativeInterpreter(), __get_toplevel_mi_from_ir(ir, Mooncake))
end

# Given some IR, generates a MethodInstance suitable for passing to infer_ir!, if you don't
# already have one with the right argument types. Credit to @oxinabox:
# https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54
function __get_toplevel_mi_from_ir(ir, _module::Module)
    mi = ccall(:jl_new_method_instance_uninit, Ref{Core.MethodInstance}, ())
    mi.specTypes = Tuple{map(_type, ir.argtypes)...}
    mi.def = _module
    return mi
end

# Run type inference and constant propagation on the ir. Credit to @oxinabox:
# https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54
function __infer_ir!(ir, interp::CC.AbstractInterpreter, mi::CC.MethodInstance)
    method_info = CC.MethodInfo(true, nothing) #=propagate_inbounds=#
    min_world = world = get_inference_world(interp)
    max_world = Base.get_world_counter()
    irsv = CC.IRInterpretationState(
        interp, method_info, ir, mi, ir.argtypes, world, min_world, max_world
    )
    rt = CC._ir_abstract_constant_propagation(interp, irsv)
    return ir
end

# In automatically generated code, it is meaningless to include code coverage effects.
# Moreover, it seems to cause some serious inference probems. Consequently, it makes sense
# to remove such effects before optimising IRCode.
function __strip_coverage!(ir::IRCode)
    for n in eachindex(stmt(ir.stmts))
        if Meta.isexpr(stmt(ir.stmts)[n], :code_coverage_effect)
            stmt(ir.stmts)[n] = nothing
        end
    end
    return ir
end

"""
    optimise_ir!(ir::IRCode, show_ir=false)

Run a fairly standard optimisation pass on `ir`. If `show_ir` is `true`, displays the IR
to `stdout` at various points in the pipeline -- this is sometimes useful for debugging.
"""
function optimise_ir!(ir::IRCode; show_ir=false, do_inline=true)
    if show_ir
        println("Pre-optimization")
        display(ir)
        println()
    end
    CC.verify_ir(ir)
    ir = __strip_coverage!(ir)
    ir = CC.compact!(ir)
    local_interp = CC.NativeInterpreter()
    mi = __get_toplevel_mi_from_ir(ir, @__MODULE__)
    ir = __infer_ir!(ir, local_interp, mi)
    if show_ir
        println("Post-inference")
        display(ir)
        println()
    end
    inline_state = CC.InliningState(local_interp)
    CC.verify_ir(ir)
    if do_inline
        ir = CC.ssa_inlining_pass!(ir, inline_state, true) #=propagate_inbounds=#
        ir = CC.compact!(ir)
    end
    ir = __strip_coverage!(ir)
    ir = CC.sroa_pass!(ir, inline_state)

    @static if VERSION < v"1.11-"
        ir = CC.adce_pass!(ir, inline_state)
    else
        ir, _ = CC.adce_pass!(ir, inline_state)
    end

    ir = CC.compact!(ir)
    # CC.verify_ir(ir, true, false, CC.optimizer_lattice(local_interp))
    CC.verify_linetable(ir.linetable, true)
    if show_ir
        println("Post-optimization")
        display(ir)
        println()
    end
    return ir
end

# Handles difference between 1.10 and 1.11.
get_matches(x::CC.MethodLookupResult) = x.matches
get_matches(x::Vector{Any}) = x

"""
    lookup_ir(
        interp::AbstractInterpreter,
        sig_or_mi::Union{Type{<:Tuple}, Core.MethodInstance},
    )::Tuple{IRCode, T}

Get the unique IR associated to `sig_or_mi` under `interp`. Throws `ArgumentError`s if
there is no code found, or if more than one `IRCode` instance returned.

Returns a tuple containing the `IRCode` and its return type.
"""
function lookup_ir(
    interp::CC.AbstractInterpreter, tt::Type{<:Tuple}; optimize_until=nothing
)
    matches = CC.findall(tt, CC.method_table(interp))
    asts = []
    for match in get_matches(matches.matches)
        match = match::Core.MethodMatch
        @static if VERSION < v"1.11-"
            meth = Base.func_for_method_checked(match.method, tt, match.sparams)
            (code, ty) = CC.typeinf_ircode(
                interp, meth, match.spec_types, match.sparams, optimize_until
            )
        else
            (code, ty) = Core.Compiler.typeinf_ircode(interp, match, optimize_until)
        end
        if code === nothing
            push!(asts, match.method => Any)
        else
            push!(asts, code => ty)
        end
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

function lookup_ir(
    interp::CC.AbstractInterpreter, mi::Core.MethodInstance; optimize_until=nothing
)
    return CC.typeinf_ircode(interp, mi.def, mi.specTypes, mi.sparam_vals, optimize_until)
end

"""
    is_reachable_return_node(x::ReturnNode)

Determine whether `x` is a `ReturnNode`, and if it is, if it is also reachable. This is
purely a function of whether or not its `val` field is defined or not.
"""
is_reachable_return_node(x::ReturnNode) = isdefined(x, :val)
is_reachable_return_node(x) = false

"""
    is_unreachable_return_node(x::ReturnNode)

Determine whehter `x` is a `ReturnNode`, and if it is, if it is also unreachable. This is
purely a function of whether or not its `val` field is defined or not.
"""
is_unreachable_return_node(x::ReturnNode) = !isdefined(x, :val)
is_unreachable_return_node(x) = false

"""
    UnhandledLanguageFeatureException(message::String)

An exception used to indicate that some aspect of the Julia language which AD cannot handle
has been encountered.
"""
struct UnhandledLanguageFeatureException <: Exception
    msg::String
end

"""
    unhandled_feature(msg::String)

Throw an `UnhandledLanguageFeatureException` with message `msg`.
"""
unhandled_feature(msg::String) = throw(UnhandledLanguageFeatureException(msg))

"""
    new_inst(stmt, type=Any, flag=CC.IR_FLAG_REFINED)::NewInstruction

Create a `NewInstruction` with fields:
- `stmt` = `stmt`
- `type` = `type`
- `info` = `CC.NoCallInfo()`
- `line` = `Int32(1)`
- `flag` = `flag`
"""
function new_inst(@nospecialize(stmt), @nospecialize(type)=Any, flag=CC.IR_FLAG_REFINED)
    return NewInstruction(stmt, type, CC.NoCallInfo(), Int32(1), flag)
end

"""
    replace_uses_with!(stmt, def::Union{Argument, SSAValue}, val)

Replace all uses of `def` with `val` in the single statement `stmt`.
Note: this function is highly incomplete, really only working correctly for a specific
function in `ir_normalisation.jl`. You probably do not want to use it.
"""
function replace_uses_with!(stmt, def::Union{Argument,SSAValue}, val)
    if stmt isa Expr
        stmt.args = Any[arg == def ? val : arg for arg in stmt.args]
        return stmt
    elseif stmt isa GotoIfNot
        if stmt.cond == def
            @assert val isa Bool
            return val === true ? nothing : GotoNode(stmt.dest)
        else
            return stmt
        end
    else
        return stmt
    end
end
