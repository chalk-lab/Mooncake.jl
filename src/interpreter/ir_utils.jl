stmt_field_name() = Compiler.instruction_statement_field()

"""
    stmt(ir::CC.InstructionStream)

Get the field containing the instructions in `ir`. This changed name in 1.11 from `inst` to
`stmt`.
"""
stmt(ir::CC.InstructionStream) = Compiler.statements(ir)

"""
    stmt(x::CC.Instruction)

Get the statement from `x`. This field changed name in 1.11 from `inst` to `stmt`.
"""
stmt(x::CC.Instruction) = Compiler.statement(x)

set_stmt!(ir::IRCode, ssa::SSAValue, a) = Compiler.set_statement!(ir, ssa, a)

get_ir(ir::IRCode, idx::SSAValue) = Compiler.instruction(ir, idx)
get_ir(ir::IRCode, idx::SSAValue, name::Symbol) = CC.getindex(get_ir(ir, idx), name)

"""

"""
function set_ir!(ir::IRCode, idx::SSAValue, name::Symbol, value)
    return CC.setindex!(Compiler.instruction(ir, idx), value, name)
end

function replace_call!(ir, idx::SSAValue, new_call)
    Compiler.replace_statement!(ir, idx, new_call)
    return nothing
end

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
    return Compiler.ircode(insts, argtypes, sptypes)
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
    return Compiler.instruction_stream_from_statements(insts)
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
    return Compiler.infer_ir!(ir)
end

# Given some IR, generates a MethodInstance suitable for passing to infer_ir!, if you don't
# already have one with the right argument types. Credit to @oxinabox:
# https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54
function __get_toplevel_mi_from_ir(ir, _module::Module)
    return Compiler.top_level_method_instance(ir, _module)
end

# Run type inference and constant propagation on the ir. Credit to @oxinabox:
# https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54
function __infer_ir!(ir, interp::CC.AbstractInterpreter, mi::CC.MethodInstance)
    return Compiler.run_constant_propagation!(ir, interp, mi)
end

# In automatically generated code, it is meaningless to include code coverage effects.
# Moreover, it seems to cause some serious inference problems. Consequently, it makes sense
# to remove such effects before optimising IRCode.
function __strip_coverage!(ir::IRCode)
    return Compiler.strip_coverage_effects!(ir)
end

"""
    optimise_ir!(ir::IRCode, show_ir=false)

Run a fairly standard optimisation pass on `ir`. If `show_ir` is `true`, displays the IR
to `stdout` at various points in the pipeline -- this is sometimes useful for debugging.
"""
function optimise_ir!(ir::IRCode; show_ir=false, do_inline=true, interp=nothing)
    return Compiler.optimize_ir!(ir; show_ir, do_inline, interp)
end

# Handles difference between 1.10 and 1.11.
get_matches(x) = Compiler.method_matches(x)

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
    return Compiler.infer_ir(interp, tt; optimize_until)
end

function lookup_ir(
    interp::CC.AbstractInterpreter, mi::Core.MethodInstance; optimize_until=nothing
)
    return Compiler.infer_ir(interp, mi; optimize_until)
end

function lookup_ir(interp::CC.AbstractInterpreter, mc::MistyClosure; optimize_until=nothing)
    return Compiler.infer_ir(interp, mc; optimize_until)
end

"""
    set_valid_world!(ir::IRCode, world::UInt)::IRCode

Compatibility shim for [`Compiler.restrict_to_world`](@ref).
"""
function set_valid_world!(ir::IRCode, world::UInt)
    return Compiler.restrict_to_world(ir, world)
end

return_type(oc::Core.OpaqueClosure) = Compiler.opaque_closure_return_type(oc)

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
            return GotoIfNot(val, stmt.dest)
        else
            return stmt
        end
    else
        return stmt
    end
end

"""
    characterised_used_ssas(stmts::Vector{Any})::Vector{Bool}

For each statement in `stmts`, determine whether the SSAValue that it corresponds to has
any uses in the other statements. In particular, if `SSAValue(n)` has any uses in `stmts`,
the `n`th element of the returned `Vector{Bool}` will be `true`, and `false` otherwise.

This function will usually be applied to the `stmts` field of an `CC.InstructionStream`.
"""
function characterised_used_ssas(stmts::Vector{Any})::Vector{Bool}
    is_used = fill(false, length(stmts))
    for stmt in stmts

        # Manually written-out iteration to avoid Core.Compiler type piracy.
        urs = CC.userefs(stmt)
        v = CC.iterate(urs)
        while v !== nothing
            (use_ref, state) = v
            use = CC.getindex(use_ref)
            if use isa SSAValue
                is_used[use.id] = true
            end
            v = CC.iterate(urs, state)
        end
    end
    return is_used
end
