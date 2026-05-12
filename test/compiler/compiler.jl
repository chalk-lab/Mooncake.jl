@testset "Compiler services" begin
    interp = Mooncake.MooncakeInterpreter(ForwardMode)
    reverse_interp = Mooncake.MooncakeInterpreter(DefaultCtx, ReverseMode)

    @test Mooncake.Compiler.interpreter_world(reverse_interp) == reverse_interp.world
    @test Mooncake.Compiler.inference_parameters(reverse_interp) ===
        reverse_interp.inf_params
    @test Mooncake.Compiler.optimization_parameters(reverse_interp) ===
        reverse_interp.opt_params
    @test Mooncake.Compiler.inference_cache(reverse_interp) === reverse_interp.inf_cache
    @test Mooncake.Compiler.code_cache_view(reverse_interp).cache ===
        reverse_interp.code_cache
    @test Mooncake.Compiler.overlay_method_table(
        reverse_interp, Mooncake.mooncake_method_table
    ) isa CC.OverlayMethodTable
    @static if VERSION >= v"1.11.0"
        @test Mooncake.Compiler.cache_owner(reverse_interp) === nothing
    end

    noinline_info = Mooncake.NoInlineCallInfo(CC.NoCallInfo(), Tuple{typeof(sin),Float64})
    @test !Mooncake.Compiler.should_inline_call(noinline_info)
    @test Mooncake.Compiler.should_inline_call(CC.NoCallInfo())

    ir, rt = Mooncake.Compiler.infer_ir(interp, Tuple{typeof(sin),Float64})

    @test ir isa CC.IRCode
    @test rt !== nothing
    @test Mooncake.Compiler.argument_count(ir) ==
        length(Mooncake.Compiler.argument_types(ir))
    @test Mooncake.Compiler.statement_stream(ir) === ir.stmts
    @test Mooncake.Compiler.statements(ir) isa Vector
    @test Mooncake.Compiler.statement_types(ir) isa Vector
    @test Mooncake.Compiler.static_parameter_map(ir, Symbol[]) == Dict{Symbol,CC.VarState}()

    restricted_ir = Mooncake.Compiler.restrict_to_world(ir, interp.world)
    @test Mooncake.Compiler.valid_at_world(restricted_ir, interp.world)

    ir = Mooncake.ircode(Any[ReturnNode(Argument(2))], Any[typeof(identity), Float64])
    @test Mooncake.Compiler.statement_type(ir, 1) == Any
    Mooncake.Compiler.set_statement_type!(ir, 1, Nothing)
    @test Mooncake.Compiler.statement_type(ir, 1) == Nothing
    Mooncake.Compiler.replace_statement!(ir, SSAValue(1), ReturnNode(Argument(1)))
    @test Mooncake.Compiler.statement(Mooncake.Compiler.instruction(ir, SSAValue(1))) isa
        ReturnNode

    bb = Mooncake.BBCode(ir)
    updated_bb = Mooncake.Compiler.with_argument_types(bb, Type[typeof(identity), Float64])
    @test Mooncake.Compiler.argument_types(updated_bb) == Any[typeof(identity), Float64]

    branch_ir = Mooncake.ircode(
        Any[
            Expr(:call, :sin, Argument(2)),
            GotoNode(3),
            PhiNode(Int32[1], Any[5]),
            ReturnNode(SSAValue(3)),
        ],
        Any[Any, Vector{Float64}],
    )
    @test Mooncake.Compiler.block_for_statement(branch_ir, 1) == 1
    @test Mooncake.Compiler.block_for_statement(branch_ir, 3) == 2
    Mooncake.Compiler.remove_control_flow_edge!(branch_ir, 1, 2)
    phi_node = Mooncake.Compiler.statements(branch_ir)[3]
    @test isempty(phi_node.edges)
    @test isempty(phi_node.values)
    @test isempty(Mooncake.Compiler.block_successors(branch_ir, 1))
    @test isempty(Mooncake.Compiler.block_predecessors(branch_ir, 2))
end
