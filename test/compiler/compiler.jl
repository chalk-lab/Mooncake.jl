@testset "Compiler services" begin
    interp = Mooncake.MooncakeInterpreter(ForwardMode)
    ir, rt = Mooncake.Compiler.infer_ir(interp, Tuple{typeof(sin),Float64})

    @test ir isa CC.IRCode
    @test rt !== nothing
    @test Mooncake.Compiler.argument_count(ir) ==
        length(Mooncake.Compiler.argument_types(ir))
    @test Mooncake.Compiler.statement_stream(ir) === ir.stmts
    @test Mooncake.Compiler.statements(ir) isa Vector

    restricted_ir = Mooncake.Compiler.restrict_to_world(ir, interp.world)
    @test Mooncake.Compiler.valid_at_world(restricted_ir, interp.world)

    ir = Mooncake.ircode(Any[ReturnNode(Argument(2))], Any[typeof(identity),Float64])
    Mooncake.Compiler.replace_statement!(ir, SSAValue(1), ReturnNode(Argument(1)))
    @test Mooncake.Compiler.statement(Mooncake.Compiler.instruction(ir, SSAValue(1))) isa
        ReturnNode

    bb = Mooncake.BBCode(ir)
    updated_bb = Mooncake.Compiler.with_argument_types(bb, Type[typeof(identity),Float64])
    @test Mooncake.Compiler.argument_types(updated_bb) == Any[typeof(identity), Float64]
end
