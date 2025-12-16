module BBCodeTestCases
test_phi_node(x::Ref{Union{Float32,Float64}}) = sin(x[])
end

@testset "bbcode" begin
    @testset "ID" begin
        id1 = ID()
        id2 = ID()
        @test id1 == id1
        @test id1 != id2
    end
    @testset "BBlock" begin
        bb = BBlock(
            ID(),
            ID[ID(), ID()],
            CC.NewInstruction[
                CC.NewInstruction(IDPhiNode([ID(), ID()], Any[true, false]), Any),
                CC.NewInstruction(:(println("hello")), Any),
            ],
        )
        @test bb isa BBlock
        @test length(bb) == 2

        ids, phi_nodes = Mooncake.phi_nodes(bb)
        @test only(ids) == bb.inst_ids[1]
        @test only(phi_nodes) == bb.insts[1]

        insert!(bb, 1, ID(), CC.NewInstruction(nothing, Nothing))
        @test length(bb) == 3
        @test bb.insts[1].stmt === nothing

        bb_copy = copy(bb)
        @test bb_copy.inst_ids !== bb.inst_ids

        @test Mooncake.terminator(bb) === nothing

        # Final statment is regular instruction, so newly inserted instruction should go at
        # the end of the block.
        @test Mooncake.insert_before_terminator!(bb, ID(), new_inst(ReturnNode(5))) ===
            nothing
        @test bb.insts[end].stmt === ReturnNode(5)

        # Final statement is now a Terminator, so insertion should happen before it.
        @test Mooncake.insert_before_terminator!(bb, ID(), new_inst(nothing)) === nothing
        @test bb.insts[end].stmt === ReturnNode(5)
        @test bb.insts[end - 1].stmt === nothing
    end
    @testset "BBCode $f" for (f, P) in [
        (TestResources.test_while_loop, Tuple{Float64}),
        (sin, Tuple{Float64}),
        (BBCodeTestCases.test_phi_node, Tuple{Ref{Union{Float32,Float64}}}),
    ]
        ir = Base.code_ircode(f, P)[1][1]
        bb_code = BBCode(ir)
        @test bb_code isa BBCode
        @test length(bb_code.blocks) == length(ir.cfg.blocks)
        new_ir = Mooncake.IRCode(bb_code)
        @test length(stmt(new_ir.stmts)) == length(stmt(ir.stmts))
        @test all(map(==, stmt(ir.stmts), stmt(new_ir.stmts)))
        @test all(map(==, ir.stmts.type, new_ir.stmts.type))
        @test all(map(==, ir.stmts.info, new_ir.stmts.info))
        @test all(map(==, ir.stmts.line, new_ir.stmts.line))
        @test all(map(==, ir.stmts.flag, new_ir.stmts.flag))
        @test length(Mooncake.collect_stmts(bb_code)) == length(stmt(ir.stmts))
        @test Mooncake.BasicBlockCode.id_to_line_map(bb_code) isa Dict{ID,Int}
    end
    @testset "control_flow_graph" begin
        ir = Base.code_ircode_by_type(Tuple{typeof(sin),Float64})[1][1]
        bb = BBCode(ir)
        new_ir = Core.Compiler.IRCode(bb)
        cfg = Mooncake.BasicBlockCode.control_flow_graph(bb)
        @test all(map((l, r) -> l.stmts == r.stmts, ir.cfg.blocks, cfg.blocks))
        @test all(map((l, r) -> sort(l.preds) == sort(r.preds), ir.cfg.blocks, cfg.blocks))
        @test all(map((l, r) -> sort(l.succs) == sort(r.succs), ir.cfg.blocks, cfg.blocks))
        @test ir.cfg.index == cfg.index
    end
    @testset "_characterise_unique_predecessor_blocks" begin
        @testset "single block" begin
            blk_id = ID()
            blks = BBlock[BBlock(blk_id, [ID()], [new_inst(ReturnNode(5))])]
            upreds, pred_is_upred = characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id] == true
            @test pred_is_upred[blk_id] == true
        end
        @testset "pair of blocks" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blks = BBlock[
                BBlock(blk_id_1, [ID()], [new_inst(IDGotoNode(blk_id_2))]),
                BBlock(blk_id_2, [ID()], [new_inst(ReturnNode(5))]),
            ]
            upreds, pred_is_upred = characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == true
            @test pred_is_upred[blk_id_1] == true
            @test pred_is_upred[blk_id_2] == true
        end
        @testset "Non-Unique Exit Node" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blk_id_3 = ID()
            blks = BBlock[
                BBlock(blk_id_1, [ID()], [new_inst(IDGotoIfNot(true, blk_id_3))]),
                BBlock(blk_id_2, [ID()], [new_inst(ReturnNode(5))]),
                BBlock(blk_id_3, [ID()], [new_inst(ReturnNode(5))]),
            ]
            upreds, pred_is_upred = characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == false
            @test upreds[blk_id_3] == false
            @test pred_is_upred[blk_id_1] == true
            @test pred_is_upred[blk_id_2] == true
            @test pred_is_upred[blk_id_3] == true
        end
        @testset "diamond structure of four blocks" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blk_id_3 = ID()
            blk_id_4 = ID()
            blks = BBlock[
                BBlock(blk_id_1, [ID()], [new_inst(IDGotoIfNot(true, blk_id_3))]),
                BBlock(blk_id_2, [ID()], [new_inst(IDGotoNode(blk_id_4))]),
                BBlock(blk_id_3, [ID()], [new_inst(IDGotoNode(blk_id_4))]),
                BBlock(blk_id_4, [ID()], [new_inst(ReturnNode(0))]),
            ]
            upreds, pred_is_upred = characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == false
            @test upreds[blk_id_3] == false
            @test upreds[blk_id_4] == true
            @test pred_is_upred[blk_id_1] == true
            @test pred_is_upred[blk_id_2] == true
            @test pred_is_upred[blk_id_3] == true
            @test pred_is_upred[blk_id_4] == false
        end
        @testset "simple loop back to first block" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blks = BBlock[
                BBlock(blk_id_1, [ID()], [new_inst(IDGotoIfNot(true, blk_id_1))]),
                BBlock(blk_id_2, [ID()], [new_inst(ReturnNode(5))]),
            ]
            upreds, pred_is_upred = characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == true
            @test pred_is_upred[blk_id_1] == false
            @test pred_is_upred[blk_id_2] == true
        end
    end
    @testset "characterise_used_ids" begin
        @testset "_find_id_uses!" begin
            @testset "Expr" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake.BasicBlockCode._find_id_uses!(d, Expr(:call, sin, 5))
                @test d[id] == false
                Mooncake.BasicBlockCode._find_id_uses!(d, Expr(:call, sin, id))
                @test d[id] == true
            end
            @testset "IDGotoIfNot" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake.BasicBlockCode._find_id_uses!(d, IDGotoIfNot(ID(), ID()))
                @test d[id] == false
                Mooncake.BasicBlockCode._find_id_uses!(d, IDGotoIfNot(true, ID()))
                @test d[id] == false
                Mooncake.BasicBlockCode._find_id_uses!(d, IDGotoIfNot(id, ID()))
                @test d[id] == true
            end
            @testset "IDGotoNode" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake.BasicBlockCode._find_id_uses!(d, IDGotoNode(ID()))
                @test d[id] == false
            end
            @testset "IDPhiNode" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake.BasicBlockCode._find_id_uses!(
                    d, IDPhiNode([ID()], Vector{Any}(undef, 1))
                )
                @test d[id] == false
                Mooncake.BasicBlockCode._find_id_uses!(d, IDPhiNode([ID()], Any[id]))
                @test d[id] == true
            end
            @testset "PiNode" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake.BasicBlockCode._find_id_uses!(d, PiNode(false, Bool))
                @test d[id] == false
                Mooncake.BasicBlockCode._find_id_uses!(d, PiNode(id, Bool))
                @test d[id] == true
            end
            @testset "ReturnNode" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake.BasicBlockCode._find_id_uses!(d, ReturnNode())
                @test d[id] == false
                Mooncake.BasicBlockCode._find_id_uses!(d, ReturnNode(5))
                @test d[id] == false
                Mooncake.BasicBlockCode._find_id_uses!(d, ReturnNode(id))
                @test d[id] == true
            end
        end
        @testset "some used some unused" begin
            id_1 = ID()
            id_2 = ID()
            id_3 = ID()
            stmts = Tuple{ID,Core.Compiler.NewInstruction}[
                (id_1, new_inst(Expr(:call, sin, Argument(1)))),
                (id_2, new_inst(Expr(:call, cos, id_1))),
                (id_3, new_inst(ReturnNode(id_2))),
            ]
            result = characterise_used_ids(stmts)
            @test result[id_1] == true
            @test result[id_2] == true
            @test result[id_3] == false
        end
    end
    @testset "_is_reachable" begin
        ir = Mooncake.ircode(
            Any[
                ReturnNode(nothing),
                Expr(:call, sin, 5),
                Core.GotoNode(4),
                ReturnNode(SSAValue(2)),
            ],
            Any[Any for _ in 1:4],
        )
        @test Mooncake.BasicBlockCode._is_reachable(BBCode(ir).blocks) ==
            [true, false, false]
    end
    @testset "remove_unreachable_blocks!" begin

        # This test case has two important features:
        # 1. the second basic block (the second statement) cannot be reached, and
        # 2. the PhiNode in the third basic block refers to the second basic block. Since
        #   the second block will be removed, the edge / value in the PhiNode corresponding
        #   to the second block must be removed as part of the call to
        #   remove_unreachable_blocks.
        ir = Mooncake.ircode(
            Any[
                GotoNode(3),
                nothing,
                PhiNode(Int32[2, 1], Any[false, true]),
                ReturnNode(SSAValue(3)),
            ],
            Any[Any for _ in 1:4],
        )
        CC.verify_ir(ir)
        bb_ir = BBCode(ir)
        new_bb_ir = Mooncake.remove_unreachable_blocks!(bb_ir)

        # Check that only the first and third block remain in the new IR.
        @test length(new_bb_ir.blocks) == 2
        @test bb_ir.blocks[1].id == new_bb_ir.blocks[1].id
        @test bb_ir.blocks[3].id == new_bb_ir.blocks[2].id

        # Check that the reference to the second block in the PhiNode has been removed.
        # Do this by checking that the only
        updated_id_phi_node = new_bb_ir.blocks[2].insts[1].stmt
        @test length(updated_id_phi_node.edges) == 1
        @test length(updated_id_phi_node.values) == 1
        @test only(updated_id_phi_node.values) == true

        # Get the IRCode, and ensure that the statements in it agree with what is expected.
        new_ir = CC.IRCode(new_bb_ir)
        expected_stmts = Any[
            GotoNode(2), PhiNode(Int32[1], Any[true]), ReturnNode(SSAValue(2))
        ]
        @test Mooncake.stmt(new_ir.stmts) == expected_stmts
    end
end
