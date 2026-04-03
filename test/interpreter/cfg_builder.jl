@testset "cfg_builder" begin
    @testset "_characterise_unique_predecessor_blocks" begin
        @testset "single block" begin
            blk_id = ID()
            blocks = Mooncake.CFGBlock[Mooncake.CFGBlock(
                blk_id, [(ID(), new_inst(ReturnNode(5)))]
            )]
            upreds, pred_is_upred = Mooncake._characterise_unique_predecessor_blocks(blocks)
            @test upreds[blk_id] == true
            @test pred_is_upred[blk_id] == true
        end

        @testset "pair of blocks" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blocks = Mooncake.CFGBlock[
                Mooncake.CFGBlock(blk_id_1, [(ID(), new_inst(IDGotoNode(blk_id_2)))]),
                Mooncake.CFGBlock(blk_id_2, [(ID(), new_inst(ReturnNode(5)))]),
            ]
            upreds, pred_is_upred = Mooncake._characterise_unique_predecessor_blocks(blocks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == true
            @test pred_is_upred[blk_id_1] == true
            @test pred_is_upred[blk_id_2] == true
        end

        @testset "non-unique exit node" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blk_id_3 = ID()
            blocks = Mooncake.CFGBlock[
                Mooncake.CFGBlock(
                    blk_id_1, [(ID(), new_inst(IDGotoIfNot(true, blk_id_3)))]
                ),
                Mooncake.CFGBlock(blk_id_2, [(ID(), new_inst(ReturnNode(5)))]),
                Mooncake.CFGBlock(blk_id_3, [(ID(), new_inst(ReturnNode(5)))]),
            ]
            upreds, pred_is_upred = Mooncake._characterise_unique_predecessor_blocks(blocks)
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
            blocks = Mooncake.CFGBlock[
                Mooncake.CFGBlock(
                    blk_id_1, [(ID(), new_inst(IDGotoIfNot(true, blk_id_3)))]
                ),
                Mooncake.CFGBlock(blk_id_2, [(ID(), new_inst(IDGotoNode(blk_id_4)))]),
                Mooncake.CFGBlock(blk_id_3, [(ID(), new_inst(IDGotoNode(blk_id_4)))]),
                Mooncake.CFGBlock(blk_id_4, [(ID(), new_inst(ReturnNode(0)))]),
            ]
            upreds, pred_is_upred = Mooncake._characterise_unique_predecessor_blocks(blocks)
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
            blocks = Mooncake.CFGBlock[
                Mooncake.CFGBlock(
                    blk_id_1, [(ID(), new_inst(IDGotoIfNot(true, blk_id_1)))]
                ),
                Mooncake.CFGBlock(blk_id_2, [(ID(), new_inst(ReturnNode(5)))]),
            ]
            upreds, pred_is_upred = Mooncake._characterise_unique_predecessor_blocks(blocks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == true
            @test pred_is_upred[blk_id_1] == false
            @test pred_is_upred[blk_id_2] == true
        end
    end

    @testset "_cfg_distance_to_entry and _canonicalise_cfg_blocks" begin
        blk_id_1 = ID()
        blk_id_2 = ID()
        blk_id_3 = ID()
        blk_id_4 = ID()
        blocks = Mooncake.CFGBlock[
            Mooncake.CFGBlock(blk_id_1, [(ID(), new_inst(IDGotoNode(blk_id_4)))]),
            Mooncake.CFGBlock(blk_id_3, [(ID(), new_inst(ReturnNode(3)))]),
            Mooncake.CFGBlock(blk_id_2, [(ID(), new_inst(ReturnNode(2)))]),
            Mooncake.CFGBlock(blk_id_4, [(ID(), new_inst(IDGotoNode(blk_id_2)))]),
        ]

        @test Mooncake._cfg_distance_to_entry(blocks) == [0, typemax(Int), 2, 1]

        sorted_blocks = Mooncake._sort_cfg_blocks!(copy(blocks))
        @test map(block -> block.id, sorted_blocks) ==
            [blk_id_1, blk_id_4, blk_id_2, blk_id_3]

        canonical_blocks = Mooncake._canonicalise_cfg_blocks(blocks)
        @test map(block -> block.id, canonical_blocks) == [blk_id_1, blk_id_4, blk_id_2]
    end

    @testset "_cfg_control_flow_graph and lower_cfg_blocks_to_ir" begin
        ir = Mooncake.ircode(Any[ReturnNode(nothing)], Any[Any])
        mid_id = ID()
        end_id = ID()
        blocks = Mooncake.CFGBlock[
            Mooncake.CFGBlock(ID(), [(ID(), new_inst(IDGotoNode(mid_id)))]),
            Mooncake.CFGBlock(mid_id, [(ID(), new_inst(IDGotoNode(end_id)))]),
            Mooncake.CFGBlock(end_id, [(ID(), new_inst(ReturnNode(1)))]),
        ]

        lowered_ir = Mooncake.lower_cfg_blocks_to_ir(ir, Any[Any], blocks)
        cfg = Mooncake._cfg_control_flow_graph(Mooncake._canonicalise_cfg_blocks(blocks))

        @test all(
            map((lhs, rhs) -> lhs.stmts == rhs.stmts, lowered_ir.cfg.blocks, cfg.blocks)
        )
        @test all(
            map((lhs, rhs) -> lhs.preds == rhs.preds, lowered_ir.cfg.blocks, cfg.blocks)
        )
        @test all(
            map((lhs, rhs) -> lhs.succs == rhs.succs, lowered_ir.cfg.blocks, cfg.blocks)
        )
        @test lowered_ir.cfg.index == cfg.index
    end
end
