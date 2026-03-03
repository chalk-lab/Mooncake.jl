include(joinpath(@__DIR__, "scripts", "ir_inspect.jl"))

# Test helper function
test_fn(x) = sin(x) * cos(x)
multi_arg_fn(x, y) = x * y + sin(x)

@testset "ir_inspect" begin
    @testset "inspect_ir reverse mode" begin
        ins = inspect_ir(sin, 1.0)
        @test ins.mode == :reverse
        @test ins.sig == Tuple{typeof(sin),Float64}
        @test ins.world isa UInt
        @test isempty(ins.notes)

        # All reverse-mode stages should be present
        expected_stages = [:raw, :normalized, :bbcode, :fwd_ir, :rvs_ir, :optimized_fwd, :optimized_rvs]
        @test ins.stage_order == expected_stages
        for s in expected_stages
            @test haskey(ins.stages, s)
        end

        # Stage metadata
        for s in expected_stages
            stage = ins.stages[s]
            @test stage.name == s
            @test stage.meta.block_count > 0
            @test stage.meta.inst_count > 0
            @test !isempty(stage.text)
        end

        # Diffs should be computed for all edges in the stage graph
        @test length(ins.diffs) == length(ins.stage_graph)
        for (from, to) in ins.stage_graph
            @test haskey(ins.diffs, from => to)
        end

        # CFGs should be computed for all stages
        @test length(ins.cfgs) == length(ins.stages)
        for (name, dot) in ins.cfgs
            @test occursin("digraph", dot)
        end
    end

    @testset "inspect_ir forward mode" begin
        ins = inspect_fwd(sin, 1.0)
        @test ins.mode == :forward

        expected_stages = [:raw, :normalized, :bbcode, :dual_ir, :optimized]
        @test ins.stage_order == expected_stages
        for s in expected_stages
            @test haskey(ins.stages, s)
            @test ins.stages[s].meta.block_count > 0
        end
    end

    @testset "inspect_ir optimize=false" begin
        ins = inspect_ir(sin, 1.0; optimize=false)
        @test ins.mode == :reverse
        @test :raw in ins.stage_order
        @test :fwd_ir in ins.stage_order
        @test :rvs_ir in ins.stage_order
        @test !haskey(ins.stages, :optimized_fwd)
        @test !haskey(ins.stages, :optimized_rvs)

        ins_fwd = inspect_fwd(sin, 1.0; optimize=false)
        @test :dual_ir in ins_fwd.stage_order
        @test !haskey(ins_fwd.stages, :optimized)
    end

    @testset "inspect_ir multi-arg function" begin
        ins = inspect_ir(multi_arg_fn, 1.0, 2.0)
        @test ins.sig == Tuple{typeof(multi_arg_fn),Float64,Float64}
        @test length(ins.stages) > 0
    end

    @testset "show_ir" begin
        ins = inspect_fwd(sin, 1.0)
        io = IOBuffer()
        show_ir(ins; io)
        output = String(take!(io))
        @test occursin("IR Inspection", output)
        @test occursin("Mode: forward", output)
        for s in ins.stage_order
            @test occursin("Stage: $s", output)
        end
    end

    @testset "show_stage" begin
        ins = inspect_ir(sin, 1.0)
        io = IOBuffer()
        show_stage(ins, :raw; io)
        output = String(take!(io))
        @test occursin("Stage: raw", output)
        @test !occursin("Stage: normalized", output)
    end

    @testset "diff_ir and show_diff" begin
        ins = inspect_fwd(sin, 1.0)

        # diff_ir returns a string
        d = diff_ir(ins; from=:raw, to=:normalized)
        @test d isa String
        @test occursin("---", d)
        @test occursin("+++", d)

        # On-demand diff for non-precomputed pairs
        d2 = diff_ir(ins; from=:raw, to=:dual_ir)
        @test d2 isa String

        # Missing stages
        d3 = diff_ir(ins; from=:nonexistent, to=:raw)
        @test occursin("not found", d3)

        # show_diff writes to IO
        io = IOBuffer()
        show_diff(ins; from=:raw, to=:normalized, io)
        output = String(take!(io))
        @test occursin("Diff:", output)
    end

    @testset "show_all_diffs" begin
        ins = inspect_fwd(sin, 1.0)
        io = IOBuffer()
        show_all_diffs(ins; io)
        output = String(take!(io))
        @test !isempty(output)
        # Should contain one diff header per edge
        for (from, to) in ins.stage_graph
            @test occursin("Diff: $from", output)
        end
    end

    @testset "cfg_dot" begin
        ins = inspect_ir(sin, 1.0)

        dot = cfg_dot(ins, :raw)
        @test occursin("digraph", dot)
        @test occursin("->", dot)
        @test occursin("Block", dot)

        dot_bb = cfg_dot(ins, :bbcode)
        @test occursin("digraph", dot_bb)
        @test occursin("ID=", dot_bb)

        # Non-existent stage returns empty string
        @test cfg_dot(ins, :nonexistent) == ""
    end

    @testset "world_age_info" begin
        ins = inspect_ir(sin, 1.0)
        report = world_age_info(ins)
        @test report isa WorldAgeReport
        @test report.inspection_world isa UInt
        @test report.inspection_world > 0
        @test length(report.stage_worlds) == length(ins.stages)
    end

    @testset "show_world_info" begin
        ins = inspect_ir(sin, 1.0)
        io = IOBuffer()
        show_world_info(ins; io)
        output = String(take!(io))
        @test occursin("World Age Report", output)
        @test occursin("Inspection world", output)
    end

    @testset "write_ir" begin
        ins = inspect_fwd(sin, 1.0)
        tmpdir = mktempdir()
        write_ir(ins, tmpdir)
        files = readdir(tmpdir)

        # Should have stage files, diff files, and CFG files
        @test length(files) == length(ins.stages) + length(ins.diffs) + length(ins.cfgs)

        # Check stage files exist
        for s in keys(ins.stages)
            @test "$(s).txt" in files
        end
        # Check CFG files exist
        for s in keys(ins.cfgs)
            @test "cfg_$(s).dot" in files
        end
        # Check diff files exist
        for (from, to) in keys(ins.diffs)
            @test "diff_$(from)_$(to).txt" in files
        end
    end

    @testset "simple_diff" begin
        d = simple_diff("line1\nline2\nline3", "line1\nchanged\nline3")
        @test occursin("-line2", d)
        @test occursin("+changed", d)
        @test !occursin("-line1", d)
        @test !occursin("-line3", d)

        # Identical strings produce no diff lines
        d_same = simple_diff("abc\ndef", "abc\ndef")
        @test !occursin("-", strip(d_same, ['-', '+', ' ', '\n']))  || occursin("---", d_same)

        # Different lengths
        d_longer = simple_diff("a", "a\nb")
        @test occursin("+b", d_longer)
    end

    @testset "render_ir" begin
        ins = inspect_ir(sin, 1.0)
        # IRCode rendering
        @test !isempty(render_ir(ins.stages[:raw].ir))
        # BBCode rendering
        @test !isempty(render_ir(ins.stages[:bbcode].ir))
        @test occursin("Block", render_ir(ins.stages[:bbcode].ir))
    end

    @testset "convenience functions" begin
        ins_rvs = inspect_rvs(sin, 1.0)
        @test ins_rvs.mode == :reverse

        ins_fwd = inspect_fwd(sin, 1.0)
        @test ins_fwd.mode == :forward

        # quick_inspect returns an IRInspection and also prints
        ins = quick_inspect(sin, 1.0; mode=:forward, stages=:raw)
        @test ins isa IRInspection
        @test ins.mode == :forward
    end

    @testset "stage graph structure" begin
        # Forward mode graph
        fg = forward_stage_graph()
        @test fg == [:raw => :normalized, :normalized => :bbcode, :bbcode => :dual_ir, :dual_ir => :optimized]

        # Reverse mode graph
        rg = reverse_stage_graph()
        @test (:raw => :normalized) in rg
        @test (:bbcode => :fwd_ir) in rg
        @test (:bbcode => :rvs_ir) in rg
        @test (:fwd_ir => :optimized_fwd) in rg
        @test (:rvs_ir => :optimized_rvs) in rg

        # Stage orders
        @test forward_stage_order() == [:raw, :normalized, :bbcode, :dual_ir, :optimized]
        @test reverse_stage_order() == [:raw, :normalized, :bbcode, :fwd_ir, :rvs_ir, :optimized_fwd, :optimized_rvs]
    end

    @testset "StageMeta" begin
        meta = StageMeta()
        @test meta.block_count == 0
        @test meta.inst_count == 0
        @test meta.edge_count == 0
        @test meta.has_ssa == true
        @test meta.uses_bbcode_ids == false
        @test meta.valid_worlds === nothing
        @test meta.misty_world === nothing

        meta2 = StageMeta(; block_count=5, inst_count=10, uses_bbcode_ids=true)
        @test meta2.block_count == 5
        @test meta2.inst_count == 10
        @test meta2.uses_bbcode_ids == true
    end

    @testset "extract_meta" begin
        ins = inspect_ir(sin, 1.0)

        # IRCode meta
        raw_meta = extract_meta(ins.stages[:raw].ir)
        @test raw_meta.block_count > 0
        @test raw_meta.inst_count > 0
        @test raw_meta.edge_count > 0
        @test raw_meta.has_ssa == true
        @test raw_meta.uses_bbcode_ids == false

        # BBCode meta
        bb_meta = extract_meta(ins.stages[:bbcode].ir)
        @test bb_meta.block_count > 0
        @test bb_meta.inst_count > 0
        @test bb_meta.has_ssa == false
        @test bb_meta.uses_bbcode_ids == true

        # Fallback meta
        fallback = extract_meta("not an IR")
        @test fallback.block_count == 0
    end

    @testset "custom function reverse mode" begin
        ins = inspect_ir(test_fn, 1.0)
        @test ins.mode == :reverse
        @test length(ins.stages) == 7
        @test all(s -> ins.stages[s].meta.inst_count > 0, ins.stage_order)
    end

    @testset "custom function forward mode" begin
        ins = inspect_fwd(test_fn, 1.0)
        @test ins.mode == :forward
        @test length(ins.stages) == 5
        @test all(s -> ins.stages[s].meta.inst_count > 0, ins.stage_order)
    end
end
