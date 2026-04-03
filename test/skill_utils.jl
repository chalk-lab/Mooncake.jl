using Mooncake.SkillUtils:
    inspect_ir,
    inspect_fwd,
    inspect_rvs,
    quick_inspect,
    show_ir,
    show_stage,
    show_diff,
    show_all_diffs,
    diff_ir,
    write_ir,
    world_age_info,
    show_world_info,
    forward_stage_graph,
    reverse_stage_graph,
    forward_stage_order,
    reverse_stage_order,
    render_ir,
    extract_meta,
    simple_diff,
    StageMeta,
    IRStage,
    IRInspection,
    WorldAgeReport

test_fn(x) = sin(x) * cos(x)
multi_arg_fn(x, y) = x * y + sin(x)
function bar_llvmcall(x)
    Base.llvmcall((
        """
    declare i64 @llvm.abs.i64(i64, i1)
    define i64 @entry(i64) {
    %x = call i64 @llvm.abs.i64(i64 %0, i1 0)
    ret i64 %x
    }
        """,
        "entry",
    ), Int64, Tuple{Int64}, x)
end
zero_derivative_llvmcall(x) = bar_llvmcall(x)
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(zero_derivative_llvmcall),Int}

@testset "ir_inspect" begin
    @testset "inspect_ir reverse mode" begin
        ins = inspect_ir(test_fn, 1.0)
        @test ins.mode == :reverse
        @test ins.sig == Tuple{typeof(test_fn),Float64}
        @test ins.world isa UInt
        @test isempty(ins.notes)

        expected_stages = [
            :raw, :normalized, :bbcode, :fwd_ir, :rvs_ir, :optimized_fwd, :optimized_rvs
        ]
        @test ins.stage_order == expected_stages
        for s in expected_stages
            @test haskey(ins.stages, s)
        end

        for s in expected_stages
            stage = ins.stages[s]
            @test stage.name == s
            @test stage.meta.block_count > 0
            @test stage.meta.inst_count > 0
            @test !isempty(stage.text)
        end

        @test length(ins.diffs) == length(ins.stage_graph)
        for (from, to) in ins.stage_graph
            @test haskey(ins.diffs, from => to)
        end
    end

    @testset "inspect_ir forward mode" begin
        ins = inspect_fwd(test_fn, 1.0)
        @test ins.mode == :forward

        expected_stages = [:raw, :normalized, :bbcode, :dual_ir, :optimized]
        @test ins.stage_order == expected_stages
        for s in expected_stages
            @test haskey(ins.stages, s)
            @test ins.stages[s].meta.block_count > 0
        end
    end

    @testset "inspect_ir optimize=false" begin
        ins = inspect_ir(test_fn, 1.0; optimize=false)
        sig = Tuple{typeof(test_fn),Float64}
        interp_rvs = get_interpreter(ReverseMode)
        dri = Mooncake.generate_ir(interp_rvs, sig; do_inline=false, do_optimize=false)
        @test ins.mode == :reverse
        @test :raw in ins.stage_order
        @test :fwd_ir in ins.stage_order
        @test :rvs_ir in ins.stage_order
        @test !haskey(ins.stages, :optimized_fwd)
        @test !haskey(ins.stages, :optimized_rvs)
        @test render_ir(ins.stages[:raw].ir) ==
            render_ir(Mooncake.primal_ir(interp_rvs, sig; normalize=false))
        @test render_ir(ins.stages[:normalized].ir) ==
            render_ir(Mooncake.primal_ir(interp_rvs, sig))
        @test render_ir(ins.stages[:fwd_ir].ir) == render_ir(dri.fwd_ir)
        @test render_ir(ins.stages[:rvs_ir].ir) == render_ir(dri.rvs_ir)

        ins_fwd = inspect_fwd(test_fn, 1.0; optimize=false)
        interp_fwd = get_interpreter(ForwardMode)
        dual_ir, _, _ = Mooncake.generate_dual_ir(
            interp_fwd, sig; do_inline=false, do_optimize=false
        )
        @test :dual_ir in ins_fwd.stage_order
        @test !haskey(ins_fwd.stages, :optimized)
        @test render_ir(ins_fwd.stages[:raw].ir) ==
            render_ir(Mooncake.primal_ir(interp_fwd, sig; normalize=false))
        @test render_ir(ins_fwd.stages[:normalized].ir) ==
            render_ir(Mooncake.primal_ir(interp_fwd, sig))
        @test render_ir(ins_fwd.stages[:dual_ir].ir) == render_ir(dual_ir)
    end

    @testset "primitive signatures report dispatch path" begin
        ins = inspect_ir(sin, 1.0)
        @test isempty(ins.stages)
        @test isempty(ins.stage_order)
        @test isempty(ins.stage_graph)
        @test isempty(ins.diffs)
        @test length(ins.notes) == 1
        @test occursin("primitive reverse-mode rule path", only(ins.notes))
        @test occursin("build_primitive_rrule", only(ins.notes))

        ins_fwd = inspect_fwd(sin, 1.0)
        @test isempty(ins_fwd.stages)
        @test isempty(ins_fwd.stage_order)
        @test isempty(ins_fwd.stage_graph)
        @test isempty(ins_fwd.diffs)
        @test length(ins_fwd.notes) == 1
        @test occursin("primitive forward-mode rule path", only(ins_fwd.notes))
        @test occursin("build_primitive_frule", only(ins_fwd.notes))

        llvm_ins = inspect_ir(zero_derivative_llvmcall, 1)
        @test isempty(llvm_ins.stages)
        @test occursin("build_primitive_rrule", only(llvm_ins.notes))
    end

    @testset "inspect_ir multi-arg function" begin
        ins = inspect_ir(multi_arg_fn, 1.0, 2.0)
        @test ins.sig == Tuple{typeof(multi_arg_fn),Float64,Float64}
        @test length(ins.stages) > 0
    end

    @testset "show_ir" begin
        ins = inspect_fwd(test_fn, 1.0)
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
        ins = inspect_ir(test_fn, 1.0)
        io = IOBuffer()
        show_stage(ins, :raw; io)
        output = String(take!(io))
        @test occursin("Stage: raw", output)
        @test !occursin("Stage: normalized", output)
    end

    @testset "diff_ir and show_diff" begin
        ins = inspect_fwd(test_fn, 1.0)

        d = diff_ir(ins; from=:raw, to=:normalized)
        @test d isa String
        @test occursin("---", d)
        @test occursin("+++", d)

        d2 = diff_ir(ins; from=:raw, to=:dual_ir)
        @test d2 isa String

        d3 = diff_ir(ins; from=:nonexistent, to=:raw)
        @test occursin("not found", d3)

        io = IOBuffer()
        show_diff(ins; from=:raw, to=:normalized, io)
        output = String(take!(io))
        @test occursin("Diff:", output)
    end

    @testset "show_all_diffs" begin
        ins = inspect_fwd(test_fn, 1.0)
        io = IOBuffer()
        show_all_diffs(ins; io)
        output = String(take!(io))
        @test !isempty(output)
        for (from, to) in ins.stage_graph
            @test occursin("Diff: $from", output)
        end
    end

    @testset "world_age_info" begin
        ins = inspect_ir(test_fn, 1.0)
        report = world_age_info(ins)
        @test report isa WorldAgeReport
        @test report.inspection_world isa UInt
        @test report.inspection_world > 0
        @test length(report.stage_worlds) == length(ins.stages)
    end

    @testset "show_world_info" begin
        ins = inspect_ir(test_fn, 1.0)
        io = IOBuffer()
        show_world_info(ins; io)
        output = String(take!(io))
        @test occursin("World Age Report", output)
        @test occursin("Inspection world", output)
    end

    @testset "write_ir" begin
        ins = inspect_fwd(test_fn, 1.0)
        tmpdir = mktempdir()
        write_ir(ins, tmpdir)
        files = readdir(tmpdir)

        @test length(files) == length(ins.stages) + length(ins.diffs)

        for s in keys(ins.stages)
            @test "$(s).txt" in files
        end
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

        d_longer = simple_diff("a", "a\nb")
        @test occursin("+b", d_longer)
    end

    @testset "render_ir" begin
        ins = inspect_ir(test_fn, 1.0)
        @test !isempty(render_ir(ins.stages[:raw].ir))
        @test !isempty(render_ir(ins.stages[:bbcode].ir))
        @test occursin("Block", render_ir(ins.stages[:bbcode].ir))
    end

    @testset "convenience functions" begin
        ins_rvs = inspect_rvs(test_fn, 1.0)
        @test ins_rvs.mode == :reverse

        ins_fwd = inspect_fwd(test_fn, 1.0)
        @test ins_fwd.mode == :forward

        ins = quick_inspect(test_fn, 1.0; mode=:forward, stages=:raw)
        @test ins isa IRInspection
        @test ins.mode == :forward
    end

    @testset "inspection failures propagate" begin
        @test_throws Exception inspect_ir(bar_llvmcall, 1)
        @test_throws Exception quick_inspect(bar_llvmcall, 1)
    end

    @testset "stage graph structure" begin
        fg = forward_stage_graph()
        @test fg == [
            :raw => :normalized,
            :normalized => :bbcode,
            :bbcode => :dual_ir,
            :dual_ir => :optimized,
        ]

        rg = reverse_stage_graph()
        @test (:raw => :normalized) in rg
        @test (:bbcode => :fwd_ir) in rg
        @test (:bbcode => :rvs_ir) in rg
        @test (:fwd_ir => :optimized_fwd) in rg
        @test (:rvs_ir => :optimized_rvs) in rg

        @test forward_stage_order() == [:raw, :normalized, :bbcode, :dual_ir, :optimized]
        @test reverse_stage_order() ==
            [:raw, :normalized, :bbcode, :fwd_ir, :rvs_ir, :optimized_fwd, :optimized_rvs]
    end

    @testset "StageMeta" begin
        meta = StageMeta()
        @test meta.block_count == 0
        @test meta.inst_count == 0
        @test meta.edge_count == 0
        @test meta.valid_worlds === nothing

        meta2 = StageMeta(; block_count=5, inst_count=10)
        @test meta2.block_count == 5
        @test meta2.inst_count == 10
    end

    @testset "extract_meta" begin
        ins = inspect_ir(test_fn, 1.0)

        raw_meta = extract_meta(ins.stages[:raw].ir)
        @test raw_meta.block_count > 0
        @test raw_meta.inst_count > 0
        @test raw_meta.edge_count > 0

        bb_meta = extract_meta(ins.stages[:bbcode].ir)
        @test bb_meta.block_count > 0
        @test bb_meta.inst_count > 0

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
