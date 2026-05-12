@testset "Compiler services" begin
    interp = Mooncake.MooncakeInterpreter(ForwardMode)
    reverse_interp = Mooncake.MooncakeInterpreter(DefaultCtx, ReverseMode)

    # TODO(boundary-cleanup): the trampolines exercised below
    # (`interpreter_world`, `inference_parameters`, `optimization_parameters`,
    # `inference_cache`, `code_cache_view`, `overlay_method_table`,
    # `cache_owner`) are pure passthroughs over `MooncakeInterpreter` fields.
    # When they're inlined into the `CC.*` overloads in
    # `src/interpreter/abstract_interpretation.jl`, these assertions should be
    # replaced with end-to-end inference probes.
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

    @testset "compiler boundary static gate" begin
        repo_root = normpath(joinpath(@__DIR__, "..", ".."))
        src_root = joinpath(repo_root, "src")
        compiler_root = joinpath(src_root, "compiler")

        function path_is_under(path::AbstractString, root::AbstractString)
            rel = relpath(path, root)
            return rel == "." || first(splitpath(rel)) != ".."
        end

        # Compiler-internal names that have either moved between Julia versions,
        # that ccall directly into Julia's C runtime, or that touch unstable
        # binding/inference internals. Their use must stay localised under
        # `src/compiler/`; any other site needs a documented exception in
        # `named_exceptions` below.
        compiler_internal_names = (
            "typeinf_ircode",
            "IRInterpretationState",
            "adce_pass!",
            "ssa_inlining_pass!",
            "sroa_pass!",
            "scan_leaf_partitions",
            "compute_ir_rettype",
            "compute_oc_signature",
            "jl_new_code_info_uninit",
            "jl_new_method_instance_uninit",
            "generate_opaque_closure",
        )
        named_exceptions = Dict(
            "IRInterpretationState" => Set([
                normpath(joinpath(src_root, "interpreter", "patch_for_319.jl")),
            ]),
            # `Mooncake.compute_ir_rettype` and `Mooncake.compute_oc_signature`
            # are legacy public names retained as compatibility shims; their
            # bodies delegate to the corresponding `Compiler.*` services.
            "compute_ir_rettype" => Set([normpath(joinpath(src_root, "utils.jl"))]),
            "compute_oc_signature" => Set([normpath(joinpath(src_root, "utils.jl"))]),
        )

        violations = String[]
        for (dir, _, filenames) in walkdir(src_root)
            for filename in filenames
                endswith(filename, ".jl") || continue
                path = normpath(joinpath(dir, filename))
                path_is_under(path, compiler_root) && continue
                for (line_number, line) in enumerate(eachline(path))
                    for name in compiler_internal_names
                        allowed_paths = get(named_exceptions, name, Set{String}())
                        if occursin(name, line) && !(path in allowed_paths)
                            push!(
                                violations,
                                "$(relpath(path, repo_root)):$line_number contains `$name`",
                            )
                        end
                    end
                end
            end
        end

        isempty(violations) || @info "Compiler boundary violations" violations
        @test isempty(violations)
    end
end
