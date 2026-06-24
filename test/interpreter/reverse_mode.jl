module S2SGlobals
using LinearAlgebra, Mooncake

non_const_global = 5.0
const const_float = 5.0
const const_int = 5
const const_bool = true

# used for regression test for issue 184
struct A
    data
end
f(a, x) = dot(a.data, x)

unstable_tester(x::Ref{Any}) = sin(x[])

# used for regression test for issue 660
struct MakeAUnionAll{T} end

end

# Regression for non-returning primals: `rule_type` must predict `Tuple{}` pullback args
# to match `build_rrule`, otherwise lazy derived-rule materialization hits a type mismatch
# (PR #1099).
rule_type_nonreturning(e::Exception) = throw(e)

# Helpers for the world-advance rule-staleness regression test (reverse mode); see the
# forward-mode analogue for the scope note. `stale_rvs_lazy` reaches the callee statically
# (LazyDerivedRule), `stale_rvs_dyn` dynamically (DynamicDerivedRule).
stale_rvs_inner(x) = Float32(x) * 2.0f0
@noinline stale_rvs_callee(x) = stale_rvs_inner(x)
stale_rvs_lazy(x) = stale_rvs_callee(x)
const STALE_RVS_FNS = Function[stale_rvs_callee]
stale_rvs_dyn(x) = (STALE_RVS_FNS[1])(x)

@testset "s2s_reverse_mode_ad" begin
    @testset "SharedDataPairs" begin
        m = SharedDataPairs()
        id = Mooncake.add_data!(m, 5.0)
        @test length(m.pairs) == 1
        @test m.pairs[1][1] == id
        @test m.pairs[1][2] == 5.0
    end
    @testset "ADInfo" begin
        GlobalsTest = Module(:GlobalsTest, true)
        Core.eval(GlobalsTest, quote
            ___x = 5.0
            ___y::Float64 = 5.0
            @static if VERSION > v"1.12-"
                const ___constx = 5.0
                const ___consty::Float64 = 5.0
            end
        end)

        arg_types = Dict{Argument,Any}(Argument(1) => Float64, Argument(2) => Int)
        id_ssa_1 = ID()
        id_ssa_2 = ID()
        ssa_insts = Dict{ID,CC.NewInstruction}(
            id_ssa_1 => CC.NewInstruction(nothing, Float64),
            id_ssa_2 => CC.NewInstruction(nothing, Any),
        )
        is_used_dict = Dict{ID,Bool}(id_ssa_1 => true, id_ssa_2 => true)
        rdata_ref = Ref{Tuple{map(Mooncake.lazy_zero_rdata_type, (Float64, Int))...}}()
        info = ADInfo(
            get_interpreter(ReverseMode),
            arg_types,
            ssa_insts,
            is_used_dict,
            false,
            rdata_ref,
            Any,
            Any,
        )

        # Verify that we can access the interpreter and terminator block ID.
        @test info.interp isa Mooncake.MooncakeInterpreter

        # Verify that we can get the type associated to Arguments, IDs, and others.
        @test Mooncake.get_primal_type(info, Argument(1)) == Float64
        @test Mooncake.get_primal_type(info, Argument(2)) == Int
        @test Mooncake.get_primal_type(info, id_ssa_1) == Float64
        @test Mooncake.get_primal_type(info, GlobalRef(Base, :sin)) == typeof(sin)
        @test Mooncake.get_primal_type(info, GlobalRef(GlobalsTest, :___x)) == Any
        @test Mooncake.get_primal_type(info, GlobalRef(GlobalsTest, :___y)) == Float64
        @test Mooncake.get_primal_type(info, GlobalRef(Base, :Float64)) == Type{Float64}
        # PARTITION_KIND_IMPLICIT_CONST
        @test Mooncake.get_primal_type(info, GlobalRef(Main, :Float64)) == Type{Float64}
        @test Mooncake.get_primal_type(info, GlobalRef(Base, :stdin)) == IO
        # PARTITION_KIND_IMPLICIT_GLOBAL
        @test Mooncake.get_primal_type(info, GlobalRef(Main, :stdin)) == IO
        @static if VERSION > v"1.12-"
            @test Mooncake.get_primal_type(info, GlobalRef(GlobalsTest, :___constx)) ==
                Float64
            @test Mooncake.get_primal_type(info, GlobalRef(GlobalsTest, :___consty)) ==
                Float64

            # Rebind globals using different types, this will increase the world age
            Core.eval(GlobalsTest, quote
                ___x = 5.0f0
                # Note that we cannot change the type of ___y nor make it constant
                const ___constx = 5.0f0
                const ___consty::Float32 = 5.0f0
            end)

            # In the info's world age, this should not affect the types
            # We have to use invokelatest otherwise the call to get_primal_type will run in the old world age
            @test invokelatest(
                Mooncake.get_primal_type, info, GlobalRef(GlobalsTest, :___x)
            ) == Any
            @test invokelatest(
                Mooncake.get_primal_type, info, GlobalRef(GlobalsTest, :___constx)
            ) == Float64
            @test invokelatest(
                Mooncake.get_primal_type, info, GlobalRef(GlobalsTest, :___consty)
            ) == Float64

            # But now we can create a new info in the new world age,
            # where these updated bindings should be visible
            info2 = ADInfo(
                get_interpreter(ReverseMode),
                arg_types,
                ssa_insts,
                is_used_dict,
                false,
                rdata_ref,
                Any,
                Any,
            )
            @test invokelatest(
                Mooncake.get_primal_type, info2, GlobalRef(GlobalsTest, :___x)
            ) == Any
            @test invokelatest(
                Mooncake.get_primal_type, info2, GlobalRef(GlobalsTest, :___constx)
            ) == Float32
            @test invokelatest(
                Mooncake.get_primal_type, info2, GlobalRef(GlobalsTest, :___consty)
            ) == Float32
        end
        @test Mooncake.get_primal_type(info, 5) == Int
        @test Mooncake.get_primal_type(info, QuoteNode(:hello)) == Symbol
        @test Mooncake.get_primal_type(info, Expr(:boundscheck)) == Bool
        @test_throws ErrorException Mooncake.get_primal_type(info, Expr(:call))
    end
    @testset "ADStmtInfo" begin
        # If the ID passes as the comms channel doesn't appear in the stmts for the forwards
        # pass, then this constructor ought to error.
        @test_throws ArgumentError ad_stmt_info(ID(), ID(), nothing, nothing)
    end
    @testset "inc_args" begin
        @test Mooncake.inc_args(Expr(:call, sin, Argument(4))) ==
            Expr(:call, sin, Argument(5))
        @test Mooncake.inc_args(ReturnNode(Argument(2))) == ReturnNode(Argument(3))
        id = ID()
        @test Mooncake.inc_args(IDGotoIfNot(Argument(1), id)) ==
            IDGotoIfNot(Argument(2), id)
        @test Mooncake.inc_args(IDGotoNode(id)) == IDGotoNode(id)
        ids = [id, ID()]
        @test ==(
            Mooncake.inc_args(IDPhiNode(ids, Any[Argument(1), 4])),
            IDPhiNode(ids, Any[Argument(2), 4]),
        )
        @test Mooncake.inc_args(nothing) === nothing
        @test Mooncake.inc_args(GlobalRef(Base, :sin)) == GlobalRef(Base, :sin)
    end
    @testset "make_ad_stmts!" begin

        # Set up ADInfo -- this state is required by `make_ad_stmts!`, and the
        # `LineToADDataMap` object can be mutated.
        id_line_1 = ID()
        id_line_2 = ID()
        info = ADInfo(
            get_interpreter(ReverseMode),
            Dict{Argument,Any}(Argument(1) => typeof(sin), Argument(2) => Float64),
            Dict{ID,CC.NewInstruction}(
                id_line_1 => new_inst(Expr(:invoke, nothing, cos, Argument(2)), Float64),
                id_line_2 => new_inst(nothing, Any),
            ),
            Dict{ID,Bool}(id_line_1 => true, id_line_2 => true),
            false,
            Ref{Tuple{map(Mooncake.lazy_zero_rdata_type, (typeof(sin), Float64))...}}(),
            Any,
            Any,
        )

        @testset "Nothing" begin
            line = ID()
            @test TestUtils.has_equal_data(
                make_ad_stmts!(nothing, line, info),
                ad_stmt_info(line, nothing, nothing, nothing),
            )
        end
        @testset "ReturnNode" begin
            line = ID()
            @testset "unreachable" begin
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(ReturnNode(), line, info),
                    ad_stmt_info(line, nothing, ReturnNode(), nothing),
                )
            end
            @testset "Argument" begin
                val = Argument(4)
                stmts = make_ad_stmts!(ReturnNode(Argument(2)), line, info)
                @test length(stmts.fwds) == 2
                @test stmts.fwds[1][2].stmt isa Expr
                @test stmts.fwds[2][2].stmt isa ReturnNode
            end
            @testset "literal" begin
                stmt_info = make_ad_stmts!(ReturnNode(5.0), line, info)
                @test length(stmt_info.fwds) == 3
                @test stmt_info isa ADStmtInfo
                @test stmt_info.fwds[3][2].stmt isa ReturnNode
            end
            @testset "GlobalRef" begin
                node = ReturnNode(GlobalRef(S2SGlobals, :const_float))
                stmt_info = make_ad_stmts!(node, line, info)
                @test length(stmt_info.fwds) == 3
                @test stmt_info isa ADStmtInfo
                @test stmt_info.fwds[3][2].stmt isa ReturnNode
            end
        end
        @testset "IDGotoNode" begin
            line = ID()
            stmt = IDGotoNode(ID())
            @test TestUtils.has_equal_data(
                make_ad_stmts!(stmt, line, info), ad_stmt_info(line, nothing, stmt, nothing)
            )
        end
        @testset "IDGotoIfNot" begin
            line = ID()
            cond_id = ID()
            stmt = IDGotoIfNot(cond_id, ID())
            ad_stmts = make_ad_stmts!(stmt, line, info)
            @test ad_stmts isa ADStmtInfo
            @test ad_stmts.rvs[1][2].stmt === nothing
            fwds = ad_stmts.fwds
            @test fwds[1][1] == fwds[2][2].stmt.cond
            @test Meta.isexpr(fwds[1][2].stmt, :call)
            @test fwds[2][2].stmt isa IDGotoIfNot
            @test fwds[2][2].stmt.dest == stmt.dest
        end
        @testset "IDPhiNode" begin
            stmt = IDPhiNode(ID[ID(), ID()], Any[ID(), 5.0])
            ad_stmts = make_ad_stmts!(stmt, id_line_1, info)
            @test ad_stmts isa ADStmtInfo
        end
        @testset "PiNode" begin
            @testset "π (nothing, Union{})" begin
                # This is a weird edge case that appeared in 1.11. See comment in src.
                line = id_line_1
                stmt_info = make_ad_stmts!(PiNode(nothing, Union{}), line, info)
                @test stmt_info isa ADStmtInfo
                @test last(stmt_info.fwds)[1] == line
            end
            @testset "π (nothing, Nothing)" begin
                stmt_info = make_ad_stmts!(PiNode(nothing, Nothing), id_line_1, info)
                @test stmt_info isa ADStmtInfo
                @test last(stmt_info.fwds)[1] == id_line_1
                fwds_stmt = last(stmt_info.fwds)[2].stmt
                @test fwds_stmt isa PiNode
                @test fwds_stmt.typ == CoDual{Nothing,NoFData}
                @test only(stmt_info.rvs)[2].stmt === nothing
            end
            @testset "π (nothing, CC.Const(nothing))" begin
                node = PiNode(nothing, CC.Const(nothing))
                stmt_info = make_ad_stmts!(node, id_line_1, info)
                @test stmt_info isa ADStmtInfo
                @test last(stmt_info.fwds)[1] == id_line_1
                fwds_stmt = last(stmt_info.fwds)[2].stmt
                @test fwds_stmt isa PiNode
                @test fwds_stmt.typ == CoDual{Nothing,NoFData}
                @test only(stmt_info.rvs)[2].stmt === nothing
            end
            @testset "π (GlobalRef, Type)" begin
                node = PiNode(GlobalRef(S2SGlobals, :const_float), Any)
                stmt_info = make_ad_stmts!(node, id_line_1, info)
                @test stmt_info isa ADStmtInfo
                fwds_stmt = last(stmt_info.fwds)[2].stmt
                @test fwds_stmt isa PiNode
                @test fwds_stmt.typ == CoDual
                @test only(stmt_info.rvs)[2].stmt === nothing
            end
            @testset "sharpen type of ID" begin
                line = id_line_1
                val = id_line_2
                stmt_info = make_ad_stmts!(PiNode(val, Float64), line, info)
                @test stmt_info isa ADStmtInfo
            end
        end
        @testset "GlobalRef" begin
            @testset "non-const" begin
                global_ref = GlobalRef(S2SGlobals, :non_const_global)
                stmt_info = make_ad_stmts!(global_ref, ID(), info)
                @test Mooncake.TestResources.non_const_global_ref(5.0) == 5.0 # run primal
                @test stmt_info isa Mooncake.ADStmtInfo
                @test Meta.isexpr(last(stmt_info.fwds)[2].stmt, :call)
                @test last(stmt_info.fwds)[2].stmt.args[1] == Mooncake.__verify_const
            end
            @testset "differentiable const globals" begin
                stmt_info = make_ad_stmts!(GlobalRef(S2SGlobals, :const_float), ID(), info)
                @test stmt_info isa Mooncake.ADStmtInfo
                @test only(stmt_info.fwds)[2].stmt isa Expr
                @test only(stmt_info.fwds)[2].stmt.args[1] === Mooncake.uninit_fcodual
            end
        end
        @testset "PhiCNode" begin
            @test_throws(
                Mooncake.UnhandledLanguageFeatureException,
                make_ad_stmts!(Core.PhiCNode(Any[]), ID(), info),
            )
        end
        @testset "UpsilonNode" begin
            @test_throws(
                Mooncake.UnhandledLanguageFeatureException,
                make_ad_stmts!(Core.UpsilonNode(5), ID(), info),
            )
        end
        @testset "Expr" begin
            @testset "assignment to GlobalRef" begin
                @test_throws(
                    Mooncake.UnhandledLanguageFeatureException,
                    make_ad_stmts!(Expr(:(=), GlobalRef(Main, :a), 5.0), ID(), info)
                )
            end
            @testset "copyast" begin
                stmt = Expr(:copyast, QuoteNode(:(hi)))
                ad_stmts = make_ad_stmts!(stmt, ID(), info)
                @test ad_stmts isa Mooncake.ADStmtInfo
                @test Meta.isexpr(ad_stmts.fwds[1][2].stmt, :call)
                @test ad_stmts.fwds[1][2].stmt.args[1] == identity
            end
            @testset "throw_undef_if_not" begin
                cond_id = ID()
                line = ID()
                fwds = Expr(:throw_undef_if_not, :x, cond_id)
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(Expr(:throw_undef_if_not, :x, cond_id), line, info),
                    ad_stmt_info(line, nothing, fwds, nothing),
                )
            end
            @testset "$stmt" for stmt in [Expr(:gc_preserve_begin)]
                line = ID()
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(stmt, line, info),
                    ad_stmt_info(line, nothing, stmt, nothing),
                )
            end
        end
    end
    @testset "rule_type $sig, $debug_mode" for sig in Any[
            Tuple{typeof(getfield),Tuple{Float64},1},
            Tuple{typeof(TestResources.foo),Float64},
            Tuple{typeof(TestResources.type_unstable_tester_0),Ref{Any}},
            Tuple{typeof(TestResources.tuple_with_union),Bool},
            Tuple{typeof(TestResources.tuple_with_union_2),Bool},
            Tuple{typeof(TestResources.tuple_with_union_3),Bool,Bool},
            Tuple{typeof(rule_type_nonreturning),ArgumentError},
        ],
        debug_mode in [true, false]

        interp = get_interpreter(ReverseMode)
        rule = Mooncake.build_rrule(interp, sig; debug_mode)
        @test rule isa Mooncake.rule_type(interp, sig; debug_mode)
    end
    @testset "MooncakeRuleCompilationError" begin
        @test_throws(Mooncake.MooncakeRuleCompilationError, Mooncake.build_rrule(sin))
        _trycatch_fn(x::Float64) =
            try
                ;
                return log(x);
            catch
                ;
                return 0.0;
            end
        @test_throws(
            Mooncake.MooncakeRuleCompilationError, Mooncake.build_rrule(_trycatch_fn, 1.0)
        )
        # showerror should include the originating method's source location (issue #649)
        function _rrule_error_test_llvmcall(x::Int64)
            Base.llvmcall(
                (
                    """
                declare i64 @llvm.abs.i64(i64, i1)
                define i64 @entry(i64) {
                %x = call i64 @llvm.abs.i64(i64 %0, i1 0)
                ret i64 %x
                }
                    """,
                    "entry",
                ), Int64, Tuple{Int64}, x
            )
        end
        err = try
            Mooncake.build_rrule(Tuple{typeof(_rrule_error_test_llvmcall),Int64})
            nothing
        catch e
            e
        end
        @test err isa Mooncake.MooncakeRuleCompilationError
        msg = sprint(showerror, err; context=:displaysize => (24, 120))
        @test startswith(msg, "Mooncake failed to differentiate the following method:")
        @test contains(msg, "_rrule_error_test_llvmcall")
        @test contains(msg, "Caused by:")
    end
    @testset "$(_typeof((f, x...)))" for (n, (interface_only, perf_flag, bnds, f, x...)) in
                                         collect(
        enumerate(TestResources.generate_test_functions())
    )
        sig = _typeof((f, x...))
        @info "$n: $sig"
        mode = ReverseMode
        TestUtils.test_rule(
            Xoshiro(123456), f, x...; perf_flag, interface_only, is_primitive=false, mode
        )
        # TestUtils.test_rule(
        #     Xoshiro(123456),
        #     f,
        #     x...;
        #     perf_flag=:none,
        #     interface_only,
        #     is_primitive=false,
        #     debug_mode=true,
        # )

        # interp = Mooncake.get_interpreter(ReverseMode)
        # codual_args = map(zero_codual, (f, x...))
        # fwds_args = map(Mooncake.to_fwds, codual_args)
        # rule = Mooncake.build_rrule(interp, sig)
        # out, pb!! = rule(fwds_args...)
        # # @code_warntype optimize=true rule(codual_args...)
        # # @code_warntype optimize=true pb!!(tangent(out), map(tangent, codual_args)...)

        # primal_time = @benchmark $f($(Ref(x))[]...)
        # s2s_time = @benchmark $rule($fwds_args...)[2]($(Mooncake.zero_rdata(primal(out))))

        # display(primal_time)
        # display(s2s_time)
        # s2s_ratio = time(s2s_time) / time(primal_time)
        # println("s2s ratio ratio: $(s2s_ratio)")

        # f(rule, fwds_args, out) = rule(fwds_args...)[2]((Mooncake.zero_rdata(primal(out))))
        # f(rule, fwds_args, out)
        # @profview(run_many_times(500, f, rule, fwds_args, out))
    end

    @testset "integration testing for invalid global ref errors" begin
        @static if VERSION > v"1.12-"
            @test_throws(
                Mooncake.MooncakeRuleCompilationError,
                Mooncake.build_rrule(
                    Tuple{typeof(Mooncake.TestResources.non_const_global_ref),Float64}
                )
            )
        end
    end

    # Tests designed to prevent accidentally re-introducing issues which we have fixed.
    @testset "regression tests" begin

        # 184
        TestUtils.test_rule(
            Xoshiro(123456),
            S2SGlobals.f,
            S2SGlobals.A(2 * ones(3)),
            ones(3);
            interface_only=false,
            is_primitive=false,
            mode=Mooncake.ReverseMode,
        )

        # BenchmarkTools not working due to world age problems. Provided that this code
        # runs successfully, everything is okay -- no need to check anything specific.
        f(x) = sin(cos(x))
        rule = Mooncake.build_rrule(f, 0.0)
        @benchmark Mooncake.value_and_gradient!!($rule, $f, $(Ref(0.0))[])

        # 660 -- ensure that the correct signature is used to construct DynamicDerivedRules
        rule = Mooncake.DynamicDerivedRule(false)
        args = (zero_fcodual(identity), zero_fcodual((v=S2SGlobals.MakeAUnionAll,)))
        @test rule(args...) isa Tuple{CoDual,Any}
    end
    @testset "_pullback_type" begin
        # On Julia 1.10, keyword functions are lowered to old-style `##foo#N` wrappers
        # rather than `Core.kwcall`. When Mooncake compiles a derived rrule for such a
        # wrapper, `pullback_type` calls `Core.Compiler.return_type` on the inner `rrule!!`
        # call. If inference gives up — e.g. because the `@from_rrule` wrapper calls
        # `ChainRulesCore.rrule` and the pullback closure type cannot be resolved — it
        # returns bare `Tuple`, whose `parameters` field is `svec(Vararg{Any})` (length 1).
        # `_pullback_type` must not index past the end of that vector.
        @test Mooncake._pullback_type(Tuple) === Any        # svec(Vararg{Any}), length 1
        @test Mooncake._pullback_type(Tuple{}) === Any      # svec(), length 0
        @test Mooncake._pullback_type(Tuple{Int}) === Any   # svec(Int), length 1
        # Sanity-check: well-formed 2-element tuple still extracts the second parameter.
        @test Mooncake._pullback_type(Tuple{Int,Float64}) === Float64
    end
    @testset "literal Strings do not appear in shared data" begin
        f() = "hello"
        @test length(build_rrule(Tuple{typeof(f)}).fwds_oc.oc.captures) == 2
    end
    @testset "Literal Types do not appear in shared data" begin
        f() = Float64
        @test length(build_rrule(Tuple{typeof(f)}).fwds_oc.oc.captures) == 2
    end
    @testset "all `Ref`s for rdata are eliminated in type unstable code" begin
        ir = Mooncake.rvs_ir(Tuple{typeof(S2SGlobals.unstable_tester),Ref{Any}})
        stmts = Mooncake.stmt(ir.stmts)
        @test !any(x -> Meta.isexpr(x, :new) && x.args[1] <: Base.RefValue, stmts)
    end
    @testset "build_rrule methods all accept kwargs" begin
        args = (sin, 5.0)
        sig = typeof(args)
        rule_sig = build_rrule(sig; debug_mode=false, silence_debug_messages=true)
        @test rule_sig == rrule!!
        rule_args = build_rrule(args...; debug_mode=false, silence_debug_messages=true)
        @test rule_args == rrule!!
        rule_debug_sig = build_rrule(sig; debug_mode=true, silence_debug_messages=true)
        @test rule_debug_sig isa Mooncake.DebugRRule
        rule_debug_args = build_rrule(args...; debug_mode=true, silence_debug_messages=true)
        @test rule_debug_args == rule_debug_sig
    end

    # Without the fix the lazy path throws a `convert` MethodError in _build_rule! after the
    # world advance; both lazy and dynamic must return the build-world result (Float32), not
    # the post-advance world's (Float64).
    @testset "stale rule build-world after world advance (issue #1218)" begin
        lazy = Mooncake.build_rrule(stale_rvs_lazy, 1.5)
        dyn = Mooncake.build_rrule(stale_rvs_dyn, 1.5)
        @eval stale_rvs_inner(x::Float64) = x * 2.0  # advance world; tightens callee's type
        lazy_y, _ = Base.invokelatest(
            lazy, Mooncake.zero_fcodual(stale_rvs_lazy), Mooncake.zero_fcodual(1.5)
        )
        dyn_y, _ = Base.invokelatest(
            dyn, Mooncake.zero_fcodual(stale_rvs_dyn), Mooncake.zero_fcodual(1.5)
        )
        @test Mooncake.primal(lazy_y) === 3.0f0
        @test Mooncake.primal(dyn_y) === 3.0f0
    end
end

# --- Working-IR layer (CFGBlock etc.) ---
# The working IR lives in src/interpreter/reverse_mode.jl (shared primitives in
# ir_utils.jl), so its tests live here too (merged from the former bbcode.jl).
module CFGBlocksTestCases
test_phi_node(x::Ref{Union{Float32,Float64}}) = sin(x[])
end

@testset "cfg_blocks" begin
    @testset "ID" begin
        id1 = ID()
        id2 = ID()
        @test id1 == id1
        @test id1 != id2
    end
    @testset "CFGBlock" begin
        bb = CFGBlock(
            ID(),
            ID[ID(), ID()],
            CC.NewInstruction[
                CC.NewInstruction(IDPhiNode([ID(), ID()], Any[true, false]), Any),
                CC.NewInstruction(:(println("hello")), Any),
            ],
        )
        @test bb isa CFGBlock
        @test length(bb) == 2

        ids, phi_nodes = Mooncake.phi_nodes(bb)
        @test only(ids) == bb.inst_ids[1]
        @test only(phi_nodes) == bb.insts[1]

        bb_copy = copy(bb)
        @test bb_copy.inst_ids !== bb.inst_ids

        @test Mooncake.terminator(bb) === nothing

        # CFGBlock is immutable: its fields cannot be reassigned.
        @test_throws ErrorException (bb.id = ID())

        # The constructor enforces inst_ids and insts having equal length.
        @test_throws AssertionError CFGBlock(ID(), ID[ID()], CC.NewInstruction[])

        # terminator returns the final statement when it is a Terminator.
        bb2 = CFGBlock(ID(), Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(5)))])
        @test Mooncake.terminator(bb2) === bb2.insts[end].stmt
    end
    @testset "insert_before_terminator" begin
        extra = Mooncake.IDInstPair[(ID(), new_inst(nothing))]

        # No terminator: `extra` is appended at the end, and the input is left untouched.
        hello = (ID(), new_inst(:(println("hello"))))
        insts = Mooncake.IDInstPair[hello]
        out = Mooncake.insert_before_terminator(insts, extra)
        @test length(out) == 2
        @test out[end] == extra[1]
        @test insts !== out  # a fresh vector is returned
        @test insts == Mooncake.IDInstPair[hello]  # input not modified (purity contract)

        # Empty `insts`: exercises the `isempty(insts)` short-circuit; `extra` is appended.
        out = Mooncake.insert_before_terminator(Mooncake.IDInstPair[], extra)
        @test out == extra

        # Terminator present: `extra` is spliced in immediately before it.
        ret = (ID(), new_inst(ReturnNode(5)))
        insts = Mooncake.IDInstPair[(ID(), new_inst(:(println("hello")))), ret]
        out = Mooncake.insert_before_terminator(insts, extra)
        @test length(out) == 3
        @test out[end] == ret
        @test out[end - 1] == extra[1]

        # Empty `extra` returns the input untouched.
        @test Mooncake.insert_before_terminator(insts, Mooncake.IDInstPair[]) === insts

        # Block led by a phi node: `extra` splices before the terminator, phi left in place.
        phi = (ID(), new_inst(IDPhiNode([ID()], Any[true])))
        insts = Mooncake.IDInstPair[phi, ret]
        out = Mooncake.insert_before_terminator(insts, extra)
        @test out[1] === phi
        @test out[end] === ret
        @test out[end - 1] == extra[1]
    end
    @testset "round-trip $f" for (f, P) in [
        (TestResources.test_while_loop, Tuple{Float64}),
        (sin, Tuple{Float64}),
        (CFGBlocksTestCases.test_phi_node, Tuple{Ref{Union{Float32,Float64}}}),
    ]
        ir = Base.code_ircode(f, P)[1][1]
        blocks = _ircode_to_cfg_blocks(ir)
        @test blocks isa Vector{CFGBlock}
        @test length(blocks) == length(ir.cfg.blocks)
        new_ir = lower_cfg_blocks_to_ir(blocks, ir)
        @test new_ir.argtypes isa Vector{Any}
        @test new_ir.argtypes == ir.argtypes
        @test length(stmt(new_ir.stmts)) == length(stmt(ir.stmts))
        @test all(map(==, stmt(ir.stmts), stmt(new_ir.stmts)))
        @test all(map(==, ir.stmts.type, new_ir.stmts.type))
        @test all(map(==, ir.stmts.info, new_ir.stmts.info))
        @test all(map(==, ir.stmts.line, new_ir.stmts.line))
        @test all(map(==, ir.stmts.flag, new_ir.stmts.flag))
        @test length(Mooncake.collect_stmts(blocks)) == length(stmt(ir.stmts))
        @test Mooncake.id_to_line_map(blocks) isa Dict{ID,Int}
    end
    @testset "lower_cfg_blocks_to_ir argtypes coercion" begin
        ir = Base.code_ircode(sin, Tuple{Float64})[1][1]
        blocks = _ircode_to_cfg_blocks(ir)
        # A non-`Vector{Any}` argtypes override must be coerced to `Vector{Any}`.
        custom = DataType[typeof(sin), Float64]
        out = lower_cfg_blocks_to_ir(blocks, ir; argtypes=custom)
        @test out.argtypes isa Vector{Any}
        @test out.argtypes == Any[typeof(sin), Float64]
    end
    @static if VERSION > v"1.12-"
        @testset "codelocs consistent after instruction insertion" begin
            # In 1.12+, codelocs (and stmts.line) pack 3 Int32 per instruction, so an
            # n-instruction IRCode has 3n entries that must stay aligned across the round-trip
            # even when instructions are inserted (regression test for Mooncake.jl#1216).
            ir = Base.code_ircode(sin, Tuple{Float64})[1][1]
            n_orig = length(ir.stmts)
            orig_ir_codelocs = copy(ir.debuginfo.codelocs)
            blocks = _ircode_to_cfg_blocks(ir)

            # Insert a recognisable instruction before the first block's terminator.
            blk = blocks[1]
            insert_idx = if isnothing(Mooncake.terminator(blk))
                length(blk.insts) + 1
            else
                length(blk.insts)
            end
            sentinel = (Int32(123), Int32(45), Int32(6))
            inst = CC.NewInstruction(
                nothing, Any, CC.NoCallInfo(), sentinel, CC.IR_FLAG_REFINED
            )
            blocks[1] = CFGBlock(
                blk.id,
                Mooncake.insert_before_terminator(
                    Mooncake.collect_stmts(blk), Mooncake.IDInstPair[(ID(), inst)]
                ),
            )
            new_ir = lower_cfg_blocks_to_ir(blocks, ir)
            n = length(new_ir.stmts)
            insert_range = (3insert_idx - 2):(3insert_idx)

            @test n == n_orig + 1
            @test length(new_ir.stmts.line) == 3n
            @test length(new_ir.debuginfo.codelocs) == 3n
            @test Tuple(new_ir.stmts.line[insert_range]) == sentinel
            # Conversion must not mutate the source `ir`'s debug info.
            @test new_ir.debuginfo.codelocs !== ir.debuginfo.codelocs
            @test ir.debuginfo.codelocs == orig_ir_codelocs
        end
    end
    @testset "control_flow_graph" begin
        ir = Base.code_ircode_by_type(Tuple{typeof(sin),Float64})[1][1]
        blocks = _ircode_to_cfg_blocks(ir)
        cfg = Mooncake.control_flow_graph(blocks)
        @test all(map((l, r) -> l.stmts == r.stmts, ir.cfg.blocks, cfg.blocks))
        @test all(map((l, r) -> sort(l.preds) == sort(r.preds), ir.cfg.blocks, cfg.blocks))
        @test all(map((l, r) -> sort(l.succs) == sort(r.succs), ir.cfg.blocks, cfg.blocks))
        @test ir.cfg.index == cfg.index
    end
    @testset "_characterise_unique_predecessor_blocks" begin
        @testset "single block" begin
            blk_id = ID()
            blks = CFGBlock[CFGBlock(blk_id, [ID()], [new_inst(ReturnNode(5))])]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id] == true
            @test pred_is_upred[blk_id] == true
        end
        @testset "pair of blocks" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blks = CFGBlock[
                CFGBlock(blk_id_1, [ID()], [new_inst(IDGotoNode(blk_id_2))]),
                CFGBlock(blk_id_2, [ID()], [new_inst(ReturnNode(5))]),
            ]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == true
            @test pred_is_upred[blk_id_1] == true
            @test pred_is_upred[blk_id_2] == true
        end
        @testset "Non-Unique Exit Node" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blk_id_3 = ID()
            blks = CFGBlock[
                CFGBlock(blk_id_1, [ID()], [new_inst(IDGotoIfNot(true, blk_id_3))]),
                CFGBlock(blk_id_2, [ID()], [new_inst(ReturnNode(5))]),
                CFGBlock(blk_id_3, [ID()], [new_inst(ReturnNode(5))]),
            ]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
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
            blks = CFGBlock[
                CFGBlock(blk_id_1, [ID()], [new_inst(IDGotoIfNot(true, blk_id_3))]),
                CFGBlock(blk_id_2, [ID()], [new_inst(IDGotoNode(blk_id_4))]),
                CFGBlock(blk_id_3, [ID()], [new_inst(IDGotoNode(blk_id_4))]),
                CFGBlock(blk_id_4, [ID()], [new_inst(ReturnNode(0))]),
            ]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
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
            blks = CFGBlock[
                CFGBlock(blk_id_1, [ID()], [new_inst(IDGotoIfNot(true, blk_id_1))]),
                CFGBlock(blk_id_2, [ID()], [new_inst(ReturnNode(5))]),
            ]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
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
                Mooncake._find_id_uses!(d, Expr(:call, sin, 5))
                @test d[id] == false
                Mooncake._find_id_uses!(d, Expr(:call, sin, id))
                @test d[id] == true
            end
            @testset "IDGotoIfNot" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake._find_id_uses!(d, IDGotoIfNot(ID(), ID()))
                @test d[id] == false
                Mooncake._find_id_uses!(d, IDGotoIfNot(true, ID()))
                @test d[id] == false
                Mooncake._find_id_uses!(d, IDGotoIfNot(id, ID()))
                @test d[id] == true
            end
            @testset "IDGotoNode" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake._find_id_uses!(d, IDGotoNode(ID()))
                @test d[id] == false
            end
            @testset "IDPhiNode" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake._find_id_uses!(d, IDPhiNode([ID()], Vector{Any}(undef, 1)))
                @test d[id] == false
                Mooncake._find_id_uses!(d, IDPhiNode([ID()], Any[id]))
                @test d[id] == true
            end
            @testset "PiNode" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake._find_id_uses!(d, PiNode(false, Bool))
                @test d[id] == false
                Mooncake._find_id_uses!(d, PiNode(id, Bool))
                @test d[id] == true
            end
            @testset "ReturnNode" begin
                id = ID()
                d = Dict{ID,Bool}(id => false)
                Mooncake._find_id_uses!(d, ReturnNode())
                @test d[id] == false
                Mooncake._find_id_uses!(d, ReturnNode(5))
                @test d[id] == false
                Mooncake._find_id_uses!(d, ReturnNode(id))
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
        @test Mooncake._is_reachable(_ircode_to_cfg_blocks(ir)) == [true, false, false]
    end
    @testset "_remove_unreachable_cfg_blocks!" begin

        # This test case has two important features:
        # 1. the second basic block (the second statement) cannot be reached, and
        # 2. the PhiNode in the third basic block refers to the second basic block. Since
        #   the second block will be removed, the edge / value in the PhiNode corresponding
        #   to the second block must be removed as part of the call to
        #   _remove_unreachable_cfg_blocks!.
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
        blocks = _ircode_to_cfg_blocks(ir)
        new_blocks = Mooncake._remove_unreachable_cfg_blocks!(blocks)

        # The returned vector is fresh and the input vector is not resized; surviving blocks
        # are the same objects shared with the input (the phi edges are mutated in place).
        @test new_blocks !== blocks
        @test length(blocks) == 3
        @test blocks[3] === new_blocks[2]

        # Check that only the first and third block remain in the new IR.
        @test length(new_blocks) == 2
        @test blocks[1].id == new_blocks[1].id
        @test blocks[3].id == new_blocks[2].id

        # Check that the reference to the second block in the PhiNode has been removed.
        updated_id_phi_node = new_blocks[2].insts[1].stmt
        @test length(updated_id_phi_node.edges) == 1
        @test length(updated_id_phi_node.values) == 1
        @test only(updated_id_phi_node.values) == true

        # Get the IRCode, and ensure that the statements in it agree with what is expected.
        new_ir = lower_cfg_blocks_to_ir(new_blocks, ir)
        expected_stmts = Any[
            GotoNode(2), PhiNode(Int32[1], Any[true]), ReturnNode(SSAValue(2))
        ]
        @test Mooncake.stmt(new_ir.stmts) == expected_stmts
    end
    @testset "Switch lowering" begin
        # The conds/dests length invariant is enforced.
        @test_throws AssertionError Mooncake.Switch(Any[true], ID[ID(), ID()], ID())

        d1, d2, fallthrough = ID(), ID(), ID()
        sw = Mooncake.Switch(Any[true, false], ID[d1, d2], fallthrough)
        blk_id = ID()
        blks = CFGBlock[
            CFGBlock(
                blk_id, Mooncake.IDInstPair[(ID(), new_inst(nothing)), (ID(), new_inst(sw))]
            ),
            CFGBlock(d1, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(1)))]),
            CFGBlock(d2, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(2)))]),
            CFGBlock(fallthrough, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(3)))]),
        ]

        # A Switch block has all dests plus the fallthrough as successors.
        @test Mooncake._compute_cfg_successors(blks)[blk_id] == ID[d1, d2, fallthrough]

        # Lowering replaces the Switch with a base block (Switch stripped), one IDGotoIfNot
        # block per (cond, dest), and a final IDGotoNode fallthrough block.
        lowered = Mooncake._cfg_lower_switch_statements(blks)
        @test length(lowered) == length(blks) + 3
        @test lowered[1].id == blk_id
        @test Mooncake.terminator(lowered[1]) === nothing  # Switch stripped
        t_d1 = Mooncake.terminator(lowered[2])
        @test t_d1 isa IDGotoIfNot && t_d1.cond == true && t_d1.dest == d1
        t_d2 = Mooncake.terminator(lowered[3])
        @test t_d2 isa IDGotoIfNot && t_d2.cond == false && t_d2.dest == d2
        t_ft = Mooncake.terminator(lowered[4])
        @test t_ft isa IDGotoNode && t_ft.label == fallthrough
    end
    @testset "_cfg_remove_double_edges" begin
        # An IDGotoIfNot whose dest is the immediate next block collapses to an IDGotoNode.
        b2 = ID()
        blks = CFGBlock[
            CFGBlock(ID(), Mooncake.IDInstPair[(ID(), new_inst(IDGotoIfNot(true, b2)))]),
            CFGBlock(b2, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(5)))]),
        ]
        t = Mooncake.terminator(Mooncake._cfg_remove_double_edges(blks)[1])
        @test t isa IDGotoNode && t.label == b2

        # A branch to a non-successor block is left unchanged (same object returned).
        elsewhere = ID()
        blkA = CFGBlock(
            ID(), Mooncake.IDInstPair[(ID(), new_inst(IDGotoIfNot(true, elsewhere)))]
        )
        blks2 = CFGBlock[
            blkA, CFGBlock(ID(), Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(1)))])
        ]
        @test Mooncake._cfg_remove_double_edges(blks2)[1] === blkA
    end
    @testset "_sort_cfg_blocks!" begin
        entry, a, b, c = ID(), ID(), ID(), ID()
        # Input out of order; `a` and `b` are both unreachable (distance typemax).
        blks = CFGBlock[
            CFGBlock(entry, Mooncake.IDInstPair[(ID(), new_inst(IDGotoNode(c)))]),
            CFGBlock(a, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(1)))]),
            CFGBlock(b, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(2)))]),
            CFGBlock(c, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(3)))]),
        ]
        out = Mooncake._sort_cfg_blocks!(blks)
        @test out === blks                      # mutates and returns the same vector
        @test out[1].id == entry                # entry (distance 0) first
        @test out[2].id == c                    # reachable (distance 1) before unreachables
        @test Set([out[3].id, out[4].id]) == Set([a, b])  # both unreachable blocks land last

        # Reachable chain with distinct distances, presented out of order (y before x):
        # ordering must follow increasing distance-from-entry, not input position.
        e2, x, y = ID(), ID(), ID()
        chain = CFGBlock[
            CFGBlock(e2, Mooncake.IDInstPair[(ID(), new_inst(IDGotoNode(x)))]),  # distance 0
            CFGBlock(y, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(0)))]),   # distance 2
            CFGBlock(x, Mooncake.IDInstPair[(ID(), new_inst(IDGotoNode(y)))]),   # distance 1
        ]
        @test [blk.id for blk in Mooncake._sort_cfg_blocks!(chain)] == [e2, x, y]
    end
    @testset "_distance_to_entry (back-edge / cycle)" begin
        # The BFS must not re-queue or overwrite a block reached via a back-edge.
        header, body, unreachable = ID(), ID(), ID()
        blks = CFGBlock[
            # header branches to itself (back-edge) and falls through to body.
            CFGBlock(
                header, Mooncake.IDInstPair[(ID(), new_inst(IDGotoIfNot(true, header)))]
            ),
            CFGBlock(body, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(1)))]),
            CFGBlock(unreachable, Mooncake.IDInstPair[(ID(), new_inst(ReturnNode(2)))]),
        ]
        d = Mooncake._distance_to_entry(blks)
        @test d[1] == 0                # header stays distance 0 despite the self back-edge
        @test d[2] == 1                # body reached via fallthrough
        @test d[3] == typemax(Int)     # unreachable
        @test Mooncake._is_reachable(blks) == [true, true, false]
    end
    @testset "new_inst defaults" begin
        ni = new_inst(nothing, Float64)
        @test ni.stmt === nothing
        @test ni.type === Float64
        @test ni.info isa CC.NoCallInfo
        @test ni.flag == CC.IR_FLAG_REFINED
    end
    @testset "is_reachable_return_node" begin
        @test Mooncake.is_reachable_return_node(ReturnNode(5)) == true
        @test Mooncake.is_reachable_return_node(ReturnNode()) == false
        @test Mooncake.is_reachable_return_node(IDGotoNode(ID())) == false
    end
    @testset "seed_id!" begin
        # Reseeding restores determinism: IDs created after each reset match (same thread).
        Mooncake.seed_id!()
        a = ID()
        Mooncake.seed_id!()
        b = ID()
        @test a == b
        @test a.id == 0
    end
end
