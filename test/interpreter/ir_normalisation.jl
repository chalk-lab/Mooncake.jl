@testset "ir_normalisation" begin
    @testset "interpolate_boundschecks" begin
        statements = Any[Expr(:boundscheck, true), Expr(:call, sin, SSAValue(1))]
        Mooncake._interpolate_boundschecks!(statements)
        @test statements[2].args[2] == true
    end
    @testset "foreigncall_to_call" begin
        foreigncall = Expr(
            :foreigncall,
            :(:jl_array_isassigned),
            Int32,
            svec(Any, UInt64),
            0,
            :(:ccall),
            Argument(2),
            0x0000000000000001,
            0x0000000000000001,
        )
        sp_map = Dict{Symbol,CC.VarState}()
        call = Mooncake.foreigncall_to_call(foreigncall, sp_map)
        @test Meta.isexpr(call, :call)
        @test call.args[1] == Mooncake._foreigncall_
    end
    @testset "fix_up_invoke_inference!" begin
        if VERSION >= v"1.11" # Base.method_instance does not exist on 1.10.
            mi = Base.method_instance(TestResources.inplace_invoke!, (Vector{Float64},))
            ir = Mooncake.ircode(
                Any[
                    Expr(:invoke, mi, TestResources.inplace_invoke!, Argument(2)),
                    ReturnNode(nothing),
                ],
                Any[Any, Vector{Float64}],
            )
            @test Mooncake.Compiler.statement_type(ir, 1) == Any
            ir = Mooncake.fix_up_invoke_inference!(ir)
            @test Mooncake.Compiler.statement_type(ir, 1) == Nothing
        end
    end
    @testset "new_to_call" begin
        new_ex = Expr(:new, GlobalRef(Mooncake, :Foo), SSAValue(1), :hi)
        call_ex = Mooncake.new_to_call(new_ex)
        @test Meta.isexpr(call_ex, :call)
        @test call_ex.args[1] == Mooncake._new_
        @test call_ex.args[2:end] == new_ex.args
    end
    @testset "splatnew_to_call" begin
        splatnew_ex = Expr(:splatnew, GlobalRef(Mooncake, :Foo), SSAValue(1))
        call_ex = Mooncake.splatnew_to_call(splatnew_ex)
        @test Meta.isexpr(call_ex, :call)
        @test call_ex.args[1] == Mooncake._splat_new_
        @test call_ex.args[2:end] == splatnew_ex.args
    end
    @testset "intrinsic_to_function" begin
        @testset "GlobalRef" begin
            intrinsic_ex = Expr(:call, GlobalRef(Core.Intrinsics, :abs_float), SSAValue(1))
            wrapper_ex = Mooncake.intrinsic_to_function(intrinsic_ex)
            @test wrapper_ex.args[1] == Mooncake.IntrinsicsWrappers.abs_float
        end
        @testset "IntrinsicFunction" begin
            intrinsic_ex = Expr(:call, Core.Intrinsics.abs_float, SSAValue(1))
            wrapper_ex = Mooncake.intrinsic_to_function(intrinsic_ex)
            @test wrapper_ex.args[1] == Mooncake.IntrinsicsWrappers.abs_float
        end
        @testset "cglobal" begin
            cglobal_ex = Expr(:call, cglobal, :jl_uv_stdout, Ptr{Cvoid})
            wrapper_ex = Mooncake.intrinsic_to_function(cglobal_ex)
            @test wrapper_ex.args[1] == Mooncake.IntrinsicsWrappers.__cglobal
        end
    end
    @testset "lift_getfield_and_others $ex" for (ex, target) in Any[
        (ReturnNode(5), ReturnNode(5)),
        (
            Expr(:call, getfield, SSAValue(1), 5),
            Expr(:call, lgetfield, SSAValue(1), Val(5)),
        ),
        (
            Expr(:call, GlobalRef(Core, :getfield), SSAValue(1), 5),
            Expr(:call, lgetfield, SSAValue(1), Val(5)),
        ),
        (
            Expr(:call, QuoteNode(getfield), SSAValue(1), 5),
            Expr(:call, lgetfield, SSAValue(1), Val(5)),
        ),
        (
            Expr(:call, getfield, SSAValue(1), SSAValue(2)),
            Expr(:call, getfield, SSAValue(1), SSAValue(2)),
        ),
        (
            Expr(:call, getfield, SSAValue(1), QuoteNode(:x)),
            Expr(:call, lgetfield, SSAValue(1), Val(:x)),
        ),
        (
            Expr(:call, GlobalRef(Core, :setfield!), SSAValue(1), 2, SSAValue(3)),
            Expr(:call, lsetfield!, SSAValue(1), Val(2), SSAValue(3)),
        ),
        (
            Expr(:call, setfield!, SSAValue(1), 2, SSAValue(3)),
            Expr(:call, lsetfield!, SSAValue(1), Val(2), SSAValue(3)),
        ),
        (
            Expr(:call, setfield!, SSAValue(1), QuoteNode(:a), SSAValue(3)),
            Expr(:call, lsetfield!, SSAValue(1), Val(:a), SSAValue(3)),
        ),
        (Expr(:call, sin, SSAValue(1)), Expr(:call, sin, SSAValue(1))),
    ]
        @test Mooncake.lift_getfield_and_others(ex) == target
    end
    @testset "gc_preserve_begin and gc_preserve_end" begin

        # Check that the placeholder function added to Mooncake.jl behaves as expected.
        @test Mooncake.gc_preserve(5.0) === nothing

        # Thanks to maleadt for this suggestion. For more info, see:
        # https://discourse.julialang.org/t/testing-gc-preserve-when-doing-compiler-passes/102241
        mutable struct FinalizerObject
            finalized::Bool
            @noinline function FinalizerObject()
                return finalizer(new(false)) do obj
                    obj.finalized = true
                end
            end
        end

        # Check that after running the primal, the object can be freed.
        function test_no_preserve()
            x = FinalizerObject()
            ptr = convert(Ptr{Bool}, Base.pointer_from_objref(x))
            GC.gc(true)
            return unsafe_load(ptr)
        end
        @test test_no_preserve()

        # Check that if you insert a call to `gc_preserve`, the object is not finalised.
        function test_preserved()
            x = FinalizerObject()
            _, pb!! = Mooncake.rrule!!(
                zero_fcodual(Mooncake.gc_preserve), Mooncake.zero_fcodual(x)
            )
            ptr = convert(Ptr{Bool}, Base.pointer_from_objref(x))
            GC.gc(true)
            return unsafe_load(ptr), pb!!
        end
        finalised, pb!! = test_preserved()
        @test !finalised

        # Check that translation of expressions happens correctly.
        @test ==(
            Mooncake.lift_gc_preservation(
                Expr(:gc_preserve_begin, Argument(1), SSAValue(2))
            ),
            Expr(:call, Mooncake.gc_preserve, Argument(1), SSAValue(2)),
        )
        @test Mooncake.lift_gc_preservation(Expr(:gc_preserve_end, SSAValue(2))) === nothing
    end
    @testset "remove_edge!" begin
        ir = Mooncake.ircode(
            Any[
                Expr(:call, :sin, Argument(2)),
                GotoNode(3),
                PhiNode(Int32[1], Any[5]),
                ReturnNode(SSAValue(3)),
            ],
            Any[Any, Vector{Float64}],
        )
        Mooncake.remove_edge!(ir, 1, 2)
        phi_node = Mooncake.Compiler.statements(ir)[3]
        @test isempty(phi_node.edges)
        @test isempty(phi_node.values)
        @test isempty(Mooncake.Compiler.block_successors(ir, 1))
        @test isempty(Mooncake.Compiler.block_predecessors(ir, 2))
    end
end
