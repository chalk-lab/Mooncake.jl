@testset "developer_tools" begin
    sig = Tuple{typeof(sin),Float64}
    @test Mooncake.primal_ir(Mooncake.MooncakeInterpreter(ForwardMode), sig) isa CC.IRCode
    @test Mooncake.primal_ir(Mooncake.MooncakeInterpreter(ReverseMode), sig) isa CC.IRCode
    @test Mooncake.dual_ir(sig) isa CC.IRCode
    @test Mooncake.fwd_ir(sig) isa CC.IRCode
    @test Mooncake.rvs_ir(sig) isa CC.IRCode

    # primal_ir should work even for functions with non-differentiable code (issue #668)
    function bar_llvmcall(x)
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
    @test Mooncake.primal_ir(
        Mooncake.MooncakeInterpreter(ReverseMode), Tuple{typeof(bar_llvmcall),Int}
    ) isa CC.IRCode
    @test Mooncake.primal_ir(
        Mooncake.MooncakeInterpreter(ForwardMode), Tuple{typeof(bar_llvmcall),Int}
    ) isa CC.IRCode
end
