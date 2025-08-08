module ContextsTestModule

using Mooncake: @is_primitive, DefaultCtx

foo(x) = x

@is_primitive DefaultCtx Tuple{typeof(foo),Float64}

end

@testset "contexts" begin
    @testset "$mode" for mode in [Mooncake.ForwardMode, Mooncake.ReverseMode]
        Tf = typeof(ContextsTestModule.foo)
        @test Mooncake.is_primitive(DefaultCtx, mode, Tuple{Tf,Float64})
        @test !Mooncake.is_primitive(DefaultCtx, mode, Tuple{Tf,Real})
    end
end
