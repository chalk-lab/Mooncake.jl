module ContextsTestModule

using Mooncake: @is_primitive, DefaultCtx, MinimalCtx

foo(x) = x

@is_primitive DefaultCtx Tuple{typeof(foo),Float64}
@is_primitive MinimalCtx Tuple{typeof(foo),Float32}

end

@testset "contexts" begin
    @testset "$mode" for mode in [Mooncake.ForwardMode, Mooncake.ReverseMode]
        Tf = typeof(ContextsTestModule.foo)
        world = Base.get_world_counter()

        # If declared a primitive in the DefaultCtx, it ought to be a primitive in this
        # context only.
        @test Mooncake.is_primitive(DefaultCtx, mode, Tuple{Tf,Float64}, world)
        @test !Mooncake.is_primitive(MinimalCtx, mode, Tuple{Tf,Float64}, world)

        # If something is declared a primitive in the MinimalCtx, it should automatically
        # also be one in the DefaultCtx.
        @test Mooncake.is_primitive(MinimalCtx, mode, Tuple{Tf,Float32}, world)
        @test Mooncake.is_primitive(DefaultCtx, mode, Tuple{Tf,Float32}, world)

        # A concrete type not directly declared a primitive should be a primitive in none
        # of the contexts.
        @test !Mooncake.is_primitive(DefaultCtx, mode, Tuple{Tf,Int}, world)
        @test !Mooncake.is_primitive(MinimalCtx, mode, Tuple{Tf,Int}, world)

        # `is_primitive` must also return true for signatures which are supertypes of the
        # declared signature. Note here that `Float64 <: Real` and `Float32 <: Real`.
        @test Mooncake.is_primitive(DefaultCtx, mode, Tuple{Tf,Real}, world)
        @test Mooncake.is_primitive(MinimalCtx, mode, Tuple{Tf,Real}, world)
    end
end
