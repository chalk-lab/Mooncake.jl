@testset "misc" begin
    @testset "stop_gradient" begin
        # Primal pass-through: same object returned.
        x = [1.0, 2.0]
        @test Mooncake.stop_gradient(3.0) === 3.0
        @test Mooncake.stop_gradient(x) === x
        @test Mooncake.stop_gradient(x, 4.0) == (x, 4.0)

        # Gradient is zero when the entire input goes through stop_gradient.
        f_zero(x) = sum(Mooncake.stop_gradient(x))
        c_zero = Mooncake.prepare_gradient_cache(f_zero, x)
        _, (_, g_zero) = Mooncake.value_and_gradient!!(c_zero, f_zero, x)
        @test iszero(g_zero)

        # Partial stop: gradient flows through the non-stopped path only.
        f_partial(x) = x[1] * Mooncake.stop_gradient(x)[2]
        c_partial = Mooncake.prepare_gradient_cache(f_partial, x)
        _, (_, g_partial) = Mooncake.value_and_gradient!!(c_partial, f_partial, x)
        @test g_partial ≈ [2.0, 0.0]

        # Multi-arg: both gradients are zero.
        f_multi(x, y) =
            sum(Mooncake.stop_gradient(x, y)[1]) + Mooncake.stop_gradient(x, y)[2]
        c_multi = Mooncake.prepare_gradient_cache(f_multi, x, 4.0)
        _, (_, gx, gy) = Mooncake.value_and_gradient!!(c_multi, f_multi, x, 4.0)
        @test iszero(gx)
        @test iszero(gy)

        # Kwargs throw in both primal and AD contexts.
        @test_throws ArgumentError Mooncake.stop_gradient(1.0; kw=1)
        f_kw(x) = sum(Mooncake.stop_gradient(x; kw=1))
        @test_throws ArgumentError Mooncake.prepare_gradient_cache(f_kw, [1.0, 2.0])
    end
    @testset "lgetfield" begin
        x = (5.0, 4)
        @test lgetfield(x, Val(1)) == getfield(x, 1)
        @test lgetfield(x, Val(2)) == getfield(x, 2)

        y = (a=5.0, b=4)
        @test lgetfield(y, Val(:a)) == getfield(y, :a)
        @test lgetfield(y, Val(:b)) == getfield(y, :b)
    end
    @testset "lsetfield!" begin
        x = TestResources.MutableFoo(5.0, randn(5))
        @test Mooncake.lsetfield!(x, Val(:a), 4.0) == 4.0
        @test x.a == 4.0

        new_b = zeros(10)
        @test Mooncake.lsetfield!(x, Val(:b), new_b) === new_b
        @test x.b === new_b
    end

    TestUtils.run_rule_test_cases(StableRNG, Val(:misc))
end
