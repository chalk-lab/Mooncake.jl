@testset "misc" begin
    @testset "stop_gradient" begin
        # Primal pass-through: same object returned.
        x = [1.0, 2.0]
        @test Mooncake.stop_gradient(3.0) === 3.0
        @test Mooncake.stop_gradient(x) === x
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

        # Multiple values: pack into a tuple, gradients for all elements are zeroed.
        y = [3.0, 4.0]
        function f_tuple(x, y)
            t = Mooncake.stop_gradient((x, y))
            return sum(t[1]) + sum(t[2])
        end
        c_tuple = Mooncake.prepare_gradient_cache(f_tuple, x, y)
        _, (_, gx_tuple, gy_tuple) = Mooncake.value_and_gradient!!(c_tuple, f_tuple, x, y)
        @test iszero(gx_tuple)
        @test iszero(gy_tuple)
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
