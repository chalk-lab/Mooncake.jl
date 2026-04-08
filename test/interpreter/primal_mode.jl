@testset "primal_mode" begin
    f(x) = sin(x) + x

    @testset "basic build_primal" begin
        primal_f = Mooncake.build_primal(f, 1.0)
        @test primal_f(2.0) == f(2.0)
        y = primal_f(Mooncake.Dual(2.0, 1.0))
        @test Mooncake.primal(y) == f(2.0)
        @test Mooncake.tangent(y) == Mooncake.NTangent((cos(2.0) + 1,))

        chunked_primal_f = Mooncake.build_primal(
            f, 1.0; tangent_mode=Mooncake.IRfwdMode{2}()
        )
        y_chunked = chunked_primal_f(Mooncake.Dual(2.0, Mooncake.NTangent((1.0, -1.0))))
        @test Mooncake.primal(y_chunked) == f(2.0)
        @test Mooncake.tangent(y_chunked) ==
            Mooncake.NTangent((cos(2.0) + 1, -(cos(2.0) + 1)))

        debug_primal_f = Mooncake.build_primal(f, 1.0; debug_mode=true)
        y_debug = debug_primal_f(Mooncake.Dual(2.0, 1.0))
        @test Mooncake.primal(y_debug) == f(2.0)
        @test Mooncake.tangent(y_debug) == Mooncake.NTangent((cos(2.0) + 1,))

        y_rule, dy_rule = Mooncake.value_and_derivative!!(
            primal_f, (f, Mooncake.NoTangent()), (2.0, 1.0)
        )
        @test y_rule == f(2.0)
        @test dy_rule == Mooncake.NTangent((cos(2.0) + 1,))

        y_grad, grad = Mooncake.value_and_gradient!!(primal_f, f, 2.0)
        @test y_grad == f(2.0)
        @test grad == (Mooncake.NoTangent(), cos(2.0) + 1)
    end

    @testset "varargs build_primal" begin
        f_vararg(x, ys...) = x + sum(ys; init=0.0)
        primal_f3 = Mooncake.build_primal(f_vararg, 1.0, 2.0, 3.0)
        primal_f1 = Mooncake.build_primal(f_vararg, 1.0)
        @test primal_f3(1.0, 2.0, 3.0) == f_vararg(1.0, 2.0, 3.0)
        @test primal_f1(1.0) == f_vararg(1.0)
    end

    @testset "lazy and dynamic primal dual calls" begin
        lazy_f = Mooncake.LazyPrimal(Tuple{typeof(f),Float64}, Base.get_world_counter())
        y_lazy = lazy_f(Mooncake.zero_dual(f), Mooncake.Dual(2.0, 1.0))
        @test Mooncake.primal(y_lazy) == f(2.0)
        @test Mooncake.tangent(y_lazy) == Mooncake.NTangent((cos(2.0) + 1,))

        dynamic_f = Mooncake.DynamicPrimal(Base.get_world_counter())
        y_dynamic = dynamic_f(Mooncake.zero_dual(f), Mooncake.Dual(2.0, 1.0))
        @test Mooncake.primal(y_dynamic) == f(2.0)
        @test Mooncake.tangent(y_dynamic) == Mooncake.NTangent((cos(2.0) + 1,))
    end

    @testset "constructor lowering" begin
        struct Box64PrimalMode
            x::Float64
        end
        g(x) = Box64PrimalMode(x).x + 1
        primal_g = Mooncake.build_primal(g, 2.0)
        @test primal_g(2.0) == g(2.0)
    end

    @testset "concrete signature" begin
        f_vec(x::Vector{Float64}) = sum(x)
        x = [1.0, 2.0, 3.0]
        primal_f_vec = Mooncake.build_primal(f_vec, x)
        @test primal_f_vec(x) == f_vec(x)
    end

    @testset "recursive callee rewrite" begin
        inner(x) = sin(x) + x
        outer(x) = 2 * inner(x)
        primal_outer = Mooncake.build_primal(outer, 1.0)
        @test primal_outer(2.0) == outer(2.0)
        y = primal_outer(Mooncake.Dual(2.0, 1.0))
        @test Mooncake.primal(y) == outer(2.0)
        @test Mooncake.tangent(y) == Mooncake.NTangent((2 * (cos(2.0) + 1),))
    end
end
