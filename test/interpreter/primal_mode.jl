using Random: Xoshiro
using Mooncake: TestResources

@testset "primal_mode" begin
    f(x) = sin(x) + x
    rng = Xoshiro(123)
    manual_forward_cases = ((
        name="real scalar",
        f=f,
        primals=(1.0,),
        dual_input=Mooncake.Dual(2.0, 1.0),
        chunked_input=Mooncake.Dual(2.0, Mooncake.NTangent((1.0, -1.0))),
        primal=f(2.0),
        tangent=Mooncake.NTangent((cos(2.0) + 1,)),
        chunked_tangent=Mooncake.NTangent((cos(2.0) + 1, -(cos(2.0) + 1))),
        gradient=(Mooncake.NoTangent(), cos(2.0) + 1),
    ),)

    @testset "basic build_primal" begin
        @testset "$(case.name)" for case in manual_forward_cases
            primal_f = Mooncake.build_primal(case.f, case.primals...)
            @test primal_f(2.0) == case.primal

            y = primal_f(case.dual_input)
            @test Mooncake.primal(y) == case.primal
            @test Mooncake.tangent(y) == case.tangent

            chunked_primal_f = Mooncake.build_primal(
                case.f, case.primals...; tangent_mode=Val(2)
            )
            y_chunked = chunked_primal_f(case.chunked_input)
            @test Mooncake.primal(y_chunked) == case.primal
            @test Mooncake.tangent(y_chunked) == case.chunked_tangent

            debug_primal_f = Mooncake.build_primal(case.f, case.primals...; debug_mode=true)
            y_debug = debug_primal_f(case.dual_input)
            @test Mooncake.primal(y_debug) == case.primal
            @test Mooncake.tangent(y_debug) == case.tangent

            y_rule, dy_rule = Mooncake.value_and_derivative!!(
                primal_f, (case.f, Mooncake.NoTangent()), (2.0, 1.0)
            )
            @test y_rule == case.primal
            @test dy_rule == case.tangent

            y_grad, grad = Mooncake.value_and_gradient!!(primal_f, case.f, 2.0)
            @test y_grad == case.primal
            @test grad == case.gradient
        end
    end

    @testset "hand-written forward-rule cases" begin
        test_cases, _ = Mooncake.hand_written_rule_test_cases(Xoshiro, Val(:primal_mode))
        @testset "$(Mooncake._typeof((fx)))" for (_, pf, _, fx...) in test_cases
            @info "$(Mooncake._typeof(fx))"
            Mooncake.TestUtils.test_rule(
                rng,
                fx...;
                perf_flag=pf,
                interface_only=false,
                is_primitive=false,
                mode=Mooncake.ForwardMode,
                chunk_sizes=(1, 2),
            )
        end
    end

    @testset "mixed primal and dual inputs fail locally" begin
        f2(x, y) = x + y
        primal_f2 = Mooncake.build_primal(f2, 1.0, 2.0)
        debug_f2 = Mooncake.build_primal(f2, 1.0, 2.0; debug_mode=true)
        frule_f2 = Mooncake.build_frule(f2, 1.0, 2.0)
        dynamic_f2 = Mooncake.DynamicPrimal(Base.get_world_counter())
        ẋ = Mooncake.Dual(1.0, 1.0)
        @test_throws ArgumentError primal_f2(ẋ, 2.0)
        @test_throws ArgumentError debug_f2(ẋ, 2.0)
        @test_throws ArgumentError frule_f2(ẋ, 2.0)
        @test_throws ArgumentError dynamic_f2(f2, ẋ, 2.0)
        @test_throws ArgumentError dynamic_f2(Mooncake.zero_dual(f2), 1.0, 2.0)
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

    @testset "derived forward-rule corpus" begin
        test_cases = collect(enumerate(TestResources.generate_test_functions()))
        @testset "$n - $(Mooncake._typeof((fx)))" for (n, (int_only, pf, _, fx...)) in
                                                      test_cases

            @info "$n: $(Mooncake._typeof(fx))"
            Mooncake.TestUtils.test_rule(
                Xoshiro(123546),
                fx...;
                perf_flag=pf,
                interface_only=int_only,
                is_primitive=false,
                mode=Mooncake.ForwardMode,
            )
        end
    end
end
