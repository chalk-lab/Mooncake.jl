using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake
using Mooncake: Mooncake
using Mooncake.TestUtils
using Mooncake.TestUtils: test_rule, test_data
using Optim: Optim
using DynamicExpressions
using DynamicExpressions: Nullable
using StableRNGs: StableRNG
using DifferentiationInterface: AutoMooncake, gradient, prepare_gradient

# Needed for certain parts of TestUtils
using JET: JET
using AllocCheck: AllocCheck

using Test

@testset "test_data on key types" begin
    let
        test_data(StableRNG(0), Nullable{Float64}(true, 1.0))
        test_data(StableRNG(0), Nullable{Float64}(false, 1.0))
        test_data(StableRNG(0), Node{Float64}(; feature=1))
        test_data(StableRNG(0), Node{Float64}(; val=1.0))
        test_data(StableRNG(0), Node{Float64}(; op=1, children=(Node{Float64}(; val=1.0),)))
        test_data(
            StableRNG(0),
            Node{Float64}(;
                op=1, children=(Node{Float64}(; val=1.0), Node{Float64}(; val=1.0))
            ),
        )
    end
end

@testset "Basic usage checks" begin
    let
        # Build up expression
        operators = OperatorEnum(1 => (cos, sin), 2 => (+, -, *, /))
        x1, x2 = (Expression(Node{Float64}(; feature=i); operators) for i in 1:2)

        f = x1 + cos(x2 - 0.2) + 0.5
        X = randn(StableRNG(0), 3, 100)

        eval_sum = let f = f
            X -> sum(f(X))
        end
        backend = AutoMooncake(; config=nothing)
        prep = prepare_gradient(eval_sum, backend, X)
        dX = gradient(eval_sum, prep, backend, X)

        # analytic derivative: df/dx1 = 1, df/dx2 = -sin(x2 - 0.2), df/dx3 = 0
        dX_ref = zeros(size(X))
        dX_ref[1, :] .= 1
        dX_ref[2, :] .= -sin.(X[2, :] .- 0.2)
        # third row already zero
        @test isapprox(dX, dX_ref; rtol=1e-10, atol=0)
    end
end

@testset "Gradient of tree parameters" begin
    let
        operators = OperatorEnum(1 => (cos, sin), 2 => (+, -, *, /))
        x1 = Expression(Node{Float64}(; feature=1); operators)

        #  simple closed‑form ground truth: ∂/∂c sum(x1 + c) = N
        N = 100
        X = randn(StableRNG(0), 3, N)
        expr = x1 + 0.0      # constant in the tree

        @testset "Simple sum" begin
            eval_sum = let X = X
                f -> sum(f(X))
            end

            backend = AutoMooncake(; config=nothing)
            prep = prepare_gradient(eval_sum, backend, expr)
            dexpr = gradient(eval_sum, prep, backend, expr)

            const_tangent = dexpr.fields.tree.children[2].x.val
            @test const_tangent ≈ N
        end

        @testset "Propagate through a tree copy" begin
            eval_sum_copy = let X = X
                f -> sum(copy(f)(X))
            end
            backend = AutoMooncake(; config=nothing)
            full_tangent = gradient(eval_sum_copy, backend, expr)
            const_tangent = full_tangent.fields.tree.children[2].x.val
            @test const_tangent ≈ 100
        end

        @testset "Propagate through multiple tree operations" begin
            expr = (x1 + 1.0) + 2.0
            function multi_operations(f)
                tree = copy(f.tree)

                # First, we double all constants
                foreach(tree) do node
                    if node.degree == 0 && node.constant
                        node.val *= 2.0
                    end
                end

                # Then, we sum the constants
                return sum(n -> n.degree == 0 && n.constant ? n.val : 0.0, tree)
            end
            backend = AutoMooncake(; config=nothing)
            tangent_multi_op = gradient(multi_operations, backend, expr)

            # [x, y] -> [2x, 2y] -> 2x + 2y
            # Thus, the gradient is [2, 2]
            grad_1 = tangent_multi_op.fields.tree.children[2].x.val  # The 2.0
            grad_2 = tangent_multi_op.fields.tree.children[1].x.children[2].x.val
            @test grad_1 ≈ 2.0
            @test grad_2 ≈ 2.0
        end
    end
end

@testset "Use in DynamicExpressions + Optim optimization" begin
    let
        operators = OperatorEnum(1 => (cos, sin, exp), 2 => (+, -, *, /))
        x1 = Expression(Node{Float64}(; feature=1); operators)
        x2 = Expression(Node{Float64}(; feature=2); operators)
        init = x1 * exp(0.7 + 0.5 * x1) + 0.9 * x2
        target = x1 * exp(0.3 + (-0.2) * x1) + 1.5 * x2

        X = randn(StableRNG(0), 2, 128)
        y = target(X)
        backend = AutoMooncake(; config=nothing)

        f = let X = X, y = y
            function (ex)
                pred = ex(X)
                return sum(i -> abs2(pred[i] - y[i]), axes(X, 2))
            end
        end
        prep = prepare_gradient(f, backend, init)
        g! = let prep = prep, backend = backend, f = f
            function (G, ex)
                grad = gradient(f, prep, backend, ex)
                G .= extract_gradient(grad, ex)
                return nothing
            end
        end
        ex0 = copy(init)
        result = Optim.optimize(f, g!, ex0, Optim.BFGS())
        constants_final = get_scalar_constants(result.minimizer)[1]
        constants_target = get_scalar_constants(get_tree(target))[1]
        @test isapprox(constants_final, constants_target, atol=1e-5)
    end
end

@testset "TestUtils systematic tests - $(T)" for T in [Float32, Float64]
    let
        operators = OperatorEnum(
            1 => (cos, sin, exp, log, abs), 2 => (+, -, *, /), 3 => (fma, max)
        )

        x1 = Expression(Node{T,3}(; feature=1); operators)
        x2 = Expression(Node{T,3}(; feature=2); operators)

        # Various expression types - using only operators that exist
        expressions = [
            Expression(Node{T,3}(; val=T(1.0)); operators),
            x1,
            x1 + T(1.0),
            cos(x1),
            x1 * x2 + sin(x1 - T(0.5)),
            fma(x1, x2, x2 + x2),
            fma(max(x1, x2, T(2.0) * x1), x2 * T(2.1), x1 * T(3.2)),
        ]

        X = randn(StableRNG(0), T, 3, 20)

        # Test derivative with respect to X
        make_eval_sum_on_X(expr) = X -> sum(expr(X))
        @testset "test_rule - dX - $(expr)" for expr in expressions
            test_rule(
                StableRNG(1),
                make_eval_sum_on_X(expr),
                X;
                interface_only=false,
                perf_flag=:none,
                is_primitive=false,
                unsafe_perturb=true,
                mode=Mooncake.ReverseMode,
            )
        end

        # Test derivative with respect to an expression object
        make_eval_sum_on_expr(X) = expr -> sum(expr(X))
        @testset "test_rule - dexpr - $(expr)" for expr in expressions
            test_rule(
                StableRNG(2),
                make_eval_sum_on_expr(X),
                expr;
                interface_only=false,
                perf_flag=:none,
                is_primitive=false,
                unsafe_perturb=true,
                mode=Mooncake.ReverseMode,
            )
        end

        @testset "test full tangent interface - $(expr)" for expr in expressions
            test_data(StableRNG(3), expr)
        end
    end
end
