# Forward-over-reverse HVP via the public `prepare_hvp_cache` /
# `value_and_hvp!!` API. The `:forward_over_reverse` HVP path is wired
# through `_GradClosure`'s primal-mode `frule!!` in
# `src/rules/high_order_derivative_patches.jl`, which routes through an
# NDual-lifted reverse rule built at cache-prep time. That keeps the
# inner pullback's fdata storage identified with the outer forward-mode
# lifted slot — gradient and HVP both accumulate in the same NDual
# storage, avoiding the `Vector{Float64}` vs `Vector{NDual{Float64,1}}`
# storage mismatch a naive nested-AD path would hit.
function _hessian_column(f, x::Vector{Float64}, i::Int)
    cache = Mooncake.prepare_hvp_cache(f, x)
    v = zeros(length(x))
    v[i] = 1.0
    _, grad, hvp = Mooncake.value_and_hvp!!(cache, f, v, x)
    return grad, hvp
end

function _compute_hessian(f, x::Vector{Float64})
    n = length(x)
    H = zeros(n, n)
    for i in 1:n
        _, H[:, i] = _hessian_column(f, x, i)
    end
    return H
end

@testset "hessian_scalar_functions" begin
    @testset "sum" begin
        g(x) = sum(x)
        x = [2.0]
        grad, hess_col = _hessian_column(g, x, 1)
        @test grad ≈ [1.0]
        @test hess_col ≈ [0.0]
    end

    @testset "x^4.0" begin
        f(x) = x[1]^4.0
        x = [2.0]
        grad, hess_col = _hessian_column(f, x, 1)
        @test grad ≈ [32.0]
        @test hess_col ≈ [48.0]
    end

    @testset "x^4" begin
        f(x) = x[1]^4
        x = [2.0]
        grad, hess_col = _hessian_column(f, x, 1)
        @test grad ≈ [32.0]
        @test hess_col ≈ [48.0]
    end

    @testset "x^6" begin
        f(x) = x[1]^6
        x = [2.0]
        grad, hess_col = _hessian_column(f, x, 1)
        @test grad ≈ [192.0]
        @test hess_col ≈ [480.0]
    end
end

@testset "hessian_multivariate" begin
    @testset "Rosenbrock" begin
        rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
        z = [1.2, 1.2]
        H = _compute_hessian(rosen, z)
        expected_H = [1250.0 -480.0; -480.0 200.0]
        @test H ≈ expected_H rtol = 1e-10
    end

    @testset "sum of squares" begin
        f(x) = sum([x[1] * x[1], x[2] * x[2]])
        x = [2.0, 3.0]
        grad, hess_col = _hessian_column(f, x, 1)
        @test grad ≈ [4.0, 6.0] rtol = 1e-10
        @test hess_col ≈ [2.0, 0.0] rtol = 1e-10
    end

    @testset "broadcast sum of squares" begin
        # Tests broadcast operations: x .* x uses broadcasting
        f(x) = sum(x .* x)
        x = [2.0, 3.0]
        H = _compute_hessian(f, x)
        # f(x) = x₁² + x₂², so ∇f = [2x₁, 2x₂] and H = 2I
        @test H ≈ [2.0 0.0; 0.0 2.0] rtol = 1e-10
    end

    @testset "GAMS objective" begin
        function gams_objective(x)
            #! format: off
            objvar = (((((((((((((((((((((((((((x[1] * x[1] + x[10] * x[10]) * (x[1] * x[1] + x[10] * x[10]) - 4 * x[1]) + 3) + (x[2] * x[2] + x[10] * x[10]) * (x[2] * x[2] + x[10] * x[10])) - 4 * x[2]) + 3) + (x[3] * x[3] + x[10] * x[10]) * (x[3] * x[3] + x[10] * x[10])) - 4 * x[3]) + 3) + (x[4] * x[4] + x[10] * x[10]) * (x[4] * x[4] + x[10] * x[10])) - 4 * x[4]) + 3) + (x[5] * x[5] + x[10] * x[10]) * (x[5] * x[5] + x[10] * x[10])) - 4 * x[5]) + 3) + (x[6] * x[6] + x[10] * x[10]) * (x[6] * x[6] + x[10] * x[10])) - 4 * x[6]) + 3) + (x[7] * x[7] + x[10] * x[10]) * (x[7] * x[7] + x[10] * x[10])) - 4 * x[7]) + 3) + (x[8] * x[8] + x[10] * x[10]) * (x[8] * x[8] + x[10] * x[10])) - 4 * x[8]) + 3) + (x[9] * x[9] + x[10] * x[10]) * (x[9] * x[9] + x[10] * x[10])) - 4 * x[9]) + 3) - 0
            #! format: on
            return objvar
        end

        x0 = [0.0; fill(1.0, 9)]
        H = _compute_hessian(gams_objective, x0)

        H_expected = zeros(10, 10)
        H_expected[1, 1] = 4.0
        for i in 2:9
            H_expected[i, i] = 16.0
            H_expected[i, 10] = 8.0
            H_expected[10, i] = 8.0
        end
        H_expected[10, 10] = 140.0

        @test H ≈ H_expected rtol = 1e-10
    end
end

@testset "forward over reverse (public interface)" begin
    function compute_hessian(
        f, x::Vector{Float64}; reverse_debug_mode=false, forward_debug_mode=false
    )
        # `prepare_hvp_cache` runs the reverse-mode rule build internally with
        # the debug_mode flag from `config`. For this regression test the
        # outer forward and inner reverse share the same debug flag — the
        # mixed-debug case (debug reverse + non-debug forward) is exercised
        # by the `reverse_debug_mode=true` variant below.
        debug_mode = reverse_debug_mode || forward_debug_mode
        cache = Mooncake.prepare_hvp_cache(f, x; config=Mooncake.Config(; debug_mode))
        n = length(x)
        H = zeros(n, n)
        for i in 1:n
            v = zeros(n)
            v[i] = 1.0
            _, _, hvp = Mooncake.value_and_hvp!!(cache, f, v, x)
            H[:, i] = hvp
        end
        return H
    end

    @testset "Rosenbrock" begin
        rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
        z = [1.2, 1.2]
        H = compute_hessian(rosen, z)
        expected_H = [1250.0 -480.0; -480.0 200.0]
        @test H ≈ expected_H rtol = 1e-10
    end

    @testset "Rosenbrock (debug reverse rule boundary)" begin
        rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
        z = [1.2, 1.2]
        H = compute_hessian(rosen, z; reverse_debug_mode=true)
        expected_H = [1250.0 -480.0; -480.0 200.0]
        @test H ≈ expected_H rtol = 1e-10
    end
end

@testset "reverse over reverse fails" begin
    rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
    z = [1.2, 1.2]

    rvscache = prepare_gradient_cache(rosen, z)
    grad(y) = value_and_gradient!!(rvscache, rosen, y)[2][2]
    # On Julia 1.10, __call_rule's inferencebarrier makes __value_and_gradient!! opaque to
    # Mooncake's rule compiler, so build_rrule fails with MooncakeRuleCompilationError
    # before reaching the ArgumentError thrown by MistyClosure.rrule!!.
    @static if VERSION >= v"1.11-"
        @test_throws "not currently supported" prepare_gradient_cache(grad, z)
    else
        @test try
            prepare_gradient_cache(grad, z)
            false
        catch e
            e isa Mooncake.MooncakeRuleCompilationError ||
                (e isa ArgumentError && occursin("not currently supported", e.msg))
        end
    end
end

@testset "native HVP interface (prepare_hvp_cache + value_and_hvp!!)" begin
    @testset "gradient correctness for x^4" begin
        f(x) = x[1]^4.0
        x = [2.0]
        cache = prepare_hvp_cache(f, x)
        f_val, grad, _ = value_and_hvp!!(cache, f, [1.0], x)
        @test f_val ≈ 16.0
        @test grad ≈ [32.0]
    end

    @testset "HVP correctness for x^4" begin
        f(x) = x[1]^4.0
        x = [2.0]
        _, _, hvp = value_and_hvp!!(prepare_hvp_cache(f, x), f, [1.0], x)
        @test hvp ≈ [48.0]
    end

    @testset "cache reuse across multiple HVP calls" begin
        # The LazyFoRRule should compile the inner rule only once; verify the cache
        # produces consistent results when called repeatedly with different directions.
        f(x) = sum(x .* x)  # H = 2I
        x = [1.0, 2.0, 3.0]
        cache = prepare_hvp_cache(f, x)
        n = length(x)
        for i in 1:n
            v = zeros(n)
            v[i] = 1.0
            _, _, hvp = value_and_hvp!!(cache, f, v, x)
            expected = 2.0 .* v
            @test hvp ≈ expected rtol = 1e-10
        end
    end

    @testset "multi-argument HVP" begin
        # f(x, y) = sum(x .* x) + sum(y .* y): H = 2I (block-diagonal, decoupled)
        f(x, y) = sum(x .* x) + sum(y .* y)
        x = [1.0, 2.0]
        y = [3.0]
        cache = prepare_hvp_cache(f, x, y)
        _, (grad_x, grad_y), (hvp_x, hvp_y) = value_and_hvp!!(
            cache, f, ([1.0, 0.0], [0.0]), x, y
        )
        @test grad_x ≈ [2.0, 4.0] rtol = 1e-10
        @test grad_y ≈ [6.0] rtol = 1e-10
        @test hvp_x ≈ [2.0, 0.0] rtol = 1e-10
        @test hvp_y ≈ [0.0] rtol = 1e-10
    end
end
