function _compute_grad(rule, f, x::Vector{Float64}, x_fdata::Vector{Float64})
    fill!(x_fdata, 0.0)
    _, pb!! = rule(zero_fcodual(f), CoDual(x, x_fdata))
    pb!!(1.0)
    return copy(x_fdata)
end

function _hessian_column(f, x::Vector{Float64}, i::Int)
    x_fdata = fdata(zero_tangent(x))
    rule = build_rrule(f, x)
    frule = build_frule(_compute_grad, rule, f, x, x_fdata)

    x_tangent = zeros(length(x))
    x_tangent[i] = 1.0
    fill!(x_fdata, 0.0)

    result = frule(
        zero_dual(_compute_grad),
        zero_dual(rule),
        zero_dual(f),
        Dual(x, x_tangent),
        Dual(x_fdata, zeros(length(x))),
    )
    return primal(result), tangent(result)
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
            return (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            (
                                                                                (
                                                                                    (
                                                                                        (
                                                                                            (
                                                                                                (
                                                                                                    (
                                                                                                        (
                                                                                                            (
                                                                                                                (
                                                                                                                    (
                                                                                                                        x[1] *
                                                                                                                        x[1] +
                                                                                                                        x[10] *
                                                                                                                        x[10]
                                                                                                                    ) *
                                                                                                                    (
                                                                                                                        x[1] *
                                                                                                                        x[1] +
                                                                                                                        x[10] *
                                                                                                                        x[10]
                                                                                                                    ) -
                                                                                                                    4 *
                                                                                                                    x[1]
                                                                                                                ) +
                                                                                                                3
                                                                                                            ) +
                                                                                                            (
                                                                                                                x[2] *
                                                                                                                x[2] +
                                                                                                                x[10] *
                                                                                                                x[10]
                                                                                                            ) *
                                                                                                            (
                                                                                                                x[2] *
                                                                                                                x[2] +
                                                                                                                x[10] *
                                                                                                                x[10]
                                                                                                            )
                                                                                                        ) -
                                                                                                        4 *
                                                                                                        x[2]
                                                                                                    ) +
                                                                                                    3
                                                                                                ) +
                                                                                                (
                                                                                                    x[3] *
                                                                                                    x[3] +
                                                                                                    x[10] *
                                                                                                    x[10]
                                                                                                ) *
                                                                                                (
                                                                                                    x[3] *
                                                                                                    x[3] +
                                                                                                    x[10] *
                                                                                                    x[10]
                                                                                                )
                                                                                            ) -
                                                                                            4 *
                                                                                            x[3]
                                                                                        ) +
                                                                                        3
                                                                                    ) +
                                                                                    (
                                                                                        x[4] *
                                                                                        x[4] +
                                                                                        x[10] *
                                                                                        x[10]
                                                                                    ) * (
                                                                                        x[4] *
                                                                                        x[4] +
                                                                                        x[10] *
                                                                                        x[10]
                                                                                    )
                                                                                ) -
                                                                                4 * x[4]
                                                                            ) + 3
                                                                        ) +
                                                                        (
                                                                            x[5] * x[5] +
                                                                            x[10] * x[10]
                                                                        ) * (
                                                                            x[5] * x[5] +
                                                                            x[10] * x[10]
                                                                        )
                                                                    ) - 4 * x[5]
                                                                ) + 3
                                                            ) +
                                                            (x[6] * x[6] + x[10] * x[10]) *
                                                            (x[6] * x[6] + x[10] * x[10])
                                                        ) - 4 * x[6]
                                                    ) + 3
                                                ) +
                                                (x[7] * x[7] + x[10] * x[10]) *
                                                (x[7] * x[7] + x[10] * x[10])
                                            ) - 4 * x[7]
                                        ) + 3
                                    ) +
                                    (x[8] * x[8] + x[10] * x[10]) *
                                    (x[8] * x[8] + x[10] * x[10])
                                ) - 4 * x[8]
                            ) + 3
                        ) + (x[9] * x[9] + x[10] * x[10]) * (x[9] * x[9] + x[10] * x[10])
                    ) - 4 * x[9]
                ) + 3
            )
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
