using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using JET, Lux, LuxLib, Mooncake, NNlib, SLEEFPirates, StableRNGs, Test
using LuxLib.Impl: sleefpirates_fast_act
using Mooncake.TestUtils: test_rule

@testset "luxlib" begin
    @testset "$(typeof(fargs))" for (interface_only, perf_flag, is_primitive, fargs...) in
                                    vcat(
        Any[
            (false, :none, true, LuxLib.Impl.matmul, randn(5, 4), randn(4, 3)),
            (false, :none, true, LuxLib.Impl.matmuladd, randn(5, 4), randn(4, 3), randn(5)),
            (
                false,
                :none,
                true,
                LuxLib.Impl.batched_matmul_fallback,
                randn(5, 4, 3),
                randn(4, 3, 3),
            ),
            (false, :none, false, LuxLib.Impl.activation, Lux.relu, randn(5, 4)),
        ],
        map(
            Any[
                LuxLib.NNlib.sigmoid_fast,
                LuxLib.NNlib.softplus,
                LuxLib.NNlib.logsigmoid,
                LuxLib.NNlib.swish,
                LuxLib.NNlib.lisht,
                Base.tanh,
                LuxLib.NNlib.tanh_fast,
            ],
        ) do f
            return (false, :stability_and_allocs, true, sleefpirates_fast_act(f), randn())
        end,
        Any[
            (
                false,
                :stability_and_allocs,
                true,
                LuxLib.Utils.static_training_mode_check,
                nothing,
                LuxLib.Utils.True(),
                LuxLib.Utils.True(),
            ),
            (
                false,
                :none,
                false,
                function (opmode, act, x, m, sigma2, gamma, beta)
                    return LuxLib.Impl.batchnorm_affine_normalize_internal(
                        opmode, act, x, m, sigma2, gamma, beta, 1e-3
                    )
                end,
                LuxLib.LoopedArrayOp(),
                Lux.relu,
                randn(5, 4, 3),
                randn(4),
                rand(4) .+ 1.0,
                nothing,
                nothing,
            ),
        ],
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [randn(5), nothing],
                    [Lux.relu, tanh, NNlib.gelu],
                ),
            ) do (opmode, bias, activation)
                (
                    false,
                    :none,
                    false,
                    LuxLib.Impl.fused_dense,
                    opmode,
                    activation,
                    randn(5, 4),
                    randn(4, 2),
                    bias,
                )
            end,
        ),
    )
        mode = Mooncake.ReverseMode
        test_rule(StableRNG(123), fargs...; perf_flag, is_primitive, interface_only, mode)
    end
end

# tests should pass for derived rules.
# Lux kernel fusion stuff
@testset "fused_dense backward pass" begin
    using Mooncake, LuxLib, NNlib, Zygote, Random
    using Test
    # @testset for wT in (Float32, Float64),
    wT = Float32
    xT = Float32
    # , Float64)
    has_bias = true
    #  false)

    x = randn(xT, 3, 6)
    weight = randn(wT, 2, 3)
    bias = has_bias ? randn(wT, 2) : nothing

    fn = sum ∘ fused_dense_bias_activation

    cache = Mooncake.build_rrule(fn, gelu, weight, x, bias)
    _, (_, _, ∂weight, ∂x, ∂bias) = value_and_gradient!!(cache, fn, gelu, weight, x, bias)

    _, ∂weight_zyg, ∂x_zyg, ∂bias_zyg = Zygote.gradient(fn, gelu, weight, x, bias)

    @test ∂x ≈ ∂x_zyg atol = 1.0e-3 rtol = 1.0e-3
    @test ∂weight ≈ ∂weight_zyg atol = 1.0e-3 rtol = 1.0e-3
    if has_bias
        @test ∂bias ≈ ∂bias_zyg atol = 1.0e-3 rtol = 1.0e-3
    end
end
