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
                LuxLib.Impl.batched_matmul,
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
