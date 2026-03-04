using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using JET, Lux, LuxLib, Mooncake, NNlib, SLEEFPirates, StableRNGs, Test
using LuxLib.Impl: sleefpirates_fast_act
using Mooncake.TestUtils: test_rule

# Access AD helper functions present in the Extension module.
const MooncakeLuxLibExt = Base.get_extension(Mooncake, :MooncakeLuxLibExt)
@assert !isnothing(MooncakeLuxLibExt) "MooncakeLuxLibExt is required for testing !"

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
            return (false, :allocs, true, sleefpirates_fast_act(f), randn())
        end,
        Any[(
            false,
            :stability_and_allocs,
            true,
            LuxLib.Utils.static_training_mode_check,
            nothing,
            LuxLib.Utils.True(),
            LuxLib.Utils.True(),
        ),],
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp()], [(nothing, nothing), (randn(4), randn(4))]
                ),
            ) do (opmode, (gamma, beta))
                (
                    false,
                    :none,
                    false,
                    function (opmode, x, m, sigma2, gamma, beta)
                        return MooncakeLuxLibExt._batchnorm_affine_normalize_identity(
                            opmode, x, m, sigma2, gamma, beta, 1e-3
                        )
                    end,
                    opmode,
                    randn(5, 4, 3),
                    randn(4),
                    rand(4) .+ 1.0,
                    gamma,
                    beta,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp()],
                    [(nothing, nothing), (randn(4), randn(4))],
                    [Lux.relu, tanh, identity],
                ),
            ) do (opmode, (gamma, beta), activation)
                (
                    false,
                    :none,
                    false,
                    function (opmode, act, x, m, sigma2, gamma, beta)
                        return LuxLib.Impl.batchnorm_affine_normalize_internal(
                            opmode, act, x, m, sigma2, gamma, beta, 1e-3
                        )
                    end,
                    opmode,
                    activation,
                    randn(5, 4, 3),
                    randn(4),
                    rand(4) .+ 1.0,
                    gamma,
                    beta,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [randn(5), nothing],
                    [Lux.relu, tanh, NNlib.gelu, identity],
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
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity],
                ),
            ) do (opmode, activation)
                (
                    false,
                    :none,
                    false,
                    function (opmode, act, x, bias)
                        return LuxLib.Impl.bias_activation(opmode, act, x, bias)
                    end,
                    opmode,
                    activation,
                    randn(5, 4),
                    randn(5),
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity],
                ),
            ) do (opmode, activation)
                (
                    false,
                    :none,
                    false,
                    function (opmode, act, x, bias)
                        return LuxLib.Impl.bias_activation!!(
                            opmode, LuxLib.Utils.True(), act, x, bias
                        )
                    end,
                    opmode,
                    activation,
                    randn(5, 4),
                    randn(5),
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity],
                ),
            ) do (opmode, activation)
                (
                    false,
                    :none,
                    false,
                    function (opmode, act, x, bias)
                        return LuxLib.Impl.bias_activation!!(
                            opmode, LuxLib.Utils.False(), act, x, bias
                        )
                    end,
                    opmode,
                    activation,
                    randn(5, 4),
                    randn(5),
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity],
                ),
            ) do (opmode, activation)
                (
                    false,
                    :none,
                    false,
                    function (opmode, act, x)
                        return LuxLib.Impl.activation!!(
                            opmode, LuxLib.Utils.True(), act, x
                        )
                    end,
                    opmode,
                    activation,
                    randn(5, 4),
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity],
                ),
            ) do (opmode, activation)
                (
                    false,
                    :none,
                    false,
                    function (opmode, act, x)
                        return LuxLib.Impl.activation!!(
                            opmode, LuxLib.Utils.False(), act, x
                        )
                    end,
                    opmode,
                    activation,
                    randn(5, 4),
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity],
                ),
            ) do (opmode, activation)
                (
                    false,
                    :none,
                    true,
                    LuxLib.Impl.activation,
                    opmode,
                    activation,
                    randn(5, 4),
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp()],
                    [randn(3), nothing],
                    [Lux.relu, tanh, NNlib.gelu, identity],
                ),
            ) do (opmode, bias, activation)
                cdims = NNlib.DenseConvDims(
                    randn(6, 6, 2, 3),
                    randn(3, 3, 2, 3);
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                )
                (
                    false,
                    :none,
                    false,
                    function (opmode, act, weight, x, bias, cdims)
                        return LuxLib.Impl.fused_conv(opmode, act, weight, x, bias, cdims)
                    end,
                    opmode,
                    activation,
                    randn(3, 3, 2, 3),
                    randn(6, 6, 2, 3),
                    bias === nothing ? nothing : randn(3),
                    cdims,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp()],
                    [Lux.relu, tanh, NNlib.gelu, identity],
                    [true, false],
                ),
            ) do (opmode, activation, affine)
                γ = affine ? randn(1, 2, 2, 1) : nothing
                β = affine ? randn(1, 2, 2, 1) : nothing
                (
                    false,
                    :none,
                    false,
                    function (opmode, act, x, μ, σ², γ, β)
                        return LuxLib.Impl.groupnorm_affine_normalize_internal(
                            opmode, act, x, μ, σ², γ, β, 1e-3
                        )
                    end,
                    opmode,
                    activation,
                    randn(4, 2, 2, 3),
                    randn(1, 1, 2, 3),
                    rand(1, 1, 2, 3) .+ 1.0,
                    γ,
                    β,
                )
            end,
        ),
    )
        mode = Mooncake.ReverseMode
        test_rule(StableRNG(123), fargs...; perf_flag, is_primitive, interface_only, mode)
    end
end
