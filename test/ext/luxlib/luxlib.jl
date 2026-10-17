include(joinpath(@__DIR__, "..", "pin_develop_or_skip.jl"))
pin_develop_or_skip(@__DIR__, "LuxLib")

using JET, Lux, LuxLib, Mooncake, NNlib, SLEEFPirates, StableRNGs, Test
using LuxLib.Impl: sleefpirates_fast_act
using Mooncake.TestUtils: TestCase, test_rule

# Custom activation to exercise fallback paths (no pre-defined rrule, needs intermediate).
_custom_act(x) = x^2 + 1

# Access AD helper functions present in the Extension module.
const MooncakeLuxLibExt = Base.get_extension(Mooncake, :MooncakeLuxLibExt)
@assert !isnothing(MooncakeLuxLibExt) "MooncakeLuxLibExt is required for testing !"

@testset "luxlib" begin
    # This suite overrides the mode at construction: these cases cover reverse rules.
    luxlib_case(f, args...; kw...) = TestCase(f, args...; kw..., mode=Mooncake.ReverseMode)
    test_cases = vcat(
        TestCase[
            luxlib_case(LuxLib.Impl.matmul, randn(5, 4), randn(4, 3)),
            luxlib_case(LuxLib.Impl.matmuladd, randn(5, 4), randn(4, 3), randn(5)),
            luxlib_case(
                LuxLib.Impl.batched_matmul_fallback, randn(5, 4, 3), randn(4, 3, 3)
            ),
            luxlib_case(LuxLib.Impl.activation, Lux.relu, randn(5, 4); is_primitive=false),
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
            return luxlib_case(
                sleefpirates_fast_act(f), randn(); perf_flag=:stability_and_allocs
            )
        end,
        TestCase[
            luxlib_case(
                LuxLib.Utils.static_training_mode_check,
                nothing,
                LuxLib.Utils.True(),
                LuxLib.Utils.True();
                perf_flag=:stability_and_allocs,
            ),
            luxlib_case(
                LuxLib.Impl.dropout_shape, randn(4, 4), :; perf_flag=:stability_and_allocs
            ),
            luxlib_case(
                LuxLib.Impl.dropout_fptype,
                randn(Float32, 4, 4);
                perf_flag=:stability_and_allocs,
            ),
            luxlib_case(
                LuxLib.Impl.check_dropout_mask_shape_mismatch,
                randn(4, 4),
                randn(4, 4),
                :;
                perf_flag=:stability_and_allocs,
            ),
            luxlib_case(
                LuxLib.Impl.generate_dropout_mask,
                StableRNG(123),
                randn(Float32, 4, 4),
                0.5f0,
                2.0f0,
                :;
                interface_only=true,
                perf_flag=:stability,
            ),
            luxlib_case(
                LuxLib.Impl.generate_alpha_dropout_noise,
                StableRNG(123),
                randn(Float32, 4, 4);
                perf_flag=:stability,
            ),
            luxlib_case(
                LuxLib.Impl.batchnorm_reduce_dims,
                randn(5, 4, 3);
                perf_flag=:stability_and_allocs,
            ),
            luxlib_case(
                LuxLib.Impl.get_batchnorm_statistics,
                randn(5, 4, 3),
                randn(4),
                randn(4),
                LuxLib.Utils.True();
                interface_only=true,
                perf_flag=:stability,
            ),
            luxlib_case(
                LuxLib.Impl.update_running_statistics,
                randn(4),
                randn(4),
                randn(4),
                randn(4),
                0.9,
                0.1;
                interface_only=true,
                perf_flag=:stability,
            ),
            luxlib_case(
                LuxLib.Impl.update_normalization_statistics,
                randn(5, 4, 3),
                zeros(1, 4, 1),
                zeros(1, 4, 1),
                zeros(1, 4, 1),
                ones(1, 4, 1),
                0.1,
                (Val(1), Val(3));
                interface_only=true,
                perf_flag=:stability,
            ),
            luxlib_case(
                LuxLib.Impl.groupnorm_reduce_dims,
                randn(4, 4, 2);
                perf_flag=:stability_and_allocs,
            ),
            luxlib_case(
                LuxLib.Impl.instancenorm_reduce_dims,
                randn(5, 4, 3);
                perf_flag=:stability_and_allocs,
            ),
            luxlib_case(
                LuxLib.Impl.compute_layernorm_dims,
                randn(4, 3),
                randn(5, 4, 1),
                randn(4, 1),
                nothing;
                perf_flag=:stability_and_allocs,
            ),
            luxlib_case(
                LuxLib.Impl.get_norm_reshape_dims,
                (4, 4, 2),
                4;
                perf_flag=:stability_and_allocs,
            ),
            luxlib_case(
                LuxLib.Impl.flattened_bias_dims,
                randn(5, 4);
                perf_flag=:stability_and_allocs,
            ),
            luxlib_case(LuxLib.Impl.get_non_heads_dim, 3, 1; perf_flag=:stability),
            luxlib_case(
                LuxLib.Impl.make_causal_mask, randn(4, 4), 4, 4; perf_flag=:stability
            ),
            luxlib_case(
                LuxLib.Impl.get_non_contracting_dim, 3, 1, (2,); perf_flag=:stability
            ),
            luxlib_case(
                LuxLib.Impl.get_batched_matmul_repeat_dims,
                randn(5, 4, 3),
                randn(4, 3, 3),
                (3,),
                (3,);
                perf_flag=:stability,
            ),
        ],
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp()], [(nothing, nothing), (randn(4), randn(4))]
                ),
            ) do (opmode, (gamma, beta))
                luxlib_case(
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
                    beta;
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp()],
                    [(nothing, nothing), (randn(4), randn(4))],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, (gamma, beta), activation)
                luxlib_case(
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
                    beta;
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [randn(5), nothing],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, bias, activation)
                luxlib_case(
                    LuxLib.Impl.fused_dense,
                    opmode,
                    activation,
                    randn(5, 4),
                    randn(4, 2),
                    bias;
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, activation)
                luxlib_case(
                    function (opmode, act, x, bias)
                        return LuxLib.Impl.bias_activation(opmode, act, x, bias)
                    end,
                    opmode,
                    activation,
                    randn(5, 4),
                    randn(5);
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, activation)
                luxlib_case(
                    function (opmode, act, x, bias)
                        return LuxLib.Impl.bias_activation!!(
                            opmode, LuxLib.Utils.True(), act, x, bias
                        )
                    end,
                    opmode,
                    activation,
                    randn(5, 4),
                    randn(5);
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, activation)
                luxlib_case(
                    function (opmode, act, x, bias)
                        return LuxLib.Impl.bias_activation!!(
                            opmode, LuxLib.Utils.False(), act, x, bias
                        )
                    end,
                    opmode,
                    activation,
                    randn(5, 4),
                    randn(5);
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, activation)
                luxlib_case(
                    function (opmode, act, x)
                        return LuxLib.Impl.activation!!(
                            opmode, LuxLib.Utils.True(), act, x
                        )
                    end,
                    opmode,
                    activation,
                    randn(5, 4);
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, activation)
                luxlib_case(
                    function (opmode, act, x)
                        return LuxLib.Impl.activation!!(
                            opmode, LuxLib.Utils.False(), act, x
                        )
                    end,
                    opmode,
                    activation,
                    randn(5, 4);
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, activation)
                luxlib_case(
                    LuxLib.Impl.activation,
                    opmode,
                    activation,
                    randn(5, 4);
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [randn(3), nothing],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, bias, activation)
                cdims = NNlib.DenseConvDims(
                    randn(6, 6, 2, 3),
                    randn(3, 3, 2, 3);
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                )
                luxlib_case(
                    function (opmode, act, weight, x, bias, cdims)
                        return LuxLib.Impl.fused_conv(opmode, act, weight, x, bias, cdims)
                    end,
                    opmode,
                    activation,
                    randn(3, 3, 2, 3),
                    randn(6, 6, 2, 3),
                    bias === nothing ? nothing : randn(3),
                    cdims;
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp{Lux.CPUDevice()}()],
                    [randn(5), nothing],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                ),
            ) do (opmode, bias, activation)
                luxlib_case(
                    LuxLib.Impl.fused_dense,
                    opmode,
                    activation,
                    randn(5, 4),
                    randn(4, 2),
                    bias;
                    is_primitive=false,
                )
            end,
        ),
        vec(
            map(
                Iterators.product(
                    [LuxLib.LoopedArrayOp()],
                    [Lux.relu, tanh, NNlib.gelu, identity, _custom_act],
                    [true, false],
                ),
            ) do (opmode, activation, affine)
                γ = affine ? randn(1, 2, 2, 1) : nothing
                β = affine ? randn(1, 2, 2, 1) : nothing
                luxlib_case(
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
                    β;
                    is_primitive=false,
                )
            end,
        ),
    )
    for (tc, name) in zip(test_cases, Mooncake.TestUtils._test_case_names(test_cases))
        test_rule(StableRNG(123), tc; name)
    end
end
