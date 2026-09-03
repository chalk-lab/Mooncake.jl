# Tests from
# https://github.com/FluxML/Flux.jl/blob/
# d15c7dc54f080dd67193e8228329d6d127952b81/test/ext_mooncake.jl
# FLUX_MODELS is inlined from Flux's test/test_utils.jl so that the integration test and
# benchmark share one model catalog without depending on Flux's internal test files.

const _gpu_enabled = true
const _gpu_disabled = false

# Tuple format: (gpu_supported, model, input, name)
const FLUX_MODELS = [
    (_gpu_enabled, Dense(2 => 4), randn(Float32, 2), "Dense(2 => 4)"),
    (
        _gpu_enabled,
        Chain(Dense(2 => 4, tanh), Dense(4 => 3)),
        randn(Float32, 2),
        "Chain(Dense(2 => 4, tanh), Dense(4 => 3))",
    ),
    (
        _gpu_enabled,
        f64(Chain(Dense(2 => 4), Dense(4 => 2))),
        randn(Float64, 2, 1),
        "f64(Chain(Dense(2 => 4), Dense(4 => 2)))",
    ),
    (
        _gpu_enabled,
        Flux.Scale([1.0f0 2.0f0 3.0f0 4.0f0], true, abs2),
        randn(Float32, 2),
        "Flux.Scale(4, abs2)",
    ),
    (
        _gpu_enabled,
        Conv((3, 3), 2 => 3),
        randn(Float32, 3, 3, 2, 1),
        "Conv((3, 3), 2 => 3)",
    ),
    (
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 3), Conv((3, 3), 3 => 1, tanh)),
        rand(Float32, 5, 5, 2, 1),
        "Chain(Conv((3, 3), 2 => 3), Conv((3, 3), 3 => 1, tanh))",
    ),
    (
        _gpu_enabled,
        Chain(Conv((4, 4), 2 => 2; pad=SamePad()), MeanPool((5, 5); pad=SamePad())),
        rand(Float32, 5, 5, 2, 2),
        "Chain(Conv((4, 4), 2 => 2), MeanPool((5, 5)))",
    ),
    (
        _gpu_enabled,
        Maxout(() -> Dense(5 => 4, tanh), 3),
        randn(Float32, 5, 1),
        "Maxout(Dense(5 => 4, tanh), 3)",
    ),
    (
        _gpu_enabled,
        SkipConnection(Dense(2 => 2), vcat),
        randn(Float32, 2, 3),
        "SkipConnection(Dense(2 => 2), vcat)",
    ),
    (
        _gpu_enabled,
        Flux.Bilinear((2, 2) => 3),
        randn(Float32, 2, 1),
        "Bilinear((2, 2) => 3)",
    ),
    (
        _gpu_enabled,
        ConvTranspose((3, 3), 3 => 2; stride=2),
        rand(Float32, 5, 5, 3, 1),
        "ConvTranspose((3, 3), 3 => 2)",
    ),
    # LayerNorm needs MooncakeCUDAExt's Statistics.varm GPU rrule!! (via LuxLib mean_var).
    (_gpu_enabled, LayerNorm(2), randn(Float32, 2, 10), "LayerNorm(2)"),
    # Uses NNlibMooncakeCUDAExt's batchnorm rrule!! (FluxML/NNlib.jl#727).
    (_gpu_enabled, BatchNorm(2), randn(Float32, 2, 10), "BatchNorm(2)"),
    (
        _gpu_enabled,
        first ∘ MultiHeadAttention(16),
        randn32(16, 20, 2),
        "MultiHeadAttention(16)",
    ),
    (_gpu_enabled, RNN(3 => 2), randn(Float32, 3, 2), "RNN(3 => 2)"),
    (_gpu_enabled, LSTM(3 => 5), randn(Float32, 3, 2), "LSTM(3 => 5)"),
    (_gpu_enabled, GRU(3 => 5), randn(Float32, 3, 10), "GRU(3 => 5)"),
    (
        _gpu_enabled,
        Chain(RNN(3 => 4), RNN(4 => 3)),
        randn(Float32, 3, 2),
        "Chain(RNN(3 => 4), RNN(4 => 3))",
    ),
    (
        _gpu_enabled,
        Chain(LSTM(3 => 5), LSTM(5 => 3)),
        randn(Float32, 3, 2),
        "Chain(LSTM(3 => 5), LSTM(5 => 3))",
    ),
]
