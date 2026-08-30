using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Test
using Bijectors, CUDA, cuDNN, Flux, Mooncake, StableRNGs

# Regression test for https://github.com/chalk-lab/Mooncake.jl/issues/661

inputdim = 4
mask = Bijectors.PartitionMask(inputdim, 1:2:inputdim)
cdim = length(1:2:inputdim)
x = randn(inputdim)
t_net = f64(Chain(Dense(cdim, 16, leakyrelu), Dense(16, 16, leakyrelu), Dense(16, cdim)))
ps, st = Optimisers.destructure(t_net)

function loss(ps, st, x, mask)
    t_net = st(ps)
    x₁, x₂, x₃ = Bijectors.partition(mask, x)
    y₁ = x₁ .+ t_net(x₂)
    y = Bijectors.combine(mask, y₁, x₂, x₃)
    return sum(abs2, y)
end

struct ACL
    mask::Bijectors.PartitionMask
    t::Flux.Chain
end
Flux.@functor ACL (t,)

psacl, stacl = Optimisers.destructure(ACL(mask, t_net))

function loss_acl(ps, st, x)
    acl = st(ps)
    x₁, x₂, x₃ = Bijectors.partition(acl.mask, x)
    y₁ = x₁ .+ acl.t(x₂)
    y = Bijectors.combine(acl.mask, y₁, x₂, x₃)
    return sum(abs2, y)
end

test_cases = Any[(loss, ps, st, x, mask), (loss_acl, psacl, stacl, x)]

@testset "bijectors regression #661" for (f, args...) in test_cases
    Mooncake.TestUtils.test_rule(
        StableRNG(1),
        f,
        args...;
        is_primitive=false,
        interface_only=true,
        unsafe_perturb=true,
        mode=Mooncake.ReverseMode,
    )
end

#
# Tests from https://github.com/FluxML/Flux.jl/blob/d15c7dc54f080dd67193e8228329d6d127952b81/test/ext_mooncake.jl
# TEST_MODELS inlined from https://github.com/FluxML/Flux.jl/blob/master/test/test_utils.jl
# to avoid a runtime dependency on Flux's internal test files.
#

const _gpu_enabled = true

# Tuple format: (gpu_supported, model, input, name)
const TEST_MODELS = [
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
    (_gpu_enabled, BatchNorm(2), randn(Float32, 2, 10), "BatchNorm(2)"),  # NNlib.batchnorm rrule!! (NNlibMooncakeCUDAExt, FluxML/NNlib.jl#727)
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

primitive_cases = (
    (Flux.Scale([1.0f0 2 3 4], true), randn(Float32, 2, 1, 3)),
    (Flux.Scale([1.0f0 2 3 4], true, abs2), randn(Float32, 2, 1, 3)),
    (LayerNorm((2, 4)), randn(Float32, 2, 4, 3)),
    (Dense(2 => 4), randn(Float32, 2, 3)),
    (Dense(2 => 4), randn(Float32, 2, 3, 2)),
    (Dense(2 => 4, tanh), randn(Float32, 2, 3)),
    (Conv((3,), 2 => 3), randn(Float32, 5, 2, 2)),
    (Conv((3, 3), 2 => 3, tanh), randn(Float32, 5, 5, 2, 2)),
    (
        Conv((2, 3), 4 => 6; groups=2, stride=(2, 1), pad=(1, 0, 2, 1), dilation=(1, 2)),
        randn(Float32, 8, 9, 4, 2),
    ),
    (ConvTranspose((3,), 3 => 2; stride=2), randn(Float32, 5, 3, 2)),
    (
        ConvTranspose(
            (2, 3),
            4 => 6;
            groups=2,
            stride=(2, 1),
            pad=(1, 0, 2, 1),
            outpad=(1, 0),
            dilation=(1, 2),
        ),
        randn(Float32, 8, 9, 4, 2),
    ),
    (MeanPool((3,); pad=SamePad()), randn(Float32, 5, 2, 2)),
)

@testset "Flux layer rules" for (layer, x) in primitive_cases
    Mooncake.TestUtils.test_rule(
        StableRNG(123), layer, x; is_primitive=true, unsafe_perturb=true
    )
end

@testset "saturated tanh layer rule" begin
    layer = Dense(zeros(Float32, 4, 2), fill(20.0f0, 4), tanh)
    x = zeros(Float32, 2, 3)
    ref = 4 * exp(-40.0f0) / (1 + exp(-40.0f0))^2

    layer_dual = Mooncake.zero_dual(layer)
    fill!(Mooncake._fields(Mooncake.tangent(layer_dual)).bias, 1)
    out = Mooncake.frule!!(layer_dual, Mooncake.zero_dual(x))
    @test all(≈(ref), Mooncake.tangent(out))

    layer_codual = Mooncake.zero_fcodual(layer)
    out, pullback = Mooncake.rrule!!(layer_codual, Mooncake.zero_fcodual(x))
    fill!(Mooncake.tangent(out), 1)
    pullback(Mooncake.NoRData())
    @test all(≈(size(x, 2) * ref), Mooncake._fields(Mooncake.tangent(layer_codual)).bias)
end

mse_inputs = (randn(Float32, 2, 3), randn(Float32, 2, 3))
@testset "Flux.Losses.mse" begin
    Mooncake.TestUtils.test_rule(
        StableRNG(123),
        Flux.Losses.mse,
        mse_inputs...;
        is_primitive=true,
        unsafe_perturb=true,
    )
end

# We only check that the gradient runs (interface_only=true), not correctness
# against a reference. Correctness is tested separately in Flux's own test suite.
@testset "mooncake gradient" begin
    for (gpu_supported, model, x, name) in TEST_MODELS
        @testset "grad check $name" begin
            @info "[CPU] testing $name"
            Mooncake.TestUtils.test_rule(
                StableRNG(123),
                m -> sum(abs2, m(x)),
                model;
                is_primitive=false,
                interface_only=true,
                unsafe_perturb=true,
                mode=Mooncake.ReverseMode,
            )
        end
    end
end

if CUDA.functional()
    @testset "Flux layer rules (GPU)" for (layer, x) in primitive_cases
        Mooncake.TestUtils.test_rule(
            StableRNG(123), gpu(layer), cu(x); is_primitive=true, unsafe_perturb=true
        )
    end

    @testset "Flux.Losses.mse (GPU)" begin
        Mooncake.TestUtils.test_rule(
            StableRNG(123),
            Flux.Losses.mse,
            cu.(mse_inputs)...;
            is_primitive=true,
            unsafe_perturb=true,
        )
    end

    @testset "mooncake gradient (GPU)" begin
        for (gpu_supported, model, x, name) in TEST_MODELS
            gpu_supported || continue  # GPU support not yet implemented
            @testset "grad check $name" begin
                @info "[GPU] testing $name"
                gpu_model = gpu(model)
                # `gpu` and `cu` convert floating-point arrays to `Float32`.
                gpu_x = cu(x)
                Mooncake.TestUtils.test_rule(
                    StableRNG(123),
                    m -> sum(abs2, m(gpu_x)),
                    gpu_model;
                    is_primitive=false,
                    interface_only=true,
                    unsafe_perturb=true,
                    mode=Mooncake.ReverseMode,
                )
            end
        end
    end
end
