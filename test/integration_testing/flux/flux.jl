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
const _gpu_disabled = false

# ── GPU AD status notes ──────────────────────────────────────────────────────────────
#
# Without an explicit rule for a GPU broadcast, Mooncake evaluates the fused scalar
# function on forward-mode `NDual{T,N}` values in one GPU kernel. `N` is the number of
# real degrees of freedom per broadcast element: one per real operand and two per complex
# operand. This covers pure element-wise Julia functions, subject to two limitations:
#
#   1. COVERAGE — GPU operations without differentiable Julia IR need explicit rules.
#
#   2. PERFORMANCE — widening each element by `N` partials increases arithmetic,
#      register pressure, and compiled code size. The pullback also accumulates
#      cotangents separately for each differentiable broadcast leaf. An explicit
#      reverse-mode `rrule!!` avoids these costs for common operations.
#
# Models marked _gpu_disabled fall into one or both of the above categories.
# ─────────────────────────────────────────────────────────────────────────────────────

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
    @testset "mooncake gradient (GPU)" begin
        for (gpu_supported, model, x, name) in TEST_MODELS
            gpu_supported || continue  # GPU support not yet implemented
            eltype(x) == Float64 && continue  # Float64 CuArrays not supported
            @testset "grad check $name" begin
                @info "[GPU] testing $name"
                gpu_model = gpu(model)
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
