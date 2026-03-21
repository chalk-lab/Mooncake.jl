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
# When Mooncake lacks an explicit rule for a GPU operation it falls back to
# differentiating through the Julia source via a forward-mode (chunked) broadcast
# using NDual{T,N} dual numbers inside GPU kernels.  N = total real DOFs across all
# broadcast inputs (1 per Float arg, 2 per Complex arg).  This works for pure
# element-wise Julia functions but has two important limitations:
#
#   1. CORRECTNESS — some GPU operations are not differentiable by Mooncake without
#      explicit rules:
#        • cuDNN operators (batchnorm_cudnn!, …) — need an rrule!!
#        • Base.permutedims(::CuArray) — called by LuxLib.batched_matmul in the
#          MultiHeadAttention path; needs an explicit rule
#      Fix: add an rrule!! or @zero_derivative for the operator (see fill!,
#      unsafe_copyto! in MooncakeCUDAExt.jl for the pattern).
#
#   2. PERFORMANCE — forward-mode broadcast is essentially chunked forward-mode AD:
#      it requires one GPU kernel launch per output DOF.  For models with many
#      parameters this scales as O(params) in memory and time, which is prohibitive
#      for large models even when it compiles.  Fix: add reverse-mode rrule!! so
#      Mooncake runs a single backward pass regardless of parameter count.
#
#   3. CPU/GPU TANGENT MISMATCH (Flux-specific) — Flux stores weights directly as
#      struct type parameters, e.g. Dense{identity, Matrix{Float32}, Vector{Float32}}.
#      gpu() replaces the runtime values but the static type params remain Matrix{Float32}.
#      Mooncake computes tangent_type from static types, so weight tangents are
#      Matrix{Float32} (CPU).  During test_rule the perturbed primal is reconstructed
#      as (primal + tangent), giving a Dense with a CPU weight matrix that is then
#      called on a GPU input:
#
#        Dense(gpu_x)
#          → weight * gpu_x                               ← Matrix{Float32} × CuArray
#            → BLAS.gemm!(A::Matrix{Float32}, B::CuArray) ← mixed CPU/GPU
#              → unsafe_convert(Ptr{Float32}, CuArray)    ← ILLEGAL: DeviceMemory has no CPU ptr
#
#      Lux avoids this because parameters live in a separate `ps` NamedTuple that is
#      explicitly moved to the GPU, so tangent_type(CuArray) = CuArray fires correctly.
#      Fix: Mooncake would need a Flux-aware rule for Dense/MHA that keeps tangents on GPU,
#      or Flux would need to update struct type params on gpu() (an Adapt.jl issue).
#
# Models marked _gpu_disabled fall into one or more of the above categories.
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
        _gpu_disabled,
        SkipConnection(Dense(2 => 2), vcat),
        randn(Float32, 2, 3),
        "SkipConnection(Dense(2 => 2), vcat)",
    ),  # vcat tangent mixes CPU/GPU: Illegal conversion of CUDA.DeviceMemory to Ptr{Float32}
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
    # LayerNorm calls varm → sum(centralizedabs2fun(m), x); requires a GPU rrule!! for
    # Statistics.varm and/or other methods before GPU AD can be supported.
    (_gpu_disabled, LayerNorm(2), randn(Float32, 2, 10), "LayerNorm(2)"),
    (_gpu_disabled, BatchNorm(2), randn(Float32, 2, 10), "BatchNorm(2)"),  # batchnorm_cudnn! not yet differentiable (category 1)
    (
        _gpu_disabled,
        first ∘ MultiHeadAttention(16),
        randn32(16, 20, 2),
        "MultiHeadAttention(16)",
    ),  # MultiHeadAttention → LuxLib.batched_matmul → Base.permutedims not yet differentiable (category 1)
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
