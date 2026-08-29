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

include("models.jl")

# ── GPU AD status notes ──────────────────────────────────────────────────────────────
#
# When Mooncake lacks an explicit rule for a GPU operation, it falls back to
# differentiating CUDA kernel via a forward-mode (chunked) broadcast
# using NDual{T,N} dual numbers inside GPU kernels.  N = total real DOFs across all
# broadcast inputs (1 per Float arg, 2 per Complex arg).  This works for pure
# element-wise Julia functions, but has two important limitations:
#
#   1. COVERAGE — some GPU operations are not differentiable by Mooncake without
#      explicit rules:
#        • cuDNN operators (batchnorm_cudnn!, …) — need an rrule!!
#        • Base.permutedims(::CuArray) — called by LuxLib.batched_matmul in the
#          MultiHeadAttention path; needs an explicit rule
#      Fix: add an rrule!! or @zero_derivative for the operator (see fill!,
#      unsafe_copyto! in MooncakeCUDAExt.jl for the pattern).
#
#   2. PERFORMANCE — forward-mode broadcast is essentially chunked forward-mode AD:
#      it requires one GPU kernel launch per output DOF.  For models with many
#      parameters, this scales as O(params) in memory and time, which is prohibitive
#      for large models even when it compiles.  Fix: add reverse-mode rrule!! so
#      Mooncake runs a single backward pass regardless of parameter count.
#
#   3. CPU/GPU TANGENT MISMATCH (Flux-specific) — Flux stores weights directly as
#      struct type parameters, e.g. Dense{identity, Matrix{Float32}, Vector{Float32}}.
#      gpu() replaces the runtime values, but the static type params remain Matrix{Float32}.
#      Mooncake computes tangent_type from static types, so weight tangents are
#      Matrix{Float32} (CPU).  During test_rule, the perturbed primal is reconstructed
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
# ─────────────────────────────────────────────────────────────────────────────────────

# We only check that the gradient runs (interface_only=true), not correctness
# against a reference. Correctness is tested separately in Flux's own test suite.
@testset "mooncake gradient" begin
    for (gpu_supported, model, x, name) in FLUX_MODELS
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
        for (gpu_supported, model, x, name) in FLUX_MODELS
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
