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
