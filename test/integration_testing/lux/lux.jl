include(joinpath(@__DIR__, "..", "..", "ext", "pin_develop_or_skip.jl"))
pin_develop_or_skip(@__DIR__, "Lux")

# Every `test_rule` below pins `mode=ReverseMode`; forward mode is not covered. Whether it
# now works is unverified, and cannot be checked here while Lux caps Mooncake at 0.5 and the
# line above skips the file.

using Mooncake, Lux, StableRNGs, Test, CUDA, cuDNN
using Mooncake.TestUtils: test_rule

sr(x) = StableRNG(x)

const P = Float32
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
# Models marked `:interface` skip comparison against a finite-difference reference because
# its perturbations can leave the primal domain. Models marked _gpu_disabled fall into one
# or both of the above categories.
# ─────────────────────────────────────────────────────────────────────────────────────

function _model_name(f)
    r = repr(f)
    return (occursin('\n', r) || length(r) > 80) ? string(nameof(typeof(f))) : r
end
function _model_name(f::Chain)
    return "Chain(" * join([_model_name(l) for l in values(f.layers)], ", ") * ")"
end
_model_name(f::StatefulRecurrentCell) = "StatefulRecurrentCell($(_model_name(f.cell)))"
function _model_name(f::Maxout)
    return "Maxout($(_model_name(first(values(f.layers)))), $(length(f.layers)))"
end
_model_name(f::SkipConnection) = "SkipConnection($(_model_name(f.layers)), $(f.connection))"
_model_name(f::MultiHeadAttention) = "MultiHeadAttention($(f.q_proj.in_dims))"

# Tuple format: (interface_only, gpu_status, model, input)
const TEST_MODELS = Any[
    (false, _gpu_enabled, Dense(2, 4), randn(sr(1), P, 2, 3)),
    # tests for https://github.com/chalk-lab/Mooncake.jl/issues/563
    # MHA needs MooncakeCUDAExt's Base.permutedims(::CuArray) rule (LuxLib.batched_matmul).
    (
        true,
        _gpu_enabled,
        MultiHeadAttention(4; attention_dropout_probability=0.1f0),
        randn(sr(1), P, 4, 4, 1),
    ),
    # tests for https://github.com/chalk-lab/Mooncake.jl/issues/622
    (
        true,
        _gpu_enabled,
        Chain(Dense(1, 10, relu), Dense(10, 10, relu), Dense(10, 1)),
        randn(sr(2), P, 1, 1_000),
    ),
    (false, _gpu_enabled, Dense(2, 4, gelu), randn(sr(2), P, 2, 3)),
    (false, _gpu_enabled, Dense(2, 4, gelu; use_bias=false), randn(sr(3), P, 2, 3)),
    (false, _gpu_enabled, Chain(Dense(2, 4, relu), Dense(4, 3)), randn(sr(4), P, 2, 3)),
    (false, _gpu_enabled, Scale(2), randn(sr(5), P, 2, 3)),
    (false, _gpu_enabled, Conv((3, 3), 2 => 3), randn(sr(6), P, 3, 3, 2, 2)),
    (
        false,
        _gpu_enabled,
        Conv((3, 3), 2 => 3, gelu; pad=SamePad()),
        randn(sr(7), P, 3, 3, 2, 2),
    ),
    (
        false,
        _gpu_enabled,
        Conv((3, 3), 2 => 3, relu; use_bias=false, pad=SamePad()),
        randn(sr(8), P, 3, 3, 2, 2),
    ),
    (
        false,
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 3, gelu), Conv((3, 3), 3 => 1, gelu)),
        rand(sr(9), P, 5, 5, 2, 2),
    ),
    (
        false,
        _gpu_enabled,
        Chain(Conv((4, 4), 2 => 2; pad=SamePad()), MeanPool((5, 5); pad=SamePad())),
        rand(sr(10), P, 5, 5, 2, 2),
    ),
    (
        false,
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 3, relu; pad=SamePad()), MaxPool((2, 2))),
        rand(sr(11), P, 5, 5, 2, 2),
    ),
    (false, _gpu_enabled, Maxout(() -> Dense(5 => 4, tanh), 3), randn(sr(12), P, 5, 2)),
    (false, _gpu_enabled, Bilinear((2, 2) => 3), randn(sr(13), P, 2, 3)),
    (false, _gpu_enabled, SkipConnection(Dense(2 => 2), vcat), randn(sr(14), P, 2, 3)),
    (
        false,
        _gpu_enabled,
        ConvTranspose((3, 3), 3 => 2; stride=2),
        rand(sr(15), P, 5, 5, 3, 1),
    ),
    (false, _gpu_enabled, StatefulRecurrentCell(RNNCell(3 => 5)), rand(sr(16), P, 3, 2)),
    (
        false,
        _gpu_enabled,
        StatefulRecurrentCell(RNNCell(3 => 5, gelu)),
        rand(sr(17), P, 3, 2),
    ),
    (
        false,
        _gpu_enabled,
        StatefulRecurrentCell(RNNCell(3 => 5, gelu; use_bias=false)),
        rand(sr(18), P, 3, 2),
    ),
    (
        false,
        _gpu_enabled,
        Chain(
            StatefulRecurrentCell(RNNCell(3 => 5)), StatefulRecurrentCell(RNNCell(5 => 3))
        ),
        rand(sr(19), P, 3, 2),
    ),
    (false, _gpu_enabled, StatefulRecurrentCell(LSTMCell(3 => 5)), rand(sr(20), P, 3, 2)),
    (
        false,
        _gpu_enabled,
        Chain(
            StatefulRecurrentCell(LSTMCell(3 => 5)), StatefulRecurrentCell(LSTMCell(5 => 3))
        ),
        rand(sr(21), P, 3, 2),
    ),
    (false, _gpu_enabled, StatefulRecurrentCell(GRUCell(3 => 5)), rand(sr(22), P, 3, 10)),
    (
        false,
        _gpu_enabled,
        Chain(
            StatefulRecurrentCell(GRUCell(3 => 5)), StatefulRecurrentCell(GRUCell(5 => 3))
        ),
        rand(sr(23), P, 3, 10),
    ),
    (true, _gpu_enabled, Chain(Dense(2, 4), BatchNorm(4)), randn(sr(24), P, 2, 3)),
    (true, _gpu_enabled, Chain(Dense(2, 4), BatchNorm(4, gelu)), randn(sr(25), P, 2, 3)),
    (
        true,
        _gpu_enabled,
        Chain(Dense(2, 4), BatchNorm(4, gelu; track_stats=false)),
        randn(sr(26), P, 2, 3),
    ),
    (
        true,
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 6), BatchNorm(6)),
        randn(sr(27), P, 6, 6, 2, 2),
    ),
    (
        true,
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 6, tanh), BatchNorm(6)),
        randn(sr(28), P, 6, 6, 2, 2),
    ),
    # Finite differences can perturb GroupNorm's positive epsilon outside its domain.
    (false, :interface, Chain(Dense(2, 4), GroupNorm(4, 2, gelu)), randn(sr(29), P, 2, 3)),
    (false, :interface, Chain(Dense(2, 4), GroupNorm(4, 2)), randn(sr(30), P, 2, 3)),
    (
        false,
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 6), GroupNorm(6, 3)),
        randn(sr(31), P, 6, 6, 2, 2),
    ),
    (
        false,
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 6, tanh), GroupNorm(6, 3)),
        randn(sr(32), P, 6, 6, 2, 2),
    ),
    (
        false,
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 3, gelu), LayerNorm((1, 1, 3))),
        randn(sr(33), P, 4, 4, 2, 2),
    ),
    (false, _gpu_enabled, InstanceNorm(6), randn(sr(34), P, 6, 6, 2, 2)),
    (
        false,
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 6), InstanceNorm(6)),
        randn(sr(35), P, 6, 6, 2, 2),
    ),
    (
        false,
        _gpu_enabled,
        Chain(Conv((3, 3), 2 => 6, tanh), InstanceNorm(6)),
        randn(sr(36), P, 6, 6, 2, 2),
    ),
    # From Flux TEST_MODELS: Scale with non-default activation (abs2)
    (false, _gpu_enabled, Scale(4, abs2), randn(sr(37), P, 4, 3)),
    (false, _gpu_enabled, LayerNorm(2), randn(sr(38), P, 2, 10)),
    (true, _gpu_enabled, BatchNorm(2), randn(sr(39), P, 2, 10)),
    # From Flux TEST_MODELS: Float64 parameters and inputs
    (false, _gpu_enabled, Chain(Dense(2, 4), Dense(4, 2)), randn(sr(40), Float64, 2, 1)),
]

@testset "lux" begin
    @testset "$(_model_name(f))" for (interface_only, gpu_status, f, x) in TEST_MODELS
        @info "[CPU] testing $(_model_name(f))"
        rng = sr(123546)
        cvt = eltype(x) == Float64 ? f64 : f32
        ps, st = cvt(Lux.setup(rng, f))
        test_rule(
            rng,
            f,
            x,
            ps,
            st;
            is_primitive=false,
            interface_only,
            unsafe_perturb=true,
            mode=Mooncake.ReverseMode,
        )
    end
end

if CUDA.functional()
    dev = gpu_device()
    @testset "lux (GPU)" begin
        @testset "$(_model_name(f))" for (interface_only, gpu_status, f, x) in TEST_MODELS
            gpu_status === _gpu_disabled && continue
            eltype(x) == Float64 && continue  # Float64 CuArrays not supported
            @info "[GPU] testing $(_model_name(f))"
            rng = sr(123546)
            cvt = f32  # Float64 inputs are skipped above; all GPU tests run as Float32
            ps, st = dev(cvt(Lux.setup(rng, f)))
            gpu_x = dev(x)
            test_rule(
                rng,
                f,
                gpu_x,
                ps,
                st;
                is_primitive=false,
                interface_only=(interface_only || gpu_status === :interface),
                unsafe_perturb=true,
                mode=Mooncake.ReverseMode,
            )
        end
    end
end
