using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake, Lux, StableRNGs, Test, CUDA, cuDNN
using Mooncake.TestUtils: test_rule

sr(x) = StableRNG(x)

const P = Float32
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
#        • GroupNorm / InstanceNorm — call varm or similar statistics functions;
#          same scalar-closure GPU sum issue as LayerNorm
#        • StatefulRecurrentCell (RNN/LSTM/GRU) — state mutation (setfield! on cell)
#          not yet differentiable on GPU
#        • SkipConnection with vcat — CPU/GPU tangent mismatch in vcat
#      Fix: add an rrule!! or @zero_derivative for the operator (see fill!,
#      unsafe_copyto! in MooncakeCUDAExt.jl for the pattern).
#
#   2. PERFORMANCE — forward-mode broadcast is essentially chunked forward-mode AD:
#      it requires one GPU kernel launch per output DOF.  For models with many
#      parameters this scales as O(params) in memory and time, which is prohibitive
#      for large models even when it compiles.  Fix: add reverse-mode rrule!! so
#      Mooncake runs a single backward pass regardless of parameter count.
#
# Models marked _gpu_disabled fall into one or both of the above categories.
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
    "Maxout($(_model_name(first(values(f.layers)))), $(length(f.layers)))"
end
_model_name(f::SkipConnection) = "SkipConnection($(_model_name(f.layers)), $(f.connection))"
_model_name(f::MultiHeadAttention) = "MultiHeadAttention($(f.q_proj.in_dims))"

# Tuple format: (interface_only, gpu_supported, model, input)
const TEST_MODELS = Any[
    (false, _gpu_enabled, Dense(2, 4), randn(sr(1), P, 2, 3)),
    # tests for https://github.com/chalk-lab/Mooncake.jl/issues/563
    # MultiHeadAttention → LuxLib.batched_matmul → Base.permutedims is not yet
    # differentiable on GPU; requires an explicit rule for permutedims(::CuArray).
    (
        true,
        _gpu_disabled,
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
    (false, _gpu_disabled, SkipConnection(Dense(2 => 2), vcat), randn(sr(14), P, 2, 3)),  # vcat: CPU/GPU tangent mismatch
    (
        false,
        _gpu_enabled,
        ConvTranspose((3, 3), 3 => 2; stride=2),
        rand(sr(15), P, 5, 5, 3, 1),
    ),
    # StatefulRecurrentCell GPU AD is disabled: state mutation (setfield! on cell) is not
    # yet differentiable on GPU.
    (false, _gpu_disabled, StatefulRecurrentCell(RNNCell(3 => 5)), rand(sr(16), P, 3, 2)),
    (
        false,
        _gpu_disabled,
        StatefulRecurrentCell(RNNCell(3 => 5, gelu)),
        rand(sr(17), P, 3, 2),
    ),
    (
        false,
        _gpu_disabled,
        StatefulRecurrentCell(RNNCell(3 => 5, gelu; use_bias=false)),
        rand(sr(18), P, 3, 2),
    ),
    (
        false,
        _gpu_disabled,
        Chain(
            StatefulRecurrentCell(RNNCell(3 => 5)), StatefulRecurrentCell(RNNCell(5 => 3))
        ),
        rand(sr(19), P, 3, 2),
    ),
    (false, _gpu_disabled, StatefulRecurrentCell(LSTMCell(3 => 5)), rand(sr(20), P, 3, 2)),
    (
        false,
        _gpu_disabled,
        Chain(
            StatefulRecurrentCell(LSTMCell(3 => 5)), StatefulRecurrentCell(LSTMCell(5 => 3))
        ),
        rand(sr(21), P, 3, 2),
    ),
    (false, _gpu_disabled, StatefulRecurrentCell(GRUCell(3 => 5)), rand(sr(22), P, 3, 10)),
    (
        false,
        _gpu_disabled,
        Chain(
            StatefulRecurrentCell(GRUCell(3 => 5)), StatefulRecurrentCell(GRUCell(5 => 3))
        ),
        rand(sr(23), P, 3, 10),
    ),
    # BatchNorm GPU AD is disabled: LuxLib's batchnorm_cudnn! is not yet differentiable.
    (true, _gpu_disabled, Chain(Dense(2, 4), BatchNorm(4)), randn(sr(24), P, 2, 3)),
    (true, _gpu_disabled, Chain(Dense(2, 4), BatchNorm(4, gelu)), randn(sr(25), P, 2, 3)),
    (
        true,
        _gpu_disabled,
        Chain(Dense(2, 4), BatchNorm(4, gelu; track_stats=false)),
        randn(sr(26), P, 2, 3),
    ),
    (
        true,
        _gpu_disabled,
        Chain(Conv((3, 3), 2 => 6), BatchNorm(6)),
        randn(sr(27), P, 6, 6, 2, 2),
    ),
    (
        true,
        _gpu_disabled,
        Chain(Conv((3, 3), 2 => 6, tanh), BatchNorm(6)),
        randn(sr(28), P, 6, 6, 2, 2),
    ),
    # GroupNorm GPU AD is disabled: calls varm → sum(centralizedabs2fun(m), x); same
    # scalar-closure GPU sum issue as LayerNorm.
    (
        false,
        _gpu_disabled,
        Chain(Dense(2, 4), GroupNorm(4, 2, gelu)),
        randn(sr(29), P, 2, 3),
    ),
    (false, _gpu_disabled, Chain(Dense(2, 4), GroupNorm(4, 2)), randn(sr(30), P, 2, 3)),
    (
        false,
        _gpu_disabled,
        Chain(Conv((3, 3), 2 => 6), GroupNorm(6, 3)),
        randn(sr(31), P, 6, 6, 2, 2),
    ),
    (
        false,
        _gpu_disabled,
        Chain(Conv((3, 3), 2 => 6, tanh), GroupNorm(6, 3)),
        randn(sr(32), P, 6, 6, 2, 2),
    ),
    # LayerNorm calls varm → sum(centralizedabs2fun(m), x); requires a GPU rrule!! for
    # Statistics.varm and/or other methods before GPU AD can be supported.
    (
        false,
        _gpu_disabled,
        Chain(Conv((3, 3), 2 => 3, gelu), LayerNorm((1, 1, 3))),
        randn(sr(33), P, 4, 4, 2, 2),
    ),
    # InstanceNorm GPU AD is disabled: same varm / scalar-closure GPU sum issue as
    # GroupNorm and LayerNorm.
    (false, _gpu_disabled, InstanceNorm(6), randn(sr(34), P, 6, 6, 2, 2)),
    (
        false,
        _gpu_disabled,
        Chain(Conv((3, 3), 2 => 6), InstanceNorm(6)),
        randn(sr(35), P, 6, 6, 2, 2),
    ),
    (
        false,
        _gpu_disabled,
        Chain(Conv((3, 3), 2 => 6, tanh), InstanceNorm(6)),
        randn(sr(36), P, 6, 6, 2, 2),
    ),
    # From Flux TEST_MODELS: Scale with non-default activation (abs2)
    (false, _gpu_enabled, Scale(4, abs2), randn(sr(37), P, 4, 3)),
    # LayerNorm calls varm → sum(centralizedabs2fun(m), x); requires a GPU rrule!! for
    # Statistics.varm and/or other methods before GPU AD can be supported.
    (false, _gpu_disabled, LayerNorm(2), randn(sr(38), P, 2, 10)),
    # From Flux TEST_MODELS: standalone BatchNorm — same batchnorm_cudnn! issue.
    (true, _gpu_disabled, BatchNorm(2), randn(sr(39), P, 2, 10)),
    # From Flux TEST_MODELS: Float64 parameters and inputs
    (false, _gpu_enabled, Chain(Dense(2, 4), Dense(4, 2)), randn(sr(40), Float64, 2, 1)),
]

@testset "lux" begin
    @testset "$(typeof(f))" for (interface_only, gpu_supported, f, x) in TEST_MODELS
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
        @testset "$(typeof(f))" for (interface_only, gpu_supported, f, x) in TEST_MODELS
            gpu_supported || continue  # GPU support not yet implemented
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
                interface_only,
                unsafe_perturb=true,
                mode=Mooncake.ReverseMode,
            )
        end
    end
end
