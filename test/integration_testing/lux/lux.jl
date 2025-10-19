using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake, Lux, StableRNGs, Test
using Mooncake.TestUtils: test_rule

sr(x) = StableRNG(x)

@testset "lux" begin
    P = Float32
    @testset "$(typeof(f))" for (interface_only, f, x_f32) in Any[
        (false, Dense(2, 4), randn(sr(1), P, 2, 3)),
        # tests for https://github.com/chalk-lab/Mooncake.jl/issues/563
        (
            true,
            MultiHeadAttention(4; attention_dropout_probability=0.1f0),
            randn(sr(1), P, 4, 4, 1),
        ),
        # tests for https://github.com/chalk-lab/Mooncake.jl/issues/622
        (
            true,
            Chain(Dense(1, 10, relu), Dense(10, 10, relu), Dense(10, 1)),
            randn(sr(2), P, 1, 1_000),
        ),
        (false, Dense(2, 4, gelu), randn(sr(2), P, 2, 3)),
        (false, Dense(2, 4, gelu; use_bias=false), randn(sr(3), P, 2, 3)),
        (false, Chain(Dense(2, 4, relu), Dense(4, 3)), randn(sr(4), P, 2, 3)),
        (false, Scale(2), randn(sr(5), P, 2, 3)),
        (false, Conv((3, 3), 2 => 3), randn(sr(6), P, 3, 3, 2, 2)),
        (false, Conv((3, 3), 2 => 3, gelu; pad=SamePad()), randn(sr(7), P, 3, 3, 2, 2)),
        (
            false,
            Conv((3, 3), 2 => 3, relu; use_bias=false, pad=SamePad()),
            randn(sr(8), P, 3, 3, 2, 2),
        ),
        (
            false,
            Chain(Conv((3, 3), 2 => 3, gelu), Conv((3, 3), 3 => 1, gelu)),
            rand(sr(9), P, 5, 5, 2, 2),
        ),
        (
            false,
            Chain(Conv((4, 4), 2 => 2; pad=SamePad()), MeanPool((5, 5); pad=SamePad())),
            rand(sr(10), P, 5, 5, 2, 2),
        ),
        (
            false,
            Chain(Conv((3, 3), 2 => 3, relu; pad=SamePad()), MaxPool((2, 2))),
            rand(sr(11), P, 5, 5, 2, 2),
        ),
        (false, Maxout(() -> Dense(5 => 4, tanh), 3), randn(sr(12), P, 5, 2)),
        (false, Bilinear((2, 2) => 3), randn(sr(13), P, 2, 3)),
        (false, SkipConnection(Dense(2 => 2), vcat), randn(sr(14), P, 2, 3)),
        (false, ConvTranspose((3, 3), 3 => 2; stride=2), rand(sr(15), P, 5, 5, 3, 1)),
        (false, StatefulRecurrentCell(RNNCell(3 => 5)), rand(sr(16), P, 3, 2)),
        (false, StatefulRecurrentCell(RNNCell(3 => 5, gelu)), rand(sr(17), P, 3, 2)),
        (
            false,
            StatefulRecurrentCell(RNNCell(3 => 5, gelu; use_bias=false)),
            rand(sr(18), P, 3, 2),
        ),
        (
            false,
            Chain(
                StatefulRecurrentCell(RNNCell(3 => 5)),
                StatefulRecurrentCell(RNNCell(5 => 3)),
            ),
            rand(sr(19), P, 3, 2),
        ),
        (false, StatefulRecurrentCell(LSTMCell(3 => 5)), rand(sr(20), P, 3, 2)),
        (
            false,
            Chain(
                StatefulRecurrentCell(LSTMCell(3 => 5)),
                StatefulRecurrentCell(LSTMCell(5 => 3)),
            ),
            rand(sr(21), P, 3, 2),
        ),
        (false, StatefulRecurrentCell(GRUCell(3 => 5)), rand(sr(22), P, 3, 10)),
        (
            false,
            Chain(
                StatefulRecurrentCell(GRUCell(3 => 5)),
                StatefulRecurrentCell(GRUCell(5 => 3)),
            ),
            rand(sr(23), P, 3, 10),
        ),
        (true, Chain(Dense(2, 4), BatchNorm(4)), randn(sr(24), P, 2, 3)),
        (true, Chain(Dense(2, 4), BatchNorm(4, gelu)), randn(sr(25), P, 2, 3)),
        (
            true,
            Chain(Dense(2, 4), BatchNorm(4, gelu; track_stats=false)),
            randn(sr(26), P, 2, 3),
        ),
        (true, Chain(Conv((3, 3), 2 => 6), BatchNorm(6)), randn(sr(27), P, 6, 6, 2, 2)),
        (
            true,
            Chain(Conv((3, 3), 2 => 6, tanh), BatchNorm(6)),
            randn(sr(28), P, 6, 6, 2, 2),
        ),
        (false, Chain(Dense(2, 4), GroupNorm(4, 2, gelu)), randn(sr(29), P, 2, 3)),
        (false, Chain(Dense(2, 4), GroupNorm(4, 2)), randn(sr(30), P, 2, 3)),
        (false, Chain(Conv((3, 3), 2 => 6), GroupNorm(6, 3)), randn(sr(31), P, 6, 6, 2, 2)),
        (
            false,
            Chain(Conv((3, 3), 2 => 6, tanh), GroupNorm(6, 3)),
            randn(sr(32), P, 6, 6, 2, 2),
        ),
        (
            false,
            Chain(Conv((3, 3), 2 => 3, gelu), LayerNorm((1, 1, 3))),
            randn(sr(33), P, 4, 4, 2, 2),
        ),
        (false, InstanceNorm(6), randn(sr(34), P, 6, 6, 2, 2)),
        (false, Chain(Conv((3, 3), 2 => 6), InstanceNorm(6)), randn(sr(35), P, 6, 6, 2, 2)),
        (
            false,
            Chain(Conv((3, 3), 2 => 6, tanh), InstanceNorm(6)),
            randn(sr(36), P, 6, 6, 2, 2),
        ),
    ]
        @info "$(typeof((f, x_f32...)))"
        rng = sr(123546)
        ps, st = f32(Lux.setup(rng, f))
        mode = Mooncake.ReverseMode
        fargs = (f, f32(x_f32), ps, st)
        test_rule(
            rng, fargs...; is_primitive=false, interface_only, unsafe_perturb=true, mode
        )
    end
end
