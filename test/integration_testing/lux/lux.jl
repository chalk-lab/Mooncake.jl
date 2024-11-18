using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake, Lux, StableRNGs, Test
using Mooncake.TestUtils: test_rule

@testset "lux" begin
    @testset "$(typeof(f))" for (f, x_f32) in Any[
        (Dense(2, 4), randn(Float32, 2, 3)),
        (Dense(2, 4, gelu), randn(Float32, 2, 3)),
        (Dense(2, 4, gelu; use_bias=false), randn(Float32, 2, 3)),
        (Chain(Dense(2, 4, relu), Dense(4, 3)), randn(Float32, 2, 3)),
        (Scale(2), randn(Float32, 2, 3)),
        (Conv((3, 3), 2 => 3), randn(Float32, 3, 3, 2, 2)),
        (Conv((3, 3), 2 => 3, gelu; pad=SamePad()), randn(Float32, 3, 3, 2, 2)),
        (Conv((3, 3), 2 => 3, relu; use_bias=false, pad=SamePad()), randn(Float32, 3, 3, 2, 2)),
        (Chain(Conv((3, 3), 2 => 3, gelu), Conv((3, 3), 3 => 1, gelu)), rand(Float32, 5, 5, 2, 2)),
        (Chain(Conv((4, 4), 2 => 2; pad=SamePad()), MeanPool((5, 5); pad=SamePad())), rand(Float32, 5, 5, 2, 2)),
        (Chain(Conv((3, 3), 2 => 3, relu; pad=SamePad()), MaxPool((2, 2))), rand(Float32, 5, 5, 2, 2)),
        (Maxout(() -> Dense(5 => 4, tanh), 3), randn(Float32, 5, 2)),
        (Bilinear((2, 2) => 3), randn(Float32, 2, 3)),
        (SkipConnection(Dense(2 => 2), vcat), randn(Float32, 2, 3)),
        (ConvTranspose((3, 3), 3 => 2; stride=2), rand(Float32, 5, 5, 3, 1)),
        (StatefulRecurrentCell(RNNCell(3 => 5)), rand(Float32, 3, 2)),
        (StatefulRecurrentCell(RNNCell(3 => 5, gelu)), rand(Float32, 3, 2)),
        (StatefulRecurrentCell(RNNCell(3 => 5, gelu; use_bias=false)), rand(Float32, 3, 2)),
        (Chain(StatefulRecurrentCell(RNNCell(3 => 5)), StatefulRecurrentCell(RNNCell(5 => 3))), rand(Float32, 3, 2)),
        (StatefulRecurrentCell(LSTMCell(3 => 5)), rand(Float32, 3, 2)),
        (Chain(StatefulRecurrentCell(LSTMCell(3 => 5)), StatefulRecurrentCell(LSTMCell(5 => 3))), rand(Float32, 3, 2)),
        (StatefulRecurrentCell(GRUCell(3 => 5)), rand(Float32, 3, 10)),
        (Chain(StatefulRecurrentCell(GRUCell(3 => 5)), StatefulRecurrentCell(GRUCell(5 => 3))), rand(Float32, 3, 10)),
        (Chain(Dense(2, 4), BatchNorm(4)), randn(Float32, 2, 3)),
        (Chain(Dense(2, 4), BatchNorm(4, gelu)), randn(Float32, 2, 3)),
        (Chain(Dense(2, 4), BatchNorm(4, gelu; track_stats=false)), randn(Float32, 2, 3)),
        (Chain(Conv((3, 3), 2 => 6), BatchNorm(6)), randn(Float32, 6, 6, 2, 2)),
        (Chain(Conv((3, 3), 2 => 6, tanh), BatchNorm(6)), randn(Float32, 6, 6, 2, 2)),
        (Chain(Dense(2, 4), GroupNorm(4, 2, gelu)), randn(Float32, 2, 3)),
        (Chain(Dense(2, 4), GroupNorm(4, 2)), randn(Float32, 2, 3)),
        (Chain(Conv((3, 3), 2 => 6), GroupNorm(6, 3)), randn(Float32, 6, 6, 2, 2)),
        (Chain(Conv((3, 3), 2 => 6, tanh), GroupNorm(6, 3)), randn(Float32, 6, 6, 2, 2)),
        (Chain(Conv((3, 3), 2 => 3, gelu), LayerNorm((1, 1, 3))), randn(Float32, 4, 4, 2, 2)),
        (InstanceNorm(6), randn(Float32, 6, 6, 2, 2)),
        (Chain(Conv((3, 3), 2 => 6), InstanceNorm(6)), randn(Float32, 6, 6, 2, 2)),
        (Chain(Conv((3, 3), 2 => 6, tanh), InstanceNorm(6)), randn(Float32, 6, 6, 2, 2)),
    ]
        @info "$(typeof((f, x_f32...)))"
        rng = StableRNG(123456)
        ps, st = f32(Lux.setup(rng, f))
        x = f32(x_f32)
        test_rule(rng, f, x, ps, st; is_primitive=false, interface_only=true)
    end
end
