using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using CUDA, cuDNN, JET, Mooncake, NNlib, StableRNGs, Test
using Mooncake.TestUtils: test_rule
using NNlib: dropout

dropout_tester_1(Trng, x, p) = dropout(Trng(1), x, p; dims=1)
dropout_tester_2(Trng, x, p) = dropout(Trng(1), x, p; dims=2)
dropout_tester_3(Trng, x, p) = dropout(Trng(1), x, p; dims=(1, 2))

@testset "nnlib" begin
    cuda = CUDA.functional()

    _rand = if cuda
        (rng, size...) -> cu(randn(rng, size...))
    else
        (rng, size...) -> randn(rng, size...)
    end
    float = cuda ? x -> Float32(x) : identity
    Trng = cuda ? CUDA.RNG : StableRNG

    rng = StableRNG(123)
    x = randn(rng, 5, 4, 3, 2)
    w = randn(rng, 2, 2, 3, 3)
    dense_cdims = DenseConvDims(x, w)
    sep_cd = DepthwiseConvDims(x, w)
    y = conv(x, w, dense_cdims)
    y_sep = depthwiseconv(x, w, sep_cd)

    pool_dims = PoolDims(size(x), 2)

    grid = Array{Float64}(undef, 2, 2, 2, 1)
    grid[:, 1, 1, 1] .= (-1, -1)
    grid[:, 2, 1, 1] .= (1, -1)
    grid[:, 1, 2, 1] .= (-1, 1)
    grid[:, 2, 2, 1] .= (1, 1)
    grid = cuda ? cu(grid) : grid
    x = cuda ? cu(x) : x
    w = cuda ? cu(w) : w
    y = cuda ? cu(y) : y
    y_sep = cuda ? cu(y_sep) : y_sep

    test_cases = Any[

        # batched_mul
        (false, :none, true, batched_mul, _rand(rng, 3, 2, 3), _rand(rng, 2, 5, 3)),

        # dropout
        (true, :none, false, dropout_tester_1, Trng, _rand(rng, 2, 2), float(0.5)),
        (true, :none, false, dropout_tester_2, Trng, _rand(rng, 2, 2), float(0.1)),
        (true, :none, false, dropout_tester_3, Trng, _rand(rng, 2, 2), float(0.4)),

        # softmax
        (false, :stability, true, softmax, _rand(rng, 2)),
        (false, :stability, true, softmax, _rand(rng, 2, 2)),
        (false, :stability, true, Core.kwcall, (dims=1,), softmax, _rand(rng, 2)),
        (false, :stability, true, Core.kwcall, (dims=1,), softmax, _rand(rng, 3, 3)),
        (false, :stability, true, Core.kwcall, (dims=2,), softmax, _rand(rng, 3, 3)),
        (false, :stability, true, Core.kwcall, (dims=(1, 2),), softmax, _rand(rng, 3, 3)),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=(1, 2),),
            softmax,
            _rand(rng, 3, 3, 2),
        ),
        (false, :none, false, x -> softmax(x; dims=1), _rand(rng, 3, 2)),
        (false, :none, false, x -> softmax(x; dims=2), _rand(rng, 3, 2)),
        (false, :none, false, x -> softmax(x; dims=(1, 2)), _rand(rng, 3, 2)),

        # logsoftmax
        (false, :stability, true, logsoftmax, _rand(rng, 2)),
        (false, :stability, true, logsoftmax, _rand(rng, 2, 3)),
        (false, :stability, true, logsoftmax, _rand(rng, 2, 3, 2)),
        (false, :stability, true, Core.kwcall, (dims=1,), logsoftmax, _rand(rng, 2)),
        (false, :stability, true, Core.kwcall, (dims=1,), logsoftmax, _rand(rng, 3, 3)),
        (false, :stability, true, Core.kwcall, (dims=2,), logsoftmax, _rand(rng, 3, 3)),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=(1, 2),),
            logsoftmax,
            _rand(rng, 3, 3),
        ),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=(1, 2),),
            logsoftmax,
            _rand(rng, 3, 3, 2),
        ),

        # logsumexp
        (false, :stability, true, logsumexp, _rand(rng, 2)),
        (false, :stability, true, logsumexp, _rand(rng, 3, 3)),
        (false, :stability, true, logsumexp, _rand(rng, 3, 3, 2)),
        (false, :stability, true, Core.kwcall, (dims=1,), logsumexp, _rand(rng, 2)),
        (false, :stability, true, Core.kwcall, (dims=1,), logsumexp, _rand(rng, 3, 3)),
        (false, :stability, true, Core.kwcall, (dims=2,), logsumexp, _rand(rng, 3, 3)),
        (false, :stability, true, Core.kwcall, (dims=(1, 2),), logsumexp, _rand(rng, 3, 3)),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=(1, 2),),
            logsumexp,
            _rand(rng, 3, 3, 2),
        ),

        # upsample_nearest
        (false, :stability, true, upsample_nearest, _rand(rng, 3), (2,)),
        (false, :stability, true, upsample_nearest, _rand(rng, 3, 2), (2, 2)),
        (false, :stability, true, upsample_nearest, _rand(rng, 3, 2, 3), (2, 2, 5)),

        # fold
        (false, :none, true, NNlib.fold, _rand(rng, 12, 12, 2), size(x), dense_cdims),

        # unfold
        (false, :none, true, NNlib.unfold, x, dense_cdims),

        # scatter
        (false, :none, true, NNlib.scatter, +, _rand(rng, 2), [1, 3]),
        (false, :none, true, Core.kwcall, (;), NNlib.scatter, +, _rand(rng, 2), [1, 3]),

        # conv
        (false, :none, true, Core.kwcall, (;), conv, x, w, dense_cdims),
        (false, :none, true, conv, x, w, dense_cdims),

        # ∇conv_data
        (false, :none, true, Core.kwcall, (;), ∇conv_data, y, w, dense_cdims),
        (false, :none, true, ∇conv_data, y, w, dense_cdims),

        # ∇conv_filter
        (false, :none, true, Core.kwcall, (;), ∇conv_filter, x, y, dense_cdims),
        (false, :none, true, ∇conv_filter, x, y, dense_cdims),

        # pooling
        (false, :none, true, maxpool, x, pool_dims),
        (false, :none, true, Core.kwcall, (;), maxpool, x, pool_dims),
        (false, :none, true, meanpool, x, pool_dims),
        (false, :none, true, Core.kwcall, (;), meanpool, x, pool_dims),

        # padding
        (false, :none, false, x -> pad_constant(x, 1, float(2.0)), x),
        (false, :none, false, x -> pad_constant(x, 1, float(2.0); dims=:), x),
    ]
    if !cuda

        # Tests here fail on CUDA.
        cpu_only_test_cases = Any[
            # softmax
            (false, :none, false, x -> softmax(5x), _rand(rng, 3, 2)),

            # conv
            (false, :none, true, Core.kwcall, (;), depthwiseconv, x, w, sep_cd),
            (false, :none, true, depthwiseconv, x, w, sep_cd),

            # ∇conv_data
            (false, :none, true, Core.kwcall, (;), ∇depthwiseconv_data, y_sep, w, sep_cd),
            (false, :none, true, ∇depthwiseconv_data, y_sep, w, sep_cd),
        ]
        test_cases = vcat(test_cases, cpu_only_test_cases)
    end
    @testset "$(typeof(fargs))" for (interface_only, perf_flag, is_primitive, fargs...) in
                                    test_cases

        @info "$(typeof(fargs))"
        perf_flag = cuda ? :none : perf_flag
        mode = Mooncake.ReverseMode
        test_rule(StableRNG(123), fargs...; perf_flag, is_primitive, interface_only, mode)
    end
end
