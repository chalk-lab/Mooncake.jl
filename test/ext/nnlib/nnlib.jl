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
    # TODO: CUDA version bound when 
    #  https://github.com/JuliaGPU/CUDA.jl/issues/2886 is fixed and released
    cuda = CUDA.functional() && pkgversion(CUDA) > v"5.9.6"

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

        # softmax with Adjoint, Transpose
        (false, :stability, true, softmax, _rand(rng, 2, 3)'),
        (false, :stability, true, Core.kwcall, (dims=1,), softmax, _rand(rng, 3, 3)'),
        (false, :stability, true, Core.kwcall, (dims=2,), softmax, _rand(rng, 3, 3)'),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=1,),
            softmax,
            transpose(_rand(rng, 3, 3)),
        ),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=2,),
            softmax,
            transpose(_rand(rng, 3, 3)),
        ),

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

        # logsoftmax with Adjoint, Transpose
        (false, :stability, true, logsoftmax, _rand(rng, 2, 3)'),
        (false, :stability, true, Core.kwcall, (dims=1,), logsoftmax, _rand(rng, 3, 3)'),
        (false, :stability, true, Core.kwcall, (dims=2,), logsoftmax, _rand(rng, 3, 3)'),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=1,),
            logsoftmax,
            transpose(_rand(rng, 3, 3)),
        ),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=2,),
            logsoftmax,
            transpose(_rand(rng, 3, 3)),
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

        # logsumexp with Adjoint, Transpose
        (false, :stability, true, logsumexp, _rand(rng, 2, 3)'),
        (false, :stability, true, Core.kwcall, (dims=1,), logsumexp, _rand(rng, 3, 3)'),
        (false, :stability, true, Core.kwcall, (dims=2,), logsumexp, _rand(rng, 3, 3)'),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=1,),
            logsumexp,
            transpose(_rand(rng, 3, 3)),
        ),
        (
            false,
            :stability,
            true,
            Core.kwcall,
            (dims=2,),
            logsumexp,
            transpose(_rand(rng, 3, 3)),
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

        # bias_act!(identity, x, b): GPU falls back to x .+ b (non-in-place)
        (false, :none, true, bias_act!, identity, _rand(rng, 8, 4), _rand(rng, 8)),
        (false, :none, true, bias_act!, identity, _rand(rng, 8), _rand(rng, 8)),
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

# Testing arrayify for Adjoint/Transpose accumulation
@testset "arrayify Adjoint/Transpose tests" begin
    rng = StableRNG(123)
    A = randn(rng, 3, 4)
    g = randn(rng, 3, 4)

    # Plain array
    xf = zeros(3, 4)
    _, dxf = Mooncake.arrayify(A, xf)
    dxf .+= g
    @test xf ≈ g

    # Plain array, scalar gradient
    xf_scalar = zeros(3, 4)
    _, dxf_scalar = Mooncake.arrayify(A, xf_scalar)
    dxf_scalar .+= 2.0
    @test xf_scalar ≈ fill(2.0, 3, 4)

    # Adjoint: gradient transposed back into parent
    parent_adj = zeros(4, 3)
    xf_adj = Mooncake.FData((parent=parent_adj,))
    _, dxf_adj = Mooncake.arrayify(A', xf_adj)
    dxf_adj .+= g
    @test parent_adj ≈ g'

    # Transpose: gradient transposed back into parent
    parent_tr = zeros(4, 3)
    xf_tr = Mooncake.FData((parent=parent_tr,))
    _, dxf_tr = Mooncake.arrayify(transpose(A), xf_tr)
    dxf_tr .+= g
    @test parent_tr ≈ transpose(g)

    # Accumulates (+=), not overwrites — Adjoint
    parent_adj2 = ones(4, 3)
    xf_adj2 = Mooncake.FData((parent=parent_adj2,))
    _, dxf_adj2 = Mooncake.arrayify(A', xf_adj2)
    dxf_adj2 .+= g
    @test parent_adj2 ≈ ones(4, 3) .+ g'

    # Accumulates (+=), not overwrites — Transpose
    parent_tr2 = ones(4, 3)
    xf_tr2 = Mooncake.FData((parent=parent_tr2,))
    _, dxf_tr2 = Mooncake.arrayify(transpose(A), xf_tr2)
    dxf_tr2 .+= g
    @test parent_tr2 ≈ ones(4, 3) .+ transpose(g)
end

@testset "logsumexp Inf/NaN stability" begin
    function test_logsumexp_inf(x, dims)
        seed = ones(eltype(x), size(logsumexp(x; dims=dims)))
        cache = Mooncake.prepare_pullback_cache(
            Core.kwcall, NamedTuple{(:dims,)}((dims=dims,)), logsumexp, x
        )
        y, (_, _, _, dx) = Mooncake.value_and_pullback!!(
            cache, seed, Core.kwcall, NamedTuple{(:dims,)}((dims=dims,)), logsumexp, x
        )
        return y, dx
    end

    # All Inf inputs
    y, dx = test_logsumexp_inf(Float32[Inf, Inf], 1)
    @test all(isinf.(y)) && all(y .> 0)
    @test !any(isnan.(dx))
    @test dx ≈ Float32[0.5, 0.5]

    # All Inf inputs - Matrix case
    y, dx = test_logsumexp_inf(Float32[Inf Inf; Inf Inf], 1)
    @test !any(isnan.(y)) && !any(isnan.(dx))
    @test dx ≈ Float32[0.5 0.5; 0.5 0.5]

    # All -Inf inputs
    y, dx = test_logsumexp_inf(Float32[-Inf, -Inf], 1)
    @test all(isinf.(y)) && all(y .< 0)
    @test !any(isnan.(dx))
    @test dx ≈ Float32[0.5, 0.5]

    # Mixed Inf and finite inputs
    y, dx = test_logsumexp_inf(Float32[Inf, 1.0f0], 1)
    @test all(isinf.(y)) && all(y .> 0)
    @test !any(isnan.(dx))
    @test dx ≈ Float32[1.0f0, 0.0f0]
end
