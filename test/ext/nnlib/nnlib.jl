using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using CUDA, cuDNN, JET, Mooncake, NNlib, StableRNGs, Test
using Mooncake.Nfwd: NDual, ndual_partial, ndual_value
using Mooncake.TestUtils: test_rule
using NNlib: dropout
using LuxLib
import LuxLib: Impl

dropout_tester_1(Trng, x, p) = dropout(Trng(1), x, p; dims=1)
dropout_tester_2(Trng, x, p) = dropout(Trng(1), x, p; dims=2)
dropout_tester_3(Trng, x, p) = dropout(Trng(1), x, p; dims=(1, 2))

# At p == 0 `dropout` returns its input itself, so mutating the result doubles the sum. `p`
# is fixed inside because `test_rule` would perturb it to `-ε`, outside `dropout`'s domain.
function dropout_alias_tester(Trng, x)
    y = dropout(Trng(1), x, zero(eltype(x)))
    y .*= 2
    return sum(x)
end

function dropout_alias_tester_dims(Trng, x)
    y = dropout(Trng(1), x, zero(eltype(x)); dims=1)
    y .*= 2
    return sum(x)
end

# TODO: drop the CUDA version bound once the fix for
# https://github.com/JuliaGPU/CUDA.jl/issues/2886 is released.
cuda = CUDA.functional() && pkgversion(CUDA) > v"5.9.6"

@testset "nnlib" begin
    _rand = if cuda
        (rng, size...) -> cu(randn(rng, size...))
    else
        (rng, size...) -> randn(rng, size...)
    end
    float = cuda ? x -> Float32(x) : identity
    _ones = cuda ? (d...) -> cu(ones(Float32, d...)) : (d...) -> ones(d...)
    # Wider than its `src` and not representable in it, so NNlib's rounding is observable;
    # `2.0` round-trips and would pass either way.
    mixed_src = cuda ? cu(ones(Float32, 3)) : ones(Float32, 3)
    _onetwo = cuda ? () -> cu(Float32[1, 2]) : () -> [1.0, 2.0]
    mixed_init = 2.1
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

        # batched_matmul_fallback for Array, NNlib.BatchedTranspose, NNlib.BatchedAdjoint
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            randn(rng, 3, 2, 3),
            randn(rng, 2, 5, 3),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            randn(rng, 3, 2, 3),
            NNlib.batched_transpose(randn(rng, 5, 2, 3)),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            randn(rng, 3, 2, 3),
            NNlib.batched_adjoint(randn(rng, 5, 2, 3)),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            NNlib.batched_transpose(randn(rng, 2, 3, 3)),
            randn(rng, 2, 5, 3),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            NNlib.batched_adjoint(randn(rng, 2, 3, 3)),
            randn(rng, 2, 5, 3),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            NNlib.batched_transpose(randn(rng, 2, 3, 3)),
            NNlib.batched_transpose(randn(rng, 5, 2, 3)),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            NNlib.batched_adjoint(randn(rng, 2, 3, 3)),
            NNlib.batched_adjoint(randn(rng, 5, 2, 3)),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            NNlib.batched_transpose(randn(rng, 2, 3, 3)),
            NNlib.batched_adjoint(randn(rng, 5, 2, 3)),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            NNlib.batched_adjoint(randn(rng, 2, 3, 3)),
            NNlib.batched_transpose(randn(rng, 5, 2, 3)),
        ),

        # batched_matmul_fallback: batch-size-1 broadcasting (exercises sum(tmp; dims=3) path)
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            randn(rng, 3, 2, 1),
            NNlib.batched_transpose(randn(rng, 5, 2, 3)),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            NNlib.batched_adjoint(randn(rng, 2, 3, 3)),
            randn(rng, 2, 5, 1),
        ),
        (
            false,
            :none,
            true,
            Impl.batched_matmul_fallback,
            NNlib.batched_transpose(randn(rng, 2, 3, 1)),
            NNlib.batched_adjoint(randn(rng, 5, 2, 3)),
        ),

        # dropout
        (true, :none, false, dropout_tester_1, Trng, _rand(rng, 2, 2), float(0.5)),
        (true, :none, false, dropout_tester_2, Trng, _rand(rng, 2, 2), float(0.1)),
        (true, :none, false, dropout_tester_3, Trng, _rand(rng, 2, 2), float(0.4)),
        (false, :none, false, dropout_alias_tester, Trng, _rand(rng, 2, 2)),
        (false, :none, false, dropout_alias_tester_dims, Trng, _rand(rng, 2, 2)),
        # The aliasing arm, `p ≤ 0`, which returns the input `CoDual`. On a device these
        # also need the `materialize!` rule for the forward broadcast. Both fail on `main`.
        (false, :none, false, dropout_alias_tester, Trng, _rand(rng, 2, 2)'),
        (false, :none, false, dropout_alias_tester, Trng, transpose(_rand(rng, 2, 2))),
        # ... and the other arm, `p > 0`, where the cotangent goes via `arrayify`'s wrapper.
        (true, :none, false, dropout_tester_1, Trng, _rand(rng, 2, 2)', float(0.5)),
        (
            true,
            :none,
            false,
            dropout_tester_2,
            Trng,
            transpose(_rand(rng, 2, 2)),
            float(0.1),
        ),

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

        # `scatter(max/min, ...)` with sources tied for one destination, hence `_ones`.
        (false, :none, true, NNlib.scatter, max, _ones(3), [1, 1, 2]),
        (false, :none, true, NNlib.scatter, min, _ones(3), [1, 1, 2]),
        (false, :none, true, Core.kwcall, (;), NNlib.scatter, max, _ones(3), [1, 1, 2]),

        # ndims(src) > ndims(idx): the tie splits per row, and the helper's gather/scatter
        # shapes have to agree on the extra leading dimension.
        (false, :none, true, NNlib.scatter, max, _ones(2, 3), [1, 1, 2]),

        # `init` is differentiable, so the keyword NamedTuple's rdata is not NoRData. The
        # cases put it below every source, above every source, and tied with one — a tie whose
        # symmetric split is the mean of the one-sided derivatives a central difference takes.
        (
            false,
            :none,
            true,
            Core.kwcall,
            (init=float(0.5),),
            NNlib.scatter,
            max,
            _ones(3),
            [1, 1, 2],
        ),
        (
            false,
            :none,
            true,
            Core.kwcall,
            (init=float(2.0),),
            NNlib.scatter,
            max,
            _ones(3),
            [1, 1, 2],
        ),
        (
            false,
            :none,
            true,
            Core.kwcall,
            (init=float(1.0),),
            NNlib.scatter,
            max,
            _onetwo(),
            [1, 2],
        ),
        (
            false,
            :none,
            true,
            Core.kwcall,
            (init=mixed_init,),
            NNlib.scatter,
            max,
            mixed_src,
            [1, 1, 2],
        ),

        # An `init` with no rdata, which used to throw. It wins outright, so nothing ties it.
        (
            false,
            :none,
            true,
            Core.kwcall,
            (init=2,),
            NNlib.scatter,
            max,
            _ones(3),
            [1, 1, 2],
        ),

        # A `NaN` in `src` must reach the gradient in both arms rather than be silenced to
        # zero by a `max(total, 1)` — removing that guard is what made the arms agree.
        # `interface_only`, since a `NaN` gradient is not finite-differenceable.
        (true, :none, true, NNlib.scatter, max, [NaN, 1.0, 2.0], [1, 1, 2]),
        (
            true,
            :none,
            true,
            Core.kwcall,
            (init=float(2.0),),
            NNlib.scatter,
            max,
            [NaN, 1.0, 2.0],
            [1, 1, 2],
        ),

        # `init` over a multi-dim `src`, where the tie count and the `init` indicator have to
        # agree on the extra leading dimension.
        (
            false,
            :none,
            true,
            Core.kwcall,
            (init=float(2.0),),
            NNlib.scatter,
            max,
            _ones(2, 3),
            [1, 1, 2],
        ),

        # `init=nothing` is NNlib's own default and takes the no-`init` path.
        (
            false,
            :none,
            true,
            Core.kwcall,
            (init=nothing,),
            NNlib.scatter,
            max,
            _ones(3),
            [1, 1, 2],
        ),

        # `dstsize` past the largest index leaves destinations no index reaches, holding
        # `scatter_empty`; nothing gathers them, so they take no part in the gradient, and the
        # slice keeps their `-Inf`s out of the sum.
        (
            false,
            :none,
            false,
            x -> sum(NNlib.scatter(max, x, [1, 1, 2]; dstsize=(4,))[1:2]),
            _ones(3),
        ),

        # gather
        (false, :none, true, NNlib.gather, _rand(rng, 2, 4), [1, 3, 1]),

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

        # bias_act!(identity, x, b): modifies x in-place
        (false, :stability, true, bias_act!, identity, _rand(rng, 8, 4), _rand(rng, 8)),
        (false, :stability, true, bias_act!, identity, _rand(rng, 8), _rand(rng, 8)),
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

# Testing arrayify for general adjoint, transpose types (LinearAlgebra.jl, NNlib.jl etc)
@testset "arrayify wrapper tests" begin
    rng = StableRNG(123)
    A2 = randn(rng, 3, 4)
    g2 = randn(rng, 3, 4)
    A3 = randn(Float32, 3, 4, 2)
    g3 = randn(Float32, 3, 4, 2)

    # Plain array
    xf = zeros(3, 4)
    _, dxf = Mooncake.arrayify(A2, xf)
    dxf .+= g2
    @test xf ≈ g2

    # Plain array, scalar gradient
    xf_scalar = zeros(3, 4)
    _, dxf_scalar = Mooncake.arrayify(A2, xf_scalar)
    dxf_scalar .+= 2.0
    @test xf_scalar ≈ fill(2.0, 3, 4)

    # Adjoint
    parent_adj = zeros(4, 3)
    _, dxf_adj = Mooncake.arrayify(A2', Mooncake.FData((parent=parent_adj,)))
    dxf_adj .+= g2
    @test parent_adj ≈ g2'

    # Transpose
    parent_tr = zeros(4, 3)
    _, dxf_tr = Mooncake.arrayify(transpose(A2), Mooncake.FData((parent=parent_tr,)))
    dxf_tr .+= g2
    @test parent_tr ≈ transpose(g2)

    # Accumulates — Adjoint
    parent_adj2 = ones(4, 3)
    _, dxf_adj2 = Mooncake.arrayify(A2', Mooncake.FData((parent=parent_adj2,)))
    dxf_adj2 .+= g2
    @test parent_adj2 ≈ ones(4, 3) .+ g2'

    # Accumulates — Transpose
    parent_tr2 = ones(4, 3)
    _, dxf_tr2 = Mooncake.arrayify(transpose(A2), Mooncake.FData((parent=parent_tr2,)))
    dxf_tr2 .+= g2
    @test parent_tr2 ≈ ones(4, 3) .+ transpose(g2)

    # BatchedTranspose
    parent_bt = zeros(Float32, 4, 3, 2)
    _, dxf_bt = Mooncake.arrayify(
        NNlib.batched_transpose(A3), Mooncake.FData((parent=parent_bt,))
    )
    dxf_bt .+= g3
    @test parent_bt ≈ permutedims(g3, (2, 1, 3))

    # BatchedAdjoint
    parent_ba = zeros(Float32, 4, 3, 2)
    _, dxf_ba = Mooncake.arrayify(
        NNlib.batched_adjoint(A3), Mooncake.FData((parent=parent_ba,))
    )
    dxf_ba .+= g3
    @test parent_ba ≈ permutedims(g3, (2, 1, 3))

    # Accumulates — BatchedTranspose
    parent_bt2 = ones(Float32, 4, 3, 2)
    _, dxf_bt2 = Mooncake.arrayify(
        NNlib.batched_transpose(A3), Mooncake.FData((parent=parent_bt2,))
    )
    dxf_bt2 .+= g3
    @test parent_bt2 ≈ ones(Float32, 4, 3, 2) .+ permutedims(g3, (2, 1, 3))

    # Accumulates — BatchedAdjoint
    parent_ba2 = ones(Float32, 4, 3, 2)
    _, dxf_ba2 = Mooncake.arrayify(
        NNlib.batched_adjoint(A3), Mooncake.FData((parent=parent_ba2,))
    )
    dxf_ba2 .+= g3
    @test parent_ba2 ≈ ones(Float32, 4, 3, 2) .+ permutedims(g3, (2, 1, 3))
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

    y_nd = NNlib.logsumexp(
        NDual{Float32,1}[
            NDual{Float32,1}(Inf32, (1.0f0,)), NDual{Float32,1}(Inf32, (0.0f0,))
        ],
    )
    @test isinf(ndual_value(y_nd)) && ndual_value(y_nd) > 0
    @test !isnan(ndual_partial(y_nd, 1))
    @test ndual_partial(y_nd, 1) ≈ 0.5f0

    y_nd_neg = NNlib.logsumexp(
        NDual{Float32,1}[
            NDual{Float32,1}(-Inf32, (1.0f0,)), NDual{Float32,1}(-Inf32, (0.0f0,))
        ],
    )
    @test isinf(ndual_value(y_nd_neg)) && ndual_value(y_nd_neg) < 0
    @test !isnan(ndual_partial(y_nd_neg, 1))
    @test ndual_partial(y_nd_neg, 1) ≈ 0.5f0
end

# NNlib's `exp(-abs(x))` form of σ has a kink at zero that the function itself does not.
@testset "sigmoid at zero and saturation: $f" for f in (NNlib.σ, NNlib.sigmoid_fast)
    test_rule(StableRNG(123), f, 0.0; perf_flag=:stability)
    # The reported failure was a gradient through a broadcast containing an exact zero.
    test_rule(StableRNG(123), x -> sum(f.(x)), [0.0, 0.5, -0.5]; is_primitive=false)
    # On GPU that broadcast is evaluated on `NDual`s in-kernel, an independent code path.
    if cuda
        test_rule(
            StableRNG(123), x -> sum(f.(x)), cu([0.0f0, 0.5f0, -0.5f0]); is_primitive=false
        )
    end
    # Saturated inputs: NaN, Inf and stability, not precision — see `test_rule`'s Limitations.
    test_rule(StableRNG(123), f, 37.0; perf_flag=:stability)
    test_rule(StableRNG(123), f, 17.0f0; perf_flag=:stability)

    # Float16 collapses first, at x >= 8, where `test_rule` fails whichever formula is in
    # place: in its `ȳ·ẏ + x̄·ẋ` check `ẏ` is exactly 0 for every step (spacing 4.88e-4
    # against a true change of 3.35e-6) and `ẋ` 2.3% off at best, so compare against a
    # wider-precision reference. `Ω * (1 - Ω)` returns exactly 0 here, so these are the only
    # tests pinning σ's precision.
    ref16 = (b=big(8.0); y=inv(1 + exp(-b)); Float64(y * (1 - y)))
    x16 = NDual{Float16,1}(Float16(8), (one(Float16),))
    @test Float64(ndual_partial(f(x16), 1)) ≈ ref16 rtol = 1e-2
    d16 = Mooncake.frule!!(
        Mooncake.Dual(f, Mooncake.NoTangent()), Mooncake.Dual(Float16(8), one(Float16))
    )
    @test Float64(Mooncake.tangent(d16)) ≈ ref16 rtol = 1e-2
    _, pb16 = Mooncake.rrule!!(Mooncake.zero_fcodual(f), Mooncake.zero_fcodual(Float16(8)))
    @test Float64(pb16(one(Float16))[2]) ≈ ref16 rtol = 1e-2
end

# `tanh_fast`'s primal discards an `ifelse` branch that overflows to `NaN`, which reverse mode
# picked up as `0 * Inf`, poisoning `gelu`/`gelu_tanh` from `|x| ≈ 21` upwards.
@testset "tanh_fast across NNlib's branches" begin
    test_rule(StableRNG(123), tanh_fast, 0.5; perf_flag=:stability)
    test_rule(StableRNG(123), tanh_fast, 400.0; perf_flag=:stability)
    # Below |x| ≈ 0.13 the primal switches to a polynomial while the rule stays analytic, so
    # this pins the gap between them as small enough for finite differences not to see it.
    test_rule(StableRNG(123), tanh_fast, 0.05; perf_flag=:stability)
    test_rule(
        StableRNG(123), x -> sum(NNlib.gelu_tanh.(x)), [1.0, 25.0]; is_primitive=false
    )
    # NNlib's `tanh_fast(::Float32)` is a different body, so none of the cases above reach it:
    # the rule kept working for `Float64` with `Float32` removed from it. Through `gelu_tanh`
    # the Float32 gradient goes `NaN` from |x| ≈ 258.8, so a scalar fails without the rule.
    test_rule(StableRNG(123), NNlib.gelu_tanh, 300.0f0; is_primitive=false)
    # As for σ, the saturated derivative's precision is beyond finite differences. `1 - Ω^2`
    # returns exactly 0 at Float16(6), against a true 2.46e-5, so it separates the two forms.
    ref16 = Float64(1 - tanh(big(6.0))^2)
    d16 = Mooncake.frule!!(
        Mooncake.Dual(tanh_fast, Mooncake.NoTangent()),
        Mooncake.Dual(Float16(6), one(Float16)),
    )
    @test Float64(Mooncake.tangent(d16)) ≈ ref16 rtol = 1e-2
    _, pb16 = Mooncake.rrule!!(
        Mooncake.zero_fcodual(tanh_fast), Mooncake.zero_fcodual(Float16(6))
    )
    @test Float64(pb16(one(Float16))[2]) ≈ ref16 rtol = 1e-2
end

# The rules below are reverse-only primitives, so forward mode traces the primal; declared for
# both modes it would find no `frule!!` and raise. CPU arrays only: a traced GPU kernel launch
# is a foreigncall, raising `MissingForeigncallRuleError`, and `gather` took the process down.
@testset "forward mode traces reverse-only rules" begin
    x = randn(StableRNG(123), 3)
    for f in (
        # `softmax` is returned whole: its outputs sum to 1, so `1ᵀJ = 0` and a summed case is
        # blind to every error in it. `logsoftmax` does not, so summing it stays a real check.
        x -> softmax(x),
        x -> softmax(x; dims=1),
        x -> sum(logsoftmax(x)),
        x -> sum(logsoftmax(x; dims=1)),
        x -> logsumexp(x),
        x -> sum(logsumexp(x; dims=1)),
        x -> sum(NNlib.gather(x, [1, 3])),
    )
        test_rule(StableRNG(123), f, x; mode=Mooncake.ForwardMode, is_primitive=false)
    end
end

# `gather`'s GPU kernel launch does not survive the forward transform — an illegal instruction
# no test can catch — so its rule raises, which this pins where a device is available.
if cuda
    @testset "forward mode over gather on a GPU array raises" begin
        gather_sum(z) = sum(NNlib.gather(z, [1, 3]))
        xg = cu(randn(StableRNG(123), Float32, 4))
        rule = Mooncake.build_frule(gather_sum, xg)
        @test_throws ArgumentError rule(
            Mooncake.zero_dual(gather_sum), Mooncake.Dual(copy(xg), CUDA.ones(Float32, 4))
        )
        cache = Mooncake.prepare_gradient_cache(gather_sum, xg)
        @test Array(Mooncake.value_and_gradient!!(cache, gather_sum, xg)[2][2]) ==
            Float32[1, 0, 1, 0]
    end
end
