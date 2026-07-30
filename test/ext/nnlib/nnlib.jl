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

# TODO: CUDA version bound when
#  https://github.com/JuliaGPU/CUDA.jl/issues/2886 is fixed and released
cuda = CUDA.functional() && pkgversion(CUDA) > v"5.9.6"

@testset "nnlib" begin
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
    # Saturated inputs. These cover NaN, Inf and type stability, not the precision itself —
    # see the Limitations section of `test_rule`'s docstring.
    test_rule(StableRNG(123), f, 37.0; perf_flag=:stability)
    test_rule(StableRNG(123), f, 17.0f0; perf_flag=:stability)

    # Float16 collapses first, at x >= 8, where `test_rule` fails whichever formula is in
    # place: both terms of its `ȳ·ẏ + x̄·ẋ` check are unusable, `ẏ` being exactly 0 for
    # every step (spacing 4.88e-4 against a true change of 3.35e-6) and `ẋ` 2.3% off at
    # best. So compare against a wider-precision reference. `Ω * (1 - Ω)` returns exactly 0
    # here, so these are the only tests pinning σ's precision, once per copy of the formula.
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

# `tanh_fast`'s primal discards an `ifelse` branch that overflows to `NaN`; reverse mode
# picked that branch up as `0 * Inf`, which poisoned `gelu`/`gelu_tanh` from `|x| ≈ 21`
# upwards.
@testset "tanh_fast across NNlib's branches" begin
    test_rule(StableRNG(123), tanh_fast, 0.5; perf_flag=:stability)
    test_rule(StableRNG(123), tanh_fast, 400.0; perf_flag=:stability)
    # Below |x| ≈ 0.13 the primal switches to a polynomial while the rule stays analytic, so
    # this pins the gap between them as small enough for finite differences not to see it.
    test_rule(StableRNG(123), tanh_fast, 0.05; perf_flag=:stability)
    test_rule(
        StableRNG(123), x -> sum(NNlib.gelu_tanh.(x)), [1.0, 25.0]; is_primitive=false
    )
    # NNlib's `tanh_fast(::Float32)` is a different body from the `Float64` one — a Remez
    # rational whose discarded branch overflows, rather than an `exp` quotient — so none of
    # the cases above reach it, and the rule kept working for `Float64` with `Float32`
    # removed from it. Through `gelu_tanh` the Float32 gradient goes `NaN` from |x| ≈ 258.8,
    # so a scalar there fails without the rule and passes with it.
    test_rule(StableRNG(123), NNlib.gelu_tanh, 300.0f0; is_primitive=false)
    # As for σ, the saturated derivative's precision is beyond finite differences, and both
    # rules carry their own copy of `4u / (1 + u)^2`. `1 - Ω^2` returns exactly 0 at
    # Float16(6), against a true 2.46e-5, so this separates the two forms outright.
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

# Each rule below exists only in reverse mode, so its `@is_primitive` says so; declared for
# both, forward mode finds a primitive with no `frule!!` and raises instead of tracing. CPU
# arrays only: on a GPU array the traced primal reaches a kernel launch, a foreigncall, and
# raises `MissingForeigncallRuleError`. `gather` took the process down, hence its own rule
# below.
@testset "forward mode traces reverse-only rules" begin
    x = randn(StableRNG(123), 3)
    for f in (
        # `softmax` is returned whole rather than summed: its outputs sum to 1 identically,
        # so `1ᵀJ = 0` and a summed case is blind to every error inside that kernel.
        # `logsoftmax` does not sum to a constant, so summing it stays a real check.
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

# `gather`'s GPU kernel launch does not survive the forward transform — an illegal
# instruction no test can catch — so its rule raises instead, which this pins where a device
# is available.
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
