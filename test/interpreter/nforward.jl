using Random: Xoshiro

struct NForwardRRuleTestFunc{N,F}
    f::F
end

(f::NForwardRRuleTestFunc)(x...) = f.f(x...)

function Mooncake.build_rrule(
    f::NForwardRRuleTestFunc{N}, x...; debug_mode=false, silence_debug_messages=true
) where {N}
    return Mooncake.nforward_build_rrule(
        f.f, x...; chunk_size=N, debug_mode, silence_debug_messages
    )
end

@testset "nforward API" begin
    f = (x, y) -> x * y + cos(x)
    x, y = 5.0, 4.0
    dx, dy = 3.0, 2.0
    z = f(x, y)
    dz = dx * y + x * dy + dx * (-sin(x))
    scalar_cases = (
        (
            name="chunk_size=1",
            chunk_size=1,
            dual_inputs=(Mooncake.Dual(x, dx), Mooncake.Dual(y, dy)),
            expected_tangent=dz,
        ),
        (
            name="chunk_size=2 scalar lanes",
            chunk_size=2,
            dual_inputs=(Mooncake.Dual(x, (dx, 0.0)), Mooncake.Dual(y, (0.0, dy))),
            expected_tangent=(dx * y + dx * (-sin(x)), x * dy),
        ),
    )

    @testset "$case.name" for case in scalar_cases
        rule = Mooncake.nforward_build_frule(f, x, y; chunk_size=case.chunk_size)
        out = rule(Mooncake.zero_dual(f), case.dual_inputs...)
        @test out isa Mooncake.Dual
        @test Mooncake.primal(out) == z
        @test Mooncake.tangent(out) == case.expected_tangent

        rrule = Mooncake.nforward_build_rrule(f, x, y; chunk_size=case.chunk_size)
        ȳ, pb!! = rrule(
            Mooncake.zero_fcodual(f), Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(y)
        )
        @test Mooncake.primal(ȳ) == z
        @test pb!!(1.0) == (Mooncake.NoRData(), y - sin(x), x)

        cache = Mooncake.nforward_prepare_cache(f, x, y; chunk_size=case.chunk_size)
        z_and_dz_dual = Mooncake.value_and_derivative!!(
            cache, Mooncake.zero_dual(f), case.dual_inputs...
        )
        @test z_and_dz_dual isa Mooncake.Dual
        @test Mooncake.primal(z_and_dz_dual) == z
        @test Mooncake.tangent(z_and_dz_dual) == case.expected_tangent

        case.chunk_size == 1 && continue
        z_and_grad = Mooncake.value_and_gradient!!(cache, f, x, y)
        @test first(z_and_grad) == z
        @test last(z_and_grad) == (Mooncake.NoTangent(), y - sin(x), x)
    end

    @testset "chunk_size=2 array lanes" begin
        g(x) = sin.(x)
        x_vec = [1.0, 2.0]
        dx_vec = reshape([1.0, 0.0, 0.0, 1.0], 2, 2)
        rrule = Mooncake.nforward_build_rrule(g, x_vec; chunk_size=2)
        value, pullback = Mooncake.value_and_pullback!!(rrule, [3.0, 4.0], g, x_vec)
        @test value == sin.(x_vec)
        @test pullback == (Mooncake.NoTangent(), [3.0 * cos(x_vec[1]), 4.0 * cos(x_vec[2])])

        cache = Mooncake.nforward_prepare_cache(g, x_vec; chunk_size=2)
        y_and_dy = Mooncake.value_and_derivative!!(
            cache, Mooncake.zero_dual(g), Mooncake.Dual(x_vec, dx_vec)
        )
        @test Mooncake.primal(y_and_dy) == sin.(x_vec)
        @test Mooncake.tangent(y_and_dy) ≈ [cos(x_vec[1]) 0.0; 0.0 cos(x_vec[2])]

        @test_throws Mooncake.ValueAndGradientReturnTypeError Mooncake.value_and_gradient!!(
            cache, g, x_vec
        )
    end

    @testset "complex inputs" begin
        fc(z) = real(z * z + cos(z))
        zc = ComplexF64(1.2, -0.3)
        dzc = ComplexF64(0.5, -0.25)
        expected_dzc = real((2zc - sin(zc)) * dzc)
        rule = Mooncake.nforward_build_frule(fc, zc; chunk_size=1)
        out = rule(Mooncake.zero_dual(fc), Mooncake.Dual(zc, dzc))
        @test Mooncake.primal(out) == fc(zc)
        @test Mooncake.tangent(out) ≈ expected_dzc

        rrule = Mooncake.nforward_build_rrule(fc, zc; chunk_size=2)
        ȳ, pb!! = rrule(Mooncake.zero_fcodual(fc), Mooncake.zero_fcodual(zc))
        @test Mooncake.primal(ȳ) == fc(zc)
        @test pb!!(1.0) == (Mooncake.NoRData(), conj(2zc - sin(zc)))

        cache_scalar = Mooncake.nforward_prepare_cache(fc, zc; chunk_size=2)
        z_and_grad = Mooncake.value_and_gradient!!(cache_scalar, fc, zc)
        @test first(z_and_grad) == fc(zc)
        @test last(z_and_grad) == (Mooncake.NoTangent(), conj(2zc - sin(zc)))

        gc(z) = sum(abs2, z)
        z_vec = ComplexF64[1.0 + 2.0im, -3.0 + 0.5im]
        dz_vec = reshape(
            ComplexF64[1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 1.0im], 2, 2
        )
        cache = Mooncake.nforward_prepare_cache(gc, z_vec; chunk_size=2)
        out_vec = Mooncake.value_and_derivative!!(
            cache, Mooncake.zero_dual(gc), Mooncake.Dual(z_vec, dz_vec)
        )
        @test Mooncake.primal(out_vec) == gc(z_vec)
        @test Mooncake.tangent(out_vec) == (2.0, 1.0)

        hc(z) = z .* z
        ȳ_vec = ComplexF64[2.0 - 1.0im, -0.5 + 0.25im]
        rrule_vec = Mooncake.nforward_build_rrule(hc, z_vec; chunk_size=2)
        value_vec, pullback_vec = Mooncake.value_and_pullback!!(
            rrule_vec, ȳ_vec, hc, z_vec
        )
        @test value_vec == hc(z_vec)
        @test pullback_vec == (Mooncake.NoTangent(), 2 .* conj.(z_vec) .* ȳ_vec)
    end

    @testset "multi-argument array input" begin
        h(x, y) = sum(x .* y)
        x_vec = [1.0, 2.0]
        y_vec = [3.0, 4.0]
        cache = Mooncake.nforward_prepare_cache(h, x_vec, y_vec; chunk_size=2)
        dx = reshape([1.0, 0.0, 0.0, 0.0], 2, 2)
        dy = reshape([0.0, 0.0, 1.0, 0.0], 2, 2)
        out = Mooncake.value_and_derivative!!(
            cache, Mooncake.zero_dual(h), Mooncake.Dual(x_vec, dx), Mooncake.Dual(y_vec, dy)
        )
        @test Mooncake.primal(out) == h(x_vec, y_vec)
        @test Mooncake.tangent(out) == (3.0, 1.0)
    end

    @testset "chunk_size=2 matrix lanes" begin
        h(X) = sin.(X)
        X = reshape([1.0, 2.0, 3.0, 4.0], 2, 2)
        dX = zeros(2, 2, 2)
        dX[1, 1, 1] = 1.0
        dX[2, 2, 2] = 1.0
        cache = Mooncake.nforward_prepare_cache(h, X; chunk_size=2)
        y_and_dy = Mooncake.value_and_derivative!!(
            cache, Mooncake.zero_dual(h), Mooncake.Dual(X, dX)
        )
        @test Mooncake.primal(y_and_dy) == sin.(X)
        expected = zeros(2, 2, 2)
        expected[1, 1, 1] = cos(X[1, 1])
        expected[2, 2, 2] = cos(X[2, 2])
        @test Mooncake.tangent(y_and_dy) ≈ expected
    end

    @testset "function instance mismatch is rejected" begin
        a = 2.0
        b = 3.0
        f_a = x -> a * x
        f_b = x -> b * x
        cache = Mooncake.nforward_prepare_cache(f_a, 5.0; chunk_size=1)
        @test_throws ArgumentError Mooncake.value_and_gradient!!(cache, f_b, 5.0)
        @test_throws ArgumentError Mooncake.value_and_derivative!!(
            cache, Mooncake.zero_dual(f_b), Mooncake.Dual(5.0, 1.0)
        )
    end

    @testset "unsupported config is rejected" begin
        @test_throws ArgumentError Mooncake.nforward_prepare_cache(
            f, x, y; chunk_size=1, config=Mooncake.Config(; friendly_tangents=true)
        )
        @test_throws ArgumentError Mooncake.nforward_prepare_cache(
            f, x, y; chunk_size=1, config=Mooncake.Config(; debug_mode=true)
        )
        @test_throws ArgumentError Mooncake.nforward_prepare_cache(
            f, view([x, y], 1:2); chunk_size=1
        )
        @test_throws ArgumentError Mooncake.nforward_build_frule(
            f, x, y; chunk_size=1, debug_mode=true
        )
        @test_throws ArgumentError Mooncake.nforward_build_rrule(
            f, x, y; chunk_size=1, debug_mode=true
        )
    end

    @testset "invalid chunk_size is rejected" begin
        @test_throws ArgumentError Mooncake.nforward_build_frule(f, x, y; chunk_size=0)
        @test_throws ArgumentError Mooncake.nforward_prepare_cache(f, x, y; chunk_size=-1)
    end

    @testset "function tangent rejection" begin
        rule = Mooncake.nforward_build_frule(f, x, y; chunk_size=1)
        @test_throws ArgumentError rule(
            Mooncake.Dual(f, 1.0), Mooncake.Dual(x, dx), Mooncake.Dual(y, dy)
        )

        rrule = Mooncake.nforward_build_rrule(f, x, y; chunk_size=1)
        @test_throws ArgumentError rrule(
            Mooncake.CoDual(f, 1.0), Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(y)
        )
    end

    @testset "array tangent validation" begin
        g(x) = sin.(x)
        x_vec = [1.0, 2.0]
        cache = Mooncake.nforward_prepare_cache(g, x_vec; chunk_size=2)
        @test_throws ArgumentError Mooncake.value_and_derivative!!(
            cache, Mooncake.zero_dual(g), Mooncake.Dual(x_vec, [1.0, 2.0, 3.0])
        )
    end

    @testset "args_to_zero validation" begin
        cache = Mooncake.nforward_prepare_cache(f, x, y; chunk_size=2)
        @test_throws ArgumentError Mooncake.value_and_gradient!!(
            cache, f, x, y; args_to_zero=(true, true)
        )
        @test_throws ArgumentError Mooncake.value_and_gradient!!(
            cache, f, x, y; args_to_zero=(true, false, true)
        )
    end

    @testset "cache size mismatch is rejected" begin
        g(x) = sum(x)
        cache = Mooncake.nforward_prepare_cache(g, [1.0, 2.0]; chunk_size=1)
        @test_throws ArgumentError Mooncake.value_and_gradient!!(cache, g, [1.0, 2.0, 3.0])
    end

    @testset "automatic chunk_size selection" begin
        cache = Mooncake.nforward_prepare_cache(f, x, y)
        z_and_grad = Mooncake.value_and_gradient!!(cache, f, x, y)
        @test first(z_and_grad) == z
        @test last(z_and_grad) == (Mooncake.NoTangent(), y - sin(x), x)
    end

    @testset "test_rule integration" begin
        nf = NForwardRRuleTestFunc{2,typeof(f)}(f)
        Mooncake.TestUtils.test_rule(
            Xoshiro(123),
            nf,
            x,
            y;
            is_primitive=false,
            perf_flag=:none,
            mode=Mooncake.ReverseMode,
        )
    end
end
