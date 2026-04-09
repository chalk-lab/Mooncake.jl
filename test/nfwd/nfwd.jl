# NDual unit tests — pure arithmetic on the NDual type; no GPU device required.
using LinearAlgebra
using Mooncake.Nfwd
@testset "NDual" begin
    # helpers
    _d(v, p1) = NDual{Float64,1}(v, (p1,))
    _d2(v, p1, p2) = NDual{Float64,2}(v, (p1, p2))
    _d32(v, p1) = NDual{Float32,1}(Float32(v), (Float32(p1),))

    @testset "construction and accessors" begin
        d = NDual{Float64,2}(3.0, (1.0, 0.0))
        @test Nfwd.ndual_value(d) === 3.0
        @test Nfwd.ndual_partial(d, 1) === 1.0
        @test Nfwd.ndual_partial(d, 2) === 0.0
        @test d.partials === (1.0, 0.0)

        # scalar constructor (zero partials)
        dc = NDual{Float64,2}(3.0)
        @test Nfwd.ndual_value(dc) === 3.0
        @test dc.partials === (0.0, 0.0)

        # isbits — critical for GPU register allocation
        @test isbits(NDual{Float64,2}(1.0, (0.0, 0.0)))
        @test isbits(NDual{Float32,3}(1.0f0, (0.0f0, 0.0f0, 0.0f0)))
    end

    @testset "zero / one / oneunit" begin
        @test zero(NDual{Float64,2}) == NDual{Float64,2}(0.0, (0.0, 0.0))
        @test one(NDual{Float64,2}) == NDual{Float64,2}(1.0, (0.0, 0.0))
        @test zero(_d(1.0, 2.0)) == NDual{Float64,1}(0.0, (0.0,))
        @test one(_d(1.0, 2.0)) == NDual{Float64,1}(1.0, (0.0,))
        # oneunit(T) must return T(1) with zero partials, not call T(one(T)) which errors
        @test oneunit(NDual{Float64,2}) == NDual{Float64,2}(1.0, (0.0, 0.0))
        @test oneunit(_d(3.0, 1.0)) == NDual{Float64,1}(1.0, (0.0,))
    end

    @testset "promote / convert" begin
        d = NDual{Float64,1}(2.0, (1.0,))
        @test convert(NDual{Float64,1}, 3) == NDual{Float64,1}(3.0, (0.0,))
        @test convert(NDual{Float64,1}, d) === d
        @test promote_type(NDual{Float64,1}, Int) === NDual{Float64,1}
        @test promote_type(NDual{Float64,1}, Float32) === NDual{Float64,1}

        # Cross-precision: NDual{Float32,N} op NDual{Float64,N} → NDual{Float64,N}
        @test promote_type(NDual{Float32,2}, NDual{Float64,2}) === NDual{Float64,2}
        @test promote_type(NDual{Float64,2}, NDual{Float32,2}) === NDual{Float64,2}

        d32 = NDual{Float32,2}(2.0f0, (1.0f0, 0.0f0))
        d64 = convert(NDual{Float64,2}, d32)
        @test d64 isa NDual{Float64,2}
        @test Nfwd.ndual_value(d64) === 2.0
        @test Nfwd.ndual_partial(d64, 1) === 1.0
        @test Nfwd.ndual_partial(d64, 2) === 0.0

        # Arithmetic between different precisions auto-promotes
        a32 = NDual{Float32,1}(2.0f0, (1.0f0,))
        b64 = NDual{Float64,1}(3.0, (0.0,))
        r = a32 + b64
        @test r isa NDual{Float64,1}
        @test Nfwd.ndual_value(r) ≈ 5.0
        @test Nfwd.ndual_partial(r, 1) ≈ 1.0

        r2 = a32 * b64
        @test r2 isa NDual{Float64,1}
        @test Nfwd.ndual_value(r2) ≈ 6.0
        @test Nfwd.ndual_partial(r2, 1) ≈ 3.0  # b.value * da

        # Float64 literal mixed with NDual{Float32} (GPU broadcast scenario)
        lit = 2.0  # Float64
        r3 = lit * a32
        @test r3 isa NDual{Float64,1}
        @test Nfwd.ndual_value(r3) ≈ 4.0
        @test Nfwd.ndual_partial(r3, 1) ≈ 2.0
    end

    @testset "arithmetic" begin
        a = _d2(2.0, 1.0, 0.0)   # represents 2 + 1*e1
        b = _d2(3.0, 0.0, 1.0)   # represents 3 + 1*e2

        @test a + b == _d2(5.0, 1.0, 1.0)
        @test a - b == _d2(-1.0, 1.0, -1.0)
        @test -a == _d2(-2.0, -1.0, 0.0)

        # product rule: d(a*b) = a*db + b*da
        r = a * b
        @test Nfwd.ndual_value(r) ≈ 6.0
        @test Nfwd.ndual_partial(r, 1) ≈ 3.0  # b.value * da/de1
        @test Nfwd.ndual_partial(r, 2) ≈ 2.0  # a.value * db/de2

        # quotient rule: d(a/b) = (da - (a/b)*db) / b
        r = a / b
        @test Nfwd.ndual_value(r) ≈ 2.0 / 3.0
        @test Nfwd.ndual_partial(r, 1) ≈ 1.0 / 3.0
        @test Nfwd.ndual_partial(r, 2) ≈ -(2.0 / 3.0) / 3.0

        # Direct Real ± NDual: partials unchanged for +/-, negated for c-x
        @test Nfwd.ndual_value(a + 1.0) ≈ 3.0
        @test Nfwd.ndual_partial(a + 1.0, 1) ≈ 1.0
        @test Nfwd.ndual_value(1.0 + a) ≈ 3.0
        @test Nfwd.ndual_partial(1.0 + a, 1) ≈ 1.0
        @test Nfwd.ndual_value(5.0 - a) ≈ 3.0
        @test Nfwd.ndual_partial(5.0 - a, 1) ≈ -1.0   # -(a.partials)
        @test Nfwd.ndual_value(a - 1.0) ≈ 1.0
        @test Nfwd.ndual_partial(a - 1.0, 1) ≈ 1.0

        # Direct Real*NDual: scales partials without product rule
        @test Nfwd.ndual_value(2.0 * a) ≈ 4.0
        @test Nfwd.ndual_partial(2.0 * a, 1) ≈ 2.0
        @test Nfwd.ndual_value(a * 3.0) ≈ 6.0
        @test Nfwd.ndual_partial(a * 3.0, 1) ≈ 3.0

        # Direct NDual / Real: scales partials by reciprocal
        @test Nfwd.ndual_value(a / 2.0) ≈ 1.0
        @test Nfwd.ndual_partial(a / 2.0, 1) ≈ 0.5

        # Direct inv: d(1/x)/dx = -1/x²
        x = _d(3.0, 1.0)
        @test Nfwd.ndual_value(inv(x)) ≈ 1/3.0
        @test Nfwd.ndual_partial(inv(x), 1) ≈ -1/9.0
    end

    @testset "power" begin
        x = _d(3.0, 1.0)
        # literal integer powers — dispatches through Base.literal_pow
        # d(x^2)/dx = 2x
        @test Nfwd.ndual_value(x^2) ≈ 9.0
        @test Nfwd.ndual_partial(x^2, 1) ≈ 6.0
        # d(x^3)/dx = 3x^2
        @test Nfwd.ndual_value(x^3) ≈ 27.0
        @test Nfwd.ndual_partial(x^3, 1) ≈ 27.0
        # literal x^0 → one, zero derivative
        @test Nfwd.ndual_value(x^0) ≈ 1.0
        @test Nfwd.ndual_partial(x^0, 1) ≈ 0.0
        # literal negative integer power: x^(-1) = 1/x, d/dx = -1/x²
        @test Nfwd.ndual_value(x^(-1)) ≈ 1/3.0
        @test Nfwd.ndual_partial(x^(-1), 1) ≈ -1/9.0
        # literal x^(-2): d/dx = -2/x³
        @test Nfwd.ndual_value(x^(-2)) ≈ 1/9.0
        @test Nfwd.ndual_partial(x^(-2), 1) ≈ -2/27.0
        # real exponent (runtime Float64, uses ^(NDual, Real))
        @test Nfwd.ndual_value(x^2.0) ≈ 9.0
        @test Nfwd.ndual_partial(x^2.0, 1) ≈ 6.0
        # real exponent b=0.0: d(x^0)/dx = 0 everywhere, including x=0 (no NaN)
        @test Nfwd.ndual_partial(_d(0.0, 1.0)^0.0, 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(_d(0.0, 1.0)^0.0, 1))

        z1 = _d2(0.0, 1.0, 0.0)
        p1 = _d2(1.0, 0.0, 0.0)
        @test Nfwd.ndual_partial(z1^p1, 1) === 1.0

        z2 = _d2(0.0, 1.0, 0.0)
        p2 = _d2(2.0, 0.0, 0.0)
        @test Nfwd.ndual_partial(z2^p2, 1) === 0.0

        zh = _d2(0.0, 1.0, 0.0)
        ph = _d2(0.5, 0.0, 0.0)
        @test isinf(Nfwd.ndual_partial(zh^ph, 1))

        xb = _d2(-2.0, 0.0, 1.0)
        pb = _d2(3.0, 1.0, 0.0)
        @test Nfwd.ndual_partial(xb^pb, 1) ≈ -8.0 * log(2.0)

        xf = _d(3.0, 1.0)
        @test Nfwd.ndual_value(Base.FastMath.pow_fast(xf, Int32(2))) ≈ 9.0
        @test Nfwd.ndual_partial(Base.FastMath.pow_fast(xf, Int32(2)), 1) ≈ 6.0
        @test Nfwd.ndual_value(Base.FastMath.pow_fast(xf, Val(3))) ≈ 27.0
        @test Nfwd.ndual_partial(Base.FastMath.pow_fast(xf, Val(3)), 1) ≈ 27.0
    end

    @testset "mod and mod2pi" begin
        m = mod(_d2(7.5, 1.0, 0.0), _d2(2.3, 0.0, 1.0))
        @test Nfwd.ndual_value(m) ≈ mod(7.5, 2.3)
        @test Nfwd.ndual_partial(m, 1) === 1.0
        @test Nfwd.ndual_partial(m, 2) ≈ -floor(7.5 / 2.3)

        ms = mod(_d2(4.0, 1.0, 0.0), _d2(2.0, 0.0, 1.0))
        @test isnan(Nfwd.ndual_partial(ms, 1))
        @test isnan(Nfwd.ndual_partial(ms, 2))

        a = mod2pi(_d(0.1, 1.0))
        @test Nfwd.ndual_value(a) ≈ mod2pi(0.1)
        @test Nfwd.ndual_partial(a, 1) === 1.0

        as = mod2pi(_d(2π, 1.0))
        @test isnan(Nfwd.ndual_partial(as, 1))
    end

    @testset "math functions" begin
        # Test each f(Dual(v,1)) matches f'(v) analytically
        for (v, fns) in [
            (
                0.5,
                [
                    (sin, cos),
                    (cos, x -> -sin(x)),
                    (tan, x -> inv(cos(x))^2),
                    (exp, exp),
                    (log, inv),
                    (sqrt, x -> inv(2sqrt(x))),
                    (abs, sign),
                    (abs2, x -> 2x),
                ],
            ),
            (
                0.3,
                [
                    (asin, x -> inv(sqrt(1 - x^2))),
                    (acos, x -> -inv(sqrt(1 - x^2))),
                    (atan, x -> inv(1 + x^2)),
                    (tanh, x -> 1 - tanh(x)^2),
                    (sinh, cosh),
                    (cosh, sinh),
                ],
            ),
        ]
            for (f, df) in fns
                d = _d(v, 1.0)
                r = f(d)
                @test Nfwd.ndual_value(r) ≈ f(v)
                @test Nfwd.ndual_partial(r, 1) ≈ df(v) rtol=1e-10
            end
        end

        # exp2 / exp10 / log2 / log10
        x = _d(2.0, 1.0)
        @test Nfwd.ndual_value(exp2(x)) ≈ exp2(2.0)
        @test Nfwd.ndual_partial(exp2(x), 1) ≈ exp2(2.0) * log(2)

        # two-argument atan(y, x): ∂/∂y = x/(x²+y²), ∂/∂x = -y/(x²+y²)
        ya, xa = 3.0, 4.0  # r² = 25
        ay = _d2(ya, 1.0, 0.0)
        ax = _d2(xa, 0.0, 1.0)
        r = atan(ay, ax)
        @test Nfwd.ndual_value(r) ≈ atan(ya, xa)
        @test Nfwd.ndual_partial(r, 1) ≈ xa / (xa^2 + ya^2)   # ∂atan/∂y
        @test Nfwd.ndual_partial(r, 2) ≈ -ya / (xa^2 + ya^2)   # ∂atan/∂x

        @test Nfwd.ndual_value(log2(x)) ≈ log2(2.0)
        @test Nfwd.ndual_partial(log2(x), 1) ≈ inv(2.0 * log(2))

        @test Nfwd.ndual_value(log10(x)) ≈ log10(2.0)
        @test Nfwd.ndual_partial(log10(x), 1) ≈ inv(2.0 * log(10))

        # expm1 / log1p
        xe = _d(0.5, 1.0)
        @test Nfwd.ndual_value(expm1(xe)) ≈ expm1(0.5)
        @test Nfwd.ndual_partial(expm1(xe), 1) ≈ exp(0.5)
        @test Nfwd.ndual_value(log1p(xe)) ≈ log1p(0.5)
        @test Nfwd.ndual_partial(log1p(xe), 1) ≈ inv(1.0 + 0.5)

        # inverse hyperbolic
        for (v, fns) in [
            (0.5, [(asinh, x -> inv(sqrt(x^2 + 1))), (atanh, x -> inv(1 - x^2))]),
            (1.5, [(acosh, x -> inv(sqrt(x^2 - 1)))]),
        ]
            for (f, df) in fns
                d = _d(v, 1.0)
                r = f(d)
                @test Nfwd.ndual_value(r) ≈ f(v)
                @test Nfwd.ndual_partial(r, 1) ≈ df(v) rtol = 1e-10
            end
        end

        # sincos
        xs = _d(1.0, 1.0)
        sv, cv = sincos(xs)
        @test Nfwd.ndual_value(sv) ≈ sin(1.0)
        @test Nfwd.ndual_partial(sv, 1) ≈ cos(1.0)
        @test Nfwd.ndual_value(cv) ≈ cos(1.0)
        @test Nfwd.ndual_partial(cv, 1) ≈ -sin(1.0)

        # sinpi / cospi
        xp = _d(0.25, 1.0)
        @test Nfwd.ndual_value(sinpi(xp)) ≈ sinpi(0.25)
        @test Nfwd.ndual_partial(sinpi(xp), 1) ≈ π * cospi(0.25)
        @test Nfwd.ndual_value(cospi(xp)) ≈ cospi(0.25)
        @test Nfwd.ndual_partial(cospi(xp), 1) ≈ -π * sinpi(0.25)

        # hypot
        xh, yh = _d(3.0, 1.0), _d(4.0, 0.0)
        h = hypot(xh, yh)
        @test Nfwd.ndual_value(h) ≈ 5.0
        @test Nfwd.ndual_partial(h, 1) ≈ 3.0 / 5.0  # d/dx hypot(x,y) = x/h

        # zero tangents must stay zero at singular derivative sites
        @test Nfwd.ndual_partial(log(_d(0.0, 0.0)), 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(log(_d(0.0, 0.0)), 1))
        @test Nfwd.ndual_partial(sqrt(_d(0.0, 0.0)), 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(sqrt(_d(0.0, 0.0)), 1))
        @test Nfwd.ndual_partial(cbrt(_d(0.0, 0.0)), 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(cbrt(_d(0.0, 0.0)), 1))
        @test Nfwd.ndual_partial(log10(_d(0.0, 0.0)), 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(log10(_d(0.0, 0.0)), 1))
        @test Nfwd.ndual_partial(log2(_d(0.0, 0.0)), 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(log2(_d(0.0, 0.0)), 1))
        @test Nfwd.ndual_partial(log1p(_d(-1.0, 0.0)), 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(log1p(_d(-1.0, 0.0)), 1))

        l0 = log(_d2(2.0, 0.0, 0.0), _d2(0.0, 0.0, 0.0))
        @test Nfwd.ndual_partial(l0, 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(l0, 1))
        @test Nfwd.ndual_partial(l0, 2) === 0.0
        @test !isnan(Nfwd.ndual_partial(l0, 2))

        h0 = hypot(_d2(0.0, 0.0, 0.0), _d2(0.0, 0.0, 0.0))
        @test Nfwd.ndual_partial(h0, 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(h0, 1))
        @test Nfwd.ndual_partial(h0, 2) === 0.0
        @test !isnan(Nfwd.ndual_partial(h0, 2))
        h3 = hypot(_d2(0.0, 0.0, 0.0), _d2(0.0, 0.0, 0.0), _d2(0.0, 0.0, 0.0))
        @test Nfwd.ndual_partial(h3, 1) === 0.0
        @test !isnan(Nfwd.ndual_partial(h3, 1))

        # max / min / clamp
        a, b = _d(3.0, 1.0), _d(1.0, 0.0)
        @test Nfwd.ndual_value(max(a, b)) ≈ 3.0
        @test Nfwd.ndual_partial(max(a, b), 1) ≈ 1.0  # a wins
        @test Nfwd.ndual_value(min(a, b)) ≈ 1.0
        @test Nfwd.ndual_partial(min(a, b), 1) ≈ 0.0  # b wins
        eq1, eq2 = _d(2.0, 1.0), _d(2.0, 0.0)
        @test Nfwd.ndual_value(max(eq1, eq2)) ≈ 2.0
        @test Nfwd.ndual_partial(max(eq1, eq2), 1) ≈ 0.0
        @test Nfwd.ndual_value(min(eq1, eq2)) ≈ 2.0
        @test Nfwd.ndual_partial(min(eq1, eq2), 1) ≈ 1.0
        zpos, zneg = _d(0.0, 1.0), _d(-0.0, 0.0)
        @test isequal(Nfwd.ndual_value(max(zpos, zneg)), 0.0)
        @test Nfwd.ndual_partial(max(zpos, zneg), 1) ≈ 1.0
        @test isequal(Nfwd.ndual_value(max(zneg, zpos)), 0.0)
        @test Nfwd.ndual_partial(max(zneg, zpos), 1) ≈ 1.0
        @test isequal(Nfwd.ndual_value(min(zpos, zneg)), -0.0)
        @test Nfwd.ndual_partial(min(zpos, zneg), 1) ≈ 0.0
        @test isequal(Nfwd.ndual_value(min(zneg, zpos)), -0.0)
        @test Nfwd.ndual_partial(min(zneg, zpos), 1) ≈ 0.0

        xc = _d(2.0, 1.0)
        @test Nfwd.ndual_value(clamp(xc, 0.0, 1.0)) ≈ 1.0
        @test Nfwd.ndual_partial(clamp(xc, 0.0, 1.0), 1) ≈ 0.0  # clamped at hi
        @test Nfwd.ndual_value(clamp(xc, 3.0, 4.0)) ≈ 3.0
        @test Nfwd.ndual_partial(clamp(xc, 3.0, 4.0), 1) ≈ 0.0  # clamped at lo
        @test Nfwd.ndual_value(clamp(xc, 0.0, 3.0)) ≈ 2.0
        @test Nfwd.ndual_partial(clamp(xc, 0.0, 3.0), 1) ≈ 1.0  # pass-through
        # NDual lo/hi variant
        lo, hi = _d(0.0, 0.0), _d(1.0, 0.0)
        @test Nfwd.ndual_value(clamp(xc, lo, hi)) ≈ 1.0
        @test Nfwd.ndual_partial(clamp(xc, lo, hi), 1) ≈ 0.0

        # flipsign / copysign
        xf = _d(2.0, 1.0)
        @test Nfwd.ndual_value(flipsign(xf, _d(-1.0, 0.0))) ≈ -2.0
        @test Nfwd.ndual_partial(flipsign(xf, _d(-1.0, 0.0)), 1) ≈ -1.0
        @test Nfwd.ndual_value(flipsign(xf, _d(1.0, 0.0))) ≈ 2.0
        @test Nfwd.ndual_partial(flipsign(xf, _d(1.0, 0.0)), 1) ≈ 1.0
        @test Nfwd.ndual_value(copysign(xf, _d(-1.0, 0.0))) ≈ -2.0
        @test Nfwd.ndual_partial(copysign(xf, _d(-1.0, 0.0)), 1) ≈ -1.0
        @test Nfwd.ndual_value(copysign(xf, _d(1.0, 0.0))) ≈ 2.0
        @test Nfwd.ndual_partial(copysign(xf, _d(1.0, 0.0)), 1) ≈ 1.0
    end

    @testset "Float32" begin
        x = _d32(2.0, 1.0)
        @test Nfwd.ndual_value(sin(x)) ≈ sin(2.0f0)
        @test Nfwd.ndual_partial(sin(x), 1) ≈ cos(2.0f0)
        @test x isa NDual{Float32,1}
    end

    @testset "real / imag / conj" begin
        d = _d(3.0, 1.0)
        @test real(d) === d
        @test imag(d) == zero(d)
        @test conj(d) === d
        @test isreal(d)
    end

    @testset "comparisons" begin
        a, b = _d(1.0, 5.0), _d(2.0, -3.0)
        @test a < b
        @test b > a
        @test a <= a
        @test !isnan(a)
        @test !isinf(a)
        @test isfinite(a)
        @test signbit(_d(-1.0, 1.0))
    end

    @testset "unsupported operations" begin
        d = _d(2.5, 1.0)
        for op in (div, mod)
            @test_throws Nfwd.NDualUnsupportedError op(d)
        end
        @test_throws Nfwd.NDualUnsupportedError floor(Int, d)
        @test_throws Nfwd.NDualUnsupportedError round(Int, d)
        err = try
            div(d)
        catch e
            e
        end
        msg = sprint(showerror, err)
        @test startswith(msg, "NDual does not support `div`.")
        @test occursin("\n  │ ", msg)
        @test occursin("This operation cannot propagate partial derivatives.", msg)
    end

    @testset "unsupported output diagnostics" begin
        err = try
            Nfwd._nfwd_output_error((1.0, [2.0, 3.0]), [1, 2])
        catch err
            err
        end
        msg = sprint(showerror, err)
        @test err isa Nfwd.UnsupportedOutputError
        @test occursin("nfwd output unsupported.", msg)
        @test occursin("Supported nfwd inputs:", msg)
        @test occursin("Supported nfwd outputs:", msg)
        @test occursin("1. Float64 (scalar)", msg)
        @test occursin("2. Vector{Float64} (size (2,))", msg)
        @test occursin("Vector{Int64} (size (2,))", msg)
    end

    @testset "unsupported input diagnostics" begin
        err = try
            Nfwd._nfwd_input_error([1, 2])
        catch err
            err
        end
        msg = sprint(showerror, err)
        @test err isa Nfwd.UnsupportedInputError
        @test occursin("nfwd input unsupported.", msg)
        @test occursin("Supported nfwd inputs:", msg)
        @test occursin("Vector{Int64} (size (2,))", msg)
    end

    @testset "Complex{NDual}" begin
        # Complex{NDual{T,N}} — each component carries its own partials.
        # Slot 1 = Re(z), slot 2 = Im(z).
        re = NDual{Float64,2}(3.0, (1.0, 0.0))
        im_ = NDual{Float64,2}(4.0, (0.0, 1.0))
        z = complex(re, im_)
        a, b = 3.0, 4.0  # primal values

        # isbits — critical for GPU register allocation
        @test isbitstype(typeof(z))

        # abs2(z) = re^2 + im^2, d/dRe = 2*re, d/dIm = 2*im
        r = abs2(z)
        @test Nfwd.ndual_value(r) ≈ 25.0
        @test Nfwd.ndual_partial(r, 1) ≈ 6.0   # 2 * re
        @test Nfwd.ndual_partial(r, 2) ≈ 8.0   # 2 * im

        # abs(z) = hypot(re, im)
        r = abs(z)
        @test Nfwd.ndual_value(r) ≈ 5.0
        @test Nfwd.ndual_partial(r, 1) ≈ a / 5.0   # re/|z|
        @test Nfwd.ndual_partial(r, 2) ≈ b / 5.0   # im/|z|

        # conj(z) = re - im*i — partials flip sign on imag part
        cz = conj(z)
        @test Nfwd.ndual_value(real(cz)) ≈ 3.0
        @test Nfwd.ndual_value(imag(cz)) ≈ -4.0
        @test Nfwd.ndual_partial(real(cz), 1) ≈ 1.0
        @test Nfwd.ndual_partial(imag(cz), 2) ≈ -1.0

        # z * conj(z) = abs2(z) as a real NDual
        r2 = real(z * conj(z))
        @test Nfwd.ndual_value(r2) ≈ 25.0

        # Helper: check value and 2-slot Jacobian against reference complex function
        function _check_cx(f, z, zv)
            r = f(z)
            rv = f(zv)
            @test Nfwd.ndual_value(real(r)) ≈ real(rv) rtol=1e-10
            @test Nfwd.ndual_value(imag(r)) ≈ imag(rv) rtol=1e-10
            ε = 1e-7
            ∂re_re = (real(f(complex(real(zv)+ε, imag(zv)))) - real(rv)) / ε
            ∂re_im = (real(f(complex(real(zv), imag(zv)+ε))) - real(rv)) / ε
            ∂im_re = (imag(f(complex(real(zv)+ε, imag(zv)))) - imag(rv)) / ε
            ∂im_im = (imag(f(complex(real(zv), imag(zv)+ε))) - imag(rv)) / ε
            @test Nfwd.ndual_partial(real(r), 1) ≈ ∂re_re rtol=1e-5
            @test Nfwd.ndual_partial(real(r), 2) ≈ ∂re_im rtol=1e-5
            @test Nfwd.ndual_partial(imag(r), 1) ≈ ∂im_re rtol=1e-5
            @test Nfwd.ndual_partial(imag(r), 2) ≈ ∂im_im rtol=1e-5
        end

        zv = complex(a, b)

        _check_cx(sin, z, zv)
        sz = sin(z)
        @test Nfwd.ndual_partial(real(sz), 1) ≈ cos(a)*cosh(b) rtol=1e-10
        @test Nfwd.ndual_partial(real(sz), 2) ≈ sin(a)*sinh(b) rtol=1e-10
        @test Nfwd.ndual_partial(imag(sz), 1) ≈ -sin(a)*sinh(b) rtol=1e-10
        @test Nfwd.ndual_partial(imag(sz), 2) ≈ cos(a)*cosh(b) rtol=1e-10

        _check_cx(cos, z, zv)
        _check_cx(exp, z, zv)
        ez = exp(z)
        @test Nfwd.ndual_partial(real(ez), 1) ≈ exp(a)*cos(b) rtol=1e-10
        @test Nfwd.ndual_partial(real(ez), 2) ≈ -exp(a)*sin(b) rtol=1e-10
        @test Nfwd.ndual_partial(imag(ez), 1) ≈ exp(a)*sin(b) rtol=1e-10
        @test Nfwd.ndual_partial(imag(ez), 2) ≈ exp(a)*cos(b) rtol=1e-10

        _check_cx(log, z, zv)
        _check_cx(sqrt, z, zv)
        _check_cx(tan, z, zv)

        # Float32 variant
        re32 = NDual{Float32,2}(3.0f0, (1.0f0, 0.0f0))
        im32 = NDual{Float32,2}(4.0f0, (0.0f0, 1.0f0))
        z32 = complex(re32, im32)
        sz32 = sin(z32)
        @test sz32 isa Complex{NDual{Float32,2}}
        @test Nfwd.ndual_value(real(sz32)) ≈ real(sin(complex(3.0f0, 4.0f0))) rtol=1e-5
    end

    @testset "chunk mode: N=3" begin
        x = NDual{Float64,3}(2.0, (1.0, 0.0, 0.0))
        y = NDual{Float64,3}(3.0, (0.0, 1.0, 0.0))
        c = NDual{Float64,3}(5.0, (0.0, 0.0, 1.0))

        r = c * sin(x) * exp(y)
        v = 5.0 * sin(2.0) * exp(3.0)
        @test Nfwd.ndual_value(r) ≈ v
        @test Nfwd.ndual_partial(r, 1) ≈ 5.0 * cos(2.0) * exp(3.0)
        @test Nfwd.ndual_partial(r, 2) ≈ 5.0 * sin(2.0) * exp(3.0)
        @test Nfwd.ndual_partial(r, 3) ≈ sin(2.0) * exp(3.0)
    end

    @testset "reciprocal trig" begin
        x = _d(0.8, 1.0)
        @test Nfwd.ndual_value(sec(x)) ≈ sec(0.8)
        @test Nfwd.ndual_partial(sec(x), 1) ≈ sec(0.8) * tan(0.8)
        @test Nfwd.ndual_value(csc(x)) ≈ csc(0.8)
        @test Nfwd.ndual_partial(csc(x), 1) ≈ -csc(0.8) * cot(0.8)
        @test Nfwd.ndual_value(cot(x)) ≈ cot(0.8)
        @test Nfwd.ndual_partial(cot(x), 1) ≈ -(1 + cot(0.8)^2)

        y = _d(1.5, 1.0)
        @test Nfwd.ndual_value(asec(y)) ≈ asec(1.5)
        @test Nfwd.ndual_partial(asec(y), 1) ≈ inv(abs(1.5) * sqrt(1.5^2 - 1))
        @test Nfwd.ndual_value(acsc(y)) ≈ acsc(1.5)
        @test Nfwd.ndual_partial(acsc(y), 1) ≈ -inv(abs(1.5) * sqrt(1.5^2 - 1))
        @test Nfwd.ndual_value(acot(x)) ≈ acot(0.8)
        @test Nfwd.ndual_partial(acot(x), 1) ≈ -inv(1 + 0.8^2)
    end

    @testset "reciprocal hyperbolic" begin
        x = _d(0.5, 1.0)
        @test Nfwd.ndual_value(sech(x)) ≈ sech(0.5)
        @test Nfwd.ndual_partial(sech(x), 1) ≈ -tanh(0.5) * sech(0.5)
        @test Nfwd.ndual_value(csch(x)) ≈ csch(0.5)
        @test Nfwd.ndual_partial(csch(x), 1) ≈ -coth(0.5) * csch(0.5)
        @test Nfwd.ndual_value(coth(x)) ≈ coth(0.5)
        @test Nfwd.ndual_partial(coth(x), 1) ≈ -(csch(0.5)^2)

        z = _d(0.4, 1.0)
        @test Nfwd.ndual_value(asech(z)) ≈ asech(0.4)
        @test Nfwd.ndual_partial(asech(z), 1) ≈ -inv(0.4 * sqrt(1 - 0.4^2))
        @test Nfwd.ndual_value(acsch(x)) ≈ acsch(0.5)
        @test Nfwd.ndual_partial(acsch(x), 1) ≈ -inv(abs(0.5) * sqrt(1 + 0.5^2))
        @test Nfwd.ndual_value(acoth(_d(2.0, 1.0))) ≈ acoth(2.0)
        @test Nfwd.ndual_partial(acoth(_d(2.0, 1.0)), 1) ≈ inv(1 - 2.0^2)
    end

    @testset "degree-based trig" begin
        x = _d(30.0, 1.0)
        @test Nfwd.ndual_value(sind(x)) ≈ sind(30.0)
        @test Nfwd.ndual_partial(sind(x), 1) ≈ deg2rad(cosd(30.0))
        @test Nfwd.ndual_value(cosd(x)) ≈ cosd(30.0)
        @test Nfwd.ndual_partial(cosd(x), 1) ≈ -deg2rad(sind(30.0))
        @test Nfwd.ndual_value(tand(x)) ≈ tand(30.0)
        @test Nfwd.ndual_partial(tand(x), 1) ≈ deg2rad(1 + tand(30.0)^2)

        y = _d(0.5, 1.0)
        @test Nfwd.ndual_value(asind(y)) ≈ asind(0.5)
        @test Nfwd.ndual_partial(asind(y), 1) ≈ inv(deg2rad(sqrt(1 - 0.5^2)))
        @test Nfwd.ndual_value(acosd(y)) ≈ acosd(0.5)
        @test Nfwd.ndual_partial(acosd(y), 1) ≈ -inv(deg2rad(sqrt(1 - 0.5^2)))
        @test Nfwd.ndual_value(atand(y)) ≈ atand(0.5)
        @test Nfwd.ndual_partial(atand(y), 1) ≈ inv(deg2rad(1 + 0.5^2))
    end

    @testset "angle conversions" begin
        x = _d(90.0, 1.0)
        @test Nfwd.ndual_value(deg2rad(x)) ≈ deg2rad(90.0)
        @test Nfwd.ndual_partial(deg2rad(x), 1) ≈ deg2rad(1.0)
        @test Nfwd.ndual_value(rad2deg(x)) ≈ rad2deg(90.0)
        @test Nfwd.ndual_partial(rad2deg(x), 1) ≈ rad2deg(1.0)
    end

    @testset "sinc" begin
        x = _d(0.5, 1.0)
        @test Nfwd.ndual_value(sinc(x)) ≈ sinc(0.5)
        @test Nfwd.ndual_partial(sinc(x), 1) ≈ cosc(0.5)
    end

    @testset "two-arg log and ldexp" begin
        x = _d(4.0, 1.0)
        @test Nfwd.ndual_value(log(2, x)) ≈ log(2, 4.0)
        @test Nfwd.ndual_partial(log(2, x), 1) ≈ inv(4.0 * log(2))

        b = _d2(2.0, 1.0, 0.0)
        x2 = _d2(4.0, 0.0, 1.0)
        r = log(b, x2)
        @test Nfwd.ndual_value(r) ≈ log(2.0, 4.0)
        @test Nfwd.ndual_partial(r, 1) ≈ -log(2.0, 4.0) / (2.0 * log(2.0))
        @test Nfwd.ndual_partial(r, 2) ≈ inv(4.0 * log(2.0))

        y = _d(1.5, 1.0)
        @test Nfwd.ndual_value(ldexp(y, 3)) ≈ ldexp(1.5, 3)
        @test Nfwd.ndual_partial(ldexp(y, 3), 1) ≈ exp2(3)
    end

    @testset "scalar-base power" begin
        a = _d(2.0, 1.0)
        r = 3.0^a
        @test Nfwd.ndual_value(r) ≈ 3.0^2.0
        @test Nfwd.ndual_partial(r, 1) ≈ 3.0^2.0 * log(3.0)
    end

    @testset "utility: eps, iszero, hash" begin
        x = _d(1.0, 0.0)
        @test eps(x) === eps(1.0)
        @test eps(NDual{Float64,1}) === eps(Float64)
        @test iszero(NDual{Float64,1}(0.0, (0.0,)))
        @test !iszero(NDual{Float64,1}(0.0, (1.0,)))
        @test !iszero(NDual{Float64,1}(1.0, (0.0,)))
        # -0.0 partials must also be treated as zero (==-based, not ===-based)
        @test iszero(NDual{Float64,1}(0.0, (-0.0,)))
        @test hash(_d(3.0, 1.0), UInt(0)) == hash(3.0, UInt(0))
    end

    @testset "precision / nextfloat / exponent" begin
        d = NDual{Float64,2}(3.0, (1.0, 0.0))
        @test Base.precision(NDual{Float64,2}) === precision(Float64)
        @test Base.precision(d) === precision(Float64)
        # nextfloat/prevfloat advance the value by one representable step and keep the
        # partials unchanged.
        nd = nextfloat(d)
        @test Nfwd.ndual_value(nd) === nextfloat(3.0)
        @test nd.partials === (1.0, 0.0)
        pd = prevfloat(d)
        @test Nfwd.ndual_value(pd) === prevfloat(3.0)
        @test pd.partials === (1.0, 0.0)
        @test Nfwd.ndual_value(nextfloat(zero(d))) === nextfloat(0.0)
        # exponent returns an Int, not an NDual
        @test Base.exponent(d) === exponent(3.0)
    end

    @testset "muladd and fma" begin
        a = _d2(2.0, 1.0, 0.0)   # 2 + e1
        b = _d2(3.0, 0.0, 1.0)   # 3 + e2
        c = _d2(1.0, 0.0, 0.0)   # 1 (constant)

        # muladd(a, b, c): value = 2*3+1 = 7; partials = (b.v=3, a.v=2, 0)
        r = muladd(a, b, c)
        @test Nfwd.ndual_value(r) ≈ 7.0
        @test Nfwd.ndual_partial(r, 1) ≈ 3.0
        @test Nfwd.ndual_partial(r, 2) ≈ 2.0

        # fma(a, b, c): same values as muladd (guaranteed single instruction)
        r = fma(a, b, c)
        @test Nfwd.ndual_value(r) ≈ 7.0
        @test Nfwd.ndual_partial(r, 1) ≈ 3.0
        @test Nfwd.ndual_partial(r, 2) ≈ 2.0

        # muladd/fma(Real, NDual, NDual): value = 2*3+1 = 7; only b's partials scaled
        r = muladd(2.0, b, c)
        @test Nfwd.ndual_value(r) ≈ 7.0
        @test Nfwd.ndual_partial(r, 1) ≈ 0.0
        @test Nfwd.ndual_partial(r, 2) ≈ 2.0
        r = fma(2.0, b, c)
        @test Nfwd.ndual_value(r) ≈ 7.0
        @test Nfwd.ndual_partial(r, 1) ≈ 0.0
        @test Nfwd.ndual_partial(r, 2) ≈ 2.0

        # muladd/fma(NDual, Real, NDual): value = 2*3+1 = 7; only a's partials scaled
        r = muladd(a, 3.0, c)
        @test Nfwd.ndual_value(r) ≈ 7.0
        @test Nfwd.ndual_partial(r, 1) ≈ 3.0
        @test Nfwd.ndual_partial(r, 2) ≈ 0.0
        r = fma(a, 3.0, c)
        @test Nfwd.ndual_value(r) ≈ 7.0
        @test Nfwd.ndual_partial(r, 1) ≈ 3.0
        @test Nfwd.ndual_partial(r, 2) ≈ 0.0

        # consistency: muladd and fma agree with a*b+c
        @test muladd(a, b, c) == a * b + c
        @test fma(a, b, c) == a * b + c
        @test muladd(2.0, b, c) == 2.0 * b + c
        @test fma(2.0, b, c) == 2.0 * b + c
        @test muladd(a, 3.0, c) == a * 3.0 + c
        @test fma(a, 3.0, c) == a * 3.0 + c
    end

    @testset "LinearAlgebra.dot" begin
        # xd = [3+t·1, 4+t·0], yd = [1+t·0, 2+t·1]
        # dot = 3·1 + 4·2 = 11; ∂dot/∂t = 1·1 + 4·1 = 5
        xd = [_d(3.0, 1.0), _d(4.0, 0.0)]
        yd = [_d(1.0, 0.0), _d(2.0, 1.0)]
        d = LinearAlgebra.dot(xd, yd)
        @test Nfwd.ndual_value(d) ≈ 3.0 * 1.0 + 4.0 * 2.0   # = 11
        @test Nfwd.ndual_partial(d, 1) ≈ 1.0 + 4.0            # = 5

        # dot(x, x) = sum(x[i]^2); ∂dot/∂t = 2·3·1 + 2·4·0 = 6
        d2 = LinearAlgebra.dot(xd, xd)
        @test Nfwd.ndual_value(d2) ≈ 3.0^2 + 4.0^2            # = 25
        @test Nfwd.ndual_partial(d2, 1) ≈ 2.0 * 3.0           # = 6

        # empty input returns zero NDual
        empty = NDual{Float64,1}[]
        @test LinearAlgebra.dot(empty, empty) == NDual{Float64,1}(0.0)

        # dimension mismatch throws
        @test_throws DimensionMismatch LinearAlgebra.dot(xd, [_d(1.0, 0.0)])
    end

    @testset "LinearAlgebra.ldiv (LU{Float64} backslash Vector{NDual})" begin
        # Verify that LU{Float64} \ Vector{NDual} uses the Float64-coefficient path
        # instead of converting to LU{NDual}, and produces correct values and partials.
        A = [4.0 1.0; 1.0 3.0]   # SPD 2×2
        F = lu(A)
        # x = NDual with value [1.0, 2.0]; seed on slot 1
        xd = [_d2(1.0, 1.0, 0.0), _d2(2.0, 0.0, 1.0)]
        # expected: A \ [1,2] = [0.2, 0.6] (since A = [4 1; 1 3], det=11)
        y_val = A \ [1.0, 2.0]
        yd = F \ xd
        @test Nfwd.ndual_value(yd[1]) ≈ y_val[1]
        @test Nfwd.ndual_value(yd[2]) ≈ y_val[2]
        # partial w.r.t. slot 1: A \ [1,0]
        dy1 = A \ [1.0, 0.0]
        @test Nfwd.ndual_partial(yd[1], 1) ≈ dy1[1]
        @test Nfwd.ndual_partial(yd[2], 1) ≈ dy1[2]
        # partial w.r.t. slot 2: A \ [0,1]
        dy2 = A \ [0.0, 1.0]
        @test Nfwd.ndual_partial(yd[1], 2) ≈ dy2[1]
        @test Nfwd.ndual_partial(yd[2], 2) ≈ dy2[2]
    end

    @testset "cholesky(Matrix{NDual})" begin
        # 2×2 SPD matrix A₀ with N=3 independent perturbation directions:
        #   slot 1 ↔ ∂/∂A₁₁, slot 2 ↔ ∂/∂A₁₂ (symmetric), slot 3 ↔ ∂/∂A₂₂
        a11, a12, a22 = 4.0, 2.0, 3.0
        A₀ = [a11 a12; a12 a22]
        A_nd = [
            NDual{Float64,3}(a11, (1.0, 0.0, 0.0)) NDual{Float64,3}(a12, (0.0, 1.0, 0.0));
            NDual{Float64,3}(a12, (0.0, 1.0, 0.0)) NDual{Float64,3}(a22, (0.0, 0.0, 1.0))
        ]

        F_nd = cholesky(A_nd)
        L_nd = F_nd.L
        L₀ = Matrix(cholesky(Hermitian(A₀)).L)

        # Primal values match Float64 reference
        @test Nfwd.ndual_value(L_nd[1, 1]) ≈ L₀[1, 1]
        @test Nfwd.ndual_value(L_nd[2, 1]) ≈ L₀[2, 1]
        @test Nfwd.ndual_value(L_nd[2, 2]) ≈ L₀[2, 2]
        @test Nfwd.ndual_value(L_nd[1, 2]) ≈ 0.0  # upper triangle zero

        # Partials verified by finite differences
        ε = 1e-7
        for (k, δA) in enumerate([
            [ε 0.0; 0.0 0.0],   # slot 1: ∂/∂A₁₁
            [0.0 ε; ε 0.0],     # slot 2: ∂/∂A₁₂ (symmetric)
            [0.0 0.0; 0.0 ε],   # slot 3: ∂/∂A₂₂
        ])
            L_pert = Matrix(cholesky(Hermitian(A₀ + δA)).L)
            L_dot = (L_pert - L₀) / ε
            @test Nfwd.ndual_partial(L_nd[1, 1], k) ≈ L_dot[1, 1] rtol = 1e-5
            @test Nfwd.ndual_partial(L_nd[2, 1], k) ≈ L_dot[2, 1] rtol = 1e-5
            @test Nfwd.ndual_partial(L_nd[2, 2], k) ≈ L_dot[2, 2] rtol = 1e-5
        end

        # Symmetric{NDual} and Hermitian{NDual} wrappers dispatch correctly
        for F_wrap in (cholesky(Hermitian(A_nd)), cholesky(Symmetric(A_nd)))
            @test Nfwd.ndual_value(F_wrap.L[1, 1]) ≈ L₀[1, 1]
            @test Nfwd.ndual_value(F_wrap.L[2, 1]) ≈ L₀[2, 1]
            @test Nfwd.ndual_value(F_wrap.L[2, 2]) ≈ L₀[2, 2]
        end

        # logdet(Cholesky{NDual}) = log(det(A₀)) verified by finite differences
        ld_nd = logdet(F_nd)
        @test Nfwd.ndual_value(ld_nd) ≈ logdet(A₀)
        ε = 1e-7
        for (k, δA) in enumerate([[ε 0.0; 0.0 0.0], [0.0 ε; ε 0.0], [0.0 0.0; 0.0 ε]])
            ld_pert = logdet(A₀ + δA)
            @test Nfwd.ndual_partial(ld_nd, k) ≈ (ld_pert - logdet(A₀)) / ε rtol = 1e-5
        end

        # Float32 sanity check
        A_nd32 = [
            NDual{Float32,1}(Float32(a11), (1.0f0,)) NDual{Float32,1}(Float32(a12), (0.0f0,));
            NDual{Float32,1}(Float32(a12), (0.0f0,)) NDual{Float32,1}(Float32(a22), (0.0f0,))
        ]
        F32 = cholesky(A_nd32)
        @test F32.L[1, 1] isa NDual{Float32,1}
        @test Nfwd.ndual_value(F32.L[1, 1]) ≈ Float32(L₀[1, 1])
    end

    @testset "Symmetric / Hermitian matrix multiply with NDual" begin
        # A_nd is a Symmetric 2×2 matrix; B is a plain Float64 matrix.
        # LinearAlgebra's BLAS path for Symmetric mul doesn't support NDual elements;
        # the materialise-then-multiply rules should intercept this.
        a11, a12, a22 = 4.0, 2.0, 3.0
        A_nd = Symmetric(
            [
                NDual{Float64,3}(a11, (1.0, 0.0, 0.0)) NDual{Float64,3}(a12, (0.0, 1.0, 0.0));
                NDual{Float64,3}(a12, (0.0, 1.0, 0.0)) NDual{Float64,3}(a22, (0.0, 0.0, 1.0))
            ],
        )
        A₀ = [a11 a12; a12 a22]
        B = [1.0 0.0; 0.0 2.0]

        # (Symmetric{NDual}) * Matrix{Float64}
        C1 = A_nd * B
        C_ref = A₀ * B
        @test Nfwd.ndual_value.(C1) ≈ C_ref

        # Matrix{Float64} * (Symmetric{NDual})
        C2 = B * A_nd
        C_ref2 = B * A₀
        @test Nfwd.ndual_value.(C2) ≈ C_ref2

        # Hermitian{NDual} * Matrix{Float64}
        A_herm = Hermitian(Matrix(A_nd))
        C3 = A_herm * B
        @test Nfwd.ndual_value.(C3) ≈ C_ref

        # Partials: (Symmetric{NDual}) * [1 0; 0 1] = Matrix(A_nd), so ∂C/∂A₁₁ slot 1
        I2 = Matrix(1.0I, 2, 2)
        Cp = A_nd * I2
        @test Nfwd.ndual_partial(Cp[1, 1], 1) ≈ 1.0   # ∂A₁₁
        @test Nfwd.ndual_partial(Cp[1, 2], 2) ≈ 1.0   # ∂A₁₂
        @test Nfwd.ndual_partial(Cp[2, 2], 3) ≈ 1.0   # ∂A₂₂
    end
end

# Slot traversal contract tests — verify _fold_slots and _unfold_slots
# agree on canonical order and produce correct results for all supported types.
@testset "slot traversal" begin
    using Mooncake.Nfwd: _fold_slots, _unfold_slots, _nfwd_input_dof

    count_slot(acc, _leaf, _slot, st) = (acc + 1, st)

    # helper: collect global slot indices via fold
    function fold_order(x)
        collect_order(acc, _leaf, _slot, st) = (push!(acc, st), st + 1)
        order, _ = _fold_slots(collect_order, Int[], x, 1)
        return order
    end

    # helper: collect global slot indices via unfold
    function unfold_order(x)
        function collect_leaf(leaf, (order, cursor))
            dof = _nfwd_input_dof(leaf)
            append!(order, cursor:(cursor + dof - 1))
            return nothing, (order, cursor + dof)
        end
        _, (order, _) = _unfold_slots(collect_leaf, x, (Int[], 1))
        return order
    end

    @testset "real scalar" begin
        @test _fold_slots(count_slot, 0, 1.0, nothing) == (1, nothing)
        @test _nfwd_input_dof(1.0) == 1
        @test fold_order(1.0) == [1]
        @test unfold_order(1.0) == [1]
    end

    @testset "complex scalar" begin
        z = 1.0 + 2.0im
        @test _fold_slots(count_slot, 0, z, nothing) == (2, nothing)
        @test _nfwd_input_dof(z) == 2
        @test fold_order(z) == [1, 2]
        @test unfold_order(z) == [1, 2]
    end

    @testset "dense real array" begin
        a = [1.0, 2.0, 3.0]
        @test _fold_slots(count_slot, 0, a, nothing) == (3, nothing)
        @test _nfwd_input_dof(a) == 3
        @test fold_order(a) == [1, 2, 3]
        @test unfold_order(a) == [1, 2, 3]
    end

    @testset "dense complex array" begin
        a = [1.0+0im, 2.0+3.0im]
        @test _fold_slots(count_slot, 0, a, nothing) == (4, nothing)
        @test _nfwd_input_dof(a) == 4
        @test fold_order(a) == [1, 2, 3, 4]
        @test unfold_order(a) == [1, 2, 3, 4]
    end

    @testset "tuple mixtures" begin
        t = (1.0, [2.0, 3.0], 4.0 + 5.0im)
        @test _fold_slots(count_slot, 0, t, nothing) == (5, nothing)
        @test _nfwd_input_dof(t) == 5
        @test fold_order(t) == [1, 2, 3, 4, 5]
        @test unfold_order(t) == [1, 2, 3, 4, 5]

        # nested tuple
        t2 = ((1.0, 2.0), [3.0 + 0im])
        @test _nfwd_input_dof(t2) == 4
        @test fold_order(t2) == [1, 2, 3, 4]
        @test unfold_order(t2) == [1, 2, 3, 4]

        # empty tuple
        @test _fold_slots(count_slot, 0, (), nothing) == (0, nothing)
        @test _nfwd_input_dof(()) == 0
        @test fold_order(()) == Int[]
        @test unfold_order(()) == Int[]
    end

    @testset "fold and unfold order agree" begin
        inputs = [
            1.0,
            1.0 + 2.0im,
            [1.0, 2.0, 3.0],
            [1.0+0im, 2.0+0im],
            (1.0, [2.0, 3.0], 4.0+5.0im),
            ((1.0, 2.0), [3.0+0im]),
            (),
        ]
        for x in inputs
            @test fold_order(x) == unfold_order(x)
        end
    end

    @testset "unfold structural rebuild" begin
        # unfold with identity preserves values
        function id_leaf(x, st)
            return x, st + _nfwd_input_dof(x)
        end

        val, st = _unfold_slots(id_leaf, 3.14, 0)
        @test val === 3.14

        val, st = _unfold_slots(id_leaf, 1.0+2.0im, 0)
        @test val === 1.0+2.0im

        a = [1.0, 2.0, 3.0]
        val, st = _unfold_slots(id_leaf, a, 0)
        @test val == a

        t = (1.0, [2.0, 3.0], 4.0+5.0im)
        val, st = _unfold_slots(id_leaf, t, 0)
        @test val[1] === 1.0
        @test val[2] == [2.0, 3.0]
        @test val[3] === 4.0+5.0im
        @test st == 5
    end

    @testset "fold accumulates correctly" begin
        # Sum all slot values (real parts for complex)
        function sum_slots(acc, leaf, slot, st)
            if leaf isa Complex
                val = slot == 1 ? real(leaf) : imag(leaf)
            elseif leaf isa AbstractArray && eltype(leaf) <: Complex
                elem = cld(slot, 2)
                val = isodd(slot) ? real(leaf[elem]) : imag(leaf[elem])
            elseif leaf isa AbstractArray
                val = leaf[slot]
            else
                val = leaf
            end
            return (acc + val, st)
        end

        total, _ = _fold_slots(sum_slots, 0.0, 3.0, nothing)
        @test total ≈ 3.0

        total, _ = _fold_slots(sum_slots, 0.0, 1.0 + 2.0im, nothing)
        @test total ≈ 3.0  # 1.0 + 2.0

        total, _ = _fold_slots(sum_slots, 0.0, [1.0, 2.0, 3.0], nothing)
        @test total ≈ 6.0

        total, _ = _fold_slots(sum_slots, 0.0, (1.0, [2.0, 3.0]), nothing)
        @test total ≈ 6.0
    end

    @testset "Float32 support" begin
        @test _nfwd_input_dof(1.0f0) == 1
        @test _nfwd_input_dof(Float32[1, 2, 3]) == 3
        @test _nfwd_input_dof(1.0f0 + 2.0f0im) == 2
        @test fold_order((1.0f0, Float32[2, 3])) == [1, 2, 3]
        @test unfold_order((1.0f0, Float32[2, 3])) == [1, 2, 3]
    end
end
