# NDual unit tests — these test pure arithmetic on the NDual type defined in the
# CUDA extension; no GPU device is required.
@testset "NDual" begin
    # helpers
    _d(v, p1) = NDual{Float64,1}(v, (p1,))
    _d2(v, p1, p2) = NDual{Float64,2}(v, (p1, p2))
    _d32(v, p1) = NDual{Float32,1}(Float32(v), (Float32(p1),))

    @testset "construction and accessors" begin
        d = NDual{Float64,2}(3.0, (1.0, 0.0))
        @test ndual_value(d) === 3.0
        @test ndual_partial(d, 1) === 1.0
        @test ndual_partial(d, 2) === 0.0
        @test d.partials === (1.0, 0.0)

        # scalar constructor (zero partials)
        dc = NDual{Float64,2}(3.0)
        @test ndual_value(dc) === 3.0
        @test dc.partials === (0.0, 0.0)

        # isbits — critical for GPU register allocation
        @test isbits(NDual{Float64,2}(1.0, (0.0, 0.0)))
        @test isbits(NDual{Float32,3}(1.0f0, (0.0f0, 0.0f0, 0.0f0)))
    end

    @testset "zero / one" begin
        @test zero(NDual{Float64,2}) == NDual{Float64,2}(0.0, (0.0, 0.0))
        @test one(NDual{Float64,2}) == NDual{Float64,2}(1.0, (0.0, 0.0))
        @test zero(_d(1.0, 2.0)) == NDual{Float64,1}(0.0, (0.0,))
        @test one(_d(1.0, 2.0)) == NDual{Float64,1}(1.0, (0.0,))
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
        @test ndual_value(d64) === 2.0
        @test ndual_partial(d64, 1) === 1.0
        @test ndual_partial(d64, 2) === 0.0

        # Arithmetic between different precisions auto-promotes
        a32 = NDual{Float32,1}(2.0f0, (1.0f0,))
        b64 = NDual{Float64,1}(3.0, (0.0,))
        r = a32 + b64
        @test r isa NDual{Float64,1}
        @test ndual_value(r) ≈ 5.0
        @test ndual_partial(r, 1) ≈ 1.0

        r2 = a32 * b64
        @test r2 isa NDual{Float64,1}
        @test ndual_value(r2) ≈ 6.0
        @test ndual_partial(r2, 1) ≈ 3.0  # b.value * da

        # Float64 literal mixed with NDual{Float32} (GPU broadcast scenario)
        lit = 2.0  # Float64
        r3 = lit * a32
        @test r3 isa NDual{Float64,1}
        @test ndual_value(r3) ≈ 4.0
        @test ndual_partial(r3, 1) ≈ 2.0
    end

    @testset "arithmetic" begin
        a = _d2(2.0, 1.0, 0.0)   # represents 2 + 1*e1
        b = _d2(3.0, 0.0, 1.0)   # represents 3 + 1*e2

        @test a + b == _d2(5.0, 1.0, 1.0)
        @test a - b == _d2(-1.0, 1.0, -1.0)
        @test -a == _d2(-2.0, -1.0, 0.0)

        # product rule: d(a*b) = a*db + b*da
        r = a * b
        @test ndual_value(r) ≈ 6.0
        @test ndual_partial(r, 1) ≈ 3.0  # b.value * da/de1
        @test ndual_partial(r, 2) ≈ 2.0  # a.value * db/de2

        # quotient rule: d(a/b) = (da - (a/b)*db) / b
        r = a / b
        @test ndual_value(r) ≈ 2.0 / 3.0
        @test ndual_partial(r, 1) ≈ 1.0 / 3.0
        @test ndual_partial(r, 2) ≈ -(2.0 / 3.0) / 3.0

        # mixing with plain numbers via promotion
        @test ndual_value(a + 1.0) ≈ 3.0
        @test ndual_partial(a + 1.0, 1) ≈ 1.0
        @test ndual_value(2.0 * a) ≈ 4.0
        @test ndual_partial(2.0 * a, 1) ≈ 2.0
    end

    @testset "power" begin
        x = _d(3.0, 1.0)
        # d(x^2)/dx = 2x
        @test ndual_value(x^2) ≈ 9.0
        @test ndual_partial(x^2, 1) ≈ 6.0
        # d(x^3)/dx = 3x^2
        @test ndual_value(x^3) ≈ 27.0
        @test ndual_partial(x^3, 1) ≈ 27.0
        # x^0
        @test ndual_value(x^0) ≈ 1.0
        @test ndual_partial(x^0, 1) ≈ 0.0
        # real exponent
        @test ndual_value(x^2.0) ≈ 9.0
        @test ndual_partial(x^2.0, 1) ≈ 6.0
        # real exponent b=0.0: d(x^0)/dx = 0 everywhere, including x=0 (no NaN)
        @test ndual_partial(_d(0.0, 1.0)^0.0, 1) === 0.0
        @test !isnan(ndual_partial(_d(0.0, 1.0)^0.0, 1))
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
                @test ndual_value(r) ≈ f(v)
                @test ndual_partial(r, 1) ≈ df(v) rtol=1e-10
            end
        end

        # exp2 / exp10 / log2 / log10
        x = _d(2.0, 1.0)
        @test ndual_value(exp2(x)) ≈ exp2(2.0)
        @test ndual_partial(exp2(x), 1) ≈ exp2(2.0) * log(2)

        # two-argument atan(y, x): ∂/∂y = x/(x²+y²), ∂/∂x = -y/(x²+y²)
        ya, xa = 3.0, 4.0  # r² = 25
        ay = _d2(ya, 1.0, 0.0)
        ax = _d2(xa, 0.0, 1.0)
        r = atan(ay, ax)
        @test ndual_value(r) ≈ atan(ya, xa)
        @test ndual_partial(r, 1) ≈ xa / (xa^2 + ya^2)   # ∂atan/∂y
        @test ndual_partial(r, 2) ≈ -ya / (xa^2 + ya^2)   # ∂atan/∂x

        @test ndual_value(log2(x)) ≈ log2(2.0)
        @test ndual_partial(log2(x), 1) ≈ inv(2.0 * log(2))

        @test ndual_value(log10(x)) ≈ log10(2.0)
        @test ndual_partial(log10(x), 1) ≈ inv(2.0 * log(10))

        # expm1 / log1p
        xe = _d(0.5, 1.0)
        @test ndual_value(expm1(xe)) ≈ expm1(0.5)
        @test ndual_partial(expm1(xe), 1) ≈ exp(0.5)
        @test ndual_value(log1p(xe)) ≈ log1p(0.5)
        @test ndual_partial(log1p(xe), 1) ≈ inv(1.0 + 0.5)

        # inverse hyperbolic
        for (v, fns) in [
            (0.5, [(asinh, x -> inv(sqrt(x^2 + 1))), (atanh, x -> inv(1 - x^2))]),
            (1.5, [(acosh, x -> inv(sqrt(x^2 - 1)))]),
        ]
            for (f, df) in fns
                d = _d(v, 1.0)
                r = f(d)
                @test ndual_value(r) ≈ f(v)
                @test ndual_partial(r, 1) ≈ df(v) rtol = 1e-10
            end
        end

        # sincos
        xs = _d(1.0, 1.0)
        sv, cv = sincos(xs)
        @test ndual_value(sv) ≈ sin(1.0)
        @test ndual_partial(sv, 1) ≈ cos(1.0)
        @test ndual_value(cv) ≈ cos(1.0)
        @test ndual_partial(cv, 1) ≈ -sin(1.0)

        # sinpi / cospi
        xp = _d(0.25, 1.0)
        @test ndual_value(sinpi(xp)) ≈ sinpi(0.25)
        @test ndual_partial(sinpi(xp), 1) ≈ π * cospi(0.25)
        @test ndual_value(cospi(xp)) ≈ cospi(0.25)
        @test ndual_partial(cospi(xp), 1) ≈ -π * sinpi(0.25)

        # hypot
        xh, yh = _d(3.0, 1.0), _d(4.0, 0.0)
        h = hypot(xh, yh)
        @test ndual_value(h) ≈ 5.0
        @test ndual_partial(h, 1) ≈ 3.0 / 5.0  # d/dx hypot(x,y) = x/h

        # max / min / clamp
        a, b = _d(3.0, 1.0), _d(1.0, 0.0)
        @test ndual_value(max(a, b)) ≈ 3.0
        @test ndual_partial(max(a, b), 1) ≈ 1.0  # a wins
        @test ndual_value(min(a, b)) ≈ 1.0
        @test ndual_partial(min(a, b), 1) ≈ 0.0  # b wins

        xc = _d(2.0, 1.0)
        @test ndual_value(clamp(xc, 0.0, 1.0)) ≈ 1.0
        @test ndual_partial(clamp(xc, 0.0, 1.0), 1) ≈ 0.0  # clamped
        @test ndual_value(clamp(xc, 0.0, 3.0)) ≈ 2.0
        @test ndual_partial(clamp(xc, 0.0, 3.0), 1) ≈ 1.0  # pass-through

        # flipsign / copysign
        xf = _d(2.0, 1.0)
        @test ndual_value(flipsign(xf, _d(-1.0, 0.0))) ≈ -2.0
        @test ndual_partial(flipsign(xf, _d(-1.0, 0.0)), 1) ≈ -1.0
        @test ndual_value(copysign(xf, _d(-1.0, 0.0))) ≈ -2.0
        @test ndual_partial(copysign(xf, _d(-1.0, 0.0)), 1) ≈ -1.0
    end

    @testset "Float32" begin
        x = _d32(2.0, 1.0)
        @test ndual_value(sin(x)) ≈ sin(2.0f0)
        @test ndual_partial(sin(x), 1) ≈ cos(2.0f0)
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
        for op in (floor, ceil, round, trunc, div, mod, rem)
            @test_throws NDualUnsupportedError op(d)
        end
        @test_throws NDualUnsupportedError floor(Int, d)
        @test_throws NDualUnsupportedError round(Int, d)
        err = try
            floor(d)
        catch e
            e
        end
        @test occursin("floor", sprint(showerror, err))
        @test occursin("NDual", sprint(showerror, err))
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
        @test ndual_value(r) ≈ 25.0
        @test ndual_partial(r, 1) ≈ 6.0   # 2 * re
        @test ndual_partial(r, 2) ≈ 8.0   # 2 * im

        # abs(z) = hypot(re, im)
        r = abs(z)
        @test ndual_value(r) ≈ 5.0
        @test ndual_partial(r, 1) ≈ a / 5.0   # re/|z|
        @test ndual_partial(r, 2) ≈ b / 5.0   # im/|z|

        # conj(z) = re - im*i — partials flip sign on imag part
        cz = conj(z)
        @test ndual_value(real(cz)) ≈ 3.0
        @test ndual_value(imag(cz)) ≈ -4.0
        @test ndual_partial(real(cz), 1) ≈ 1.0
        @test ndual_partial(imag(cz), 2) ≈ -1.0

        # z * conj(z) = abs2(z) as a real NDual
        r2 = real(z * conj(z))
        @test ndual_value(r2) ≈ 25.0

        # Helper: check value and 2-slot Jacobian against reference complex function
        function _check_cx(f, z, zv)
            r = f(z)
            rv = f(zv)
            @test ndual_value(real(r)) ≈ real(rv) rtol=1e-10
            @test ndual_value(imag(r)) ≈ imag(rv) rtol=1e-10
            ε = 1e-7
            ∂re_re = (real(f(complex(real(zv)+ε, imag(zv)))) - real(rv)) / ε
            ∂re_im = (real(f(complex(real(zv), imag(zv)+ε))) - real(rv)) / ε
            ∂im_re = (imag(f(complex(real(zv)+ε, imag(zv)))) - imag(rv)) / ε
            ∂im_im = (imag(f(complex(real(zv), imag(zv)+ε))) - imag(rv)) / ε
            @test ndual_partial(real(r), 1) ≈ ∂re_re rtol=1e-5
            @test ndual_partial(real(r), 2) ≈ ∂re_im rtol=1e-5
            @test ndual_partial(imag(r), 1) ≈ ∂im_re rtol=1e-5
            @test ndual_partial(imag(r), 2) ≈ ∂im_im rtol=1e-5
        end

        zv = complex(a, b)

        _check_cx(sin, z, zv)
        sz = sin(z)
        @test ndual_partial(real(sz), 1) ≈ cos(a)*cosh(b) rtol=1e-10
        @test ndual_partial(real(sz), 2) ≈ sin(a)*sinh(b) rtol=1e-10
        @test ndual_partial(imag(sz), 1) ≈ -sin(a)*sinh(b) rtol=1e-10
        @test ndual_partial(imag(sz), 2) ≈ cos(a)*cosh(b) rtol=1e-10

        _check_cx(cos, z, zv)
        _check_cx(exp, z, zv)
        ez = exp(z)
        @test ndual_partial(real(ez), 1) ≈ exp(a)*cos(b) rtol=1e-10
        @test ndual_partial(real(ez), 2) ≈ -exp(a)*sin(b) rtol=1e-10
        @test ndual_partial(imag(ez), 1) ≈ exp(a)*sin(b) rtol=1e-10
        @test ndual_partial(imag(ez), 2) ≈ exp(a)*cos(b) rtol=1e-10

        _check_cx(log, z, zv)
        _check_cx(sqrt, z, zv)
        _check_cx(tan, z, zv)

        # Float32 variant
        re32 = NDual{Float32,2}(3.0f0, (1.0f0, 0.0f0))
        im32 = NDual{Float32,2}(4.0f0, (0.0f0, 1.0f0))
        z32 = complex(re32, im32)
        sz32 = sin(z32)
        @test sz32 isa Complex{NDual{Float32,2}}
        @test ndual_value(real(sz32)) ≈ real(sin(complex(3.0f0, 4.0f0))) rtol=1e-5
    end

    @testset "chunk mode: N=3" begin
        x = NDual{Float64,3}(2.0, (1.0, 0.0, 0.0))
        y = NDual{Float64,3}(3.0, (0.0, 1.0, 0.0))
        c = NDual{Float64,3}(5.0, (0.0, 0.0, 1.0))

        r = c * sin(x) * exp(y)
        v = 5.0 * sin(2.0) * exp(3.0)
        @test ndual_value(r) ≈ v
        @test ndual_partial(r, 1) ≈ 5.0 * cos(2.0) * exp(3.0)
        @test ndual_partial(r, 2) ≈ 5.0 * sin(2.0) * exp(3.0)
        @test ndual_partial(r, 3) ≈ sin(2.0) * exp(3.0)
    end

    @testset "reciprocal trig" begin
        x = _d(0.8, 1.0)
        @test ndual_value(sec(x)) ≈ sec(0.8)
        @test ndual_partial(sec(x), 1) ≈ sec(0.8) * tan(0.8)
        @test ndual_value(csc(x)) ≈ csc(0.8)
        @test ndual_partial(csc(x), 1) ≈ -csc(0.8) * cot(0.8)
        @test ndual_value(cot(x)) ≈ cot(0.8)
        @test ndual_partial(cot(x), 1) ≈ -(1 + cot(0.8)^2)

        y = _d(1.5, 1.0)
        @test ndual_value(asec(y)) ≈ asec(1.5)
        @test ndual_partial(asec(y), 1) ≈ inv(abs(1.5) * sqrt(1.5^2 - 1))
        @test ndual_value(acsc(y)) ≈ acsc(1.5)
        @test ndual_partial(acsc(y), 1) ≈ -inv(abs(1.5) * sqrt(1.5^2 - 1))
        @test ndual_value(acot(x)) ≈ acot(0.8)
        @test ndual_partial(acot(x), 1) ≈ -inv(1 + 0.8^2)
    end

    @testset "reciprocal hyperbolic" begin
        x = _d(0.5, 1.0)
        @test ndual_value(sech(x)) ≈ sech(0.5)
        @test ndual_partial(sech(x), 1) ≈ -tanh(0.5) * sech(0.5)
        @test ndual_value(csch(x)) ≈ csch(0.5)
        @test ndual_partial(csch(x), 1) ≈ -coth(0.5) * csch(0.5)
        @test ndual_value(coth(x)) ≈ coth(0.5)
        @test ndual_partial(coth(x), 1) ≈ -(csch(0.5)^2)

        z = _d(0.4, 1.0)
        @test ndual_value(asech(z)) ≈ asech(0.4)
        @test ndual_partial(asech(z), 1) ≈ -inv(0.4 * sqrt(1 - 0.4^2))
        @test ndual_value(acsch(x)) ≈ acsch(0.5)
        @test ndual_partial(acsch(x), 1) ≈ -inv(abs(0.5) * sqrt(1 + 0.5^2))
        @test ndual_value(acoth(_d(2.0, 1.0))) ≈ acoth(2.0)
        @test ndual_partial(acoth(_d(2.0, 1.0)), 1) ≈ inv(1 - 2.0^2)
    end

    @testset "degree-based trig" begin
        x = _d(30.0, 1.0)
        @test ndual_value(sind(x)) ≈ sind(30.0)
        @test ndual_partial(sind(x), 1) ≈ deg2rad(cosd(30.0))
        @test ndual_value(cosd(x)) ≈ cosd(30.0)
        @test ndual_partial(cosd(x), 1) ≈ -deg2rad(sind(30.0))
        @test ndual_value(tand(x)) ≈ tand(30.0)
        @test ndual_partial(tand(x), 1) ≈ deg2rad(1 + tand(30.0)^2)

        y = _d(0.5, 1.0)
        @test ndual_value(asind(y)) ≈ asind(0.5)
        @test ndual_partial(asind(y), 1) ≈ inv(deg2rad(sqrt(1 - 0.5^2)))
        @test ndual_value(acosd(y)) ≈ acosd(0.5)
        @test ndual_partial(acosd(y), 1) ≈ -inv(deg2rad(sqrt(1 - 0.5^2)))
        @test ndual_value(atand(y)) ≈ atand(0.5)
        @test ndual_partial(atand(y), 1) ≈ inv(deg2rad(1 + 0.5^2))
    end

    @testset "angle conversions" begin
        x = _d(90.0, 1.0)
        @test ndual_value(deg2rad(x)) ≈ deg2rad(90.0)
        @test ndual_partial(deg2rad(x), 1) ≈ deg2rad(1.0)
        @test ndual_value(rad2deg(x)) ≈ rad2deg(90.0)
        @test ndual_partial(rad2deg(x), 1) ≈ rad2deg(1.0)
    end

    @testset "sinc" begin
        x = _d(0.5, 1.0)
        @test ndual_value(sinc(x)) ≈ sinc(0.5)
        @test ndual_partial(sinc(x), 1) ≈ cosc(0.5)
    end

    @testset "two-arg log and ldexp" begin
        x = _d(4.0, 1.0)
        @test ndual_value(log(2, x)) ≈ log(2, 4.0)
        @test ndual_partial(log(2, x), 1) ≈ inv(4.0 * log(2))

        y = _d(1.5, 1.0)
        @test ndual_value(ldexp(y, 3)) ≈ ldexp(1.5, 3)
        @test ndual_partial(ldexp(y, 3), 1) ≈ exp2(3)
    end

    @testset "scalar-base power" begin
        a = _d(2.0, 1.0)
        r = 3.0^a
        @test ndual_value(r) ≈ 3.0^2.0
        @test ndual_partial(r, 1) ≈ 3.0^2.0 * log(3.0)
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
end
