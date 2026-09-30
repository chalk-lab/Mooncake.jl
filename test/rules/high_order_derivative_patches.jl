struct _RefCaptureWrap{F}
    f::F
end
(w::_RefCaptureWrap)(x) = w.f(x)
Mooncake.tangent_type(::Type{<:_RefCaptureWrap}) = Mooncake.NoTangent

const _EMPTY_FDATA_EXCEPTION = ErrorException("x")
_throw_empty_fdata_exception(x) = x < 0 ? throw(_EMPTY_FDATA_EXCEPTION) : x^2

function _compute_grad(rule, f, x::Vector{Float64}, x_fdata::Vector{Float64})
    fill!(x_fdata, 0.0)
    _, pb!! = rule(zero_fcodual(f), CoDual(x, x_fdata))
    pb!!(1.0)
    return copy(x_fdata)
end

function _hessian_column(f, x::Vector{Float64}, i::Int)
    x_fdata = fdata(zero_tangent(x))
    rule = build_rrule(f, x)
    frule = build_frule(_compute_grad, rule, f, x, x_fdata)

    x_tangent = zeros(length(x))
    x_tangent[i] = 1.0
    fill!(x_fdata, 0.0)

    result = frule(
        zero_dual(_compute_grad),
        zero_dual(rule),
        zero_dual(f),
        Dual(x, x_tangent),
        Dual(x_fdata, zeros(length(x))),
    )
    return primal(result), tangent(result)
end

function _compute_hessian(f, x::Vector{Float64})
    n = length(x)
    H = zeros(n, n)
    for i in 1:n
        _, H[:, i] = _hessian_column(f, x, i)
    end
    return H
end

@testset "hessian_scalar_functions" begin
    @testset "sum" begin
        g(x) = sum(x)
        x = [2.0]
        grad, hess_col = _hessian_column(g, x, 1)
        @test grad ≈ [1.0]
        @test hess_col ≈ [0.0]
    end

    @testset "x^4.0" begin
        f(x) = x[1]^4.0
        x = [2.0]
        grad, hess_col = _hessian_column(f, x, 1)
        @test grad ≈ [32.0]
        @test hess_col ≈ [48.0]
    end

    @testset "x^4" begin
        f(x) = x[1]^4
        x = [2.0]
        grad, hess_col = _hessian_column(f, x, 1)
        @test grad ≈ [32.0]
        @test hess_col ≈ [48.0]
    end

    @testset "x^6" begin
        f(x) = x[1]^6
        x = [2.0]
        grad, hess_col = _hessian_column(f, x, 1)
        @test grad ≈ [192.0]
        @test hess_col ≈ [480.0]
    end
end

@testset "hessian_multivariate" begin
    @testset "Rosenbrock" begin
        rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
        z = [1.2, 1.2]
        H = _compute_hessian(rosen, z)
        expected_H = [1250.0 -480.0; -480.0 200.0]
        @test H ≈ expected_H rtol = 1e-10
    end

    @testset "sum of squares" begin
        f(x) = sum([x[1] * x[1], x[2] * x[2]])
        x = [2.0, 3.0]
        grad, hess_col = _hessian_column(f, x, 1)
        @test grad ≈ [4.0, 6.0] rtol = 1e-10
        @test hess_col ≈ [2.0, 0.0] rtol = 1e-10
    end

    @testset "broadcast sum of squares" begin
        # Tests broadcast operations: x .* x uses broadcasting
        f(x) = sum(x .* x)
        x = [2.0, 3.0]
        H = _compute_hessian(f, x)
        # f(x) = x₁² + x₂², so ∇f = [2x₁, 2x₂] and H = 2I
        @test H ≈ [2.0 0.0; 0.0 2.0] rtol = 1e-10
    end

    @testset "GAMS objective" begin
        function gams_objective(x)
            #! format: off
            objvar = (((((((((((((((((((((((((((x[1] * x[1] + x[10] * x[10]) * (x[1] * x[1] + x[10] * x[10]) - 4 * x[1]) + 3) + (x[2] * x[2] + x[10] * x[10]) * (x[2] * x[2] + x[10] * x[10])) - 4 * x[2]) + 3) + (x[3] * x[3] + x[10] * x[10]) * (x[3] * x[3] + x[10] * x[10])) - 4 * x[3]) + 3) + (x[4] * x[4] + x[10] * x[10]) * (x[4] * x[4] + x[10] * x[10])) - 4 * x[4]) + 3) + (x[5] * x[5] + x[10] * x[10]) * (x[5] * x[5] + x[10] * x[10])) - 4 * x[5]) + 3) + (x[6] * x[6] + x[10] * x[10]) * (x[6] * x[6] + x[10] * x[10])) - 4 * x[6]) + 3) + (x[7] * x[7] + x[10] * x[10]) * (x[7] * x[7] + x[10] * x[10])) - 4 * x[7]) + 3) + (x[8] * x[8] + x[10] * x[10]) * (x[8] * x[8] + x[10] * x[10])) - 4 * x[8]) + 3) + (x[9] * x[9] + x[10] * x[10]) * (x[9] * x[9] + x[10] * x[10])) - 4 * x[9]) + 3) - 0
            #! format: on
            return objvar
        end

        x0 = [0.0; fill(1.0, 9)]
        H = _compute_hessian(gams_objective, x0)

        H_expected = zeros(10, 10)
        H_expected[1, 1] = 4.0
        for i in 2:9
            H_expected[i, i] = 16.0
            H_expected[i, 10] = 8.0
            H_expected[10, i] = 8.0
        end
        H_expected[10, 10] = 140.0

        @test H ≈ H_expected rtol = 1e-10
    end
end

# Previous tests use build_f/rrule,
# here we use the public interface directly.
@testset "forward over reverse (public interface)" begin
    function compute_hessian(f, x::Vector{Float64}; debug_mode=false)
        config = Mooncake.Config(; debug_mode)
        function grad(y)
            rvscache = prepare_gradient_cache(f, y; config)
            value_and_gradient!!(rvscache, f, y)[2][2]
        end
        fwdcache = prepare_derivative_cache(grad, x; config)
        hvp(y) = tangent(value_and_derivative!!(fwdcache, zero_dual(grad), Dual(x, y)))
        n = length(x)
        H = zeros(n, n)
        for i in 1:n
            y = zeros(Float64, n)
            y[i] = 1
            H[:, i] = hvp(y)
        end
        return H
    end

    @testset "Rosenbrock" begin
        rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
        z = [1.2, 1.2]
        H = compute_hessian(rosen, z)
        expected_H = [1250.0 -480.0; -480.0 200.0]
        @test H ≈ expected_H rtol = 1e-10
    end

    @testset "Rosenbrock (debug_mode=true)" begin
        rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
        z = [1.2, 1.2]
        H = compute_hessian(rosen, z; debug_mode=true)
        expected_H = [1250.0 -480.0; -480.0 200.0]
        @test H ≈ expected_H rtol = 1e-10
    end
end

@testset "reverse over reverse fails" begin
    rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
    z = [1.2, 1.2]

    rvscache = prepare_gradient_cache(rosen, z)
    grad(y) = value_and_gradient!!(rvscache, rosen, y)[2][2]
    # On Julia 1.10, __call_rule's inferencebarrier makes __value_and_gradient!! opaque to
    # Mooncake's rule compiler, so build_rrule fails with MooncakeRuleCompilationError
    # before reaching the ArgumentError thrown by MistyClosure.rrule!!.
    @static if VERSION >= v"1.11-"
        @test_throws "not currently supported" prepare_gradient_cache(grad, z)
    else
        @test try
            prepare_gradient_cache(grad, z)
            false
        catch e
            e isa Mooncake.MooncakeRuleCompilationError ||
                (e isa ArgumentError && occursin("not currently supported", e.msg))
        end
    end
end

@testset "get_inner_rrule is forward-over-reverse only" begin
    for_rule = Mooncake.compile_for_rule(x -> sum(x .* x), [1.0, 2.0])
    @test_throws "forward-over-reverse only" Mooncake.rrule!!(
        zero_fcodual(Mooncake.get_inner_rrule), zero_fcodual(for_rule)
    )
end

@testset "native HVP interface (prepare_hvp_cache + value_and_hvp!!)" begin
    @testset "BLAS zero cotangents with live perturbations" begin
        fscal(a) = (BLAS.scal!(1, a, [2.0], 1)[1] - 2.0)^2
        faxpy(a) = (BLAS.axpy!(1, a, [2.0], 1, [0.0], 1)[1] - 2.0)^2
        fgemv(a) = (BLAS.gemv!('N', a, ones(1, 1), ones(1), 0.0, zeros(1))[1] - 1.0)^2
        for (f, h) in ((fscal, 8.0), (faxpy, 8.0), (fgemv, 2.0))
            cache = prepare_hvp_cache(f, 1.0)
            @test value_and_hvp!!(cache, f, 1.0, 1.0) == (0.0, 0.0, h)
        end
    end

    @testset "BLAS zero coefficients with live perturbations" begin
        fgemm(b) = only(BLAS.gemm!('N', 'N', 2.0, ones(1, 1), ones(1, 1), b, ones(1, 1)))^2
        fsymm(a) = only(BLAS.symm!('L', 'U', a, ones(1, 1), ones(1, 1), 1.0, ones(1, 1)))^2
        fhemm(a) =
            real(
                only(
                    BLAS.hemm!(
                        'L',
                        'U',
                        complex(a),
                        ones(ComplexF64, 1, 1),
                        ones(ComplexF64, 1, 1),
                        1.0 + 0im,
                        ones(ComplexF64, 1, 1),
                    ),
                ),
            )^2
        for (f, value, gradient) in
            ((fgemm, 4.0, 4.0), (fsymm, 1.0, 2.0), (fhemm, 1.0, 2.0))
            cache = prepare_hvp_cache(f, 0.0)
            @test value_and_hvp!!(cache, f, 1.0, 0.0) == (value, gradient, 2.0)
        end
    end

    @testset "TwicePrecision cotangent accumulation (#1328)" begin
        f(x) = abs2(typeof(x)(TwicePrecision(x)))
        for x in (0.5f0, 0.5)
            cache = prepare_hvp_cache(f, x)
            for v in (one(x), -one(x))
                @test value_and_hvp!!(cache, f, v, x) == (abs2(x), 2x, 2v)
            end
        end
    end

    @testset "captured constant with empty structural fdata" begin
        # The closure-captured variant follows a separate failing path tracked in #1286.
        f = _throw_empty_fdata_exception
        @test value_and_hvp!!(prepare_hvp_cache(f, 1.0), f, 1.0, 1.0) == (1.0, 2.0, 2.0)
    end

    @testset "gradient correctness for x^4" begin
        f(x) = x[1]^4.0
        x = [2.0]
        cache = prepare_hvp_cache(f, x)
        f_val, grad, _ = value_and_hvp!!(cache, f, [1.0], x)
        @test f_val ≈ 16.0
        @test grad ≈ [32.0]
    end

    @testset "HVP correctness for x^4" begin
        f(x) = x[1]^4.0
        x = [2.0]
        _, _, hvp = value_and_hvp!!(prepare_hvp_cache(f, x), f, [1.0], x)
        @test hvp ≈ [48.0]
    end

    # Regression test for #1246.
    @testset "triangular solve" begin
        L = LowerTriangular([2.0 0.0; 1.0 3.0])
        f(x) = sum(abs2, L \ x)
        x = [1.0, 2.0]

        value, grad, H = value_gradient_and_hessian!!(prepare_hessian_cache(f, x), f, x)

        @test value ≈ 1 / 2
        @test grad ≈ [1 / 3, 1 / 3]
        @test H ≈ [5 / 9 -1 / 9; -1 / 9 2 / 9]
    end

    @testset "cache reuse across multiple HVP calls" begin
        # The `DerivedFoRRule`'s cached `Dual` is reused across calls without copying.
        f(x) = sum(x .* x)  # H = 2I
        x = [1.0, 2.0, 3.0]
        cache = prepare_hvp_cache(f, x)
        n = length(x)
        for i in 1:n
            v = zeros(n)
            v[i] = 1.0
            _, _, hvp = value_and_hvp!!(cache, f, v, x)
            expected = 2.0 .* v
            @test hvp ≈ expected rtol = 1e-10
        end
    end

    @testset "multi-argument HVP" begin
        # f(x, y) = sum(x .* x) + sum(y .* y): H = 2I (block-diagonal, decoupled)
        f(x, y) = sum(x .* x) + sum(y .* y)
        x = [1.0, 2.0]
        y = [3.0]
        cache = prepare_hvp_cache(f, x, y)
        _, (grad_x, grad_y), (hvp_x, hvp_y) = value_and_hvp!!(
            cache, f, ([1.0, 0.0], [0.0]), x, y
        )
        @test grad_x ≈ [2.0, 4.0] rtol = 1e-10
        @test grad_y ≈ [6.0] rtol = 1e-10
        @test hvp_x ≈ [2.0, 0.0] rtol = 1e-10
        @test hvp_y ≈ [0.0] rtol = 1e-10
    end

    @testset "primitive f (DerivedFoRRule{Nothing} path)" begin
        # Primitive `f` ⇒ `compile_for_rule` returns `DerivedFoRRule{Nothing}`, so `grad_f`
        # routes through `value_and_gradient!!`, not an inner derived rrule.
        @testset "single argument" begin
            f = sum  # linear ⇒ zero Hessian
            x = [1.0, 2.0, 3.0]
            fval, grad, hvp = value_and_hvp!!(
                prepare_hvp_cache(f, x), f, [1.0, 0.0, 0.0], x
            )
            @test fval ≈ 6.0
            @test grad ≈ [1.0, 1.0, 1.0]
            @test hvp ≈ [0.0, 0.0, 0.0]
        end
        @testset "multiple arguments" begin
            f = hypot  # r = √(a²+b²); H = [b² -ab; -ab a²]/r³
            a, b = 3.0, 4.0
            fval, grads, hvps = value_and_hvp!!(
                prepare_hvp_cache(f, a, b), f, (1.0, 0.0), a, b
            )
            @test fval ≈ 5.0
            @test grads[1] ≈ 0.6 rtol = 1e-10
            @test grads[2] ≈ 0.8 rtol = 1e-10
            @test hvps[1] ≈ 0.128 rtol = 1e-10
            @test hvps[2] ≈ -0.096 rtol = 1e-10
        end
    end

    @testset "Ref-capture under NoTangent wrapper (issue #1193)" begin
        f = _RefCaptureWrap(
            let r = Ref(3.0);
                x -> r[] * sum(abs2, x);
            end,
        )
        x = [1.0, 2.0]
        _, _, hvp = value_and_hvp!!(prepare_hvp_cache(f, x), f, [1.0, 0.0], x)
        @test hvp ≈ [6.0, 0.0]
        _, _, H = value_gradient_and_hessian!!(prepare_hessian_cache(f, x), f, x)
        @test H ≈ [6.0 0.0; 0.0 6.0]
    end

    @test Mooncake.tangent_type(typeof(get_interpreter(ForwardMode))) == Mooncake.NoTangent
end

@testset "BLAS coefficient HVPs" begin
    # First-order registry checks cannot detect lost perturbations in a pullback.
    for P in (Float64, ComplexF64),
        op in (
            BLAS.gemm!,
            BLAS.gemv!,
            BLAS.symm!,
            BLAS.symv!,
            BLAS.syrk!,
            BLAS.trmm!,
            BLAS.trsm!,
            (P <: Complex ? (BLAS.hemm!, BLAS.hemv!, BLAS.herk!) : ())...,
        )

        p = one(P)
        f = if op in (BLAS.gemm!, BLAS.symm!, BLAS.hemm!)
            flags = op === BLAS.gemm! ? ('N', 'N') : ('L', 'U')
            x -> sum(
                abs2,
                op(
                    flags...,
                    oftype(p, x[1]),
                    fill(oftype(p, x[2]), 1, 1),
                    fill(p, 1, 1),
                    oftype(p, x[3]),
                    fill(p, 1, 1),
                ),
            )
        elseif op in (BLAS.gemv!, BLAS.symv!, BLAS.hemv!)
            flag = op === BLAS.gemv! ? 'N' : 'U'
            x -> sum(
                abs2,
                op(
                    flag,
                    oftype(p, x[1]),
                    fill(oftype(p, x[2]), 1, 1),
                    fill(p, 1),
                    oftype(p, x[3]),
                    fill(p, 1),
                ),
            )
        elseif op in (BLAS.syrk!, BLAS.herk!)
            q = op === BLAS.herk! ? real(p) : p
            x -> sum(
                abs2,
                op(
                    'U',
                    'N',
                    oftype(q, x[1]),
                    fill(oftype(p, x[2]), 1, 1),
                    oftype(q, x[3]),
                    fill(p, 1, 1),
                ),
            )
        else
            x -> sum(
                abs2,
                op(
                    'L',
                    'U',
                    'N',
                    'N',
                    oftype(p, x[1]),
                    fill(oftype(p, x[2]), 1, 1),
                    fill(p, 1, 1),
                ),
            )
        end
        @testset "$P $op $a $b" for a in (0.0, 1.0, 2.0), b in (0.0, 1.0)
            x, v = [a, 3.0, b], ones(3)
            cache = Mooncake.prepare_gradient_cache(f, x)
            grad(z) = copy(Mooncake.value_and_gradient!!(cache, f, z)[2][2])
            fd = (grad(x + 1e-5v) - grad(x - 1e-5v)) / 2e-5
            h = Mooncake.value_and_hvp!!(Mooncake.prepare_hvp_cache(f, x), f, v, x)[3]
            @test h ≈ fd rtol=1e-7 atol=1e-7
        end
    end
end

@testset "zero alpha extreme HVP" for (A, B, seed, da) in (
    (1e-200, 1e200, 1e200, 1e-300), (1e-100, 1e-200, 1e-200, 1e300)
)
    f(x) =
        seed *
        only(BLAS.gemm!('N', 'N', x[1], fill(x[2], 1, 1), fill(B, 1, 1), 0.0, zeros(1, 1)))
    x = [0.0, A]
    h = value_and_hvp!!(prepare_hvp_cache(f, x), f, [da, 0.0], x)[3]
    @test isequal(h[2], only(da * fill(seed, 1, 1) * fill(B, 1, 1)))
end

@testset "zero alpha coefficient direction HVP" for (big, small) in
                                                    ((1e200, 1e-200), (1e-200, 1e200))
    f(x) =
        small * only(
            BLAS.gemm!('N', 'N', x[1], fill(big, 1, 1), fill(x[2], 1, 1), 0.0, zeros(1, 1))
        )
    x = [0.0, small]
    h = value_and_hvp!!(prepare_hvp_cache(f, x), f, [big, 0.0], x)[3]
    @test isequal(h, [0.0, big])
end

@testset "nonzero alpha extreme HVP" for unit in (1.0, 1.0 + 0.0im),
    (a, seed) in ((1e200, 1e-200), (1e-200, 1e200))

    function f(x)
        A = fill(oftype(unit, x[1]), 1, 1)
        B = fill(oftype(unit, x[2]), 1, 1)
        C = fill(zero(unit), 1, 1)
        return seed * real(only(BLAS.gemm!('N', 'N', oftype(unit, a), A, B, zero(unit), C)))
    end
    x = [1e-200, 1e-200]
    value, _, h = value_and_hvp!!(prepare_hvp_cache(f, x), f, [a, 0.0], x)
    @test isfinite(value)
    @test h ≈ [0.0, a]
end

@testset "mixed extreme HVP directions" begin
    f(x) =
        x[4] * only(
            BLAS.gemm!('N', 'N', x[1], fill(x[2], 1, 1), fill(x[3], 1, 1), 0.0, zeros(1, 1))
        )
    x = [2.0, 1.0, 1.0, 1.0]
    v = [1e308, 0.0, -5e307, 5e307]
    h = value_and_hvp!!(prepare_hvp_cache(f, x), f, v, x)[3]
    @test isequal(h, [0.0, 1e308, Inf, 0.0])
end
