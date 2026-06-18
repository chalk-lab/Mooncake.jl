struct _RefCaptureWrap{F}
    f::F
end
(w::_RefCaptureWrap)(x) = w.f(x)
Mooncake.tangent_type(::Type{<:_RefCaptureWrap}) = Mooncake.NoTangent
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
        Mooncake.lift(x, x_tangent),
        Mooncake.lift(x_fdata, zeros(length(x))),
    )
    return primal(result), last(Mooncake.unlift(result))
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
        hvp(y) = last(value_and_derivative!!(fwdcache, (grad, NoTangent()), (x, y)))
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

    @testset "cache reuse across multiple HVP calls" begin
        # The `DerivedFoRRule`'s cached `Dual` is reused across calls without copying;
        # verify the cache produces consistent results when called repeatedly with
        # different directions.
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
        # When `f` is itself a reverse-mode primitive, `compile_for_rule` returns
        # `DerivedFoRRule{Nothing}` and `grad_f` routes through `value_and_gradient!!`
        # rather than an inner derived rrule.
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

    # Regression for https://github.com/chalk-lab/Mooncake.jl/issues/1193.
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
end

# The `jl_genericmemory_owner` forward rule (a forward-over-reverse `dataids`-inlining workaround)
# must return the canonical forward V for its `Memory` result — `NDualArray`, not a bare `NoDual`
# (which would yield `NoTangent()` on a lane read and diverge from the reverse oracle). Review #7.
@static if VERSION >= v"1.11-"
    @testset "jl_genericmemory_owner frule canonical V (#7)" begin
        m = Memory{Float64}(undef, 3) .= [1.0, 2.0, 3.0]
        zl(x) = Mooncake.zero_lifted(Val(1), x)
        l = Mooncake.frule!!(
            zl(Mooncake._foreigncall_),
            zl(Val(:jl_genericmemory_owner)),
            zl(Val(Any)),
            zl((Val(Any),)),
            zl(Val(0)),
            zl(Val(:ccall)),
            zl(m),
        )
        @test Mooncake.verify_lifted_type(l)
        @test Mooncake.tangent(l, 1) isa AbstractArray{Float64}
    end
end
