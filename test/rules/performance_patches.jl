@testset "performance_patches" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:performance_patches))
    P = Float16
    A = P[0.2 0.3; 0.4 0.5]
    B = P[0.3 0.4; 0.5 0.6]
    cases = [
        (zeros(P, 2, 2), A, A, true, false),
        (ones(P, 2, 2), A, B, false, true),
        (zeros(P, 2, 2), transpose(A), adjoint(B), true, false),
        (ones(P, 2, 2), A, B, P(0.5), zero(P)),
        (ones(P, 2, 2), A, B, true, P(0.25)),
        (ones(P, 2, 2), A, B, P(0.5), true),
    ]
    for (wrap_a, wrap_b) in
        ((identity, transpose), (transpose, adjoint), (adjoint, identity))
        X = wrap_a(view(reshape(copy(A), 2, 2, 1),:,:,1))
        Y = wrap_b(view(reshape(copy(B), 2, 2, 1),:,:,1))
        C = view(fill(P(0.1), 2, 2, 1),:,:,1)
        push!(cases, (C, X, Y, P(0.5), P(0.25)))
    end
    @testset "Float16 mul! $(map(typeof, xs))" for xs in cases
        TestUtils.test_rule(
            StableRNG(123),
            mul!,
            xs...;
            is_primitive=true,
            atol=5e-2,
            rtol=5e-2,
        )
    end

    @testset "disjoint batch slices: $wrap, $side" for wrap in
                                                       (identity, transpose, adjoint),
        side in (:A, :B)

        X = fill(P(0.25), 2, 2, 2)
        B = fill(P(0.5), 2, 2)
        function f(X, B)
            C, V = view(X,:,:,1), wrap(view(X,:,:,2))
            return side === :A ? mul!(C, V, B, true, false) : mul!(C, B, V, true, false)
        end
        TestUtils.test_rule(
            StableRNG(123), f, X, B; is_primitive=false, atol=5e-2, rtol=5e-2
        )
    end

    @testset "aliased mul! destination" begin
        A = ones(P, 4, 4)
        B = copy(A)
        V = view(reshape(A, 4, 4, 1),:,:,1)
        for (C, X, Y) in ((A, A, B), (B, A, B), (V, transpose(V), B), (V, B, adjoint(V)))
            TestUtils.test_rule_throws(
                StableRNG(123),
                mul!,
                C,
                X,
                Y,
                true,
                false;
                err=ArgumentError,
                msg="output matrix must not be aliased with input matrix",
            )
        end
    end

end
