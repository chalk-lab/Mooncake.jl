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
end
