@testset "lapack" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:lapack))

    @testset "real/complex logdet" begin
        d = 3
        logdet_realmat(x) = logdet(reshape(x, d, d))
        x0 = vec(randn(d, d)^2)
        TestUtils.test_rule(StableRNG(1), logdet_realmat, x0; is_primitive=false)
        logdet_complexmat(x) = real(
            logdet(
                reshape(x[1:(d ^ 2)], d, d) + im * reshape(x[(d ^ 2 + 1):(2d ^ 2)], d, d)
            ),
        )
        z0 = randn(ComplexF64, d, d)
        y0 = z0'z0
        x0 = [vec(real(y0)); vec(imag(y0))]
        TestUtils.test_rule(StableRNG(1), logdet_complexmat, x0; is_primitive=false)
    end
end
