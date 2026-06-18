@testset "complex" begin
    rng = sr(123)
    p = Complex{Float64}(5.0, 4.0)
    TestUtils.test_data(rng, p)
    p = Complex{Float32}(5.0, 4.0)
    TestUtils.test_data(rng, p)

    # Exercise the Complex-scalar `lgetfield` (real/imag) and `_new_` (construction) forward rules
    # through `test_rule` (finite-difference correctness): there is no `:complex` rule registry, so
    # these were otherwise covered only via whole-program composition, not the per-rule harness.
    @testset "complex scalar getfield / new via test_rule" begin
        TestUtils.test_rule(sr(1), real, 1.0 + 2.0im; is_primitive=false, perf_flag=:none)
        TestUtils.test_rule(sr(2), imag, 1.0 + 2.0im; is_primitive=false, perf_flag=:none)
        TestUtils.test_rule(
            sr(3), z -> z.re * z.im, 1.5 - 0.5im; is_primitive=false, perf_flag=:none
        )
        TestUtils.test_rule(
            sr(4), (a, b) -> Complex(a, b), 1.0, 2.0; is_primitive=false, perf_flag=:none
        )
        TestUtils.test_rule(
            sr(5),
            (a, b) -> abs2(Complex(a, b)),
            1.0,
            2.0;
            is_primitive=false,
            perf_flag=:none,
        )
    end
end
