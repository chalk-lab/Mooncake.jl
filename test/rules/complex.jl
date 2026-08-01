@testset "complex" begin
    rng = sr(123)
    p = Complex{Float64}(5.0, 4.0)
    TestUtils.test_data(rng, p)
    p = Complex{Float32}(5.0, 4.0)
    TestUtils.test_data(rng, p)

    # Complex-scalar `lgetfield` (real/imag) and `_new_` (construction) rules, registered as
    # `derived_rule_test_cases(:complex)` so they get both modes and widths 1-3 from the harness.
    TestUtils.run_rule_test_cases(StableRNG, Val(:complex))
end
