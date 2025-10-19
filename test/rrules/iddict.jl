@testset "iddict" begin
    @testset "IdDict tangent functionality" begin
        p = IdDict(true => 5.0, false => 4.0)
        T = IdDict{Bool,Float64}
        TestUtils.test_tangent(sr(123456), p, T; interface_only=false, perf=false)
        TestUtils.test_tangent_splitting(sr(123456), p)
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:iddict))
end
