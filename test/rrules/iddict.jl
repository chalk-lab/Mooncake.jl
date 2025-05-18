@testset "iddict" begin
    @testset "IdDict tangent functionality" begin
        p = IdDict(true => 5.0, false => 4.0)
        x = IdDict(true => 1.0, false => 1.0)
        y = IdDict(true => 2.0, false => 1.0)
        z = IdDict(true => 3.0, false => 2.0)
        TestUtils.test_tangent(sr(123456), p, x, y, z; interface_only=false, perf=false)
        TestUtils.test_fwds_rvs_data(sr(123456), p)
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:iddict))
end
