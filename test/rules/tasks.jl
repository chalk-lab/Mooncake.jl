@testset "tasks" begin
    @testset "Task tangent functionality" begin
        p = Task(() -> nothing)
        T = Mooncake.TaskTangent
        TestUtils.test_tangent(sr(123456), p, T; interface_only=false, perf=false)
        TestUtils.test_rule_and_type_interactions(sr(123456), p)
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:tasks))
end
