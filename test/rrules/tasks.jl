@testset "tasks" begin
    @testset "Task tangent functionality" begin
        p = Task(() -> nothing)
        T = Mooncake.TaskTangent
        TestUtils.test_tangent(sr(123456), p, T; interface_only=false, perf=false)
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:tasks))
end
