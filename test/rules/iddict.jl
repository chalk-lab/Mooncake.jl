@testset "iddict" begin
    @testset "IdDict tangent functionality" begin
        p = IdDict(true => 5.0, false => 4.0)
        T = IdDict{Bool,Float64}
        TestUtils.test_tangent(sr(123456), p, T; interface_only=false, perf=false)
        TestUtils.test_tangent_splitting(sr(123456), p)
        # Abstract V must survive lane traversal. The tangent registry's allocation
        # checks exclude abstract-valued IdDicts, so exercise test_lifted directly.
        TestUtils.test_lifted(sr(123456), IdDict{Symbol,Any}(:a => 5.0))
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:iddict))

    @testset "forward lift preserves IdDict value aliasing" begin
        # Two keys sharing an array must share its forward and reverse tangent storage.
        arr = [1.0, 2.0]
        d = IdDict{Symbol,Any}(:x => arr, :y => arr)
        t = Mooncake.zero_tangent(d)
        @test t[:x] === t[:y]                       # reverse oracle shares
        v = Mooncake.tangent(Mooncake.lift(d, t))
        @test v[:x] === v[:y]                       # forward V shares too
    end
end
