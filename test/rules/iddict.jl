@testset "iddict" begin
    @testset "IdDict tangent functionality" begin
        p = IdDict(true => 5.0, false => 4.0)
        T = IdDict{Bool,Float64}
        TestUtils.test_tangent(sr(123456), p, T; interface_only=false, perf=false)
        TestUtils.test_tangent_splitting(sr(123456), p)
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:iddict))

    @testset "forward lift preserves IdDict value aliasing" begin
        # Two keys map to one array: the forward V must share a single tangent buffer (matching the
        # reverse-mode aliasing invariant), not build a fresh one per value. The width-1 `lift`
        # boundary previously took no aliasing cache, so the shared array was lifted twice.
        arr = [1.0, 2.0]
        d = IdDict{Symbol,Any}(:x => arr, :y => arr)
        t = Mooncake.zero_tangent(d)
        @test t[:x] === t[:y]                       # reverse oracle shares
        v = Mooncake.tangent(Mooncake.lift(d, t))
        @test v[:x] === v[:y]                       # forward V shares too
    end
end
