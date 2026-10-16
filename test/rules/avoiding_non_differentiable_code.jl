@testset "avoiding_non_differentiable_code" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:avoiding_non_differentiable_code))
    @testset "descriptor shift allocations and inference" begin
        for T in (UInt8, Cvoid), width in (0, 1, 8), op in (+, -), n in (0, 2, 1024, -1024)
            if width == 0
                z = zero_fcodual
                rule = Mooncake.rrule!!
            else
                z = x -> Mooncake.zero_lifted(Val(width), x)
                rule = Mooncake.frule!!
            end
            args = (z(op), z(Ptr{T}(4096)), z(n))
            @inferred rule(args...)
            @test TestUtils.count_allocs(rule, args...) == 0
        end
    end
end
