@testset "avoiding_non_differentiable_code" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:avoiding_non_differentiable_code))

    # Reading a ScopedValue differentiates outside and inside `with`, where the read
    # looks the value up in the self-referential scope storage.
    @static if VERSION ≥ v"1.11-"
        @testset "ScopedValue read, $mode" for mode in (ForwardMode, ReverseMode)
            scale = Base.ScopedValues.ScopedValue(2)
            f = x -> sum(x) * scale[]
            test_rule(sr(123), f, [1.0, 2.0]; is_primitive=false, mode)
            Base.ScopedValues.with(scale => 3) do
                test_rule(sr(123), f, [1.0, 2.0]; is_primitive=false, mode)
            end
        end
    end
end
