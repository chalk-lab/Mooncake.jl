using Random: Xoshiro

@testset "primitive_wrappers" begin
    rng = Xoshiro(123)

    @testset "hand-written rule cases" begin
        test_cases, _ = Mooncake.hand_written_rule_test_cases(
            Xoshiro, Val(:primitive_wrappers)
        )
        @testset "$(case.name)" for case in test_cases
            if case.kind === :dual
                Mooncake.TestUtils.test_dual(
                    rng, case.primal, case.dual_type; width=case.width, perf=false
                )
            else
                Mooncake.TestUtils.test_rule(
                    rng,
                    case.f,
                    case.args...;
                    is_primitive=case.is_primitive,
                    perf_flag=:none,
                    chunk_sizes=case.chunk_sizes,
                    output_tangent=case.output_tangent,
                )
            end
        end
    end

    @testset "derived rule cases" begin
        test_cases, _ = Mooncake.derived_rule_test_cases(Xoshiro, Val(:primitive_wrappers))
        @testset "$(case.name)" for case in test_cases
            Mooncake.TestUtils.test_rule(
                rng,
                case.f,
                case.args...;
                is_primitive=false,
                perf_flag=:none,
                chunk_sizes=case.chunk_sizes,
                output_tangent=case.output_tangent,
            )
        end
    end
end
