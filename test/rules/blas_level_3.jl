# The level-3 hand-written test cases are split across four sets (3a–3d) to reduce
# peak memory: each test case tuple holds live references to its primal arrays, so all
# arrays for a set stay allocated until the @testset loop finishes. Running sequentially
# lets GC reclaim one set before the next is constructed (~8–10 MB peak per set vs
# ~33 MB if run as one). See the comment in src/rules/blas.jl for the per-set breakdown.
#
# gemm! mat×mat is the dominant cost and is split by tB value across 3a/3b/3c.
# The derived tests (aliased gemm!) are kept under the original Val(:blas_level_3) and
# called directly to avoid needing a dummy hand_written companion.
@testset "blas (level 3a)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3a))
end
@testset "blas (level 3b)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3b))
end
@testset "blas (level 3c)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3c))
end
@testset "blas (level 3d)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3d))
end
@testset "blas (level 3, derived)" begin
    TestUtils.run_derived_rule_test_cases(StableRNG, Val(:blas_level_3), ForwardMode)
    TestUtils.run_derived_rule_test_cases(StableRNG, Val(:blas_level_3), ReverseMode)
end
