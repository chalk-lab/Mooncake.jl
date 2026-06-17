@testset "blas (ComplexF64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_ComplexF64))

    # dotc/dotu are forward-mode-only primitives whose reverse descends to the `_foreigncall_`
    # rrule (derived), so the registry exercises them only at chunk width 1. Drive the forward
    # rule directly (`mode=ForwardMode`, `is_primitive=true`) so the width-N>1 per-lane assembly
    # is checked. Branch-vs-main review #3.
    @testset "$f chunked forward" for f in (BLAS.dotc, BLAS.dotu)
        rng = sr(123)
        TestUtils.test_rule(
            rng,
            f,
            3,
            randn(rng, ComplexF64, 6),
            2,
            randn(rng, ComplexF64, 9),
            3;
            mode=ForwardMode,
        )
    end
end
