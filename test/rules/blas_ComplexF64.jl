@testset "blas (ComplexF64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_ComplexF64))

    # Regression: the forward dotc/dotu `@is_primitive` must bind the two array arguments to
    # independent type vars, so a differently-typed pair (dense `Vector` + a strided `SubArray`) is
    # still a primitive. A single shared type var left such pairs non-primitive, dropping them onto
    # the derived forward path that cannot land complex per-lane partials.
    @testset "dotc/dotu forward on differently-typed array args ($f)" for f in (
        BLAS.dotc, BLAS.dotu
    )
        W = Base.get_world_counter()
        sig = Tuple{
            typeof(f),
            Int,
            Vector{ComplexF64},
            Int,
            SubArray{ComplexF64,1,Vector{ComplexF64},Tuple{StepRange{Int,Int}},true},
            Int,
        }
        @test Mooncake.is_primitive(DefaultCtx, Mooncake.ForwardMode, sig, W)
        g(a, b) = f(4, a, 1, view(b, 1:2:8), 1)
        x = randn(StableRNG(1), ComplexF64, 4)
        ybuf = randn(StableRNG(2), ComplexF64, 8)
        TestUtils.test_rule(
            StableRNG(3), g, x, ybuf; is_primitive=false, mode=Mooncake.ForwardMode
        )
    end
end
