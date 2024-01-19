@testset "misc" begin

    @testset "misc utility" begin
        x = randn(4, 5)
        p = Base.unsafe_convert(Ptr{Float64}, x)
        @test Taped.wrap_ptr_as_view(p, 4, 4, 5) == x
        @test Taped.wrap_ptr_as_view(p, 4, 2, 5) == x[1:2, :]
        @test Taped.wrap_ptr_as_view(p, 4, 2, 3) == x[1:2, 1:3]
    end

    @testset "lgetfield" begin
        x = (5.0, 4)
        @test lgetfield(x, Val(1)) == getfield(x, 1)
        @test lgetfield(x, Val(2)) == getfield(x, 2)

        y = (a=5.0, b=4)
        @test lgetfield(y, Val(:a)) == getfield(y, :a)
        @test lgetfield(y, Val(:b)) == getfield(y, :b)
    end
    @testset "lsetfield!" begin
        x = TestResources.MutableFoo(5.0, randn(5))
        @test Taped.lsetfield!(x, Val(:a), 4.0) == 4.0
        @test x.a == 4.0

        new_b = zeros(10)
        @test Taped.lsetfield!(x, Val(:b), new_b) === new_b
        @test x.b === new_b
    end

    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:misc))
end
