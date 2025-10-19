@testset "test_utils" begin
    @testset "has_equal_data" begin
        @test !has_equal_data(5.0, 4.0)
        @test has_equal_data(5.0, 5.0)
        @test has_equal_data(Float64(NaN), Float64(NaN))
        @test !has_equal_data(5.0, NaN)
        @test has_equal_data(Float64, Float64)
        @test !has_equal_data(Float64, Float32)
        @test has_equal_data(Float64.name, Float64.name)
        @test !has_equal_data(Float64.name, Float32.name)
        @test !has_equal_data(ones(1), ones(2))
        @test !has_equal_data(randn(5), randn(5))
        @test has_equal_data(ones(5), ones(5))
        @test has_equal_data(Base, Base)
        @test !has_equal_data(Base, Core)
        @test has_equal_data(GlobalRef(Base, :sin), GlobalRef(Base, :sin))
        @test !has_equal_data(GlobalRef(Base, :sin), GlobalRef(Base, :cos))
        @test !has_equal_data(GlobalRef(Base, :sin), GlobalRef(Core, :sin))
        @test has_equal_data(Complex(5.0, 4.0), Complex(5.0, 4.0))[]
        @test !has_equal_data(Complex(5.0, 4.0), Complex(5.0, 5.0))
        @test !has_equal_data(Diagonal(randn(5)), Diagonal(randn(5)))
        @test has_equal_data(Diagonal(ones(5)), Diagonal(ones(5)))
        @test has_equal_data("hello", "hello")
        @test !has_equal_data("hello", "goodbye")
        @test has_equal_data(
            TypeUnstableMutableStruct(4.0, 5), TypeUnstableMutableStruct(4.0, 5)
        )
        @test !has_equal_data(
            TypeUnstableMutableStruct(4.0, 5), TypeUnstableMutableStruct(4.0, 6)
        )
        @test has_equal_data(TypeUnstableStruct(4.0, 5), TypeUnstableStruct(4.0, 5))
        @test !has_equal_data(TypeUnstableStruct(0.0), TypeUnstableStruct(4.0))
        @test has_equal_data(
            make_circular_reference_struct(), make_circular_reference_struct()
        )
        @test has_equal_data(
            make_indirect_circular_reference_struct(),
            make_indirect_circular_reference_struct(),
        )
        @test has_equal_data(
            make_circular_reference_array(), make_circular_reference_array()
        )
        @test has_equal_data(
            make_indirect_circular_reference_array(),
            make_indirect_circular_reference_array(),
        )
        @test !has_equal_data(
            (s=make_circular_reference_struct(); s.a=1.0; s),
            (t=make_circular_reference_struct(); t.a=2.0; t),
        )
        @test !has_equal_data(
            (a=make_indirect_circular_reference_array(); a[1][1]=1.0; a),
            (b=make_indirect_circular_reference_array(); b[1][1]=2.0; b),
        )
        @test !has_equal_data(
            (s=make_indirect_circular_reference_struct(); s.b.a=1.0; s),
            (t=make_indirect_circular_reference_struct(); t.b.a=2.0; t),
        )
        @test !has_equal_data(
            (a=make_indirect_circular_reference_array(); a[1][1]=1.0; a),
            (b=make_indirect_circular_reference_array(); b[1][1]=2.0; b),
        )
    end
    @testset "populate_address_map" begin
        @testset "primitive types" begin
            @test isempty(populate_address_map(5.0, 5.0))
            @test isempty(populate_address_map(5, NoTangent()))
        end
        @testset "ComplexF64" begin
            x = ComplexF64(5.0, 4.0)
            dx = zero_tangent(x)
            @test isempty(populate_address_map(x, dx))
        end
        @testset "Array" begin
            m = populate_address_map(randn(2), randn(2))
            @test length(m) == 1

            p = [randn(2), randn(1)]
            t = zero_tangent(p)
            @test length(populate_address_map(p, t)) == 3

            p2 = [p, p]
            t2 = [t, t]
            @test length(populate_address_map(p2, t2)) == 4
            @test_throws AssertionError populate_address_map(
                p2, [zero_tangent(p), zero_tangent(p)]
            )
        end
        @testset "immutable type" begin
            p = TestResources.StructFoo(5.0, randn(2))
            t = zero_tangent(p)
            m = populate_address_map(p, t)
            @test length(m) == 1
        end
        @testset "mutable type" begin
            p = TestResources.MutableFoo(5.0, randn(2))
            t = zero_tangent(p)
            @test length(populate_address_map(p, t)) == 2
        end
        @testset "Tuple" begin
            p = (5.0, randn(3))
            t = zero_tangent(p)
            @test length(populate_address_map(p, t)) == 1

            p2 = (p, p)
            @test_throws AssertionError populate_address_map(
                p2, (zero_tangent(p), zero_tangent(p))
            )

            p = TestResources.MutableFoo(5.0, randn(2))
            p2 = (p, p)
            @test_throws AssertionError populate_address_map(
                p2, (zero_tangent(p), zero_tangent(p))
            )
        end
        @testset "views" begin
            p = view(randn(5, 4), 1:2, 1:3)
            t = zero_tangent(p)
            @test length(populate_address_map(p, zero_tangent(p))) == 1
        end
    end
    @testset "address_maps_are_consistent" begin
        f = TestResources.my_setfield!
        x = (TestResources.MutableFoo(5.0, randn(5)), :b, randn(5))
        x̄ = map(zero_tangent, x)
        input_addr_map = populate_address_map(x, x̄)
        f_f̄ = CoDual(f, zero_tangent(f))
        x_x̄ = map(CoDual, x, x̄)
        y_ȳ, _ = Mooncake.rrule!!(f_f̄, x_x̄...)
        z = (x..., primal(y_ȳ))
        z̄ = (x̄..., tangent(y_ȳ))
        @test_throws AssertionError populate_address_map(z, z̄)
    end
end
