using Mooncake.TestUtils: count_allocs, has_equal_data
using Mooncake:
    prepare_gradient_cache,
    prepare_hessian_cache,
    prepare_pullback_cache,
    value_and_gradient!!,
    value_and_hessian!!,
    value_and_pullback!!

struct SimplePair
    x1::Float64
    x2::Float64
end

@testset "interface" begin
    @testset "$(typeof((f, x...)))" for (ȳ, f, x...) in Any[
        (1.0, (x, y) -> x * y + sin(x) * cos(y), 5.0, 4.0),
        ([1.0, 1.0], x -> [sin(x), sin(2x)], 3.0),
        (1.0, x -> sum(5x), [5.0, 2.0]),
    ]
        @testset "debug_mode=$debug_mode" for debug_mode in Bool[false, true]
            rule = build_rrule(f, x...; debug_mode)
            v, (df, dx...) = value_and_pullback!!(rule, ȳ, f, x...)
            @test v ≈ f(x...)
            @test df isa tangent_type(typeof(f))
            for (_dx, _x) in zip(dx, x)
                @test _dx isa tangent_type(typeof(_x))
            end
        end
    end
    @testset "sensible error when CoDuals are passed to `value_and_pullback!!" begin
        foo(x) = sin(cos(x))
        rule = build_rrule(foo, 5.0)
        @test_throws ArgumentError value_and_pullback!!(rule, 1.0, foo, CoDual(5.0, 0.0))
    end
    @testset "value_and_gradient!!" begin
        @testset "($(typeof(fargs))" for fargs in Any[
            (sin, randn(Float64)),
            (sin, randn(Float32)),
            (x -> sin(cos(x)), randn(Float64)),
            (x -> sin(cos(x)), randn(Float32)),
            ((x, y) -> x + sin(y), randn(Float64), randn(Float64)),
            ((x, y) -> x + sin(y), randn(Float32), randn(Float32)),
            ((x...) -> x[1] + x[2], randn(Float64), randn(Float64)),
            (sum, randn(10)),
            (x -> (x .*= 2; sum(x)), randn(10)),
            # Regression test for https://github.com/chalk-lab/Mooncake.jl/issues/1020:
            # passing a function-valued arg previously caused perf regressions due to
            # missing specialisation; @inline on the interface functions fixes this.
            ((xs, f) -> f(xs), randn(10), sum),
        ]
            kwargs = (debug_mode=false, silence_debug_messages=true)
            rule = build_rrule(fargs...; kwargs...)
            v, dfargs = value_and_gradient!!(rule, deepcopy(fargs)...)
            f, args... = deepcopy(fargs)
            @test v == f(args...)
            for (arg, darg) in zip(fargs, dfargs)
                @test tangent_type(typeof(arg)) == typeof(darg)
            end

            # Create cache and verify that mutation is undone.
            original_fargs = deepcopy(fargs)
            cache = Mooncake.prepare_gradient_cache(
                fargs...; config=Mooncake.Config(; kwargs...)
            )
            @test fargs == original_fargs

            _v, _dfargs = value_and_gradient!!(cache, fargs...)
            @test _v == v
            for (arg, darg) in zip(fargs, _dfargs)
                @test tangent_type(typeof(arg)) == typeof(darg)
            end
            alloc_count = count_allocs(value_and_gradient!!, cache, fargs...)
            if alloc_count > 0
                @test_broken alloc_count == 0
            else
                @test alloc_count == 0
            end
        end

        rule = build_rrule(identity, (5.0, 4.0))
        @test_throws(
            Mooncake.ValueAndGradientReturnTypeError,
            value_and_gradient!!(rule, identity, (5.0, 4.0)),
        )
        @test_throws(
            Mooncake.ValueAndGradientReturnTypeError,
            Mooncake.prepare_gradient_cache(identity, (5.0, 4.0)),
        )

        @testset "friendly tangents" begin
            f = (x::SimplePair) -> x.x1^2 + sin(x.x2)
            x = SimplePair(1.0, 2.0)

            cache = Mooncake.prepare_gradient_cache(f, x)
            v, dx = Mooncake.value_and_gradient!!(cache, f, x)
            @test dx[2] isa Mooncake.Tangent{@NamedTuple{x1::Float64,x2::Float64}}
            @test dx[2].fields == (; x1=2 * x.x1, x2=cos(x.x2))

            cache = Mooncake.prepare_gradient_cache(
                f, x; config=Mooncake.Config(; friendly_tangents=true)
            )
            v, dx = Mooncake.value_and_gradient!!(cache, f, x)
            @test dx[2] isa SimplePair
            @test dx[2] == SimplePair(2 * x.x1, cos(x.x2))

            rule = build_rrule(f, x)

            v, dx = Mooncake.value_and_gradient!!(rule, f, x)
            @test dx[2] isa Mooncake.Tangent{@NamedTuple{x1::Float64,x2::Float64}}
            @test dx[2].fields == (; x1=2 * x.x1, x2=cos(x.x2))

            v, dx = Mooncake.value_and_gradient!!(rule, f, x; friendly_tangents=true)
            @test dx[2] isa SimplePair
            @test dx[2] == SimplePair(2 * x.x1, cos(x.x2))
        end
    end
    @testset "value_and_pullback!!" begin
        @testset "($(typeof(fargs))" for (ȳ, fargs...) in Any[
            (randn(10), identity, randn(10)),
            (randn(10), x -> (x .*= 2; x), randn(10)),
            (randn(), sin, randn(Float64)),
            (randn(), sum, randn(Float64)),
        ]
            kwargs = (debug_mode=false, silence_debug_messages=true)
            rule = build_rrule(fargs...; kwargs...)
            f, args... = fargs
            v, dfargs = value_and_pullback!!(rule, ȳ, deepcopy(fargs)...)
            @test v == f(deepcopy(args)...)
            for (arg, darg) in zip(fargs, dfargs)
                @test tangent_type(typeof(arg)) == typeof(darg)
            end

            # Create cache and verify fargs is unchanged afterwards.
            original_args = deepcopy(fargs)
            cache = Mooncake.prepare_pullback_cache(
                fargs...; config=Mooncake.Config(; kwargs...)
            )
            @test original_args == fargs

            _v, _dfargs = value_and_pullback!!(cache, ȳ, fargs...)
            @test _v == v
            for (arg, darg) in zip(fargs, _dfargs)
                @test tangent_type(typeof(arg)) == typeof(darg)
            end
            alloc_count = count_allocs(value_and_pullback!!, cache, ȳ, fargs...)
            if alloc_count > 0
                @test_broken alloc_count == 0
            else
                @test alloc_count == 0
            end
        end

        @testset "friendly tangents" begin
            testf(x::SimplePair) = SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2)
            x = SimplePair(1.0, 2.0)
            x̄ = SimplePair(0.5, 0.3)
            x̄_unfriendly = Mooncake.Tangent((; x1=0.5, x2=0.3))

            cache = Mooncake.prepare_pullback_cache(testf, x)
            v, pb = Mooncake.value_and_pullback!!(cache, x̄_unfriendly, testf, x)
            @test has_equal_data(v, SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2))
            @test has_equal_data(
                pb[2],
                Mooncake.Tangent((;
                    x1=2x.x1 * x̄.x1 + x.x2 * x̄.x2, x2=cos(x.x2) * x̄.x1 + x.x1 * x̄.x2
                )),
            )

            cache = Mooncake.prepare_pullback_cache(
                testf, x; config=Mooncake.Config(; friendly_tangents=true)
            )
            v, pb = Mooncake.value_and_pullback!!(cache, x̄, testf, x)
            @test has_equal_data(v, SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2))
            @test has_equal_data(
                pb[2],
                SimplePair(2x.x1 * x̄.x1 + x.x2 * x̄.x2, cos(x.x2) * x̄.x1 + x.x1 * x̄.x2),
            )

            rrule = build_rrule(testf, x)
            v, pb = Mooncake.value_and_pullback!!(rrule, x̄_unfriendly, testf, x)
            @test has_equal_data(v, SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2))
            @test has_equal_data(
                pb[2],
                Mooncake.Tangent((;
                    x1=2x.x1 * x̄.x1 + x.x2 * x̄.x2, x2=cos(x.x2) * x̄.x1 + x.x1 * x̄.x2
                )),
            )

            v, pb = Mooncake.value_and_pullback!!(
                rrule, x̄, testf, x; friendly_tangents=true
            )
            @test has_equal_data(v, SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2))
            @test has_equal_data(
                pb[2],
                SimplePair(2x.x1 * x̄.x1 + x.x2 * x̄.x2, cos(x.x2) * x̄.x1 + x.x1 * x̄.x2),
            )

            # Regression test for "invalid struct allocation" and `TypeError` error. See #1024.
            struct ImmutableWithNothingFields
                a::Float64
                b::Float64
                c::Nothing
            end
            nothing_struct = ImmutableWithNothingFields(1.0, 2.0, nothing)
            f_nothing_struct = let s = nothing_struct
                function (x::Vector{Float64})
                    return x .* s.a .+ s.b
                end
            end
            x_vec = randn(3)
            cache_ns = Mooncake.prepare_pullback_cache(
                f_nothing_struct, x_vec; config=Mooncake.Config(; friendly_tangents=true)
            )
            ȳ_vec = ones(3)
            v_ns, pb_ns = Mooncake.value_and_pullback!!(
                cache_ns, ȳ_vec, f_nothing_struct, x_vec
            )
            @test v_ns ≈ x_vec .* nothing_struct.a .+ nothing_struct.b
            @test pb_ns[2] ≈ ȳ_vec .* nothing_struct.a
        end
    end

    @testset "prepare_pullback_cache errors" begin
        # Test when function outputs a valid type.
        struct UserDefinedStruct
            a::Int64
            b::Vector{Float64}
            c::Vector{Vector{Float64}}
        end

        mutable struct UserDefinedMutableStruct
            a::Int64
            b::Vector{Float64}
            c::Vector{Vector{Float64}}
        end

        test_to_pass_cases = [
            (1, (1.0, 1.0)),
            (1.0, 1.0),
            (1, [[1.0, 1, 1.0], 1.0]),
            (1.0, [1.0]),
            UserDefinedStruct(1, [1.0, 1.0, 1.0], [[1.0]]),
            UserDefinedMutableStruct(1, [1.0, 1.0, 1.0], [[1.0]]),
            Dict(:a => [1, 2], :b => [3, 4]),
            Set([1, 2]),
        ]
        VERSION >= v"1.11" &&
            push!(test_to_pass_cases, fill!(Memory{Float64}(undef, 3), 3.0))

        @testset "Valid Output types" for res in test_to_pass_cases
            @test isnothing(Mooncake.__exclude_unsupported_output(res))
        end

        # Test when function outputs an invalid type. 
        test_to_fail_cases = []

        # Aliasing Cases
        alias_vector = [rand(Int64, 2), rand(Int64, 2)]
        alias_vector[2] = alias_vector[1]
        push!(test_to_fail_cases, (identity, alias_vector))

        alias_tuple = (rand(2), rand(2))
        alias_tuple = (alias_tuple[1], alias_tuple[1])
        push!(test_to_fail_cases, (identity, alias_tuple))

        # Circular Referencing Cases
        circular_vector = Any[rand(2)]
        push!(circular_vector, circular_vector)
        push!(test_to_fail_cases, (identity, circular_vector))

        mutable struct CircularStruct
            data::Any
            numeric::Int64
        end

        circ_obj = CircularStruct(nothing, rand(Int64, 1)[1])
        circ_obj.data = circ_obj  # Self-referential struct
        push!(test_to_fail_cases, (identity, circ_obj))

        # Exclude `Ptr` typed input arguments and returned values
        push!(test_to_fail_cases, ((x) -> Ptr{Float64}(x[1]), rand(UInt, 1)))
        push!(
            test_to_fail_cases,
            ((x) -> (rand(UInt, 1), [Ptr{Float64}(x_i) for x_i in x]), rand(UInt, 5)),
        )

        @testset "prepare_pullback_cache checks" for (f, test_case) in test_to_fail_cases
            @test_throws(
                Mooncake.ValueAndPullbackReturnTypeError,
                Mooncake.__exclude_unsupported_output(f(test_case))
            )
            @test_throws(
                Mooncake.ValueAndPullbackReturnTypeError,
                Mooncake.prepare_pullback_cache(f, test_case)
            )
        end

        additional_test_set = Mooncake.tangent_test_cases()

        @testset "__exclude_unsupported_output , $(test_set)" for test_set in
                                                                  additional_test_set

            try
                Mooncake.__exclude_unsupported_output(test_set[2])
            catch err
                @test isa(err, Mooncake.ValueAndPullbackReturnTypeError)
            end
        end

        @testset "_copy_output & _copy_to_output!!, $(test_set)" for test_set in
                                                                     additional_test_set

            original = test_set[2]
            try
                if isnothing(Mooncake.__exclude_unsupported_output(original))
                    test_copy = Mooncake._copy_output(original)
                    test_inplace_copy = Mooncake._copy_to_output!!(test_copy, original)

                    @test Mooncake.TestUtils.has_equal_data(original, test_copy)
                    @test Mooncake.TestUtils.has_equal_data(original, test_inplace_copy)
                    @test typeof(test_copy) == typeof(original)
                end
            catch err
                @test isa(err, Mooncake.ValueAndPullbackReturnTypeError)
            end
        end

        # `_copy_output` needs to be able handle `Type`, `Core.TypeName`,
        # and `Module` values. See #1024.
        @testset "_copy_output non-deep-copyable types" begin
            # Type values
            @test Mooncake._copy_output(Float64) === Float64
            @test Mooncake._copy_output(Vector{Float64}) === Vector{Float64}
            @test Mooncake._copy_output(Union{Float64,Int64}) === Union{Float64,Int64}

            # Core.TypeName
            @test Mooncake._copy_output(Float64.name) === Float64.name

            # Module
            @test Mooncake._copy_output(Base) === Base

            # _copy_to_output!! for the same non-deep-copyable types
            @test Mooncake._copy_to_output!!(Float64, Float64) === Float64
            @test Mooncake._copy_to_output!!(Float64.name, Float64.name) === Float64.name
            @test Mooncake._copy_to_output!!(Base, Base) === Base

            # Mutable struct containing a Type field.
            mutable struct MutableWithTypeField
                t::Type
                x::Float64
            end
            obj = MutableWithTypeField(Float64, 1.0)
            obj_copy = Mooncake._copy_output(obj)
            @test typeof(obj_copy) == MutableWithTypeField
            @test obj_copy.t === Float64
            @test obj_copy.x == 1.0
            obj2 = MutableWithTypeField(Int64, 2.0)
            Mooncake._copy_to_output!!(obj_copy, obj2)
            @test obj_copy.t === Int64
            @test obj_copy.x == 2.0
        end

        # Fix for #1033: opaque mutable types (nfields == 0).
        @testset "_copy_output opaque mutable types (Symbol, String, Dict)" begin
            # Symbol and String are mutable with 0 user-visible fields
            @test Mooncake._copy_output(:hello) === :hello
            @test Mooncake._copy_output("hello") === "hello"

            # _copy_to_output!! must return src for opaque mutable types, not dst
            @test Mooncake._copy_to_output!!(:hello, :world) === :world
            @test Mooncake._copy_to_output!!("hello", "world") === "world"

            # Dict contains a Memory{Symbol} (keys) internally
            d = Dict(:x => 1, :y => 2)
            d_copy = Mooncake._copy_output(d)
            @test d_copy == d
            @test d_copy !== d

            # Dict{Symbol, Any}
            d2 = Dict{Symbol,Any}(:x => [1.0, 2.0], :n => 3)
            d2_copy = Mooncake._copy_output(d2)
            @test d2_copy == d2
            @test d2_copy !== d2

            # Struct containing a Dict must also be copyable
            struct DataStoreForTest
                _n::Int
                _data::Dict{Symbol,Any}
            end
            ds = DataStoreForTest(3, Dict{Symbol,Any}(:x => randn(Float32, 2)))
            ds_copy = Mooncake._copy_output(ds)
            @test ds_copy._n == ds._n
            @test ds_copy._data == ds._data
            ds2 = DataStoreForTest(5, Dict{Symbol,Any}(:y => randn(Float32, 2)))
            ds_copy2 = Mooncake._copy_to_output!!(ds_copy, ds2)
            @test ds_copy2._n == ds2._n
            @test ds_copy2._data == ds2._data
        end
    end
    @testset "forwards mode ($kwargs)" for kwargs in [
        (;),
        (; debug_mode=true),
        (; debug_mode=false),
        (; debug_mode=true, silence_debug_messages=true),
    ]
        f = (x, y) -> x * y + cos(x)
        g = (sp::SimplePair) -> SimplePair(f(sp.x1, sp.x2), 2.0)

        x, y = 5.0, 4.0
        dx, dy = 3.0, 2.0
        fx = (f, x, y)
        dfx = (Mooncake.zero_tangent(f), dx, dy)
        z = f(x, y)
        dz = dx * y + x * dy + dx * (-sin(x))

        fx_sp = (g, SimplePair(x, y))
        dfx_sp = (Mooncake.zero_tangent(g), SimplePair(dx, dy))
        z_sp = g(SimplePair(x, y))

        @testset "Simple types" begin
            cache = Mooncake.prepare_derivative_cache(
                fx...; config=Mooncake.Config(; kwargs...)
            )

            # legacy Dual interface
            z_and_dz_dual = Mooncake.value_and_derivative!!(
                cache, map(Mooncake.Dual, fx, dfx)...
            )
            @test z_and_dz_dual isa Mooncake.Dual
            @test Mooncake.primal(z_and_dz_dual) == z
            @test Mooncake.tangent(z_and_dz_dual) == dz

            # new tuple interface
            z_and_dz_tup = Mooncake.value_and_derivative!!(cache, zip(fx, dfx)...)
            @test z_and_dz_tup isa Tuple{Float64,Float64}
            @test first(z_and_dz_tup) == z
            @test last(z_and_dz_tup) == dz
        end

        @testset "Structured types" begin
            cache_sp_friendly = Mooncake.prepare_derivative_cache(
                fx_sp...; config=Mooncake.Config(; friendly_tangents=true, kwargs...)
            )
            # friendly input doesn't error
            z_and_dz_sp = Mooncake.value_and_derivative!!(
                cache_sp_friendly, zip(fx_sp, dfx_sp)...
            )
            # output is friendly
            @test z_and_dz_sp isa Tuple{SimplePair,SimplePair}
            @test first(z_and_dz_sp) == SimplePair(z, 2.0)
            @test last(z_and_dz_sp) == SimplePair(dz, 0.0)

            cache_sp_unfriendly = Mooncake.prepare_derivative_cache(
                fx_sp...; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
            )
            @test_throws ArgumentError Mooncake.value_and_derivative!!(
                cache_sp_unfriendly, zip(fx_sp, dfx_sp)...
            )
            @test_throws "Tangent types do not match primal types:" Mooncake.value_and_derivative!!(
                cache_sp_unfriendly, zip(fx_sp, dfx_sp)...
            )
        end
    end

    @testset "value_and_hessian!!" begin
        rosen(z) = (1 - z[1])^2 + 100 * (z[2] - z[1]^2)^2
        function rosen_H(z)
            h11 = 2 - 400 * (z[2] - z[1]^2) + 800 * z[1]^2
            h12 = -400 * z[1]
            return [h11 h12; h12 200.0]
        end
        rosen_g(z) = [-2*(1 - z[1]) - 400*z[1]*(z[2] - z[1]^2), 200*(z[2] - z[1]^2)]

        @testset "Rosenbrock Float64" begin
            z = [1.2, 1.2]
            cache = prepare_hessian_cache(rosen, z)
            v, g, H = value_and_hessian!!(cache, rosen, z)
            @test v ≈ rosen(z)
            @test g ≈ rosen_g(z) rtol = 1e-10
            @test H ≈ rosen_H(z) rtol = 1e-10
        end

        @testset "Rosenbrock Float32" begin
            z = Float32[1.2, 1.2]
            cache = prepare_hessian_cache(rosen, z)
            v, g, H = value_and_hessian!!(cache, rosen, z)
            @test v isa Float32
            @test g isa Vector{Float32}
            @test H isa Matrix{Float32}
            @test v ≈ rosen(z) rtol = 1e-4
            @test H ≈ rosen_H(Float64[1.2, 1.2]) rtol = 1e-4
        end

        @testset "quadratic (diagonal Hessian)" begin
            f(x) = sum(x .^ 2)
            x = [1.0, 2.0, 3.0]
            cache = prepare_hessian_cache(f, x)
            v, g, H = value_and_hessian!!(cache, f, x)
            @test v ≈ 14.0
            @test g ≈ [2.0, 4.0, 6.0]
            @test H ≈ 2 * I
        end

        @testset "cache reuse with different x" begin
            f(x) = sum(x .^ 2)
            x1 = [1.0, 0.0]
            x2 = [2.0, 3.0]
            cache = prepare_hessian_cache(f, x1)
            v1, g1, H1 = value_and_hessian!!(cache, f, x1)
            v2, g2, H2 = value_and_hessian!!(cache, f, x2)
            @test v1 ≈ 1.0
            @test v2 ≈ 13.0
            @test g1 ≈ [2.0, 0.0]
            @test g2 ≈ [4.0, 6.0]
            @test H1 ≈ H2  # Hessian of sum(x^2) is 2I everywhere
        end

        @testset "debug_mode=true" begin
            z = [1.2, 1.2]
            cache = prepare_hessian_cache(
                rosen, z; config=Mooncake.Config(; debug_mode=true)
            )
            v, g, H = value_and_hessian!!(cache, rosen, z)
            @test v ≈ rosen(z)
            @test H ≈ rosen_H(z) rtol = 1e-10
        end

        @testset "n=0 edge case" begin
            f(x) = 0.0
            x = Float64[]
            cache = prepare_hessian_cache(f, x)
            v, g, H = value_and_hessian!!(cache, f, x)
            @test v == 0.0
            @test g == Float64[]
            @test H == zeros(0, 0)
        end

        @testset "multi-arg: two vectors" begin
            # f(x, y) = x'x + y'y + x[1]*y[1]
            # ∇_x f = 2x + [y[1], 0, ...]
            # ∇_y f = 2y + [x[1], 0, ...]
            # H_xx = 2I, H_yy = 2I
            # H_xy = H_yx = [[1, 0, ...]; [0, ...]] (only [1,1] entry is 1)
            f(x, y) = sum(x .^ 2) + sum(y .^ 2) + x[1] * y[1]
            x = [1.0, 2.0]
            y = [3.0, 4.0]
            cache = prepare_hessian_cache(f, x, y)
            val, (gx, gy), ((Hxx, Hxy), (Hyx, Hyy)) = value_and_hessian!!(cache, f, x, y)
            @test val ≈ f(x, y)
            @test gx ≈ 2x + [y[1], 0.0] rtol = 1e-10
            @test gy ≈ 2y + [x[1], 0.0] rtol = 1e-10
            @test Hxx ≈ 2 * I rtol = 1e-10
            @test Hyy ≈ 2 * I rtol = 1e-10
            @test Hxy ≈ [1.0 0.0; 0.0 0.0] rtol = 1e-10
            @test Hyx ≈ [1.0 0.0; 0.0 0.0] rtol = 1e-10
        end

        @testset "multi-arg: cache reuse" begin
            f(x, y) = sum(x .^ 2) + sum(y .^ 2)
            x1, y1 = [1.0, 0.0], [0.0, 1.0]
            x2, y2 = [2.0, 3.0], [4.0, 5.0]
            cache = prepare_hessian_cache(f, x1, y1)
            v1, (gx1, gy1), ((Hxx1, _), (_, Hyy1)) = value_and_hessian!!(cache, f, x1, y1)
            v2, (gx2, gy2), ((Hxx2, _), (_, Hyy2)) = value_and_hessian!!(cache, f, x2, y2)
            @test v1 ≈ f(x1, y1)
            @test v2 ≈ f(x2, y2)
            @test gx1 ≈ 2x1
            @test gx2 ≈ 2x2
            @test Hxx1 ≈ 2 * I
            @test Hxx2 ≈ 2 * I  # Hessian is constant for sum of squares
            @test Hyy1 ≈ 2 * I
            @test Hyy2 ≈ 2 * I
        end

        @testset "multi-arg: first arg empty" begin
            # Regression test: when ns[1]==0 but other args non-empty, value and grads
            # must still be initialised (previously caused UndefVarError).
            f(x, y) = sum(y .^ 2)
            x = Float64[]
            y = [1.0, 2.0]
            cache = prepare_hessian_cache(f, x, y)
            val, (gx, gy), ((Hxx, Hxy), (Hyx, Hyy)) = value_and_hessian!!(cache, f, x, y)
            @test val ≈ f(x, y)
            @test gx == Float64[]
            @test gy ≈ 2y
            @test Hxx == zeros(0, 0)
            @test Hyy ≈ 2 * I
        end
    end

    @testset "selective zeroing of cotangents" begin
        f = (x, y) -> sum(abs2, x) - sum(abs2, y)
        x = [1.0, 2.0]
        y = [3.0, 4.0]

        @testset "Pullback cache" begin
            cache_pb = prepare_pullback_cache(f, x, y)
            value_and_pullback!!(cache_pb, 1.0, f, x, y)
            @test cache_pb.tangents[2] == 2x
            @test cache_pb.tangents[3] == -2y
            value_and_pullback!!(cache_pb, 1.0, f, x, y)
            @test cache_pb.tangents[2] == 2x
            @test cache_pb.tangents[3] == -2y
            value_and_pullback!!(cache_pb, 1.0, f, x, y; args_to_zero=(true, false, true))
            @test cache_pb.tangents[2] == 4x
            @test cache_pb.tangents[3] == -2y
            value_and_pullback!!(cache_pb, 1.0, f, x, y; args_to_zero=(true, true, false))
            @test cache_pb.tangents[2] == 2x
            @test cache_pb.tangents[3] == -4y
        end

        @testset "Gradient cache" begin
            cache_grad = prepare_gradient_cache(f, x, y)
            value_and_gradient!!(cache_grad, f, x, y)
            @test cache_grad.tangents[2] == 2x
            @test cache_grad.tangents[3] == -2y
            value_and_gradient!!(cache_grad, f, x, y)
            @test cache_grad.tangents[2] == 2x
            @test cache_grad.tangents[3] == -2y
            value_and_gradient!!(cache_grad, f, x, y; args_to_zero=(true, false, true))
            @test cache_grad.tangents[2] == 4x
            @test cache_grad.tangents[3] == -2y
            value_and_gradient!!(cache_grad, f, x, y; args_to_zero=(true, true, false))
            @test cache_grad.tangents[2] == 2x
            @test cache_grad.tangents[3] == -4y
        end
    end
end
