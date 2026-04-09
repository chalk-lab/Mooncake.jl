using Mooncake: TestUtils
using Mooncake:
    prepare_gradient_cache,
    prepare_hvp_cache,
    prepare_hessian_cache,
    prepare_pullback_cache,
    value_and_gradient!!,
    value_and_hvp!!,
    value_gradient_and_hessian!!,
    value_and_pullback!!

struct SimplePair
    x1::Float64
    x2::Float64
end

struct WithSymField
    m::LinearAlgebra.Symmetric{Float64,Matrix{Float64}}
    v::Float64
end

struct ScalarBox
    x::Float64
end

mutable struct AliasedPair
    a::Vector{Float64}
    b::Vector{Float64}
end

mutable struct AnyCycleNode
    next::Any
    weight::Float64
end

mutable struct MaybeInitBox
    x::Float64
    y::Float64
    MaybeInitBox(x::Float64) = new(x)
end

const CHUNK_SCALAR_EVAL_COUNT = Ref(0)
struct CountedChunkScalarCall end
(::CountedChunkScalarCall)(x, y) = (CHUNK_SCALAR_EVAL_COUNT[] += 1; x * y + cos(x))

const CHUNK_ARRAY_EVAL_COUNT = Ref(0)
struct CountedChunkArrayCall end
(::CountedChunkArrayCall)(x) = (CHUNK_ARRAY_EVAL_COUNT[] += 1; sum(abs2, x))

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
            alloc_count = TestUtils.count_allocs(value_and_gradient!!, cache, fargs...)
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

        @testset "cache display" begin
            reverse_cache = Mooncake.prepare_gradient_cache(
                sin, 1.0; config=Mooncake.Config(; debug_mode=false, friendly_tangents=true)
            )
            reverse_show = sprint(show, reverse_cache)
            @test occursin("Mooncake.Cache(", reverse_show)
            @test occursin("mode=:reverse", reverse_show)
            @test occursin("friendly_tangents=true", reverse_show)

            reverse_plain = repr(MIME"text/plain"(), reverse_cache)
            @test occursin("Mooncake.Cache", reverse_plain)
            @test occursin("mode: reverse", reverse_plain)
            @test occursin("friendly_tangents: true", reverse_plain)
            @test occursin("input_1: Float64 (scalar)", reverse_plain)
            @test occursin("output: Float64 (scalar)", reverse_plain)

            forward_cache = Mooncake.prepare_derivative_cache(
                sin, 1.0; config=Mooncake.Config(; chunk_size=2)
            )
            forward_show = sprint(show, forward_cache)
            @test occursin("Mooncake.NfwdCache(", forward_show)
            @test occursin("mode=:forward", forward_show)
            @test occursin("chunk_size=2", forward_show)

            forward_plain = repr(MIME"text/plain"(), forward_cache)
            @test occursin("Mooncake.NfwdCache", forward_plain)
            @test occursin("mode: forward", forward_plain)
            @test occursin("chunk_size: 2", forward_plain)
            @test occursin("inputs: 2", forward_plain)

            hvp_cache = Mooncake.prepare_hvp_cache(sin, 1.0)
            hvp_show = sprint(show, hvp_cache)
            @test occursin("Mooncake.HVPCache(", hvp_show)
            @test occursin("mode=:forward_over_reverse", hvp_show)

            hvp_plain = repr(MIME"text/plain"(), hvp_cache)
            @test occursin("Mooncake.HVPCache", hvp_plain)
            @test occursin("mode: forward_over_reverse", hvp_plain)
            @test occursin("input_1: Float64 (scalar)", hvp_plain)
            @test occursin("output: Float64 (scalar)", hvp_plain)
        end

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
            # SimplePair has no :as_primal opt-in; friendly tangent is a NamedTuple of fields
            @test dx[2] isa @NamedTuple{x1::Float64, x2::Float64}
            @test dx[2] == (; x1=2 * x.x1, x2=cos(x.x2))

            rule = build_rrule(f, x)

            v, dx = Mooncake.value_and_gradient!!(rule, f, x)
            @test dx[2] isa Mooncake.Tangent{@NamedTuple{x1::Float64,x2::Float64}}
            @test dx[2].fields == (; x1=2 * x.x1, x2=cos(x.x2))

            v, dx = Mooncake.value_and_gradient!!(rule, f, x; friendly_tangents=true)
            # SimplePair has no :as_primal opt-in; friendly tangent is a NamedTuple of fields
            @test dx[2] isa @NamedTuple{x1::Float64, x2::Float64}
            @test dx[2] == (; x1=2 * x.x1, x2=cos(x.x2))

            # Struct with a Symmetric field: friendly gradient unpacks the Symmetric tangent
            # to a plain Matrix (MWE 1 & 2 from temp/friendly_tangent_mwes.jl).
            foo = WithSymField(LinearAlgebra.Symmetric([1.0 2.0; 3.0 4.0]), 3.14)
            # Use element access rather than sum: Base.sum uses Base._InitialValue as its
            # initial accumulator, producing Union{Base._InitialValue, Float64} during
            # tracing. fcodual_type then returns a non-concrete Union type, which
            # DispatchDoctor flags as a type instability (pre-existing Base behaviour).
            f_sym = (x::WithSymField) -> x.m[1, 1] + x.m[2, 1] + x.v^2

            rule_sym = build_rrule(f_sym, foo)
            _, grads_sym = Mooncake.value_and_gradient!!(
                rule_sym, f_sym, foo; friendly_tangents=true
            )
            @test grads_sym[2] isa NamedTuple{(:m, :v)}
            @test grads_sym[2].m isa Matrix{Float64}
            # m[1,1] and m[2,1] both read from data[1,1] and data[1,2] respectively
            # (Symmetric :U stores upper triangle; m[2,1] aliases data[1,2]).
            @test grads_sym[2].m ≈ [1.0 1.0; 0.0 0.0]
            @test grads_sym[2].v ≈ 2 * foo.v

            cache_sym = Mooncake.prepare_gradient_cache(
                f_sym, foo; config=Mooncake.Config(; friendly_tangents=true)
            )
            _, dx_sym = Mooncake.value_and_gradient!!(cache_sym, f_sym, foo)
            @test dx_sym[2] isa NamedTuple{(:m, :v)}
            @test dx_sym[2].m isa Matrix{Float64}
            @test dx_sym[2].m == grads_sym[2].m
            @test dx_sym[2].v ≈ grads_sym[2].v
            _, dx_sym2 = Mooncake.value_and_gradient!!(cache_sym, f_sym, foo)
            @test dx_sym2[2].m === dx_sym[2].m
            @test dx_sym2[2].m == grads_sym[2].m

            # Vector of structs: friendly gradient returns a Vector of the same struct type
            # (MWE 3 from temp/friendly_tangent_mwes.jl).
            f_vec = (v::Vector{ScalarBox}) -> sum(b.x^2 for b in v)
            v_boxes = [ScalarBox(1.0), ScalarBox(2.0), ScalarBox(3.0)]
            rule_vec = build_rrule(f_vec, v_boxes)
            _, grads_vec = Mooncake.value_and_gradient!!(
                rule_vec, f_vec, v_boxes; friendly_tangents=true
            )
            # ScalarBox is a struct so friendly tangent is a NamedTuple; the Vector of such
            # NamedTuples is returned as a Vector{@NamedTuple{x::Float64}}.
            @test grads_vec[2] isa AbstractVector
            @test [g.x for g in grads_vec[2]] ≈ [2.0, 4.0, 6.0]
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
            alloc_count = TestUtils.count_allocs(value_and_pullback!!, cache, ȳ, fargs...)
            if alloc_count > 0
                @test_broken alloc_count == 0
            else
                @test alloc_count == 0
            end
        end

        @testset "pullback cache mismatch errors" begin
            f_arr = x -> sum(abs2, x)
            x_arr = [1.0, 2.0]
            cache = Mooncake.prepare_pullback_cache(f_arr, x_arr)

            @test_throws r"Cached autodiff call has a size mismatch for `x1`" Mooncake.value_and_pullback!!(
                cache, 1.0, f_arr, [1.0, 2.0, 3.0]
            )
            @test_throws r"Cached autodiff call has a type mismatch for `x1`" Mooncake.value_and_pullback!!(
                cache, 1.0, f_arr, Float32[1.0, 2.0]
            )
            @test_throws r"Cached autodiff call has a type mismatch for `x1`" Mooncake.value_and_pullback!!(
                cache, 1.0, f_arr, reshape([1.0, 2.0], 2, 1)
            )
        end

        @testset "friendly tangents" begin
            testf(x::SimplePair) = SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2)
            x = SimplePair(1.0, 2.0)
            x̄ = SimplePair(0.5, 0.3)
            x̄_unfriendly = Mooncake.Tangent((; x1=0.5, x2=0.3))

            cache = Mooncake.prepare_pullback_cache(testf, x)
            v, pb = Mooncake.value_and_pullback!!(cache, x̄_unfriendly, testf, x)
            @test TestUtils.has_equal_data(v, SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2))
            @test TestUtils.has_equal_data(
                pb[2],
                Mooncake.Tangent((;
                    x1=2x.x1 * x̄.x1 + x.x2 * x̄.x2, x2=cos(x.x2) * x̄.x1 + x.x1 * x̄.x2
                )),
            )

            cache = Mooncake.prepare_pullback_cache(
                testf, x; config=Mooncake.Config(; friendly_tangents=true)
            )
            # SimplePair has no :as_primal opt-in; friendly tangent is a NamedTuple of fields.
            # ȳ is passed as a primal (SimplePair); output gradient is a NamedTuple.
            v, pb = Mooncake.value_and_pullback!!(cache, x̄, testf, x)
            @test TestUtils.has_equal_data(v, SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2))
            @test TestUtils.has_equal_data(
                pb[2],
                (; x1=2x.x1 * x̄.x1 + x.x2 * x̄.x2, x2=cos(x.x2) * x̄.x1 + x.x1 * x̄.x2),
            )

            rrule = build_rrule(testf, x)
            v, pb = Mooncake.value_and_pullback!!(rrule, x̄_unfriendly, testf, x)
            @test TestUtils.has_equal_data(v, SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2))
            @test TestUtils.has_equal_data(
                pb[2],
                Mooncake.Tangent((;
                    x1=2x.x1 * x̄.x1 + x.x2 * x̄.x2, x2=cos(x.x2) * x̄.x1 + x.x1 * x̄.x2
                )),
            )

            v, pb = Mooncake.value_and_pullback!!(
                rrule, x̄, testf, x; friendly_tangents=true
            )
            @test TestUtils.has_equal_data(v, SimplePair(x.x1^2 + sin(x.x2), x.x1 * x.x2))
            @test TestUtils.has_equal_data(
                pb[2],
                (; x1=2x.x1 * x̄.x1 + x.x2 * x̄.x2, x2=cos(x.x2) * x̄.x1 + x.x1 * x̄.x2),
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

                    @test TestUtils.has_equal_data(original, test_copy)
                    @test TestUtils.has_equal_data(original, test_inplace_copy)
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
    @testset "forwards mode" begin
        f = (x, y) -> x * y + cos(x)

        x, y = 5.0, 4.0
        dx, dy = 3.0, 2.0
        fx = (f, x, y)
        dfx = (Mooncake.zero_tangent(f), dx, dy)
        z = f(x, y)
        dz = dx * y + x * dy + dx * (-sin(x))

        @testset "Simple types" begin
            cache = Mooncake.prepare_derivative_cache(fx...)

            # legacy Dual interface
            z_and_dz_dual = Mooncake.value_and_derivative!!(
                cache, map(Mooncake.Dual, fx, dfx)...
            )
            @test z_and_dz_dual isa Mooncake.Dual
            @test Mooncake.primal(z_and_dz_dual) == z
            @test Mooncake.tangent(z_and_dz_dual) == Mooncake.NTangent((dz,))

            # new tuple interface
            z_and_dz_tup = Mooncake.value_and_derivative!!(cache, zip(fx, dfx)...)
            @test z_and_dz_tup isa Tuple{Float64,Mooncake.NTangent}
            @test first(z_and_dz_tup) == z
            @test last(z_and_dz_tup) == Mooncake.NTangent((dz,))

            z_and_dz_explicit_dual = Mooncake.value_and_derivative!!(
                cache,
                Mooncake.Dual(f, Mooncake.NoTangent()),
                Mooncake.Dual{Float64,Float64}(x, dx),
                Mooncake.Dual{Float64,Float64}(y, dy),
            )
            @test z_and_dz_explicit_dual isa Mooncake.Dual
            @test Mooncake.primal(z_and_dz_explicit_dual) == z
            @test Mooncake.tangent(z_and_dz_explicit_dual) == Mooncake.NTangent((dz,))
        end

        @testset "Array inputs" begin
            f_arr = x -> sum(abs2, x)
            x_arr = [x, y]
            dx_arr_1 = [dx, 0.0]
            dx_arr_2 = [0.0, dy]

            cache_arr = Mooncake.prepare_derivative_cache(f_arr, x_arr)
            @test Mooncake.value_and_derivative!!(
                cache_arr, (f_arr, Mooncake.zero_tangent(f_arr)), (x_arr, dx_arr_1)
            ) == (sum(abs2, x_arr), Mooncake.NTangent((2 * dot(x_arr, dx_arr_1),)))

            width2_rule = Mooncake.build_frule(sum, x_arr; chunk_size=2)
            @test Mooncake.value_and_derivative!!(
                width2_rule,
                (sum, Mooncake.NoTangent()),
                (x_arr, Mooncake.NTangent((ones(2), fill(2.0, 2)))),
            ) == (sum(x_arr), Mooncake.NTangent((2.0, 4.0)))

            counted_scalar = CountedChunkScalarCall()
            counted_scalar_cache = Mooncake.prepare_derivative_cache(
                counted_scalar, x, y; config=Mooncake.Config(; chunk_size=2)
            )
            CHUNK_SCALAR_EVAL_COUNT[] = 0
            @test Mooncake.value_and_derivative!!(
                counted_scalar_cache,
                (counted_scalar, Mooncake.NoTangent()),
                (x, Mooncake.NTangent((dx, 0.0))),
                (y, Mooncake.NTangent((0.0, dy))),
            ) == (z, Mooncake.NTangent((dx * y + dx * (-sin(x)), x * dy)))
            @test CHUNK_SCALAR_EVAL_COUNT[] == 1

            counted_array = CountedChunkArrayCall()
            counted_array_cache = Mooncake.prepare_derivative_cache(
                counted_array, x_arr; config=Mooncake.Config(; chunk_size=2)
            )
            CHUNK_ARRAY_EVAL_COUNT[] = 0
            @test Mooncake.value_and_derivative!!(
                counted_array_cache,
                (counted_array, Mooncake.NoTangent()),
                (x_arr, Mooncake.NTangent((dx_arr_1, dx_arr_2))),
            ) == (sum(abs2, x_arr), Mooncake.NTangent((2 * x * dx, 2 * y * dy)))
            @test CHUNK_ARRAY_EVAL_COUNT[] == 1
        end

        @testset "Non-differentiable outputs" begin
            f_int = x -> x > 0 ? 1 : 2
            cache_int = Mooncake.prepare_derivative_cache(f_int, x)
            @test Mooncake.value_and_derivative!!(
                cache_int, (f_int, Mooncake.zero_tangent(f_int)), (x, dx)
            ) == (f_int(x), Mooncake.NoTangent())
        end

        @testset "cached forward gradient/pullback" begin
            cache = Mooncake.prepare_derivative_cache(f, x, y)
            @test Mooncake.value_and_gradient!!(cache, f, x, y) ==
                (z, (Mooncake.NoTangent(), y - sin(x), x))
            @test Mooncake.value_and_pullback!!(cache, 1.0, f, x, y) ==
                (z, (Mooncake.NoTangent(), y - sin(x), x))
        end

        @testset "call-time derivative width is inferred from the call" begin
            counted_scalar = CountedChunkScalarCall()
            cache = Mooncake.prepare_derivative_cache(
                counted_scalar, x, y; config=Mooncake.Config(; chunk_size=2)
            )
            @test Mooncake.value_and_derivative!!(
                cache, (counted_scalar, Mooncake.NoTangent()), (x, dx), (y, dy)
            ) == (z, Mooncake.NTangent((dz,)))

            cache_width_1 = Mooncake.prepare_derivative_cache(counted_scalar, x, y)
            @test Mooncake.value_and_derivative!!(
                cache_width_1,
                (counted_scalar, Mooncake.NoTangent()),
                (x, Mooncake.NTangent((dx, -dx))),
                (y, Mooncake.NTangent((dy, -dy))),
            ) == (z, Mooncake.NTangent((dz, -dz)))
        end

        @testset "zero-dof prepared gradients return zero gradients" begin
            f0() = 3.0
            cache0 = Mooncake.prepare_derivative_cache(f0)
            @test Mooncake.value_and_gradient!!(cache0, f0) ==
                (3.0, (Mooncake.NoTangent(),))
            @test Mooncake.value_and_pullback!!(cache0, 2.0, f0) ==
                (3.0, (Mooncake.NoTangent(),))

            f_empty(x) = 7.0
            x_empty = Float64[]
            cache_empty = Mooncake.prepare_derivative_cache(f_empty, x_empty)
            @test Mooncake.value_and_gradient!!(cache_empty, f_empty, x_empty) ==
                (7.0, (Mooncake.NoTangent(), Float64[]))
            @test Mooncake.value_and_pullback!!(cache_empty, 2.0, f_empty, x_empty) ==
                (7.0, (Mooncake.NoTangent(), Float64[]))
        end

        @testset "complex scalar prepared pullback" begin
            f_complex(z) = z^2
            z_complex = 1.0 + 2.0im
            cache_complex = Mooncake.prepare_derivative_cache(f_complex, z_complex)
            @test Mooncake.value_and_pullback!!(
                cache_complex, 1.0 + 0.0im, f_complex, z_complex
            ) == (f_complex(z_complex), (Mooncake.NoTangent(), 2 * conj(z_complex)))
            @test Mooncake.value_and_pullback!!(
                cache_complex, 0.0 + 1.0im, f_complex, z_complex
            ) == (f_complex(z_complex), (Mooncake.NoTangent(), 2im * conj(z_complex)))
        end

        @testset "forward cache mismatch errors" begin
            f_arr = x -> sum(abs2, x)
            x_arr = [x, y]
            dx_arr = [dx, 0.0]
            cache = Mooncake.prepare_derivative_cache(f_arr, x_arr)

            @test_throws r"Cached autodiff call has a size mismatch for `x1`" Mooncake.value_and_derivative!!(
                cache, (f_arr, Mooncake.NoTangent()), ([x, y, 3.0], [dx, 0.0, 0.0])
            )
            @test_throws r"Cached autodiff call has a type mismatch for `x1`" Mooncake.value_and_derivative!!(
                cache, (f_arr, Mooncake.NoTangent()), (Float32[x, y], Float32[dx, 0.0])
            )
            @test_throws r"Cached autodiff call has a type mismatch for `x1`" Mooncake.value_and_derivative!!(
                cache,
                (f_arr, Mooncake.NoTangent()),
                (reshape([x, y], 2, 1), reshape([dx, 0.0], 2, 1)),
            )
        end

        @testset "NfwdCache boundary rejection" begin
            @test_throws ArgumentError Mooncake.prepare_derivative_cache(
                sin, 1.0; config=Mooncake.Config(; debug_mode=true)
            )
            @test_throws ArgumentError Mooncake.prepare_derivative_cache(
                sin, 1.0; config=Mooncake.Config(; friendly_tangents=true)
            )
            f_sp = (sp::SimplePair) -> sp.x1^2 + sin(sp.x2)
            @test_throws ArgumentError Mooncake.prepare_derivative_cache(
                f_sp, SimplePair(1.0, 2.0)
            )
            @test_throws ArgumentError Mooncake.prepare_derivative_cache(
                t -> t[1]^2 + sin(t[2]), (1.0, 2.0)
            )
            @test_throws ArgumentError Mooncake.prepare_derivative_cache(
                nt -> nt.a * sin(nt.b), (; a=1.0, b=2.0)
            )
        end

        @testset "reverse cache mismatch errors" begin
            f_arr = x -> sum(abs2, x)
            x_arr = [x, y]
            cache = Mooncake.prepare_gradient_cache(f_arr, x_arr)

            @test_throws r"Cached autodiff call has a size mismatch for `x1`" Mooncake.value_and_gradient!!(
                cache, f_arr, [x, y, 3.0]
            )
            @test_throws r"Cached autodiff call has a type mismatch for `x1`" Mooncake.value_and_gradient!!(
                cache, f_arr, Float32[x, y]
            )
            @test_throws r"Cached autodiff call has a type mismatch for `x1`" Mooncake.value_and_gradient!!(
                cache, f_arr, reshape([x, y], 2, 1)
            )
        end
    end

    @testset "value_and_hvp!!" begin
        TestUtils.test_hook(Val(:allow_unstable_hvp_interface_test)) do
            @testset "primal dof skips undefined builtin-array slots" begin
                x = Vector{Any}(undef, 2)
                x[1] = 1.0
                @test Mooncake._fold_slots(
                    (acc, _, _) -> acc + 1, 0, x, (seen=IdDict{Any,Any}(),)
                ) == 1
            end

            @testset "multi-argument HVP validates direction arity" begin
                f(x, y) = sum(x .* x) + sum(y .* y)
                x = [1.0, 2.0]
                y = [3.0]
                cache = prepare_hvp_cache(f, x, y)
                @test_throws ArgumentError value_and_hvp!!(cache, f, ([1.0, 0.0],), x, y)
            end

            @testset "HVP validates tangent shapes" begin
                f(x, y) = sum(x .* x) + sum(y .* y)
                x = [1.0, 2.0]
                y = [3.0]
                cache1 = prepare_hvp_cache(sum, x)
                @test_throws ArgumentError value_and_hvp!!(cache1, sum, [1.0], x)

                cache2 = prepare_hvp_cache(f, x, y)
                @test_throws ArgumentError value_and_hvp!!(cache2, f, ([1.0], [0.0]), x, y)
            end

            @testset "HVP cache mismatch errors" begin
                f(x) = sum(x .* x)
                x = [1.0, 2.0]
                cache = prepare_hvp_cache(f, x)
                @test_throws r"Cached autodiff call has a size mismatch for `x1`" value_and_hvp!!(
                    cache, f, [1.0, 0.0, 0.0], [1.0, 2.0, 3.0]
                )
                @test_throws r"Cached autodiff call has a type mismatch for `x1`" value_and_hvp!!(
                    cache, f, Float32[1.0, 0.0], Float32[1.0, 2.0]
                )
                @test_throws r"Cached autodiff call has a type mismatch for `x1`" value_and_hvp!!(
                    cache, f, reshape([1.0, 0.0], 2, 1), reshape([1.0, 2.0], 2, 1)
                )
            end
        end
    end

    @testset "value_gradient_and_hessian!!" begin
        TestUtils.test_hook(Val(:allow_unstable_hessian_interface_test)) do
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
                v, g, H = value_gradient_and_hessian!!(cache, rosen, z)
                @test v ≈ rosen(z)
                @test g ≈ rosen_g(z) rtol = 1e-10
                @test H ≈ rosen_H(z) rtol = 1e-10
            end

            @testset "Rosenbrock Float32" begin
                z = Float32[1.2, 1.2]
                cache = prepare_hessian_cache(rosen, z)
                v, g, H = value_gradient_and_hessian!!(cache, rosen, z)
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
                v, g, H = value_gradient_and_hessian!!(cache, f, x)
                @test v ≈ 14.0
                @test g ≈ [2.0, 4.0, 6.0]
                @test H ≈ 2 * I
            end

            @testset "cache reuse with different x" begin
                f(x) = sum(x .^ 2)
                x1 = [1.0, 0.0]
                x2 = [2.0, 3.0]
                cache = prepare_hessian_cache(f, x1)
                v1, g1, H1 = value_gradient_and_hessian!!(cache, f, x1)
                v2, g2, H2 = value_gradient_and_hessian!!(cache, f, x2)
                @test v1 ≈ 1.0
                @test v2 ≈ 13.0
                @test g1 ≈ [2.0, 0.0]
                @test g2 ≈ [4.0, 6.0]
                @test H1 ≈ H2
            end

            @testset "debug_mode=true is rejected" begin
                z = [1.2, 1.2]
                @test_throws ArgumentError prepare_hessian_cache(
                    rosen, z; config=Mooncake.Config(; debug_mode=true)
                )
            end

            @testset "n=0 edge case" begin
                f(x) = 0.0
                x = Float64[]
                cache = prepare_hessian_cache(f, x)
                v, g, H = value_gradient_and_hessian!!(cache, f, x)
                @test v == 0.0
                @test g == Float64[]
                @test H == zeros(0, 0)
            end

            @testset "n=0 edge case with cache reuse" begin
                f(x) = 0.0
                x = Float64[]
                cache = prepare_hessian_cache(f, x)
                v1, g1, H1 = value_gradient_and_hessian!!(cache, f, x)
                v2, g2, H2 = value_gradient_and_hessian!!(cache, f, x)
                @test (v1, g1, H1) == (0.0, Float64[], zeros(0, 0))
                @test (v2, g2, H2) == (0.0, Float64[], zeros(0, 0))
            end

            @testset "multi-arg: two vectors" begin
                f(x, y) = sum(x .^ 2) + sum(y .^ 2) + x[1] * y[1]
                x = [1.0, 2.0]
                y = [3.0, 4.0]
                cache = prepare_hessian_cache(f, x, y)
                val, (gx, gy), ((Hxx, Hxy), (Hyx, Hyy)) = value_gradient_and_hessian!!(
                    cache, f, x, y
                )
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
                v1, (gx1, gy1), ((Hxx1, _), (_, Hyy1)) = value_gradient_and_hessian!!(
                    cache, f, x1, y1
                )
                v2, (gx2, gy2), ((Hxx2, _), (_, Hyy2)) = value_gradient_and_hessian!!(
                    cache, f, x2, y2
                )
                @test v1 ≈ f(x1, y1)
                @test v2 ≈ f(x2, y2)
                @test gx1 ≈ 2x1
                @test gx2 ≈ 2x2
                @test Hxx1 ≈ 2 * I
                @test Hxx2 ≈ 2 * I
                @test Hyy1 ≈ 2 * I
                @test Hyy2 ≈ 2 * I
            end

            @testset "multi-arg: first arg empty" begin
                f(x, y) = sum(y .^ 2)
                x = Float64[]
                y = [1.0, 2.0]
                cache = prepare_hessian_cache(f, x, y)
                val, (gx, gy), ((Hxx, Hxy), (Hyx, Hyy)) = value_gradient_and_hessian!!(
                    cache, f, x, y
                )
                @test val ≈ f(x, y)
                @test gx == Float64[]
                @test gy ≈ 2y
                @test Hxx == zeros(0, 0)
                @test Hyy ≈ 2 * I
            end

            @testset "multi-arg: all args empty" begin
                f(x, y) = 0.0
                x = Float64[]
                y = Float64[]
                cache = prepare_hessian_cache(f, x, y)
                val, (gx, gy), ((Hxx, Hxy), (Hyx, Hyy)) = value_gradient_and_hessian!!(
                    cache, f, x, y
                )
                @test val == 0.0
                @test gx == Float64[]
                @test gy == Float64[]
                @test Hxx == zeros(0, 0)
                @test Hxy == zeros(0, 0)
                @test Hyx == zeros(0, 0)
                @test Hyy == zeros(0, 0)
            end

            @testset "reject non-vector inputs" begin
                f(x) = sum(x .^ 2)
                x = [1.0 2.0; 3.0 4.0]
                cache = prepare_hessian_cache(f, x)
                @test_throws ArgumentError value_gradient_and_hessian!!(cache, f, x)
            end

            @testset "reject non-IEEEFloat element types" begin
                f(x) = sum(abs2, x)
                x = ComplexF64[1 + 0im, 2 + 0im]
                cache = prepare_hessian_cache(f, x)
                @test_throws ArgumentError value_gradient_and_hessian!!(cache, f, x)
            end

            @testset "reject mismatched element types across arguments" begin
                f(x, y) = sum(x .^ 2) + sum(y .^ 2)
                x = Float64[1.0, 2.0]
                y = Float32[3.0, 4.0]
                cache = prepare_hessian_cache(f, x, y)
                @test_throws ArgumentError value_gradient_and_hessian!!(cache, f, x, y)
            end

            @testset "reject mismatched function object" begin
                f(x) = sum(x .^ 2)
                g(x) = sum(3 .* x .^ 2)
                x = [1.0, 2.0]
                cache = prepare_hessian_cache(f, x)
                @test_throws ArgumentError value_gradient_and_hessian!!(cache, g, x)
            end

            @testset "hessian cache mismatch errors" begin
                f(x) = sum(x .^ 2)
                x = [1.0, 2.0]
                cache = prepare_hessian_cache(f, x)
                @test_throws r"Cached autodiff call has a size mismatch for `x1`" value_gradient_and_hessian!!(
                    cache, f, [1.0, 2.0, 3.0]
                )
                @test_throws r"Cached autodiff call has a type mismatch for `x1`" value_gradient_and_hessian!!(
                    cache, f, Float32[1.0, 2.0]
                )
            end
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
