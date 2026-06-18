using Mooncake:
    prepare_gradient_cache,
    prepare_hvp_cache,
    prepare_hessian_cache,
    prepare_pullback_cache,
    value_and_gradient!!,
    value_and_hvp!!,
    value_and_jacobian!!,
    value_gradient_and_hessian!!,
    value_and_pullback!!,
    CoDual,
    TestUtils,
    build_rrule,
    tangent_type

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

mutable struct IntScaler
    a::Int
end
(s::IntScaler)(v) = s.a * sum(v)

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

const NFWD_PREPARE_COUNTER = Ref(0)
_ndual_prepare_side_effect(x) = (NFWD_PREPARE_COUNTER[] += 1; x^2 + one(x))

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
                sin,
                1.0;
                config=Mooncake.Config(;
                    debug_mode=false, friendly_tangents=true, chunk_size=2
                ),
            )
            forward_show = sprint(show, forward_cache)
            @test occursin("Mooncake.FCache(", forward_show)
            @test occursin("mode=:forward", forward_show)
            @test occursin("friendly_tangents=true", forward_show)
            # A scalar input has 1 dof, so no width-`W` chunk rule is built.
            @test occursin("chunk=false", forward_show)
            @test occursin("chunk_size=1", forward_show)

            forward_plain = repr(MIME"text/plain"(), forward_cache)
            @test occursin("Mooncake.FCache", forward_plain)
            @test occursin("mode: forward", forward_plain)
            @test occursin("friendly_tangents: true", forward_plain)
            @test occursin("chunk: false", forward_plain)
            @test occursin("chunk_size: 1", forward_plain)
            @test occursin("input_1: Float64 (scalar)", forward_plain)
            @test occursin("output: Float64 (scalar)", forward_plain)

            forward_cache_chunk2 = Mooncake.prepare_derivative_cache(
                (x, y) -> x * y + sin(x),
                1.0,
                2.0;
                config=Mooncake.Config(;
                    debug_mode=false, friendly_tangents=true, chunk_size=2
                ),
            )
            forward_chunk2_show = sprint(show, forward_cache_chunk2)
            # 2 dof at chunk_size=2 builds a width-2 native chunk rule.
            @test occursin("chunk=true", forward_chunk2_show)
            @test occursin("chunk_size=2", forward_chunk2_show)

            forward_chunk2_plain = repr(MIME"text/plain"(), forward_cache_chunk2)
            @test occursin("chunk_size: 2", forward_chunk2_plain)

            hvp_cache = Mooncake.prepare_hvp_cache(sin, 1.0)
            hvp_show = sprint(show, hvp_cache)
            @test occursin("Mooncake.HVPCache(", hvp_show)
            @test occursin("mode=:forward_over_reverse", hvp_show)
            @test occursin("chunk=", hvp_show)

            hvp_plain = repr(MIME"text/plain"(), hvp_cache)
            @test occursin("Mooncake.HVPCache", hvp_plain)
            @test occursin("mode: forward_over_reverse", hvp_plain)
            @test occursin("chunk: ", hvp_plain)
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

            # tuple interface
            z_and_dz_tup = Mooncake.value_and_derivative!!(cache, zip(fx, dfx)...)
            @test z_and_dz_tup isa Tuple{Float64,Float64}
            @test first(z_and_dz_tup) == z
            @test last(z_and_dz_tup) == dz

            # multi-argument single-direction tuple interface
            z_and_dz_multi = Mooncake.value_and_derivative!!(
                cache, (f, Mooncake.zero_tangent(f)), (x, dx), (y, dy)
            )
            @test z_and_dz_multi isa Tuple{Float64,Float64}
            @test first(z_and_dz_multi) == z
            @test last(z_and_dz_multi) == dz
        end

        @testset "Array inputs" begin
            f_arr = x -> sum(abs2, x)
            x_arr = [x, y]
            dir = [dx, dy]

            cache_arr = Mooncake.prepare_derivative_cache(
                f_arr, x_arr; config=Mooncake.Config(; kwargs...)
            )
            z_and_dz_arr = Mooncake.value_and_derivative!!(
                cache_arr, (f_arr, Mooncake.zero_tangent(f_arr)), (x_arr, dir)
            )
            @test first(z_and_dz_arr) == sum(abs2, x_arr)
            # directional derivative of sum(abs2, x) is 2x ⋅ dir
            @test last(z_and_dz_arr) == 2 * x * dx + 2 * y * dy
        end

        @testset "Non-differentiable outputs" begin
            f_int = x -> x > 0 ? 1 : 2
            cache_int = Mooncake.prepare_derivative_cache(
                f_int, x; config=Mooncake.Config(; kwargs...)
            )
            z_and_dz_int = Mooncake.value_and_derivative!!(
                cache_int, (f_int, Mooncake.zero_tangent(f_int)), (x, dx)
            )
            @test first(z_and_dz_int) == 1
            @test last(z_and_dz_int) == Mooncake.NoTangent()
        end

        @testset "Structured types" begin
            cache_sp_friendly = Mooncake.prepare_derivative_cache(
                fx_sp...; config=Mooncake.Config(; friendly_tangents=true, kwargs...)
            )
            # friendly input doesn't error
            z_and_dz_sp = Mooncake.value_and_derivative!!(
                cache_sp_friendly, zip(fx_sp, dfx_sp)...
            )
            # primal output is friendly; tangent is a NamedTuple of per-field gradients.
            @test first(z_and_dz_sp) == SimplePair(z, 2.0)
            dz_sp = last(z_and_dz_sp)
            @test dz_sp.x1 ≈ dz
            @test dz_sp.x2 == 0.0

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

        @testset "Tuple-like inputs" begin
            f_tuple = t -> t[1]^2 + sin(t[2])
            tuple_x = (x, y)
            cache_tuple = Mooncake.prepare_derivative_cache(
                f_tuple,
                tuple_x;
                config=Mooncake.Config(; friendly_tangents=true, kwargs...),
            )
            z_and_dz_tuple = Mooncake.value_and_derivative!!(
                cache_tuple, (f_tuple, Mooncake.zero_tangent(f_tuple)), (tuple_x, (dx, dy))
            )
            @test first(z_and_dz_tuple) == x^2 + sin(y)
            @test last(z_and_dz_tuple) == 2 * x * dx + cos(y) * dy

            f_named = nt -> nt.a * sin(nt.b)
            named_x = (; a=x, b=y)
            cache_named = Mooncake.prepare_derivative_cache(
                f_named,
                named_x;
                config=Mooncake.Config(; friendly_tangents=true, kwargs...),
            )
            z_and_dz_named = Mooncake.value_and_derivative!!(
                cache_named,
                (f_named, Mooncake.zero_tangent(f_named)),
                (named_x, (; a=dx, b=dy)),
            )
            @test first(z_and_dz_named) == x * sin(y)
            @test last(z_and_dz_named) == dx * sin(y) + x * cos(y) * dy
        end

        @testset "value_and_gradient!! via FCache" begin
            cache_grad_fwd = Mooncake.prepare_derivative_cache(
                f, x, y; config=Mooncake.Config(; kwargs...)
            )
            @test Mooncake.value_and_gradient!!(cache_grad_fwd, f, x, y) ==
                (z, (Mooncake.NoTangent(), y - sin(x), x))

            f_scalar = x -> x^2 + sin(x)
            scalar_cache_grad_fwd = Mooncake.prepare_derivative_cache(
                f_scalar, x; config=Mooncake.Config(; kwargs...)
            )
            @test Mooncake.value_and_gradient!!(scalar_cache_grad_fwd, f_scalar, x) ==
                (f_scalar(x), (Mooncake.NoTangent(), 2 * x + cos(x)))

            f_tuple = t -> t[1]^2 + sin(t[2])
            tuple_x = (x, y)
            tuple_cache_grad_fwd = Mooncake.prepare_derivative_cache(
                f_tuple, tuple_x; config=Mooncake.Config(; kwargs...)
            )
            @test Mooncake.value_and_gradient!!(tuple_cache_grad_fwd, f_tuple, tuple_x) ==
                (x^2 + sin(y), (Mooncake.NoTangent(), (2 * x, cos(y))))

            # A differentiable `Ref` within a multi-dof gradient input forces the chunked
            # `basis_lifted!!` seeding path (2 dofs at chunk_size=2); `_basis_seed!!` had no
            # `NDualRef` method (#4), so this threw a MethodError. Forward must match the reverse
            # oracle (the `Ref`'s cotangent is a `MutableTangent`).
            g_ref = t -> t[1][]^2 + sin(t[2])
            ref_fwd = Mooncake.prepare_derivative_cache(
                g_ref, (Ref(x), y); config=Mooncake.Config(; chunk_size=2, kwargs...)
            )
            ref_rev = Mooncake.prepare_gradient_cache(g_ref, (Ref(x), y))
            yf_ref, gf_ref = Mooncake.value_and_gradient!!(ref_fwd, g_ref, (Ref(x), y))
            yr_ref, gr_ref = Mooncake.value_and_gradient!!(ref_rev, g_ref, (Ref(x), y))
            @test yf_ref == yr_ref
            @test TestUtils.has_equal_data(gf_ref, gr_ref)

            # Complex `Ref` exercises the distinct complex `NDualRef` `_basis_seed!!` (two cursor
            # steps per dof: real then imag).
            g_cref = t -> abs2(t[1][]) + sin(t[2])
            cref0 = ComplexF64(x, y)
            cref_fwd = Mooncake.prepare_derivative_cache(
                g_cref, (Ref(cref0), y); config=Mooncake.Config(; chunk_size=2, kwargs...)
            )
            cref_rev = Mooncake.prepare_gradient_cache(g_cref, (Ref(cref0), y))
            yf_cref, gf_cref = Mooncake.value_and_gradient!!(
                cref_fwd, g_cref, (Ref(cref0), y)
            )
            yr_cref, gr_cref = Mooncake.value_and_gradient!!(
                cref_rev, g_cref, (Ref(cref0), y)
            )
            @test yf_cref == yr_cref
            @test TestUtils.has_equal_data(gf_cref, gr_cref)

            h = (sp::SimplePair) -> sp.x1^2 + sin(sp.x2)
            sp = SimplePair(x, y)
            cache_sp_fwd_friendly = Mooncake.prepare_derivative_cache(
                h, sp; config=Mooncake.Config(; friendly_tangents=true, kwargs...)
            )
            @test Mooncake.value_and_gradient!!(cache_sp_fwd_friendly, h, sp) ==
                (h(sp), (h, SimplePair(2 * x, cos(y))))

            f_vec = x -> (x, 2x)
            cache_vec_fwd = Mooncake.prepare_derivative_cache(
                f_vec, x; config=Mooncake.Config(; kwargs...)
            )
            @test_throws Mooncake.ValueAndGradientReturnTypeError Mooncake.value_and_gradient!!(
                cache_vec_fwd, f_vec, x
            )

            alias_f = ap -> sum(ap.a) + sum(ap.b)
            shared = [x, y]
            alias_pair = AliasedPair(shared, shared)
            alias_cache = Mooncake.prepare_derivative_cache(
                alias_f,
                alias_pair;
                config=Mooncake.Config(; friendly_tangents=false, kwargs...),
            )
            alias_val, alias_grad = Mooncake.value_and_gradient!!(
                alias_cache, alias_f, alias_pair
            )
            alias_pair_grad = alias_grad[2]
            alias_a_grad = Mooncake.get_tangent_field(alias_pair_grad, :a)
            alias_b_grad = Mooncake.get_tangent_field(alias_pair_grad, :b)
            @test alias_val == 2 * sum(shared)
            @test alias_a_grad === alias_b_grad
            @test alias_a_grad == fill(2.0, length(shared))

            cycle_f = node -> node.weight + node.next.weight
            cycle_node = AnyCycleNode(nothing, x)
            cycle_node.next = cycle_node
            cycle_cache = Mooncake.prepare_derivative_cache(
                cycle_f,
                cycle_node;
                config=Mooncake.Config(; friendly_tangents=false, kwargs...),
            )
            cycle_val, cycle_grad = Mooncake.value_and_gradient!!(
                cycle_cache, cycle_f, cycle_node
            )
            cycle_node_grad = cycle_grad[2]
            @test cycle_val == 2 * x
            @test Mooncake.get_tangent_field(cycle_node_grad, :next) === cycle_node_grad
            @test Mooncake.get_tangent_field(cycle_node_grad, :weight) == 2 * one(x)

            uninit_f = box -> box.x^2
            uninit_box = MaybeInitBox(x)
            uninit_cache = Mooncake.prepare_derivative_cache(
                uninit_f,
                uninit_box;
                config=Mooncake.Config(; friendly_tangents=false, kwargs...),
            )
            uninit_val, uninit_grad = Mooncake.value_and_gradient!!(
                uninit_cache, uninit_f, uninit_box
            )
            uninit_box_grad = uninit_grad[2]
            uninit_y_grad = getfield(uninit_box_grad.fields, :y)
            @test uninit_val == x^2
            @test Mooncake.get_tangent_field(uninit_box_grad, :x) == 2 * x
            @test !Mooncake.is_init(uninit_y_grad) || Mooncake.val(uninit_y_grad) == 0.0

            # The packable path must evaluate the CALL-time `f`, not the prepare-time
            # instance captured in the seed (regression: a value-stateful non-diff
            # callable silently used stale state).
            sc_cache = Mooncake.prepare_derivative_cache(
                IntScaler(1), collect(1.0:4.0); config=Mooncake.Config(; kwargs...)
            )
            sc_val, sc_grad = Mooncake.value_and_gradient!!(
                sc_cache, IntScaler(2), collect(1.0:4.0)
            )
            @test sc_val == 20.0
            @test sc_grad[2] == fill(2.0, 4)

            # An in-place-mutating `f` must not compound across packable chunks: the seed
            # primals are restored from the user's arrays at the top of every chunk
            # (regression: y and the chunk-2 gradient slots were silently wrong).
            mut_f = v -> (s=sum(abs2, v); v .*= 2; s)
            mut_x = collect(1.0:12.0)  # dof > default chunk width 8 → two chunks
            mut_cache = Mooncake.prepare_derivative_cache(
                mut_f, copy(mut_x); config=Mooncake.Config(; kwargs...)
            )
            mut_y, mut_grad = Mooncake.value_and_gradient!!(mut_cache, mut_f, copy(mut_x))
            @test mut_y == sum(abs2, mut_x)
            @test mut_grad[2] == 2 .* mut_x

            # A differentiable closure `f` takes the generic path on a scalar input: the
            # width-1 fast path cannot represent `f`'s own dofs (regression: it hard-coded
            # NoTangent for `f` and seeded uninitialised tangent storage).
            closure_f = let c = 3.0
                v -> c * v
            end
            closure_cache = Mooncake.prepare_derivative_cache(
                closure_f, x; config=Mooncake.Config(; kwargs...)
            )
            closure_y, closure_grad = Mooncake.value_and_gradient!!(
                closure_cache, closure_f, x
            )
            @test closure_y == 3.0 * x
            @test closure_grad[2] == 3.0
            @test Mooncake.get_tangent_field(closure_grad[1], :c) == x

            # Width-N Lifted inputs against a cache without a chunk rule (scalar dof → no
            # chunk built), and against a chunk rule of a different width, must raise a
            # clear PreparedCacheError, not a MethodError/typeassert.
            sq = z -> z^2
            scalar_cache = Mooncake.prepare_derivative_cache(
                sq, 1.5; config=Mooncake.Config(; kwargs...)
            )
            w3 = Mooncake.Lifted{Float64,3}(
                1.5, Mooncake.Nfwd.NDual{Float64,3}(1.5, (1.0, 0.0, 0.0))
            )
            @test_throws Mooncake.PreparedCacheError Mooncake.value_and_derivative!!(
                scalar_cache, Mooncake.zero_lifted(Val(3), sq), w3
            )
            wide_cache = Mooncake.prepare_derivative_cache(
                mut_f, collect(1.0:12.0); config=Mooncake.Config(; kwargs...)
            )
            @test_throws Mooncake.PreparedCacheError Mooncake.value_and_derivative!!(
                wide_cache,
                Mooncake.zero_lifted(Val(3), mut_f),
                Mooncake.randn_lifted(Val(3), Xoshiro(1), collect(1.0:12.0)),
            )
            # Mixed-width slots whose FIRST slot matches the cache's chunk width must still raise a
            # clear PreparedCacheError (every slot must share the width; checking only `first`
            # would let a trailing differently-sized slot reach the chunk rule's OC as a typeassert).
            chunk2_cache = Mooncake.prepare_derivative_cache(
                f, x, y; config=Mooncake.Config(; chunk_size=2, kwargs...)
            )
            @test_throws Mooncake.PreparedCacheError Mooncake.value_and_derivative!!(
                chunk2_cache,
                Mooncake.zero_lifted(Val(2), f),
                Mooncake.zero_lifted(Val(2), x),
                Mooncake.zero_lifted(Val(3), y),
            )

            # Derived vararg rules at chunk width > 1 exercise `__unflatten_dual_varargs`' width-W
            # group assembly (`Lifted{GP,W}(group_primal, group_v)`), which `test_frule` skips
            # (derived rules run width 1 only) and other chunked tests miss (all fixed-arity). A W=1
            # regression in that path throws a typeassert; cover it at chunk_size=2.
            vararg_f = (a, bs...) -> a + sum(bs)
            vararg_cache = Mooncake.prepare_derivative_cache(
                vararg_f, x, y, 3.0; config=Mooncake.Config(; chunk_size=2, kwargs...)
            )
            @test Mooncake.value_and_gradient!!(vararg_cache, vararg_f, x, y, 3.0) ==
                (x + y + 3.0, (Mooncake.NoTangent(), 1.0, 1.0, 1.0))
            # All-non-differentiable vararg group: `dual_type(Val(2), Tuple{Int,Int})` is `NoDual`,
            # exercising the `group_v === NoDual ? NoDual()` collapse branch at width 2.
            vararg_nd = (a, ns::Vararg{Int}) -> a + sum(ns)
            vararg_nd_cache = Mooncake.prepare_derivative_cache(
                vararg_nd, x, 2, 3; config=Mooncake.Config(; chunk_size=2, kwargs...)
            )
            @test Mooncake.value_and_gradient!!(vararg_nd_cache, vararg_nd, x, 2, 3) == (
                x + 5,
                (Mooncake.NoTangent(), 1.0, Mooncake.NoTangent(), Mooncake.NoTangent()),
            )

            f32_scalar = x -> Float32(x^2 + sin(x))
            x32 = Float32(x)
            f32_scalar_cache = Mooncake.prepare_derivative_cache(
                f32_scalar, x32; config=Mooncake.Config(; kwargs...)
            )
            @test Mooncake.value_and_gradient!!(f32_scalar_cache, f32_scalar, x32) ==
                (f32_scalar(x32), (Mooncake.NoTangent(), Float32(2x32 + cos(x32))))

            f32_vec = x -> Float32(sum(abs2, x))
            x32_vec = Float32[x, y]
            f32_vec_cache = Mooncake.prepare_derivative_cache(
                f32_vec, x32_vec; config=Mooncake.Config(; kwargs...)
            )
            @test Mooncake.value_and_gradient!!(f32_vec_cache, f32_vec, x32_vec) ==
                (f32_vec(x32_vec), (Mooncake.NoTangent(), Float32.(2 .* x32_vec)))

            f32_tuple = t -> Float32(t[1]^2 + sin(t[2]))
            tuple_x32 = (Float32(x), Float32(y))
            f32_tuple_cache = Mooncake.prepare_derivative_cache(
                f32_tuple, tuple_x32; config=Mooncake.Config(; kwargs...)
            )
            @test Mooncake.value_and_gradient!!(f32_tuple_cache, f32_tuple, tuple_x32) == (
                f32_tuple(tuple_x32),
                (
                    Mooncake.NoTangent(),
                    (Float32(2 * tuple_x32[1]), Float32(cos(tuple_x32[2]))),
                ),
            )

            # A view input cannot use the flat packable seed (`similar(::SubArray)` is a plain
            # `Vector`, mismatching the view type the rule and cache spec expect). It must fall
            # through to the structured path and return the structural (parent-field) gradient,
            # matching reverse mode (regression: the flat seed threw a PreparedCacheError, and the
            # AbstractVector fast method then mis-dispatched the StructuredGradSeed).
            view_f = v -> sum(abs2, v)
            view_x = view(collect(1.0:6.0), 1:3)
            view_cache = Mooncake.prepare_derivative_cache(
                view_f, view_x; config=Mooncake.Config(; kwargs...)
            )
            view_val, view_grad = Mooncake.value_and_gradient!!(view_cache, view_f, view_x)
            @test view_val == sum(abs2, view_x)
            @test Mooncake.get_tangent_field(view_grad[2], :parent) ==
                vcat(2 .* collect(1.0:3.0), zeros(3))

            # A structured input whose NESTED array is reused at the same length but a different
            # shape must be rejected (size, not just length, is validated) instead of silently
            # computing on the stale cache-owned shape (regression: returned the wrong primal).
            nested_f = t -> sum(t[1] * permutedims(t[1]))
            nested_cache = Mooncake.prepare_derivative_cache(
                nested_f,
                (reshape(collect(1.0:6.0), 2, 3),);
                config=Mooncake.Config(; kwargs...),
            )
            @test_throws Mooncake.PreparedCacheError Mooncake.value_and_gradient!!(
                nested_cache, nested_f, (reshape(collect(1.0:6.0), 3, 2),)
            )
            nested_A2 = reshape(collect(7.0:12.0), 2, 3)
            @test first(
                Mooncake.value_and_gradient!!(nested_cache, nested_f, (nested_A2,))
            ) == nested_f((nested_A2,))

            if get(kwargs, :debug_mode, false)
                @test true
            else
                scalar_allocs = TestUtils.count_allocs(
                    Mooncake.value_and_gradient!!, scalar_cache_grad_fwd, f_scalar, x
                )
                @test scalar_allocs == 0

                scalar_f = CountedChunkScalarCall()
                scalar_cache_grad_fwd = Mooncake.prepare_derivative_cache(
                    scalar_f,
                    x,
                    y;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
                )
                CHUNK_SCALAR_EVAL_COUNT[] = 0
                @test Mooncake.value_and_gradient!!(
                    scalar_cache_grad_fwd, scalar_f, x, y
                ) == (z, (Mooncake.NoTangent(), y - sin(x), x))
                @test CHUNK_SCALAR_EVAL_COUNT[] == 1

                scalar_cache_grad_fwd_chunked = Mooncake.prepare_derivative_cache(
                    scalar_f,
                    x,
                    y;
                    config=Mooncake.Config(;
                        debug_mode=false, friendly_tangents=false, chunk_size=1
                    ),
                )
                CHUNK_SCALAR_EVAL_COUNT[] = 0
                @test Mooncake.value_and_gradient!!(
                    scalar_cache_grad_fwd_chunked, scalar_f, x, y
                ) == (z, (Mooncake.NoTangent(), y - sin(x), x))
                @test CHUNK_SCALAR_EVAL_COUNT[] == 2

                array_f = CountedChunkArrayCall()
                x_arr = [x, y]
                array_cache_grad_fwd = Mooncake.prepare_derivative_cache(
                    array_f,
                    x_arr;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
                )
                CHUNK_ARRAY_EVAL_COUNT[] = 0
                @test Mooncake.value_and_gradient!!(array_cache_grad_fwd, array_f, x_arr) ==
                    (sum(abs2, x_arr), (Mooncake.NoTangent(), 2 .* x_arr))
                @test CHUNK_ARRAY_EVAL_COUNT[] == 1
                @test TestUtils.count_allocs(
                    Mooncake.value_and_gradient!!, array_cache_grad_fwd, array_f, x_arr
                ) == 0

                array_cache_grad_fwd_chunked = Mooncake.prepare_derivative_cache(
                    array_f,
                    x_arr;
                    config=Mooncake.Config(;
                        debug_mode=false, friendly_tangents=false, chunk_size=1
                    ),
                )
                CHUNK_ARRAY_EVAL_COUNT[] = 0
                @test Mooncake.value_and_gradient!!(
                    array_cache_grad_fwd_chunked, array_f, x_arr
                ) == (sum(abs2, x_arr), (Mooncake.NoTangent(), 2 .* x_arr))
                @test CHUNK_ARRAY_EVAL_COUNT[] == 2

                singleton_x_arr = [x]
                singleton_array_cache_grad_fwd = Mooncake.prepare_derivative_cache(
                    array_f,
                    singleton_x_arr;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
                )
                CHUNK_ARRAY_EVAL_COUNT[] = 0
                @test Mooncake.value_and_gradient!!(
                    singleton_array_cache_grad_fwd, array_f, singleton_x_arr
                ) == (
                    sum(abs2, singleton_x_arr), (Mooncake.NoTangent(), 2 .* singleton_x_arr)
                )
                @test CHUNK_ARRAY_EVAL_COUNT[] == 1
                @test TestUtils.count_allocs(
                    Mooncake.value_and_gradient!!,
                    singleton_array_cache_grad_fwd,
                    array_f,
                    singleton_x_arr,
                ) == 0

                singleton_array_cache_grad_fwd_friendly = Mooncake.prepare_derivative_cache(
                    array_f,
                    singleton_x_arr;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=true),
                )
                CHUNK_ARRAY_EVAL_COUNT[] = 0
                @test Mooncake.value_and_gradient!!(
                    singleton_array_cache_grad_fwd_friendly, array_f, singleton_x_arr
                ) == (sum(abs2, singleton_x_arr), (array_f, 2 .* singleton_x_arr))
                @test CHUNK_ARRAY_EVAL_COUNT[] == 1

                # Regression: _validate_prepared_cache must not allocate.
                # length-5 vector: a single full-width (chunk_size=5) native chunk pass.
                x5 = collect(1.0:5.0)
                f5 = x -> sum(abs2, x)
                cache_5 = Mooncake.prepare_derivative_cache(
                    f5,
                    x5;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
                )
                @test Mooncake.value_and_gradient!!(cache_5, f5, x5) ==
                    (sum(abs2, x5), (Mooncake.NoTangent(), 2 .* x5))
                @test TestUtils.count_allocs(
                    Mooncake.value_and_gradient!!, cache_5, f5, x5
                ) == 0

                # length-10 vector: DOF > max chunk width (8), so two chunks (8 + 2).
                x10 = collect(1.0:10.0)
                f10 = x -> sum(abs2, x)
                cache_10 = Mooncake.prepare_derivative_cache(
                    f10,
                    x10;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
                )
                @test Mooncake.value_and_gradient!!(cache_10, f10, x10) ==
                    (sum(abs2, x10), (Mooncake.NoTangent(), 2 .* x10))
                @test TestUtils.count_allocs(
                    Mooncake.value_and_gradient!!, cache_10, f10, x10
                ) == 0

                # Non-packable inputs (here a NamedTuple) also chunk through the generic
                # chunked gradient path: multi-dof builds a native chunk rule and the
                # gradient is correct. (Such inputs were previously pinned to width 1.)
                nt_x = (; a=1.3, b=2.1, c=0.7)
                f_nt = nt -> nt.a^2 * nt.b + sin(nt.a) * nt.c
                cache_nt = Mooncake.prepare_derivative_cache(
                    f_nt, nt_x; config=Mooncake.Config(; friendly_tangents=true)
                )
                @test getfield(cache_nt, :gradient_chunk_size) > 1
                @test getfield(cache_nt, :chunk_rule) !== nothing
                y_nt, g_nt = Mooncake.value_and_gradient!!(cache_nt, f_nt, nt_x)
                @test y_nt == f_nt(nt_x)
                @test g_nt[2].a ≈ 2 * nt_x.a * nt_x.b + cos(nt_x.a) * nt_x.c
                @test g_nt[2].b ≈ nt_x.a^2
                @test g_nt[2].c ≈ sin(nt_x.a)

                # Array-backed structured inputs take the zero-allocation leaf-table path
                # (StructuredGradSeed): tuple/Matrix of float arrays — correct + zero-alloc.
                ft = t -> sum(abs2, t[1]) + sum(abs2, t[2])
                tx = ([1.0, 2.0, 3.0], [4.0, 5.0])
                ct = Mooncake.prepare_derivative_cache(
                    ft, tx; config=Mooncake.Config(; friendly_tangents=false)
                )
                @test getfield(ct, :gradient_seed) isa Mooncake.StructuredGradSeed
                yt, gt = Mooncake.value_and_gradient!!(ct, ft, tx)
                @test yt == ft(tx)
                @test gt[2][1] ≈ 2 .* tx[1]
                @test gt[2][2] ≈ 2 .* tx[2]
                @test TestUtils.count_allocs(Mooncake.value_and_gradient!!, ct, ft, tx) == 0

                fA = A -> sum(abs2, A)
                Ax = [1.0 2.0; 3.0 4.0]
                cA = Mooncake.prepare_derivative_cache(
                    fA, Ax; config=Mooncake.Config(; friendly_tangents=false)
                )
                @test getfield(cA, :gradient_seed) isa Mooncake.StructuredGradSeed
                @test TestUtils.count_allocs(Mooncake.value_and_gradient!!, cA, fA, Ax) == 0
                _, gA = Mooncake.value_and_gradient!!(cA, fA, Ax)
                @test gA[2] ≈ 2 .* Ax

                # Primal refresh: prepare at one point, evaluate at another.
                cr = Mooncake.prepare_derivative_cache(
                    ft,
                    ([1.0, 1.0, 1.0], [1.0, 1.0]);
                    config=Mooncake.Config(; friendly_tangents=false),
                )
                tx2 = ([2.0, 3.0, 4.0], [5.0, 6.0])
                yr, gr = Mooncake.value_and_gradient!!(cr, ft, tx2)
                @test yr == ft(tx2)
                @test gr[2][1] ≈ 2 .* tx2[1]
                @test gr[2][2] ≈ 2 .* tx2[2]

                # In-place-mutating `f` whose array spans >1 chunk: the seed primal must be
                # restored (and partials re-zeroed) every chunk, else a later chunk runs on an
                # earlier chunk's mutated primal. dof 10 > max chunk width forces two chunks.
                fip = t -> begin
                    t[1] .= t[1] .* 2.0
                    sum(abs2, t[1])
                end
                tip0 = (collect(1.0:10.0),)
                cip = Mooncake.prepare_derivative_cache(
                    fip, tip0; config=Mooncake.Config(; friendly_tangents=false)
                )
                tip = (collect(1.0:10.0),)
                _, gip = Mooncake.value_and_gradient!!(cip, fip, tip)
                @test gip[2][1] ≈ 8 .* collect(1.0:10.0)   # d/dt Σ(2t)² = 8t, across both chunks
                @test tip == (collect(1.0:10.0),)          # user input not mutated

                # The seed must not alias the user's prepare-time arrays: prepare AND evaluate at
                # the SAME object with an in-place `f` — the input must be left unchanged.
                fsame = t -> begin
                    t[1] .= t[1] .* 2.0
                    sum(abs2, t[1]) + sum(abs2, t[2])
                end
                tsame = ([1.0, 2.0, 3.0], [4.0, 5.0])
                csame = Mooncake.prepare_derivative_cache(
                    fsame, tsame; config=Mooncake.Config(; friendly_tangents=false)
                )
                _, gsame = Mooncake.value_and_gradient!!(csame, fsame, tsame)  # same object
                @test tsame == ([1.0, 2.0, 3.0], [4.0, 5.0])   # user input not clobbered
                @test gsame[2][1] ≈ 8 .* [1.0, 2.0, 3.0]
                @test gsame[2][2] ≈ 2 .* [4.0, 5.0]

                # Zero-dof input (no float dofs) with an in-place `f`: the total_dof==0 generic
                # branch must also snapshot/restore the user's input.
                fz0 = x -> (x[1] += 1; 2.5)
                xz0 = [10, 20, 30]
                cz0 = Mooncake.prepare_derivative_cache(
                    fz0, xz0; config=Mooncake.Config(; friendly_tangents=false)
                )
                yz0, _ = Mooncake.value_and_gradient!!(cz0, fz0, xz0)
                @test yz0 == 2.5
                @test xz0 == [10, 20, 30]                      # user input not mutated

                # Mixed array + scalar input has a non-array dof, so the gather bails and the
                # generic chunked path runs — still correct.
                fmix = nt -> sum(nt.v) + nt.s^2
                mx = (; v=[1.0, 2.0], s=3.0)
                cmix = Mooncake.prepare_derivative_cache(
                    fmix, mx; config=Mooncake.Config(; friendly_tangents=true)
                )
                @test !(getfield(cmix, :gradient_seed) isa Mooncake.StructuredGradSeed)
                _, gmix = Mooncake.value_and_gradient!!(cmix, fmix, mx)
                @test gmix[2].v ≈ ones(2)
                @test gmix[2].s ≈ 2 * mx.s

                # Scalar-only structured inputs (isbits V) take the concrete-barrier path
                # (IsbitsGradSeed): tuple/NamedTuple/immutable-struct of scalars — correct +
                # zero-alloc. (Previously the generic chunked path, ~52 allocations.)
                fnt = nt -> nt.a^2 * nt.b + sin(nt.a) * nt.c
                ntx = (; a=1.3, b=2.1, c=0.7)
                cnt = Mooncake.prepare_derivative_cache(
                    fnt, ntx; config=Mooncake.Config(; friendly_tangents=false)
                )
                @test getfield(cnt, :gradient_seed) isa Mooncake.IsbitsGradSeed
                ynt, gnt = Mooncake.value_and_gradient!!(cnt, fnt, ntx)
                @test ynt == fnt(ntx)
                @test gnt[2].a ≈ 2 * ntx.a * ntx.b + cos(ntx.a) * ntx.c
                @test gnt[2].b ≈ ntx.a^2
                @test gnt[2].c ≈ sin(ntx.a)
                @test TestUtils.count_allocs(
                    Mooncake.value_and_gradient!!, cnt, fnt, ntx
                ) == 0

                # immutable struct of scalars: native gradient is a `Tangent` (scattered via the
                # `Tangent` branch), and prepare-at-x0/evaluate-at-x1 (primal refresh) is correct.
                fsp = p -> p.x1^2 * p.x2
                csp = Mooncake.prepare_derivative_cache(
                    fsp,
                    SimplePair(1.0, 1.0);
                    config=Mooncake.Config(; friendly_tangents=false),
                )
                @test getfield(csp, :gradient_seed) isa Mooncake.IsbitsGradSeed
                ysp, gsp = Mooncake.value_and_gradient!!(csp, fsp, SimplePair(3.0, 4.0))
                @test ysp == fsp(SimplePair(3.0, 4.0))
                @test gsp[2].fields.x1 ≈ 2 * 3.0 * 4.0
                @test gsp[2].fields.x2 ≈ 3.0^2

                # Multi-chunk scalar input (dof 10 > max chunk width): correct + zero-alloc.
                nt10 = NamedTuple{Tuple(Symbol.("x", 1:10))}(ntuple(Float64, 10))
                f10 = nt -> sum(abs2, values(nt))
                c10 = Mooncake.prepare_derivative_cache(
                    f10, nt10; config=Mooncake.Config(; friendly_tangents=false)
                )
                @test getfield(c10, :gradient_chunk_size) < 10
                _, g10 = Mooncake.value_and_gradient!!(c10, f10, nt10)
                @test g10[2].x1 ≈ 2.0
                @test g10[2].x10 ≈ 20.0
                @test TestUtils.count_allocs(
                    Mooncake.value_and_gradient!!, c10, f10, nt10
                ) == 0

                # Complex scalar dofs have an isbits V but two dofs per element, which the isbits
                # barrier's scatter cannot handle — they must take the generic path, not crash.
                fz = z -> abs2(z)
                cz = Mooncake.prepare_derivative_cache(
                    fz, 1.0 + 2.0im; config=Mooncake.Config(; friendly_tangents=false)
                )
                @test !(getfield(cz, :gradient_seed) isa Mooncake.IsbitsGradSeed)
                yz, gz = Mooncake.value_and_gradient!!(cz, fz, 1.0 + 2.0im)
                @test yz == abs2(1.0 + 2.0im)
                @test gz[2] ≈ 2.0 + 4.0im
                fzt = t -> abs2(t[1]) + t[2]^2
                czt = Mooncake.prepare_derivative_cache(
                    fzt,
                    (1.0 + 2.0im, 3.0);
                    config=Mooncake.Config(; friendly_tangents=false),
                )
                @test !(getfield(czt, :gradient_seed) isa Mooncake.IsbitsGradSeed)
                _, gzt = Mooncake.value_and_gradient!!(czt, fzt, (1.0 + 2.0im, 3.0))
                @test gzt[2] == (2.0 + 4.0im, 6.0)

                # A non-isbits `f` (closure capturing a Vector) over scalar args must NOT take the
                # isbits barrier (its per-chunk seed rebuild would allocate) — generic path instead.
                clo = let k = [10.0]
                    x -> k[1] * x.a + x.b^2
                end
                cclo = Mooncake.prepare_derivative_cache(
                    clo, (; a=1.0, b=2.0); config=Mooncake.Config(; friendly_tangents=false)
                )
                @test !(getfield(cclo, :gradient_seed) isa Mooncake.IsbitsGradSeed)
                _, gclo = Mooncake.value_and_gradient!!(cclo, clo, (; a=1.0, b=2.0))
                @test gclo[2].a ≈ 10.0
                @test gclo[2].b ≈ 4.0
            end
        end

        @testset "forward cache mismatch errors" begin
            f_arr = x -> sum(abs2, x)
            x_arr = [x, y]
            dx_arr = [dx, 0.0]
            cache = Mooncake.prepare_derivative_cache(
                f_arr, x_arr; config=Mooncake.Config(; kwargs...)
            )

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

        @testset "reverse cache mismatch errors" begin
            f_arr = x -> sum(abs2, x)
            x_arr = [x, y]
            cache = Mooncake.prepare_gradient_cache(
                f_arr, x_arr; config=Mooncake.Config(; kwargs...)
            )

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

        @testset "prepare_derivative_cache chunk_size config" begin
            @test_throws ArgumentError Mooncake.prepare_derivative_cache(
                sin, x; config=Mooncake.Config(; chunk_size=0)
            )
            @test_throws ArgumentError Mooncake.prepare_derivative_cache(
                sin, x; config=Mooncake.Config(; chunk_size=-1)
            )
        end

        @testset "native chunk cache" begin
            # A multi-dof signature builds a native width-`W` chunk frule on the cache.
            cache_supported = Mooncake.prepare_derivative_cache(
                (a, b) -> a * b + sin(a),
                x,
                y;
                config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
            )
            @test !isnothing(getfield(cache_supported, :chunk_rule))

            # One width-2 native chunk pass covers both directions, so the primal runs once.
            @testset "$(label)" for (label, f, args, counter) in (
                ("scalar", CountedChunkScalarCall(), (x, y), CHUNK_SCALAR_EVAL_COUNT),
                ("array", CountedChunkArrayCall(), ([x, y],), CHUNK_ARRAY_EVAL_COUNT),
            )
                cache = Mooncake.prepare_derivative_cache(
                    f,
                    args...;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
                )
                counter[] = 0
                Mooncake.value_and_gradient!!(cache, f, args...)
                @test counter[] == 1
            end
        end

        @testset "value_and_jacobian!!" begin
            f_jac = x -> [x[1]^2 + x[2], x[1] * x[2], sin(x[2])]
            x_jac = [x, y]
            expected_jac = [2x 1.0; y x; 0.0 cos(y)]

            for prepare_cache in
                (Mooncake.prepare_derivative_cache, Mooncake.prepare_pullback_cache)
                cache_jac = prepare_cache(
                    f_jac,
                    x_jac;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
                )
                val_jac, jac = Mooncake.value_and_jacobian!!(cache_jac, f_jac, x_jac)
                @test val_jac == f_jac(x_jac)
                @test jac ≈ expected_jac

                x_jac2 = [x + 1, y - 1]
                expected_jac2 = [2x_jac2[1] 1.0; x_jac2[2] x_jac2[1]; 0.0 cos(x_jac2[2])]
                @test Mooncake.value_and_jacobian!!(cache_jac, f_jac, x_jac2) ==
                    (f_jac(x_jac2), expected_jac2)
            end

            # Allocation regression: with an allocation-free primal the packable forward path
            # reuses the cached seed and Jacobian buffer and must not allocate, matching the
            # zero-allocation `value_and_gradient!!`. Covers width-1 and a chunked width.
            for cs in (1, 2)
                af_cache = Mooncake.prepare_derivative_cache(
                    identity, x_jac; config=Mooncake.Config(; chunk_size=cs)
                )
                Mooncake.value_and_jacobian!!(af_cache, identity, x_jac)  # warm up / size buffer
                @test TestUtils.count_allocs(
                    Mooncake.value_and_jacobian!!, af_cache, identity, x_jac
                ) == 0
            end

            scalar_out_fwd_cache = Mooncake.prepare_derivative_cache(
                sum,
                x_jac;
                config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
            )
            @test_throws "value_and_jacobian!! only supports AbstractVector outputs" Mooncake.value_and_jacobian!!(
                scalar_out_fwd_cache, sum, x_jac
            )

            scalar_out_rev_cache = Mooncake.prepare_pullback_cache(
                sum,
                x_jac;
                config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
            )
            @test_throws "value_and_jacobian!! only supports AbstractVector outputs" Mooncake.value_and_jacobian!!(
                scalar_out_rev_cache, sum, x_jac
            )

            f_empty_jac = x -> Float64[]
            expected_empty = (Float64[], zeros(Float64, 0, length(x_jac)))
            fwd_empty_cache = Mooncake.prepare_derivative_cache(
                f_empty_jac,
                x_jac;
                config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
            )
            @test Mooncake.value_and_jacobian!!(fwd_empty_cache, f_empty_jac, x_jac) ==
                expected_empty

            rev_empty_cache = Mooncake.prepare_pullback_cache(
                f_empty_jac,
                x_jac;
                config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
            )
            @test Mooncake.value_and_jacobian!!(rev_empty_cache, f_empty_jac, x_jac) ==
                expected_empty
            @test Mooncake.value_and_jacobian!!(
                rev_empty_cache, f_empty_jac, [x + 1, y - 1]
            ) == expected_empty

            fwd_cache_jac_chunk1 = Mooncake.prepare_derivative_cache(
                f_jac, x_jac; config=Mooncake.Config(; chunk_size=1)
            )
            @test Mooncake.value_and_jacobian!!(fwd_cache_jac_chunk1, f_jac, x_jac) ==
                (f_jac(x_jac), expected_jac)

            hvp_cache = Mooncake.prepare_hvp_cache(sin, 1.0)
            @test_throws "value_and_jacobian!! only supports cache types Cache and FCache" Mooncake.value_and_jacobian!!(
                hvp_cache, sin, 1.0
            )

            # Multi-argument calls get a clear error, not an opaque MethodError.
            multi_cache = Mooncake.prepare_derivative_cache(x -> [sum(x)], [1.0, 2.0])
            @test_throws "supports only a single AbstractVector input" Mooncake.value_and_jacobian!!(
                multi_cache, x -> [sum(x)], [1.0, 2.0], [3.0]
            )

            f_mut_jac = x -> (x .*= 2; x .^ 2)
            x_mut_jac = [1.5, -2.0]
            rev_cache_mut_jac = Mooncake.prepare_pullback_cache(
                f_mut_jac,
                copy(x_mut_jac);
                config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
            )
            x_mut_jac_work = copy(x_mut_jac)
            val_mut_jac, jac_mut_jac = Mooncake.value_and_jacobian!!(
                rev_cache_mut_jac, f_mut_jac, x_mut_jac_work
            )
            @test x_mut_jac_work == x_mut_jac
            @test val_mut_jac == 4 .* x_mut_jac .^ 2
            @test jac_mut_jac ≈ [8 * x_mut_jac[1] 0.0; 0.0 8 * x_mut_jac[2]]

            x_mut_jac_chunked = [1.0, 2.0, 3.0]
            fwd_cache_mut_jac = Mooncake.prepare_derivative_cache(
                f_mut_jac,
                copy(x_mut_jac_chunked);
                config=Mooncake.Config(;
                    chunk_size=2, debug_mode=false, friendly_tangents=false
                ),
            )
            x_mut_jac_chunked_work = copy(x_mut_jac_chunked)
            val_mut_jac_chunked, jac_mut_jac_chunked = Mooncake.value_and_jacobian!!(
                fwd_cache_mut_jac, f_mut_jac, x_mut_jac_chunked_work
            )
            @test x_mut_jac_chunked_work == x_mut_jac_chunked
            @test val_mut_jac_chunked == 4 .* x_mut_jac_chunked .^ 2
            @test jac_mut_jac_chunked ≈ Diagonal(8 .* x_mut_jac_chunked)

            # The forward gradient and derivative must likewise leave a mutating `f`'s
            # input unchanged and give correct results across chunk sizes (the chunked
            # sweeps re-run `f` on shared input storage; without snapshot/restore an
            # in-place `f` compounds across chunks — see the FCache `input_snapshot` buffer).
            x_mut0 = [1.0, 2.0, 3.0]
            g_mut(x) = sum((x .*= 2; x .^ 2))   # true grad 8x
            for cs in (1, 2, 3)
                gc = Mooncake.prepare_gradient_cache(
                    g_mut, copy(x_mut0); config=Mooncake.Config(; chunk_size=cs)
                )
                xg = copy(x_mut0)
                _, (_, grad_mut) = Mooncake.value_and_gradient!!(gc, g_mut, xg)
                @test grad_mut ≈ 8 .* x_mut0
                @test xg == x_mut0
            end
            dc = Mooncake.prepare_derivative_cache(f_mut_jac, copy(x_mut0))
            xd = copy(x_mut0)
            Mooncake.value_and_derivative!!(
                dc, (f_mut_jac, Mooncake.NoTangent()), (xd, [1.0, 0.0, 0.0])
            )
            @test xd == x_mut0

            x_jac_parent = [x, y, 0.0]
            x_jac_view = @view x_jac_parent[1:2]
            f_view_jac = x -> [x[1]^2, x[1] + x[2]]
            for prepare_cache in
                (Mooncake.prepare_derivative_cache, Mooncake.prepare_pullback_cache)
                view_cache_jac = prepare_cache(
                    f_view_jac,
                    x_jac_view;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
                )
                @test_throws ArgumentError Mooncake.value_and_jacobian!!(
                    view_cache_jac, f_view_jac, x_jac_view
                )
            end
        end

        @testset "prepare_derivative_cache does not execute the function" begin
            let
                # Cache construction transforms IR but never runs the primal.
                NFWD_PREPARE_COUNTER[] = 0
                cache = Mooncake.prepare_derivative_cache(
                    _ndual_prepare_side_effect,
                    x;
                    config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
                )
                @test NFWD_PREPARE_COUNTER[] == 0

                # The scalar gradient then runs the primal exactly once.
                NFWD_PREPARE_COUNTER[] = 0
                @test Mooncake.value_and_gradient!!(cache, _ndual_prepare_side_effect, x) ==
                    (x^2 + one(x), (Mooncake.NoTangent(), 2 * x))
                @test NFWD_PREPARE_COUNTER[] == 1
            end
        end
    end

    @testset "value_and_hvp!!" begin
        TestUtils.test_hook(Val(:allow_unstable_hvp_interface_test)) do
            @testset "fcache dof skips undefined builtin-array slots" begin
                x = Vector{Any}(undef, 2)
                x[1] = 1.0
                @test Mooncake.dof(x) == 1
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

            # Single-direction (width-1) forward-over-reverse value correctness.
            # Regression guard for the two forward-mode V-drops these exercise:
            # (1) `lgetfield` `.ref` projection on `NDualArray` slots (array reads
            # through `getindex`/broadcast must keep their partials); (2) the
            # reverse rule's `fwds_oc`/`pb_oc` sharing one forward-tangent buffer
            # for their common capture stacks. Either drop silently zeroes `hvp`.
            @testset "HVP value correctness" begin
                # Scalar: hvp = f''(x)·v is distinct from the gradient f'(x).
                let f = x -> x^4, x = 2.0, v = 1.0
                    val, g, hv = value_and_hvp!!(prepare_hvp_cache(f, x), f, v, x)
                    @test val ≈ x^4
                    @test g ≈ 4x^3          # 32
                    @test hv ≈ 12x^2 * v    # 48 — would be 0 if a partial were dropped
                end
                # Array, Hessian 2I: hvp = 2v. Reads x via getindex/broadcast.
                let f = x -> sum(x .* x), x = [2.0, 3.0, 4.0], v = [1.0, 0.0, 0.0]
                    val, g, hv = value_and_hvp!!(prepare_hvp_cache(f, x), f, (v,), x)
                    @test val ≈ sum(x .* x)
                    @test g ≈ 2 .* x
                    @test hv ≈ 2 .* v
                end
                # Fused-primitive path (`sum(abs2, ·)`), same Hessian.
                let f = x -> sum(abs2, x), x = [2.0, 3.0, 4.0], v = [0.0, 1.0, 0.0]
                    _, _, hv = value_and_hvp!!(prepare_hvp_cache(f, x), f, (v,), x)
                    @test hv ≈ 2 .* v
                end
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
                # `cache` owns the returned `g`/`H`; snapshot before reusing the cache.
                g1, H1 = copy(g1), copy(H1)
                v2, g2, H2 = value_gradient_and_hessian!!(cache, f, x2)
                @test v1 ≈ 1.0
                @test v2 ≈ 13.0
                @test g1 ≈ [2.0, 0.0]
                @test g2 ≈ [4.0, 6.0]
                @test H1 ≈ H2
            end

            @testset "debug_mode=true" begin
                z = [1.2, 1.2]
                cache = prepare_hessian_cache(
                    rosen, z; config=Mooncake.Config(; debug_mode=true)
                )
                v, g, H = value_gradient_and_hessian!!(cache, rosen, z)
                @test v ≈ rosen(z)
                @test H ≈ rosen_H(z) rtol = 1e-10
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
                # `cache` owns the returned tuples; snapshot before reusing the cache.
                gx1, gy1, Hxx1, Hyy1 = copy(gx1), copy(gy1), copy(Hxx1), copy(Hyy1)
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
                @test_throws ArgumentError prepare_hessian_cache(f, x)
            end

            @testset "reject non-IEEEFloat element types" begin
                f(x) = sum(abs2, x)
                x = ComplexF64[1 + 0im, 2 + 0im]
                @test_throws ArgumentError prepare_hessian_cache(f, x)
            end

            @testset "reject mismatched element types across arguments" begin
                f(x, y) = sum(x .^ 2) + sum(y .^ 2)
                x = Float64[1.0, 2.0]
                y = Float32[3.0, 4.0]
                @test_throws ArgumentError prepare_hessian_cache(f, x, y)
            end

            @testset "reject mismatched function object" begin
                f(x) = sum(x .^ 2)
                g(x) = sum(3 .* x .^ 2)
                x = [1.0, 2.0]
                cache = prepare_hessian_cache(f, x)
                @test_throws ArgumentError value_gradient_and_hessian!!(cache, g, x)
            end

            @testset "reject HVP-only cache" begin
                f(x) = sum(x .^ 2)
                x = [1.0, 2.0]
                cache = Mooncake.prepare_hvp_cache(f, x)
                @test_throws ArgumentError value_gradient_and_hessian!!(cache, f, x)
            end

            @testset "cache buffer reuse (output aliasing)" begin
                f(x) = sum(x .^ 2)
                x = [1.0, 2.0, 3.0]
                cache = prepare_hessian_cache(f, x)
                _, g1, H1 = value_gradient_and_hessian!!(cache, f, x)
                _, g2, H2 = value_gradient_and_hessian!!(cache, f, x)
                # Both calls return the same cache-owned buffers.
                @test g1 === g2
                @test H1 === H2
            end

            @testset "multi-arg cache buffer reuse" begin
                f(x, y) = sum(x .^ 2) + sum(y .^ 2) + x[1] * y[1]
                x = [1.0, 2.0]
                y = [3.0, 4.0]
                cache = prepare_hessian_cache(f, x, y)
                _, (gx1, gy1), ((Hxx1, Hxy1), (Hyx1, Hyy1)) = value_gradient_and_hessian!!(
                    cache, f, x, y
                )
                _, (gx2, gy2), ((Hxx2, Hxy2), (Hyx2, Hyy2)) = value_gradient_and_hessian!!(
                    cache, f, x, y
                )
                @test gx1 === gx2 && gy1 === gy2
                @test Hxx1 === Hxx2 && Hxy1 === Hxy2
                @test Hyx1 === Hyx2 && Hyy1 === Hyy2
            end

            @testset "reject mismatched cache arity" begin
                f(x) = sum(abs2, x)
                f(x, y) = sum(abs2, x) + sum(abs2, y)
                # 2-arg cache, 3-arg call.
                cache2 = prepare_hessian_cache(f, [1.0], [2.0])
                @test_throws r"cache was prepared for 2 arguments but called with 3" value_gradient_and_hessian!!(
                    cache2, f, [1.0], [2.0], [3.0]
                )
                # 2-arg cache, 1-arg call — single-arg dispatch must report arity, not
                # the generic "not a hessian cache" error.
                @test_throws r"cache was prepared for 2 arguments but called with 1" value_gradient_and_hessian!!(
                    cache2, f, [1.0]
                )
                # 1-arg cache, 2-arg call — multi-arg dispatch, same expectation.
                cache1 = prepare_hessian_cache(f, [1.0, 2.0])
                @test_throws r"cache was prepared for 1 argument but called with 2" value_gradient_and_hessian!!(
                    cache1, f, [1.0, 2.0], [3.0]
                )
            end

            @testset "empty-cache reused at non-empty input" begin
                f(x) = sum(x .^ 2)
                cache = prepare_hessian_cache(f, Float64[])
                @test_throws ArgumentError value_gradient_and_hessian!!(
                    cache, f, [1.0, 2.0]
                )
                g(x, y) = sum(x .^ 2) + sum(y .^ 2)
                cache2 = prepare_hessian_cache(g, Float64[], Float64[])
                @test_throws ArgumentError value_gradient_and_hessian!!(
                    cache2, g, [1.0], Float64[]
                )
            end

            @testset "hessian cache mismatch errors" begin
                f(x) = sum(x .^ 2)
                x = [1.0, 2.0]
                cache = prepare_hessian_cache(f, x)
                @test_throws r"input vector has length 3 but cache was prepared for length 2" value_gradient_and_hessian!!(
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
