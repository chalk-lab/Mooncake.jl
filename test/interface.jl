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

# A callable holding an array, so the callable can share storage with an argument.
struct FwdAliasHolder{V}
    v::V
end
(h::FwdAliasHolder)(x) = sum(h.v .* x)

# A callable holding an array that returns its mutated argument. Its own dofs put the Jacobian
# sweep on the non-packable path, where the returned value aliases the caller's input.
struct FwdInPlaceScaler{V}
    v::V
end
(h::FwdInPlaceScaler)(x) = (x.=(h.v .* x); x)

# Two fields that may be the same array: aliasing inside ONE argument.
struct FwdAliasPair{A}
    p::A
    q::A
end

# A differentiable array beside a non-differentiable `Int` that selects how much of it is read.
# Passed as an ARGUMENT (a differentiable callable would leave the zero-allocation structured
# path, which is where the prepare-time state was kept).
struct FwdPrefixSum{V}
    w::V
    k::Int
end
fwd_prefix_sum(m::FwdPrefixSum) = sum(m.w[1:m.k])

# An abstractly-typed non-differentiable field: prepared at an `Int`, a call passing a `Float64`
# needs derivative storage the cache does not have.
struct FwdAbstractField
    a::Real
    w::Vector{Float64}
end
fwd_abstract_field(s::FwdAbstractField) = s.a * sum(s.w)

# A MUTABLE non-differentiable argument that `f` mutates.
mutable struct FwdCounter
    n::Int
end
fwd_counting(w, c::FwdCounter) = (c.n += 1; sum(w) + c.n)

# A `const` field cannot be written with `setfield!`, and a mutable NESTED in the argument stays
# shared with the caller unless the copy into the cache's object recurses.
mutable struct FwdConstTag
    const tag::Int
    n::Int
end
fwd_const_tag(w, c::FwdConstTag) = (c.n += 1; sum(w) + c.n + c.tag)

mutable struct FwdNestedCounter
    inner::FwdCounter
end
fwd_nested_counting(w, o::FwdNestedCounter) = (o.inner.n += 1; sum(w) + o.inner.n)

struct StructuredPair{A,B}
    u::A
    v::B
end

fwd_load_ptr(p::Ptr{Float64}) = unsafe_load(p)

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
            # The prepared-cache zero-allocation contract. Asserted, rather than the
            # `alloc_count > 0 ? @test_broken : @test` it replaces, which could not fail in
            # either branch and so asserted nothing on any version. On 1.11 and 1.12 every case
            # here is exactly zero; on 1.10 the `__call_rule` dispatch barrier (julia#61368, see
            # the note in `src/utils.jl`) costs 3-4 for all but `sum`, so bound it there.
            alloc_count = TestUtils.count_allocs(value_and_gradient!!, cache, fargs...)
            @static if VERSION < v"1.11-"
                @test alloc_count <= 4
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
            @test occursin("chunk=false", hvp_show)

            hvp_plain = repr(MIME"text/plain"(), hvp_cache)
            @test occursin("Mooncake.HVPCache", hvp_plain)
            @test occursin("mode: forward_over_reverse", hvp_plain)
            @test occursin("chunk: false", hvp_plain)
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
            # As for the gradient above: exactly zero on 1.11 and 1.12, and on 1.10 bounded by
            # the `__call_rule` dispatch barrier (julia#61368).
            alloc_count = TestUtils.count_allocs(value_and_pullback!!, cache, ȳ, fargs...)
            @static if VERSION < v"1.11-"
                @test alloc_count <= 3
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

        # Fix for #1242: compiled callables retain reflection/IR objects with cyclic
        # reference graphs, so descending into their fields never terminates.
        @testset "_copy_output compiled callables" begin
            oc = Base.Experimental.@opaque x -> x + 1
            @test Mooncake._copy_output(oc) === oc

            # End-to-end: friendly forward-over-reverse over a genuinely non-primitive array
            # function (`sum(x.^2)` broadcasts, unlike the primitive `sum(abs2, x)`) builds a
            # gradient closure that captures the compiled reverse rule. `friendly_tangents=true`
            # must (a) prepare without descending into that rule's reflection graph, and (b)
            # evaluate HVP/Hessian correctly — the inner forward cache is always non-friendly, so
            # the flag does not change results. Regression for the two HVP/Hessian friendly bugs.
            x = [1.0, 2.0, 3.0]
            v = [1.0, 0.0, 0.0]
            for f in (x -> sum(x .^ 2), x -> sum(abs2, x)), ft in (false, true)
                cfg = Mooncake.Config(; friendly_tangents=ft)
                hvp_cache = Mooncake.prepare_hvp_cache(f, x; config=cfg)
                @test hvp_cache isa Mooncake.HVPCache
                _, g, h = Mooncake.value_and_hvp!!(hvp_cache, f, v, x)
                @test g ≈ 2 .* x
                @test h ≈ 2 .* v
                hess_cache = Mooncake.prepare_hessian_cache(f, x; config=cfg)
                @test hess_cache isa Mooncake.HVPCache
                _, _, H = Mooncake.value_gradient_and_hessian!!(hess_cache, f, x)
                @test H ≈ 2 * LinearAlgebra.I(3)
            end
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

            # Regression: the FULL gradient must fill every element, not just the first. The
            # packable path seeds the element-major block by linear index; a 2-D `block[lane,
            # elem]` index silently mis-seeds all but element 1 on Julia 1.10 (the block is a
            # flat `Vector` there), giving e.g. [2x₁, 0, 0]. Exercise widths that split and span.
            x3 = [1.0, 2.0, 3.0]
            for cs in (1, 2, 3)
                gc = Mooncake.prepare_derivative_cache(
                    f_arr, x3; config=Mooncake.Config(; chunk_size=cs, kwargs...)
                )
                _, (_, g) = Mooncake.value_and_gradient!!(gc, f_arr, x3)
                @test g == 2 .* x3
            end
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

        @testset "forward gradient accepts a `Dict` / `Set` argument" begin
            # `_basis_seed!!` walks the forward V writing standard-basis lanes. A lifted `Dict`
            # (and a `Set`, which wraps one) has bare `Memory` backing its `slots`/`keys`, which
            # carry no derivative at all, and the walk had no method for it: forward threw a raw
            # `MethodError` naming an internal for an input reverse mode accepts, `_check_liftable_input`
            # admits, and Julia 1.10 forward already handled (1.10 `Dict`s use `Vector`s).
            #
            # Distinct coefficients, so a gradient entry landing in the wrong slot cannot pass by
            # symmetry: the seed walk must advance the dof cursor in the order `dof` counts, and a
            # mismatch there misplaces entries silently rather than erroring. `kwargs` comes from
            # the enclosing loop, so the four iterations cover the seed path with debug mode on and
            # off rather than repeating one configuration.
            # A `Dict` whose two VALUES are one array: the seed walks the backing `Memory`,
            # which registered the container but delegated to the cache-free factory, so the two
            # elements got independent partials and each position saw only its own contribution.
            galias(dd) = sum(dd[1]) + sum(dd[2])
            # A multi-line function, not `mkalias() = (v = …; …)`: inside a `@testset`, a
            # one-line function whose body is a `;`-block assigns to the ENCLOSING scope, so a
            # local named `z` there silently overwrites the testset's own `z`.
            function mkalias()
                shared = [1.0, 2.0, 3.0]
                return Dict{Int,Vector{Float64}}(1 => shared, 2 => shared)
            end
            vals_of(g) = [
                g.fields.vals[i] for
                i in eachindex(g.fields.vals) if isassigned(g.fields.vals, i)
            ]
            _, g_rev = Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(galias, mkalias()), galias, mkalias()
            )
            @testset "aliased Dict values, chunk_size=$w" for w in (1, 2)
                _, g_fwd = Mooncake.value_and_gradient!!(
                    Mooncake.prepare_derivative_cache(
                        galias, mkalias(); config=Mooncake.Config(; chunk_size=w, kwargs...)
                    ),
                    galias,
                    mkalias(),
                )
                # Analytic: the primal is 2*sum(z), so every entry is 2.0.
                @test all(v -> v == fill(2.0, 3), vals_of(g_fwd[2]))
                @test vals_of(g_fwd[2]) == vals_of(g_rev[2])
            end

            fdict(d, v) = d[:a] * v[1] + 10.0 * d[:b] * v[2] + 100.0 * sum(v)
            mkd() = Dict(:a => 2.0, :b => 3.0)
            v0 = [5.0, 7.0]
            vr, gr = Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(fdict, mkd(), v0), fdict, mkd(), copy(v0)
            )
            @testset "chunk_size=$w" for w in (1, 2, 3)
                vf, gf = Mooncake.value_and_gradient!!(
                    Mooncake.prepare_derivative_cache(
                        fdict, mkd(), v0; config=Mooncake.Config(; chunk_size=w, kwargs...)
                    ),
                    fdict,
                    mkd(),
                    copy(v0),
                )
                @test vf == vr
                @test gf[3] == gr[3]
                @test collect(gf[2].fields.vals) == collect(gr[2].fields.vals)
            end
            # An `IdDict` interleaves keys and values in one backing `ht`, so its V has no separate
            # non-differentiable field and it needed its own seed method. The per-key gradients are
            # checked, not just `d/dv`: the seed walk has to advance the dof cursor in the order
            # `dof` counts, and a mismatch misplaces entries silently rather than erroring.
            fid(d, v) = d[:a] * v[1] + 10.0 * d[:b] * v[2] + 100.0 * sum(v)
            mkid() = IdDict{Symbol,Float64}(:a => 2.0, :b => 3.0)
            vir, gir = Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(fid, mkid(), v0), fid, mkid(), copy(v0)
            )
            @testset "IdDict, chunk_size=$w" for w in (1, 2, 3)
                vif, gif = Mooncake.value_and_gradient!!(
                    Mooncake.prepare_derivative_cache(
                        fid, mkid(), v0; config=Mooncake.Config(; chunk_size=w, kwargs...)
                    ),
                    fid,
                    mkid(),
                    copy(v0),
                )
                @test vif == vir
                @test gif[3] == gir[3]
                @test gif[2][:a] == gir[2][:a]
                @test gif[2][:b] == gir[2][:b]
            end
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
            # `NDualRef` method, so this threw a MethodError. Forward must match the reverse
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

            # Debug-mode rules wrap every rule and allocate, so the zero-allocation assertions
            # below are checked outside debug mode only. Everything else here runs under it.
            check_allocs = !get(kwargs, :debug_mode, false)
            scalar_allocs = TestUtils.count_allocs(
                Mooncake.value_and_gradient!!, scalar_cache_grad_fwd, f_scalar, x
            )
            check_allocs && @test scalar_allocs == 0

            scalar_f = CountedChunkScalarCall()
            scalar_cache_grad_fwd = Mooncake.prepare_derivative_cache(
                scalar_f,
                x,
                y;
                config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
            )
            CHUNK_SCALAR_EVAL_COUNT[] = 0
            @test Mooncake.value_and_gradient!!(scalar_cache_grad_fwd, scalar_f, x, y) ==
                (z, (Mooncake.NoTangent(), y - sin(x), x))
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
            check_allocs && @test TestUtils.count_allocs(
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
            ) == (sum(abs2, singleton_x_arr), (Mooncake.NoTangent(), 2 .* singleton_x_arr))
            @test CHUNK_ARRAY_EVAL_COUNT[] == 1
            check_allocs && @test TestUtils.count_allocs(
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
                f5, x5; config=Mooncake.Config(; debug_mode=false, friendly_tangents=false)
            )
            @test Mooncake.value_and_gradient!!(cache_5, f5, x5) ==
                (sum(abs2, x5), (Mooncake.NoTangent(), 2 .* x5))
            check_allocs && @test TestUtils.count_allocs(
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
            check_allocs && @test TestUtils.count_allocs(
                Mooncake.value_and_gradient!!, cache_10, f10, x10
            ) == 0

            # Non-packable inputs (here a NamedTuple) also chunk through the generic
            # chunked gradient path: multi-dof builds a native chunk rule and the
            # gradient is correct. (Such inputs were previously pinned to width 1.)
            nt_x = (; a=1.3, b=2.1, c=0.7)
            f_nt = nt -> nt.a^2 * nt.b + sin(nt.a) * nt.c
            cache_nt = Mooncake.prepare_derivative_cache(
                f_nt, nt_x; config=Mooncake.Config(; friendly_tangents=true, kwargs...)
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
                ft, tx; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
            )
            @test getfield(ct, :gradient_seed) isa Mooncake.StructuredGradSeed
            yt, gt = Mooncake.value_and_gradient!!(ct, ft, tx)
            @test yt == ft(tx)
            @test gt[2][1] ≈ 2 .* tx[1]
            @test gt[2][2] ≈ 2 .* tx[2]
            check_allocs &&
                @test TestUtils.count_allocs(Mooncake.value_and_gradient!!, ct, ft, tx) == 0

            fA = A -> sum(abs2, A)
            Ax = [1.0 2.0; 3.0 4.0]
            cA = Mooncake.prepare_derivative_cache(
                fA, Ax; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
            )
            @test getfield(cA, :gradient_seed) isa Mooncake.StructuredGradSeed
            check_allocs &&
                @test TestUtils.count_allocs(Mooncake.value_and_gradient!!, cA, fA, Ax) == 0
            _, gA = Mooncake.value_and_gradient!!(cA, fA, Ax)
            @test gA[2] ≈ 2 .* Ax

            # Primal refresh: prepare at one point, evaluate at another.
            cr = Mooncake.prepare_derivative_cache(
                ft,
                ([1.0, 1.0, 1.0], [1.0, 1.0]);
                config=Mooncake.Config(; friendly_tangents=false, kwargs...),
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
                fip, tip0; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
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
                fsame, tsame; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
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
                fz0, xz0; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
            )
            yz0, _ = Mooncake.value_and_gradient!!(cz0, fz0, xz0)
            @test yz0 == 2.5
            @test xz0 == [10, 20, 30]                      # user input not mutated

            # Mixed array + scalar input has a non-array dof, so the gather bails and the
            # generic chunked path runs — still correct.
            fmix = nt -> sum(nt.v) + nt.s^2
            mx = (; v=[1.0, 2.0], s=3.0)
            cmix = Mooncake.prepare_derivative_cache(
                fmix, mx; config=Mooncake.Config(; friendly_tangents=true, kwargs...)
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
                fnt, ntx; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
            )
            @test getfield(cnt, :gradient_seed) isa Mooncake.IsbitsGradSeed
            ynt, gnt = Mooncake.value_and_gradient!!(cnt, fnt, ntx)
            @test ynt == fnt(ntx)
            @test gnt[2].a ≈ 2 * ntx.a * ntx.b + cos(ntx.a) * ntx.c
            @test gnt[2].b ≈ ntx.a^2
            @test gnt[2].c ≈ sin(ntx.a)
            check_allocs && @test TestUtils.count_allocs(
                Mooncake.value_and_gradient!!, cnt, fnt, ntx
            ) == 0

            # immutable struct of scalars: native gradient is a `Tangent` (scattered via the
            # `Tangent` branch), and prepare-at-x0/evaluate-at-x1 (primal refresh) is correct.
            fsp = p -> p.x1^2 * p.x2
            csp = Mooncake.prepare_derivative_cache(
                fsp,
                SimplePair(1.0, 1.0);
                config=Mooncake.Config(; friendly_tangents=false, kwargs...),
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
                f10, nt10; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
            )
            @test getfield(c10, :gradient_chunk_size) < 10
            _, g10 = Mooncake.value_and_gradient!!(c10, f10, nt10)
            @test g10[2].x1 ≈ 2.0
            @test g10[2].x10 ≈ 20.0
            check_allocs && @test TestUtils.count_allocs(
                Mooncake.value_and_gradient!!, c10, f10, nt10
            ) == 0

            # Complex scalar dofs have an isbits V but two dofs per element, which the isbits
            # barrier's scatter cannot handle — they must take the generic path, not crash.
            fz = z -> abs2(z)
            cz = Mooncake.prepare_derivative_cache(
                fz,
                1.0 + 2.0im;
                config=Mooncake.Config(; friendly_tangents=false, kwargs...),
            )
            @test !(getfield(cz, :gradient_seed) isa Mooncake.IsbitsGradSeed)
            yz, gz = Mooncake.value_and_gradient!!(cz, fz, 1.0 + 2.0im)
            @test yz == abs2(1.0 + 2.0im)
            @test gz[2] ≈ 2.0 + 4.0im
            fzt = t -> abs2(t[1]) + t[2]^2
            czt = Mooncake.prepare_derivative_cache(
                fzt,
                (1.0 + 2.0im, 3.0);
                config=Mooncake.Config(; friendly_tangents=false, kwargs...),
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
                clo,
                (; a=1.0, b=2.0);
                config=Mooncake.Config(; friendly_tangents=false, kwargs...),
            )
            @test !(getfield(cclo, :gradient_seed) isa Mooncake.IsbitsGradSeed)
            _, gclo = Mooncake.value_and_gradient!!(cclo, clo, (; a=1.0, b=2.0))
            @test gclo[2].a ≈ 10.0
            @test gclo[2].b ≈ 4.0
        end

        @testset "a mutating `f` over one repeated argument shares partials" begin
            # Every argument-tuple lift needs ONE shared aliasing cache: the float-array `lift`
            # packs its seed into a fresh partials block per call, so two arguments over one
            # storage otherwise get independent partials while the PRIMAL still aliases. The
            # returned pair is then self-contradictory — the value says the mutation was seen, the
            # derivative says it was not.
            #
            # The `sum(a .* b)` case below cannot expose this: with no mutation, independent blocks
            # give the right answer anyway by the product rule. The trigger is mutation THROUGH the
            # aliased storage.
            fmut(x, y) = (x .*= 2.0; sum(y))
            mk() = collect(1.0:4.0)
            # Analytic: g(a) = 2*sum(a), so the value is 20.0 and the JVP along ones(4) is 8.0.
            @testset "friendly_tangents=$fr" for fr in (false, true)
                a = mk()
                cache = Mooncake.prepare_derivative_cache(
                    fmut, a, a; config=Mooncake.Config(; friendly_tangents=fr, kwargs...)
                )
                # ONE array passed at both positions — `mk()` twice would be two distinct
                # arrays, and the prepared-aliased cache rightly refuses that.
                aa, dd = mk(), ones(4)
                v, d = Mooncake.value_and_derivative!!(
                    cache, (fmut, Mooncake.NoTangent()), (aa, dd), (aa, dd)
                )
                @test (v, d) == (20.0, 8.0)
            end
            # The rule-level tuple method takes no `Config` at all and had the same gap.
            let a = mk()
                aa, dd = mk(), ones(4)
                v, d = Mooncake.value_and_derivative!!(
                    Mooncake.build_frule(fmut, a, a),
                    (fmut, Mooncake.NoTangent()),
                    (aa, dd),
                    (aa, dd),
                )
                @test (v, d) == (20.0, 8.0)
            end
        end

        @testset "friendly cache refuses only prepared-aliased/called-distinct" begin
            # Prepare-time aliased arguments share one tangent buffer, so a call with distinct
            # arguments leaves both positions holding the last tangent written: 6.0 for a truth
            # of 5.0. Reverse already rejects this.
            g_al = (a, b) -> sum(a .* b)
            X_al = [1.0, 2.0]
            A_al, dA_al = [1.0, 2.0], [1.0, 0.0]
            B_al, dB_al = [3.0, 4.0], [0.0, 1.0]
            friendly = Mooncake.Config(; friendly_tangents=true, kwargs...)
            c_al = Mooncake.prepare_derivative_cache(g_al, X_al, X_al; config=friendly)
            @test_throws Mooncake.PreparedCacheError Mooncake.value_and_derivative!!(
                c_al, (g_al, Mooncake.NoTangent()), (A_al, dA_al), (B_al, dB_al)
            )
            # The opposite direction is correct and must keep working: distinct buffers each hold
            # the caller's seed, and the aliased primal receives both.
            c_di = Mooncake.prepare_derivative_cache(g_al, A_al, B_al; config=friendly)
            dX_al = [1.0, 1.0]
            v_di, d_di = Mooncake.value_and_derivative!!(
                c_di, (g_al, Mooncake.NoTangent()), (X_al, dX_al), (X_al, dX_al)
            )
            @test d_di ≈ 2 * sum(dX_al .* X_al)
            # The non-friendly method needs no check: it lifts the caller's own tangents afresh.
            c_nf = Mooncake.prepare_derivative_cache(g_al, X_al, X_al)
            _, d_nf = Mooncake.value_and_derivative!!(
                c_nf, (g_al, Mooncake.NoTangent()), (A_al, dA_al), (B_al, dB_al)
            )
            @test d_nf ≈ sum(dA_al .* B_al) + sum(A_al .* dB_al)

            # A REPEATED mutable argument given two DIFFERENT tangents is ill-posed: the two
            # positions are one array, so they are one tangent, and only one direction exists to
            # carry. Both tuple methods used to answer it, differently and silently — the
            # unfriendly path kept the first tangent (2.0) and the friendly path the last (4.0),
            # which are the JVPs along `dA_al` and `dB_al` respectively.
            for cfg in (friendly, Mooncake.Config(; friendly_tangents=false, kwargs...))
                c_rep = Mooncake.prepare_derivative_cache(g_al, X_al, X_al; config=cfg)
                @test_throws ArgumentError Mooncake.value_and_derivative!!(
                    c_rep, (g_al, Mooncake.NoTangent()), (X_al, dA_al), (X_al, dB_al)
                )
                # The same tangent at both positions is well-posed and still answered.
                _, d_rep = Mooncake.value_and_derivative!!(
                    c_rep, (g_al, Mooncake.NoTangent()), (X_al, dA_al), (X_al, dA_al)
                )
                @test d_rep ≈ 2 * sum(dA_al .* X_al)
            end

            # The same ill-posedness ONE LEVEL DOWN, where the top-level `===` scan cannot see
            # it: a closure capturing the array that is also passed as the argument. The two
            # positions are different objects, so that scan passes, but they are one storage and
            # so one tangent. `zero_tangent` for the callable is the natural way to write "do not
            # perturb the function", and it silently returned 0.0 where the directional
            # derivative is 2 * sum(d) * sum(cap) = 6.0. Only the non-friendly method needs the
            # check; the friendly one converts into prepared buffers that already share.
            mk_capturing(a) = y -> sum(y) * sum(a)
            cap_arr = [1.0, 2.0]
            d_cap = [1.0, 0.0]
            f_cap = mk_capturing(cap_arr)
            c_cap = Mooncake.prepare_derivative_cache(
                f_cap, cap_arr; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
            )
            @test_throws ArgumentError Mooncake.value_and_derivative!!(
                c_cap, (f_cap, Mooncake.zero_tangent(f_cap)), (cap_arr, d_cap)
            )
            # One tangent shared across both positions is well-posed and still answered.
            _, d_shared = Mooncake.value_and_derivative!!(
                c_cap, (f_cap, Mooncake.Tangent((a=d_cap,))), (cap_arr, d_cap)
            )
            @test d_shared ≈ 2 * sum(d_cap) * sum(cap_arr)
        end

        @testset "reused cache reads call-time non-differentiable state" begin
            # `_refresh_seed!` restores the differentiable leaves of the prepare-time `deepcopy`;
            # everything non-differentiable used to keep its prepare-time value for the life of
            # the cache. Prepared at `k = 2` and called at `k = 4`, this returned 3.0 with a
            # gradient of [1, 1, 0, 0].
            w491 = [1.0, 2.0, 3.0, 4.0]
            plain = Mooncake.Config(; friendly_tangents=false, kwargs...)
            c491 = Mooncake.prepare_derivative_cache(
                fwd_prefix_sum, FwdPrefixSum(w491, 2); config=plain
            )
            @test getfield(c491, :gradient_seed) isa Mooncake.StructuredGradSeed
            v491, g491 = Mooncake.value_and_gradient!!(
                c491, fwd_prefix_sum, FwdPrefixSum(w491, 4)
            )
            @test v491 ≈ sum(w491)
            @test g491[2].fields.w ≈ ones(4)

            # A `SubArray`'s indices are non-differentiable too, and `_validate_prepared_cache`
            # cannot catch a change in them: both views have the same type and the same size.
            p491 = collect(1.0:6.0)
            fv491 = v -> sum(v)
            cv491 = Mooncake.prepare_derivative_cache(fv491, view(p491, 1:3); config=plain)
            vv491, gv491 = Mooncake.value_and_gradient!!(cv491, fv491, view(p491, 4:6))
            @test vv491 ≈ sum(view(p491, 4:6))
            @test gv491[2].fields.parent ≈ [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        end

        @testset "refreshing non-differentiable state keeps the slot coherent and the caller's" begin
            plain498 = Mooncake.Config(; friendly_tangents=false, kwargs...)
            # The refresh reads the CALL's non-differentiable state. A `NoDual` slot asserts the
            # position has no derivative, so a call-time value whose canonical dual DOES have one
            # must be refused here rather than reaching the dual IR's typeassert as a raw
            # `TypeError`. `_validate_prepared_cache` cannot see it: both calls pass an `S`.
            s498 = Mooncake.prepare_derivative_cache(
                fwd_abstract_field, FwdAbstractField(2, [1.0, 2.0, 3.0]); config=plain498
            )
            @test Mooncake.value_and_gradient!!(
                s498, fwd_abstract_field, FwdAbstractField(5, [1.0, 2.0, 3.0])
            )[1] ≈ 30.0
            @test_throws Mooncake.PreparedCacheError Mooncake.value_and_gradient!!(
                s498, fwd_abstract_field, FwdAbstractField(3.0, [1.0, 2.0, 3.0])
            )

            # A MUTABLE non-differentiable argument must be copied into the cache's own object, not
            # taken from the call: `f` mutates it, and taking it wrote through to the user's value.
            c500 = FwdCounter(0)
            w500 = collect(1.0:16.0)
            cache500 = Mooncake.prepare_derivative_cache(
                fwd_counting, w500, c500; config=plain498
            )
            v500, g500 = Mooncake.value_and_gradient!!(cache500, fwd_counting, w500, c500)
            @test c500.n == 0                 # the caller's object is untouched
            @test g500[2] ≈ ones(16)
            # 16 dof at chunk 8 is TWO chunks, and the refresh runs per chunk: without that,
            # chunk 1's mutation carried into chunk 2 and the value came from the last chunk
            # (138.0 for a truth of 137.0). One chunk was already right, so `n` must exceed the
            # chunk width for this to bite.
            @test v500 ≈ fwd_counting(collect(1.0:16.0), FwdCounter(0))

            # A `const` field: the copy into the cache's object must write it, which `setfield!`
            # cannot do at all: it threw for every call, whether or not the field had changed.
            cache501 = Mooncake.prepare_derivative_cache(
                fwd_const_tag, w500, FwdConstTag(1, 0); config=plain498
            )
            c501 = FwdConstTag(7, 0)
            v501, g501 = Mooncake.value_and_gradient!!(cache501, fwd_const_tag, w500, c501)
            @test v501 ≈ fwd_const_tag(collect(1.0:16.0), FwdConstTag(7, 0))
            @test (c501.tag, c501.n) == (7, 0)
            @test g501[2] ≈ ones(16)

            # A mutable NESTED in the argument: copying one level deep left it shared with the
            # caller, so `f`'s in-place update wrote through and compounded across chunks.
            o502 = FwdNestedCounter(FwdCounter(0))
            cache502 = Mooncake.prepare_derivative_cache(
                fwd_nested_counting, w500, FwdNestedCounter(FwdCounter(0)); config=plain498
            )
            v502, g502 = Mooncake.value_and_gradient!!(
                cache502, fwd_nested_counting, w500, o502
            )
            @test o502.inner.n == 0
            @test v502 ≈
                fwd_nested_counting(collect(1.0:16.0), FwdNestedCounter(FwdCounter(0)))
            @test g502[2] ≈ ones(16)

            # A `Ptr` argument: the aliasing check must use the same tangent entry point as the
            # shared-dof count it is compared against, which returns the documented placeholder
            # rather than throwing.
            p499 = [3.0]
            @test Mooncake.prepare_derivative_cache(
                fwd_load_ptr, pointer(p499); config=plain498
            ) isa Any
        end

        @testset "cache copy refuses a size it cannot hold" begin
            # The cached buffer is sized at preparation time and the copy indexes it under
            # `@inbounds` while iterating the SOURCE, and `isassigned(dst, i)` reports false out of
            # range rather than throwing. A longer source therefore wrote past the end: a segfault
            # here, and below a silently truncated value handed back before the corruption showed
            # up at an unrelated GC.
            f_nest = x -> sum(sum, x)
            c_nest = Mooncake.prepare_derivative_cache(f_nest, [collect(1.0:2.0)])
            @test_throws Mooncake.PreparedCacheError Mooncake.value_and_gradient!!(
                c_nest, f_nest, [collect(1.0:400.0)]
            )
            # The OUTPUT side: the input shape is unchanged, so input validation passes, and only
            # the non-differentiable `n` moves the output length.
            f_out = (v, n) -> fill(sum(v), n)
            c_out = Mooncake.prepare_pullback_cache(f_out, ones(2), 2)
            @test_throws Mooncake.PreparedCacheError Mooncake.value_and_pullback!!(
                c_out, ones(400), f_out, ones(2), 400
            )
            # A self-referential reference-element array: the two-argument copy recursed into
            # elements two-argument and never reached the cycle-aware family, so it overflowed the
            # stack. Reverse mode always handled this shape.
            f_cyc = v::Vector{Any} -> v[1]::Float64 * 2.0
            mk_cyc = () -> (a=Any[1.0]; push!(a, a); a)
            c_cyc = Mooncake.prepare_derivative_cache(f_cyc, mk_cyc())
            @test Mooncake.value_and_gradient!!(c_cyc, f_cyc, mk_cyc())[1] == 2.0
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

        @testset "chunked forward through array growth" begin
            # Growing a lifted array grows its partials block, and every lane must survive the
            # growth — not just lane 1. On Julia 1.11+ growth is a derived rule, which the
            # registered test cases run at width 1 only, so this is the width>1 cover.
            grow(v) = (w=copy(v); push!(w, 2 * v[1]); pushfirst!(w, sum(v)); sum(abs2, w))
            v = randn(StableRNG(123), 6)
            oracle = Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(grow, v), grow, v
            )[2][2]
            @testset "chunk_size $W" for W in (1, 2, 3, 5)
                cache = Mooncake.prepare_derivative_cache(
                    grow, v; config=Mooncake.Config(; chunk_size=W)
                )
                @test Mooncake.value_and_gradient!!(cache, grow, v)[2][2] ≈ oracle
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

            @testset "returned value survives the input restore and a later call" begin
                # Non-packable: the seed aliases the caller's `x`, so for an `f` returning its
                # mutated argument the final restore rewrote the value already returned.
                scale2!(v) = (v.=2 .* v; v)
                sc = FwdInPlaceScaler([2.0, 2.0, 2.0])
                cache_ip = Mooncake.prepare_derivative_cache(sc, [1.0, 2.0, 3.0])
                v_ip, _ = Mooncake.value_and_jacobian!!(cache_ip, sc, [1.0, 2.0, 3.0])
                @test v_ip == [2.0, 4.0, 6.0]
                # The zero-allocation path returns a value aliasing a cache-owned buffer, as it
                # does for `J`; the docstring says so and the allocation test pins the guarantee
                # that forbids copying it. Assert the documented behaviour so a future copy has to
                # change the contract deliberately rather than by accident.
                cache_pk = Mooncake.prepare_derivative_cache(scale2!, [1.0, 2.0, 3.0])
                v_first, _ = Mooncake.value_and_jacobian!!(
                    cache_pk, scale2!, [1.0, 2.0, 3.0]
                )
                @test v_first == [2.0, 4.0, 6.0]
                Mooncake.value_and_jacobian!!(cache_pk, scale2!, [10.0, 20.0, 30.0])
                @test v_first == [20.0, 40.0, 60.0]
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

            f_wrapper_out = x -> view(x .* x, 1:2)
            wrapper_out_cache = Mooncake.prepare_derivative_cache(
                f_wrapper_out,
                x_jac;
                config=Mooncake.Config(; debug_mode=false, friendly_tangents=false),
            )
            @test_throws "value_and_jacobian!! does not support a" Mooncake.value_and_jacobian!!(
                wrapper_out_cache, f_wrapper_out, x_jac
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

            # Differentiable `f` (captures a vector ⇒ dof(f) ≥ 1) with a short input and a vector
            # output takes the non-packable forward path, where the per-chunk width
            # `W = gradient_chunk_size` includes `f`'s dofs, so `W > length(x)`. The first-chunk
            # J-write loop must guard `lane <= total_dof`, else it writes past `J`'s `length(x)`
            # columns (BoundsError under --check-bounds=yes). Forward must match the reverse oracle.
            let
                g_cap = let w = collect(1.0:7.0)
                    z -> [z[1] * sum(w), z[1] + z[2]]
                end
                z = [0.5, 0.5]
                cf = Mooncake.prepare_derivative_cache(g_cap, z)
                @test getfield(cf, :gradient_chunk_size) > length(z)  # W > total_dof
                _, Jf = Mooncake.value_and_jacobian!!(cf, g_cap, z)
                _, Jr = Mooncake.value_and_jacobian!!(
                    Mooncake.prepare_pullback_cache(g_cap, z), g_cap, z
                )
                @test Jf == Jr == [28.0 0.0; 1.0 1.0]
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

            @testset "multi-argument HVP is rejected" begin
                # Like `value_and_jacobian!!`, HVP supports only a single vector input;
                # rejected eagerly at prepare and at the compute call.
                f(x, y) = sum(x .* x) + sum(y .* y)
                x = [1.0, 2.0]
                y = [3.0]
                @test_throws ArgumentError prepare_hvp_cache(f, x, y)
                cache = prepare_hvp_cache(sum, x)
                @test_throws ArgumentError value_and_hvp!!(
                    cache, sum, ([1.0, 0.0], [1.0]), x, y
                )
            end

            @testset "HVP validates tangent shapes" begin
                x = [1.0, 2.0]
                cache1 = prepare_hvp_cache(sum, x)
                @test_throws ArgumentError value_and_hvp!!(cache1, sum, [1.0], x)
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
                    val, g, hv = value_and_hvp!!(prepare_hvp_cache(f, x), f, v, x)
                    @test val ≈ sum(x .* x)
                    @test g ≈ 2 .* x
                    @test hv ≈ 2 .* v
                end
                # Fused-primitive path (`sum(abs2, ·)`), same Hessian.
                let f = x -> sum(abs2, x), x = [2.0, 3.0, 4.0], v = [0.0, 1.0, 0.0]
                    _, _, hv = value_and_hvp!!(prepare_hvp_cache(f, x), f, v, x)
                    @test hv ≈ 2 .* v
                end
                # BLAS `dot` path: the reverse `dot` pullback threads `Ptr{NoTangent}`
                # fdata pointers, whose forward V must keep the per-lane partial pointers (else the
                # forward-over-reverse `_new_` backing mismatches and crashes).
                let f = x -> dot(x, x), x = [2.0, 3.0, 4.0], v = [1.0, 0.0, 0.0]
                    val, g, hv = value_and_hvp!!(prepare_hvp_cache(f, x), f, v, x)
                    @test val ≈ dot(x, x)
                    @test g ≈ 2 .* x
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

            @testset "BLAS quadratic form (dot)" begin
                # `dot(x, A*x)/2` has gradient `A*x` and Hessian `A`; its reverse rule runs through
                # BLAS on raw pointers, so forward-over-reverse threads `Ptr{NoTangent}` fdata pointers
                # that must keep their per-lane V. Previously untested (all other cases are elementwise).
                A = [2.0 0.5 0.0; 0.5 3.0 0.1; 0.0 0.1 4.0]  # symmetric ⇒ Hessian is exactly A
                f(x) = dot(x, A * x) / 2
                x = [0.5, -0.2, 0.9]
                v, g, H = value_gradient_and_hessian!!(prepare_hessian_cache(f, x), f, x)
                @test g ≈ A * x
                @test H ≈ A
            end

            @testset "chunked Hessian == width-1 (chunk_size $W)" for W in (1, 2, 3, 5)
                # The Hessian sweep batches W forward-over-reverse columns per pass; results must
                # match the width-1 column loop across widths (incl. n not divisible by W), and the
                # input must be left unchanged. value_and_hvp!! must stay width-1 regardless.
                f(x) = sum(abs2, x) + x[1] * x[2] + 0.5 * x[2] * x[3]
                x0 = [0.3, -0.7, 1.1, 0.5, -0.2]
                ref = prepare_hessian_cache(
                    f, copy(x0); config=Mooncake.Config(; chunk_size=1)
                )
                _, g1, H1 = value_gradient_and_hessian!!(ref, f, copy(x0))
                xc = copy(x0)
                c = prepare_hessian_cache(
                    f, copy(x0); config=Mooncake.Config(; chunk_size=W)
                )
                _, g, H = value_gradient_and_hessian!!(c, f, xc)
                @test H ≈ H1 rtol = 1e-10
                @test g ≈ g1 rtol = 1e-10
                @test xc == x0
                # A chunk-configured cache still serves a width-1 single-direction HVP.
                v = [1.0, 0.0, 0.0, 0.0, 0.0]
                hc = prepare_hvp_cache(f, copy(x0); config=Mooncake.Config(; chunk_size=W))
                _, _, hv = value_and_hvp!!(hc, f, v, copy(x0))
                @test hv ≈ H1[:, 1] rtol = 1e-10
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

            # `FwdAliasHolder(w)(w)` is `sum(w .^ 2)`, whose Hessian is `2I`. A basis sweep
            # reaches the shared leaf at one position per column, so the chunked sweep returned
            # `I`. Both sweeps must refuse: the width-1 one inherits the guard from
            # `value_and_hvp!!`, the chunked one only from the entry-point check.
            @testset "aliased input is refused on both sweeps" begin
                w = [1.0, 2.0, 3.0]
                for cfg in (Mooncake.Config(), Mooncake.Config(; chunk_size=1))
                    cache = prepare_hessian_cache(FwdAliasHolder(w), w; config=cfg)
                    @test_throws ArgumentError value_gradient_and_hessian!!(
                        cache, FwdAliasHolder(w), w
                    )
                end
                # The same shape without sharing is unaffected.
                u = copy(w)
                cache = prepare_hessian_cache(FwdAliasHolder(w), u)
                _, g, H = value_gradient_and_hessian!!(cache, FwdAliasHolder(w), u)
                @test g ≈ w
                @test H ≈ zeros(3, 3)
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

            @testset "multi-argument Hessian is rejected" begin
                # Like `value_and_jacobian!!`, the Hessian supports only a single vector input;
                # rejected eagerly at prepare and at the compute call.
                f(x, y) = sum(x .^ 2) + sum(y .^ 2) + x[1] * y[1]
                x = [1.0, 2.0]
                y = [3.0, 4.0]
                @test_throws ArgumentError prepare_hessian_cache(f, x, y)
                cache = prepare_hessian_cache(sum, x)
                @test_throws ArgumentError value_gradient_and_hessian!!(cache, sum, x, y)
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

            @testset "repeated mutable argument shares one gradient" begin
                # `f(x, x)` mutates through one slot and reads through the other, so the aliased
                # gradient is [4,2,2]; treating the slots as independent gives [1,1,1] and [2,1,1],
                # which sum to [3,2,2]. Seeds were built per argument, so the two slots got separate
                # tangent storage and the reverse aliasing invariant (aliased primals share fdata)
                # was broken at the interface boundary.
                f(x, y) = (x[1] += y[1]; sum(x) + sum(y))
                xp = [1.0, 2.0, 3.0]
                cache = prepare_gradient_cache(f, xp, xp)
                xg = [1.0, 2.0, 3.0]
                v, g = Mooncake.value_and_gradient!!(cache, f, xg, xg)
                @test v == 14.0                      # the aliased primal, not 13.0
                @test g[2] === g[3]                  # one storage, per the aliasing invariant
                @test g[2] == [4.0, 2.0, 2.0]
                # Distinct arguments must keep independent gradients.
                cache2 = prepare_gradient_cache(f, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
                _, g2 = Mooncake.value_and_gradient!!(
                    cache2, f, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]
                )
                @test g2[2] !== g2[3]
                @test g2[2] == [1.0, 1.0, 1.0]
                @test g2[3] == [2.0, 1.0, 1.0]
            end

            @testset "aliasing mismatch between preparation and call" begin
                # A prepared cache holds one cotangent buffer per argument, so the aliasing among
                # arguments is part of the shape it was prepared for. Reusing it with different
                # aliasing accumulates into the wrong buffers: prepared-distinct-called-aliased gave
                # [1,1,1] and [2,1,1] where the aliased gradient is [4,2,2], and the converse gave
                # [4,2,2] for BOTH slots where they should be independent. Neither is visible from
                # types or sizes, so both directions are rejected.
                f(x, y) = (x[1] += y[1]; sum(x) + sum(y))
                x0 = [1.0, 2.0, 3.0]
                distinct_cache = prepare_gradient_cache(f, copy(x0), copy(x0))
                xg = copy(x0)
                @test_throws Mooncake.PreparedCacheError Mooncake.value_and_gradient!!(
                    distinct_cache, f, xg, xg
                )
                xp = copy(x0)
                aliased_cache = prepare_gradient_cache(f, xp, xp)
                @test_throws Mooncake.PreparedCacheError Mooncake.value_and_gradient!!(
                    aliased_cache, f, copy(x0), copy(x0)
                )
                # Matching aliasing keeps working in both directions.
                xg2 = copy(x0)
                _, ga = Mooncake.value_and_gradient!!(aliased_cache, f, xg2, xg2)
                @test ga[2] == [4.0, 2.0, 2.0]
                _, gd = Mooncake.value_and_gradient!!(distinct_cache, f, copy(x0), copy(x0))
                @test gd[2] == [1.0, 1.0, 1.0]
                @test gd[3] == [2.0, 1.0, 1.0]

                # The same mismatch with each array wrapped in a one-tuple. The wrapper is
                # immutable, so comparing only the top-level tangents saw nothing and this
                # returned [3,2,2] against an aliased truth of [4,2,2], silently.
                # `_mutable_tangent_paths` finds the nested position from the type, so the
                # comparison costs a `===` per call rather than a traversal.
                g(t, u) = (t[1][1] += u[1][1]; sum(t[1]) + sum(u[1]))
                nested_distinct = prepare_gradient_cache(g, (copy(x0),), (copy(x0),))
                tg = (copy(x0),)
                @test_throws Mooncake.PreparedCacheError Mooncake.value_and_gradient!!(
                    nested_distinct, g, tg, tg
                )
                tp = (copy(x0),)
                nested_aliased = prepare_gradient_cache(g, tp, tp)
                @test_throws Mooncake.PreparedCacheError Mooncake.value_and_gradient!!(
                    nested_aliased, g, (copy(x0),), (copy(x0),)
                )
                # Matching aliasing keeps working through the wrapper too.
                tg2 = (copy(x0),)
                _, gna = Mooncake.value_and_gradient!!(nested_aliased, g, tg2, tg2)
                @test gna[2][1] == [4.0, 2.0, 2.0]
                @test gna[2][1] === gna[3][1]
                _, gnd = Mooncake.value_and_gradient!!(
                    nested_distinct, g, (copy(x0),), (copy(x0),)
                )
                @test gnd[2][1] == [1.0, 1.0, 1.0]
                @test gnd[3][1] == [2.0, 1.0, 1.0]
            end

            @testset "both modes refuse an input with no concrete representation" begin
                # A NamedTuple with an abstract field widens to a non-concrete `dual_type` and
                # `tangent_type` alike, and the slot wrappers are invariant in that parameter, so
                # no annotation works in either mode. Both checks replace a `TypeError` naming
                # internal slot types — reverse used to report only that.
                NTA = NamedTuple{(:a,),Tuple{Any}}
                nt_field(t) = t.a * 2.0
                @test_throws ArgumentError Mooncake.prepare_derivative_cache(
                    nt_field, NTA((1.0,))
                )
                @test_throws ArgumentError Mooncake.prepare_gradient_cache(
                    nt_field, NTA((1.0,))
                )
                @test_throws ArgumentError Mooncake.prepare_pullback_cache(
                    nt_field, NTA((1.0,))
                )
                # `Mooncake.TestResources.Foo` has the same abstract field (`x::Real`) but keeps its
                # declared field type, so it is supported — what the error tells the caller to use.
                struct_field(s) = s.x * 2.0
                foo = Mooncake.TestResources.Foo(1.0)
                cache = Mooncake.prepare_derivative_cache(struct_field, foo)
                v, _ = Mooncake.value_and_derivative!!(
                    cache,
                    (struct_field, Mooncake.NoTangent()),
                    (foo, Mooncake.zero_tangent(foo)),
                )
                @test v == 2.0
            end

            @testset "gradient and shared storage agree across versions" begin
                # Two containers over one buffer. 1.10 used to RETURN d/t[1]=1.0, d/t[2]=2.0 where
                # the shared buffer's derivative is 3.0 per element -- an answer no perturbation can
                # produce -- because its tangents, unlike 1.11+'s, do not alias, so a tangent-keyed
                # check saw nothing to refuse.
                a_v = collect(1.0:6.0)
                g_v = t -> sum(t[1]) + 2 * sum(t[2])
                c_v = Mooncake.prepare_derivative_cache(g_v, (a_v, reshape(a_v, 2, 3)))
                @test_throws ArgumentError Mooncake.value_and_gradient!!(
                    c_v, g_v, (a_v, reshape(a_v, 2, 3))
                )
                # The mirror image: positions sharing a NON-differentiable buffer contribute no
                # dofs, so there is nothing to scale by a count and nothing to refuse. 1.11+ threw
                # here, rejecting a gradient it computes correctly.
                n_v = collect(1:6)
                h_v = (x, m, r) -> sum(x) * (length(m) + length(r))
                c_n = Mooncake.prepare_derivative_cache(
                    h_v, collect(1.0:6.0), n_v, reshape(n_v, 2, 3)
                )
                v_n, g_n = Mooncake.value_and_gradient!!(
                    c_n, h_v, collect(1.0:6.0), n_v, reshape(n_v, 2, 3)
                )
                @test v_n == sum(1.0:6.0) * 12
                @test g_n[2] ≈ fill(12.0, 6)
                # The same sharing NESTED IN A STRUCT. 1.10 reads the sharing off the primals, and
                # walked only arrays and tuples, so a struct holding two positions over one buffer
                # stayed silent there and returned d/u = 1.0, d/v = 2.0 where the shared buffer's
                # derivative is 3.0 per element.
                a_s = collect(1.0:6.0)
                g_s = p -> sum(p.u) + 2 * sum(p.v)
                p_s = StructuredPair(a_s, reshape(a_s, 2, 3))
                c_s = Mooncake.prepare_derivative_cache(g_s, p_s)
                @test_throws ArgumentError Mooncake.value_and_gradient!!(c_s, g_s, p_s)
            end

            @static if VERSION >= v"1.11-"
                @testset "forward gradient refuses inputs sharing backing storage" begin
                    # `a` and `reshape(a)` are DISTINCT objects over ONE `Memory`, which `dof`'s
                    # identity-keyed de-duplication cannot see, so the dof comparison agrees with
                    # the per-position sum. The gradient buffers alias all the same and the sweep
                    # writes each dof exactly once, so one position's contribution overwrote the
                    # other's: 2.0 where reverse gives 3.0 for `sum(t[1]) + 2*sum(t[2])`.
                    a_st = collect(1.0:6.0)
                    g_st = t -> sum(t[1]) + 2 * sum(t[2])
                    # An `Array` beside its own backing `Memory` is the same sharing reached a
                    # different way: the array tangent's `Memory` IS the `Memory`'s tangent.
                    v_st = collect(1.0:3.0)
                    for t_st in (
                        (a_st, reshape(a_st, 2, 3)),
                        (reshape(a_st, 2, 3), a_st),
                        (v_st, v_st.ref.mem),
                    )
                        c_st = Mooncake.prepare_derivative_cache(g_st, t_st)
                        @test_throws ArgumentError Mooncake.value_and_gradient!!(
                            c_st, g_st, t_st
                        )
                    end
                    # The same sharing one level down, which the traversal used to walk past:
                    # inside an array ELEMENT, and behind a `Ref`, whose tangent field is a
                    # `PossiblyUninitTangent`. Both returned 4.0 for a per-position derivative of
                    # 1.0, with no refusal.
                    g_el = t -> sum(sum, t[1]) + 2 * sum(sum, t[2])
                    t_el = ([a_st], [reshape(a_st, 2, 3)])
                    c_el = Mooncake.prepare_derivative_cache(g_el, t_el)
                    @test_throws ArgumentError Mooncake.value_and_gradient!!(
                        c_el, g_el, t_el
                    )
                    g_ref = t -> sum(t[1][]) + 2 * sum(t[2])
                    t_ref = (Base.RefValue(a_st), reshape(a_st, 2, 3))
                    c_ref = Mooncake.prepare_derivative_cache(g_ref, t_ref)
                    @test_throws ArgumentError Mooncake.value_and_gradient!!(
                        c_ref, g_ref, t_ref
                    )
                    # Distinct storage is unaffected. So are two NON-overlapping views, whose
                    # tangents get their own parents; two EMPTY arrays, which all share Julia's one
                    # global empty `Memory` and so would look aliased on identity alone; and the
                    # SAME `Memory` at both positions, which the aliasing cache already handles by
                    # giving them one tangent object.
                    c_ok = Mooncake.prepare_derivative_cache(
                        g_st, (collect(1.0:6.0), collect(1.0:6.0))
                    )
                    _, g_ok = Mooncake.value_and_gradient!!(
                        c_ok, g_st, (collect(1.0:6.0), collect(1.0:6.0))
                    )
                    @test g_ok[2][1] ≈ ones(6)
                    @test g_ok[2][2] ≈ 2 .* ones(6)
                    m_ok = collect(1.0:3.0).ref.mem
                    for t_ok in (
                        (view(a_st, 1:3), view(a_st, 4:6)),
                        (Float64[], Float64[]),
                        (m_ok, m_ok),
                    )
                        @test Mooncake.prepare_derivative_cache(g_st, t_ok) isa Any
                    end
                end
            end

            @testset "forward gradient refuses a repeated mutable argument" begin
                # The gradient is assembled from one standard-basis dof range per argument, which
                # cannot represent a repeated argument: the seeds are per-argument, so the seeded
                # primal stops aliasing and a mutating `f` reported the value for DISTINCT
                # arguments (13.0 instead of 14.0). Sharing the seed slot instead double-counts.
                # `value_and_derivative!!` handles it, since the caller shares one tangent.
                f(x, y) = (x[1] += y[1]; sum(x) + sum(y))
                x0 = [1.0, 2.0, 3.0]
                xp = copy(x0)
                cache = Mooncake.prepare_derivative_cache(f, xp, xp)
                xg = copy(x0)
                @test_throws ArgumentError Mooncake.value_and_gradient!!(cache, f, xg, xg)
                # The supported route gives the aliased truth: value 14.0, all-ones direction 8.0.
                xs = copy(x0)
                ss = [1.0, 1.0, 1.0]
                v, d = Mooncake.value_and_derivative!!(
                    cache, (f, Mooncake.zero_tangent(f)), (xs, ss), (xs, ss)
                )
                @test v == 14.0
                @test d == 8.0
                # Distinct arguments are unaffected, and an immutable repeated argument is fine
                # (a scalar cannot be mutated, so there is no aliasing to represent).
                g(a, b) = sum(a .* b)
                cg = Mooncake.prepare_derivative_cache(g, [1.0, 2.0], [3.0, 4.0])
                _, gg = Mooncake.value_and_gradient!!(cg, g, [1.0, 2.0], [3.0, 4.0])
                @test gg[2] == [3.0, 4.0]
                @test gg[3] == [1.0, 2.0]
                h(a, b) = a * b
                ch = Mooncake.prepare_derivative_cache(h, 2.0, 3.0)
                @test Mooncake.value_and_gradient!!(ch, h, 4.0, 4.0)[1] == 16.0
                # A repeated MUTABLE argument with no differentiable dof is representable: it has
                # no gradient to assemble, so the dof ranges still line up with the arguments.
                # `ismutabletype` alone refuses it, and reverse mode accepts it.
                k(a, b, v) = (a.a + b.a) * sum(v)
                ck = IntScaler(3)
                cache_k = Mooncake.prepare_derivative_cache(k, ck, ck, [1.0, 2.0])
                _, gk = Mooncake.value_and_gradient!!(cache_k, k, ck, ck, [1.0, 2.0])
                @test gk[4] == [6.0, 6.0]
            end

            @testset "forward gradient refuses inputs sharing storage across positions" begin
                # `_check_gradient_arg_aliasing` sees only a repeated top-level MUTABLE argument,
                # and only the arguments — never `f`. A callable holding an array also passed as
                # an argument shares storage at depth, so the shared leaf was differentiated once
                # per position and came back scaled by that count: [4,8,12] for a gradient that is
                # [2,4,6]. Refuse, as for the repeated argument.
                w = [1.0, 2.0, 3.0]
                aliased = Mooncake.prepare_derivative_cache(FwdAliasHolder(w), w)
                @test_throws ArgumentError Mooncake.value_and_gradient!!(
                    aliased, FwdAliasHolder(w), w
                )
                # `value_and_derivative!!` shares the cache and handles aliased inputs, so the
                # refusal must not have moved into construction.
                # ONE tangent object for the one shared storage: the holder's field and the
                # argument tangent are the same array. Two equal but distinct arrays are refused,
                # the same identity rule the repeated-argument check applies at the top level —
                # a storage carries one direction, and two objects cannot be shown to agree
                # without a value walk.
                dw = fill(1.0, 3)
                dh = Mooncake.Tangent((; v=dw))
                v, d = Mooncake.value_and_derivative!!(
                    aliased, (FwdAliasHolder(w), dh), (w, dw)
                )
                @test v == sum(abs2, w)
                @test d == 2 * sum(w)                    # both occurrences move
                # No over-refusal: distinct storage still matches reverse mode.
                distinct = Mooncake.prepare_derivative_cache(FwdAliasHolder(w), copy(w))
                _, gd = Mooncake.value_and_gradient!!(distinct, FwdAliasHolder(w), copy(w))
                @test gd[2] == w
                # Sharing WITHIN one argument is representable (one dof range covers both
                # occurrences) and must keep working.
                intra(t) = sum(t.p .* t.q)
                ci = Mooncake.prepare_derivative_cache(intra, FwdAliasPair(w, w))
                _, gi = Mooncake.value_and_gradient!!(ci, intra, FwdAliasPair(w, w))
                @test gi[2].fields.p == 2 .* w
            end

            @testset "empty-cache reused at non-empty input" begin
                f(x) = sum(x .^ 2)
                cache = prepare_hessian_cache(f, Float64[])
                @test_throws ArgumentError value_gradient_and_hessian!!(
                    cache, f, [1.0, 2.0]
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
