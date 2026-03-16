using LinearAlgebra
using Mooncake
using Mooncake:
    Dual,
    DefaultCtx,
    ForwardMode,
    ReverseMode,
    @is_primitive,
    @reverse_from_forward,
    prepare_pullback_cache,
    value_and_pullback!!,
    primal,
    tangent
using Mooncake.TestUtils: test_rule, count_allocs
using StableRNGs
using Test

## Working

# we define rules for fi that pretend fi_true was run instead, to check that Mooncake doesn't recurse inside the function
# we still use fi_true in the Mooncake.TestUtils.test_rule

cube(x) = x^3
fourthpow(x) = x^4

f1(x::Float64) = sin(x)
f1_true(x::Float64) = cube(x)

f2(x::Float64) = [sin(x), cos(x)]
f2_true(x::Float64) = [cube(x), fourthpow(x)]

f3(x::Vector{Float64}) = sum(sin, x)
f3_true(x::Vector{Float64}) = sum(cube, x)

f4(x::Vector{Float64}) = map(sin, x)
f4_true(x::Vector{Float64}) = map(cube, x)

# not primitives, we'll rely on derived forward rules
f5(x::AbstractArray{<:Real}, y::Real) = sin(y) * sum(abs2, x)
f6(y, x) = f5(x, y)

# fperf_v: array → scalar with an allocation-free frule!!, used in perf regression tests.
# With an alloc-free frule!!, any allocations detected in the round-trip indicate a
# regression in the ForwardModeRRule!! wrapper itself.
fperf_v(x::Vector{Float64}) = sum(cube, x)

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f1),Float64}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f1_true),Float64}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f2),Float64}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f2_true),Float64}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f3),Vector{Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f3_true),Vector{Float64}}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f4),Vector{Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f4_true),Vector{Float64}}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(fperf_v),Vector{Float64}}

function Mooncake.frule!!(
    ::Union{Dual{typeof(f1)},Dual{typeof(f1_true)}}, x_dual::Dual{Float64}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y = x^3
    dy = 3x^2 * dx
    return Dual(y, dy)
end

function Mooncake.frule!!(
    ::Union{Dual{typeof(f2)},Dual{typeof(f2_true)}}, x_dual::Dual{Float64}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y, z = x^3, x^4
    dy, dz = 3x^2 * dx, 4x^3 * dx
    return Dual([y, z], [dy, dz])
end

function Mooncake.frule!!(
    ::Union{Dual{typeof(f3)},Dual{typeof(f3_true)}}, x_dual::Dual{Vector{Float64}}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y = sum(_x -> _x^3, x)
    dy = dot(map(_x -> 3 * _x^2, x), dx)
    return Dual(y, dy)
end

function Mooncake.frule!!(
    ::Union{Dual{typeof(f4)},Dual{typeof(f4_true)}}, x_dual::Dual{Vector{Float64}}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y = map(_x -> _x^3, x)
    J = diagm(map(_x -> 3 * _x^2, x))
    dy = J * dx
    return Dual(y, dy)
end

function Mooncake.frule!!(::Dual{typeof(fperf_v)}, x_dual::Dual{Vector{Float64}})
    x, dx = primal(x_dual), tangent(x_dual)
    return Dual(sum(cube, x), sum(i -> 3x[i]^2 * dx[i], eachindex(x)))
end

@reverse_from_forward Tuple{typeof(f1),Float64}
@reverse_from_forward Tuple{typeof(f1_true),Float64}

@reverse_from_forward Tuple{typeof(f2),Float64}
@reverse_from_forward Tuple{typeof(f2_true),Float64}

@reverse_from_forward Tuple{typeof(f3),Vector{Float64}}
@reverse_from_forward Tuple{typeof(f3_true),Vector{Float64}}

@reverse_from_forward Tuple{typeof(f4),Vector{Float64}}
@reverse_from_forward Tuple{typeof(f4_true),Vector{Float64}}

@reverse_from_forward Tuple{typeof(fperf_v),Vector{Float64}}

# use abstract types
@reverse_from_forward Tuple{typeof(f5),AbstractArray{<:AbstractFloat},AbstractFloat}
@reverse_from_forward Tuple{typeof(f6),AbstractFloat,AbstractArray{<:AbstractFloat}}

# is_primitive=false because test_rule's is_primitive=true asserts the rule is exactly rrule!!,
# but @reverse_from_forward registers a ForwardModeRRule!! via build_primitive_rrule. With
# is_primitive=false the type-equality check is skipped while the rule is still built and
# numerically validated via finite differences.
@testset verbose = true "Working cases" begin
    @testset "Scalar to scalar" begin
        x = 5.0
        dy = 7.0
        cache = prepare_pullback_cache(f1, zero(x))
        val, pb = value_and_pullback!!(cache, dy, f1, x)
        @test val == x^3
        @test pb[1] == Mooncake.NoTangent()
        @test pb[2] == dy * 3x^2
        test_rule(StableRNG(63), f1_true, x; is_primitive=false, perf_flag=:allocs)
    end
    @testset "Scalar to array" begin
        x = 5.0
        dy = [7.0, 11.0]
        cache = prepare_pullback_cache(f2, zero(x))
        val, pb = value_and_pullback!!(cache, dy, f2, x)
        @test val == [x^3, x^4]
        @test pb[1] == Mooncake.NoTangent()
        @test pb[2] == dy[1] * 3x^2 + dy[2] * 4x^3
        test_rule(StableRNG(63), f2_true, x; is_primitive=false)
    end
    @testset "Array to scalar" begin
        x = [5.0, 13.0]
        dy = 7.0
        cache = prepare_pullback_cache(f3, zero(x))
        val, pb = value_and_pullback!!(cache, dy, f3, x)
        @test val == sum(cube, x)
        @test pb[1] == Mooncake.NoTangent()
        @test pb[2] == dy .* map(_x -> 3 * _x^2, x)
        test_rule(StableRNG(63), f3_true, x; is_primitive=false)
    end
    @testset "Array to array" begin
        x = [5.0, 13.0]
        dy = [7.0, 11.0]
        cache = prepare_pullback_cache(f4, zero(x))
        val, pb = value_and_pullback!!(cache, dy, f4, x)
        @test val == map(cube, x)
        @test pb[1] == Mooncake.NoTangent()
        @test pb[2] ≈ dy .* map(_x -> 3 * _x^2, x)
        test_rule(StableRNG(63), f4_true, x; is_primitive=false)
    end
    @testset "Multiple arguments" begin
        x = [5.0, 13.0]
        y = 11.0
        dz = 17.0
        cache = prepare_pullback_cache(f5, zero(x), zero(y))
        val, pb = value_and_pullback!!(cache, dz, f5, x, y)
        @test val == sin(y) * sum(abs2, x)
        @test pb[1] == Mooncake.NoTangent()
        test_rule(StableRNG(63), f5, x, y; is_primitive=false)
        test_rule(StableRNG(63), f6, y, x; is_primitive=false)
    end
end;

## Allocations

@testset verbose = true "Allocations" begin
    @testset "Array input, scalar output: zero allocations" begin
        # fperf_v has an alloc-free frule!!, so any allocations detected indicate a
        # regression in ForwardModeRRule!! itself. perf_flag=:allocs warms up then counts.
        test_rule(StableRNG(63), fperf_v, [3.0, 5.0]; is_primitive=false, perf_flag=:allocs)
    end
    @testset "Buffer resize: correctness with variable-size arrays" begin
        # Regression for the size-mismatch bug: the same cached rule must produce
        # correct gradients when called with arrays of different sizes.
        cache = prepare_pullback_cache(fperf_v, zeros(2))
        _, pb2 = value_and_pullback!!(cache, 1.0, fperf_v, [1.0, 2.0])
        @test pb2[2] ≈ 3 .* [1.0, 2.0] .^ 2
        _, pb3 = value_and_pullback!!(cache, 1.0, fperf_v, [1.0, 2.0, 3.0])
        @test pb3[2] ≈ 3 .* [1.0, 2.0, 3.0] .^ 2
        # After resizing, steady-state calls should be allocation-free.
        x3 = [1.0, 2.0, 3.0]
        value_and_pullback!!(cache, 1.0, fperf_v, x3)  # warm up new size
        @test count_allocs(value_and_pullback!!, cache, 1.0, fperf_v, x3) == 0
    end
end;

## Failing

# forgot macro
f5_nomacro(x, y) = f5(x, y)
# unsupported types
f_tup(x::Tuple{Float64}) = x[1]

@reverse_from_forward(Tuple{typeof(f_tup),Tuple{Float64}})

struct Multiplier
    a::Float64
end
(m::Multiplier)(x) = m.a * x

@reverse_from_forward Tuple{Multiplier,Float64}

@testset verbose = true "Failing cases" begin
    @testset "Wrong expression (not a Tuple{...})" begin
        @test_throws "LoadError: ArgumentError: The provided signature must be of the form `Tuple{typeof(f), ...}`." @eval @reverse_from_forward(
            :(f3(x::Float64))
        )
    end
    @testset "Closures" begin
        @test_throws "ArgumentError: `Mooncake.@reverse_from_forward` does not support functions which close over data." prepare_gradient_cache(
            Multiplier(2.0), 5.0
        )
    end
    @testset "Unsupported input types" begin
        @test_throws ArgumentError prepare_pullback_cache(f_tup, (5.0,))
    end
    @testset "Forgot macro" begin
        @test_throws "MethodError: no method matching rrule!!" prepare_pullback_cache(
            f5_nomacro, [1.0], 2.0
        )
    end
    @testset "Wrong signature (no method)" begin
        @test_throws "MethodError: no method matching f1(::Float32)" prepare_pullback_cache(
            f1, Float32(1.0)
        )
    end
end;
