# Resources for @from_chainrules / @from_rrule tests
module ToolsForRulesResources

# Note: do not `using Mooncake` in this module to ensure that all of the macros work
# correctly if `Mooncake` is not in scope.
using ChainRulesCore, LinearAlgebra
using Base: IEEEFloat
using Mooncake:
    @mooncake_overlay,
    @zero_derivative,
    @from_rrule,
    MinimalCtx,
    DefaultCtx,
    @from_chainrules,
    ForwardMode,
    ReverseMode,
    NoFData,
    NoRData

const CRC = ChainRulesCore

local_function(x) = 3x
overlay_tester(x) = 2x
@mooncake_overlay overlay_tester(x) = local_function(x)

zero_tester(x) = 0
@zero_derivative MinimalCtx Tuple{typeof(zero_tester),Float64}

vararg_zero_tester(x...) = 0
@zero_derivative MinimalCtx Tuple{typeof(vararg_zero_tester),Vararg}

typed_vararg_zero_tester(x::Float64...) = 0
@zero_derivative MinimalCtx Tuple{typeof(typed_vararg_zero_tester),Vararg{Float64}}

counted_vararg_zero_tester(x::Float64...) = 0
@zero_derivative MinimalCtx Tuple{
    typeof(counted_vararg_zero_tester),Vararg{Float64,N}
} where {N}

nested_vararg_zero_tester(x::Vector{Float64}...) = 0
@zero_derivative MinimalCtx Tuple{
    typeof(nested_vararg_zero_tester),Vararg{Vector{Float64},N}
} where {N}

mixed_vararg_zero_tester(n::Int, x::Float64...) = 0
@zero_derivative MinimalCtx Tuple{typeof(mixed_vararg_zero_tester),Int,Vararg{Float64}}
@zero_derivative MinimalCtx Tuple{
    typeof(mixed_vararg_zero_tester),Int,Vararg{Float64,N}
} where {N}

zero_tester_forward_only(x) = 0
@zero_derivative MinimalCtx Tuple{typeof(zero_tester_forward_only),Float64} ForwardMode

zero_tester_reverse_only(x) = 0
@zero_derivative MinimalCtx Tuple{typeof(zero_tester_reverse_only),Float64} ReverseMode

# Test case with isbits data.

bleh(x::Float64, y::Int) = x * y

CRC.frule((_, dx, _), ::typeof(bleh), x::Float64, y::Int) = x * y, dx * y

function CRC.rrule(::typeof(bleh), x::Float64, y::Int)
    return x * y, dz -> (CRC.NoTangent(), dz * y, CRC.NoTangent())
end

@from_chainrules DefaultCtx Tuple{typeof(bleh),Float64,Int} false

# Test case with heap-allocated input.

test_sum(x) = sum(x)

CRC.frule((_, dx), ::typeof(test_sum), x::AbstractArray{<:Real}) = sum(x), sum(dx)

function CRC.rrule(::typeof(test_sum), x::AbstractArray{<:Real})
    test_sum_pb(dy::Real) = CRC.NoTangent(), fill(dy, size(x))
    return test_sum(x), test_sum_pb
end

@from_chainrules DefaultCtx Tuple{typeof(test_sum),Array{<:Base.IEEEFloat}} false

# Test case with heap-allocated output.

test_scale(x::Real, y::AbstractVector{<:Real}) = x * y

function CRC.frule((_, dx, dy), ::typeof(test_scale), x::Real, y::AbstractVector{<:Real})
    return x * y, dx * y + x * dy
end

function CRC.rrule(::typeof(test_scale), x::Real, y::AbstractVector{<:Real})
    test_scale_pb(dout::AbstractVector{<:Real}) = CRC.NoTangent(), dot(dout, y), dout * x
    return x * y, test_scale_pb
end

@from_chainrules(
    DefaultCtx, Tuple{typeof(test_scale),Base.IEEEFloat,Vector{<:Base.IEEEFloat}}, false
)

# Test case with non-differentiable type as output.

test_nothing() = nothing

CRC.frule(_, ::typeof(test_nothing)) = (nothing, CRC.NoTangent())

function CRC.rrule(::typeof(test_nothing))
    test_nothing_pb(::CRC.NoTangent) = (CRC.NoTangent(),)
    return nothing, test_nothing_pb
end

@from_chainrules DefaultCtx Tuple{typeof(test_nothing)} false

# Test case in which ChainRulesCore returns a tangent which is of the "wrong" type from the
# perspective of Mooncake.jl. In this instance, some kind of error should be thrown, rather
# than it being possible for the error to propagate.

test_bad_rdata(x::Real) = 5x

function CRC.rrule(::typeof(test_bad_rdata), x::Float64)
    test_bad_rdata_pb(dy::Float64) = CRC.NoTangent(), Float32(dy * 5)
    return 5x, test_bad_rdata_pb
end

@from_rrule DefaultCtx Tuple{typeof(test_bad_rdata),Float64} false

# Test case for rule with diagonal dispatch.
test_add(x, y) = x + y
function CRC.rrule(::typeof(test_add), x, y)
    test_add_pb(dout) = CRC.NoTangent(), dout, dout
    return x + y, test_add_pb
end
@from_rrule DefaultCtx Tuple{typeof(test_add),T,T} where {T<:IEEEFloat} false

# Test case for rule with non-differentiable kwargs.
test_kwargs(x; y::Bool=false) = y ? x : 2x

function CRC.frule((_, dx), ::typeof(test_kwargs), x::Float64; y::Bool=false)
    return test_kwargs(x; y), y ? dx : 2dx
end

function CRC.rrule(::typeof(test_kwargs), x::Float64; y::Bool=false)
    test_kwargs_pb(dz::Float64) = CRC.NoTangent(), y ? dz : 2dz
    return y ? x : 2x, test_kwargs_pb
end

@from_chainrules(DefaultCtx, Tuple{typeof(test_kwargs),Float64}, true)

# Test case for rule with differentiable types used in a non-differentiable way.
test_kwargs_conditional(x; y::Float64=1.0) = y > 0 ? x : 2x

function CRC.frule((_, dx), ::typeof(test_kwargs_conditional), x::Float64; y::Float64=1.0)
    return test_kwargs_conditional(x; y), y > 0 ? dx : 2dx
end

function CRC.rrule(::typeof(test_kwargs_conditional), x::Float64; y::Float64=1.0)
    test_kwargs_cond_pb(dz::Float64) = CRC.NoTangent(), y > 0 ? dz : 2dz
    return y > 0 ? x : 2x, test_kwargs_cond_pb
end

@from_chainrules(DefaultCtx, Tuple{typeof(test_kwargs_conditional),Float64}, true)

# Test case for mode-specific @from_chainrules.

fwd_only_chainrules(x::Float64) = 3x
rev_only_chainrules(x::Float64) = 4x

CRC.frule((_, dx), ::typeof(fwd_only_chainrules), x::Float64) = (3x, 3dx)
function CRC.rrule(::typeof(fwd_only_chainrules), x::Float64)
    pb(dy::Float64) = (CRC.NoTangent(), 3dy)
    return 3x, pb
end

CRC.frule((_, dx), ::typeof(rev_only_chainrules), x::Float64) = (4x, 4dx)
function CRC.rrule(::typeof(rev_only_chainrules), x::Float64)
    pb(dy::Float64) = (CRC.NoTangent(), 4dy)
    return 4x, pb
end

@from_chainrules DefaultCtx Tuple{typeof(fwd_only_chainrules),Float64} false ForwardMode
@from_chainrules DefaultCtx Tuple{typeof(rev_only_chainrules),Float64} false ReverseMode

end

# Resources for @from_forward tests
module FromForwardResources

using LinearAlgebra
using Base: IEEEFloat
import Mooncake: frule!!

using Mooncake:
    Dual,
    DefaultCtx,
    ForwardMode,
    ReverseMode,
    @is_primitive,
    @from_forward,
    primal,
    tangent

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

# matrix inputs: element-wise map (matrix → matrix)
f4m(x::Matrix{Float64}) = map(sin, x)

# not primitives, we'll rely on derived forward rules
f5(x::AbstractArray{<:Real}, y::Real) = sin(y) * sum(abs2, x)
f6(y, x) = f5(x, y)

# fperf_v: array → scalar with an allocation-free frule!!, used in perf regression tests.
fperf_v(x::Vector{Float64}) = sum(cube, x)

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f1),Float64}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f1_true),Float64}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f2),Float64}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f2_true),Float64}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f3),Vector{Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f3_true),Vector{Float64}}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f4),Vector{Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f4_true),Vector{Float64}}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f4m),Matrix{Float64}}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(fperf_v),Vector{Float64}}

# We define rules for fi that pretend fi_true was run instead, to check that Mooncake
# doesn't recurse inside the function; we still use fi_true in TestUtils.test_rule.
function frule!!(::Union{Dual{typeof(f1)},Dual{typeof(f1_true)}}, x_dual::Dual{Float64})
    x, dx = primal(x_dual), tangent(x_dual)
    y = x^3
    dy = 3x^2 * dx
    return Dual(y, dy)
end

function frule!!(::Union{Dual{typeof(f2)},Dual{typeof(f2_true)}}, x_dual::Dual{Float64})
    x, dx = primal(x_dual), tangent(x_dual)
    y, z = x^3, x^4
    dy, dz = 3x^2 * dx, 4x^3 * dx
    return Dual([y, z], [dy, dz])
end

function frule!!(
    ::Union{Dual{typeof(f3)},Dual{typeof(f3_true)}}, x_dual::Dual{Vector{Float64}}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y = sum(_x -> _x^3, x)
    dy = dot(map(_x -> 3 * _x^2, x), dx)
    return Dual(y, dy)
end

function frule!!(
    ::Union{Dual{typeof(f4)},Dual{typeof(f4_true)}}, x_dual::Dual{Vector{Float64}}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y = map(_x -> _x^3, x)
    J = diagm(map(_x -> 3 * _x^2, x))
    dy = J * dx
    return Dual(y, dy)
end

function frule!!(::Dual{typeof(f4m)}, x_dual::Dual{Matrix{Float64}})
    x, dx = primal(x_dual), tangent(x_dual)
    return Dual(map(sin, x), map((_x, _dx) -> cos(_x) * _dx, x, dx))
end

function frule!!(::Dual{typeof(fperf_v)}, x_dual::Dual{Vector{Float64}})
    x, dx = primal(x_dual), tangent(x_dual)
    return Dual(sum(cube, x), sum(i -> 3x[i]^2 * dx[i], eachindex(x)))
end

@from_forward Tuple{typeof(f1),Float64}
@from_forward Tuple{typeof(f1_true),Float64}

@from_forward Tuple{typeof(f2),Float64}
@from_forward Tuple{typeof(f2_true),Float64}

@from_forward Tuple{typeof(f3),Vector{Float64}}
@from_forward Tuple{typeof(f3_true),Vector{Float64}}

@from_forward Tuple{typeof(f4),Vector{Float64}}
@from_forward Tuple{typeof(f4_true),Vector{Float64}}

@from_forward Tuple{typeof(f4m),Matrix{Float64}}

@from_forward Tuple{typeof(fperf_v),Vector{Float64}}

@from_forward Tuple{typeof(f5),AbstractArray{<:AbstractFloat},AbstractFloat}
@from_forward Tuple{typeof(f6),AbstractFloat,AbstractArray{<:AbstractFloat}}

# forgot macro: marked as a reverse-mode primitive (so the interpreter won't
# recurse) but @from_forward was never called, leaving no rrule!! or
# build_primitive_rrule. prepare_pullback_cache must therefore throw.
f_forgot_macro(x::Float64) = x^2
@is_primitive DefaultCtx ReverseMode Tuple{typeof(f_forgot_macro),Float64}

# unsupported types
f_tup(x::Tuple{Float64}) = x[1]
@from_forward(Tuple{typeof(f_tup),Tuple{Float64}})

struct Multiplier
    a::Float64
end
(m::Multiplier)(x) = m.a * x
@from_forward Tuple{Multiplier,Float64}

end

@testset "tools_for_rules" begin
    @testset "mooncake_overlay" begin
        f = ToolsForRulesResources.overlay_tester
        rule = Mooncake.build_rrule(Tuple{typeof(f),Float64})
        @test value_and_gradient!!(rule, f, 5.0) == (15.0, (NoTangent(), 3.0))
    end
    @testset "zero_derivative" begin
        test_rule(
            sr(123),
            ToolsForRulesResources.zero_tester,
            5.0;
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )
        test_rule(
            sr(123),
            ToolsForRulesResources.vararg_zero_tester,
            5.0,
            4.0;
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )
        test_rule(
            sr(123),
            ToolsForRulesResources.typed_vararg_zero_tester,
            5.0,
            4.0;
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )
        test_rule(
            sr(123),
            ToolsForRulesResources.counted_vararg_zero_tester,
            5.0,
            4.0;
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )
        test_rule(
            sr(123),
            ToolsForRulesResources.counted_vararg_zero_tester;
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )
        test_rule(
            sr(123),
            ToolsForRulesResources.nested_vararg_zero_tester,
            [5.0],
            [4.0];
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )
        test_rule(
            sr(123),
            ToolsForRulesResources.mixed_vararg_zero_tester,
            3,
            5.0,
            4.0;
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )
        test_rule(
            sr(123),
            ToolsForRulesResources.mixed_vararg_zero_tester,
            3;
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )

        @test_throws(
            r"@zero_derivative: `Vararg` may only appear as the last element of",
            Mooncake.@zero_derivative MinimalCtx Tuple{Vararg,typeof(zero_tester)}
        )

        world = Base.get_world_counter()
        perf_flag = :stability_and_allocs
        @testset "forward mode only" begin
            sig = Tuple{typeof(ToolsForRulesResources.zero_tester_forward_only),Float64}
            @test is_primitive(MinimalCtx, ForwardMode, sig, world)
            @test !is_primitive(MinimalCtx, ReverseMode, sig, world)
            args = (ToolsForRulesResources.zero_tester_forward_only, 5.0)
            test_rule(sr(123), args...; is_primitive=true, perf_flag, mode=ForwardMode)
        end
        @testset "reverse mode only" begin
            sig = Tuple{typeof(ToolsForRulesResources.zero_tester_reverse_only),Float64}
            @test !is_primitive(MinimalCtx, ForwardMode, sig, world)
            @test is_primitive(MinimalCtx, ReverseMode, sig, world)
            args = (ToolsForRulesResources.zero_tester_reverse_only, 5.0)
            test_rule(sr(123), args...; is_primitive=true, perf_flag, mode=ReverseMode)
        end
    end
    @testset "chain_rules_macro" begin
        @testset "to_cr_tangent" for (t, t_cr) in Any[
            (5.0, 5.0),
            (ones(5), ones(5)),
            (NoTangent(), ChainRulesCore.NoTangent()),
            ((5.0, 4.0), ChainRulesCore.Tangent{Any}(5.0, 4.0)),
            ((a=5.0, b=4.0), ChainRulesCore.Tangent{Any}(; a=5.0, b=4.0)),
            ([ones(5), NoTangent()], [ones(5), ChainRulesCore.NoTangent()]),
            (
                Tangent((a=5.0, b=NoTangent())),
                ChainRulesCore.Tangent{Any}(; a=5.0, b=ChainRulesCore.NoTangent()),
            ),
            (
                MutableTangent((a=5.0, b=ones(3))),
                ChainRulesCore.Tangent{Any}(; a=5.0, b=ones(3)),
            ),
            (PossiblyUninitTangent{Float64}(5.0), 5.0),
            (PossiblyUninitTangent{Vector{Float64}}([5.0]), [5.0]),
            (PossiblyUninitTangent{Vector{Float64}}(), ChainRulesCore.ZeroTangent()),
        ]
            @test Mooncake.to_cr_tangent(t) == t_cr
        end
        @testset "mooncake_tangent($(typeof(p)), $(typeof(t)))" for (p, t) in Any[
            (5, ChainRulesCore.NoTangent()),
            (5.0, 4.0),
            (randn(5), randn(5)),
            ([randn(5)], [randn(5)]),
            ((5.0, 4), (4.0, ChainRulesCore.NoTangent())),
        ]
            @test Mooncake.mooncake_tangent(p, t) isa tangent_type(typeof(p))
        end
        @testset "rules: $(typeof(fargs))" for fargs in Any[
            (ToolsForRulesResources.bleh, 5.0, 4),
            (ToolsForRulesResources.test_sum, ones(5)),
            (ToolsForRulesResources.test_scale, 5.0, randn(3)),
            (ToolsForRulesResources.test_nothing,),
            (Core.kwcall, (y=true,), ToolsForRulesResources.test_kwargs, 5.0),
            (Core.kwcall, (y=false,), ToolsForRulesResources.test_kwargs, 5.0),
            (ToolsForRulesResources.test_kwargs, 5.0),
            (Core.kwcall, (y=-1.0,), ToolsForRulesResources.test_kwargs_conditional, 5.0),
            (Core.kwcall, (y=1.0,), ToolsForRulesResources.test_kwargs_conditional, 5.0),
            (ToolsForRulesResources.test_kwargs_conditional, 5.0),
        ]
            test_rule(sr(1), fargs...; perf_flag=:none, is_primitive=true, mode=ForwardMode)
            test_rule(
                sr(1), fargs...; perf_flag=:stability, is_primitive=true, mode=ReverseMode
            )
        end
        @testset "bad rdata" begin
            f = ToolsForRulesResources.test_bad_rdata
            out, pb!! = Mooncake.rrule!!(zero_fcodual(f), zero_fcodual(3.0))
            @test_throws ArgumentError pb!!(5.0)
        end
        @testset "forward mode only" begin
            world = Base.get_world_counter()
            sig = Tuple{typeof(ToolsForRulesResources.fwd_only_chainrules),Float64}
            @test is_primitive(DefaultCtx, ForwardMode, sig, world)
            @test !is_primitive(DefaultCtx, ReverseMode, sig, world)
            test_rule(
                sr(1),
                ToolsForRulesResources.fwd_only_chainrules,
                5.0;
                perf_flag=:none,
                is_primitive=true,
                mode=ForwardMode,
            )
            rrule_sig = Tuple{
                CoDual{typeof(ToolsForRulesResources.fwd_only_chainrules)},CoDual{Float64}
            }
            @test !hasmethod(rrule!!, rrule_sig)
        end
        @testset "reverse mode only" begin
            world = Base.get_world_counter()
            sig = Tuple{typeof(ToolsForRulesResources.rev_only_chainrules),Float64}
            @test !is_primitive(DefaultCtx, ForwardMode, sig, world)
            @test is_primitive(DefaultCtx, ReverseMode, sig, world)
            test_rule(
                sr(1),
                ToolsForRulesResources.rev_only_chainrules,
                5.0;
                perf_flag=:none,
                is_primitive=true,
                mode=ReverseMode,
            )
            frule_sig = Tuple{
                Dual{typeof(ToolsForRulesResources.rev_only_chainrules)},Dual{Float64}
            }
            @test !hasmethod(Mooncake.frule!!, frule_sig)
        end
        @testset "invalid mode" begin
            bad_mode_exprs = [
                :(Mooncake.@from_chainrules DefaultCtx Tuple{
                    typeof(ToolsForRulesResources.test_sum),Array{<:Base.IEEEFloat}
                } false BadMode),
                :(Mooncake.@from_chainrules DefaultCtx Tuple{
                    typeof(ToolsForRulesResources.test_sum),Array{<:Base.IEEEFloat}
                } false Mooncake.ForwardMode),
            ]
            for expr in bad_mode_exprs
                err = @test_throws LoadError eval(expr)
                @test err.value.error isa ArgumentError
            end
        end

        @testset "increment_and_get_rdata!(f, r, t) specialized dispatches" begin
            f_no = NoFData()
            r_no = NoRData()

            @testset "NoFData - homogeneous Tuple rdata" begin
                # f_no, r, t - NoFData, Tuple{Float64, Float64}, Tangent{Any, <:Tuple}
                r = (rand(), rand())
                dr = (rand(), rand())
                t = ChainRulesCore.Tangent{Any,typeof(r)}(dr)

                result = Mooncake.increment_and_get_rdata!(f_no, r, t)

                @test result isa typeof(r)
                @test result[1] ≈ r[1] + dr[1]
                @test result[2] ≈ r[2] + dr[2]
            end

            @testset "NoFData - heterogeneous Tuple rdata" begin
                # f_no, r, t - NoFData, Tuple{Tuple{Float64,Float64}, Float64}, Tangent{Any, <:Tuple}
                r = ((rand(), rand()), rand())
                dr = ((rand(), rand()), rand())
                t = ChainRulesCore.Tangent{Any,typeof(r)}(dr)

                result = Mooncake.increment_and_get_rdata!(f_no, r, t)

                @test result isa typeof(r)
                @test result[1][1] ≈ r[1][1] + dr[1][1]
                @test result[1][2] ≈ r[1][2] + dr[1][2]
                @test result[2] ≈ r[2] + dr[2]
            end

            @testset "NoRData - homogeneous Tuple fdata" begin
                # f, r_no, t - Tuple{Vector{Float64}, Vector{Float64}}, NoRData, Tangent{Any, <:Tuple}
                f1_orig = rand(3)
                f2_orig = rand(3)
                f = (copy(f1_orig), copy(f2_orig))

                df1, df2 = rand(3), rand(3)
                t = ChainRulesCore.Tangent{Any,typeof(f)}((df1, df2))

                result = Mooncake.increment_and_get_rdata!(f, r_no, t)

                @test result isa typeof(r_no)
                @test f[1] ≈ f1_orig + df1
                @test f[2] ≈ f2_orig + df2
            end

            @testset "NoRData - heterogeneous Tuple fdata" begin
                # f, r_no, t - Tuple{Vector{Float64}, Vector{Float64}} (different lengths), NoRData, Tangent{Any, <:Tuple}
                f1_orig = rand(3)
                f2_orig = rand(2)
                f = (copy(f1_orig), copy(f2_orig))

                df1, df2 = rand(3), rand(2)
                t = ChainRulesCore.Tangent{Any,typeof(f)}((df1, df2))

                result = Mooncake.increment_and_get_rdata!(f, r_no, t)

                @test result isa typeof(r_no)
                @test f[1] ≈ f1_orig + df1
                @test f[2] ≈ f2_orig + df2
            end

            @testset "Mixed - Tuple{Vector, Tuple} fdata and rdata" begin
                # f1, r1, t1 - Tuple{Vector{Float64}, NoFData}, Tuple{NoRData, Tuple{Float64,Float64}}, Tangent{Any, <:Tuple}
                f1_orig = rand(3)
                f1 = (copy(f1_orig), f_no)

                r2 = (rand(), rand())
                r1 = (r_no, r2)

                df1 = rand(3)
                dr2 = (rand(), rand())
                t1 = ChainRulesCore.Tangent{Any,typeof((f1_orig, r2))}((df1, dr2))

                result = Mooncake.increment_and_get_rdata!(f1, r1, t1)

                @test result isa typeof(r1)
                @test result[1] isa typeof(r_no)
                @test result[2][1] ≈ r2[1] + dr2[1]
                @test result[2][2] ≈ r2[2] + dr2[2]
                @test f1[1] ≈ f1_orig + df1
            end
        end
    end

    @testset "@from_forward" begin
        R = FromForwardResources

        # is_primitive=false because test_rule's is_primitive=true asserts the rule is exactly rrule!!,
        # but @from_forward registers a ForwardModeRRule!! via build_primitive_rrule. With
        # is_primitive=false the type-equality check is skipped while the rule is still built and
        # numerically validated via finite differences.
        @testset "Working cases" begin
            @testset "Scalar to scalar" begin
                x = 5.0
                dy = 7.0
                cache = Mooncake.prepare_pullback_cache(R.f1, zero(x))
                val, pb = value_and_pullback!!(cache, dy, R.f1, x)
                @test val == x^3
                @test pb[1] == Mooncake.NoTangent()
                @test pb[2] == dy * 3x^2
                test_rule(
                    StableRNG(63), R.f1_true, x; is_primitive=false, perf_flag=:allocs
                )
            end
            @testset "Scalar to array" begin
                x = 5.0
                dy = [7.0, 11.0]
                cache = Mooncake.prepare_pullback_cache(R.f2, zero(x))
                val, pb = value_and_pullback!!(cache, dy, R.f2, x)
                @test val == [x^3, x^4]
                @test pb[1] == Mooncake.NoTangent()
                @test pb[2] == dy[1] * 3x^2 + dy[2] * 4x^3
                test_rule(StableRNG(63), R.f2_true, x; is_primitive=false)
            end
            @testset "Array to scalar" begin
                x = [5.0, 13.0]
                dy = 7.0
                cache = Mooncake.prepare_pullback_cache(R.f3, zero(x))
                val, pb = value_and_pullback!!(cache, dy, R.f3, x)
                @test val == sum(R.cube, x)
                @test pb[1] == Mooncake.NoTangent()
                @test pb[2] == dy .* map(_x -> 3 * _x^2, x)
                test_rule(StableRNG(63), R.f3_true, x; is_primitive=false)
            end
            @testset "Array to array" begin
                x = [5.0, 13.0]
                dy = [7.0, 11.0]
                cache = Mooncake.prepare_pullback_cache(R.f4, zero(x))
                val, pb = value_and_pullback!!(cache, dy, R.f4, x)
                @test val == map(R.cube, x)
                @test pb[1] == Mooncake.NoTangent()
                @test pb[2] ≈ dy .* map(_x -> 3 * _x^2, x)
                test_rule(StableRNG(63), R.f4_true, x; is_primitive=false)
            end
            @testset "Matrix to matrix" begin
                x = [5.0 13.0; 3.0 7.0]
                dy = [7.0 11.0; 2.0 4.0]
                cache = Mooncake.prepare_pullback_cache(R.f4m, zero(x))
                val, pb = value_and_pullback!!(cache, dy, R.f4m, x)
                @test val == map(sin, x)
                @test pb[1] == Mooncake.NoTangent()
                @test pb[2] ≈ dy .* map(cos, x)
                test_rule(StableRNG(63), R.f4m, x; is_primitive=false)
            end
            # Julia bug on < 1.11: JIT codegen crash (specsig OC call), fixed in 1.11.
            @static if VERSION >= v"1.11"
                @testset "Multiple arguments" begin
                    x = [5.0, 13.0]
                    y = 11.0
                    dz = 17.0
                    cache = Mooncake.prepare_pullback_cache(R.f5, zero(x), zero(y))
                    val, pb = value_and_pullback!!(cache, dz, R.f5, x, y)
                    @test val == sin(y) * sum(abs2, x)
                    @test pb[1] == Mooncake.NoTangent()
                    @test pb[2] ≈ dz .* sin(y) .* 2 .* x   # ∂/∂x (array fdata)
                    @test pb[3] ≈ dz * cos(y) * sum(abs2, x)  # ∂/∂y (scalar rdata)
                    test_rule(StableRNG(63), R.f5, x, y; is_primitive=false)
                    test_rule(StableRNG(63), R.f6, y, x; is_primitive=false)
                end
            end
        end

        @testset "Allocations" begin
            @testset "Array input, scalar output: zero allocations" begin
                test_rule(
                    StableRNG(63),
                    R.fperf_v,
                    [3.0, 5.0];
                    is_primitive=false,
                    perf_flag=:allocs,
                )
            end
            @testset "Buffer resize: correctness with variable-size arrays" begin
                cache = Mooncake.prepare_pullback_cache(R.fperf_v, zeros(2))
                _, pb2 = value_and_pullback!!(cache, 1.0, R.fperf_v, [1.0, 2.0])
                @test pb2[2] ≈ 3 .* [1.0, 2.0] .^ 2
                _, pb3 = value_and_pullback!!(cache, 1.0, R.fperf_v, [1.0, 2.0, 3.0])
                @test pb3[2] ≈ 3 .* [1.0, 2.0, 3.0] .^ 2
                x3 = [1.0, 2.0, 3.0]
                value_and_pullback!!(cache, 1.0, R.fperf_v, x3)  # warm up new size
                @test Mooncake.TestUtils.count_allocs(
                    value_and_pullback!!, cache, 1.0, R.fperf_v, x3
                ) == 0
            end
        end

        @testset "Failing cases" begin
            @testset "Wrong expression (not a Tuple{...})" begin
                @test_throws "LoadError: ArgumentError: The provided signature must be of the form `Tuple{typeof(f), ...}`." @eval Mooncake.@from_forward(
                    :(f3(x::Float64))
                )
            end
            @testset "Closures" begin
                @test_throws "ArgumentError: `Mooncake.@from_forward` does not support functions which close over data." Mooncake.prepare_gradient_cache(
                    R.Multiplier(2.0), 5.0
                )
            end
            @testset "Unsupported input types" begin
                @test_throws ArgumentError Mooncake.prepare_pullback_cache(R.f_tup, (5.0,))
            end
            @testset "Forgot macro" begin
                @test_throws MethodError Mooncake.prepare_pullback_cache(
                    R.f_forgot_macro, 1.0
                )
            end
            @testset "Wrong signature (no method)" begin
                let ex = @test_throws MethodError Mooncake.prepare_pullback_cache(
                        R.f1, Float32(1.0)
                    )
                    @test ex.value.f === R.f1
                end
            end
        end
    end
end
