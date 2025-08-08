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
    ReverseMode

const CRC = ChainRulesCore

local_function(x) = 3x
overlay_tester(x) = 2x
@mooncake_overlay overlay_tester(x) = local_function(x)

zero_tester(x) = 0
@zero_derivative MinimalCtx Tuple{typeof(zero_tester),Float64}

vararg_zero_tester(x...) = 0
@zero_derivative MinimalCtx Tuple{typeof(vararg_zero_tester),Vararg}

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

        perf_flag = :stability_and_allocs
        @testset "forward mode only" begin
            sig = Tuple{typeof(ToolsForRulesResources.zero_tester_forward_only),Float64}
            @test is_primitive(MinimalCtx, ForwardMode, sig)
            @test !is_primitive(MinimalCtx, ReverseMode, sig)
            args = (ToolsForRulesResources.zero_tester_forward_only, 5.0)
            test_rule(sr(123), args...; is_primitive=true, perf_flag, mode=ForwardMode)
        end
        @testset "reverse mode only" begin
            sig = Tuple{typeof(ToolsForRulesResources.zero_tester_reverse_only),Float64}
            @test !is_primitive(MinimalCtx, ForwardMode, sig)
            @test is_primitive(MinimalCtx, ReverseMode, sig)
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
            ([ones(5), NoTangent()], [ones(5), ChainRulesCore.NoTangent()]),
            (
                Tangent((a=5.0, b=NoTangent())),
                ChainRulesCore.Tangent{Any}(; a=5.0, b=ChainRulesCore.NoTangent()),
            ),
            (
                MutableTangent((a=5.0, b=ones(3))),
                ChainRulesCore.Tangent{Any}(; a=5.0, b=ones(3)),
            ),
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
            @test_throws MethodError pb!!(5.0)
        end
    end
end
