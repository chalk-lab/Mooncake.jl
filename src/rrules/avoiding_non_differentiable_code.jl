# Avoid troublesome bitcast magic -- we can't handle converting from pointer to UInt,
# because we drop the gradient, because the tangent type of integers is NoTangent.
# https://github.com/JuliaLang/julia/blob/9f9e989f241fad1ae03c3920c20a93d8017a5b8f/base/pointer.jl#L282
@is_primitive MinimalCtx Tuple{typeof(Base.:(+)),Ptr,Integer}
function frule!!(::Dual{typeof(Base.:(+))}, x::Dual{<:Ptr}, y::Dual{<:Integer})
    return Dual(primal(x) + primal(y), tangent(x) + primal(y))
end
function rrule!!(f::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    return CoDual(primal(x) + primal(y), tangent(x) + primal(y)), NoPullback(f, x, y)
end

@zero_derivative MinimalCtx Tuple{typeof(randn),AbstractRNG,Vararg}
@zero_derivative MinimalCtx Tuple{typeof(string),Vararg}
@zero_derivative MinimalCtx Tuple{Type{Symbol},Vararg}
@zero_derivative MinimalCtx Tuple{Type{Float64},Any,RoundingMode}
@zero_derivative MinimalCtx Tuple{Type{Float32},Any,RoundingMode}
@zero_derivative MinimalCtx Tuple{Type{Float16},Any,RoundingMode}
@zero_derivative MinimalCtx Tuple{typeof(==),Type,Type}

# logging, String related primitive rules
using Base: getindex, getproperty
using Base.Threads: Atomic
using Mooncake: zero_fcodual, MinimalCtx, @is_primitive, NoPullback, CoDual
using Base.CoreLogging: LogLevel, handle_message, invokelatest
import Base.CoreLogging as CoreLogging

# Rule for accessing an Atomic{T} wrapped Integer with Base.getindex as deriving a rule
# results in encountering a Atomic->Int address bitcast followed by a llvm atomic load call.
@zero_derivative MinimalCtx Tuple{typeof(getindex),Atomic{I}} where {I<:Integer}

# Some Base String related rrules :
@zero_derivative MinimalCtx Tuple{typeof(print),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(println),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(show),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(normpath),String}

# seperate kwargs, non-kwargs Base.sprint rules are required. Julia compilation only gives a
# common lowered IR for any Base.sprint calls. Refer to issue #558 and PR
# https://github.com/chalk-lab/Mooncake.jl/pull/659 for another sneaky appearance of this
# problem + fix.
@zero_derivative MinimalCtx Tuple{typeof(sprint),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(Core.kwcall),NamedTuple,typeof(sprint),Vararg}

# Base.CoreLogging @logmsg related primitives.
@zero_derivative MinimalCtx Tuple{
    typeof(Base._replace_init),String,Tuple{Pair{String,String}},Int64
}
@zero_derivative MinimalCtx Tuple{
    typeof(CoreLogging.current_logger_for_env),LogLevel,Symbol,Module
}
@zero_derivative MinimalCtx Tuple{
    typeof(Core._call_latest),
    typeof(Base.CoreLogging.shouldlog),
    Any,
    LogLevel,
    Module,
    Symbol,
    Symbol,
}
@zero_derivative MinimalCtx Tuple{
    typeof(Core._call_latest),
    typeof(CoreLogging.handle_message),
    Any,
    Base.CoreLogging.LogLevel,
    String,
    Module,
    Symbol,
    Symbol,
    String,
    Int64,
}
# specialized case for Builtin primitive Core._call_latest rrule for CoreLogging.handle_message kwargs call.
@zero_derivative(
    MinimalCtx,
    Tuple{
        typeof(Core._call_latest),
        typeof(Core.kwcall),
        NamedTuple,
        typeof(CoreLogging.handle_message),
        Any,
        Base.CoreLogging.LogLevel,
        String,
        Module,
        Symbol,
        Symbol,
        String,
        Int64,
    }
)

function generate_hand_written_rrule!!_test_cases(
    rng_ctor, ::Val{:avoiding_non_differentiable_code}
)
    _x = Ref(5.0)
    _dx = Ref(4.0)
    test_cases = vcat(
        Any[
            # Rules to avoid pointer type conversions.
            (
                true,
                :stability_and_allocs,
                nothing,
                +,
                CoDual(
                    bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                    bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
                ),
                2,
            ),

            # Rules for handling Atomic read operations.
            (false, :stability_and_allocs, nothing, getindex, Atomic{Int64}(rand(1:100))),
            (false, :stability_and_allocs, nothing, getindex, Atomic{Int32}(rand(1:100))),
            (false, :stability_and_allocs, nothing, getindex, Atomic{Int16}(rand(1:100))),
        ],

        # Rules in order to avoid introducing determinism.
        reduce(
            vcat,
            map([Xoshiro(1), TaskLocalRNG()]) do rng
                return Any[
                    (true, :stability_and_allocs, nothing, randn, rng),
                    (true, :stability, nothing, randn, rng, 2),
                    (true, :stability, nothing, randn, rng, 3, 2),
                ]
            end,
        ),

        # Rules to make string-related functionality work properly.
        (false, :stability, nothing, string, 'H'),
        (false, :stability, nothing, Base.normpath, "/home/user/../folder/./file.txt"),
        (
            false,
            :stability,
            nothing,
            Base._replace_init,
            "hello world",
            ("hello" => "hi",),
            1,
        ),
        (false, :none, nothing, print, "Testing print"),
        (false, :none, nothing, println, "Testing println"),
        (false, :none, nothing, show, "Testing show"),

        # non-kwargs sprint rule test
        (false, :stability, nothing, sprint, show, "Testing sprint"),

        # Rules to make Symbol-related functionality work properly.
        (false, :stability_and_allocs, nothing, Symbol, "hello"),
        (false, :stability_and_allocs, nothing, Symbol, UInt8[1, 2]),
        (false, :stability_and_allocs, nothing, Float64, π, RoundDown),
        (false, :stability_and_allocs, nothing, Float64, π, RoundUp),
        (true, :stability_and_allocs, nothing, Float32, π, RoundDown),
        (true, :stability_and_allocs, nothing, Float32, π, RoundUp),
        (true, :stability_and_allocs, nothing, Float16, π, RoundDown),
        (true, :stability_and_allocs, nothing, Float16, π, RoundUp),
    )
    memory = Any[_x, _dx]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(
    rng_ctor, ::Val{:avoiding_non_differentiable_code}
)
    function testloggingmacro1(x)
        @warn "Testing @warn macro"
    end

    function testloggingmacro2(x)
        @info "Testing @info macro"
    end

    function testloggingmacro3(x)
        @error "Testing @error macro"
    end

    function testloggingmacro4(x)
        @debug "Testing @debug macro"
    end

    function testloggingmacro5(x; kw1=rand(1:100))
        @info "Testing @info macro with kwargs" x kw1
    end

    # Base.sprint kwargs rule test
    function testloggingmacro6(x)
        return sprint(show, x; context=nothing)
    end

    function testloggingmacro7(x)
        return repr(x; context=nothing)
    end

    function testloggingmacro8(x)
        return repr(x)
    end

    function testloggingmacro9(x)
        @show x
    end

    test_cases = vcat(
        Any[
            # Tests for Base.CoreLogging, @show macros and string related functions.
            (false, :none, nothing, testloggingmacro1, rand(1:100)),
            (false, :none, nothing, testloggingmacro2, rand(1:100)),
            (false, :none, nothing, testloggingmacro3, rand(1:100)),
            (false, :none, nothing, testloggingmacro4, rand(1:100)),
            (false, :none, nothing, testloggingmacro5, rand(1:100)),
            (false, :none, nothing, testloggingmacro6, rand(1:100)),
            (false, :none, nothing, testloggingmacro7, rand(1:100)),
            (false, :none, nothing, testloggingmacro8, rand(1:100)),
            (false, :none, nothing, testloggingmacro9, rand(1:100)),
        ],
    )
    return test_cases, Any[]
end
