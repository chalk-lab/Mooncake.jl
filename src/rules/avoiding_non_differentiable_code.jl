# Avoid troublesome bitcast magic -- we can't handle converting from pointer to UInt,
# because we drop the gradient, because the tangent type of integers is NoTangent.
# https://github.com/JuliaLang/julia/blob/9f9e989f241fad1ae03c3920c20a93d8017a5b8f/base/pointer.jl#L282
@is_primitive MinimalCtx Tuple{typeof(Base.:(+)),Ptr,Integer}
# Shift each lane pointer, including abstract-element pointers such as Ptr{Real}.
function frule!!(
    ::Lifted{typeof(Base.:(+)),Nw}, x::Lifted{P,Nw,<:NTuple{Nw,Ptr}}, y::Lifted{<:Integer}
) where {Nw,P<:Ptr}
    yp = primal(y)
    # Read the V's lane pointer, not `tangent(x, lane)`: that accessor materialises lane `lane`'s
    # REVERSE tangent, a `VoidPtrTangent` for `Ptr{Nothing}`, not the address the lane holds.
    return Lifted{P,Nw}(primal(x) + yp, ntuple(lane -> tangent(x)[lane] + yp, Val(Nw)))
end
# NoDual pointers (e.g. from bitcast) must match the primitive's full Ptr coverage.
function frule!!(
    ::Lifted{typeof(Base.:(+)),Nw}, x::Lifted{<:Ptr,Nw,NoDual}, y::Lifted{<:Integer}
) where {Nw}
    p = primal(x) + primal(y)
    return Lifted{typeof(p),Nw}(p, NoDual())
end
# `@is_primitive` above claims EVERY `Ptr`, so this must shift whatever a pointer's fdata is. For a
# `Ptr{Cvoid}` that is a `VoidPtrTangent`, which shifts its address and keeps what it erased.
@inline _shift_ptr_fdata(dx::Ptr, n::Integer) = iszero(UInt(dx)) ? dx : dx + n
@inline _shift_ptr_fdata(dx::VoidPtrTangent, n::Integer) = VoidPtrTangent(
    _shift_ptr_fdata(dx.p, n), dx.elt
)

function rrule!!(f::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    return CoDual(primal(x) + primal(y), _shift_ptr_fdata(tangent(x), primal(y))),
    NoPullback(f, x, y)
end

@zero_derivative MinimalCtx Tuple{typeof(randn),AbstractRNG,Vararg}
@zero_derivative MinimalCtx Tuple{typeof(string),Vararg}
# These Bool-valued predicates reach utf8proc foreign calls, including through
# LinearAlgebra wrapper-char dispatch. isdigit/isspace/iscntrl/isxdigit use ASCII fast paths.
for f in (:isuppercase, :islowercase, :isletter, :isnumeric, :ispunct, :isprint)
    @eval @zero_derivative MinimalCtx Tuple{typeof($f),AbstractChar}
end
@zero_derivative MinimalCtx Tuple{typeof(Base.Unicode.category_code),AbstractChar}
@zero_derivative MinimalCtx Tuple{Type{Symbol},Vararg}
@zero_derivative MinimalCtx Tuple{Type{Float64},Any,RoundingMode}
@zero_derivative MinimalCtx Tuple{Type{Float32},Any,RoundingMode}
@zero_derivative MinimalCtx Tuple{Type{Float16},Any,RoundingMode}
@zero_derivative MinimalCtx Tuple{typeof(==),Type,Type}

# Optional rule to avoid unnecessary allocations on Julia 1.10
@zero_derivative DefaultCtx Tuple{typeof(count),Any,Any}

# Logging: String-related primitive rules.
using Base.Threads: Atomic
using Base.CoreLogging: LogLevel
import Base.CoreLogging as CoreLogging

# Rule for accessing an Atomic{T}-wrapped Integer with Base.getindex, since deriving a rule
# results in encountering an Atomic->Int address bitcast followed by an LLVM atomic load call.
@zero_derivative MinimalCtx Tuple{typeof(getindex),Atomic{I}} where {I<:Integer}

# Some Base String-related rrules:
@zero_derivative MinimalCtx Tuple{typeof(print),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(println),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(show),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(normpath),String}

# Separate kwargs and non-kwargs Base.sprint rules are required. Julia compilation only gives a
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

# On 1.12+, @invokelatest uses Base.invokelatest/invokelatest_gr, not Core._call_latest.
@static if VERSION < v"1.12-"
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
end

# Package loading internals; also needed for extension code paths.
@zero_derivative MinimalCtx Tuple{Type{Base.PkgId},Module}
@zero_derivative MinimalCtx Tuple{typeof(Base.get_extension),Base.PkgId,Symbol}

@static if VERSION ≥ v"1.12-"
    @zero_derivative MinimalCtx Tuple{typeof(Base.fixup_stdlib_path),String}
    @zero_derivative MinimalCtx Tuple{
        typeof(CoreLogging.handle_message_nothrow),
        Any,
        CoreLogging.LogLevel,
        String,
        Module,
        Symbol,
        Symbol,
        String,
        Int64,
    }
    @zero_derivative MinimalCtx Tuple{
        typeof(Core.kwcall),NamedTuple,typeof(CoreLogging.handle_message_nothrow),Vararg
    }
end

# The kwargs variant is dead on 1.12+ for the same reason.
@static if VERSION < v"1.12-"
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
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:avoiding_non_differentiable_code})
    _x = Ref(5.0)
    _dx = Ref(4.0)
    test_cases = vcat(
        TestCase[
            # Rules to avoid pointer type conversions.
            TestCase(
                +,
                CoDual(
                    bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                    bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
                ),
                2;
                interface_only=true,
                perf_flag=:stability_and_allocs,
            ),

            # Rules for handling Atomic read operations.
            TestCase(getindex, Atomic{Int64}(rand(1:100)); perf_flag=:stability_and_allocs),
            TestCase(getindex, Atomic{Int32}(rand(1:100)); perf_flag=:stability_and_allocs),
            TestCase(getindex, Atomic{Int16}(rand(1:100)); perf_flag=:stability_and_allocs),
        ],

        # Rules in order to avoid introducing determinism.
        reduce(
            vcat,
            map([Xoshiro(1), TaskLocalRNG()]) do rng
                return TestCase[
                    TestCase(
                        randn, rng; interface_only=true, perf_flag=:stability_and_allocs
                    ),
                    TestCase(randn, rng, 2; interface_only=true, perf_flag=:stability),
                    TestCase(randn, rng, 3, 2; interface_only=true, perf_flag=:stability),
                ]
            end,
        ),

        # Rules to make string-related functionality work properly.
        TestCase(string, 'H'; perf_flag=:stability),
        TestCase(Base.normpath, "/home/user/../folder/./file.txt"; perf_flag=:stability),
        TestCase(
            Base._replace_init, "hello world", ("hello" => "hi",), 1; perf_flag=:stability
        ),

        # non-kwargs sprint rule test
        TestCase(sprint, show, "Testing sprint"; perf_flag=:stability),

        # Rules to make Symbol-related functionality work properly.
        TestCase(Symbol, "hello"; perf_flag=:stability_and_allocs),
        TestCase(Symbol, UInt8[1, 2]; perf_flag=:stability_and_allocs),

        # Julia Base functions have type stability issues in version 1.12
        TestCase(
            Float64,
            π,
            RoundDown;
            perf_flag=VERSION >= v"1.12-" ? :none : :stability_and_allocs,
        ),
        TestCase(
            Float64,
            π,
            RoundUp;
            perf_flag=VERSION >= v"1.12-" ? :none : :stability_and_allocs,
        ),
        TestCase(
            Float32,
            π,
            RoundDown;
            interface_only=true,
            perf_flag=VERSION >= v"1.12-" ? :none : :stability_and_allocs,
        ),
        TestCase(
            Float32,
            π,
            RoundUp;
            interface_only=true,
            perf_flag=VERSION >= v"1.12-" ? :none : :stability_and_allocs,
        ),

        # F16 works fine even in 1.12
        TestCase(
            Float16, π, RoundDown; interface_only=true, perf_flag=:stability_and_allocs
        ),
        TestCase(Float16, π, RoundUp; interface_only=true, perf_flag=:stability_and_allocs),
    )
    memory = Any[_x, _dx]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:avoiding_non_differentiable_code})
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
        TestCase[
            TestCase(
                (x -> unsafe_load(Ptr{Float64}(pointer(x)) + 8)),
                zeros(UInt8, 16);
                mode=ReverseMode,
                throws=(ArgumentError, "tangent pointer is NULL"),
            ),
            # Package loading internals: Module can't be deepcopied, so test via closures
            # that capture the module and take a differentiable Float64 arg instead.
            TestCase((x) -> (Base.PkgId(Base); x), 1.0),
            TestCase(
                (x) -> (Base.get_extension(Base.PkgId(Base), :GenericTestExt); x), 1.0
            ),

            # Matrix wrappers exercise utf8proc-backed char dispatch. A barrier keeps direct
            # predicate tests from constant-folding away before AD sees them.
            map(((X, Y) -> Symmetric(X) * Y, (X, Y) -> Hermitian(X) * Y)) do f
                return TestCase(
                    f, randn(rng_ctor(123), 4, 4), randn(rng_ctor(124), 4, 3)
                )
            end...,
            map((
                isuppercase, islowercase, isletter, isnumeric, ispunct, isprint
            )) do pred
                return TestCase(
                    (x -> (pred(Base.inferencebarrier('U')::Char) ? 2.0 : 3.0) * x), 1.0
                )
            end...,

            # Tests for Base.CoreLogging, @show macros and string related functions.
            TestCase((x) -> print(x), "Testing print"),
            TestCase((x) -> println(x), "Testing println"),
            TestCase((x) -> show(x), "Testing show"),
            TestCase(testloggingmacro1, rand(1:100)),
            TestCase(testloggingmacro2, rand(1:100)),
            TestCase(testloggingmacro3, rand(1:100)),
            TestCase(testloggingmacro4, rand(1:100)),
            TestCase(testloggingmacro5, rand(1:100)),
            TestCase(testloggingmacro6, rand(1:100)),
            TestCase(testloggingmacro7, rand(1:100)),
            TestCase(testloggingmacro8, rand(1:100)),
            TestCase(testloggingmacro9, rand(1:100)),
        ],
    )
    return test_cases, Any[]
end
