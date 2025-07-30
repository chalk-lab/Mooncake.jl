# Avoid troublesome bitcast magic -- we can't handle converting from pointer to UInt,
# because we drop the gradient, because the tangent type of integers is NoTangent.
# https://github.com/JuliaLang/julia/blob/9f9e989f241fad1ae03c3920c20a93d8017a5b8f/base/pointer.jl#L282
@is_primitive MinimalCtx Tuple{typeof(Base.:(+)),Ptr,Integer}
function rrule!!(f::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    return CoDual(primal(x) + primal(y), tangent(x) + primal(y)), NoPullback(f, x, y)
end

@zero_adjoint MinimalCtx Tuple{typeof(randn),AbstractRNG,Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(string),Vararg}
@zero_adjoint MinimalCtx Tuple{Type{Symbol},Vararg}
@zero_adjoint MinimalCtx Tuple{Type{Float64},Any,RoundingMode}
@zero_adjoint MinimalCtx Tuple{Type{Float32},Any,RoundingMode}
@zero_adjoint MinimalCtx Tuple{Type{Float16},Any,RoundingMode}
@zero_adjoint MinimalCtx Tuple{typeof(==),Type,Type}

# logging, String related primitive rules
using Base: getindex, getproperty
using Base.Threads: Atomic
using Mooncake: zero_fcodual, MinimalCtx, @is_primitive, NoPullback, CoDual
using Base.CoreLogging: LogLevel, handle_message, invokelatest
import Base.CoreLogging as CoreLogging

# Rule for accessing an Atomic{T} wrapped Integer with Base.getindex as deriving a rule results
# in encountering a Atomic->Int address bitcast followed by a llvm atomic load call 
@is_primitive MinimalCtx Tuple{typeof(getindex),Atomic{T}} where {T<:Integer}
function rrule!!(::CoDual{typeof(getindex)}, x::CoDual{Atomic{T}}) where {T<:Integer}
    return zero_fcodual(getindex(x.x)), NoPullback()
end

@is_primitive MinimalCtx Tuple{typeof(Base.normpath),String}
function rrule!!(::CoDual{typeof(Base.normpath)}, path::CoDual{String})
    return zero_fcodual(Base.normpath(path.x)), NoPullback()
end

@is_primitive MinimalCtx Tuple{
    typeof(Base._replace_init),String,Tuple{Pair{String,String}},Int64
}
function rrule!!(
    ::CoDual{typeof(Base._replace_init)},
    str::CoDual{String},
    replacements::CoDual{Tuple{Pair{String,String}}},
    count::CoDual{Int64},
)
    return zero_fcodual(Base._replace_init(str.x, replacements.x, count.x)), NoPullback()
end

@is_primitive MinimalCtx Tuple{
    typeof(CoreLogging.current_logger_for_env),LogLevel,Symbol,Module
}
function rrule!!(
    ::CoDual{typeof(CoreLogging.current_logger_for_env)},
    level::CoDual{LogLevel},
    group::CoDual{Symbol},
    _module::CoDual{Module},
)
    logger = CoreLogging.current_logger_for_env(level.x, group.x, _module.x)
    return zero_fcodual(logger), NoPullback()
end

@is_primitive MinimalCtx Tuple{
    typeof(Core._call_latest),
    typeof(Base.CoreLogging.shouldlog),
    Any,
    LogLevel,
    Module,
    Symbol,
    Symbol,
}
function rrule!!(
    ::CoDual{typeof(Core._call_latest)},
    ::CoDual{typeof(Base.CoreLogging.shouldlog)},
    logger::CoDual,
    level::CoDual{LogLevel},
    _module::CoDual{Module},
    group::CoDual{Symbol},
    id::CoDual{Symbol},
)
    result = Core._call_latest(
        Base.CoreLogging.shouldlog, logger.x, level.x, _module.x, group.x, id.x
    )
    return zero_fcodual(result), NoPullback()
end

@is_primitive MinimalCtx Tuple{
    typeof(Core._call_latest),typeof(handle_message),Any,Vararg{Any}
}
function rrule!!(
    ::CoDual{typeof(Core._call_latest)},
    ::CoDual{typeof(CoreLogging.handle_message)},
    logger::CoDual,
    args::Vararg{CoDual},
)
    prim_args = map(primal, args)
    result = Core._call_latest(CoreLogging.handle_message, logger.x, prim_args...)
    return zero_fcodual(result), NoPullback()
end

@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),
    NamedTuple,
    typeof(CoreLogging.invokelatest),
    typeof(CoreLogging.handle_message),
    Vararg{Any},
}
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs::CoDual,
    ::CoDual{typeof(Core._call_latest)},
    ::CoDual{typeof(CoreLogging.handle_message)},
    args::Vararg{CoDual},
)
    prim_args = map(primal, args)
    Core._call_latest(CoreLogging.handle_message, logger.x, prim_args...; kwargs...)
    return zero_fcodual(nothing), NoPullback()
end

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
        ),],

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
    return Any[], Any[]
end
