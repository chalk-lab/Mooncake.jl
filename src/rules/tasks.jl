# Tasks are recursively-defined, so their tangent type needs to be done manually.
# Occassionally one encountered tasks in code, but they don't actually get called. For
# example, calls to `rand` with a `TaskLocalRNG` will query the local task, purely for the
# sake of getting random number generator state associated to it.
# The goal of the code in this file is to ensure that this kind of usage of tasks is handled
# well, rather than attempting to properly handle tasks.

mutable struct TaskTangent end

tangent_type(::Type{Task}) = TaskTangent

function zero_tangent_internal(p::Task, dict::MaybeCache)
    if haskey(dict, p)
        return dict[p]::TaskTangent
    else
        t = TaskTangent()
        dict[p] = t
        return t
    end
end

function randn_tangent_internal(rng::AbstractRNG, p::Task, dict::MaybeCache)
    if haskey(dict, p)
        return dict[p]::TaskTangent
    else
        t = TaskTangent()
        dict[p] = t
        return t
    end
end

increment_internal!!(::IncCache, t::TaskTangent, s::TaskTangent) = t

set_to_zero_internal!!(::SetToZeroCache, t::TaskTangent) = t

_add_to_primal_internal(::MaybeCache, p::Task, t::TaskTangent, ::Bool) = p

_diff_internal(::MaybeCache, ::Task, ::Task) = TaskTangent()

_dot_internal(::MaybeCache, ::TaskTangent, ::TaskTangent) = 0.0

_scale_internal(::MaybeCache, ::Float64, t::TaskTangent) = t

TestUtils.populate_address_map_internal(m::TestUtils.AddressMap, ::Task, ::TaskTangent) = m

fdata_type(::Type{TaskTangent}) = TaskTangent

rdata_type(::Type{TaskTangent}) = NoRData

tangent(t::TaskTangent, ::NoRData) = t

@inline function _get_tangent_field(t::TaskTangent, f)
    f === :rngState0 && return NoTangent()
    f === :rngState1 && return NoTangent()
    f === :rngState2 && return NoTangent()
    f === :rngState3 && return NoTangent()
    f === :rngState4 && return NoTangent()
    return error("Unhandled field $f")
end
@inline function _get_fdata_field(_, t::TaskTangent, f)
    f === :rngState0 && return NoFData()
    f === :rngState1 && return NoFData()
    f === :rngState2 && return NoFData()
    f === :rngState3 && return NoFData()
    f === :rngState4 && return NoFData()
    return error("Unhandled field $f")
end

@inline increment_field_rdata!(::TaskTangent, ::NoRData, ::Val) = nothing

function get_tangent_field(t::TaskTangent, f)
    f === :rngState0 && return NoTangent()
    f === :rngState1 && return NoTangent()
    f === :rngState2 && return NoTangent()
    f === :rngState3 && return NoTangent()
    f === :rngState4 && return NoTangent()
    throw(error("Unhandled field $f"))
end

const TaskDual = Dual{Task,TaskTangent}
const TaskCoDual = CoDual{Task,TaskTangent}

function frule!!(::Dual{typeof(lgetfield)}, x::TaskDual, ::Dual{Val{f}}) where {f}
    return Dual(getfield(primal(x), f), _get_tangent_field(tangent(x), f))
end
function rrule!!(::CoDual{typeof(lgetfield)}, x::TaskCoDual, ::CoDual{Val{f}}) where {f}
    dx = x.dx
    function mutable_lgetfield_pb!!(dy)
        increment_field_rdata!(dx, dy, Val{f}())
        return NoRData(), NoRData(), NoRData()
    end
    y = CoDual(getfield(x.x, f), _get_fdata_field(x.x, x.dx, f))
    return y, mutable_lgetfield_pb!!
end

function frule!!(::Dual{typeof(getfield)}, x::TaskDual, f::Dual)
    return Dual(getfield(primal(x), primal(f)), _get_tangent_field(tangent(x), primal(f)))
end
function rrule!!(::CoDual{typeof(getfield)}, x::TaskCoDual, f::CoDual)
    return rrule!!(zero_fcodual(lgetfield), x, zero_fcodual(Val(primal(f))))
end

function frule!!(::Dual{typeof(lsetfield!)}, task::TaskDual, name::Dual, val::Dual)
    return lsetfield_frule(task, name, val)
end
function rrule!!(::CoDual{typeof(lsetfield!)}, task::TaskCoDual, name::CoDual, val::CoDual)
    return lsetfield_rrule(task, name, val)
end

set_tangent_field!(t::TaskTangent, f, ::NoTangent) = NoTangent()

@zero_derivative MinimalCtx Tuple{typeof(current_task)}

__verify_fdata_value(::IdDict{Any,Nothing}, ::Task, ::TaskTangent) = nothing

function hand_written_rule_test_cases(rng_ctor, ::Val{:tasks})
    test_cases = Any[
        (false, :none, nothing, lgetfield, Task(() -> nothing), Val(:rngState1)),
        (false, :none, nothing, getfield, Task(() -> nothing), :rngState1),
        (
            false,
            :none,
            nothing,
            lsetfield!,
            Task(() -> nothing),
            Val(:rngState1),
            UInt64(5),
        ),
        (false, :stability, nothing, current_task),
    ]
    memory = Any[]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:tasks})
    test_cases = Any[(
        false,
        :none,
        nothing,
        (rng) -> (Random.seed!(rng, 0); rand(rng)),
        Random.default_rng(),
    ),]
    memory = Any[]
    return test_cases, memory
end
