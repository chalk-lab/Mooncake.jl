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

tangent_to_primal_internal!!(x::Task, t, ::MaybeCache) = x
primal_to_tangent_internal!!(t, x::Task, ::MaybeCache) = t

_dot_internal(::MaybeCache, ::TaskTangent, ::TaskTangent) = 0.0

_scale_internal(::MaybeCache, ::Float64, t::TaskTangent) = t

TestUtils.populate_address_map_internal(m::TestUtils.AddressMap, ::Task, ::TaskTangent) = m

fdata_type(::Type{TaskTangent}) = TaskTangent

rdata_type(::Type{TaskTangent}) = NoRData

tangent(t::TaskTangent, ::NoRData) = t

@inline function _get_fdata_field(_, t::TaskTangent, f)
    f === :rngState0 && return NoFData()
    f === :rngState1 && return NoFData()
    f === :rngState2 && return NoFData()
    f === :rngState3 && return NoFData()
    f === :rngState4 && return NoFData()
    # All other Task fields are non-differentiable runtime infrastructure.
    return NoFData()
end

@inline increment_field_rdata!(::TaskTangent, ::NoRData, ::Val) = nothing

function get_tangent_field(t::TaskTangent, f)
    f === :rngState0 && return NoTangent()
    f === :rngState1 && return NoTangent()
    f === :rngState2 && return NoTangent()
    f === :rngState3 && return NoTangent()
    f === :rngState4 && return NoTangent()
    # All other Task fields are non-differentiable runtime infrastructure.
    return NoTangent()
end

const TaskCoDual = CoDual{Task,TaskTangent}

# Forward-mode canonical V for Task — same `TaskTangent` reverse mode uses.
# All Task fields are non-differentiable, so the tangent carries no lane
# data; we keep one shared `TaskTangent` per slot independent of width N.
@inline dual_type(::Val{N}, ::Type{Task}) where {N} = TaskTangent
@inline lifted_type(::Val{N}, ::Type{Task}) where {N} = Lifted{Task,N,TaskTangent}

# Forward seed factories: a `Task`'s V is the singleton `TaskTangent` (= its reverse tangent),
# not a structural lift, so the generic `@generated` seed factory cannot build it (a `Task`
# has 16 fields but no NamedTuple-backed dual). Seed it directly, mirroring reverse
# `zero_tangent_internal` / `randn_tangent_internal`. `zero_lifted` / `randn_lifted` enter via
# the `*_internal` family.
for f in (:_zero_dual_internal, :_uninit_dual_internal)
    @eval @inline $f(::Val{N}, ::Task, ::MaybeCache) where {N} = TaskTangent()
end
@inline _randn_dual_internal(::Val{N}, ::AbstractRNG, ::Task, ::MaybeCache) where {N} = TaskTangent()
# Per-lane tangent accessor and the width-1 lift boundary for the singleton V.
@inline tangent(::Lifted{Task,N,TaskTangent}, ::Integer) where {N} = TaskTangent()
@inline lift(x::Task, ::TaskTangent) = Lifted{Task,1}(x, TaskTangent())

function frule!!(
    ::Lifted{typeof(lgetfield),N}, x::Lifted{Task,N,TaskTangent}, ::Lifted{Val{f},N}
) where {N,f}
    # All Task fields are non-differentiable, so the read carries no forward derivative.
    y = getfield(primal(x), f)
    return Lifted{typeof(y),N}(y, NoDual())
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

function frule!!(
    ::Lifted{typeof(getfield),N}, x::Lifted{Task,N,TaskTangent}, f::Lifted
) where {N}
    y = getfield(primal(x), primal(f))
    return Lifted{typeof(y),N}(y, NoDual())
end
function rrule!!(::CoDual{typeof(getfield)}, x::TaskCoDual, f::CoDual)
    return rrule!!(zero_fcodual(lgetfield), x, zero_fcodual(Val(primal(f))))
end

function frule!!(
    ::Lifted{typeof(lsetfield!),N},
    task::Lifted{Task,N,TaskTangent},
    ::Lifted{Val{name},N},
    val::Lifted,
) where {N,name}
    # Inline body — `set_tangent_field!(::TaskTangent, ::Symbol, ::NoTangent)` is
    # a no-op (Task fields are non-differentiable), so we only mutate the
    # user's Task primal and return the new-value slot unchanged.
    setfield!(primal(task), name, primal(val))
    return val
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
