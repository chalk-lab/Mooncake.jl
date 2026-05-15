@inline function _threading_foreigncall_frule(name::Val, args...)
    return zero_dual(_foreigncall_(name, tuple_map(primal, args)...))
end

function _threading_foreigncall_rrule()
    throw(
        ErrorException(
            "Differentiating through threading is not safe and is unsupported " *
            "in reverse mode. Use forward mode instead.",
        ),
    )
end

for name in [
    :jl_enter_threaded_region,
    :jl_exit_threaded_region,
    :jl_in_threaded_region,
    :jl_new_task,
    :jl_set_task_tid,
    :jl_set_task_threadpoolid,
    :jl_get_task_tid,
    :jl_get_task_threadpoolid,
    :jl_is_task_started,
    :jl_wakeup_thread,
    :jl_set_next_task,
    :jl_switch,
    :jl_get_next_task,
    :jl_task_get_next,
]
    @eval frule!!(::Dual{typeof(_foreigncall_)}, ::Dual{Val{$(QuoteNode(name))}}, args...) = _threading_foreigncall_frule(
        Val($(QuoteNode(name))), args...
    )

    @eval rrule!!(::CoDual{typeof(_foreigncall_)}, ::CoDual{Val{$(QuoteNode(name))}}, args...) = _threading_foreigncall_rrule()
end

@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(Base.Threads.threading_run),F,Bool
} where {F}

# `build_frule` already memoises the worker rule per `(F, world)` via `interp.oc_cache`,
# but each call still pays for `[_copy(worker_rule) for _ in 1:threadpoolsize()]` —
# `_copy` recursively copies captures and runs once per thread. Cache the per-thread
# array keyed by signature + threadpoolsize + width (the threadpoolsize changes if
# the user reconfigures threads at runtime; width is part of the rule's call shape).
const _THREADING_RUN_RULES_LOCK = ReentrantLock()
const _THREADING_RUN_RULES = Dict{Tuple{Any,Int,UInt,Int},Vector{Any}}()
push!(
    _EXTRA_CACHE_CLEANERS,
    () -> lock(() -> empty!(_THREADING_RUN_RULES), _THREADING_RUN_RULES_LOCK),
)

function _threading_run_worker_rules(::Type{F}, ::Val{N}, world::UInt) where {F,N}
    n = Threads.threadpoolsize()
    key = (F, n, world, N)
    lock(_THREADING_RUN_RULES_LOCK) do
        cached = get(_THREADING_RUN_RULES, key, nothing)
        cached === nothing || return cached
        interp = get_interpreter(ForwardMode)
        worker_rule = build_frule(interp, Tuple{F,Int}, Val(N))
        rules = Any[_copy(worker_rule) for _ in 1:n]
        _THREADING_RUN_RULES[key] = rules
        return rules
    end
end

# `Base.Threads.threading_run` implementation kernel. Worker rules are built via
# `build_frule(interp, Tuple{F, Int}, Val(N))` so they accept the IR-emit's
# `Lifted{F, N}` / `Lifted{Int, N}` call shape. The frule passes its `Lifted`
# arguments through unchanged so the worker rule's first-arg dispatch on `F`
# survives the structural lift (which would otherwise expose a bare NamedTuple
# from `_unlift` and miss the closure type).
@inline function _threading_run_kernel(
    ::Val{N}, fun::Mooncake.Lifted{F,N}, static::Mooncake.Lifted{Bool,N}
) where {N,F}
    worker_rules = _threading_run_worker_rules(
        F, Val(N), get_interpreter(ForwardMode).world
    )
    Base.Threads.threading_run(primal(static)) do tid
        1 <= tid <= length(worker_rules) ||
            throw(ArgumentError("unexpected thread id $tid"))
        worker_rules[tid](fun, zero_lifted(Val(N), tid))
        nothing
    end
    return zero_lifted(Val(N), nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base.Threads.threading_run),N},
    fun::Mooncake.Lifted{F,N},
    static::Mooncake.Lifted{Bool,N},
) where {N,F}
    return _threading_run_kernel(Val(N), fun, static)
end
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Base.Threads.threading_run),Any,Bool}}
) = true
