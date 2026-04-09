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

function frule!!(
    ::Dual{typeof(Base.Threads.threading_run)}, fun::Dual{F}, static::Dual{Bool}
) where {F}
    worker_rule = build_frule(get_interpreter(ForwardMode), Tuple{F,Int})
    worker_rules = [_copy(worker_rule) for _ in 1:Threads.threadpoolsize()]
    Base.Threads.threading_run(primal(static)) do tid
        # `threading_run` hands worker ids in the default pool's 1-based tid space.
        1 <= tid <= length(worker_rules) ||
            throw(ArgumentError("unexpected thread id $tid"))
        worker_rules[tid](fun, zero_dual(tid))
        nothing
    end
    return zero_dual(nothing)
end
