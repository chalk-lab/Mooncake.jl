module MooncakeJETExt

using JET, Mooncake

# Test-utility state only, never reached from rules.
# JET entry analyses are retained through compiler backedges; memoisation avoids
# repeating them until an upstream JET fix makes this workaround unnecessary.
# Entries are reused only within one world age: any top-level definition between
# calls clears them. Reuse comes from many calls in one statement, as in registry runners.
const reports = Dict{Any,Any}()
const report_world = Ref(UInt(0))
const report_lock = ReentrantLock()

signature(tt::Type{<:Tuple}) = tt
signature(mi::Core.MethodInstance) = mi
signature(f, types=Base.default_tt(f)) = Base.signature_type(f, types)

function cached_report(x...; options...)
    return lock(report_lock) do
        world = Base.get_world_counter()
        if report_world[] != world
            empty!(reports)
            report_world[] = world
        end
        # OpaqueClosure analysis depends on its source and its captures' types.
        first(x) isa Core.OpaqueClosure && return JET.report_opt(x...; options...)
        get!(reports, (signature(x...), (; options...))) do
            JET.report_opt(x...; options...)
        end
    end
end

function Mooncake.TestUtils.test_opt_internal(::Mooncake.TestUtils.Shim, x...; options...)
    # JET.func_test uses the same callable for analysis and display, so we must name the cache.
    return JET.func_test(cached_report, :test_opt, x...; options...)
end
function Mooncake.TestUtils.report_opt_internal(::Mooncake.TestUtils.Shim, x...; options...)
    return cached_report(x...; options...)
end

end
