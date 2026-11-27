module MooncakeJETExt

using JET, Mooncake

# Avoid repeated JET entry analyses, which compiler backedges retain.
const reports = Dict{Any,Any}()
const report_world = Ref(UInt(0))
const report_lock = ReentrantLock()

signature(tt::Type{<:Tuple}) = tt
signature(f, types) = Base.signature_type(f, types)

function cached_report(x...; options...)
    return lock(report_lock) do
        world = Base.get_world_counter()
        if report_world[] != world
            empty!(reports)
            report_world[] = world
        end
        # OpaqueClosure reports depend on source and captures, not only signature.
        first(x) isa Core.OpaqueClosure && return JET.report_opt(x...; options...)
        get!(reports, (signature(x...), (; options...))) do
            JET.report_opt(x...; options...)
        end
    end
end

function Mooncake.TestUtils.test_opt_internal(::Mooncake.TestUtils.Shim, x...)
    return JET.func_test(cached_report, :test_opt, x...)
end
Mooncake.TestUtils.report_opt_internal(::Mooncake.TestUtils.Shim, tt) = cached_report(tt)

end
