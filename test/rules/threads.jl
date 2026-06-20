function threaded_sin_sum(x::Vector{Float64})
    y = similar(x)
    Threads.@threads for i in eachindex(y, x)
        y[i] = sin(x[i])
    end
    return sum(y)
end

@testset "threads" begin
    x = randn(4)

    TestUtils.test_rule(
        StableRNG(123), threaded_sin_sum, x; is_primitive=false, mode=ForwardMode
    )

    # Regression: a `Task`-returning threading foreigncall (jl_new_task) must yield a
    # canonical-coherent slot. The frule wraps results with `zero_lifted`, so the Task gets a
    # `TaskTangent` V; a blanket `Lifted(y, NoDual())` would fail `verify_canonical_dual_type`
    # (dual_type(Task) === TaskTangent). The Task's V is never consumed in simple forward AD, so
    # this is checked at the frule boundary. `TaskTangent` is width-invariant, so width 1 suffices.
    # Args mirror the normalized `_foreigncall_(Val(:jl_new_task), RT, AT, nreq, cc, f, cf, ssize)`.
    @testset "jl_new_task slot coherence" begin
        zl(v) = Mooncake.zero_lifted(Val(1), v)
        r = Mooncake.frule!!(
            zl(Mooncake._foreigncall_),
            zl(Val(:jl_new_task)),
            zl(Val{Ref{Task}}()),
            zl((Val{Any}(), Val{Any}(), Val{Int64}())),
            zl(Val{0}()),
            zl(Val{:ccall}()),
            zl(() -> nothing),
            zl(nothing),
            zl(0),
        )
        @test primal(r) isa Task
        @test tangent(r) isa Mooncake.TaskTangent
        @test Mooncake.verify_canonical_dual_type(r) === nothing
    end
end
