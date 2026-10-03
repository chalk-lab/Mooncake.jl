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

    # Check coherence at the boundary: forward AD never consumes this fieldless Task V.
    # TaskTangent is width-independent. Match ssize to native Int, including on 32-bit.
    @testset "jl_new_task slot coherence" begin
        zl(v) = Mooncake.zero_lifted(Val(1), v)
        r = Mooncake.frule!!(
            zl(Mooncake._foreigncall_),
            zl(Val(:jl_new_task)),
            zl(Val{Ref{Task}}()),
            zl((Val{Any}(), Val{Any}(), Val{Int}())),
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
