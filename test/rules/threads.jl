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
        StableRNG(123),
        threaded_sin_sum,
        x;
        is_primitive=false,
        mode=ForwardMode,
    )
end
