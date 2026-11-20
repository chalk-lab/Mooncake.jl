# Only the explicit prerelease profile loads these hooks. Ordinary tests still require a
# working AllocCheck extension; unavailable analysis is never a passing check.
function Mooncake.TestUtils.test_hook(
    ::Any, ::typeof(Mooncake.TestUtils.check_allocs), f::F, args...
) where {F}
    @test_skip AllocCheck_static_allocation_analysis
    # Callers also check the result, so preserve execution, mutation, and exceptions.
    return f(args...)
end
