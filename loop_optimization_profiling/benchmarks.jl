# Test function definitions from Issue #156

function _sum(f::F, x::AbstractArray{<:Real}) where {F}
    y = 0.0
    n = 0
    while n < length(x)
        n += 1
        y += f(x[n])
    end
    return y
end

_map_sin_cos_exp(x::AbstractArray{<:Real}) = sum(map(x -> sin(cos(exp(x))), x))

# Benchmark definitions
function get_benchmarks()
    return [
        ("sum_1000", sum, (randn(1_000),)),
        ("_sum_1000", _sum, (identity, randn(1_000))),
        ("map_sin_cos_exp", _map_sin_cos_exp, (randn(10, 10),)),
    ]
end
