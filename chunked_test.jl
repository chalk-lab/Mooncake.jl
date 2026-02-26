using Mooncake
import Mooncake: Dual, @is_primitive, frule!!, MinimalCtx, primal, tangents, zero_dual, tuple_map

@noinline function mycos(x)
    return cos(x)
end

@is_primitive MinimalCtx Tuple{typeof(mycos),Any}

function frule!!(::Dual{typeof(mycos)}, _x::Dual)
    x = primal(_x)
    dxs = tangents(_x)
    msin = -sin(x)
    return Dual(mycos(x), tuple_map(dx -> dx * msin, dxs))
end

function cos_of_square(x)
    return mycos(x^2)
end

function test_chunking()
    f = cos_of_square
    x = 0.5
    dxs = (1.0, 0.5, 0.25)
    x_dxs = Dual(x, dxs)
    cache = prepare_derivative_cache(f, x, chunksize=3)
    y = value_and_derivative!!(cache, zero_dual(f, Val(length(dxs))), x_dxs)
    @assert primal(y) ≈ cos(0.25)
    @assert [t for t in tangents(y)] ≈ [-sin(0.25), -sin(0.25)/2, -sin(0.25)/4]
end

test_chunking()