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

function test_chunking()
    x = Dual(0.5, (1.0, 0.5, 0.25))
    y = frule!!(zero_dual(mycos, Val(3)), x)
    @assert primal(y) ≈ cos(0.5)
    @assert [t for t in tangents(y)] ≈ [-sin(0.5), -sin(0.5)/2, -sin(0.5)/4]
end

test_chunking()