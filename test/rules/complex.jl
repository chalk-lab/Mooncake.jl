@testset "complex" begin
    rng = sr(123)
    p = Complex{Float64}(5.0, 4.0)
    TestUtils.test_data(rng, p)
    p = Complex{Float32}(5.0, 4.0)
    TestUtils.test_data(rng, p)

    TestUtils.run_rule_test_cases(StableRNG, Val(:complex))
end

# Registry finite differences check first derivatives; these exercise the pullbacks' tangents.
@testset "reinterpret HVP" for T in (Float16, Float32, Float64), E in (T, Complex{T})
    function f(x)
        p = E <: Real ? complex.(x[1:2], x[3:4]) : copy(x)
        r = reinterpret(E, p)
        r[1] = 2r[2]
        return sum(abs2, r) + real(dot(r, r)) + real(sum(p))
    end
    for x in (T[1, 2, 3, 4], zeros(T, 4))
        v = T[2, 3, 4, 5]
        cache = Mooncake.prepare_gradient_cache(f, x)
        grad(z) = copy(Mooncake.value_and_gradient!!(cache, f, z)[2][2])
        h = if T === Float16
            T(0.125)
        elseif T === Float32
            T(0.01)
        else
            T(1e-5)
        end
        fd = (grad(x + h*v) - grad(x - h*v)) / (2h)
        actual = Mooncake.value_and_hvp!!(Mooncake.prepare_hvp_cache(f, x), f, v, x)[3]
        @test actual ≈ fd rtol=1e-3
    end
end

# The registry reference sees returned rdata, not the mutated parent fdata checked here.
@testset "reinterpret dot strong zeros" for T in (Float16, Float32, Float64),
    E in (T, Complex{T})

    S = E <: Real ? Complex{T} : T
    for a in (E(Inf), E(NaN)), b in (zero(E), one(E))
        x = reinterpret(E, fill(S(0), E <: Real ? 1 : 2))
        x .= a
        y = fill(b, length(x))
        cx, cy = zero_fcodual(x), zero_fcodual(y)
        _, pb = rrule!!(zero_fcodual(dot), cx, cy)
        pb(zero(E))
        @test all(iszero, Mooncake._reinterpret_tangent(cx))
        @test all(iszero, tangent(cy))
        for w in (1, 8)
            lx, ly = Mooncake.zero_lifted(Val(w), x), Mooncake.zero_lifted(Val(w), y)
            out = Mooncake.frule!!(Mooncake.zero_lifted(Val(w), dot), lx, ly)
            @test all(k -> iszero(tangent(out, k)), 1:w)
            if iszero(b)
                foreach(d -> fill!(d, E(Inf)), Mooncake._reinterpret_lanes(lx))
                out = Mooncake.frule!!(Mooncake.zero_lifted(Val(w), dot), lx, ly)
                @test all(k -> iszero(tangent(out, k)), 1:w)
            end
        end
        if iszero(b)
            _, pb = rrule!!(zero_fcodual(dot), cx, cy)
            pb(E(Inf))
            @test all(iszero, Mooncake._reinterpret_tangent(cx))
        end
    end
end
