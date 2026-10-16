@testset "dict" begin
    @testset "keyed tangent data" for p in (
        Dict(1 => 2.0, 23 => 3.0),
        Dict(:a => [1.0, 2.0]),
        Dict{Symbol,Any}(:a => 2.0),
        [Dict(1 => 2.0)],
        (d=Dict(1 => 2.0),),
    )
        TestUtils.test_data(sr(123456), p)
        TestUtils.test_lifted(sr(123456), p)
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:dict))
    # Specific seed provenance and storage identity are not expressible as registry rows.
    @testset "key provenance and aliases" begin
        p = Dict(1 => 1.0, 23 => 2.0)
        t = Dict(23 => 0.0, 1 => 1.0)
        sizehint!(t, 1000)
        f(d) = d[1]^2 + 5d[23]^2
        for n in (1, 8)
            cache = Mooncake.build_frule(f, p; chunk_size=n)
            out = cache(
                Mooncake.zero_lifted(Val(n), f),
                Lifted{typeof(p),n}(
                    p,
                    sizehint!(
                        Dict(
                            k => Mooncake.Nfwd.NDual(p[k], ntuple(_ -> v, n)) for
                            (k, v) in t
                        ),
                        1000,
                    ),
                ),
            )
            @test primal(out) == 21.0
            @test all(i -> tangent(out, i) == 2.0, 1:n)
            empty!(p)
            p[23] = 2.0
            p[1] = 1.0
            out = cache(
                Mooncake.zero_lifted(Val(n), f),
                Lifted{typeof(p),n}(
                    p,
                    sizehint!(
                        Dict(
                            k => Mooncake.Nfwd.NDual(p[k], ntuple(_ -> v, n)) for
                            (k, v) in t
                        ),
                        1000,
                    ),
                ),
            )
            @test primal(out) == 21.0
            @test all(i -> tangent(out, i) == 2.0, 1:n)
        end
        # Seed an actual closure capture; the registry cannot perturb closure constructors.
        captured_f = let captured=p
            x -> captured[1] * x
        end
        for n in (1, 8)
            rule = Mooncake.build_frule(captured_f, 3.0; chunk_size=n)
            slot = Mooncake.zero_lifted(Val(n), captured_f)
            tangent(slot).fields.captured[1] = Mooncake.Nfwd.NDual(
                p[1], ntuple(_ -> 1.0, n)
            )
            out = rule(slot, Mooncake.zero_lifted(Val(n), 3.0))
            @test primal(out) == 3.0
            @test all(i -> tangent(out, i) == 3.0, 1:n)
        end
        _, captured_grad = Mooncake.value_and_gradient!!(
            Mooncake.build_rrule(captured_f, 3.0), captured_f, 3.0
        )
        @test captured_grad[1].fields.captured[1] == 3.0
        a = [1.0, 2.0]
        d = Dict(:a => a, :b => a)
        dt = zero_tangent(d)
        @test dt[:a] === dt[:b]
        dv = tangent(Mooncake.lift(d, dt))
        @test dv[:a] === dv[:b]
        for n in (1, 8)
            dv = Mooncake.zero_dual(Val(n), d)
            @test dv[:a] === dv[:b]
        end
    end
    # The registry compares values, not the identity/type of equal stored key objects.
    @testset "stored key restoration" for (stored, query) in ((1, 1.0), ([1], [1]))
        for width in (1, 8)
            d = Dict{Any,Float64}(stored => 2.0)
            slots = map(
                x -> Mooncake.zero_lifted(Val(width), x),
                (Mooncake._replace_dict_value!, d, 3.0, query),
            )
            result = Mooncake.frule!!(slots...)
            @test only(keys(primal(result))) === stored
            @test only(keys(tangent(result))) === stored
        end
        d = Dict{Any,Float64}(stored => 2.0)
        result, pb = Mooncake.rrule!!(
            map(Mooncake.zero_fcodual, (Mooncake._replace_dict_value!, d, 3.0, query))...
        )
        @test only(keys(primal(result))) === stored
        pb(Mooncake.NoRData())
        @test d[stored] == 2.0
        for f in (
            d -> (delete!(d, query); 1.0),
            d -> pop!(d, query),
            d -> (d[query]=3.0; d[query]),
        )
            d = Dict{Any,Float64}(stored => 2.0)
            c = Mooncake.build_rrule(f, d)
            for _ in 1:2
                _, g = Mooncake.value_and_gradient!!(c, f, d)
                @test only(keys(d)) === stored
                @test only(keys(g[2])) === stored
                @test d[stored] == 2.0
            end
        end
    end
    @testset "output cotangent keys" for friendly in (false, true), nested in (false, true)
        f(x) = Dict(1 => x[1]^2, 2 => x[1] * x[2])
        wrap(x) = nested ? (x,) : x
        g(x) = wrap(f(x))
        x = [1.3, 0.7]
        config = Mooncake.Config(; friendly_tangents=friendly)
        c = Mooncake.prepare_pullback_cache(g, x; config)
        @test Mooncake.value_and_pullback!!(c, wrap(Dict(1 => 1.0, 2 => 1.0)), g, x)[2][2] ≈
            [3.3, 1.3]
        for dy in (Dict(1 => 1.0), Dict(1 => 1.0, 2 => 1.0, 3 => 9.0))
            @test_throws "cotangent must have exactly the primal's keys" Mooncake.value_and_pullback!!(
                c, wrap(dy), g, x
            )
        end
    end
end
