using Mooncake: generate_data_test_cases

function generate_mem()
    return rrule!!(zero_fcodual(Memory{Float64}), zero_fcodual(undef), zero_fcodual(10))
end

@testset "memory" begin
    @testset "$(typeof(p))" for p in generate_data_test_cases(StableRNG, Val(:memory))
        TestUtils.test_data(sr(123), p)
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:memory))

    # Check that the rule for `Memory{P}` only produces two allocations.
    generate_mem()
    @test TestUtils.count_allocs(generate_mem) <= 2

    # Check that zero_tangent and randn_tangent yield consistent results.
    @testset "$f" for f in [zero_tangent, Base.Fix1(randn_tangent, Xoshiro(123))]
        arr = randn(2)
        p = [arr, arr.ref.mem]
        @test pointer(p[1].ref.mem) === pointer(p[2])
        t = f(p)
        @test pointer(t[1].ref.mem) === pointer(t[2])
    end

    # Regression: an isbits element-wise `Memory{P}(undef, n)` V must be filled with
    # coherent zero duals — readable garbage partials would propagate as nonzero
    # derivatives via whole-buffer copies.
    @testset "element-wise undef Memory V zero partials (width $N)" for N in (1, 2, 3)
        # Dirty the heap so an unzeroed partial buffer would read back nonzero — without this the
        # guard is vacuous (a fresh `undef` buffer commonly reads zero by chance).
        let junk = Memory{Float64}[]
            for _ in 1:200
                m = Memory{Float64}(undef, 16)
                fill!(m, 12345.0)
                push!(junk, m)
            end
        end
        GC.gc(false)
        r = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(N), Memory{Tuple{Float64,Int}}),
            Mooncake.zero_lifted(Val(N), undef),
            Mooncake.zero_lifted(Val(N), 4),
        )
        @test all(i -> all(iszero, tangent(r)[i][1].partials), 1:4)
        @test all(i -> tangent(r)[i][1].value === primal(r)[i][1], 1:4)
    end

    # Regression: the NDualEltype `Core.memorynew` frule (Julia 1.12+ array lowering) must zero
    # each partial buffer like its `Memory{P}(undef, n)` sibling — bare `Core.memorynew` returns
    # uninitialized memory that whole-buffer copies would propagate as nonzero partials.
    @static if VERSION >= v"1.12-"
        @testset "Core.memorynew NDualEltype V zero partials (width $N)" for N in (1, 2, 3)
            # Dirty the heap so an unzeroed buffer would read back nonzero (deterministic guard).
            let junk = Memory{Float64}[]
                for _ in 1:200
                    m = Core.memorynew(Memory{Float64}, 16)
                    fill!(m, 12345.0)
                    push!(junk, m)
                end
            end
            GC.gc(false)
            r = Mooncake.frule!!(
                Mooncake.zero_lifted(Val(N), Core.memorynew),
                Mooncake.zero_lifted(Val(N), Memory{Float64}),
                Mooncake.zero_lifted(Val(N), 8),
            )
            @test all(p -> all(iszero, p), tangent(r).partials)
        end
    end
end
