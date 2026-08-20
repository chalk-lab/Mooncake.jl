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
            # Dirty the heap so an unzeroed buffer would likely read back nonzero.
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
            @test all(iszero, tangent(r).partials_block)
        end
    end

    @static if VERSION >= v"1.11-rc4"
        @testset "no tangent pointer for a zero-size element type" begin
            # A `Memory{UInt8}`'s tangent is a `Memory{NoTangent}`, whose buffer holds no bytes, so
            # handing out its address let a re-typed `pointerset` write `sizeof(Float64)` bytes into
            # a zero-byte allocation — a segfault. The registry cannot express this: seeding a `Ptr`
            # primal yields the `uninit_*` placeholder, never NULL.
            m8 = Memory{UInt8}(undef, 8)
            o = Mooncake.rrule!!(
                Mooncake.zero_fcodual(Mooncake.lgetfield),
                Mooncake.CoDual(m8, Mooncake.zero_tangent(m8)),
                Mooncake.zero_fcodual(Val(:ptr)),
                Mooncake.zero_fcodual(Val(:not_atomic)),
            )[1]
            @test iszero(UInt(tangent(o)))
            # A differentiable element type still gets its real tangent buffer's address.
            mf = Memory{Float64}(undef, 2)
            tf = Mooncake.zero_tangent(mf)
            o = Mooncake.rrule!!(
                Mooncake.zero_fcodual(Mooncake.lgetfield),
                Mooncake.CoDual(mf, tf),
                Mooncake.zero_fcodual(Val(:ptr)),
                Mooncake.zero_fcodual(Val(:not_atomic)),
            )[1]
            @test UInt(tangent(o)) == UInt(tf.ptr)
            # Dereferencing a NULL tangent pointer is refused instead of faulting.
            @test_throws ArgumentError Mooncake.rrule!!(
                Mooncake.zero_fcodual(Mooncake.IntrinsicsWrappers.pointerref),
                Mooncake.CoDual(Ptr{Float64}(pointer(m8)), Ptr{Float64}(0)),
                Mooncake.zero_fcodual(1),
                Mooncake.zero_fcodual(1),
            )
            # The guard must NOT fire for a zero-size tangent element type: a non-differentiable
            # store carries `Ptr{NoTangent}`, whose deref touches no bytes and is safe even at
            # NULL. A blanket null check rejected this and broke a registered `pointerset` case.
            Mooncake.rrule!!(
                Mooncake.zero_fcodual(Mooncake.IntrinsicsWrappers.pointerset),
                Mooncake.CoDual(Ptr{UInt8}(pointer(m8)), Ptr{Mooncake.NoTangent}(0)),
                Mooncake.zero_fcodual(UInt8(3)),
                Mooncake.zero_fcodual(1),
                Mooncake.zero_fcodual(1),
            )
            @test true  # reaching here without throwing is the assertion

            # End to end: reinterpreting a byte buffer as floats under reverse AD used to segfault.
            fbytes(buf, x) =
                (p=Ptr{Float64}(pointer(buf)); unsafe_store!(p, x); unsafe_load(p))
            @test_throws ArgumentError Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(fbytes, Memory{UInt8}(undef, 8), 7.0),
                fbytes,
                Memory{UInt8}(undef, 8),
                7.0,
            )
        end
    end
end
