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

    # Isbits undef Memory V needs coherent zero partials before whole-buffer copies.
    @testset "element-wise undef Memory V zero partials (width $N)" for N in (1, 2, 3)
        # Dirty the heap so an unzeroed buffer cannot pass merely by reusing zeroed pages.
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

    # Core.memorynew must zero NDualEltype partials too (Julia 1.12+ array lowering).
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
            # Zero-byte tangent buffers need a NoTangent layout tag to refuse retyped accesses.
            # The registry cannot express this: Ptr seeding yields an uninit_* placeholder.
            m8 = Memory{UInt8}(undef, 8)
            o = Mooncake.rrule!!(
                Mooncake.zero_fcodual(Mooncake.lgetfield),
                Mooncake.CoDual(m8, Mooncake.zero_tangent(m8)),
                Mooncake.zero_fcodual(Val(:ptr)),
                Mooncake.zero_fcodual(Val(:not_atomic)),
            )[1]
            @test tangent(o).elt === Mooncake.NoTangent
            # Differentiable elements retain the real tangent address and layout tag.
            mf = Memory{Float64}(undef, 2)
            tf = Mooncake.zero_tangent(mf)
            o = Mooncake.rrule!!(
                Mooncake.zero_fcodual(Mooncake.lgetfield),
                Mooncake.CoDual(mf, tf),
                Mooncake.zero_fcodual(Val(:ptr)),
                Mooncake.zero_fcodual(Val(:not_atomic)),
            )[1]
            @test UInt(tangent(o).p) == UInt(tf.ptr)
            @test tangent(o).elt === Float64
            # Dereferencing a NULL tangent pointer is refused instead of faulting.
            @test_throws ArgumentError Mooncake.rrule!!(
                Mooncake.zero_fcodual(Mooncake.IntrinsicsWrappers.pointerref),
                Mooncake.CoDual(Ptr{Float64}(pointer(m8)), Ptr{Float64}(0)),
                Mooncake.zero_fcodual(1),
                Mooncake.zero_fcodual(1),
            )
            # Ptr{NoTangent} dereferences touch no bytes and must remain safe even at NULL.
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

        @static if VERSION >= v"1.12-"
            @testset "Core.memorynew hands back a ZERO tangent" begin
                # Dirty the heap and test the fresh tangent directly: end-to-end gradients can
                # hide uninitialised partials when freed pages happen to contain zeros.
                junk = [fill(-77.0, 64) for _ in 1:400]
                GC.@preserve junk nothing
                junk = nothing
                GC.gc(false)
                for _ in 1:8
                    out, _ = Mooncake.rrule!!(
                        Mooncake.zero_fcodual(Core.memorynew),
                        Mooncake.zero_fcodual(Memory{Float64}),
                        Mooncake.zero_fcodual(64),
                    )
                    @test all(iszero, tangent(out))
                end
            end
        end
    end
end
