include(joinpath(@__DIR__, "..", "pin_develop_or_skip.jl"))
pin_develop_or_skip(@__DIR__, "FunctionWrappers")

using AllocCheck, FunctionWrappers, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils:
    TestCase, test_rule, test_tangent_interface, test_tangent_splitting
using FunctionWrappers: FunctionWrapper

@testset "function_wrappers" begin
    include("forward_storage.jl")

    # Tangent interface tests.
    rng = StableRNG(123)
    _data = Ref{Float64}(5.0)
    @testset "$p" for p in Any[
        FunctionWrapper{Float64,Tuple{Float64}}(sin),
        FunctionWrapper{Float64,Tuple{Float64}}(x -> x * _data[]),
    ]
        test_tangent_interface(rng, p)
        test_tangent_splitting(rng, p)

        # Check that we can run `to_cr_tangent` on tangents for FunctionWrappers.
        t = Mooncake.zero_tangent(p)
        @test Mooncake.to_cr_tangent(t) === t
    end

    @testset "lifted mutable capture" begin
        state = Ref(1.0)
        fw = FunctionWrapper{Float64,Tuple{Float64}}(x -> (state[] += x))
        dfw = Mooncake.zero_tangent(fw)
        x = Mooncake.lift(2.0, 1.0)
        @test_throws r"use lift" Mooncake.Lifted{typeof(fw),1}(fw, dfw)
        raw = Mooncake.lift(fw, dfw)
        @test Mooncake.tangent(Mooncake.frule!!(raw, x), 1) == 1.0
        @test Mooncake.tangent(Mooncake.frule!!(raw, x), 1) == 2.0
        for _ in 1:2
            Mooncake.set_to_zero!!(dfw)
            cache = IdDict()
            slot = @inferred Mooncake.lift(fw, dfw, cache)
            @test Mooncake.lift(fw, dfw, cache) === slot
            @test Mooncake.tangent(Mooncake.frule!!(slot, x), 1) == 1.0
            @test Mooncake.tangent(Mooncake.frule!!(slot, x), 1) == 2.0
        end
    end

    for mode in (Mooncake.ForwardMode, Mooncake.ReverseMode)
        Mooncake.TestUtils.run_rule_test_cases(
            StableRNG, Val(:function_wrappers), mode, false
        )
    end

    # Rule testing.
    test_cases = [
        TestCase(function (x, y)
            p = FunctionWrapper{Float64,Tuple{Float64}}(x -> x * y)
            out = 0.0
            for _ in 1:1_000
                out += p(x)
            end
            return out
        end, 5.0, 4.0),
        TestCase(
            function (x::Vector{Float64}, y::Float64)
                p = FunctionWrapper{Float64,Tuple{Float64}}(x -> x * y)
                out = 0.0
                for _x in x
                    out += p(_x)
                end
                return out
            end, randn(100), randn()
        ),
    ]
    for (tc, name) in zip(test_cases, Mooncake.TestUtils._test_case_names(test_cases))
        test_rule(rng, tc; fallbacks=(is_primitive=false,), name)
    end

    test_cases = [
        TestCase(FunctionWrapper{Float64,Tuple{Float64}}, sin),
        TestCase(FunctionWrapper{Float64,Tuple{Float64}}(sin), 5.0),
        TestCase(
            FunctionWrapper{
                Nothing,Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Float64}
            },
            (du, u, p, t) -> (du[1]=p[1] * u[1]; nothing),
        ),
        # Calling that same Nothing-return wrapper (#1005).
        TestCase(
            FunctionWrapper{
                Nothing,Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Float64}
            }(
                (du, u, p, t) -> (du[1]=p[1] * u[1]; nothing)
            ),
            [0.0],
            [1.0],
            [2.0],
            0.5,
        ),
    ]
    for (tc, name) in zip(test_cases, Mooncake.TestUtils._test_case_names(test_cases))
        test_rule(rng, tc; name)
    end

    # Check each output lane for both zero-seeded wrappers and differentiated captures.
    @testset "chunked forward (width N)" begin
        FW = FunctionWrapper{Float64,Tuple{Float64}}
        ndual(x, seeds) = Mooncake.Lifted{Float64,length(seeds)}(
            x, Mooncake.Nfwd.NDual{Float64,length(seeds)}(x, seeds)
        )
        zl(N, v) = Mooncake.zero_lifted(Val(N), v)
        # Fresh interpreter per build (the testset's function defs advance the world age).
        mkfr(sig, N) = Mooncake.build_frule(
            Mooncake.get_interpreter(Mooncake.ForwardMode), sig; chunk_size=N
        )

        call_fw(fw, x) = fw(x)
        fw_sin = FW(sin)
        f_construct(x, y) = FW(t -> t * y)(x)   # x*y; d/dx = y, d/dy = x

        @testset "width $N" for N in (2, 3)
            seeds = ntuple(Float64, N)

            r1 = mkfr(Tuple{typeof(call_fw),FW,Float64}, N)
            o1 = r1(zl(N, call_fw), zl(N, fw_sin), ndual(1.3, seeds))
            @test Mooncake.primal(o1) ≈ sin(1.3)
            @test all(k -> Mooncake.tangent(o1, k) ≈ cos(1.3) * seeds[k], 1:N)

            xs, ys = ntuple(Float64, N), ntuple(k -> -Float64(k), N)
            x0, y0 = 1.3, 2.5
            r2 = mkfr(Tuple{typeof(f_construct),Float64,Float64}, N)
            o2 = r2(zl(N, f_construct), ndual(x0, xs), ndual(y0, ys))
            @test Mooncake.primal(o2) ≈ x0 * y0
            @test all(k -> Mooncake.tangent(o2, k) ≈ y0 * xs[k] + x0 * ys[k], 1:N)

            @test Mooncake.TestUtils.has_equal_data(
                Mooncake.tangent(zl(N, fw_sin), 1), Mooncake.zero_tangent(fw_sin)
            )
        end
        # Width 1 materialises a native tangent without exposing the forward slot.
        let s1 = zl(1, fw_sin)
            @test Mooncake.TestUtils.has_equal_data(
                Mooncake.tangent(s1, 1), Mooncake.zero_tangent(fw_sin)
            )
        end
    end
end
