using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, CUDA, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_tangent_interface, test_tangent_splitting, test_rule

@testset "cuda" begin
    if CUDA.functional()
        # Check we can operate on CuArrays of various element types.
        @testset for ET in (Float32,)
            # , Float64, ComplexF32, ComplexF64)
            # p = CuArray{ET,2,CUDA.DeviceMemory}(undef, 8, 8)

            # its the undef CuArray that causes issues.
            # we might have a memory block with too large values already present.
            p = CuArray{ET,2,CUDA.DeviceMemory}(
                [
                    3.8854233f24+1.4832072f0im 2.6339196f-16-1.7828647f0im 1.2679982f-31-1.3956923f0im 3.6161333f-34-1.8966274f0im 1.1784188f-14+1.866959f0im 3.833234f32-1.8736119f0im -2.5026026f14+1.745722f0im -3.0818833f-16+1.8844136f0im
                    8.246902f-17+1.6706587f0im 1.2110038f28+1.6966934f0im 9.297032f-25+1.754505f0im 2.352584f-18+1.7683662f0im -3.883463f-37-1.8168418f0im -3.8974874f37+1.764631f0im 6.4534347f14+1.7580796f0im 4.3797045f-24+1.8355393f0im
                    -1.7671618f22-1.7331566f0im -5.7503433f30+1.8785174f0im 3.850432f35+1.077262f0im -6.508398f-5+1.5934811f0im 3.0128017f-16-1.5721159f0im -1.1220466f-23+1.6327752f0im -1.7502249f-38-1.7113417f0im -2.7640808f-12+1.5592481f0im
                    -2.4716033f38+1.8172125f0im -6.031176f-6-1.6571676f0im -30091.836f0+1.9276128f0im -446.08618f0+1.8145128f0im -6.6243224f-13-1.6064521f0im -9.4413705f21+1.154524f0im -1.6669508f22-1.0461488f0im -3.419574f27-1.7609674f0im
                    -2.2760783f22+1.4688432f0im -5.942711f8+1.572029f0im -2.0228646f-26-1.5227581f0im -8.404612f36-1.8330257f0im 19.479572f0+1.3451102f0im 2.0556223f-30+1.9849584f0im -1.5266167f-13+1.5752516f0im 0.0009217479f0-1.6794002f0im
                    6.396492f13+1.8614336f0im 1.1194592f-37-1.1724205f0im -6.1285413f-24-1.6526783f0im 5.589568f-39-1.993986f0im 7.6547125f33+1.7877249f0im -4.8981534f30+1.6293379f0im 1.2818821f31+1.7093843f0im 2.7704724f-25-1.7584853f0im
                    -127.94369f0-1.8032475f0im 3.182849f-39-1.5041441f0im 6.163061f-8+1.6968126f0im 3.0026957f28-1.8026869f0im -1.2914952f35-1.8850138f0im -2.1154748f-27-1.8465796f0im -322.78043f0-1.7801483f0im 0.123182215f0+1.8484458f0im
                    0.27871618f0+1.6751668f0im 3041.2847f0-1.9276897f0im 7.82246f-32+1.3190353f0im 7574.0405f0-1.7277507f0im -2.100707f19-1.8368495f0im -1.837451f27+1.897618f0im -3.4904618f-28+1.8024384f0im 9.3388335f35+1.7806036f0im
                ],
            )

            test_tangent_interface(StableRNG(123456), p; interface_only=false)
            test_tangent_splitting(StableRNG(123456), p)

            # Check we can instantiate a CuArray.
            test_rule(
                StableRNG(123456),
                CuArray{ET,1,CUDA.DeviceMemory},
                undef,
                256;
                interface_only=true,
                is_primitive=true,
                debug_mode=true,
                mode=Mooncake.ReverseMode,
            )
            dp = Mooncake.zero_codual(p)
            if ET <: Real
                @test Mooncake.arrayify(dp) == (p, Mooncake.zero_tangent(p))
            elseif ET <: Complex
                primal_p, tangent_p = Mooncake.arrayify(dp)
                @test (primal_p, tangent_p) isa
                    Tuple{CuArray{ET,2,CUDA.DeviceMemory},CuArray{ET,2,CUDA.DeviceMemory}}
                @test all(iszero, tangent_p)
            end
        end
    else
        println("Tests are skipped since no CUDA device was found. ")
    end
end
