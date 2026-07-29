struct MirrorPoly{T}
    coeffs::Vector{T}
end

Mooncake.@mirror_tangent MirrorPoly{T} => MirrorPoly{tangent_type(T)}

# A field carrying reverse data cannot be represented by a primal-shaped tangent.
struct MirrorScaled{T}
    coeffs::Vector{T}
    scale::T
end

Mooncake.@mirror_tangent MirrorScaled{T} => MirrorScaled{tangent_type(T)}

@testset "mirror_tangents" begin
    @test tangent_type(MirrorPoly{Float64}) == MirrorPoly{Float64}
    TestUtils.test_data(Random.default_rng(), MirrorPoly([1.0, 2.0, 3.0]))

    @testset "gradient is returned as the primal's own type" begin
        f(p) = sum(abs2, p.coeffs)
        p = MirrorPoly([1.0, 2.0, 3.0])
        _, grads = Mooncake.value_and_gradient!!(build_rrule(f, p), f, p)
        @test grads[2] isa MirrorPoly{Float64}
        @test grads[2].coeffs ≈ 2 .* p.coeffs
    end

    # The aliasing invariant: a wrapper sharing storage with a loose array shares its tangent.
    @testset "aliasing" begin
        v = randn(3)
        t = Mooncake.zero_tangent((v, MirrorPoly(v)))
        @test t[1] === t[2].coeffs
    end

    # A scalar field's tangent is reverse data, so the mirror would have to be split in two.
    @testset "rejects fields carrying reverse data" begin
        @test_throws ArgumentError rdata_type(MirrorScaled{Float64})
    end
end
