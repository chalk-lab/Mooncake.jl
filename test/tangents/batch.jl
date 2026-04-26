using Test, LinearAlgebra
using Mooncake
using Mooncake:
    _VmapScalar, BatchContainer, batch_type, _make_batch, _pack_batch,
    _wrap_input, struct_batchable, Tangent, MutableTangent, get_tangent_field,
    vmap_rule!!, vmap, vmap_apply, VmappedFn

# ── Test structs ──────────────────────────────────────────────────────────────
struct BPoint; x::Float64; y::Float64; end
mutable struct BMParticle; pos::BPoint; mass::Float64; end
Mooncake.struct_batchable(::Type{BPoint})     = true
Mooncake.struct_batchable(::Type{BMParticle}) = true

@testset "vmap / batch" begin

@testset "batch_type" begin
    # _VmapScalar scalars
    @test batch_type(Float64) == BatchContainer{Float64, Vector{Float64}}
    @test batch_type(Float32) == BatchContainer{Float32, Vector{Float32}}
    @test batch_type(ComplexF64) == BatchContainer{ComplexF64, Vector{ComplexF64}}

    # _VmapScalar arrays
    @test batch_type(Vector{Float64}) == BatchContainer{Vector{Float64}, Matrix{Float64}}
    @test batch_type(Matrix{Float64}) == BatchContainer{Matrix{Float64}, Array{Float64,3}}
    @test batch_type(Array{Float64,3}) == BatchContainer{Array{Float64,3}, Array{Float64,4}}

    # non-_VmapScalar arrays
    @test batch_type(Vector{Int}) == BatchContainer{Vector{Int}, Matrix{Int}}
    @test batch_type(Array{String,2}) == BatchContainer{Array{String,2}, Array{String,3}}

    # non-differentiable scalars — identity sentinel
    @test batch_type(Bool)    == Bool
    @test batch_type(Int64)   == Int64
    @test batch_type(Symbol)  == Symbol
    @test batch_type(Nothing) == Nothing

    # Tuples
    @test batch_type(Tuple{Float64, Float64}) ==
        Tangent{NamedTuple{(:_1,:_2), Tuple{BatchContainer{Float64,Vector{Float64}}, BatchContainer{Float64,Vector{Float64}}}}}
    @test batch_type(Tuple{Float64, Int}) ==
        Tangent{NamedTuple{(:_1,:_2), Tuple{BatchContainer{Float64,Vector{Float64}}, Int64}}}

    # NamedTuples
    NT = @NamedTuple{x::Float64, y::Float64}
    bt = batch_type(NT)
    @test bt == Tangent{NamedTuple{(:x,:y), Tuple{BatchContainer{Float64,Vector{Float64}}, BatchContainer{Float64,Vector{Float64}}}}}

    # @struct_batch structs
    @test batch_type(BPoint) == Tangent{NamedTuple{(:x,:y), Tuple{BatchContainer{Float64,Vector{Float64}}, BatchContainer{Float64,Vector{Float64}}}}}
    @test batch_type(BMParticle) <: MutableTangent
end

@testset "BatchContainer basics" begin
    # Scalar batch
    bc = BatchContainer{Float64, Vector{Float64}}([1.0, 2.0, 3.0], 1)
    @test length(bc) == 3
    @test bc[1] == 1.0
    @test bc[2] == 2.0

    # Vector batch
    data = [1.0 3.0; 2.0 4.0]  # 2×2, each column is a batch element
    bc2 = BatchContainer{Vector{Float64}, Matrix{Float64}}(data, 2)
    @test length(bc2) == 2
    @test bc2[1] == [1.0, 2.0]
    @test bc2[2] == [3.0, 4.0]
end

@testset "_make_batch" begin
    # scalar
    bc = _make_batch(2.0, 4)
    @test bc isa BatchContainer{Float64, Vector{Float64}}
    @test length(bc) == 4
    @test all(bc.data .== 2.0)

    # vector
    bc = _make_batch([1.0, 2.0], 3)
    @test bc isa BatchContainer{Vector{Float64}, Matrix{Float64}}
    @test size(bc.data) == (2, 3)
    @test bc.data[:, 1] == [1.0, 2.0]

    # matrix
    A = [1.0 2.0; 3.0 4.0]
    bc = _make_batch(A, 3)
    @test bc isa BatchContainer{Matrix{Float64}, Array{Float64,3}}
    @test size(bc.data) == (2, 2, 3)

    # Tuple (@generated — type-stable)
    bc = _make_batch((1.0, 2.0f0), 4)
    @test bc isa Tangent
    @test get_tangent_field(bc, 1) isa BatchContainer{Float64, Vector{Float64}}
    @test get_tangent_field(bc, 2) isa BatchContainer{Float32, Vector{Float32}}
    @test all(get_tangent_field(bc, 1).data .== 1.0)
    @inferred _make_batch((1.0, 2.0f0), 4)

    # NamedTuple (@generated — type-stable)
    bc = _make_batch((x=1.0, y=2.0f0), 4)
    @test bc isa Tangent
    @test get_tangent_field(bc, :x) isa BatchContainer{Float64, Vector{Float64}}
    @test get_tangent_field(bc, :y) isa BatchContainer{Float32, Vector{Float32}}
    @inferred _make_batch((x=1.0, y=2.0f0), 4)

    # @struct_batch immutable
    bc = _make_batch(BPoint(1.0, 2.0), 4)
    @test bc isa Tangent
    @test get_tangent_field(bc, :x) isa BatchContainer{Float64, Vector{Float64}}
    @test all(get_tangent_field(bc, :x).data .== 1.0)
    @test all(get_tangent_field(bc, :y).data .== 2.0)

    # @struct_batch mutable (nested struct)
    mp = BMParticle(BPoint(1.0, 2.0), 3.0)
    bc = _make_batch(mp, 2)
    @test bc isa MutableTangent
    @test get_tangent_field(bc, :mass) isa BatchContainer{Float64, Vector{Float64}}
    @test get_tangent_field(bc, :pos) isa Tangent

    # non-_VmapScalar array
    bc = _make_batch(Int[1, 2, 3], 4)
    @test bc isa BatchContainer{Vector{Int}, Matrix{Int}}
    @test size(bc.data) == (3, 4)

    # non-batchable type — fill fallback
    @test _make_batch(:hello, 3) == [:hello, :hello, :hello]
    @test _make_batch(true, 2)   == [true, true]
end

@testset "_pack_batch" begin
    # scalars
    bc = _pack_batch([1.0, 2.0, 3.0])
    @test bc isa BatchContainer{Float64, Vector{Float64}}
    @test bc.data == [1.0, 2.0, 3.0]

    # vectors
    bc = _pack_batch([[1.0, 2.0], [3.0, 4.0]])
    @test bc isa BatchContainer{Vector{Float64}, Matrix{Float64}}
    @test size(bc.data) == (2, 2)
    @test bc.data[:, 1] == [1.0, 2.0]
    @test bc.data[:, 2] == [3.0, 4.0]

    # Tuples (@generated — type-stable)
    bc = _pack_batch([(1.0, 2.0f0), (3.0, 4.0f0)])
    @test bc isa Tangent
    @test get_tangent_field(bc, 1) isa BatchContainer{Float64, Vector{Float64}}
    @test get_tangent_field(bc, 2) isa BatchContainer{Float32, Vector{Float32}}
    @test get_tangent_field(bc, 1).data == [1.0, 3.0]
    @inferred _pack_batch([(1.0, 2.0f0), (3.0, 4.0f0)])

    # NamedTuples (@generated — type-stable)
    bc = _pack_batch([(x=1.0, y=2.0f0), (x=3.0, y=4.0f0)])
    @test bc isa Tangent
    @test get_tangent_field(bc, :x).data == [1.0, 3.0]
    @test get_tangent_field(bc, :y).data == [2.0f0, 4.0f0]
    @inferred _pack_batch([(x=1.0, y=2.0f0), (x=3.0, y=4.0f0)])

    # @struct_batch
    bc = _pack_batch([BPoint(1.0, 2.0), BPoint(3.0, 4.0)])
    @test bc isa Tangent
    @test get_tangent_field(bc, :x).data == [1.0, 3.0]
    @test get_tangent_field(bc, :y).data == [2.0, 4.0]

    # non-_VmapScalar arrays
    bc = _pack_batch([Int[1,2], Int[3,4], Int[5,6]])
    @test bc isa BatchContainer{Vector{Int}, Matrix{Int}}
    @test size(bc.data) == (2, 3)
end

@testset "_wrap_input" begin
    # scalars
    bc = _wrap_input([1.0, 2.0, 3.0])
    @test bc isa BatchContainer{Float64, Vector{Float64}}

    # vectors
    bc = _wrap_input([[1.0, 2.0], [3.0, 4.0]])
    @test bc isa BatchContainer{Vector{Float64}, Matrix{Float64}}
    @test size(bc.data) == (2, 2)

    # Tuples (@generated — type-stable)
    bc = _wrap_input([(1.0, 2.0f0), (3.0, 4.0f0)])
    @test bc isa Tangent
    @test get_tangent_field(bc, 1) isa BatchContainer{Float64, Vector{Float64}}
    @test get_tangent_field(bc, 2) isa BatchContainer{Float32, Vector{Float32}}
    @test get_tangent_field(bc, 1).data == [1.0, 3.0]
    @inferred _wrap_input([(1.0, 2.0f0), (3.0, 4.0f0)])

    # NamedTuples (@generated — type-stable)
    bc = _wrap_input([(x=1.0, y=2.0f0), (x=3.0, y=4.0f0)])
    @test bc isa Tangent
    @test get_tangent_field(bc, :x).data == [1.0, 3.0]
    @test get_tangent_field(bc, :y).data == [2.0f0, 4.0f0]
    @inferred _wrap_input([(x=1.0, y=2.0f0), (x=3.0, y=4.0f0)])

    # @struct_batch structs
    bc = _wrap_input([BPoint(1.0, 2.0), BPoint(3.0, 4.0)])
    @test bc isa Tangent
    @test get_tangent_field(bc, :x).data == [1.0, 3.0]
    @test get_tangent_field(bc, :y).data == [2.0, 4.0]

    # non-batchable — pass through unchanged
    @test _wrap_input([:a, :b, :c]) == [:a, :b, :c]
end

@testset "vmap_rule!! scalar ops" begin
    a = BatchContainer{Float64,Vector{Float64}}([1.0,2.0,3.0], 1)
    b = BatchContainer{Float64,Vector{Float64}}([4.0,5.0,6.0], 1)
    @test vmap_rule!!(+, a, b).data == [5.0, 7.0, 9.0]
    @test vmap_rule!!(-, a, b).data == [-3.0, -3.0, -3.0]
    @test vmap_rule!!(*, a, b).data == [4.0, 10.0, 18.0]
    @test vmap_rule!!(*, 2.0, a).data == [2.0, 4.0, 6.0]
    @test vmap_rule!!(*, a, 2.0).data == [2.0, 4.0, 6.0]
end

@testset "vmap_rule!! BLAS matmul" begin
    A = [1.0 0.0; 0.0 2.0]
    # matrix × batch of vectors
    data = [1.0 2.0; 3.0 4.0]   # 2 batch elements of length-2 vectors
    x = BatchContainer{Vector{Float64}, Matrix{Float64}}(data, 2)
    result = vmap_rule!!(*, A, x)
    @test result isa BatchContainer{Vector{Float64}, Matrix{Float64}}
    @test result.data[:, 1] ≈ A * data[:, 1]
    @test result.data[:, 2] ≈ A * data[:, 2]

    # matrix × batch of matrices (reshape trick)
    d3 = cat([1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0]; dims=3)
    xm = BatchContainer{Matrix{Float64}, Array{Float64,3}}(d3, 3)
    result2 = vmap_rule!!(*, A, xm)
    @test result2 isa BatchContainer{Matrix{Float64}, Array{Float64,3}}
    @test result2.data[:,:,1] ≈ A * d3[:,:,1]
    @test result2.data[:,:,2] ≈ A * d3[:,:,2]
end

@testset "vmap_rule!! reductions" begin
    data = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2×3, batch size 3
    bc = BatchContainer{Vector{Float64}, Matrix{Float64}}(data, 2)
    s = vmap_rule!!(sum, bc)
    @test s isa BatchContainer{Float64, Vector{Float64}}
    @test s.data ≈ [5.0, 7.0, 9.0]

    n = vmap_rule!!(LinearAlgebra.norm, bc)
    @test n isa BatchContainer{Float64, Vector{Float64}}
    @test n.data ≈ [norm(data[:,k]) for k in 1:3]
end

@testset "vmap_rule!! element-wise catch-all" begin
    bc = BatchContainer{Float64,Vector{Float64}}([0.0, π/2, π], 1)
    result = vmap_rule!!(sin, bc)
    @test result isa BatchContainer{Float64,Vector{Float64}}
    @test result.data ≈ sin.([0.0, π/2, π])
end

@testset "vmap_rule!! Tangent/Tuple field access" begin
    t = Tangent((
        _1 = BatchContainer{Float64,Vector{Float64}}([1.0,2.0], 1),
        _2 = BatchContainer{Float64,Vector{Float64}}([3.0,4.0], 1),
    ))
    @test vmap_rule!!(Base.getindex, t, 1) === get_tangent_field(t, 1)
    @test vmap_rule!!(Base.getfield, t, :_1) === get_tangent_field(t, :_1)
end

@testset "vmap end-to-end: scalars" begin
    sq = vmap(x -> x^2 + 1.0)
    @test sq isa VmappedFn
    xs = [1.0, 2.0, 3.0, 4.0]
    result = sq(xs)
    @test result isa BatchContainer
    @test collect(result) ≈ [2.0, 5.0, 10.0, 17.0]

    # second call reuses cache — no new entry
    n_before = length(sq.cache)
    result2 = sq(xs)
    @test length(sq.cache) == n_before
    @test collect(result2) ≈ collect(result)

    # different element type recompiles
    sq([1.0f0, 2.0f0, 3.0f0])
    @test length(sq.cache) == n_before + 1

    # empty input fast-path
    @test sq(Float64[]) == Any[]
end

@testset "vmap end-to-end: scalar ops" begin
    # Scalar ops go through our SIMD broadcast rules directly.
    sq2 = vmap(x -> x * x + 1.0)
    xs = [1.0, 2.0, 3.0, 4.0]
    result = sq2(xs)
    @test result isa BatchContainer{Float64, Vector{Float64}}
    @test result.data ≈ [2.0, 5.0, 10.0, 17.0]
end

@testset "vmap end-to-end: matrix multiply (vmap_rule!! direct)" begin
    # End-to-end via vmap: A is a constant in the closure, so accessing it goes
    # through the lifted IR's closure getfield machinery. Test the rule directly instead.
    A = [1.0 0.0; 0.0 2.0]
    data = [1.0 2.0 3.0; 4.0 5.0 6.0]
    x_batch = BatchContainer{Vector{Float64}, Matrix{Float64}}(data, 2)
    result = vmap_rule!!(*, A, x_batch)
    @test result isa BatchContainer{Vector{Float64}, Matrix{Float64}}
    for k in 1:3
        @test result.data[:, k] ≈ A * data[:, k]
    end
end

@testset "vmap end-to-end: NamedTuple input" begin
    f = vmap(nt -> nt.x + nt.y)
    xs = [(x=1.0, y=2.0), (x=3.0, y=4.0), (x=5.0, y=6.0)]
    result = f(xs)
    @test result isa BatchContainer{Float64, Vector{Float64}}
    @test result.data ≈ [3.0, 7.0, 11.0]
end

@testset "vmap end-to-end: @struct_batch" begin
    f = vmap(p -> p.x + p.y)
    pts = [BPoint(1.0, 2.0), BPoint(3.0, 4.0), BPoint(5.0, 6.0)]
    result = f(pts)
    @test result isa BatchContainer{Float64, Vector{Float64}}
    @test result.data ≈ [3.0, 7.0, 11.0]
end

@testset "vmap end-to-end: divergent branch throws" begin
    f = vmap(x -> x > 0.0 ? 1.0 : -1.0)
    @test_throws ErrorException f([1.0, -1.0])
end

@testset "vmap_apply: no cache" begin
    xs = [1.0, 2.0, 3.0]
    result = vmap_apply(x -> x + 1.0, xs)
    @test result isa BatchContainer{Float64, Vector{Float64}}
    @test result.data ≈ [2.0, 3.0, 4.0]
end

end  # @testset "vmap / batch"
