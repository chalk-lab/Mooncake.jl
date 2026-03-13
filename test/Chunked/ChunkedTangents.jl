# NOTE !!!!
# This prototype assumes that the functions undergoing Chunked Forward Mode AD are Frechet and Gateaux differentiable at the specific primal Point.
# This condition is critical for the seeding probes (current version) to work correctly. 

# Fun Fact: Mooncake Testing Infra abides by both these assumptions, but the rules themselves are valid for all Gateaux differentiable function.
# This means we can handle Chunked FM AD for even Gateaux differentiable points!!
# TODO - Going beyond this ? handle Handle Gateux Differentiability (see below) :
# For a ChunkedPrimitiveTangent{T,N} we must do N derivative lane passes (all N directions seeding probes into the Jacobian) - N input sets to the function f, scaling with Chunk size itself.
# This completely handles non-Fréchet differentiable spaces in F. The tradeof however, is that we scale with N times more operations instead of 1 rule lane pass.
# This is critical and imho even acceptable to get the gradients correct for "crumpled" manifolds - because there is simply more exotic "information".

using Mooncake
import Mooncake:
    tangent_type,
    frule!!,
    NoTangent,
    zero_tangent,
    uninit_tangent,
    Dual,
    primal,
    tangent,
    @is_primitive,
    MinimalCtx

# For Randomized Linearity Testing.
using LinearAlgebra, Random

# For testing
using Test

"""
Union of types that can appear inside a `ChunkedPrimitiveTangent` Tangent.
"""
const PrimitiveTangents = Union{Base.IEEEFloat,NoTangent}

"""
    ChunkedPrimitiveTangent{T<:PrimitiveTangents, N}

Holds N tangent directions for a single primitive-typed primal value.
Stored as NTuple{N,T} - contiguous per field (Struct-of-Arrays), SIMD friendly.

For N <= 16: lives entirely in registers, fully unrolled by the Julia compiler.
"""
# TODO : For N > 16: consider `ChunkArrayTangent` (Based on Option 2) - values isa Matrix etc ?
# Treat Array
# struct ChunkedArrayTangent{T<:PrimitiveTangents,N}
# values::Matrix{T}
# end

struct ChunkedPrimitiveTangent{T<:PrimitiveTangents,N}
    values::NTuple{N,T}
end

function ChunkedPrimitiveTangent(
    ::Type{T}, ::Val{N}; init=zero(T)
) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(ntuple(_ -> init, Val(N)))
end

# MOONCAKE TANGENT INTERFACE

function Mooncake.zero_tangent(
    p::R, c::ChunkedPrimitiveTangent{T,N}
) where {R<:Real,T<:PrimitiveTangents,N}
    @assert Mooncake.tangent_type(p) == T
    return ChunkedPrimitiveTangent{T,N}(map(x -> Mooncake.zero_tangent(x), c.values))
end

function Mooncake.tangent_type(::Type{T}, ::Val{N}) where {T<:Real,N}
    return ChunkedPrimitiveTangent{Mooncake.tangent_type(T),N}
end

# SOME USEFULL CHUNK UTILITIES
"""
Create a Mooncake Chunked Tangent being 0 in all directions.
Currently only supports `ChunkedPrimitiveTangent` creation.
"""
# TODO: add dipsatch for NamedTuples, Matrices, Arrays etc.
function zero_chunk(::Type{T}, ::Val{N}) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(ntuple(_ -> zero(T), Val(N)))
end

"""
Create a Mooncake Chunked Tangent with `N` explicit tangent directions from `dirs`.
Currently only supports `ChunkedPrimitiveTangent` creation.
"""
# TODO: add dipsatch for NamedTuples, Matrices, Arrays etc.
function make_chunked_tangent(dirs::NTuple{N,T}) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(dirs)
end

"""
Create a Mooncake Chunked Tangent with value 1 in direction `dir` (Index of value), 0 elsewhere.
Used to seed the dir-th column within a chunk for the JVP.
Currently only supports seeding `ChunkedPrimitiveTangent`.
"""
# TODO: add dipsatch for NamedTuples, Matrices, Arrays etc.
function seed_chunk(::Type{T}, ::Val{N}, dir::Int) where {T<:PrimitiveTangents,N}
    @assert 1 <= dir <= N "dir=$dir out of range 1:$N"
    return ChunkedPrimitiveTangent{T,N}(ntuple(i -> i == dir ? one(T) : zero(T), Val(N)))
end

"""
Pull direction i out of Mooncake Chunked Tangent - usefull for calling frule!!
Currently supports only `ChunkedPrimitiveTangent`, `PrimitiveTangents`.
"""
# TODO: add dipsatch for NamedTuples, Matrices, Arrays etc.
extract_direction(d::ChunkedPrimitiveTangent{T,N}, i::Int) where {T,N} = d.values[i]
extract_direction(d::T, i::Int) where {T<:PrimitiveTangents} = d

"""
Flatten any output tangent into a flat Vector{T}
"""
flatten_tangent(t::T) where {T<:PrimitiveTangents} = [t]
flatten_tangent(t::AbstractArray) = vec(collect(t))
flatten_tangent(t::Tuple) = reduce(vcat, map(flatten_tangent, t))
flatten_tangent(t::NamedTuple) = reduce(vcat, map(flatten_tangent, values(t)))
function flatten_tangent(t)
    isempty(fieldnames(typeof(t))) &&
        error("No flatten_tangent dispatch for the type $(typeof(t)), it has no fields.")
    return reduce(vcat, map(f -> flatten_tangent(getfield(t, f)), fieldnames(typeof(t))))
end

"""
How many scalar dims does this primal's tangent occupy ?
"""
tangent_dim(x::Real) = 1
tangent_dim(x::AbstractArray{<:Real}) = length(x)
tangent_dim(x::Tuple) = sum(tangent_dim, x)
tangent_dim(x::NamedTuple) = sum(tangent_dim, values(x))
function tangent_dim(x)
    isempty(fieldnames(typeof(x))) &&
        error("No tangent_dim dispatch for the type $(typeof(x)), it has no fields.")
    return sum(f -> tangent_dim(getfield(x, f)), fieldnames(typeof(x)))
end

"""
Unflatten chunks back into the right output tangent structure.
"""
function unflatten_tangent(chunks::Vector, primal_ref::Real, offset::Int)
    return chunks[offset + 1], offset + 1
end

function unflatten_tangent(chunks::Vector, primal_ref::AbstractArray, offset::Int)
    k = length(primal_ref)
    # Reshape flat slice to original shape
    result = reshape(chunks[(offset + 1):(offset + k)], size(primal_ref))
    return result, offset + k
end

function unflatten_tangent(chunks::Vector, primal_ref::Tuple, offset::Int)
    results = []
    for elem in primal_ref
        val, offset = unflatten_tangent(chunks, elem, offset)
        push!(results, val)
    end
    return Tuple(results), offset
end

function unflatten_tangent(chunks::Vector, primal_ref::NamedTuple, offset::Int)
    results = []
    for elem in values(primal_ref)
        val, offset = unflatten_tangent(chunks, elem, offset)
        push!(results, val)
    end
    return NamedTuple{keys(primal_ref)}(Tuple(results)), offset
end

function unflatten_tangent(chunks::Vector, primal_ref, offset::Int)
    isempty(fieldnames(typeof(primal_ref))) && error(
        "No unflatten_tangent dispatch for the type $(typeof(primal_ref)), it has no fields",
    )
    field_vals = []
    for f in fieldnames(typeof(primal_ref))
        val, offset = unflatten_tangent(chunks, getfield(primal_ref, f), offset)
        push!(field_vals, val)
    end
    return typeof(primal_ref)(field_vals...), offset
end

"""
Recursive approximate equality for any output type. Usefull in Testing Chunked Forward Mode AD primal output, Tangents.s
"""
approx_equal(a::Real, b::Real) = a ≈ b
approx_equal(a::AbstractArray, b::AbstractArray) = all(a .≈ b)
approx_equal(a::Tuple, b::Tuple) = all(approx_equal(ai, bi) for (ai, bi) in zip(a, b))
function approx_equal(a, b)
    try
        return a == b
    catch
        error("no approx_equal() dispatch for the type $(typeof(a)), write the same.")
    end
end

# GENERAL ACCESSORS

Base.getindex(c::ChunkedPrimitiveTangent, ind::Int) = c.values[ind]
Base.length(::ChunkedPrimitiveTangent{T,N}) where {T,N} = N
chunk_size(::ChunkedPrimitiveTangent{T,N}) where {T,N} = N
chunk_size(::Type{ChunkedPrimitiveTangent{T,N}}) where {T,N} = N
function Base.show(io::IO, c::ChunkedPrimitiveTangent{T,N}) where {T,N}
    return print(io, "ChunkedPrimitiveTangent{$T,$N}$(c.values)")
end

# ARITHMETIC OPERATIONS

# ChunkedPrimitiveTangent scaling, translation :
# Note : Chunks must have same T,N because they live in the same tangent space.
function Base.:+(
    a::ChunkedPrimitiveTangent{T,N}, b::ChunkedPrimitiveTangent{T,N}
) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(+, a.values, b.values))
end

function Base.:-(
    a::ChunkedPrimitiveTangent{T,N}, b::ChunkedPrimitiveTangent{T,N}
) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(-, a.values, b.values))
end

function Base.:-(c::ChunkedPrimitiveTangent{T,N}) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(-, c.values))
end

function Base.:*(
    a::ChunkedPrimitiveTangent{T,N}, b::ChunkedPrimitiveTangent{T,N}
) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(*, a.values, b.values))
end

function Base.:/(
    a::ChunkedPrimitiveTangent{T,N}, b::ChunkedPrimitiveTangent{T,N}
) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(/, a.values, b.values))
end

# NoTangent absorption
Base.:+(::NoTangent, ::NoTangent) = NoTangent()
Base.:-(::NoTangent, ::NoTangent) = NoTangent()
Base.:*(::NoTangent, ::NoTangent) = NoTangent()
Base.:/(::NoTangent, ::NoTangent) = NoTangent()

# Scalar & ChunkedPrimitiveTangent scaling, translation :
# Note : Must preserve tangent space properties (primitive tangent type T), chunksize N
function Base.:+(s::T, c::ChunkedPrimitiveTangent{T,N}) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(x -> s + x, c.values))
end

function Base.:+(c::ChunkedPrimitiveTangent{T,N}, s::T) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(x -> x + s, c.values))
end

function Base.:-(s::T, c::ChunkedPrimitiveTangent{T,N}) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(x -> s - x, c.values))
end

function Base.:-(c::ChunkedPrimitiveTangent{T,N}, s::T) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(x -> x - s, c.values))
end

function Base.:*(s::T, c::ChunkedPrimitiveTangent{T,N}) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(x -> s * x, c.values))
end

function Base.:*(c::ChunkedPrimitiveTangent{T,N}, s::T) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(x -> x * s, c.values))
end

function Base.:/(c::ChunkedPrimitiveTangent{T,N}, s::T) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(x -> x / s, c.values))
end

function Base.:/(s::T, c::ChunkedPrimitiveTangent{T,N}) where {T<:PrimitiveTangents,N}
    return ChunkedPrimitiveTangent{T,N}(map(x -> s / x, c.values))
end

"""
    apply_frechet_linearmap(basis_jvps, chunked_tangents...)

Apply a precomputed basis JVP to a set of chunked tangents.
`basis_jvps` is a m × n Matrix (m outputs, n inputs), where basis_jvps[m,n] = ∂out_m/∂input_n
`chunked_tangents` are n Mooncake Chunked Tangents, all with same chunk size N.

Currently this supports only `ChunkedPrimitiveTangents`.
"""
function apply_frechet_linearmap(
    basis_jvps::Matrix, chunked_tangents::ChunkedPrimitiveTangent{T,N}...
) where {N,T<:PrimitiveTangents}
    num_outputs, num_inputs = size(basis_jvps)

    return map(1:num_outputs) do m
        # Seed with first input's Tangent
        acc = map(v -> basis_jvps[m, 1] * v, chunked_tangents[1].values)

        # Accumulate remaining inputs Tangents
        foldl(2:num_inputs; init=acc) do acc, n
            jac_mn = basis_jvps[m, n]              # ∂out_m/∂input_n
            chunk = chunked_tangents[n].values    # NTuple{T, N}
            map((a, v) -> a + jac_mn * v, acc, chunk)
        end
    end
end

"""
    Chunked Tangent dispatch for Mooncake.frule!!

For a function f: R^n -> R^m (with input space R^n and output space R^m)
Triggered when we call to `Mooncake.frule!!` over `Dual` Type arguments with Mooncake tangent.

Note :
1. In General, Forward Mode AD is best used for `f` with n < m. Given this - my Implementation must have optimal Time and Space Complexity.
2. Currently Chunked Forward Mode AD works for `f` where :
    a. Domain : only works for single, multiple arguments with `Real` primal type & `ChunkedPrimitiveTangent` Tangent type.
    b. Range : Single & Multiple for Tuples, AbstractArray, NamedTuple outputs are all supported.
(Point 2 is because currently we have only written helper methods for `ChunkedPrimitiveTangent` input tangents)

# Criteria for Chunked Forward Mode AD to work :
- `f` must be Fréchet differentiable at the primal point for the chunked linear map application to be correct.
#  TODO: make this configurable, by allowing rule derivation, evaluation for whole chunk/intelligent detection of Information Boundaries.
- The callable function `f` can be anything as long as it does not modify it's own state.
"""
# TODO: Allow frule!! to dispatch on Mixed Dual + ChunkedPrimitiveTangent or unequal sized ChunkedPrimitiveTangent arguments.
function Mooncake.frule!!(
    f::Dual{F}, args::Dual{P,ChunkedPrimitiveTangent{T,N}}...
) where {F,N,P<:Real,T<:PrimitiveTangents}

    # We Assume Frechet Differentiability here :
    # 1. This allows for using pre-existing frules dispatched on `Duals`. (Automatically expands with that coverage also)
    # 2. Space complexity is O(N) where N=chunksize, max memory needed is only for Basis JVP & Chunks Matrix Multiplication (can be optimized Further)
    # 3. Time Complexity is just O(n * cost(f)) where :
    #         a. n is number of arguments/inputs.
    #         b. cost(f) is the cost of deriving, evaluating primal/differential operator for function f (whichever is higher).

    _args = map(Mooncake.primal, args)
    dargs_chunks = map(Mooncake.tangent, args)
    num_inputs = length(_args)

    # Scalar single rule run for each input
    atomic_frule = Mooncake.build_frule(primal(f), _args...)

    # TODO: Allow calculating derivative for Gateux differentiable functions by calling the frule for each Tangent direction within the Chunks.
    # (Maybe this can be done intelligently ?)
    # Now we apply the Randomized Linearity Test. (supported by the Schwartz Zippel Lemma for Polynomial Surfaces)
    # we continue using this rule ONLY if we can safely say we are working with a function f :
    #       a) Where `f` is Frechet differentiable in all directions within our Chunk.
    #       b) In case the `f` is only Gateux Differentiable, we informatively error out..
    # TODO : Implement the Randomized Linearity Test function - check_linearity_residue(atomic_frule, chunk; tol=1e-8)

    # NOTE: THE ABOVE COMMENTS, TODO are fine to avoid rn as Mooncake's documentation itself mentions that Mooncake's ForwardMode AD Assumes Frechet Differentiability.
    # https://chalk-lab.github.io/Mooncake.jl/dev/tutorial/#Mooncake.jl-Functions

    atomic_directions = map(dargs_chunk -> extract_direction(dargs_chunk, 1), dargs_chunks)   # extract the direction[1] from all chunks

    # TODO: write a one() Mooncake specific helper to cover NamedTuples, all tangent types.
    # For `ChunkedPrimitiveTangent` we only need a scalar unit basis, so one() works directly on scalars.
    atomic_unit_directions = map(
        atomic_direction -> one(atomic_direction), atomic_directions
    )   # converted to a set of standard basis Duals
    _atomic_unit_directions = map(deepcopy, atomic_unit_directions) # mutation safety

    # TODO: write a Diagonal() function to turn a vector of Duals into a Diagonal Matrix.
    # we create a set of standard Basis Vectors in the input space of f (R^n -> R^m).
    basis_dual_sets = map(1:num_inputs) do n
        map(enumerate(_args)) do (i, primal_arg_i)
            if i == n
                Dual(primal_arg_i, _atomic_unit_directions[i])
            else
                Mooncake.zero_dual(primal_arg_i)
            end
        end
    end

    # Call atomic_frule over each standard Basis Vector to get the JVP along that EigenVector.
    # Scales as O(n) with the n = Number of inputs to the function f (R^n -> R^m). Also, We reuse the same rule (avoids scaling with chunksize - N rule derivations)
    # This is good for us as Forward Mode works best with m>n.

    # Preallocate memory for basis JVP 
    atomic_basis_JVP = Vector{Vector{T}}(undef, num_inputs)
    primal_out, _ = foldl(
        enumerate(basis_dual_sets); init=(nothing, atomic_basis_JVP)
    ) do (primal_acc, cols), (n, dual_set)
        temp = atomic_frule(f, dual_set...)
        primal_n, jvp_n = Mooncake.primal(temp), Mooncake.tangent(temp)

        # Since this primal evaluation atleast n times is unavoidable :
        # store the primal output value, error out if it changes across loop.
        if !isnothing(primal_acc)
            approx_equal(primal_acc, primal_n) || error("""
            Mutation detected across basis probes.
            Your function is mutating its state and has memory of its calls!
            This means the primal output changed between basis probes, which 
            invalidates the Jacobian computation. This is not supported yet.
            """)
        end

        cols[n] = flatten_tangent(jvp_n)
        (primal_n, cols)
    end
    basis_jvps = reduce(hcat, atomic_basis_JVP)     # basis_jvps is a m × n Matrix (m outputs, n inputs), where basis_jvps[m,n] = ∂out_m/∂input_n

    # Note: Frechet Differentiablity ensures that we can directly apply a Linear Map across different directions, simply because the manifold is smooth.
    # Since we are dealing with a level set on f's domain that is Frechet Differentiable,
    # We simply scale the above atomic_basis_tangent with all the directions within the chunks.
    # This is normal Matrix/Ntuple multiplication lol (can used optimized routines).
    scaled_tangents = apply_frechet_linearmap(basis_jvps, dargs_chunks...)

    chunks = map(make_chunked_tangent, scaled_tangents)  # Vector{ChunkedPrimitiveTangent}
    chunked_tangents, _ = unflatten_tangent(chunks, primal_out, 0)

    # We already have `primal_out`, this is obviously the primal output for the whole chunk.
    # Note that this is got in O(n) Operations where n = inputs of Function f (R^n -> R^m).
    return Dual(primal_out, chunked_tangents)
end

# TEST CODE

# TODO: Need to maybe write mode code based on : https://chalk-lab.github.io/Mooncake.jl/dev/developer_documentation/custom_tangent_type/#Writing-Custom-Tangent-Types
## Derive frule for f_scalar, then call it with chunkedPrimitiveTangent.
# chunked_frule = Mooncake.build_frule(f_scalar, a, b) # builds the atomic frule for f_scalar
# derived_dual_output = chunked_frule(Mooncake.zero_dual(f_scalar), Dual(a, da), Dual(b, db)) # calls the chunked frule for f_scalar (crashes unfortunately)

Random.seed!(42)
const N = 4 # Chunk size for all tests

# EXAMPLE 1 : f: R^2 -> R
function f_scalar(a, b)
    return sin(a + 2b)
end

# Tangent data for the Duals is randomnly generated.
a, b = randn(), randn()
da_vals = ntuple(_ -> randn(), Val(N))
db_vals = ntuple(_ -> randn(), Val(N))
da, db = ChunkedPrimitiveTangent(da_vals), ChunkedPrimitiveTangent(db_vals)

# Ideal (current Mooncake)
scalar_rule_1 = Mooncake.build_frule(f_scalar, a, b)
# Proposed Chunked Forward Mode AD
chunked_out_1 = Mooncake.frule!!(Mooncake.zero_dual(f_scalar), Dual(a, da), Dual(b, db))

# Test for all directions.
for dir in 1:N
    scalar_ref = scalar_rule_1(
        Mooncake.zero_dual(f_scalar), Dual(a, da_vals[dir]), Dual(b, db_vals[dir])
    )
    @test primal(chunked_out_1) ≈ primal(scalar_ref)
    @test extract_direction(tangent(chunked_out_1), dir) ≈ tangent(scalar_ref)
end

# EXAMPLE 2 : f: R^2 -> R^3
function f_vector(a, b)
    return [sin(a), cos(b), a * b]
end

# Tangent data for the Duals is randomnly generated.
a, b = randn(), randn()
da_vals = ntuple(_ -> randn(), Val(N))
db_vals = ntuple(_ -> randn(), Val(N))
da, db = ChunkedPrimitiveTangent(da_vals), ChunkedPrimitiveTangent(db_vals)

# Ideal (current Mooncake)
scalar_rule_2 = Mooncake.build_frule(f_vector, a, b)
# Proposed Chunked Forward Mode AD
chunked_out_2 = Mooncake.frule!!(Mooncake.zero_dual(f_vector), Dual(a, da), Dual(b, db))

# Test for all directions.
for dir in 1:N
    scalar_ref = scalar_rule_2(
        Mooncake.zero_dual(f_vector), Dual(a, da_vals[dir]), Dual(b, db_vals[dir])
    )
    @test primal(chunked_out_2) ≈ primal(scalar_ref)
    t_chunked = tangent(chunked_out_2)
    t_scalar = tangent(scalar_ref)
    @test all(
        extract_direction(t_chunked[i], dir) ≈ t_scalar[i] for i in eachindex(t_scalar)
    )
end

# EXAMPLE 3 : f: R^2 -> R^(2x2)
function f_matrix(a, b)
    return [sin(a) cos(b); a*b exp(a + b)]
end

# Tangent data for the Duals is randomnly generated.
a, b = randn(), randn()
da_vals = ntuple(_ -> randn(), Val(N))
db_vals = ntuple(_ -> randn(), Val(N))
da, db = ChunkedPrimitiveTangent(da_vals), ChunkedPrimitiveTangent(db_vals)

# Ideal (current Mooncake)
scalar_rule_3 = Mooncake.build_frule(f_matrix, a, b)
# Proposed Chunked Forward Mode AD
chunked_out_3 = Mooncake.frule!!(Mooncake.zero_dual(f_matrix), Dual(a, da), Dual(b, db))

# Test for all directions.
for dir in 1:N
    scalar_ref = scalar_rule_3(
        Mooncake.zero_dual(f_matrix), Dual(a, da_vals[dir]), Dual(b, db_vals[dir])
    )
    @test primal(chunked_out_3) ≈ primal(scalar_ref)
    t_chunked = tangent(chunked_out_3)
    t_scalar = tangent(scalar_ref)
    @test all(
        extract_direction(t_chunked[i, j], dir) ≈ t_scalar[i, j] for
        i in axes(t_scalar, 1), j in axes(t_scalar, 2)
    )
end

# EXAMPLE 4 : f: R^3 -> (R, R^(2x2), R^2)
function f_mixed(a, b, c)
    return (a * b * c, [sin(a) cos(b); b*c exp(a)], [cos(c), a + b])
end

# Tangent data for the Duals is randomnly generated.
a, b, c = randn(), randn(), randn()
da_vals = ntuple(_ -> randn(), Val(N))
db_vals = ntuple(_ -> randn(), Val(N))
dc_vals = ntuple(_ -> randn(), Val(N))
da, db, dc = ChunkedPrimitiveTangent(da_vals),
ChunkedPrimitiveTangent(db_vals),
ChunkedPrimitiveTangent(dc_vals)

# Ideal (current Mooncake)
scalar_rule_5 = Mooncake.build_frule(f_mixed, a, b, c)
# Proposed Chunked Forward Mode AD
chunked_out_5 = Mooncake.frule!!(
    Mooncake.zero_dual(f_mixed), Dual(a, da), Dual(b, db), Dual(c, dc)
)

# Test for all directions.
for dir in 1:N
    scalar_ref = scalar_rule_5(
        Mooncake.zero_dual(f_mixed),
        Dual(a, da_vals[dir]),
        Dual(b, db_vals[dir]),
        Dual(c, dc_vals[dir]),
    )
    @test all(primal(chunked_out_5) .≈ primal(scalar_ref))
    t_chunked = tangent(chunked_out_5)
    t_scalar = tangent(scalar_ref)
    @test extract_direction(t_chunked[1], dir) ≈ t_scalar[1]
    @test all(
        extract_direction(t_chunked[2][i, j], dir) ≈ t_scalar[2][i, j] for
        i in axes(t_scalar[2], 1), j in axes(t_scalar[2], 2)
    )
    @test all(
        extract_direction(t_chunked[3][i], dir) ≈ t_scalar[3][i] for
        i in eachindex(t_scalar[3])
    )
end

# IN PROGRESS STUFF :

# TODO: padding/augmentation for unequal N ? (More support)
# """
# Auto promote Duals with scalar OR Chunked Mooncake Tangents to a uniform ChunkedTangent of chunk size N.
# 
# Currently this handles only `ChunkedPrimitiveTangent`.
# """

# function _to_chunked(d::Dual{P,T}, ::Val{N}) where {P<:Real,T<:PrimitiveTangents,N}
#     t = tangent(d)
#     return Dual(primal(d), ChunkedPrimitiveTangent{T,N}(ntuple(_ -> t, Val(N))))
# end

# _to_chunked(d::Dual{P,ChunkedPrimitiveTangent{T,N}}, ::Val{N}) where {P<:Real,T<:PrimitiveTangents,N} = d

# function _to_chunked(d::Dual{P,ChunkedPrimitiveTangent{T,M}}, ::Val{N}) where {P<:Real,T<:PrimitiveTangents,M,N}
#     error("""
#     Chunk size mismatch in chunking candidate $d - found N = $M but expected N = $N.
#     """)
# end
