using Mooncake
using Mooncake: tangent_type, frule!!, rrule!!

# OPTION 1
# Keeping mutable because:
# 1. For large multidim arrays in-place mutations are efficient.
# 2. Can default to accessor functions (set_element!) etc that control access.

mutable struct ChunkedPrimitiveTangents7{T, DataDim}
    ts::Array{T, DataDim}
    _chunk_size::Int
    
    # Directions to take for ChunkedTangents{T,N} for Complex,Int,Floats,Vectors or Matrices :
    # ts cannot be made private in Julia, so document and only access through accessors - The user can change ts length != N
    # or alLow chunksize as another parameter within ChunkedPrimitiveTangents - Probably better.
    # or use StaticArrays, MVectors? - Not Suitable for large N > 100.
    
    # enforce correct chunk size for all outer constructor calls.
    function ChunkedPrimitiveTangents7{T, DataDim}(ts::Array{T, DataDim}, _chunk_size::Int) where {T, DataDim}
        size(ts, DataDim) == _chunk_size || error("Size mismatch")
        new(ts, _chunk_size)
    end
end

# Scalar chunks - (Float64, Int, Complex, etc.)
function ChunkedPrimitiveTangents7(::Type{T}, ::Val{ChunkSize}; init=zero(T)) where {ChunkSize, T<:Union{Base.IEEEFloat, Integer, Complex}}
    ts = fill(init, ChunkSize)
    ChunkedPrimitiveTangents7{T, 1}(ts, ChunkSize)
end

# Vector chunks
function ChunkedPrimitiveTangents7(::Type{Vector{T}}, vec_length::Int, ::Val{ChunkSize}; init=zero(T)) where {ChunkSize, T<:Union{Base.IEEEFloat, Integer, Complex}}
    ts = fill(init, vec_length, ChunkSize)
    ChunkedPrimitiveTangents7{T, 2}(ts, ChunkSize)
end

# matrix chunks
function ChunkedPrimitiveTangents7(::Type{Matrix{T}}, rows::Int, cols::Int, ::Val{ChunkSize}; init=zero(T)) where {ChunkSize, T<:Union{Base.IEEEFloat, Integer, Complex}}
    ts = fill(init, rows, cols, ChunkSize)
    ChunkedPrimitiveTangents7{T, 3}(ts, ChunkSize)
end

# Accessor methods
function get_element(chunk::ChunkedPrimitiveTangents7, idx...)
    size(chunk.ts, ndims(chunk.ts)) == chunk._chunk_size || error("Corrupted chunk! Size mismatch")
    return chunk.ts[idx...]
end

function set_element!(chunk::ChunkedPrimitiveTangents7, val, idx...)
    size(chunk.ts, ndims(chunk.ts)) == chunk._chunk_size || error("Corrupted chunk! Size mismatch")
    chunk.ts[idx...] = val
end


# USAGE
# Scalar chunks
chunk1 = ChunkedPrimitiveTangents7(Float64, Val(100))
chunk2 = ChunkedPrimitiveTangents7(Float64, Val(100); init=2.0)

# using accessors
get_element(chunk2, 5)
set_element!(chunk2, 0.0, 5)
get_element(chunk2, 5)

# Invalid chunk size checking (when user modifies chunks without accessors)
chunk_temp = ChunkedPrimitiveTangents7(Float64, Val(10))
get_element(chunk_temp, 50)  # Out of bounds error
push!(chunk_temp.ts, 5.0)
get_element(chunk_temp, 1)  # Chunk corrupted

# Vector chunks
chunk3 = ChunkedPrimitiveTangents7(Vector{Float64}, 5, Val(100))
chunk4 = ChunkedPrimitiveTangents7(Vector{Float64}, 5, Val(100); init=2.0)

# using accessors
get_element(chunk4, 2, 3)
set_element!(chunk4, 0.0, 2, 3)
get_element(chunk4, 2, 3)

# Invalid chunk size checking (when user modifies chunks without accessors)
chunk_temp = ChunkedPrimitiveTangents7(Vector{Float64}, 5, Val(10))
get_element(chunk_temp, 2, 50)  # Out of bounds error
chunk_temp.ts = cat(chunk_vec.ts, zeros(5, 1); dims=2)  # Add one chunk
get_element(chunk_temp, 1, 1)  # Chunk corrupted

# Matrix chunks
chunk5 = ChunkedPrimitiveTangents7(Matrix{Float64}, 3, 4, Val(100))
chunk6 = ChunkedPrimitiveTangents7(Matrix{Float64}, 3, 4, Val(100); init=2.5)
chunk7 = ChunkedPrimitiveTangents7(Matrix{Int64}, 3, 4, Val(100);init=2)
chunk8 = ChunkedPrimitiveTangents7(Matrix{ComplexF64}, 3, 4, Val(100); init=2.0 + 2.0im)

# using accessors
get_element(chunk8, 1, 2, 5)
set_element!(chunk8, 0.0, 1, 2, 5)
get_element(chunk8, 1, 2, 5)

# Invalid chunk size checking (when user modifies chunks without accessors)
chunk_temp = ChunkedPrimitiveTangents7(Matrix{Float64}, 3, 4, Val(10))
get_element(chunk_temp, 2, 50,50)  # Out of bounds error
chunk_temp.ts = cat(chunk_temp.ts, zeros(3, 4, 1); dims=3)  # Add one chunk
get_element(chunk_temp, 1, 1, 1)  # Chunk corrupted


# TODO : FRULES AND RRULES MUST HANDLE ChunkedPrimitiveTangent
# TODO : Define accessors for relevant operations & Document access violation cases.