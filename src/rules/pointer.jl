# Avoid troublesome bitcast magic -- we can't handle converting from pointer to UInt,
# because we drop the gradient, because the tangent type of integers is NoTangent.
# https://github.com/JuliaLang/julia/blob/9f9e989f241fad1ae03c3920c20a93d8017a5b8f/base/pointer.jl#L282
@is_primitive MinimalCtx Tuple{typeof(Base.:(+)),Ptr,Integer}
function frule!!(::Dual{typeof(Base.:(+))}, x::Dual{<:Ptr}, y::Dual{<:Integer})
    # A null tangent must stay null: an offset one passes every check and segfaults.
    dx = tangent(x)
    return Dual(primal(x) + primal(y), dx == C_NULL ? dx : dx + primal(y))
end
function rrule!!(f::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    dx = tangent(x)
    return CoDual(primal(x) + primal(y), dx == C_NULL ? dx : dx + primal(y)),
    NoPullback(f, x, y)
end

# Tangent storage must mirror primal bytes: `Memory{NoTangent}` has no storage, and
# the tangent of `Tuple{Int,Float64}` stores only the `Float64`.
"""
    _build_tangent_ptr(dp::Ptr, ::Type{P}, ::Type{T})
    _build_tangent_ptr(dp::Ptr, x::DenseArray{P}, dx::DenseArray{T})

Return the tangent pointer that accompanies a primal pointer, or a null pointer where the
tangent memory does not mirror the primal memory.

`dp` is the address in tangent memory at the primal pointer's offset, and `T` the tangent
type of the elements it points to; `P` is the primal type the pointer reads. `x` and `dx`
are a primal array and its tangent.

Pointer rules repeat each primal access through `dp`. Where the layouts differ, `dp`
points at unrelated bytes or at no storage, so the access gives wrong gradients or a
segfault. Null marks these cases while Base can still take and offset the pointer, e.g. to
check aliasing.

`T` mirrors `P` when their values occupy the same bytes, field by field: `Float64` mirrors
`Float64`, but `NoTangent` does not mirror `Int`, and a bits-union member without a
tangent, such as `Missing`, breaks it.

- Layout: `dp` if `T` mirrors `P`, else null of the same type.
- Storage: the layout check on `P` and `T`, as a `Ptr{NoTangent}`; null for zero-size
  tangent elements, which have no storage (in a `MemoryRef`, `dp` is then an offset, not
  an address).

# Examples

With `dp` a non-null `Ptr{Float64}`:

```julia
_build_tangent_ptr(dp, Float64, Float64)                   # dp
_build_tangent_ptr(dp, Tuple{Int,Float64}, Tuple{NoTangent,Float64})  # null: 16 vs 8 bytes
_build_tangent_ptr(dp, Union{Missing,Float64}, Union{NoTangent,Float64})  # null
_build_tangent_ptr(dp, [1.0], [0.0])                       # dp, as a Ptr{NoTangent}
_build_tangent_ptr(dp, ["a"], [NoTangent()])               # null
```
"""
@generated function _build_tangent_ptr(dp::Ptr, ::Type{P}, ::Type{T}) where {P,T}
    function mirrors(P, T)
        P === T && return true
        inline = Base.allocatedinline(P)
        inline == Base.allocatedinline(T) || return false
        inline || return true  # both stored as references
        Base.aligned_sizeof(P) == Base.aligned_sizeof(T) || return false
        Base.aligned_sizeof(P) == 0 && return true  # e.g. `Nothing` and `NoTangent`
        P <: Ptr && T <: Ptr && return true
        if Base.isbitsunion(P)
            # A member without a tangent, e.g. the `Int` in `Union{Int,Float64}` or `missing`,
            # leaves the tangent bytes of the value it replaces behind.
            Ws = Base.isbitsunion(T) ? Base.uniontypes(T) : Any[T]
            NoTangent in Ws && return false
            Ps = Base.uniontypes(P)
            return all(Q -> any(W -> mirrors(Q, W), Ws), Ps) &&
                   all(W -> any(Q -> mirrors(Q, W), Ps), Ws)
        end
        (P isa DataType && T isa DataType) || return false
        T <: Tangent && (T = fieldtype(T, :fields))
        n = fieldcount(P)
        (n > 0 && n == fieldcount(T)) || return false
        for i in 1:n
            fieldoffset(P, i) == fieldoffset(T, i) || return false
            mirrors(fieldtype(P, i), fieldtype(T, i)) || return false
        end
        return true
    end
    return mirrors(P, T) ? :(dp) : :(typeof(dp)(C_NULL))
end

# MemoryRef fields with zero-sized elements contain an offset rather than an address.
function _build_tangent_ptr(dp::Ptr, x::DenseArray{P}, dx::DenseArray{T}) where {P,T}
    Base.elsize(dx) == 0 && return Ptr{NoTangent}(C_NULL)
    return _build_tangent_ptr(bitcast(Ptr{NoTangent}, dp), P, T)
end

@doc """
    _check_tangent_ptr(dp::Ptr{T}, n::Integer)
    _check_tangent_ptr(dp::Ptr{T})
    _check_tangent_ptr(dp::Ptr{<:Ptr})

Return `true` if a load, store or copy may go through tangent pointer `dp`, `false` if it
can safely be skipped.

`T` is the tangent type of the elements `dp` points to: `Float64` for a `Float64` array,
`NoTangent` for the tangent memory of a `Vector{Int}`, and itself a `Ptr` when the
elements are pointers, as in `Ptr{Ptr{Float64}}`. `n` is the number of elements the access
moves; without it, one.

A non-null `dp` returns `true`. A null `dp` means no tangent storage lies behind the
primal pointer (see [`_build_tangent_ptr`](@ref)), so skipping the access is safe only if
it moves no derivative, which `T` and `n` decide: `T == NoTangent`, `n <= 0`, or a single
pointer value (`T <: Ptr`, no `n`), since a pointer's own tangent may be null. These
return `false`; anything else throws an `ArgumentError` rather than drop a derivative.

# Examples

```julia
_check_tangent_ptr(Ptr{NoTangent}(C_NULL), 3)     # false: skip
_check_tangent_ptr(Ptr{Ptr{Float64}}(C_NULL))     # false: skip
_check_tangent_ptr(Ptr{Float64}(C_NULL))          # throws
_check_tangent_ptr(dp) && unsafe_store!(dp, dv)   # guarded store
```
""" _check_tangent_ptr
# Throws where `false` would make the caller skip an access that carries a derivative.
function _check_tangent_ptr(dp::Ptr{T}, n::Integer) where {T}
    dp != C_NULL && return true
    (T == NoTangent || n <= 0) && return false
    throw(
        ArgumentError(
            "Mooncake cannot differentiate pointer operations without matching tangent storage, " *
            "e.g. a `Ptr{Float64}` into a `Vector{UInt8}` or a `Vector{Tuple{Int,Float64}}`, " *
            "or casting a pointer into differentiable memory to `Ptr{Int}`.",
        ),
    )
end

# Without a count, the access moves one element.
_check_tangent_ptr(dp::Ptr) = _check_tangent_ptr(dp, 1)

# A pointer value may carry a null tangent, so a null `dp` is skipped here; copies, casts
# and `unsafe_pointer_to_objref` pass `n` to require tangent storage.
_check_tangent_ptr(dp::Ptr{<:Ptr}) = dp != C_NULL

#! format: off
function hand_written_rule_test_cases(rng_ctor, ::Val{:pointer})
    _x = Ref(5.0)
    _dx = Ref(4.0)
    test_cases = vcat(
        Any[
            # Rules to avoid pointer type conversions.
            (
                true,
                :stability_and_allocs,
                nothing,
                +,
                CoDual(
                    bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                    bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
                ),
                2,
            ),
        ],
    )
    memory = Any[_x, _dx]
    return test_cases, memory
end
#! format: on

derived_rule_test_cases(rng_ctor, ::Val{:pointer}) = Any[], Any[]
