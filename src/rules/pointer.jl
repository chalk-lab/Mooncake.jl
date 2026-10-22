# Avoid troublesome bitcast magic -- we can't handle converting from pointer to UInt,
# because we drop the gradient, because the tangent type of integers is NoTangent.
# https://github.com/JuliaLang/julia/blob/9f9e989f241fad1ae03c3920c20a93d8017a5b8f/base/pointer.jl#L282
@is_primitive MinimalCtx Tuple{typeof(Base.:(+)),Ptr,Integer}
function frule!!(::Dual{typeof(Base.:(+))}, x::Dual{<:Ptr}, y::Dual{<:Integer})
    return Dual(primal(x) + primal(y), _build_tangent_ptr(tangent(x), primal(y)))
end
function rrule!!(f::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    return CoDual(primal(x) + primal(y), _build_tangent_ptr(tangent(x), primal(y))),
    NoPullback(f, x, y)
end

# Tangent storage must mirror primal bytes: `Memory{NoTangent}` has no storage, and
# the tangent of `Tuple{Int,Float64}` stores only the `Float64`.
"""
    _build_tangent_ptr(dp::Ptr, ::Type{P}, ::Type{T})
    _build_tangent_ptr(dp::Ptr, x::DenseArray, dx::DenseArray)
    _build_tangent_ptr(dp::Ptr, offset::Integer)
    _build_tangent_ptr(::Type{Ptr{P}}, dp::Ptr)

Build the tangent pointer for an access through primal pointer `p`; return null where the
tangent memory does not mirror the primal memory.

Here `dp` is the address at `p`'s offset in tangent memory, and `x` and `dx` are a primal
array and its tangent.

Pointer rules repeat each primal access through `dp`. Where the layouts differ, `dp`
points at unrelated bytes or at no storage, so the access gives wrong gradients or a
segfault. Null marks these cases while Base can still take and offset the pointer, e.g. to
check aliasing.

Tangent type `T` mirrors primal type `P` when their values occupy the same bytes, field by
field; a bits-union member without a tangent, such as `Missing`, breaks this.

- Layout: `dp` if `T` mirrors `P`, else null of the same type.
- Storage: the layout check on the element types of `x` and `dx`, as a `Ptr{NoTangent}`;
  null for zero-size tangent elements, which have no storage (in a `MemoryRef`, `dp` is
  then an offset, not an address).
- Offset: `dp + offset` (in bytes); null stays null.
- Cast: `dp` as a `Ptr{tangent_type(P)}`; throws an `ArgumentError` if `dp` is non-null
  and `tangent_type(P)` does not mirror `P`, unless `P` or `dp`'s pointee is zero-size.

# Examples

With `dp` a non-null `Ptr{Float64}`:

```julia
_build_tangent_ptr(dp, Float64, Float64)                   # dp
_build_tangent_ptr(dp, Tuple{Int,Float64}, Tuple{NoTangent,Float64})  # null: 16 vs 8 bytes
_build_tangent_ptr(dp, Union{Missing,Float64}, Union{NoTangent,Float64})  # null
_build_tangent_ptr(dp, [1.0], [0.0])                       # dp, as a Ptr{NoTangent}
_build_tangent_ptr(dp, ["a"], [NoTangent()])               # null
_build_tangent_ptr(Ptr{Float64}(C_NULL), 8)                # null
_build_tangent_ptr(Ptr{Int}, dp)                           # throws: tangent `NoTangent`
```

Known gaps: the cast compares `P` only with its own tangent, never with the memory `dp`
points into, and skips zero-size pointees. So casting a `Float64` tangent pointer to
`Ptr{Float32}` reinterprets its bytes, and a store through a `Ptr{Int}` reached via
`Ptr{Cvoid}` leaves the old tangent in place.
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

_build_tangent_ptr(dp::Ptr, offset::Integer) = dp == C_NULL ? dp : dp + offset

# Casting to `Ptr{Int}` must not leave an overwritten `Float64`'s tangent in place.
function _build_tangent_ptr(::Type{Ptr{P}}, dp::Ptr{T}) where {P,T}
    if dp != C_NULL && Base.aligned_sizeof(T) > 0 && Base.aligned_sizeof(P) > 0
        _check_tangent_ptr(_build_tangent_ptr(dp, P, tangent_type(P)), 1)
    end
    return bitcast(Ptr{tangent_type(P)}, dp)
end

"""
    _check_tangent_ptr(dp::Ptr)
    _check_tangent_ptr(dp::Ptr, n::Integer)

Return whether a load, store or copy may go through tangent pointer `dp`; throw where
skipping it would drop a derivative.

A null tangent pointer is harmless until used, so loads, stores and copies check it at the
access, not where it is built. A cast checks when made: if `Ptr{Float64}` became
`Ptr{Int}`, its tangent would be a non-null `Ptr{NoTangent}`, so a later store would write
no tangent and pass every check.

A non-null `dp` returns `true`. A null `dp`, from [`_build_tangent_ptr`](@ref), has no
tangent storage behind it, so the result depends on what the access moves:

- `n` elements of `NoTangent`, or `n <= 0`: `false`, as nothing differentiable is lost;
- one pointer value, `dp::Ptr{<:Ptr}` without `n`: `false`, as a pointer value may carry
  a null tangent;
- anything else, including `n` pointer values: throws an `ArgumentError`.

Without `n`, a `dp` that is not a `Ptr{<:Ptr}` is checked as one element. Pass `n` wherever
the access needs real tangent storage, as copies, casts and `unsafe_pointer_to_objref` do:
a null `Ptr{<:Ptr}` then throws instead of returning `false`.

# Examples

With `dp` a non-null tangent pointer and `dv` a tangent:

```julia
_check_tangent_ptr(dp, 3)                         # true
_check_tangent_ptr(Ptr{NoTangent}(C_NULL), 3)     # false: no derivative
_check_tangent_ptr(Ptr{Float64}(C_NULL), 0)       # false: empty copy
_check_tangent_ptr(Ptr{Ptr{Float64}}(C_NULL))     # false: one pointer value
_check_tangent_ptr(Ptr{Ptr{Float64}}(C_NULL), 1)  # throws
_check_tangent_ptr(Ptr{Float64}(C_NULL))          # throws
_check_tangent_ptr(dp) && unsafe_store!(dp, dv)   # guarded store
```
"""
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

_check_tangent_ptr(dp::Ptr) = _check_tangent_ptr(dp, 1)

# Individual pointer values can carry a null tangent; copying pointer storage cannot.
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
