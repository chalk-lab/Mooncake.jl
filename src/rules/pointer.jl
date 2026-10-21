# Avoid troublesome bitcast magic -- we can't handle converting from pointer to UInt,
# because we drop the gradient, because the tangent type of integers is NoTangent.
# https://github.com/JuliaLang/julia/blob/9f9e989f241fad1ae03c3920c20a93d8017a5b8f/base/pointer.jl#L282
@is_primitive MinimalCtx Tuple{typeof(Base.:(+)),Ptr,Integer}
function frule!!(::Dual{typeof(Base.:(+))}, x::Dual{<:Ptr}, y::Dual{<:Integer})
    return Dual(primal(x) + primal(y), _tangent_ptr_add(tangent(x), primal(y)))
end
function rrule!!(f::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    return CoDual(primal(x) + primal(y), _tangent_ptr_add(tangent(x), primal(y))),
    NoPullback(f, x, y)
end

# A pointer into tangent memory is only usable if that memory mirrors the primal memory byte
# for byte. If it has no storage (e.g. `Memory{NoTangent}`, or `Array{NoTangent}` before
# Julia 1.11) or another layout (e.g. the
# tangent of `Tuple{Int,Float64}` holds only the `Float64`), a pointer into it gets a null
# tangent pointer, and the pointer rules refuse to load or store a tangent through it. (For a
# `MemoryRef`, `ptr_or_offset` is an index when the elements take up no space.)
function _tangent_data_ptr(ptr::Ptr, ::DenseArray{P}, t::DenseArray{V}) where {P,V}
    if Base.elsize(t) == 0 || !_tangent_layout_mirrors(P, V)
        return Ptr{NoTangent}(C_NULL)
    end
    return bitcast(Ptr{NoTangent}, ptr)
end

# Whether values of type `P` and their tangents of type `V` take up the same bytes.
function _layout_mirrors(P::Type, V::Type)
    P === V && return true
    inline = Base.allocatedinline(P)
    inline == Base.allocatedinline(V) || return false
    inline || return true  # both stored as references
    Base.aligned_sizeof(P) == Base.aligned_sizeof(V) || return false
    Base.aligned_sizeof(P) == 0 && return true  # e.g. `Nothing` and `NoTangent`
    P <: Ptr && V <: Ptr && return true
    if Base.isbitsunion(P)
        # A member without a tangent, e.g. the `Int` in `Union{Int,Float64}` or `missing`,
        # leaves the tangent bytes of the value it replaces behind.
        Ws = Base.isbitsunion(V) ? Base.uniontypes(V) : Any[V]
        NoTangent in Ws && return false
        Ts = Base.uniontypes(P)
        return all(T -> any(W -> _layout_mirrors(T, W), Ws), Ts) &&
               all(W -> any(T -> _layout_mirrors(T, W), Ts), Ws)
    end
    (P isa DataType && V isa DataType) || return false
    V <: Tangent && (V = fieldtype(V, :fields))
    n = fieldcount(P)
    (n > 0 && n == fieldcount(V)) || return false
    for i in 1:n
        fieldoffset(P, i) == fieldoffset(V, i) || return false
        _layout_mirrors(fieldtype(P, i), fieldtype(V, i)) || return false
    end
    return true
end

# Defined after `_layout_mirrors`, which its generator calls.
@generated _tangent_layout_mirrors(::Type{P}, ::Type{V}) where {P,V} = _layout_mirrors(P, V)

# A null tangent pointer stays null (see `_tangent_data_ptr`).
_tangent_ptr_add(dx::Ptr, y::Integer) = dx == C_NULL ? dx : dx + y

# The tangent of a value loaded or stored through a null tangent pointer. Only values without
# a derivative, and pointers (which get a null tangent), can go through one.
function _null_ptr_tangent(::Type{V}) where {V}
    V == NoTangent && return NoTangent()
    V <: Ptr && return V(C_NULL)
    return throw(ArgumentError(NULL_TANGENT_PTR_MSG))
end
const NULL_TANGENT_PTR_MSG =
    "Mooncake cannot differentiate loading, storing or copying tangents through a pointer " *
    "into memory whose tangent has no storage or a different layout, e.g. a `Ptr{Float64}` " *
    "into a `Vector{UInt8}` or a `Vector{Tuple{Int,Float64}}`."

# Copying or wrapping `n` tangents through a null tangent pointer is only possible when they
# take up no space.
function _check_null_copy(dx::Ptr{V}, n::Integer) where {V}
    (dx == C_NULL && n > 0 && V != NoTangent) && throw(ArgumentError(NULL_TANGENT_PTR_MSG))
    return nothing
end

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
