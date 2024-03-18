"""
    Stack{T}()

A stack specialised for reverse-mode AD.

Semantically equivalent to a usual stack, but never de-allocates memory once allocated.
"""
mutable struct Stack{T}
    const memory::Vector{T}
    position::Int
    Stack{T}() where {T} = new{T}(Vector{T}(undef, 0), 0)
end

function Stack{T}(x) where {T}
    stack = Stack{T}()
    push!(stack, x)
    return stack
end

Stack(x::T) where {T} = Stack{T}(x)

@inline function Base.push!(x::Stack{T}, val::T) where {T}
    position = x.position + 1
    memory = x.memory
    x.position = position
    if position <= length(memory)
        @inbounds memory[position] = val
        return nothing
    else
        @noinline push!(memory, val)
        return nothing
    end
end

@inline function Base.pop!(x::Stack)
    position = x.position
    val = x.memory[position]
    x.position = position - 1
    return val
end

Base.isempty(x::Stack) = x.position == 0

Base.length(x::Stack) = x.position

"""
    Base.getindex(x::Stack)

Return the value at the top of `x` without popping it.
"""
Base.getindex(x::Stack) = x.memory[x.position]

"""
    Base.setindex!(x::Stack, v)

Set the value of the element at the top of the `x` to `v`.
"""
function Base.setindex!(x::Stack, v)
    x.memory[x.position] = v
    return v
end

Base.eltype(::Stack{T}) where {T} = T

top_ref(x::Stack) = Ref(getfield(x, :memory), getfield(x, :position))

"""
    NoTangentStack()

If a type has `NoTangent` as its tangent type, it should use one of these stacks.
Probably needs to be generalised to an inactive-tangent stack in future, as we also need to
handle constants, which aren't always active.
"""
struct NoTangentStack end

Base.push!(::NoTangentStack, ::Any) = nothing
Base.getindex(::NoTangentStack) = NoTangent()
Base.setindex!(::NoTangentStack, ::NoTangent) = nothing
Base.pop!(::NoTangentStack) = NoTangent()

struct NoTangentRef <: Ref{NoTangent} end

Base.getindex(::NoTangentRef) = NoTangent()
Base.setindex!(::NoTangentRef, ::NoTangent) = nothing

top_ref(::NoTangentStack) = NoTangentRef()

"""
    NoTangentRefStack

Stack for `NoTangentRef`s.
"""
struct NoTangentRefStack end

Base.push!(::NoTangentRefStack, ::Any) = nothing
Base.pop!(::NoTangentRefStack) = NoTangentRef()


struct SingletonStack{T} end

Base.push!(::SingletonStack, ::Any) = nothing
@generated Base.pop!(::SingletonStack{T}) where {T} = T.instance


function tangent_stack_type(::Type{P}) where {P}
    P === DataType && return Stack{Any}
    T = tangent_type(P)
    return T === NoTangent ? NoTangentStack : Stack{T}
end

__array_ref_type(::Type{P}) where {P} = Base.RefArray{P, Vector{P}, Nothing}

function tangent_ref_type_ub(::Type{P}) where {P}
    P === DataType && return Ref
    T = tangent_type(P)
    T === NoTangent && return NoTangentRef
    return isconcretetype(P) ? __array_ref_type(T) : Ref
end

tangent_ref_type_ub(::Type{Type{P}}) where {P} = NoTangentRef

struct InactiveStack{T}
    zero_tangent::T
end

Base.pop!(s::InactiveStack{T}) where {T} = s.zero_tangent

struct InactiveRef{T}
    x::T
end

Base.getindex(x::InactiveRef{T}) where {T} = x.x

increment_ref!(::InactiveRef{T}, ::T) where {T} = nothing

top_ref(::Nothing) = InactiveRef(nothing)

increment_ref!(::InactiveRef{Nothing}, ::T) where {T} = nothing


struct FixedStackTangentRefStack{T}
    x::Stack{T}
end

Base.push!(x::FixedStackTangentRefStack, t) = nothing

Base.pop!(x::FixedStackTangentRefStack) = top_ref(x.x)
