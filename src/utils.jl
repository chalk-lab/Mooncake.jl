"""
    _typeof(x)

Central definition of typeof, which is specific to the use-required in this package.
"""
_typeof(x) = Base._stable_typeof(x)
_typeof(x::Tuple) = Tuple{map(_typeof, x)...}
_typeof(x::NamedTuple{names}) where {names} = NamedTuple{names, _typeof(Tuple(x))}

"""
    tuple_map(f::F, x::Tuple) where {F}

This function is largely equivalent to `map(f, x)`, but always specialises on all of
the element types of `x`, regardless the length of `x`. This contrasts with `map`, in which
the number of element types specialised upon is a fixed constant in the compiler.

As a consequence, if `x` is very long, this function may have very large compile times.

    tuple_map(f::F, x::Tuple, y::Tuple) where {F}

Binary extension of `tuple_map`. Nearly equivalent to `map(f, x, y)`, but guaranteed to
specialise on all element types of `x` and `y`. Furthermore, errors if `x` and `y` aren't
the same length, while `map` will just produce a new tuple whose length is equal to the
shorter of `x` and `y`.
"""
@inline @generated function tuple_map(f::F, x::Tuple) where {F}
    return Expr(:call, :tuple, map(n -> :(f(getfield(x, $n))), eachindex(x.parameters))...)
end

@inline @generated function tuple_map(f::F, x::Tuple, y::Tuple) where {F}
    if length(x.parameters) != length(y.parameters)
        return :(throw(ArgumentError("length(x) != length(y)")))
    else
        stmts = map(n -> :(f(getfield(x, $n), getfield(y, $n))), eachindex(x.parameters))
        return Expr(:call, :tuple, stmts...)
    end
end

function tuple_map(f::F, x::NamedTuple{names}) where {F, names}
    return NamedTuple{names}(tuple_map(f, Tuple(x)))
end

function tuple_map(f::F, x::NamedTuple{names}, y::NamedTuple{names}) where {F, names}
    return NamedTuple{names}(tuple_map(f, Tuple(x), Tuple(y)))
end

for N in 1:256
    @eval @inline function tuple_splat(f, x::Tuple{Vararg{Any, $N}})
        return $(Expr(:call, :f, map(n -> :(getfield(x, $n)), 1:N)...))
    end
end

@inline @generated function tuple_splat(f, x::Tuple)
    return Expr(:call, :f, map(n -> :(x[$n]), 1:length(x.parameters))...)
end

@inline @generated function tuple_splat(f, v, x::Tuple)
    return Expr(:call, :f, :v, map(n -> :(x[$n]), 1:length(x.parameters))...)
end

@inline @generated function tuple_fill(val ,::Val{N}) where {N}
    return Expr(:call, :tuple, map(_ -> :val, 1:N)...)
end


#=
    _map_if_assigned!(f, y::Array, x::Array{P}) where {P}

For all `n`, if `x[n]` is assigned, then writes the value returned by `f(x[n])` to `y[n]`,
otherwise leaves `y[n]` unchanged.

Equivalent to `map!(f, y, x)` if `P` is a bits type as element will always be assigned.

Requires that `y` and `x` have the same size.
=#
function _map_if_assigned!(f::F, y::Array, x::Array{P}) where {F, P}
    @assert size(y) == size(x)
    @inbounds for n in eachindex(y)
        if isbitstype(P) || isassigned(x, n)
            y[n] = f(x[n])
        end
    end
    return y
end

#=
    _map_if_assigned!(f::F, y::Array, x1::Array{P}, x2::Array)

Similar to the other method of `_map_if_assigned!` -- for all `n`, if `x1[n]` is assigned,
writes `f(x1[n], x2[n])` to `y[n]`, otherwise leaves `y[n]` unchanged.

Requires that `y`, `x1`, and `x2` have the same size.
=#
function _map_if_assigned!(f::F, y::Array, x1::Array{P}, x2::Array) where {F, P}
    @assert size(y) == size(x1)
    @assert size(y) == size(x2)
    @inbounds for n in eachindex(y)
        if isbitstype(P) || isassigned(x1, n)
            y[n] = f(x1[n], x2[n])
        end
    end
    return y
end

#=
    _map(f, x...)

Same as `map` but requires all elements of `x` to have equal length.
The usual function `map` doesn't enforce this for `Array`s.
=#
@inline function _map(f::F, x::Vararg{Any, N}) where {F, N}
    @assert allequal(map(length, x))
    return map(f, x...)
end

#=
    is_vararg_sig_and_sparam_names(sig)::Tuple{Bool, Vector{Symbol}}

Returns a 2-tuple. The first element is true if the method associated to `sig` is a vararg
method, and false if not. The second element contains all of the names of the static
parameters associated to said method.
=#
function is_vararg_sig_and_sparam_names(sig)::Tuple{Bool, Vector{Symbol}}
    world = Base.get_world_counter()
    min = Base.RefValue{UInt}(typemin(UInt))
    max = Base.RefValue{UInt}(typemax(UInt))
    ms = Base._methods_by_ftype(sig, nothing, -1, world, true, min, max, Ptr{Int32}(C_NULL))::Vector
    m = only(ms).method
    return m.isva, sparam_names(m)
end

# Returns the names of all of the static parameters in `m`.
function sparam_names(m::Core.Method)::Vector{Symbol}
    whereparams = ExprTools.where_parameters(m.sig)
    whereparams === nothing && return Symbol[]
    return map(whereparams) do name
        name isa Symbol && return name
        Meta.isexpr(name, :(<:)) && return name.args[1]
        Meta.isexpr(name, :(>:)) && return name.args[1]
        error("unrecognised type param $name")
    end
end

"""
    is_always_initialised(::Type{P}, n::Int)

True if the `n`th field of `P` is always initialised. If the `n`th fieldtype of `P`
`isbitstype`, then this is distinct from asking whether the `n`th field is always defined.
An isbits field is always defined, but is not always explicitly initialised.
"""
function is_always_initialised(::Type{P}, n::Int) where {P}
    return n <= Core.Compiler.datatype_min_ninitialized(P)
end

"""
    is_always_fully_initialised(::Type{P}) where {P}

True if all fields in `P` are always initialised. Put differently, there are no inner
constructors which permit partial initialisation.
"""
function is_always_fully_initialised(::Type{P}) where {P}
    return Core.Compiler.datatype_min_ninitialized(P) == fieldcount(P)
end
