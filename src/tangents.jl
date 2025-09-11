"""
    NoTangent

The type in question has no meaningful notion of a tangent space.
Generally, you shouldn't use this -- just let the default recursive tangent
construction work.
You might need to use this for `primitive type`s though.
"""
struct NoTangent end

"""
    PossiblyUninitTangent{T}

Represents a `T` which maybe or may not be present. Does not distinguish between 0 and
not being present.
"""
struct PossiblyUninitTangent{T}
    tangent::T
    PossiblyUninitTangent{T}(tangent::T) where {T} = new{T}(tangent)
    PossiblyUninitTangent{T}() where {T} = new{T}()
end

# Copy only if initialized, otherwise create new uninitialized instance
_copy(x::P) where {P<:PossiblyUninitTangent} = is_init(x) ? P(_copy(x.tangent)) : P()

@inline PossiblyUninitTangent(tangent::T) where {T} = PossiblyUninitTangent{T}(tangent)
@inline PossiblyUninitTangent(T::Type) = PossiblyUninitTangent{T}()

@inline is_init(t::PossiblyUninitTangent) = isdefined(t, :tangent)
is_init(t) = true

@unstable @inline val(x::PossiblyUninitTangent) =
    (!is_init(x) && error("Uninitialised"); x.tangent)
@inline val(x) = x

"""
    Tangent{Tfields<:NamedTuple}

Default type used to represent the tangent of a `struct`. See [`tangent_type`](@ref) for
more info.
"""
struct Tangent{Tfields<:NamedTuple}
    fields::Tfields
end

Base.:(==)(x::Tangent, y::Tangent) = x.fields == y.fields

"""
    MutableTangent{Tfields<:NamedTuple}

Default type used to represent the tangent of a `mutable struct`. See [`tangent_type`](@ref)
for more info.
"""
mutable struct MutableTangent{Tfields<:NamedTuple}
    fields::Tfields
    MutableTangent{Tfields}() where {Tfields} = new{Tfields}()
    MutableTangent(fields::Tfields) where {Tfields} = MutableTangent{Tfields}(fields)
    function MutableTangent{Tfields}(
        fields::NamedTuple{names}
    ) where {names,Tfields<:NamedTuple{names}}
        return new{Tfields}(fields)
    end
end

Base.:(==)(x::MutableTangent, y::MutableTangent) = x.fields == y.fields

fields_type(::Type{MutableTangent{Tfields}}) where {Tfields<:NamedTuple} = Tfields
fields_type(::Type{Tangent{Tfields}}) where {Tfields<:NamedTuple} = Tfields
fields_type(::Type{<:Union{MutableTangent,Tangent}}) = NamedTuple

const PossiblyMutableTangent{T} = Union{MutableTangent{T},Tangent{T}}

"""
    get_tangent_field(t::Union{MutableTangent, Tangent}, i::Int)

Gets the `i`th field of data in `t`.

Has the same semantics that `getfield!` would have if the data in the `fields` field of `t`
were actually fields of `t`. This is the moral equivalent of `getfield` for
`MutableTangent`.
"""
@unstable @inline get_tangent_field(t::PossiblyMutableTangent, i::Int) = val(
    getfield(t.fields, i)
)

@unstable @inline function get_tangent_field(
    t::PossiblyMutableTangent{F}, s::Symbol
) where {F}
    return get_tangent_field(t, _sym_to_int(F, Val(s)))
end

"""
    set_tangent_field!(t::MutableTangent{Tfields}, i::Int, x) where {Tfields}

Sets the value of the `i`th field of the data in `t` to value `x`.

Has the same semantics that `setfield!` would have if the data in the `fields` field of `t`
were actually fields of `t`. This is the moral equivalent of `setfield!` for
`MutableTangent`.
"""
@inline function set_tangent_field!(t::MutableTangent{Tfields}, i::Int, x) where {Tfields}
    fields = t.fields
    Ti = fieldtype(Tfields, i)
    new_val = Ti <: PossiblyUninitTangent ? Ti(x) : x
    new_fields = Tfields(ntuple(n -> n == i ? new_val : fields[n], fieldcount(Tfields)))
    t.fields = new_fields
    return x
end

@inline function set_tangent_field!(t::MutableTangent{T}, s::Symbol, x) where {T}
    return set_tangent_field!(t, _sym_to_int(T, Val(s)), x)
end

@generated function _sym_to_int(::Type{Tfields}, ::Val{s}) where {Tfields,s}
    return findfirst(==(s), fieldnames(Tfields))
end

@unstable function tangent_field_types_exprs(P::Type)
    tangent_type_exprs = map(fieldtypes(P), always_initialised(P)) do _P, init
        T_expr = Expr(:call, :tangent_type, _P)
        return init ? T_expr : Expr(:curly, PossiblyUninitTangent, T_expr)
    end
    return tangent_type_exprs
end

# It is essential that this gets inlined. If it does not, then we run into performance
# issues with the recursion to compute tangent types for nested types.
@generated function tangent_field_types(::Type{P}) where {P}
    return Expr(:call, :tuple, tangent_field_types_exprs(P)...)
end

function build_tangent(::Type{P}, fields::Vararg{Any,N}) where {P,N}
    TP = tangent_type(P)
    _ftypes = tangent_field_types(P)
    ftypes = Tuple{_ftypes...}
    fnames = fieldnames(P)
    return _build_tangent_cartesian(
        TP, fields, ftypes, Val(fnames), Val(length(_ftypes))
    )::TP
end
@generated function _build_tangent_cartesian(
    ::Type{TP}, fields::Tuple{Vararg{Any,N}}, ::Type{ftypes}, ::Val{fnames}, ::Val{nfields}
) where {TP,N,ftypes,fnames,nfields}
    quote
        full_fields = Base.Cartesian.@ntuple(
            $nfields, n -> let
                tt = ftypes.types[n]
                if tt <: PossiblyUninitTangent
                    n <= $N ? tt(fields[n]) : tt()
                else
                    fields[n]
                end
            end
        )
        return TP(NamedTuple{$fnames}(full_fields))
    end
end

function build_tangent(
    ::Type{P}, fields::Vararg{Any,N}
) where {P<:Union{Tuple,NamedTuple},N}
    TP = tangent_type(P)
    TP === NoTangent && return NoTangent()::TP
    isconcretetype(P) && return TP(fields)
    return __tangent_from_non_concrete(P, fields)::TP
end

"""
    macro foldable def

Shorthand for `Base.@assume_effects :foldable function f(x)...`.
"""
macro foldable(expr)
    return esc(:(Base.@assume_effects :foldable $expr))
end

"""
    tangent_type(P)

There must be a single type used to represents tangents of primals of type `P`, and it must
be given by `tangent_type(P)`.

Warning: this function assumes the effects `:removable` and `:consistent`. This is necessary
to ensure good performance, but imposes precise constraints on your implementation. If
adding new methods to `tangent_type`, you should consult the extended help of
`Base.@assume_effects` to see what this imposes upon your implementation.

# Extended help

The tangent types which Mooncake.jl uses are quite similar in spirit to ChainRules.jl.
For example, tangent "vectors" for
1. `Float64`s are `Float64`s,
1. `Vector{Float64}`s are `Vector{Float64}`s, and
1. `struct`s are other another (special) `struct` with field types specified recursively.

There are, however, some major differences.
Firstly, while it is certainly true that the above tangent types are permissible in
ChainRules.jl, they are not the uniquely permissible types. For example, `ZeroTangent` is
also a permissible type of tangent for any of them, and `Float32` is permissible for
`Float64`. This is a general theme in ChainRules.jl -- it intentionally declines to place
restrictions on what type can be used to represent the tangent of a given type.

Mooncake.jl differs from this.
**It insists that each primal type is associated to a _single_ tangent type.**
Furthermore, this type is _always_ given by the function `Mooncake.tangent_type(primal_type)`.

Consider some more worked examples.

#### Int

`Int` is not a differentiable type, so its tangent type is `NoTangent`:
```jldoctest
julia> tangent_type(Int)
NoTangent
```

#### Tuples

The tangent type of a `Tuple` is defined recursively based on its field types. For example
```jldoctest
julia> tangent_type(Tuple{Float64, Vector{Float64}, Int})
Tuple{Float64, Vector{Float64}, NoTangent}
```

There is one edge case to be aware of: if all of the field of a `Tuple` are
non-differentiable, then the tangent type is `NoTangent`. For example,
```jldoctest
julia> tangent_type(Tuple{Int, Int})
NoTangent
```

#### Structs

As with `Tuple`s, the tangent type of a struct is, by default, given recursively.
In particular, the tangent type of a `struct` type is `Tangent`.
This type contains a `NamedTuple` containing the tangent to each field in the primal `struct`.

As with `Tuple`s, if all field types are non-differentiable, the tangent type of the entire
struct is `NoTangent`.

There are a couple of additional subtleties to consider over `Tuple`s though. Firstly, not
all fields of a `struct` have to be defined. Fortunately, Julia makes it easy to determine
how many of the fields might possibly not be defined. The tangent associated to any field
which might possibly not be defined is wrapped in a `PossiblyUninitTangent`.

Furthermore, `struct`s can have fields whose static type is abstract. For example
```jldoctest foo
julia> struct Foo
           x
       end
```
If you ask for the tangent type of `Foo`, you will see that it is
```jldoctest foo
julia> tangent_type(Foo)
Tangent{@NamedTuple{x}}
```
Observe that the field type associated to `x` is `Any`. The way to understand this result is
to observe that
1. `x` could have literally any type at runtime, so we know nothing about what its tangent
    type must be until runtime, and
1. we require that the tangent type of `Foo` be unique.
The consequence of these two considerations is that the tangent type of `Foo` must be able
to contain any type of tangent in its `x` field. It follows that the fieldtype of the `x`
field of `Foo`s tangent must be `Any`.



#### Mutable Structs

The tangent type for `mutable struct`s have the same set of considerations as `struct`s.
The only difference is that they must themselves be mutable.
Consequently, we use a type called `MutableTangent` to represent their tangents.
It is a `mutable struct` with the same structure as `Tangent`.

For example, if you ask for the `tangent_type` of
```jldoctest bar
julia> mutable struct Bar
           x::Float64
       end
```
you will find that it is
```jldoctest bar
julia> tangent_type(Bar)
MutableTangent{@NamedTuple{x::Float64}}
```


#### Primitive Types

We've already seen a couple of primitive types (`Float64` and `Int`).
The basic story here is that all primitive types require an explicit specification of what their tangent type must be.

One interesting case are `Ptr` types.
The tangent type of a `Ptr{P}` is `Ptr{T}`, where `T = tangent_type(P)`.
For example
```julia
julia> tangent_type(Ptr{Float64})
Ptr{Float64}
```

"""
tangent_type(T)

tangent_type(x) = throw(error("$x is not a type. Perhaps you meant typeof(x)?"))

# The "Bottom" type.
@foldable tangent_type(::Type{Union{}}) = Union{}

# This is essential for DataType, as the recursive definition always recurses infinitely,
# because one of the fieldtypes is itself always a DataType. In particular, we'll always
# eventually hit `Any`, whose `super` field is `Any`.
# This makes it clear that we can't recursively construct tangents for data structures which
# refer to themselves...
tangent_type(::Type{<:Type}) = NoTangent

tangent_type(::Type{<:TypeVar}) = NoTangent

@unstable @foldable tangent_type(::Type{Ptr{P}}) where {P} = Ptr{tangent_type(P)}

tangent_type(::Type{<:Ptr}) = NoTangent

tangent_type(::Type{Bool}) = NoTangent

tangent_type(::Type{Char}) = NoTangent

tangent_type(::Type{Symbol}) = NoTangent

tangent_type(::Type{Cstring}) = NoTangent

tangent_type(::Type{Cwstring}) = NoTangent

tangent_type(::Type{Module}) = NoTangent

tangent_type(::Type{Nothing}) = NoTangent

tangent_type(::Type{Expr}) = NoTangent

tangent_type(::Type{Core.TypeofVararg}) = NoTangent

@unstable tangent_type(::Type{SimpleVector}) = Vector{Any}

tangent_type(::Type{P}) where {P<:Union{UInt8,UInt16,UInt32,UInt64,UInt128}} = NoTangent

tangent_type(::Type{P}) where {P<:Union{Int8,Int16,Int32,Int64,Int128,BigInt}} = NoTangent

tangent_type(::Type{<:Core.Builtin}) = NoTangent

@foldable tangent_type(::Type{P}) where {P<:IEEEFloat} = P

tangent_type(::Type{<:Core.LLVMPtr}) = NoTangent

tangent_type(::Type{String}) = NoTangent

@foldable tangent_type(::Type{<:Array{P,N}}) where {P,N} = Array{tangent_type(P),N}

@unstable tangent_type(::Type{<:Array{P,N} where {P}}) where {N} = Array

tangent_type(::Type{<:MersenneTwister}) = NoTangent

tangent_type(::Type{Core.TypeName}) = NoTangent

tangent_type(::Type{Core.MethodTable}) = NoTangent

tangent_type(::Type{DimensionMismatch}) = NoTangent

tangent_type(::Type{Method}) = NoTangent

tangent_type(::Type{<:Enum}) = NoTangent

tangent_type(::Type{<:Base.TTY}) = NoTangent

tangent_type(::Type{<:IOStream}) = NoTangent

tangent_type(::Type{<:Base.CoreLogging.AbstractLogger}) = NoTangent

function split_union_tuple_type(tangent_types)

    # Create first split.
    ta_types = map(tangent_types) do T
        return T isa Union ? T.a : T
    end
    ta = Tuple{ta_types...}

    # Create second split.
    tb_types = map(tangent_types) do T
        return T isa Union ? T.b : T
    end
    tb = Tuple{tb_types...}

    return Union{ta,tb}
end

# Generated functions cannot emit closures, so this is defined here for use below.
isconcrete_or_union(p) = p isa Union || isconcretetype(p)

@foldable @generated function tangent_type(::Type{P}) where {N,P<:Tuple{Vararg{Any,N}}}

    # As with other types, tangent type of Union is Union of tangent types.
    P isa Union && return :(Union{tangent_type($(P.a)),tangent_type($(P.b))})

    # Determine whether P isa a Tuple with a Vararg, e.g, Tuple, or Tuple{Float64, Vararg}.
    # Need to exclude `UnionAll`s from this, by checking `isa(P, DataType)`, in order to
    # ensure that `Base.datatype_fieldcount(P)` will run successfully.
    isa(P, DataType) && !(@isdefined(N)) && return Any

    # Tuple{} can only have `NoTangent` as its tangent type. As before, verify we don't have
    # a UnionAll before running to ensure that datatype_fieldcount will run.
    isa(P, DataType) && N == 0 && return NoTangent

    # Expression to construct `Tuple` type containing tangent type for all fields.
    tangent_type_exprs = map(n -> :(tangent_type(fieldtype(P, $n))), 1:N)
    tangent_types = Expr(:call, tuple, tangent_type_exprs...)

    # Construct a Tuple type of the same length as `P`, containing all `NoTangent`s.
    T_all_notangent = Tuple{Vararg{NoTangent,N}}

    return quote

        # Get tangent types for all fields. If they're all `NoTangent`, return `NoTangent`.
        # i.e. if `P = Tuple{Int, Int}`, do not return `Tuple{NoTangent, NoTangent}`.
        # Simplify and return `NoTangent`.
        tangent_types = $tangent_types
        T = Tuple{tangent_types...}
        T <: $T_all_notangent && return NoTangent

        # If exactly one of the field types is a Union, then split.
        union_fields = _findall(Base.Fix2(isa, Union), tangent_types)
        if length(union_fields) == 1 && all(tuple_map(isconcrete_or_union, tangent_types))
            return split_union_tuple_type(tangent_types)
        end

        # If it's _possible_ for a subtype of `P` to have tangent type `NoTangent`, then we
        # must account for that by returning the union of `NoTangent` and `T`. For example,
        # if `P = Tuple{Any, Int}`, then `P2 = Tuple{Int, Int}` is a subtype. Since `P2` has
        # tangent type `NoTangent`, it must be true that `NoTangent <: tangent_type(P)`. If,
        # on the other hand, it's not possible for `NoTangent` to be the tangent type, e.g.
        # for `Tuple{Float64, Any}`, then there's no need to take the union.
        return $T_all_notangent <: T ? Union{T,NoTangent} : T
    end
end

@unstable @foldable function tangent_type(::Type{P}) where {N,P<:NamedTuple{N}}
    P isa Union && return Union{tangent_type(P.a),tangent_type(P.b)}
    !isconcretetype(P) && return Union{NoTangent,NamedTuple{N}}
    TT = tangent_type(Tuple{fieldtypes(P)...})
    TT == NoTangent && return NoTangent
    return isconcretetype(TT) ? NamedTuple{N,TT} : Any
end

@foldable @generated function tangent_type(::Type{P}) where {P}

    # This method can only handle struct types. Something has gone wrong if P is primitive.
    if isprimitivetype(P)
        return error("$P is a primitive type. Implement a method of `tangent_type` for it.")
    end

    # If the type is a Union, then take the union type of its arguments.
    P isa Union && return :(Union{tangent_type($(P.a)),tangent_type($(P.b))})

    # If the type is itself abstract, it's tangent could be anything.
    # The same goes for if the type has any undetermined type parameters.
    (isabstracttype(P) || !isconcretetype(P)) && return Any

    tangent_fields_types_expr = Expr(:curly, Tuple, tangent_field_types_exprs(P)...)
    T_all_notangent = Tuple{Vararg{NoTangent,fieldcount(P)}}
    return quote

        # Construct a `Tuple{...}` whose fields are the tangent types of the fields of `P`.
        tangent_field_types_tuple = $tangent_fields_types_expr

        # If all fields are definitely `NoTangent`s, then return `NoTangent`.
        tangent_field_types_tuple <: $T_all_notangent && return NoTangent

        # Derive tangent type.
        bt = NamedTuple{$(fieldnames(P)),tangent_field_types_tuple}
        return $(ismutabletype(P) ? MutableTangent : Tangent){bt}
    end
end

backing_type(P::Type) = NamedTuple{fieldnames(P),Tuple{tangent_field_types(P)...}}

struct NoCache end

Base.haskey(::NoCache, x) = false
Base.setindex!(::NoCache, v, x) = nothing

const MaybeCache = Union{NoCache,IdDict{Any,Any}}

"""
    zero_tangent(x)

Returns the unique zero element of the tangent space of `x`.
It is an error for the zero element of the tangent space of `x` to be represented by
anything other than that which this function returns.
"""
zero_tangent(x)
function zero_tangent(x::P) where {P}
    return zero_tangent_internal(x, isbitstype(P) ? NoCache() : IdDict())
end

"""
    zero_tangent_internal(x, d::MaybeCache)

Implementation of [`zero_tangent`](@ref). Makes use of `d` in the same way that
`Base.deepcopy_internal` makes use of an `IdDict` (see the docstring for `Base.deepcopy` for
information).

In particular, it should be used to ensure that aliasing relationships are respected,
meaning that if in the tuple `x = (a, b)`, `a === b`, then in
`(da, db) = zero_tangent((a, b))` it must hold that should have that `da === db`.
You may want to consult the method of `zero_tangent_internal` for `struct` and
`mutable struct` types for inspiration if implementing this for your own type.

Similarly, if `x` contains a one or more circular, its tangent will probably need to contain
similar circular references (unless it is something trivial like `NoTangent`). Again,
consult existing implementations for inspiration.

If `d` is a `NoCache` assume that `x` contains neither aliasing nor circular references.
"""
zero_tangent_internal(::Union{Int8,Int16,Int32,Int64,Int128}, ::MaybeCache) = NoTangent()
zero_tangent_internal(x::IEEEFloat, ::MaybeCache) = zero(x)
@generated function zero_tangent_internal(x::Tuple, dict::MaybeCache)
    zt_exprs = map(n -> :(zero_tangent_internal(x[$n], dict)), 1:fieldcount(x))
    return quote
        tangent_type($x) == NoTangent && return NoTangent()
        return $(Expr(:call, :tuple, zt_exprs...))
    end
end
function zero_tangent_internal(x::NamedTuple, dict::MaybeCache)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    return tuple_map(Base.Fix2(zero_tangent_internal, dict), x)
end
function zero_tangent_internal(x::Ptr, ::MaybeCache)
    return throw(ArgumentError("zero_tangent not available for pointers."))
end
function zero_tangent_internal(x::SimpleVector, dict::MaybeCache)
    return map!(
        n -> zero_tangent_internal(x[n], dict), Vector{Any}(undef, length(x)), eachindex(x)
    )
end
@inline @generated function zero_tangent_internal(x::P, d::MaybeCache) where {P}

    # Loop over fields, constructing expressions to construct zeros depending on the
    # field type and initialisation status.
    inits = always_initialised(P)
    tangent_field_exprs = map(1:fieldcount(P)) do n
        if inits[n]
            return :(zero_tangent_internal(getfield(x, $n), d))
        else
            P_field = fieldtype(P, n)
            T_field_expr = :(PossiblyUninitTangent{tangent_type($P_field)})
            return quote
                if isdefined(x, $n)
                    $T_field_expr(zero_tangent_internal(getfield(x, $n), d))
                else
                    $T_field_expr()
                end
            end
        end
    end
    tangent_fields_tuple_expr = Expr(:call, :tuple, tangent_field_exprs...)

    return quote
        tangent_type(P) == NoTangent && return NoTangent()

        # If dealing with a mutable type, ensure that we have an entry in `d`.
        if tangent_type(P) <: MutableTangent
            haskey(d, x) && return d[x]::tangent_type(P)
            d[x] = tangent_type(P)() # create a uninitialised MutableTangent
        end

        # For each field in `x`, construct its zero tangent. This is where the generated
        # expression above it used. Everything else is regular code.
        fields = backing_type(P)($tangent_fields_tuple_expr)

        if tangent_type(P) <: MutableTangent
            # if circular reference exists, then the recursive call will first look up d
            # and return the uninitialised MutableTangent
            # after the recursive call returns, d will be initialised
            d[x].fields = fields
            return d[x]::tangent_type(P)
        else
            return tangent_type(P)(fields)
        end
        return t
    end
end

"""
    uninit_tangent(x)

Related to `zero_tangent`, but a bit different. Check current implementation for
details -- this docstring is intentionally non-specific in order to avoid becoming outdated.
"""
@inline uninit_tangent(x) = zero_tangent(x)
@inline uninit_tangent(x::Ptr{P}) where {P} = bitcast(Ptr{tangent_type(P)}, x)

"""
    randn_tangent(rng::AbstractRNG, x::P) where {P}

Required for testing. Generate a randomly-chosen tangent to `x`. Very similar to
[`Mooncake.zero_tangent`](@ref), except that the elements are randomly chosen, rather than
being equal to zero.
"""
function randn_tangent(rng::AbstractRNG, x::P) where {P}
    return randn_tangent_internal(rng, x, isbitstype(P) ? NoCache() : IdDict())
end

"""
    randn_tangent_internal(rng::AbstractRNG, x, dict::MaybeCache)

Implementation for [`Mooncake.randn_tangent`](@ref). Please consult the docstring for
[`zero_tangent_internal`](@ref) for more information on how this implementation works, As
the same implementation strategy is adopted for both this function and that one.
"""
function randn_tangent_internal(rng::AbstractRNG, ::P, ::MaybeCache) where {P<:IEEEFloat}
    return randn(rng, P)
end
@generated function randn_tangent_internal(rng::AbstractRNG, x::Tuple, dict::MaybeCache)
    rt_exprs = map(n -> :(randn_tangent_internal(rng, x[$n], dict)), 1:fieldcount(x))
    return quote
        tangent_type($x) == NoTangent && return NoTangent()
        return $(Expr(:call, :tuple, rt_exprs...))
    end
end
function randn_tangent_internal(rng::AbstractRNG, x::NamedTuple, dict::MaybeCache)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    return tuple_map(x -> randn_tangent_internal(rng, x, dict), x)
end
function randn_tangent_internal(rng::AbstractRNG, x::SimpleVector, dict::MaybeCache)
    return map!(Vector{Any}(undef, length(x)), eachindex(x)) do n
        return randn_tangent_internal(rng, x[n], dict)
    end
end
@generated function randn_tangent_internal(rng::AbstractRNG, x::P, d::MaybeCache) where {P}

    # Loop over fields, constructing expressions to construct randn tangents depending on
    # the field type and initialisation status.
    inits = always_initialised(P)
    tangent_field_exprs = map(1:fieldcount(P)) do n
        if inits[n]
            return :(randn_tangent_internal(rng, getfield(x, $n), d))
        else
            P_field = fieldtype(P, n)
            T_field_expr = :(PossiblyUninitTangent{tangent_type($P_field)})
            return quote
                if isdefined(x, $n)
                    $T_field_expr(randn_tangent_internal(rng, getfield(x, $n), d))
                else
                    $T_field_expr()
                end
            end
        end
    end
    tangent_fields_tuple_expr = Expr(:call, :tuple, tangent_field_exprs...)

    return quote
        tangent_type(P) == NoTangent && return NoTangent()

        # If dealing with a mutable type, ensure that we have an entry in `d`.
        if tangent_type(P) <: MutableTangent
            haskey(d, x) && return d[x]::tangent_type(P)
            d[x] = tangent_type(P)() # create a uninitialised MutableTangent
        end

        # For each field in `x`, construct its randn tangent. This is where the generated
        # expression above it used. Everything else is regular code.
        fields = backing_type(P)($tangent_fields_tuple_expr)

        if tangent_type(P) <: MutableTangent
            # if circular reference exists, then the recursive call will first look up d
            # and return the uninitialised MutableTangent
            # after the recursive call returns, d will be initialised
            d[x].fields = fields
            return d[x]::tangent_type(P)
        else
            return tangent_type(P)(fields)
        end
        return t
    end
end

"""
    require_tangent_cache(::Type{P}) where {P}

Determines whether operations on tangents of primal type `P` require a cache to handle potential 
circular references or aliasing. Returns `Val{true}()` if caching is required (the default),
or `Val{false}()` if tangents of type `tangent_type(P)` are guaranteed to be free of circular references,
uninitialized fields that could create circular references, and aliasing.

This function is used internally by operations like `set_to_zero!!`. Returning `Val{false}()` 
can improve performance by avoiding cache overhead, but is only safe when the memory layout
of the tangent type is provably tree-like. 

!!! warning "Advanced Performance Optimization"
    This is an advanced optimization hook. The default behavior (returning `Val{true}()`)
    is always correct but may have performance overhead. Only implement custom methods
    if you have:
    1. Measured a significant performance impact from caching
    2. Proven your tangent types cannot contain circular references or aliasing
    3. Thoroughly tested your implementation
    
    See the Extended Help section for detailed safety requirements and examples.

# Extended help

### Understanding Tangent Caching

This function makes decisions based on the primal type `P` by answering the key question:
"Could tangents of type `tangent_type(P)` contain circular references or aliasing?"

The cache prevents infinite loops and incorrect results when traversing tangents that might contain:
- Circular references (A references B, B references A)
- Aliasing (multiple references to the same object)

### Safety Requirements for `Val{false}()`

Returning `Val{false}()` is only safe when the tangent type memory layout is guaranteed to be tree-like,
with no possibility of circular references or aliasing.

#### Safe Cases (non-exhaustive, can return `Val{false}()`):

1. **Pure immutable structures**: Structures with only `Tangent` types (no `MutableTangent`)
   cannot create cycles because immutable objects cannot reference themselves after construction
   
2. **Concrete-only mutable structs**: `MutableTangent` types where ALL fields have concrete types
   that cannot hold references (e.g., `mutable struct Foo; x::Float64; end`)
   
3. **Concrete `PossiblyUninitTangent`**: When parameterized by concrete non-reference types
   like `PossiblyUninitTangent{Float64}`. For example, `mutable struct Bar; x::Ref{Float64}; end` 
   produces this safe tangent type because `Ref{Float64}` is concrete

#### Unsafe Cases (non-exhaustive, must return `Val{true}()`): 

1. **Abstract typed fields**: Any field typed as `Any` or other abstract types can hold
   arbitrary values at runtime, including circular references
   
2. **Potentially self-referential types**: Types where fields could reference the containing object,
   either directly or through a chain of references
   
3. **Shared mutable state**: Multiple fields that might reference the same mutable object,
   creating aliasing issues

### Common Patterns Requiring Caching

The following examples demonstrate why certain patterns create circular references or aliasing in tangents,
requiring caching to avoid infinite loops or incorrect results.

#### Example 1: Circular References with Abstract-Typed Fields

`Ref` with abstract types can lead to circular references:

```jldoctest
julia> # Ref{Any} is dangerous because Any can hold circular references
       struct Evil
           r::Ref{Any}
           data::Float64
       end

julia> e = Evil(Ref{Any}(nothing), 1.0);

julia> e.r[] = e;  # Store the struct in its own field!

julia> # The tangent type has PossiblyUninitTangent{Any}
       tangent_type(Evil)
Tangent{@NamedTuple{r, data::Float64}}

julia> # Let's trace what happens with zero_tangent
       zt = zero_tangent(e);

julia> # The Ref field's tangent is a MutableTangent
       typeof(zt.fields.r)
MutableTangent{@NamedTuple{x::Mooncake.PossiblyUninitTangent{Any}}}

julia> # And it contains a circular reference to zt itself!
       zt.fields.r.fields.x.tangent === zt
true
```

#### Example 2: Aliasing in Tangent Structures

When a primal contains aliased references, the tangent must preserve this aliasing for correctness.
Without caching, operations would incorrectly process aliased tangents multiple times:

```jldoctest
julia> # Create a mutable primal with aliased references
       mutable struct Container
           data::Vector{Float64}
       end

julia> x = Container([1.0, 2.0, 3.0]);

julia> # Create aliasing: both fields reference the same Container
       primal_object = (x, x);

julia> # The tangent preserves the aliasing structure
       zt = zero_tangent(primal_object);

julia> # Verify the tangent type: tuple of two MutableTangents
       typeof(zt)
Tuple{MutableTangent{@NamedTuple{data::Vector{Float64}}}, MutableTangent{@NamedTuple{data::Vector{Float64}}}}

julia> # Crucially, both elements are the SAME tangent object (aliased)
       zt[1] === zt[2]
true
```

This aliasing is essential for correctness! If `zt[1]` and `zt[2]` were different objects, 
the tangent wouldn't correctly represent derivatives w.r.t. the shared primal `x`.

The aliasing in tangents mirrors the aliasing in primals. Without caching to track visited 
objects, operations like [`increment!!`](@ref) would visit `zt[1]` and `zt[2]` separately, not 
realizing they're the same object. This would lead to double-counting, incrementing the 
same tangent twice and producing incorrect results.

"""
require_tangent_cache(::Type{P}) where {P} = Val{!isbitstype(P)}()
require_tangent_cache(::Type{<:Array{P}}) where {P} = Val{!isbitstype(P)}()

const IncCache = Union{NoCache,IdDict{Any,Bool}}
const SetToZeroCache = Union{NoCache,Vector{UInt}}

"""
    _already_tracked!(c::SetToZeroCache, x)

Check if an object has already been tracked and add it to the cache if not.
Returns `true` if the object was already tracked, `false` otherwise.
Mutates the cache by adding untracked objects.
"""
@inline function _already_tracked!(c::Vector{UInt}, x)
    oid = objectid(x)
    oid in c && return true
    push!(c, oid)
    return false
end

@inline _already_tracked!(::NoCache, x) = false

"""
    increment!!(x::T, y::T) where {T}

Add `x` to `y`. If `ismutabletype(T)`, then `increment!!(x, y) === x` must hold.
That is, `increment!!` will mutate `x`.
This must apply recursively if `T` is a composite type whose fields are mutable.
"""
function increment!!(x::T, y::T) where {T}
    return increment_internal!!(isbitstype(T) ? NoCache() : IdDict{Any,Bool}(), x, y)
end

"""
    increment_internal!!(c::IncCache, x::T, y::T) where {T}

Implementation of [`Mooncake.increment!!`](@ref). Make use the cache `c` to avoid "double
counting". If `c` is a `NoCache`, assume no aliasing or circular referencing.
"""
increment_internal!!(::IncCache, ::NoTangent, ::NoTangent) = NoTangent()
increment_internal!!(::IncCache, x::T, y::T) where {T<:IEEEFloat} = x + y
function increment_internal!!(::IncCache, x::Ptr{T}, y::Ptr{T}) where {T}
    return x === y ? x : throw(error("Incrementing pointers is not supported!"))
end
@generated function increment_internal!!(c::IncCache, x::T, y::T) where {T<:Tuple}
    inc_exprs = map(n -> :(increment_internal!!(c, x[$n], y[$n])), 1:fieldcount(T))
    return Expr(:call, :tuple, inc_exprs...)
end
@generated function increment_internal!!(c::IncCache, x::T, y::T) where {T<:NamedTuple}
    inc_exprs = map(n -> :(increment_internal!!(c, x[$n], y[$n])), 1:fieldcount(T))
    return Expr(:new, T, inc_exprs...)
end
function increment_internal!!(c::IncCache, x::T, y::T) where {T<:PossiblyUninitTangent}
    is_init(x) && is_init(y) && return T(increment_internal!!(c, val(x), val(y)))
    is_init(x) && !is_init(y) && error("x is initialised, but y is not")
    !is_init(x) && is_init(y) && error("x is not initialised, but y is")
    return x
end
function increment_internal!!(c::IncCache, x::T, y::T) where {T<:Tangent}
    return T(increment_internal!!(c, x.fields, y.fields))
end
function increment_internal!!(c::IncCache, x::T, y::T) where {T<:MutableTangent}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x.fields = increment_internal!!(c, x.fields, y.fields)
    return x
end

"""
    set_to_zero!!(x)

Set `x` to its zero element (`x` should be a tangent, so the zero must exist).
"""
set_to_zero!!(x) = set_to_zero!!(x, require_tangent_cache(typeof(x)))
set_to_zero!!(x, ::Val{true}) = set_to_zero_internal!!(Vector{UInt}(), x)
set_to_zero!!(x, ::Val{false}) = set_to_zero_internal!!(NoCache(), x)

"""
    set_to_zero_internal!!(c::SetToZeroCache, x)

Implementation for [`Mooncake.set_to_zero!!`](@ref). Use `c` to ensure that circular
references are correctly handled. If `c` is a `NoCache`, assume no circular references.
"""
set_to_zero_internal!!(::SetToZeroCache, ::NoTangent) = NoTangent()
set_to_zero_internal!!(::SetToZeroCache, x::Base.IEEEFloat) = zero(x)
function set_to_zero_internal!!(c::SetToZeroCache, x::Union{Tuple,NamedTuple})
    return tuple_map(Base.Fix1(set_to_zero_internal!!, c), x)
end
function set_to_zero_internal!!(c::SetToZeroCache, x::T) where {T<:PossiblyUninitTangent}
    return is_init(x) ? T(set_to_zero_internal!!(c, val(x))) : x
end
function set_to_zero_internal!!(c::SetToZeroCache, x::T) where {T<:Tangent}
    return T(set_to_zero_internal!!(c, x.fields))
end
function set_to_zero_internal!!(c::SetToZeroCache, x::MutableTangent)
    _already_tracked!(c, x) && return x
    x.fields = set_to_zero_internal!!(c, x.fields)
    return x
end

"""
    _scale(a::Float64, t::T) where {T}

Required for testing.
Should be defined for all standard tangent types.

Multiply tangent `t` by scalar `a`. Always possible because any given tangent type must
correspond to a vector field. Not using `*` in order to avoid piracy.
"""
_scale(a::Float64, t) = _scale_internal(IdDict{Any,Any}(), a, t)

"""
    _scale_internal(c::MaybeCache, a::Float64, t)

Implementation for [`_scale`](@ref). Use `c` to handle circular references and aliasing in
`t`. If `c` is a `NoCache` assume no circular references or aliasing in `c`.
"""
_scale_internal(::MaybeCache, ::Float64, ::NoTangent) = NoTangent()
_scale_internal(::MaybeCache, a::Float64, t::T) where {T<:IEEEFloat} = T(a * t)
@unstable function _scale_internal(c::MaybeCache, a::Float64, t::Union{Tuple,NamedTuple})
    return map(ti -> _scale_internal(c, a, ti)::typeof(ti), t)
end
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:PossiblyUninitTangent}
    return is_init(t) ? T(_scale_internal(c, a, val(t))) : T()
end
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:Tangent}
    return T(_scale_internal(c, a, t.fields))
end
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:MutableTangent}
    haskey(c, t) && return c[t]::T
    y = T()
    c[t] = y
    y.fields = _scale_internal(c, a, t.fields)
    return y
end

struct FieldUndefined end

"""
    _dot(t::T, s::T)::Float64 where {T}

Required for testing.
Should be defined for all standard tangent types.

Inner product between tangents `t` and `s`. Must return a `Float64`.
Always available because all tangent types correspond to finite-dimensional vector spaces.
"""
_dot(t::T, s::T) where {T} = _dot_internal(IdDict{Any,Any}(), t, s)::Float64

"""
    _dot_internal(c::MaybeCache, t::T, s::T) where {T}

Implementation for [`_dot`](@ref). Use `c` to handle circular references and aliasing.
If `c` is a `NoCache`, assume that neither `t` nor `s` contain either circular references
or aliasing.
"""
_dot_internal(::MaybeCache, ::NoTangent, ::NoTangent) = 0.0
_dot_internal(::MaybeCache, t::T, s::T) where {T<:IEEEFloat} = Float64(t * s)
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:Union{Tuple,NamedTuple}}
    return sum(map((t, s) -> _dot_internal(c, t, s)::Float64, t, s); init=0.0)::Float64
end
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:PossiblyUninitTangent}
    is_init(t) && is_init(s) && return _dot_internal(c, val(t), val(s))::Float64
    return 0.0
end
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:Union{Tangent,MutableTangent}}
    key = (t, s)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    return sum(
        _map((t, s) -> _dot_internal(c, t, s)::Float64, t.fields, s.fields); init=0.0
    )::Float64
end

"""
    _add_to_primal(p::P, t::T, unsafe::Bool=false) where {P, T}

Adds `t` to `p`, returning a `P`. It must be the case that `tangent_type(P) == T`.

If `unsafe` is `true` and `P` is a composite type, then `_add_to_primal` will construct a
new instance of `P` by directly invoking the `:new` instruction for `P`, rather than
attempting to use the default constructor for `P`. This is fine if you are confident that
the new `P` constructed by adding `t` to `p` will always be a valid instance of `P`, but
could cause problems if you are not confident of this.

This is, for example, fine for the following type:
```julia
struct Foo{T}
    x::Vector{T}
    y::Vector{T}
    function Foo(x::Vector{T}, y::Vector{T}) where {T}
        @assert length(x) == length(y)
        return new{T}(x, y)
    end
end
```
Here, the value returned by `_add_to_primal` will satisfy the invariant asserted in the
inner constructor for `Foo`.
"""
function _add_to_primal(p, t, unsafe::Bool=false)
    return _add_to_primal_internal(IdDict{Any,Any}(), p, t, unsafe)::typeof(p)
end

"""
    _add_to_primal_internal(c::MaybeCache, x, t, ::Bool)

Implementation for [`_add_to_primal`](@ref). Use `c` to handle circular referencing and
aliasing correctly. If `c` is a `NoCache`, assume there is no circular references or
aliasing in either `x` or `t`.
"""
_add_to_primal_internal(::MaybeCache, x, ::NoTangent, ::Bool) = x
_add_to_primal_internal(::MaybeCache, x::T, t::T, ::Bool) where {T<:IEEEFloat} = x + t
function _add_to_primal_internal(
    c::MaybeCache, x::SimpleVector, t::Vector{Any}, unsafe::Bool
)
    haskey(c, (x, t, unsafe)) && return c[(x, t, unsafe)]::SimpleVector
    x′ = svec(map(n -> _add_to_primal_internal(c, x[n], t[n], unsafe), eachindex(x))...)
    c[(x, t, unsafe)] = x′
    return x′
end
function _add_to_primal_internal(c::MaybeCache, x::Tuple, t::Tuple, unsafe::Bool)
    return _map((x, t) -> _add_to_primal_internal(c, x, t, unsafe), x, t)::typeof(x)
end
function _add_to_primal_internal(c::MaybeCache, x::NamedTuple, t::NamedTuple, unsafe::Bool)
    return _map((x, t) -> _add_to_primal_internal(c, x, t, unsafe), x, t)::typeof(x)
end

struct AddToPrimalException <: Exception
    primal_type::Type
end

function Base.showerror(io::IO, err::AddToPrimalException)
    msg =
        "Attempted to construct an instance of $(err.primal_type) using the default " *
        "constuctor. In most cases, this error is caused by the lack of existence of the " *
        "default constructor for this type. There are two approaches to dealing with " *
        "this problem. The first is to avoid having to call `_add_to_primal` on this " *
        "type, which can be achieved by avoiding testing functions whose arguments are " *
        "of this type. If this cannot be avoided, you should consider using calling " *
        "`Mooncake._add_to_primal` with its third positional argument set to `true`. " *
        "If you are using some of Mooncake's testing functionality, this can be achieved " *
        "by setting the `unsafe_perturb` setting to `true` -- check the docstring " *
        "for `Mooncake._add_to_primal` to ensure that your use case is unlikely to " *
        "cause problems."
    return println(io, msg)
end

function __construct_type(::Type{P}, unsafe::Bool, fields::Vararg{Any,N})::P where {P,N}
    i = findfirst(==(FieldUndefined()), fields)

    # If unsafe mode is enabled, then call `_new_` directly, and avoid the possibility that
    # the default inner constructor for `P` does not exist.
    if unsafe
        return i === nothing ? _new_(P, fields...) : _new_(P, fields[1:(i - 1)]...)
    end

    # If unsafe mode is disabled, try to use the default constructor for `P`. If this does
    # not work, then throw an informative error message.
    try
        return i === nothing ? P(fields...) : P(fields[1:(i - 1)]...)
    catch e
        if e isa MethodError
            throw(AddToPrimalException(P))
        else
            rethrow(e)
        end
    end
end

function _add_to_primal_internal(
    c::MaybeCache, p::P, t::T, unsafe::Bool
) where {P,T<:Tangent}
    Tt = tangent_type(P)
    if Tt != typeof(t)
        throw(ArgumentError("p of type $P has tangent_type $Tt, but t is of type $T"))
    end
    fields = map(fieldnames(P)) do f
        tf = getfield(t.fields, f)
        isdefined(p, f) &&
            is_init(tf) &&
            return _add_to_primal_internal(c, getfield(p, f), val(tf), unsafe)
        !isdefined(p, f) && !is_init(tf) && return FieldUndefined()
        throw(error("unable to handle undefined-ness"))
    end
    return __construct_type(P, unsafe, fields...)::P
end

function _add_to_primal_internal(
    c::MaybeCache, p::P, t::T, unsafe::Bool
) where {P,T<:MutableTangent}

    # Do not recompute if we already have a perturbed primal.
    key = (p, t, unsafe)
    haskey(c, key) && return c[key]::P

    # Check that `T` is the correct tangent type for `P`.
    Tt = tangent_type(P)
    if Tt != typeof(t)
        throw(ArgumentError("p of type $P has tangent_type $Tt, but t is of type $T"))
    end

    # For all const fields, it is safe to immediately recurse and construct the primal, as
    # it is not possible to have a field marked as const which contains a circular reference
    # to `p`. Other (defined) fields are given placeholder values, and revisited in a second
    # pass over the data structure.
    init_fields = map(fieldnames(P)) do f
        tf = getfield(t.fields, f)
        if isdefined(p, f) && is_init(tf) && isconst(P, f)
            return _add_to_primal_internal(c, getfield(p, f), val(tf), unsafe)
        elseif isdefined(p, f) && is_init(tf) && !isconst(P, f)
            return getfield(p, f)
        elseif !isdefined(p, f) && !is_init(tf)
            return FieldUndefined()
        else
            throw(error("unable to handle undefined-ness"))
        end
    end

    # Construct an initial version of perturbed `p`, in which all (defined) constants fields
    # are perturbed, but all fields which are not marked as const are the same as in `p`.
    p′ = __construct_type(P, unsafe, init_fields...)
    c[key] = p′

    # Now that we are protected against circular references in `p`, perturb each defined
    # mutable field in `p′`.
    map(fieldnames(P)) do f
        tf = getfield(t.fields, f)
        if isdefined(p, f) && is_init(tf) && !isconst(P, f)
            setfield!(p′, f, _add_to_primal_internal(c, getfield(p, f), val(tf), unsafe))
        end
    end
    return p′::P
end

"""
    _diff(p::P, q::P) where {P}

Required for testing.

Computes the difference between `p` and `q`, which _must_ be of the same type, `P`.
Returns a tangent of type `tangent_type(P)`.
"""
_diff(p::P, q::P) where {P} = _diff_internal(IdDict{Any,Any}(), p, q)::tangent_type(P)

"""
    _diff_internal(c::MaybeCache, p::P, q::P) where {P}

Implmentation for [`_diff`](@ref). Use `c` to correctly handle circular references and
aliasing. If `c` is a `NoCache` then assume no circular references or aliasing in either
`p` or `q`.
"""
function _diff_internal(c::MaybeCache, p::P, q::P) where {P}
    @assert typeof(p) == typeof(q) # this function implicitly assumes p and q have identical type
    TP = tangent_type(P)
    TP === NoTangent && return NoTangent()
    T = Tangent{NamedTuple{(),Tuple{}}}
    TP === T && return T((;))
    key = (p, q)
    haskey(c, key) && return c[key]::TP
    return _containerlike_diff(c, p, q)::TP
end
_diff_internal(::MaybeCache, p::P, q::P) where {P<:IEEEFloat} = p - q
function _diff_internal(c::MaybeCache, p::P, q::P) where {P<:SimpleVector}
    key = (p, q)
    haskey(c, key) && return c[key]::tangent_type(P)
    t = Any[_diff_internal(c, a, b) for (a, b) in zip(p, q)]
    c[key] = t
    return t
end
function _diff_internal(c::MaybeCache, p::P, q::P) where {P<:Union{Tuple,NamedTuple}}
    TP = tangent_type(P)
    TP == NoTangent && return NoTangent()
    return _map((p, q) -> _diff_internal(c, p, q), p, q)::TP
end

function _containerlike_diff(c::MaybeCache, p::P, q::P) where {P}
    return _containerlike_diff_cartesian(c, p, q, Val(ismutabletype(P)), Val(fieldcount(P)))
end
@generated function _containerlike_diff_cartesian(
    c::MaybeCache, p::P, q::P, ::Val{mutable}, ::Val{nfield}
) where {P,mutable,nfield}
    quote
        t = if mutable
            _t = tangent_type(P)()
            c[(p, q)] = _t
            _t
        else
            nothing
        end
        Base.Cartesian.@nif(
            $(nfield + 1),
            n -> let
                defined_p = isdefined(p, n)
                defined_q = isdefined(q, n)
                defined_p != defined_q && throw(error("Unhandleable undefinedness"))

                !defined_p
            end,
            # We have found the first undefined field, or,
            # if n == $(nfield + 1), then we have found the last field,
            # and all fields are defined.
            n -> _containerlike_diff_cartesian_internal(
                Val(n), c, p, q, t, Val(mutable), Val(nfield)
            )
        )
    end
end
@generated function _containerlike_diff_cartesian_internal(
    ::Val{n}, c::MaybeCache, p::P, q::P, t, ::Val{mutable}, ::Val{nfield}
) where {P,n,mutable,nfield}
    quote
        diffed_fields = Base.Cartesian.@ntuple(
            $(n - 1),
            m -> _diff_internal(
                c, getfield(p, m), getfield(q, m)
            )::tangent_type(fieldtype(P, m))
        )
        if mutable
            return _build_tangent(P, t, diffed_fields...)
        else
            return build_tangent(P, diffed_fields...)
        end
    end
end

# For mutable types.
@generated function _build_tangent(::Type{P}, t::T, fields::Vararg{Any,N}) where {P,T,N}
    tangent_values_exprs = map(enumerate(tangent_field_types(P))) do (n, tt)
        tt <: PossiblyUninitTangent && return n <= N ? :($tt(fields[$n])) : :($tt())
        return :(fields[$n])
    end
    nt_expr = Expr(:call, backing_type(P), Expr(:tuple, tangent_values_exprs...))
    return Expr(:block, Expr(:call, :setfield!, :t, :(:fields), nt_expr), :(return t))
end

"""
    increment_field!!(x::T, y::V, f) where {T, V}

`increment!!` the field `f` of `x` by `y`, and return the updated `x`.
"""
@inline @generated function increment_field!!(x::Tuple, y, ::Val{i}) where {i}
    exprs = map(n -> n == i ? :(increment!!(x[$n], y)) : :(x[$n]), fieldnames(x))
    return Expr(:tuple, exprs...)
end

# Optimal for homogeneously-typed Tuples with dynamic field choice. Implementation using
# `ifelse` chosen to ensure that the entire function comprises a single basic block. If
# instead one wrote `n -> n == i ? v : x[n]` we would get one basic block per element of
# `x`. This is fine for small-medium `x`, but causes a great deal of trouble for large `x`
# (certainly for length > 1_000, but probably also for smaller sizes than that).
function increment_field!!(x::Tuple, y, i::Int)
    v = increment!!(x[i], y)
    return ntuple(n -> ifelse(n == i, v, x[n]), Val(length(x)))
end

@inline @generated function increment_field!!(x::T, y, ::Val{f}) where {T<:NamedTuple,f}
    i = f isa Symbol ? findfirst(==(f), fieldnames(T)) : f
    new_fields = Expr(:call, increment_field!!, :(Tuple(x)), :y, :(Val($i)))
    return Expr(:call, T, new_fields)
end

# Optimal for homogeneously-typed NamedTuples with dynamic field choice.
function increment_field!!(x::T, y, i::Int) where {T<:NamedTuple}
    return T(increment_field!!(Tuple(x), y, i))
end
function increment_field!!(x::T, y, s::Symbol) where {T<:NamedTuple}
    return T(tuple_map(n -> n == s ? increment!!(x[n], y) : x[n], fieldnames(T)))
end

function increment_field!!(x::Tangent{T}, y, f::Val{F}) where {T,F}
    y isa NoTangent && return x
    new_val = fieldtype(T, F) <: PossiblyUninitTangent ? fieldtype(T, F)(y) : y
    return Tangent(increment_field!!(x.fields, new_val, f))
end
function increment_field!!(x::MutableTangent{T}, y, f::V) where {T,F,V<:Val{F}}
    y isa NoTangent && return x
    new_val = fieldtype(T, F) <: PossiblyUninitTangent ? fieldtype(T, F)(y) : y
    setfield!(x, :fields, increment_field!!(x.fields, new_val, f))
    return x
end

@unstable @inline increment_field!!(x, y, f::Symbol) = increment_field!!(x, y, Val(f))
@unstable @inline increment_field!!(x, y, n::Int) = increment_field!!(x, y, Val(n))

# Fallback method for when a tangent type for a struct is declared to be `NoTangent`.
for T in [Symbol, Int, Val]
    @eval increment_field!!(::NoTangent, ::NoTangent, f::Union{$T}) = NoTangent()
end

"""
    tangent_test_cases()

Constructs a `Vector` of `Tuple`s containing test cases for the tangent infrastructure.

If the returned tuple has 2 elements, the elements should be interpreted as follows:
1 - interface_only
2 - primal value

interface_only is a Bool which will be used to determine which subset of tests to run.

If the returned tuple has 5 elements, then the elements are interpreted as follows:
1 - interface_only
2 - primal value
3, 4, 5 - tangents, where <5> == increment!!(<3>, <4>).

Test cases in the first format make use of `zero_tangent` / `randn_tangent` etc to generate
tangents, but they're unable to check that `increment!!` is correct in an absolute sense.
"""
@unstable function tangent_test_cases()
    N_large = 33
    _names = Tuple(map(n -> Symbol("x$n"), 1:N_large))

    abs_test_cases = [
        (sin, NoTangent),
        (Float16(5.0), Float16),
        (5.0f0, Float32),
        (5.1, Float64),
        (svec(5.0), Vector{Any}),
        ([3.0, 2.0], Vector{Float64}),
        (Float64[], Vector{Float64}),
        ([1, 2], Vector{NoTangent}),
        ([[1.0], [1.0, 2.0]], Vector{Vector{Float64}}),
        (setindex!(Vector{Vector{Float64}}(undef, 2), [1.0], 1), Vector{Vector{Float64}}),
        (setindex!(Vector{Vector{Float64}}(undef, 2), [1.0], 2), Vector{Vector{Float64}}),
        ((6.0, [1.0, 2.0]), Tuple{Float64,Vector{Float64}}),
        ((), NoTangent),
        ((1,), NoTangent),
        ((2, 3), NoTangent),
        (Mooncake.tuple_fill(5.0, Val(N_large)), NTuple{N_large,Float64}),
        ((a=6.0, b=[1.0, 2.0]), @NamedTuple{a::Float64, b::Vector{Float64}}),
        ((;), NoTangent),
        (
            NamedTuple{_names}(Mooncake.tuple_fill(5.0, Val(N_large))),
            NamedTuple{_names,NTuple{N_large,Float64}},
        ),
        (UnitRange{Int}(5, 7), NoTangent),
        (Array, NoTangent),
        (Float64, NoTangent),
        (BigInt, NoTangent),
        (Union{Float64,Float32}, NoTangent),
        (Union, NoTangent),
        (UnionAll, NoTangent),
        (typeof(<:), NoTangent),
        (Base.CoreLogging.SimpleLogger, NoTangent),
        (IOStream(""), NoTangent),
    ]
    # Construct test cases containing circular references. These typically require multiple
    # lines of code to construct, so we build them before adding them to `rel_test_cases`.
    circular_vector = Any[5.0]
    circular_vector[1] = circular_vector

    rel_test_cases = Any[
        TestResources.StructFoo(6.0, [1.0, 2.0]),
        TestResources.StructFoo(6.0),
        TestResources.MutableFoo(6.0, [1.0, 2.0]),
        TestResources.MutableFoo(6.0),
        TestResources.StructNoFwds(5.0),
        TestResources.StructNoRvs([5.0]),
        TestResources.TypeStableMutableStruct{Float64}(5.0, 3.0),
        LowerTriangular{Float64,Matrix{Float64}}(randn(2, 2)),
        UpperTriangular{Float64,Matrix{Float64}}(randn(2, 2)),
        UnitLowerTriangular{Float64,Matrix{Float64}}(randn(2, 2)),
        UnitUpperTriangular{Float64,Matrix{Float64}}(randn(2, 2)),
        (2.0, 3),
        (3, 2.0),
        (2.0, 1.0),
        (randn(10), 3),
        (3, randn(10)),
        (randn(10), randn(10)),
        (a=2.0, b=3),
        (a=3, b=2.0),
        (a=randn(10), b=3),
        (a=3, b=randn(10)),
        (a=randn(10), b=randn(10)),
        (Base.TOML.ErrorType(1), NoTangent()), # Enum
        circular_vector,
        TestResources.make_circular_reference_struct(),
        TestResources.make_indirect_circular_reference_array(),
        # Regression tests to catch type inference failures, see https://github.com/chalk-lab/Mooncake.jl/pull/422
        (((((randn(33)...,),),),),),
        (((((((((randn(33)...,),),),),), randn(5)...),),),),
        Base.OneTo{Int},
        TestResources.build_big_isbits_struct(),
    ]
    VERSION >= v"1.11" && push!(rel_test_cases, fill!(Memory{Float64}(undef, 3), 3.0))
    return vcat(
        map(x -> (false, x...), abs_test_cases),
        map(x -> (false, x), rel_test_cases),
        map(Mooncake.TestTypes.instantiate, Mooncake.TestTypes.PRIMALS),
    )
end
