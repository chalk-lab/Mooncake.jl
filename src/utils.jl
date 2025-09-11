"""
    _typeof(x)

Central definition of typeof, which is specific to the use-required in this package.
"""
@unstable _typeof(x) = Base._stable_typeof(x)
@unstable _typeof(x::Tuple) = Tuple{tuple_map(_typeof, x)...}
@unstable _typeof(x::NamedTuple{names}) where {names} = NamedTuple{names,_typeof(Tuple(x))}

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
    return Expr(:call, :tuple, map(n -> :(f(getfield(x, $n))), 1:fieldcount(x))...)
end

@inline @generated function tuple_map(f::F, x::Tuple, y::Tuple) where {F}
    if length(x.parameters) != length(y.parameters)
        return :(throw(ArgumentError("length(x) != length(y)")))
    else
        stmts = map(n -> :(f(getfield(x, $n), getfield(y, $n))), 1:fieldcount(x))
        return Expr(:call, :tuple, stmts...)
    end
end

@generated function tuple_map(f, x::NamedTuple{names}) where {names}
    getfield_exprs = map(n -> :(f(getfield(x, $n))), 1:fieldcount(x))
    return :(NamedTuple{names}($(Expr(:call, :tuple, getfield_exprs...))))
end

@generated function tuple_map(f, x::NamedTuple{names}, y::NamedTuple{names}) where {names}
    if fieldcount(x) != fieldcount(y)
        return :(throw(ArgumentError("length(x) != length(y)")))
    end
    getfield_exprs = map(n -> :(f(getfield(x, $n), getfield(y, $n))), 1:fieldcount(x))
    return :(NamedTuple{names}($(Expr(:call, :tuple, getfield_exprs...))))
end

for N in 1:256
    @eval @inline function tuple_splat(f, x::Tuple{Vararg{Any,$N}})
        return $(Expr(:call, :f, map(n -> :(getfield(x, $n)), 1:N)...))
    end
end

@inline @generated function tuple_splat(f, x::Tuple)
    return Expr(:call, :f, map(n -> :(x[$n]), 1:length(x.parameters))...)
end

@inline @generated function tuple_splat(f, v, x::Tuple)
    return Expr(:call, :f, :v, map(n -> :(x[$n]), 1:length(x.parameters))...)
end

@inline @generated function tuple_fill(val, ::Val{N}) where {N}
    return Expr(:call, :tuple, map(_ -> :val, 1:N)...)
end

"""
    _findall(cond, x::Tuple)

Type-stable version of `findall` for `Tuple`s. Should constant-fold if `cond` can be
determined from the type of `x`.
"""
@inline @generated function _findall(cond, x::Tuple)

    # Initially we have found nothing.
    y = :(y = ())

    # For each element in `x`, if it satisfies `cond`, insert its index into `y`.
    exprs = map(n -> :(y = cond(x[$n]) ? ($n, y...) : y), 1:fieldcount(x))

    # Combine all expressions into a single block and return.
    return Expr(:block, y, exprs...)
end

"""
    stable_all(x::NTuple{N, Bool}) where {N}

`all(x::NTuple{N, Bool})` does not constant-fold nicely on 1.10 if the values of `x` are
known statically. This implementation constant-folds nicely on both 1.10 and 1.11, so can
be used in its place in situations where this is important.
"""
@generated function stable_all(x::NTuple{N,Bool}) where {N}

    # For each element in `x`, if it is `false`, return `false`.
    exprs = map(n -> :(x[$n] || return false), 1:N)

    # I've we've not found any false elements, return `true`.
    return Expr(:block, exprs..., :(return true))
end

"""
    _map_if_assigned!(f, y::DenseArray, x::DenseArray{P}) where {P}

For all `n`, if `x[n]` is assigned, then writes the value returned by `f(x[n])` to `y[n]`,
otherwise leaves `y[n]` unchanged.

Equivalent to `map!(f, y, x)` if `P` is a bits type as element will always be assigned.

Requires that `y` and `x` have the same size.
"""
function _map_if_assigned!(f::F, y::DenseArray, x::DenseArray{P}) where {F,P}
    @assert size(y) == size(x)
    @inbounds for n in eachindex(y)
        if isbitstype(P) || isassigned(x, n)
            y[n] = f(x[n])
        end
    end
    return y
end

"""
    _map_if_assigned!(f::F, y::DenseArray, x1::DenseArray{P}, x2::DenseArray)

Similar to the other method of `_map_if_assigned!` -- for all `n`, if `x1[n]` is assigned,
writes `f(x1[n], x2[n])` to `y[n]`, otherwise leaves `y[n]` unchanged.

Requires that `y`, `x1`, and `x2` have the same size.
"""
function _map_if_assigned!(
    f::F, y::DenseArray, x1::DenseArray{P}, x2::DenseArray
) where {F,P}
    @assert size(y) == size(x1)
    @assert size(y) == size(x2)
    @inbounds for n in eachindex(y)
        if isbitstype(P) || isassigned(x1, n)
            y[n] = f(x1[n], x2[n])
        end
    end
    return y
end

"""
    _map(f, x...)

Same as `map` but requires all elements of `x` to have equal length.
The usual function `map` doesn't enforce this for `Array`s.
"""
@unstable @inline function _map(f::F, x::Vararg{Any,N}) where {F,N}
    @assert allequal(map(length, x))
    return map(f, x...)
end

"""
    is_vararg_and_sparam_names(m::Method)

Returns a 2-tuple. The first element is true if `m` is a vararg method, and false if not.
The second element contains the names of the static parameters associated to `m`.
"""
is_vararg_and_sparam_names(m::Method)::Tuple{Bool,Vector{Symbol}} = m.isva, sparam_names(m)

"""
    is_vararg_and_sparam_names(sig)::Tuple{Bool, Vector{Symbol}}

Finds the method associated to `sig`, and calls `is_vararg_and_sparam_names` on it.
"""
function is_vararg_and_sparam_names(sig)::Tuple{Bool,Vector{Symbol}}
    world = Base.get_world_counter()
    min = Base.RefValue{UInt}(typemin(UInt))
    max = Base.RefValue{UInt}(typemax(UInt))
    ms = Base._methods_by_ftype(
        sig, nothing, -1, world, true, min, max, Ptr{Int32}(C_NULL)
    )::Vector
    return is_vararg_and_sparam_names(only(ms).method)
end

"""
    is_vararg_and_sparam_names(mi::Core.MethodInstance)

Calls `is_vararg_and_sparam_names` on `mi.def::Method`.
"""
function is_vararg_and_sparam_names(mi::Core.MethodInstance)::Tuple{Bool,Vector{Symbol}}
    return is_vararg_and_sparam_names(mi.def)
end

"""
    is_vararg_and_sparam_names(mc::MistyClosure)::Tuple{Bool,Vector{Symbol}}

Basic implementation for MistyClosure. Assumes is not a varargs function, and has no static
parameter names appearing in the source.
"""
is_vararg_and_sparam_names(::MistyClosure)::Tuple{Bool,Vector{Symbol}} = false, Symbol[]

"""
    sparam_names(m::Core.Method)::Vector{Symbol}

Returns the names of all of the static parameters in `m`.
"""
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
    always_initialised(::Type{P}) where {P}

Returns a tuple with number of fields equal to the number of fields in `P`. The nth field
is set to `true` if the nth field of `P` is initialised, and `false` otherwise.
"""
@generated function always_initialised(::Type{P}) where {P}
    P isa DataType || return :(error("$P is not a DataType."))
    num_init = CC.datatype_min_ninitialized(P)
    return (map(n -> n <= num_init, 1:fieldcount(P))...,)
end

"""
    is_always_initialised(P::DataType, n::Int)::Bool

True if the `n`th field of `P` is always initialised. If the `n`th fieldtype of `P`
`isbitstype`, then this is distinct from asking whether the `n`th field is always defined.
An isbits field is always defined, but is not always explicitly initialised.
"""
function is_always_initialised(P::DataType, n::Int)::Bool
    return n <= CC.datatype_min_ninitialized(P)
end

"""
    is_always_fully_initialised(P::DataType)::Bool

True if all fields in `P` are always initialised. Put differently, there are no inner
constructors which permit partial initialisation.
"""
function is_always_fully_initialised(P::DataType)::Bool
    return CC.datatype_min_ninitialized(P) == fieldcount(P)
end

"""
    lgetfield(x, ::Val{f}, ::Val{order}) where {f, order}

Like `getfield`, but with the field and access order encoded as types.
"""
lgetfield(x, ::Val{f}, ::Val{order}) where {f,order} = getfield(x, f, order)

"""
    lsetfield!(value, name::Val, x, [order::Val])

This function is to `setfield!` what `lgetfield` is to `getfield`. It will always hold that
```julia
setfield!(copy(x), :f, v) == lsetfield!(copy(x), Val(:f), v)
setfield!(copy(x), 2, v) == lsetfield(copy(x), Val(2), v)
```
"""
lsetfield!(value, ::Val{name}, x) where {name} = setfield!(value, name, x)

"""
    _new_(::Type{T}, x::Vararg{Any, N}) where {T, N}

One-liner which calls the `:new` instruction with type `T` with arguments `x`.
"""
@inline @generated function _new_(::Type{T}, x::Vararg{Any,N}) where {T,N}
    return Expr(:new, :T, map(n -> :(x[$n]), 1:N)...)
end

"""
    flat_product(xs...)

Equivalent to `vec(collect(Iterators.product(xs...)))`.
"""
flat_product(xs...) = vec(collect(Iterators.product(xs...)))

"""
    map_prod(f, xs...)

Equivalent to `map(f, flat_product(xs...))`.
"""
map_prod(f, xs...) = map(f, flat_product(xs...))

"""
    opaque_closure(
        ret_type::Type,
        ir::IRCode,
        @nospecialize env...;
        isva::Bool=false,
        do_compile::Bool=true,
    )::Core.OpaqueClosure{<:Tuple, ret_type}

Construct a `Core.OpaqueClosure`. Almost equivalent to
`Core.OpaqueClosure(ir, env...; isva, do_compile)`, but instead of letting
`Core.compute_oc_rettype` figure out the return type from `ir`, impose `ret_type` as the
return type.

# Warning

User beware: if the `Core.OpaqueClosure` produced by this function ever returns anything
which is not an instance of a subtype of `ret_type`, you should expect all kinds of awful
things to happen, such as segfaults. You have been warned!

# Extended Help

This is needed in Mooncake.jl because make extensive use of our ability to know the return
type of a couple of specific `OpaqueClosure`s without actually having constructed them --
see `LazyDerivedRule`. Without the capability to specify the return type, we have to guess
what type `compute_ir_rettype` will return for a given `IRCode` before we have constructed
the `IRCode` and run type inference on it. This exposes us to details of type inference,
which are not part of the public interface of the language, and can therefore vary from
Julia version to Julia version (including patch versions). Moreover, even for a fixed Julia
version it can be extremely hard to predict exactly what type inference will infer to be the
return type of a function.

Failing to correctly guess the return type can happen for a number of reasons, and the kinds
of errors that tend to be generated when this fails tell you very little about the
underlying cause of the problem.

By specifying the return type ourselves, we remove this dependence. The price we pay for
this is the potential for segfaults etc if we fail to specify `ret_type` correctly.
"""
function opaque_closure(
    ret_type::Type,
    ir::IRCode,
    @nospecialize env...;
    isva::Bool=false,
    do_compile::Bool=true,
)
    # This implementation is copied over directly from `Core.OpaqueClosure`.
    ir = CC.copy(ir)
    nargs = length(ir.argtypes) - 1
    sig = Base.Experimental.compute_oc_signature(ir, nargs, isva)
    src = ccall(:jl_new_code_info_uninit, Ref{CC.CodeInfo}, ())
    src.slotnames = fill(:none, nargs + 1)
    src.slotflags = fill(zero(UInt8), length(ir.argtypes))
    src.slottypes = copy(ir.argtypes)
    src.rettype = ret_type
    src = CC.ir_to_codeinf!(src, ir)
    return Base.Experimental.generate_opaque_closure(
        sig, Union{}, ret_type, src, nargs, isva, env...; do_compile
    )::Core.OpaqueClosure{sig,ret_type}
end

"""
    misty_closure(
        ret_type::Type,
        ir::IRCode,
        @nospecialize env...;
        isva::Bool=false,
        do_compile::Bool=true,
    )

Identical to [`Mooncake.opaque_closure`](@ref), but returns a `MistyClosure` closure rather
than a `Core.OpaqueClosure`.
"""
function misty_closure(
    ret_type::Type,
    ir::IRCode,
    @nospecialize env...;
    isva::Bool=false,
    do_compile::Bool=true,
)
    return MistyClosure(opaque_closure(ret_type, ir, env...; isva, do_compile), Ref(ir))
end

"""
    _copytrito!(B::AbstractMatrix, A::AbstractMatrix, uplo::AbstractChar)

Literally just copied over from Julia's LinearAlgebra std lib, in order to produce a method
which works on 1.10.
"""
function _copytrito!(B::AbstractMatrix, A::AbstractMatrix, uplo::AbstractChar)
    @static if VERSION >= v"1.11"
        return copytrito!(B, A, uplo)
    end
    Base.require_one_based_indexing(A, B)
    BLAS.chkuplo(uplo)
    m, n = size(A)
    m1, n1 = size(B)
    A = Base.unalias(B, A)
    if uplo == 'U'
        if n < m
            (m1 < n || n1 < n) && throw(
                DimensionMismatch(
                    lazy"B of size ($m1,$n1) should have at least size ($n,$n)"
                ),
            )
        else
            (m1 < m || n1 < n) && throw(
                DimensionMismatch(
                    lazy"B of size ($m1,$n1) should have at least size ($m,$n)"
                ),
            )
        end
        for j in 1:n, i in 1:min(j, m)
            @inbounds B[i, j] = A[i, j]
        end
    else # uplo == 'L'
        if m < n
            (m1 < m || n1 < m) && throw(
                DimensionMismatch(
                    lazy"B of size ($m1,$n1) should have at least size ($m,$m)"
                ),
            )
        else
            (m1 < m || n1 < n) && throw(
                DimensionMismatch(
                    lazy"B of size ($m1,$n1) should have at least size ($m,$n)"
                ),
            )
        end
        for j in 1:n, i in j:m
            @inbounds B[i, j] = A[i, j]
        end
    end
    return B
end

"""
    _copy(x)

*Note:* this is not part of the public Mooncake.jl interface, and may change without warning.

Internal protocol for creating copies of AD-related data structures. This function is used 
throughout the automatic differentiation system to create appropriate copies of rules, 
caches, and other internal data structures when building new AD contexts.

# Semantics

The `_copy` protocol defines how different types should be copied within the AD system:

- For immutable AD types (like `CoDual`, `Dual`), typically returns the same object
- For mutable containers, creates new instances with copied contents
- For composite types, recursively applies `_copy` to fields
- Falls back to `Base.copy` for general types

# Implementation Requirements

When implementing `_copy` for a new type, consider:
- Whether the type represents mutable state that needs actual copying
- Whether fields should be recursively copied or can be shared
- Performance implications of copying vs. sharing immutable data

# Examples

```julia
# For immutable AD types - no copying needed
_copy(x::CoDual) = x

# For `Stack` type - create new empty instance
_copy(::Stack{T}) where {T} = Stack{T}()

# For composite types - recursive copying
_copy(x::Tuple) = map(_copy, x)

# For rule types - create new instances with appropriately copied captures/caches
function _copy(x::DerivedRule)
    new_captures = _copy(x.fwds_oc.oc.captures)
    new_fwds_oc = replace_captures(x.fwds_oc, new_captures)
    new_pb_oc_ref = Ref(replace_captures(x.pb_oc_ref[], new_captures))
    return typeof(x)(new_fwds_oc, new_pb_oc_ref, x.nargs)
end

# For misty closure reverse data - deep copy the captures data
_copy(r::MistyClosureRData) = MistyClosureRData(deepcopy(r.captures_rdata))

# For tangent types - copy conditionally based on initialization state
_copy(x::PossiblyUninitTangent) = is_init(x) ? typeof(x)(_copy(x.tangent)) : typeof(x)()

# For forwards/reverse data types - recursively copy wrapped data
_copy(x::FData) = typeof(x)(_copy(x.data))
_copy(x::RData) = typeof(x)(_copy(x.data))
_copy(x::LazyZeroRData) = typeof(x)(_copy(x.data))

# Fallback to Base.copy
_copy(x) = copy(x)
```
"""

# Generic implementations that work with any type
_copy(::Nothing) = nothing
_copy(x::Symbol) = x
_copy(x::Tuple) = map(_copy, x)
_copy(x::NamedTuple) = map(_copy, x)
_copy(x::Ref{T}) where {T} = isassigned(x) ? Ref{T}(_copy(x[])) : Ref{T}()
_copy(x::Type) = x

# Fallback to Base.copy for all other types
_copy(x) = copy(x)
