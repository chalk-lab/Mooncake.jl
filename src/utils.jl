"""
    _typeof(x)

Central definition of typeof, which is specific to the use-required in this package.
"""
@unstable _typeof(x) = Base._stable_typeof(x)
@unstable _typeof(x::Tuple) = Tuple{tuple_map(_typeof, x)...}
@unstable _typeof(x::NamedTuple{names}) where {names} = NamedTuple{names,_typeof(Tuple(x))}

function _print_boxed_message(io::IO, level::AbstractString, lines; footer=nothing)
    first_item = iterate(lines)
    isnothing(first_item) && return nothing
    line, state = first_item
    first_prefix = "┌ " * level * ": "
    rest_prefix = "│ "
    first_width = _boxed_message_width(io, first_prefix)
    rest_width = _boxed_message_width(io, rest_prefix)
    first_wrapped = _wrap_boxed_line(line, first_width)
    println(io, first_prefix, first(first_wrapped))
    for wrapped_line in Base.tail(first_wrapped)
        println(io, rest_prefix, wrapped_line)
    end
    while true
        item = iterate(lines, state)
        isnothing(item) && break
        line, state = item
        for wrapped_line in _wrap_boxed_line(line, rest_width)
            println(io, rest_prefix, wrapped_line)
        end
    end
    return isnothing(footer) ? println(io, "└") : println(io, "└ ", footer)
end

function _print_boxed_block(io::IO, first_prefix::AbstractString, lines; footer=nothing)
    first_item = iterate(lines)
    isnothing(first_item) && return nothing
    line, state = first_item
    rest_prefix = "│ "
    first_width = _boxed_message_width(io, first_prefix)
    rest_width = _boxed_message_width(io, rest_prefix)
    first_wrapped = _wrap_boxed_line(line, first_width)
    println(io, first_prefix, first(first_wrapped))
    for wrapped_line in Base.tail(first_wrapped)
        println(io, rest_prefix, wrapped_line)
    end
    while true
        item = iterate(lines, state)
        isnothing(item) && break
        line, state = item
        for wrapped_line in _wrap_boxed_line(line, rest_width)
            println(io, rest_prefix, wrapped_line)
        end
    end
    return isnothing(footer) ? println(io, "└") : println(io, "└ ", footer)
end

@inline function _boxed_message_width(io::IO, prefix::AbstractString)
    cols = get(io, :displaysize, displaysize(io))[2]
    return max(20, cols - textwidth(prefix))
end

function _wrap_boxed_line(line, width::Int)
    text = string(line)
    isempty(text) && return (text,)
    width < 1 && return (text,)
    textwidth(text) <= width && return (text,)

    wrapped = String[]
    remaining = text
    while textwidth(remaining) > width
        split_idx = nothing
        for idx in eachindex(remaining)
            textwidth(SubString(remaining, 1, idx)) > width && break
            remaining[idx] == ' ' && (split_idx = idx)
        end
        if isnothing(split_idx)
            split_idx = firstindex(remaining)
            for idx in eachindex(remaining)
                textwidth(SubString(remaining, firstindex(remaining), idx)) > width && break
                split_idx = idx
            end
        end
        push!(wrapped, rstrip(SubString(remaining, firstindex(remaining), split_idx)))
        remaining = lstrip(SubString(remaining, nextind(remaining, split_idx)))
        isempty(remaining) && break
    end
    isempty(remaining) || push!(wrapped, remaining)
    return Tuple(wrapped)
end

function _print_boxed_error(io::IO, lines; footer=nothing)
    _print_boxed_block(io, "", lines; footer)
end

function _print_boxed_info(io::IO, lines; footer=nothing)
    _print_boxed_message(io, "Info", lines; footer)
end

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

    tuple_map(f::F, x::Tuple, y::Tuple, z::Tuple) where {F}

Ternary extension of `tuple_map`. Nearly equivalent to `map(f, x, y, z)`.
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

@inline @generated function tuple_map(f::F, x::Tuple, y::Tuple, z::Tuple) where {F}
    if length(x.parameters) != length(y.parameters) ||
        length(x.parameters) != length(z.parameters)
        return :(throw(ArgumentError("x, y, and z must have the same length")))
    else
        stmts = map(
            n -> :(f(getfield(x, $n), getfield(y, $n), getfield(z, $n))), 1:fieldcount(x)
        )
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

@noinline function _gradient_throw_uninit_field_error(::Type{P}, n::Int) where {P}
    throw(
        ArgumentError(
            "Gradient basis traversal encountered an undefined field " *
            "`$(fieldname(P, n))` in a value of type `$P`, but that field is marked " *
            "always-initialised. This object is in a partially initialised state that " *
            "Mooncake cannot traverse automatically.",
        ),
    )
end

#
# Generic primal-value fold/rebuild kernels
#
# These helpers only walk assigned builtin-array entries, ordinary array values,
# tuple/named-tuple values, and defined fields. Callers provide the leaf policy; the
# traversal kernel owns the structural walk, alias handling, and immutable tangent rebuild.
#
# In particular, `_primal_rebuild` is designed around Mooncake's aliasing invariance:
# if two references in the primal alias the same object, then the rebuilt tangents must
# alias the same tangent object too. That is why the rebuild path accepts `seen` and
# memoizes mutable/shared primals while rebuilding their tangent structure.
#
# MWE (fold):
# `total = _fold_tuple_values(
#     (acc, x, _seen) -> acc + x,
#     0,
#     (1, 2, 3),
#     nothing,
# )`
# returns `6` because the helper walks the tuple values and threads the accumulator.
#
# MWE (rebuild):
# `_primal_rebuild(recur, ops, (1.0, 2.0), Ref(0), nothing)` can rebuild a
# tangent-shaped object by keeping the tuple walk in the kernel while `ops.leaf` decides
# what to put at each scalar leaf. Use an `IdDict` in `seen` when rebuilt tangents need
# to preserve primal aliasing.
#
@inline function _fold_builtin_array_assigned(f, init, x::AbstractArray, seen)
    acc = init
    for i in eachindex(x)
        isassigned(x, i) || continue
        acc = f(acc, x[i], seen)
    end
    return acc
end

@inline function _fold_array_values(f, init, x::AbstractArray, seen)
    acc = init
    for xi in x
        acc = f(acc, xi, seen)
    end
    return acc
end

@inline function _fold_tuple_values(f, init, x::Tuple, seen)
    acc = init
    for xi in x
        acc = f(acc, xi, seen)
    end
    return acc
end

@inline function _fold_namedtuple_values(f, init, x::NamedTuple, seen)
    acc = init
    for xi in values(x)
        acc = f(acc, xi, seen)
    end
    return acc
end

@inline function _fold_fields(f, init, x::P, seen) where {P}
    acc = init
    inits = always_initialised(P)
    for n in 1:fieldcount(P)
        if isdefined(x, n)
            acc = f(acc, getfield(x, n), seen)
        elseif inits[n]
            _gradient_throw_uninit_field_error(P, n)
        end
    end
    return acc
end

@inline _make_array_tangent(x, _state, _seen) = zero_tangent(x)
@inline _make_struct_tangent(x, _state, _seen) = zero_tangent(x)
@inline _make_tuple_tangent(::Type, fields, _state, _seen) = fields
@inline _make_namedtuple_tangent(::Type{P}, fields, _state, _seen) where {P<:NamedTuple} = P(
    fields
)
@inline function _store_array_leaf!(leaf, dx, I, xI, state, seen)
    dx[I] = leaf(xI, state, seen)
    return nothing
end
@inline function _store_array_child!(recur, ops, dx, I, xI, state, seen)
    dx[I] = _primal_rebuild(recur, ops, xI, state, seen)
    return nothing
end
@inline function _store_struct_field!(recur, ops, tx, n::Int, x, state, seen)
    set_tangent_field!(tx, n, _primal_rebuild(recur, ops, getfield(x, n), state, seen))
    return nothing
end

# TODO: `zero_tangent` still has its own recursive structural walk. It should eventually
# reuse `_primal_rebuild` with zero-valued leaf hooks so aliasing invariance and
# partially-initialised-field handling live in one traversal kernel.
#
# TODO: `copy_to!!` and similar structural copy/update paths still duplicate shape walks.
# They should eventually reuse `_primal_rebuild` or `_primal_fold` with copy-specific
# hooks, while keeping backend-specific storage policies explicit.
#
# NOTE: `_primal_fold` is for pure structural reductions over the primal shape, like
# `dof` counting. Queries that need tangent allocation or mutation should use
# `_primal_rebuild` instead.
#
# NOTE: the shared traversal kernel does not imply one shared reconstruction policy.
# IRfwd rebuilds Mooncake tangents, while nfwd can rebuild raw seeded tuples or packed
# arrays. Share the walk; keep backend-specific container reconstruction configurable.
@inline _primal_rebuild_ops(; leaf, nodiff, make_array=_make_array_tangent, make_tuple_tangent=_make_tuple_tangent, make_namedtuple_tangent=_make_namedtuple_tangent, store_array_leaf=(_store_array_leaf!), store_array_child=(_store_array_child!), make_struct_tangent=_make_struct_tangent, store_struct_field=(_store_struct_field!)) = (;
    leaf,
    nodiff,
    make_array,
    make_tuple_tangent,
    make_namedtuple_tangent,
    store_array_leaf,
    store_array_child,
    make_struct_tangent,
    store_struct_field,
)

@inline _primal_fold(ops, init, ::CRC.NoTangent, seen) = ops.nodiff(init, NoTangent(), seen)
@inline _primal_fold(ops, init, x::IEEEFloat, seen) = ops.leaf(init, x, seen)
@inline _primal_fold(ops, init, x::Complex{<:IEEEFloat}, seen) = ops.leaf(init, x, seen)
@inline function _primal_fold(ops, init, x::AbstractArray, seen)
    tangent_type(typeof(x)) == NoTangent && return ops.nodiff(init, x, seen)
    if seen isa IdDict{Any,Any}
        haskey(seen, x) && return init
        seen[x] = nothing
    end
    return ops.combine_array(
        (acc, xi, seen) -> _primal_fold(ops, acc, xi, seen), init, x, seen
    )
end
@inline function _primal_fold(ops, init, x::Tuple, seen)
    return ops.combine_tuple(
        (acc, xi, seen) -> _primal_fold(ops, acc, xi, seen), init, x, seen
    )
end
@inline function _primal_fold(ops, init, x::NamedTuple, seen)
    return ops.combine_namedtuple(
        (acc, xi, seen) -> _primal_fold(ops, acc, xi, seen), init, x, seen
    )
end
@inline function _primal_fold(ops, init, x::P, seen) where {P}
    tangent_type(P) == NoTangent && return ops.nodiff(init, x, seen)
    if Base.ismutabletype(P) && seen isa IdDict{Any,Any}
        haskey(seen, x) && return init
        seen[x] = nothing
    end
    return ops.combine_struct(
        (acc, xi, seen) -> _primal_fold(ops, acc, xi, seen), init, x, seen
    )
end

@inline function _primal_input_dof(x, seen)
    # DOF is a pure reduction over the primal shape, so it uses the fold kernel rather than
    # rebuilding tangents.
    ops = (;
        leaf=(total, x, _seen) -> total + (x isa Complex ? 2 : 1),
        nodiff=(total, _x, _seen) -> total,
        combine_array=(f, init, x, seen) -> if x isa _BuiltinArrays
            _fold_builtin_array_assigned(f, init, x, seen)
        else
            _fold_array_values(f, init, x, seen)
        end,
        combine_tuple=_fold_tuple_values,
        combine_namedtuple=_fold_namedtuple_values,
        combine_struct=_fold_fields,
    )
    return _primal_fold(ops, 0, x, seen)
end

@inline _primal_rebuild(recur, ops, ::CRC.NoTangent, state, seen) = ops.nodiff(
    NoTangent(), state, seen
)
@inline _primal_rebuild(recur, ops, x::IEEEFloat, state, seen) = ops.leaf(x, state, seen)
@inline _primal_rebuild(recur, ops, x::Complex{<:IEEEFloat}, state, seen) = ops.leaf(
    x, state, seen
)

@inline function _primal_rebuild(
    recur, ops, x::AbstractArray{T}, state, seen
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    tangent_type(typeof(x)) == NoTangent && return ops.nodiff(x, state, seen)
    if seen isa IdDict{Any,Any}
        existing = get(seen, x, nothing)
        !isnothing(existing) && return existing
    end
    # Scalar/complex leaf arrays are rebuilt elementwise through the leaf policy, letting
    # each backend choose the concrete storage layout.
    dx = ops.make_array(x, state, seen)
    seen isa IdDict{Any,Any} && (seen[x] = dx)
    @inbounds for I in eachindex(x)
        ops.store_array_leaf(ops.leaf, dx, I, x[I], state, seen)
    end
    return dx
end

@inline function _primal_rebuild(recur, ops, x::AbstractArray, state, seen)
    tangent_type(typeof(x)) == NoTangent && return ops.nodiff(x, state, seen)
    if seen isa IdDict{Any,Any}
        existing = get(seen, x, nothing)
        !isnothing(existing) && return existing
    end
    # Non-leaf arrays recurse into their elements, while still memoizing shared mutable
    # primals so rebuilt tangents preserve aliasing invariance.
    dx = ops.make_array(x, state, seen)
    seen isa IdDict{Any,Any} && (seen[x] = dx)
    if x isa _BuiltinArrays
        for I in eachindex(x)
            isassigned(x, I) || continue
            ops.store_array_child(recur, ops, dx, I, x[I], state, seen)
        end
    else
        for I in eachindex(x)
            ops.store_array_child(recur, ops, dx, I, x[I], state, seen)
        end
    end
    return dx
end

@inline function _primal_rebuild(recur, ops, x::P, state, seen) where {P<:Tuple}
    tangent_type(P) == NoTangent && return ops.nodiff(x, state, seen)
    fields = ntuple(n -> _primal_rebuild(recur, ops, x[n], state, seen), Val(fieldcount(P)))
    return ops.make_tuple_tangent(P, fields, state, seen)
end

@inline function _primal_rebuild(recur, ops, x::P, state, seen) where {P<:NamedTuple}
    tangent_type(P) == NoTangent && return ops.nodiff(x, state, seen)
    fields = ntuple(n -> _primal_rebuild(recur, ops, x[n], state, seen), Val(fieldcount(P)))
    return ops.make_namedtuple_tangent(P, fields, state, seen)
end

@inline function _primal_rebuild(recur, ops, x::P, state, seen) where {P}
    tangent_type(P) == NoTangent && return ops.nodiff(x, state, seen)
    if Base.ismutabletype(P)
        if seen isa IdDict{Any,Any}
            existing = get(seen, x, nothing)
            !isnothing(existing) && return existing
        end
        tx = ops.make_struct_tangent(x, state, seen)
        seen isa IdDict{Any,Any} && (seen[x] = tx)
        if tx isa MutableTangent
            inits = always_initialised(P)
            for n in 1:fieldcount(P)
                if isdefined(x, n)
                    ops.store_struct_field(recur, ops, tx, n, x, state, seen)
                elseif inits[n]
                    _gradient_throw_uninit_field_error(P, n)
                end
            end
        end
        return tx
    end

    inits = always_initialised(P)
    fields = ntuple(Val(fieldcount(P))) do n
        if isdefined(x, n)
            _primal_rebuild(recur, ops, getfield(x, n), state, seen)
        elseif inits[n]
            _gradient_throw_uninit_field_error(P, n)
        else
            PossiblyUninitTangent{tangent_type(fieldtype(P, n))}()
        end
    end
    return build_tangent(P, fields...)
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
    lookup_method(sig)::Union{Method,Nothing}

Returns the first `Method` matching `sig`, or `nothing` if no match is found. Unlike
`is_vararg_and_sparam_names`, this does not throw if there are zero or multiple matches.
"""
function lookup_method(sig)::Union{Method,Nothing}
    world = Base.get_world_counter()
    min = Base.RefValue{UInt}(typemin(UInt))
    max = Base.RefValue{UInt}(typemax(UInt))
    ms = Base._methods_by_ftype(sig, nothing, 1, world, true, min, max, Ptr{Int32}(C_NULL))
    return (ms isa Vector && !isempty(ms)) ? first(ms).method : nothing
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
    ir.argtypes[1] = _typeof(env)
    nargtypes = length(ir.argtypes)
    nargs = nargtypes - 1
    sig = compute_oc_signature(ir, nargs, isva)
    src = ccall(:jl_new_code_info_uninit, Ref{CC.CodeInfo}, ())
    src.slotnames = [Symbol(:_, i) for i in 1:nargtypes]
    src.slotflags = fill(zero(UInt8), length(ir.argtypes))
    src.slottypes = copy(ir.argtypes)
    @static if VERSION > v"1.12-"
        ir.debuginfo.def === nothing &&
            (ir.debuginfo.def = :var"generated IR for OpaqueClosure")
        src.min_world = ir.valid_worlds.min_world
        src.max_world = ir.valid_worlds.max_world
        src.isva = isva
        src.nargs = nargtypes
    end
    src = CC.ir_to_codeinf!(src, ir)
    src.rettype = ret_type
    oc = Base.Experimental.generate_opaque_closure(
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

@unstable begin
    @static if VERSION > v"1.12-"
        compute_ir_rettype(ir) = CC.compute_ir_rettype(ir)
        compute_oc_signature(ir, nargs, isva) = CC.compute_oc_signature(ir, nargs, isva)
    else
        compute_ir_rettype(ir) = Base.Experimental.compute_ir_rettype(ir)
        compute_oc_signature(ir, nargs, isva) = Base.Experimental.compute_oc_signature(
            ir, nargs, isva
        )
    end
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

!!! warning
    This is an internal utility, not part of the public `Mooncake.jl` API, and may change
    without notice. Its behaviour can vary across data types, so the description below
    should be treated as guidance, not a strict contract.

Utility for copying AD-related data structures (e.g. rules, caches, and other internals).

# Examples of current behaviour

Currently, `_copy` has the following behaviours for specific types:

- `CoDual`, `Dual` types → no copying needed  
- `Stack` types → create a new empty instance  
- Composite types (e.g. `Tuple`, `NamedTuple`) → recursively copy elements  
- Misty closure → construct a new Misty closure with deep copy of captured data  
- Tangent types (`PossiblyUninitTangent`) → copy conditionally based on initialisation state  
- Forward/reverse data types (e.g. `FData`, `RData`, `LazyZeroRData`) → recursively copy wrapped data  
- `RRuleZeroWrapper` → recursively copy the wrapped rule into a new instance
- `DerivedRule` → construct new instances with copied captures and caches  
- `LazyFRule`, `LazyDerivedRule` → construct new lazy rules with the same method instance and debug mode  
- `DynamicFRule`, `DynamicDerivedRule` → construct new dynamic rules with an empty cache and the same debug mode  
"""

# Generic fallback to Base.copy
_copy(x) = copy(x)

_copy(::Nothing) = nothing
_copy(x::Symbol) = x
_copy(x::Tuple) = map(_copy, x)
_copy(x::NamedTuple) = map(_copy, x)
_copy(x::Ref{T}) where {T} = isassigned(x) ? Ref{T}(_copy(x[])) : Ref{T}()
_copy(x::Type) = x

# TODO: remove the Julia < 1.11 branch (and the corresponding @static VERSION guards in
# test_utils.jl) once https://github.com/JuliaLang/julia/issues/61368 is fixed and
# Julia 1.10 support is dropped.
#
# Julia 1.10 codegen bug (julia#61368 / #51016): jl_compile_workqueue crashes in
# emit_specsig_oc_call when compiling an OC body that transitively calls another OC
# type whose CodeInstance has been invalidated (null specfun) by a world-counter advance
# (e.g. from loading DispatchDoctor or defining new methods).
#
# Fix: __call_rule type-erases rule via Base.inferencebarrier on Julia 1.10 so the
# compiled code calls through jl_apply_generic (which never invokes emit_specsig_oc_call).
# The @noinline wrapper ensures this erased call is a separate compilation unit. On Julia
# 1.11+ the bug is absent and __call_rule is a direct call. The type-erasure causes
# isbits argument boxing at each nested rule callsite (jl_apply_generic requires boxed
# args), so zero-allocation performance checks are skipped on Julia < 1.11 in test_utils.
#
# This generic fallback returns Any. Specialised overloads restore type stability:
#   - DerivedFRule, DerivedRule (forward_mode.jl, reverse_mode.jl): type assertion via params
#   - DebugFRule, DebugRRule (debug_mode.jl): direct call (no OC, no barrier needed)
#   - typeof(rrule!!) (interface.jl): direct call (plain function, no OC)
# Dynamic rules (DynamicFRule, DynamicDerivedRule) cannot be handled statically and remain
# unstable on Julia 1.10.
#
# Used by LazyFRule, DynamicFRule, LazyDerivedRule, DynamicDerivedRule, RRuleZeroWrapper,
# DebugFRule, and DebugRRule.
#
# Approximate Mooncake-free MWE (Julia 1.10):
#   dep(x::Float64) = x * 2.0
#   inner = Base.Experimental.@opaque (x::Float64) -> dep(x)
#   inner(1.0)                    # compile inner at world W₀
#   dep(x::Float64) = x * 3.0    # redefine dep → invalidates inner's CodeInstance
#   struct Wrapper{T}; oc::T; end
#   (w::Wrapper)(x) = w.oc(x)    # call chain exposes typeof(inner) to the workqueue
#   w = Wrapper(inner)            # concrete Wrapper{typeof(inner)}
#   Base.Experimental.@opaque (x::Float64) -> w(x)  # crash: workqueue compiles
#                                 # Wrapper::call, reaches typeof(inner) with null specfun
@static if VERSION < v"1.11-"
    @noinline __call_rule_erased!(rule, args) = rule(args...)
    @inline __call_rule(rule, args) = __call_rule_erased!(Base.inferencebarrier(rule), args)

    # TODO: if all internal OpaqueClosure / MistyClosure callsites are routed through
    # `__call_opaque_closure`, this helper may be able to subsume the OpaqueClosure-specific
    # part of `__call_rule`.
    @noinline function __call_opaque_closure_erased!(oc::OpaqueClosure, args)
        return Core._apply_iterate(iterate, oc, args)
    end
    @inline function __call_opaque_closure(oc::OpaqueClosure, args)
        return __call_opaque_closure_erased!(
            Base.inferencebarrier(oc), Base.inferencebarrier(args)
        )
    end
else
    @inline __call_rule(rule, args) = rule(args...)
    @inline __call_opaque_closure(oc::OpaqueClosure, args) = Core._apply_iterate(
        iterate, oc, args
    )
end

@inline __call_opaque_closure(mc::MistyClosure, args) = __call_opaque_closure(mc.oc, args)
