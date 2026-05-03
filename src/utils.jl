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
    return _print_boxed_block(io, "", lines; footer)
end

function _print_boxed_info(io::IO, lines; footer=nothing)
    return _print_boxed_message(io, "Info", lines; footer)
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
    return oc = Base.Experimental.generate_opaque_closure(
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
        function compute_oc_signature(ir, nargs, isva)
            return Base.Experimental.compute_oc_signature(ir, nargs, isva)
        end
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
- `LazyPrimal`, `LazyDerivedRule` → construct new lazy rules with the same method instance and debug mode  
- `DynamicPrimal`, `DynamicDerivedRule` → construct new dynamic rules with an empty cache and the same debug mode  
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
#   - DerivedPrimal, DerivedRule (primal_mode.jl, reverse_mode.jl): type assertion via params
#   - DebugFRule, DebugRRule (debug_mode.jl): direct call (no OC, no barrier needed)
#   - typeof(rrule!!) (interface.jl): direct call (plain function, no OC)
# Dynamic rules (DynamicPrimal, DynamicDerivedRule) cannot be handled statically and remain
# unstable on Julia 1.10.
#
# Used by LazyPrimal, DynamicPrimal, LazyDerivedRule, DynamicDerivedRule, RRuleZeroWrapper,
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
else
    @inline __call_rule(rule, args) = rule(args...)
end

# ── Scalar slot traversal ────────────────────────────────────────────────────
#
# Left-fold and map-like rebuild over the scalar degrees of freedom of a value.
# These generalise Nfwd._nfwd_fold_slots / _nfwd_unfold_slots to NamedTuples,
# generic AbstractArrays, and arbitrary struct types.
#
# Aliasing: when `state` is an `IdDict{Any,Any}`, mutable containers (arrays,
# mutable structs) are tracked — revisits are skipped (_fold_slots) or return
# their cached rebuild (_unfold_slots).

# ── _fold_slots ──────────────────────────────────────────────────────────────
#
#     _fold_slots(f, init, x, state) -> (acc, state)
#
# Callback signature: f(acc, x_container, slot_index, state) -> (acc, state).

@inline function _fold_slots(f::F, init, x::IEEEFloat, state) where {F}
    return f(init, x, 1, state)
end

@inline function _fold_slots(f::F, init, x::Complex{<:IEEEFloat}, state) where {F}
    acc, state = f(init, x, 1, state)
    return f(acc, x, 2, state)
end

@inline function _fold_slots(f::F, init, x::AbstractArray{T}, state) where {F,T<:IEEEFloat}
    if state isa IdDict{Any,Any}
        haskey(state, x) && return (init, state)
        state[x] = nothing
    end
    acc = init
    @inbounds for i in eachindex(x)
        acc, state = f(acc, x, i, state)
    end
    return acc, state
end

@inline function _fold_slots(
    f::F, init, x::AbstractArray{Complex{T}}, state
) where {F,T<:IEEEFloat}
    if state isa IdDict{Any,Any}
        haskey(state, x) && return (init, state)
        state[x] = nothing
    end
    acc = init
    @inbounds for i in eachindex(x)
        acc, state = f(acc, x, 2i - 1, state)
        acc, state = f(acc, x, 2i, state)
    end
    return acc, state
end

@inline _fold_slots(::F, init, ::Tuple{}, state) where {F} = (init, state)
@inline function _fold_slots(f::F, init, x::Tuple, state) where {F}
    acc, state = _fold_slots(f, init, first(x), state)
    return _fold_slots(f, acc, Base.tail(x), state)
end

@inline _fold_slots(f::F, init, x::NamedTuple, state) where {F} =
    _fold_slots(f, init, values(x), state)

@inline function _fold_slots(f::F, init, x::AbstractArray, state) where {F}
    tangent_type(typeof(x)) == NoTangent && return (init, state)
    if state isa IdDict{Any,Any}
        haskey(state, x) && return (init, state)
        state[x] = nothing
    end
    acc = init
    if x isa Array
        for i in eachindex(x)
            isassigned(x, i) || continue
            acc, state = _fold_slots(f, acc, x[i], state)
        end
    else
        for xi in x
            acc, state = _fold_slots(f, acc, xi, state)
        end
    end
    return (acc, state)
end

@inline function _fold_slots(f::F, init, x::P, state) where {F,P}
    tangent_type(P) == NoTangent && return (init, state)
    if Base.ismutabletype(P) && state isa IdDict{Any,Any}
        haskey(state, x) && return (init, state)
        state[x] = nothing
    end
    acc = init
    inits = always_initialised(P)
    for n in 1:fieldcount(P)
        if isdefined(x, n)
            acc, state = _fold_slots(f, acc, getfield(x, n), state)
        elseif inits[n]
            throw(
                ArgumentError(
                    "Encountered undefined field `$(fieldname(P, n))` in `$P` " *
                    "that is marked always-initialised.",
                ),
            )
        end
    end
    return (acc, state)
end

# ── _unfold_slots ────────────────────────────────────────────────────────────
#
#     _unfold_slots(f, x, state) -> (rebuilt, state)
#
# Callback signature: f(x_leaf, state) -> (result, state).

@inline function _unfold_slots(
    f::F, x::Union{IEEEFloat,Complex{<:IEEEFloat}}, state
) where {F}
    return f(x, state)
end

@inline function _unfold_slots(
    f::F, x::Union{AbstractArray{<:IEEEFloat},AbstractArray{<:Complex{<:IEEEFloat}}}, state
) where {F}
    if state isa IdDict{Any,Any}
        haskey(state, x) && return state[x], state
    end
    result, state = f(x, state)
    if state isa IdDict{Any,Any}
        state[x] = result
    end
    return result, state
end

@inline _unfold_slots(::F, ::Tuple{}, state) where {F} = ((), state)
@inline function _unfold_slots(f::F, x::Tuple, state) where {F}
    head, state = _unfold_slots(f, first(x), state)
    tail, state = _unfold_slots(f, Base.tail(x), state)
    return (head, tail...), state
end

@inline function _unfold_slots(f::F, x::NamedTuple{names}, state) where {F,names}
    vals, state = _unfold_slots(f, values(x), state)
    return NamedTuple{names}(vals), state
end

# ── _count_slots ──────────────────────────────────────────────────────────────

@inline _count_slots(x::IEEEFloat) = 1
@inline _count_slots(x::Complex{<:IEEEFloat}) = 2
@inline _count_slots(x::AbstractArray{<:IEEEFloat}) = length(x)
@inline _count_slots(x::AbstractArray{<:Complex{<:IEEEFloat}}) = 2 * length(x)
# Tuples/NamedTuples are by definition acyclic, so recurse without an IdDict —
# the generic fallback below allocates one per call.
@inline _count_slots(::Tuple{}) = 0
@inline _count_slots(x::Tuple) = _count_slots(first(x)) + _count_slots(Base.tail(x))
@inline _count_slots(x::NamedTuple) = _count_slots(values(x))
@inline _count_slots(x) =
    first(_fold_slots((acc, _, _, s) -> (acc + 1, s), 0, x, IdDict{Any,Any}()))
