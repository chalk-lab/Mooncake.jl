# Writing Custom Tangent Types

```@meta
CurrentModule = Mooncake
```

Mooncake.jl associates each **primal type** (the original data structure) with a unique **tangent type** (the type that stores its derivative information). By default, Mooncake can automatically derive tangent types for most Julia structs. However, for *recursive types*—that is, types that reference themselves (directly or indirectly)—the default mechanism can fail, often resulting in a stack overflow. In such cases, you must manually define a custom tangent type and implement the required interface.

This guide walks you through the process, from understanding Mooncake’s tangent design to testing your custom tangent type.

## 1. Tangent Types and the FData/RData Split

Before diving in, let's review how Mooncake represents tangents (gradients) and why it splits them into **forward data** (`fdata`) and **reverse data** (`rdata`). For more details, see the [Mooncake.jl Rule System documentation](https://chalk-lab.github.io/Mooncake.jl/stable/understanding_mooncake/rule_system/).

### Tangent Types

For a given primal type `P`, `Mooncake.tangent_type(P)` returns the tangent type associated with `P`. By default, Mooncake uses generic `Tangent{...}` structs to hold fieldwise derivatives. For example, a simple struct’s tangent might be `Tangent{NamedTuple}` with the same field names as the primal. Mutable structs get a `MutableTangent` type. Each field’s tangent is itself of type `tangent_type(field_type)`.

### Forward Data vs. Reverse Data

Mooncake splits a tangent object into two parts:

- **fdata**: Forward-pass data, typically components identified by address (e.g., arrays or mutable fields), which are carried along and updated in-place.
- **rdata**: Reverse-pass data, typically value-identified components (e.g., plain numbers), only needed for the reverse pass.

This design improves performance by minimizing what needs to be propagated during the forward pass.

**Example:**  
Consider `Tuple{Float64, Vector{Float64}, Int}`. Its tangent type is `Tuple{Float64, Vector{Float64}, NoTangent}` (since `Int` is non-differentiable). The `fdata` type is `Tuple{NoFData, Vector{Float64}, NoFData}`—only the vector is forwarded. The `rdata` type is `Tuple{Float64, NoRData, NoRData}`—only the float’s derivative is carried in reverse. Mooncake ensures that for any tangent `t`, if `f = Mooncake.fdata(t)` and `r = Mooncake.rdata(t)`, then `Mooncake.tangent(f, r)` reconstructs the original `t`.

## 2. Why Recursive Types Are Challenging

A *recursive type* is a struct that contains itself (directly or indirectly) as a field. For example:

```@setup custom_tangent_type
using Mooncake: Mooncake
using DifferentiationInterface
using JET
using AllocCheck
using Test
using Random
```

```@example custom_tangent_type
mutable struct A{T}
    x::T
    a::Union{A{T},Nothing}

    A(x::T) where {T} = new{T}(x, nothing)
    A(x::T, child::A{T}) where {T} = new{T}(x, child)
end
```

Here, `A` has a self-referential field `a`. If you ask Mooncake for the tangent type of `A{Float64}`, it tries to construct something like `Tangent{Tuple{Float64, tangent_type(A)}}`, which leads to infinite recursion. Calling `tangent_type(A)` in this scenario will overflow the stack.

To solve this, you must manually define a custom tangent type that breaks this circular dependency.

## 3. Defining a Custom Tangent Type for Recursion

The first step is to define a new type to represent the tangent of `A`. This custom tangent should mimic the structure of `A`, but in a way that resolves the recursion:

```@example custom_tangent_type
mutable struct TangentForA{Tx}
    x::Tx
    a::Union{TangentForA{Tx}, Mooncake.NoTangent}

    function TangentForA{Tx}(x_tangent::Tx) where {Tx}
        new{Tx}(x_tangent, Mooncake.NoTangent())
    end

    function TangentForA{Tx}(x_tangent::Tx, a_tangent::Union{TangentForA{Tx}, Mooncake.NoTangent}) where {Tx}
        new{Tx}(x_tangent, a_tangent)
    end

    # This constructor is required by Mooncake's internal machinery for constructing tangents from named tuples
    function TangentForA{Tx}(nt::@NamedTuple{x::Tx, a::Union{Mooncake.NoTangent, TangentForA{Tx}}}) where {Tx}
        return new{Tx}(nt.x, nt.a)
    end
end
```

This `TangentForA` type mirrors `A`'s fields. Its `a` field is either another `TangentForA` (for nested or cyclic primal structures) or [`Mooncake.NoTangent`](@ref) (if the primal `A.a` is `nothing`). This explicit definition breaks the infinite type recursion that would occur with naive tangent derivation.

## 4. Registering Your Tangent Type with Mooncake

Defining the tangent type is not enough—you must **register it with Mooncake’s interface** so Mooncake knows to use it and how to split it into [`fdata`](@ref)/[`rdata`](@ref). Implement these methods:

### 4.1. `tangent_type`

Tell Mooncake that the tangent of `A` is `TangentForA`:

```@example custom_tangent_type
function Mooncake.tangent_type(::Type{A{T}}) where {T}
    Tx = Mooncake.tangent_type(T)
    return Tx == Mooncake.NoTangent ? Mooncake.NoTangent : TangentForA{Tx}
end
```

This overrides the default mechanism and associates `A` with your custom tangent type.

### 4.2. `fdata_type` and `rdata_type`

Define the types of forward and reverse data for `TangentForA`. In this example, since both `A` and `TangentForA` are mutable, all updates can be done in-place, so the `fdata` is the tangent itself and `rdata` is [`NoRData`](@ref). We shouldn't need to specifically define `fdata_type` and `rdata_type` because Mooncake can infer them in this case. In other cases, you may need to split these more carefully and define them explicitly.

### 4.3. `tangent` (Combining Function)

Mooncake provides [`Mooncake.tangent`](@ref) to reassemble a tangent from `fdata` and `rdata`. For your type:

```@example custom_tangent_type
Mooncake.tangent(t::TangentForA{Tx}, ::Mooncake.NoRData) where {Tx} = t
```

This ensures that `Mooncake.tangent(Mooncake.fdata(t), Mooncake.rdata(t)) === t`, which is a core requirement of Mooncake's tangent interface (see [`fdata_type`](@ref)). Mooncake’s tests will check that the reassembled tangent is the exact same object as the original.

With these methods, your custom type is now connected to Mooncake’s AD system.

## 5. Bottom-Up Integration: Implement Only What You Need

Mooncake provides extensive coverage and thorough testing. To get started, you can implement just enough to differentiate simple functions and add more as needed. For example, consider:

```@example custom_tangent_type
f1(a::A) = 2.0 * a.x
```

When you try to differentiate this, Mooncake will complain it lacks an `rrule!!` for [`lgetfield`](@ref). The `lgetfield` function is Mooncake's internal version of `getfield` that accepts a `Val` type for the field name, enabling better type stability. You need to implement it:

### 5.1. Field Access (`lgetfield`) Rule

```@example custom_tangent_type
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(Mooncake.lgetfield),A{T},Val} where {T}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    field_name_cd::Mooncake.CoDual{Val{FieldName}},
) where {T,Tx,FieldName}
    a = Mooncake.primal(obj_cd)
    a_tangent = Mooncake.tangent(obj_cd)

    value_primal = getfield(a, FieldName)
    actual_field_tangent_value = FieldName === :x ? a_tangent.x :
                                FieldName === :a ? a_tangent.a :
                                throw(ArgumentError("lgetfield: Unknown field '$FieldName' for type A."))

    value_output_fdata = Mooncake.fdata(actual_field_tangent_value)
    y_cd = Mooncake.CoDual(value_primal, value_output_fdata)

    function lgetfield_A_pullback(Δy_rdata)
        if FieldName === :x
            if !(Δy_rdata isa Mooncake.NoRData)
                a_tangent.x = Mooncake.increment_rdata!!(a_tangent.x, Δy_rdata)
            end
        elseif FieldName === :a
            @assert Δy_rdata isa Mooncake.NoRData  # for mutable TangentForA, rdata is not used
        end
        return (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData())
    end
    return y_cd, lgetfield_A_pullback
end
```

### 5.2. Zeroing Out the Tangent

Mooncake will next require [`set_to_zero!!`](@ref):

```@example custom_tangent_type
function Mooncake.set_to_zero!!(t::TangentForA{Tx}) where Tx
    t.x = Mooncake.set_to_zero!!(t.x)
    if !(t.a isa Mooncake.NoTangent)
        Mooncake.set_to_zero!!(t.a)
    end
    return t
end
```

With these, you can now differentiate simple functions:

```@example custom_tangent_type
a = A(1.0)
val, grad = DifferentiationInterface.value_and_gradient(f1, AutoMooncake(; config=nothing), a)
```

Another example:

```@example custom_tangent_type
function prod_x(a::A{T}) where {T}
    a_val = a.x
    return a.a === nothing ? a_val : a_val * prod_x(a.a)
end
sum_a = A(1.0, A(2.0, A(3.0)))
val_f5, grad_f5 = DifferentiationInterface.value_and_gradient(prod_x, AutoMooncake(; config=nothing), sum_a)
```

Depending on your use case, this may be sufficient.

## 6. From "It Works!" to Passing [`TestUtils.test_data`](@ref)

To fully integrate with Mooncake, you must implement additional operations on your tangent type so Mooncake’s algorithms can manipulate it robustly. At minimum, Mooncake expects the following functions for any custom tangent type:

### Checklist: Functions Needed for Recursive Struct Support

Below is a checklist of most functions you need to make [`Mooncake.TestUtils.test_data`](@ref) pass for the recursive struct `A` and its tangent `TangentForA`. They are grouped by their role in Mooncake’s test suite.

#### Primitive rrules (Mandatory Differentiation Hooks)

You must provide adjoints for every `getfield`/`lgetfield` variant that appears in tests.

| Primitive            | Variants to implement                                                      |
| -------------------- | -------------------------------------------------------------------------- |
| [`lgetfield`](@ref)  | `(A, Val{:x})`, `(A, Val{:a})`, plus Symbol, Int, and (Val, Val) fallbacks |
| `Base.getfield`      | Same coverage as `lgetfield`                                               |
| [`_new_`](@ref)      | `A(x)`, `A(x, a::A)`, `A(x, nothing)`—three separate `rrule!!` methods     |
| [`lsetfield!`](@ref) | `(A, Val{:field}, new_value)` including both Symbol & Int field IDs        |

#### Core Tangent Operations

| Function                          | Purpose/feature tested                                           |
| --------------------------------- | ---------------------------------------------------------------- |
| [`zero_tangent_internal`](@ref)   | Structure-preserving zero generation with cycle cache            |
| [`randn_tangent_internal`](@ref)  | Random tangent generator (for stochastic interface tests)        |
| [`set_to_zero_internal!!`](@ref)  | Recursive in-place reset with cycle protection                   |
| [`increment_internal!!`](@ref)    | In-place accumulation used in reverse pass                       |
| [`_add_to_primal_internal`](@ref) | Adds a tangent to a primal (needed for finite-difference checks) |
| [`_diff_internal`](@ref)          | Structural diff between two primals → tangent                    |
| [`_dot_internal`](@ref)           | Inner-product between tangents (dual-number consistency)         |
| [`_scale_internal`](@ref)         | Scalar × tangent scaling                                         |

#### Test Utilities

| Override                                                                                                       | What it proves                                             |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [`TestUtils.populate_address_map_internal`](@ref)                                                              | Tangent-to-primal pointer correspondence (cycle safety)    |
| `TestUtils.has_equal_data_internal` (internal version of [`TestUtils.has_equal_data`](@ref) (primal & tangent) | Deep equality ignoring pointer identity; handles recursion |

By following this process—starting with a minimal set of methods and expanding as Mooncake requests more—you can support recursive types robustly in Mooncake.jl.

## Appendix: Full Implementations

Before defining the full implementation, [`TestUtils.test_data`](@ref) will fail.

```@example custom_tangent_type
try
    Mooncake.TestUtils.test_data(Random.default_rng(), A(1.0, A(2.0, A(3.0))))
catch e
    @show e
end
```

### Basic tangent interface methods

First, implement the core tangent interface methods:

```@example custom_tangent_type
Mooncake.rdata(::TangentForA{Tx}) where {Tx} = Mooncake.NoRData()
Mooncake.tangent(t::TangentForA{Tx}, ::Mooncake.NoRData) where {Tx} = t

function Mooncake.tangent_type(::Type{TangentForA{Tx}}) where {Tx}
    return TangentForA{Tx}
end
Mooncake.tangent_type(::Type{TangentForA{Tx}}, ::Type{Mooncake.NoRData}) where {Tx} = TangentForA{Tx}
```

### Field access helper functions

Define utility functions for field access:

```@example custom_tangent_type

_field_symbol(f::Symbol) = f
_field_symbol(i::Int) = i == 1 ? :x : i == 2 ? :a :
    throw(ArgumentError("Invalid field index '$i' for type A."))
_field_symbol(::Type{Val{F}}) where F = _field_symbol(F)
_field_symbol(::Val{F}) where F = _field_symbol(F)
```

### Common getfield rule implementation

Define a shared helper for getfield rules:

```@example custom_tangent_type
function _rrule_getfield_common(obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
                                field_sym::Symbol,
                                n_args::Int) where {T,Tx}
    a = Mooncake.primal(obj_cd)
    a_t = Mooncake.tangent(obj_cd)

    value_primal = getfield(a, field_sym)

    field_tan = field_sym === :x ? a_t.x : field_sym === :a ? a_t.a :
        throw(ArgumentError("Unknown field '$field_sym' for type A."))

    y_cd = Mooncake.CoDual(value_primal, Mooncake.fdata(field_tan))

    function pb(Δy_rdata)
        if field_sym === :x
            if !(Δy_rdata isa Mooncake.NoRData)
                a_t.x = Mooncake.increment_rdata!!(a_t.x, Δy_rdata)
            end
        else
            @assert Δy_rdata isa Mooncake.NoRData
        end
        return ntuple(_ -> Mooncake.NoRData(), n_args)
    end

    return y_cd, pb
end
```

### lgetfield and getfield rules

Implement the various field access rules:

```@example custom_tangent_type
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(Mooncake.lgetfield),A{T},Val{S}} where {T,S<:Symbol}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lgetfield),Mooncake.NoFData},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    ::Mooncake.CoDual{Val{FieldName},Mooncake.NoFData},
) where {T,Tx,FieldName}
    field_symbol = _field_symbol(FieldName)
    return _rrule_getfield_common(obj_cd, field_symbol, 3)
end

# Rule for lgetfield(A, Val{Field}, Val{Order})
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(Mooncake.lgetfield),A{T},Val,Val} where {T}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lgetfield),F},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    ::Mooncake.CoDual{Val{VFieldName},Mooncake.NoFData},
    ::Mooncake.CoDual{Val{VOrderName},Mooncake.NoFData}
) where {F,T,Tx,VFieldName,VOrderName}
    field_symbol = _field_symbol(VFieldName)
    return _rrule_getfield_common(obj_cd, field_symbol, 4)
end

# Rule for getfield(A, ::Symbol)
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(getfield),A{T},Symbol} where {T}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(getfield)},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    field_name_symbol_cd::Mooncake.CoDual{Symbol,Mooncake.NoFData},
) where {T,Tx}
    field_sym = Mooncake.primal(field_name_symbol_cd)
    return _rrule_getfield_common(obj_cd, field_sym, 3)
end

# Rule for getfield(A, ::Int)
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(getfield),A{T},Int} where {T}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(getfield)},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    field_idx_cd::Mooncake.CoDual{Int,Mooncake.NoFData},
) where {T,Tx}
    field_sym = _field_symbol(Mooncake.primal(field_idx_cd))
    return _rrule_getfield_common(obj_cd, field_sym, 3)
end

# Rule for getfield(A, ::Symbol, ::Symbol) e.g. getfield(obj, :field, :not_atomic)
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(getfield),A{T},Symbol,Symbol} where {T}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(getfield)},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    field_name_symbol_cd::Mooncake.CoDual{Symbol,Mooncake.NoFData},
    ::Mooncake.CoDual{Symbol,Mooncake.NoFData}
) where {T,Tx}
    field_sym = Mooncake.primal(field_name_symbol_cd)
    return _rrule_getfield_common(obj_cd, field_sym, 4)
end

# Rule for getfield(A, ::Int, ::Symbol) e.g. getfield(obj, 1, :not_atomic)
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(getfield),A{T},Int,Symbol} where {T}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(getfield)},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    field_idx_cd::Mooncake.CoDual{Int,Mooncake.NoFData},
    ::Mooncake.CoDual{Symbol,Mooncake.NoFData}
) where {T,Tx}
    field_sym = _field_symbol(Mooncake.primal(field_idx_cd))
    return _rrule_getfield_common(obj_cd, field_sym, 4)
end
```

### Core tangent operations

Implement the essential tangent manipulation functions:

```@example custom_tangent_type
function Mooncake.zero_tangent_internal(p::A{T}, dict::Mooncake.MaybeCache) where {T}
    Tx = Mooncake.tangent_type(T)
    Tx == Mooncake.NoTangent && return Mooncake.NoTangent()
    if haskey(dict, p)
        return dict[p]::TangentForA{Tx}
    end
    x_t = Mooncake.zero_tangent_internal(p.x, dict)::Tx
    t = TangentForA{Tx}(x_t)
    dict[p] = t
    if p.a === nothing
        t.a = Mooncake.NoTangent()
    else
        t.a = Mooncake.zero_tangent_internal(p.a, dict)::Union{TangentForA{Tx},Mooncake.NoTangent}
    end
    return t
end

function Mooncake.randn_tangent_internal(rng::AbstractRNG, p::A{T}, dict::Mooncake.MaybeCache) where {T}
    Tx = Mooncake.tangent_type(T)
    Tx == Mooncake.NoTangent && return Mooncake.NoTangent()
    if haskey(dict, p)
        return dict[p]::TangentForA{Tx}
    end
    x_t = Mooncake.randn_tangent_internal(rng, p.x, dict)::Tx
    t = TangentForA{Tx}(x_t)
    dict[p] = t
    if p.a === nothing
        t.a = Mooncake.NoTangent()
    else
        t.a = Mooncake.randn_tangent_internal(rng, p.a, dict)::Union{TangentForA{Tx},Mooncake.NoTangent}
    end
    return t
end

function Mooncake.increment_internal!!(c::Mooncake.IncCache, t::TangentForA{Tx}, s::TangentForA{Tx}) where {Tx}
    (haskey(c, t) || t === s) && return t
    c[t] = true
    t.x = Mooncake.increment_internal!!(c, t.x, s.x)
    if !(t.a isa Mooncake.NoTangent)
        t.a = Mooncake.increment_internal!!(c, t.a, s.a)
    end
    return t
end

function Mooncake.set_to_zero_internal!!(c::Mooncake.SetToZeroCache, t::TangentForA{Tx}) where {Tx}
    Mooncake._already_tracked!(c, t) && return t
    t.x = Mooncake.set_to_zero_internal!!(c, t.x)
    if !(t.a isa Mooncake.NoTangent)
        t.a = Mooncake.set_to_zero_internal!!(c, t.a)
    end
    return t
end

function Mooncake._add_to_primal_internal(c::Mooncake.MaybeCache, p::A{T}, t::TangentForA{Tx}, unsafe::Bool) where {T,Tx}
    key = (p, t, unsafe)
    haskey(c, key) && return c[key]::A{T}
    x_new = Mooncake._add_to_primal_internal(c, p.x, t.x, unsafe)
    a_new = p.a === nothing ? nothing : Mooncake._add_to_primal_internal(c, p.a, t.a, unsafe)
    p_new = a_new === nothing ? A(x_new) : A(x_new, a_new)
    c[key] = p_new
    return p_new
end

function Mooncake._diff_internal(c::Mooncake.MaybeCache, p::A{T}, q::A{T}) where {T}
    key = (p, q)
    haskey(c, key) && return c[key]::Union{TangentForA{Mooncake.tangent_type(T)},Mooncake.NoTangent}
    Tx = Mooncake.tangent_type(T)
    if Tx == Mooncake.NoTangent
        t = Mooncake.NoTangent()
        c[key] = t
        return t
    end
    x_t = Mooncake._diff_internal(c, p.x, q.x)
    a_t = if p.a === nothing
        Mooncake.NoTangent()
    else
        Mooncake._diff_internal(c, p.a, q.a)
    end
    t = TangentForA{Tx}(x_t, a_t)
    c[key] = t
    return t
end

function Mooncake._dot_internal(c::Mooncake.MaybeCache, t::TangentForA{Tx}, s::TangentForA{Tx}) where {Tx}
    key = (t, s)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    res = Mooncake._dot_internal(c, t.x, s.x)
    if !(t.a isa Mooncake.NoTangent)
        res += Mooncake._dot_internal(c, t.a, s.a)
    end
    c[key] = res
    return res
end

function Mooncake._scale_internal(c::Mooncake.MaybeCache, a::Float64, t::TangentForA{Tx}) where {Tx}
    haskey(c, t) && return c[t]::TangentForA{Tx}
    x_new = Mooncake._scale_internal(c, a, t.x)
    a_new = t.a isa Mooncake.NoTangent ? Mooncake.NoTangent() : Mooncake._scale_internal(c, a, t.a)
    t_new = TangentForA{Tx}(x_new, a_new)
    c[t] = t_new
    return t_new
end

@inline function Mooncake.get_tangent_field(t::TangentForA, f)
    if f === :x
        return t.x
    elseif f === :a
        return t.a
    else
        throw(error("Unhandled field $f"))
    end
end

Mooncake.__verify_fdata_value(::IdDict{Any,Nothing}, ::A{T}, ::TangentForA{Tx}) where {T,Tx} = nothing
```

### Constructor rules

Implement rrules for the A constructors:

```@example custom_tangent_type
# rrule for A(x::T)
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(Mooncake._new_),Type{A{T}},T} where {T}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake._new_)},
    ::Mooncake.CoDual{Type{A{T}}},
    x_cd::Mooncake.CoDual{T},
) where {T}
    primal_x = Mooncake.primal(x_cd)
    y_primal = A(primal_x)

    Tx_for_field = Mooncake.tangent_type(T)

    y_fdata = if Tx_for_field == Mooncake.NoTangent
        Mooncake.NoTangent()
    else
        raw_x_tan = Mooncake.tangent(x_cd)
        processed_x_tan = if (raw_x_tan isa Mooncake.NoTangent) || (raw_x_tan isa Mooncake.NoFData)
            Mooncake.zero_tangent(primal_x)::Tx_for_field
        else
            raw_x_tan
        end
        TangentForA{Tx_for_field}(processed_x_tan)
    end

    y_cd = Mooncake.CoDual(y_primal, y_fdata)

    function _new_A_x_pullback(Δy_rdata)
        # For scalar types, return the appropriate zero value
        if T <: AbstractFloat || T <: Integer
            return (Mooncake.NoRData(), Mooncake.NoRData(), zero(T))
        else
            x_tangent_val = Mooncake.tangent(x_cd)
            rdata_for_x = (x_tangent_val isa Mooncake.NoTangent) || (x_tangent_val isa Mooncake.NoFData) ? Mooncake.NoRData() : zero(Mooncake.rdata(x_tangent_val))
            return (Mooncake.NoRData(), Mooncake.NoRData(), rdata_for_x)
        end
    end
    return y_cd, _new_A_x_pullback
end

# A(x::T, a::A{T})
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(Mooncake._new_),Type{A{T}},T,A{T}} where {T}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake._new_)},
    ::Mooncake.CoDual{Type{A{T}}},
    x_cd::Mooncake.CoDual{T},
    a_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
) where {T,Tx}
    primal_x = Mooncake.primal(x_cd)

    raw_tangent_x = Mooncake.tangent(x_cd)

    final_tangent_for_x_field = if (raw_tangent_x isa Mooncake.NoTangent) || (raw_tangent_x isa Mooncake.NoFData)
        Mooncake.zero_tangent(primal_x)::Tx
    else
        raw_tangent_x
    end

    primal_a = Mooncake.primal(a_cd)
    tangent_a = Mooncake.tangent(a_cd)

    y_primal = A(primal_x, primal_a)

    y_fdata = TangentForA{Tx}(final_tangent_for_x_field, tangent_a)

    y_cd = Mooncake.CoDual(y_primal, y_fdata)

    function _new_A_x_a_pullback(Δy_rdata)
        # For scalar types, return the appropriate zero value
        if T <: AbstractFloat || T <: Integer
            rdata_for_x = zero(T)
        else
            x_tangent_val = Mooncake.tangent(x_cd)
            rdata_for_x = (x_tangent_val isa Mooncake.NoTangent) || (x_tangent_val isa Mooncake.NoFData) ? Mooncake.NoRData() : zero(Mooncake.rdata(x_tangent_val))
        end

        rdata_for_a = Mooncake.NoRData()

        return (Mooncake.NoRData(), Mooncake.NoRData(), rdata_for_x, rdata_for_a)
    end
    return y_cd, _new_A_x_a_pullback
end

# A(x::T, a::Nothing)
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(Mooncake._new_),Type{A{T}},T,Nothing} where {T}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake._new_)},
    ::Mooncake.CoDual{Type{A{T}}},
    x_cd::Mooncake.CoDual{T},
    a_nothing_cd::Mooncake.CoDual{Nothing,Mooncake.NoFData},
) where {T}
    primal_x = Mooncake.primal(x_cd)

    y_primal = A(primal_x)

    Tx = Mooncake.tangent_type(T)

    y_fdata = if Tx == Mooncake.NoTangent
        Mooncake.NoTangent()
    else
        raw_tangent_x = Mooncake.tangent(x_cd)
        processed_tx = (raw_tangent_x isa Mooncake.NoTangent) || (raw_tangent_x isa Mooncake.NoFData) ? Mooncake.zero_tangent(primal_x) : raw_tangent_x
        TangentForA{Tx}(processed_tx)
    end

    y_cd = Mooncake.CoDual(y_primal, y_fdata)

    function _new_A_x_nothing_pullback(Δy_rdata)
        # For Float64 inputs, we need to return Float64 rdata, not NoRData
        if T <: AbstractFloat
            return (Mooncake.NoRData(), Mooncake.NoRData(), zero(T), Mooncake.NoRData())
        else
            x_tangent_val = Mooncake.tangent(x_cd)
            rdata_for_x = (x_tangent_val isa Mooncake.NoTangent) || (x_tangent_val isa Mooncake.NoFData) ? Mooncake.NoRData() : zero(Mooncake.rdata(x_tangent_val))
            return (Mooncake.NoRData(), Mooncake.NoRData(), rdata_for_x, Mooncake.NoRData())
        end
    end
    return y_cd, _new_A_x_nothing_pullback
end

# rrule for lsetfield!(A)
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(Mooncake.lsetfield!),A{T},Val{F},Any} where {T,F}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lsetfield!)},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    field_val_cd::Mooncake.CoDual{Val{FieldName}},
    new_val_cd::Mooncake.CoDual{V}
) where {T,Tx,FieldName,V}
    a = Mooncake.primal(obj_cd)
    a_t = Mooncake.tangent(obj_cd)
    new_val_primal = Mooncake.primal(new_val_cd)
    new_val_tangent = Mooncake.tangent(new_val_cd)

    field_sym = if FieldName isa Symbol
        FieldName
    elseif FieldName isa Int
        FieldName == 1 ? :x : FieldName == 2 ? :a : throw(ArgumentError("lsetfield!: Invalid integer field '$FieldName' for type A."))
    else
        throw(ArgumentError("lsetfield!: Unsupported field type for lsetfield!"))
    end

    old_val = getfield(a, field_sym)
    old_tangent = if field_sym === :x
        a_t.x
    elseif field_sym === :a
        a_t.a
    else
        throw(ArgumentError("lsetfield!: Unknown field '$field_sym' for type A."))
    end

    Mooncake.lsetfield!(a, Val(field_sym), new_val_primal)
    new_field_tangent = if (new_val_tangent isa Mooncake.NoTangent) || (new_val_tangent isa Mooncake.NoFData)
        Mooncake.zero_tangent(new_val_primal)
    else
        new_val_tangent
    end
    if field_sym === :x
        a_t.x = new_field_tangent
    elseif field_sym === :a
        a_t.a = new_field_tangent
    end

    y_fdata = Mooncake.fdata(new_field_tangent)
    y_cd = Mooncake.CoDual(new_val_primal, y_fdata)

    function lsetfield_A_pullback(dy_rdata)
        Mooncake.lsetfield!(a, Val(field_sym), old_val)
        if field_sym === :x
            a_t.x = old_tangent
        elseif field_sym === :a
            a_t.a = old_tangent
        end
        return (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), dy_rdata)
    end

    return y_cd, lsetfield_A_pullback
end
```

### Test utilities

Implement the test utility functions:

```@example custom_tangent_type
function Mooncake.TestUtils.populate_address_map_internal(m::Mooncake.TestUtils.AddressMap, p::A{T}, t::TangentForA{Tx}) where {T,Tx}
    k = Base.pointer_from_objref(p)
    v = Base.pointer_from_objref(t)
    if haskey(m, k)
        @assert m[k] == v
        return m
    end
    m[k] = v
    Mooncake.TestUtils.populate_address_map_internal(m, p.x, t.x)
    if !(t.a isa Mooncake.NoTangent)
        Mooncake.TestUtils.populate_address_map_internal(m, p.a, t.a)
    end
    return m
end

function Mooncake.TestUtils.has_equal_data_internal(x::A{T}, y::A{T}, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}) where {T}
    id_pair = (objectid(x), objectid(y))
    haskey(d, id_pair) && return d[id_pair]
    d[id_pair] = true
    eq = Mooncake.TestUtils.has_equal_data_internal(x.x, y.x, equal_undefs, d)
    if (x.a === nothing) != (y.a === nothing)
        return false
    elseif x.a === nothing
        return eq
    else
        return eq && Mooncake.TestUtils.has_equal_data_internal(x.a, y.a, equal_undefs, d)
    end
end

function Mooncake.TestUtils.has_equal_data_internal(t::TangentForA{Tx}, s::TangentForA{Tx}, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}) where {Tx}
    id_pair = (objectid(t), objectid(s))
    haskey(d, id_pair) && return d[id_pair]
    d[id_pair] = true
    eq = Mooncake.TestUtils.has_equal_data_internal(t.x, s.x, equal_undefs, d)
    if (t.a isa Mooncake.NoTangent) != (s.a isa Mooncake.NoTangent)
        return false
    elseif t.a isa Mooncake.NoTangent
        return eq
    else
        return eq && Mooncake.TestUtils.has_equal_data_internal(t.a, s.a, equal_undefs, d)
    end
end
```

Now we can run it again and successfully check if all the tangent / fdata / rdata and other required functionality works correctly for the self-referential type A.

```@example custom_tangent_type
Mooncake.TestUtils.test_data(Random.default_rng(), A(1.0, A(2.0, A(3.0))))
```
