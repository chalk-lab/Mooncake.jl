#
# General utilities
#

@unstable function parse_signature_expr(sig::Expr)
    # Different parsing is required for `Tuple{...}` vs `Tuple{...} where ...`.
    if sig.head == :curly
        @assert sig.args[1] == :Tuple
        arg_type_symbols = map(esc, sig.args[2:end])
        where_params = nothing
    elseif sig.head == :where
        @assert sig.args[1].args[1] == :Tuple
        arg_type_symbols = map(esc, sig.args[1].args[2:end])
        where_params = map(esc, sig.args[2:end])
    else
        throw(ArgumentError("Expected either a `Tuple{...}` or `Tuple{...} where {...}"))
    end
    return arg_type_symbols, where_params
end

function construct_rrule_def(arg_names, arg_types, where_params, body)
    return construct_rule_def(:(Mooncake.rrule!!), arg_names, arg_types, where_params, body)
end

function construct_frule_def(arg_names, arg_types, where_params, body)
    return construct_rule_def(:(Mooncake.frule!!), arg_names, arg_types, where_params, body)
end

function construct_rule_def(name, arg_names, arg_types, where_params, body)
    arg_exprs = map((n, t) -> :($n::$t), arg_names, arg_types)
    def = Dict(:head => :function, :name => name, :args => arg_exprs, :body => body)
    where_params !== nothing && setindex!(def, where_params, :whereparams)
    return ExprTools.combinedef(def)
end

#
# Functionality supporting @mooncake_overlay
#

"""
    @mooncake_overlay method_expr

Define a method of a function which only Mooncake can see. This can be used to write
versions of methods which can be successfully differentiated by Mooncake if the original
cannot be.

For example, suppose that you have a function
```jldoctest overlay
julia> foo(x::Float64) = bar(x)
foo (generic function with 1 method)
```
where Mooncake.jl fails to differentiate `bar` for some reason.
If you have access to another function `baz`, which does the same thing as `bar`, but does
    so in a way which Mooncake.jl can differentiate, you can simply write:
```jldoctest overlay
julia> Mooncake.@mooncake_overlay foo(x::Float64) = baz(x)

```
When looking up the code for `foo(::Float64)`, Mooncake.jl will see this method, rather than
the original, and differentiate it instead.

# A Worked Example

To demonstrate how to use `@mooncake_overlay`s in practice, we here demonstrate how the
answer that Mooncake.jl gives changes if you change the definition of a function using a
`@mooncake_overlay`.
Do not do this in practice -- this is just a simple way to demonostrate how to use overlays!

First, consider a simple example:
```jldoctest overlay-doctest
julia> scale(x) = 2x
scale (generic function with 1 method)

julia> rule = Mooncake.build_rrule(Tuple{typeof(scale), Float64});

julia> Mooncake.value_and_gradient!!(rule, scale, 5.0)
(10.0, (NoTangent(), 2.0))
```

We can use `@mooncake_overlay` to change the definition which Mooncake.jl sees:
```jldoctest overlay-doctest
julia> Mooncake.@mooncake_overlay scale(x) = 3x

julia> rule = Mooncake.build_rrule(Tuple{typeof(scale), Float64});

julia> Mooncake.value_and_gradient!!(rule, scale, 5.0)
(15.0, (NoTangent(), 3.0))
```
As can be seen from the output, the result of differentiating using Mooncake.jl has changed
to reflect the overlay-ed definition of the method.

Additionally, it is possible to use the usual multi-line syntax to declare an overlay:
```jldoctest overlay-doctest
julia> Mooncake.@mooncake_overlay function scale(x)
           return 4x
       end

julia> rule = Mooncake.build_rrule(Tuple{typeof(scale), Float64});

julia> Mooncake.value_and_gradient!!(rule, scale, 5.0)
(20.0, (NoTangent(), 4.0))
```
"""
macro mooncake_overlay(method_expr)
    def = splitdef(method_expr)
    __mooncake_method_table = gensym("mooncake_method_table")
    def[:name] = Expr(:overlay, __mooncake_method_table, def[:name])
    return quote
        $(esc(__mooncake_method_table)) = Mooncake.mooncake_method_table
        $(esc(combinedef(def)))
    end
end

#
# Functionality supporting @zero_adjoint
#

"""
    zero_adjoint(f::CoDual, x::Vararg{CoDual, N}) where {N}

Utility functionality for constructing `rrule!!`s for functions whose adjoints always return
zero.

NOTE: you should only make use of this function if you cannot make use of the
[`@zero_adjoint`](@ref) macro.

You make use of this functionality by writing a method of `Mooncake.rrule!!`, and
passing all of its arguments (including the function itself) to this function. For example:
```jldoctest
julia> import Mooncake: zero_adjoint, DefaultCtx, zero_fcodual, rrule!!, is_primitive, CoDual

julia> foo(x::Vararg{Int}) = 5
foo (generic function with 1 method)

julia> is_primitive(::Type{DefaultCtx}, ::Type{<:Tuple{typeof(foo), Vararg{Int}}}) = true;

julia> rrule!!(f::CoDual{typeof(foo)}, x::Vararg{CoDual{Int}}) = zero_adjoint(f, x...);

julia> rrule!!(zero_fcodual(foo), zero_fcodual(3), zero_fcodual(2))[2](NoRData())
(NoRData(), NoRData(), NoRData())
```

WARNING: this is only correct if the output of `primal(f)(map(primal, x)...)` does not alias
anything in `f` or `x`. This is always the case if the result is a bits type, but more care
may be required if it is not.
```
"""
@inline function zero_adjoint(f::CoDual, x::Vararg{CoDual,N}) where {N}
    return zero_fcodual(primal(f)(map(primal, x)...)), NoPullback(f, x...)
end

"""
    zero_derivative(f::Dual, x::Vararg{Dual,N}) where {N}

Utility functionality for constructing `frule!!`s for functions whose derivatives always
return zero.

NOTE: you should only make use of this function if you cannot make use of the
[`@zero_derivative`](@ref) macro.

You make use of this functionality by writing a method of `Mooncake.frule!!`, and
passing all of its arguments (including the function itself) to this function. For example:
```jldoctest
julia> import Mooncake: zero_derivative, DefaultCtx, zero_dual, frule!!, Dual

julia> foo(x::Vararg{Int}) = 5
foo (generic function with 1 method)

julia> frule!!(f::Dual{typeof(foo)}, x::Vararg{Dual{Int}}) = zero_derivative(f, x...);

julia> frule!!(zero_dual(foo), zero_dual(3), zero_dual(2))
Dual{Int64, NoTangent}(5, NoTangent())
```
"""
@inline function zero_derivative(f::Dual, x::Vararg{Dual,N}) where {N}
    return zero_dual(primal(f)(map(primal, x)...))
end

"""
    zero_derivative(ctx, sig, [mode=Mode])

Declares that the derivative of the mode for `sig` is always zero, for all arguments. This
also implies that the adjoint of the derivative is always zero for all arguments.

Accordingly, if `mode===Mode` (the default) this macro creates a method of
[`is_primitive`](@ref) which returns `true` for `ctx`, `sig`, and both [`ForwardMode`](@ref)
and [`ReverseMode`](@ref). It additionally creates methods of [`frule!!`](@ref) and
[`rrule!!`](@ref) which always return zero / do not increment tangents and fdata.

Users of ChainRules.jl should be familiar with this functionality -- it is morally the same
as `ChainRulesCore.@non_differentiable`.

For example:
```jldoctest
julia> using Mooncake: @zero_derivative, DefaultCtx, zero_dual, zero_fcodual, frule!!, rrule!!, is_primitive, ForwardMode, ReverseMode

julia> foo(x) = 5
foo (generic function with 1 method)

julia> @zero_derivative DefaultCtx Tuple{typeof(foo), Any}

julia> is_primitive(DefaultCtx, ForwardMode, Tuple{typeof(foo), Any})
true

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo), Any})
true

julia> frule!!(zero_dual(foo), zero_dual(3.0))
Mooncake.Dual{Int64, NoTangent}(5, NoTangent())

julia> rrule!!(zero_fcodual(foo), zero_fcodual(3.0))[2](NoRData())
(NoRData(), 0.0)
```

Limited support for `Vararg`s is also available. For example
```jldoctest
julia> using Mooncake: @zero_derivative, DefaultCtx, zero_fcodual, rrule!!, is_primitive, ReverseMode

julia> foo_varargs(x...) = 5
foo_varargs (generic function with 1 method)

julia> @zero_derivative DefaultCtx Tuple{typeof(foo_varargs), Vararg}

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo_varargs), Any, Float64, Int})
true

julia> rrule!!(zero_fcodual(foo_varargs), zero_fcodual(3.0), zero_fcodual(5))[2](NoRData())
(NoRData(), 0.0, NoRData())
```
Be aware that it is not currently possible to specify any of the type parameters of the
`Vararg`. For example, the signature `Tuple{typeof(foo), Vararg{Float64, 5}}` will not work
with this macro.

WARNING: this is only correct if the output of the function does not alias any fields of the
function, or any of its arguments. For example, applying this macro to the function `x -> x`
will yield incorrect results.

As always, you should use [`TestUtils.test_rule`](@ref) to ensure that you've not
made a mistake.

# Signatures Unsupported By This Macro

If the signature you wish to apply `@zero_derivative` to is not supported, for example because
it uses a `Vararg` with a type parameter, you can still make use of
[`zero_derivative`](@ref).

"""
macro zero_derivative(ctx, sig, mode=Mode)
    mode = mode == :ForwardMode ? ForwardMode : mode
    mode = mode == :ReverseMode ? ReverseMode : mode
    return _zero_derivative_impl(ctx, sig, mode)
end

function _zero_derivative_impl(ctx, sig, mode)

    # Parse the signature, and construct the rule definition. If it is a vararg definition,
    # then the last argument requires special treatment.
    arg_type_symbols, where_params = parse_signature_expr(sig)
    arg_names = map(n -> Symbol("x_$n"), eachindex(arg_type_symbols))
    is_vararg = arg_type_symbols[end] == Expr(:escape, :Vararg)
    if is_vararg
        arg_types_deriv = vcat(
            map(t -> :(Mooncake.Dual{<:$t}), arg_type_symbols[1:(end - 1)]),
            :(Vararg{Mooncake.Dual}),
        )
        arg_types_adjoint = vcat(
            map(t -> :(Mooncake.CoDual{<:$t}), arg_type_symbols[1:(end - 1)]),
            :(Vararg{Mooncake.CoDual}),
        )
        splat_symbol = Expr(Symbol("..."), arg_names[end])
        tmp = arg_names[1:(end - 1)]
        body_deriv = Expr(:call, Mooncake.zero_derivative, tmp..., splat_symbol)
        body_adjoint = Expr(:call, Mooncake.zero_adjoint, tmp..., splat_symbol)
    else
        arg_types_deriv = map(t -> :(Mooncake.Dual{<:$t}), arg_type_symbols)
        arg_types_adjoint = map(t -> :(Mooncake.CoDual{<:$t}), arg_type_symbols)
        body_deriv = Expr(:call, Mooncake.zero_derivative, arg_names...)
        body_adjoint = Expr(:call, Mooncake.zero_adjoint, arg_names...)
    end

    # Construct is_primitive statement. If no mode is provided, then construct a statement
    # which does not escape the mode argument. This will work even if the names `Mooncake`
    # or `Mooncake.Mode` are not available in the scope which calls this macro.
    is_primitive_ex = quote
        function Mooncake.is_primitive(
            ::Type{$(esc(ctx))}, ::Type{<:$mode}, ::Type{<:$(esc(sig))}
        )
            return true
        end
    end

    # Figuring out which mode argument was actually provided is going to be very hard in
    # general, and rather error prone, because the mode might appear as a `Type`, one of
    # several `Symbol`s, or possibly something else not considered. As a result, we always
    # define both the frule and rrule, and rely on the method of `is_primitive` defined
    # above to determine whether or not they do anything. This might inflate the method
    # table a bit for `frule!!` and `rrule!!` unnecessarily, but it will be robust.
    frule_ex = construct_frule_def(arg_names, arg_types_deriv, where_params, body_deriv)
    rrule_ex = construct_rrule_def(arg_names, arg_types_adjoint, where_params, body_adjoint)

    return Expr(:block, is_primitive_ex, frule_ex, rrule_ex)
end

"""
    @zero_adjoint ctx sig

Equivalent to `@zero_derivative ctx sig ReverseMode`. Consult the docstring for
[`@zero_derivative`](@ref) for more information.
"""
macro zero_adjoint(ctx, sig)
    return _zero_derivative_impl(ctx, sig, ReverseMode)
end

#
# Functionality supporting @from_rrule
#

"""
    to_cr_tangent(t)

Convert a Mooncake tangent into a type that ChainRules.jl `rrule`s expect to see.
"""
to_cr_tangent(t::IEEEFloat) = t
to_cr_tangent(t::Array{<:IEEEFloat}) = t
to_cr_tangent(t::Array) = map(to_cr_tangent, t)
to_cr_tangent(::NoTangent) = CRC.NoTangent()
to_cr_tangent(t::Tangent) = CRC.Tangent{Any}(; map(to_cr_tangent, t.fields)...)
to_cr_tangent(t::MutableTangent) = CRC.Tangent{Any}(; map(to_cr_tangent, t.fields)...)
to_cr_tangent(t::Tuple) = CRC.Tangent{Any}(map(to_cr_tangent, t)...)

"""
    mooncake_tangent(p, cr_tangent)

For primal `p` and a tangent used by ChainRules `cr_tangent`, returns the tangent of type
`tangent_type(typeof(p))`. Useful for converting the result of a `ChainRules.frule` into
something that Mooncake can use.
"""
mooncake_tangent(p, ::CRC.NoTangent) = NoTangent()
mooncake_tangent(p, t::IEEEFloat) = t
mooncake_tangent(p::Array, t::Array{<:IEEEFloat}) = t
mooncake_tangent(p::Array, t::Array) = map(mooncake_tangent, p, t)
mooncake_tangent(p, t::CRC.ZeroTangent) = zero_tangent(p)
function mooncake_tangent(p::P, t::T) where {P,T<:Tuple}
    return tangent_type(P) == NoTangent ? NoTangent() : map(mooncake_tangent, p, t)
end
function mooncake_tangent(p::P, t::T) where {P<:Tuple,T<:CRC.Tangent}
    return tangent_type(P) == NoTangent ? NoTangent() : map(mooncake_tangent, p, t.backing)
end

"""
    increment_and_get_rdata!(fdata, zero_rdata, cr_tangent)

Increment `fdata` by the fdata component of the ChainRules.jl-style tangent, `cr_tangent`,
and return the rdata component of `cr_tangent` by adding it to `zero_rdata`.
"""
increment_and_get_rdata!(::NoFData, r::T, t::T) where {T<:IEEEFloat} = r + t
function increment_and_get_rdata!(f::Array{P}, ::NoRData, t::Array{P}) where {P<:IEEEFloat}
    increment!!(f, t)
    return NoRData()
end
increment_and_get_rdata!(::Any, r, ::CRC.NoTangent) = r
function increment_and_get_rdata!(f, r, t::CRC.Thunk)
    return increment_and_get_rdata!(f, r, CRC.unthunk(t))
end

"""
    frule_wrapper(f::Dual, args::Dual...)

Implements an `frule!!` for `f` applied to `args` by calling `ChainRulesCore.frule`.
"""
function frule_wrapper(fargs::Vararg{Dual,N}) where {N}
    tangents = tuple_map(to_cr_tangent ∘ tangent, fargs)
    Ω, dΩ = CRC.frule(tangents, tuple_map(primal, fargs)...)
    return Dual(Ω, mooncake_tangent(Ω, dΩ))
end

function frule_wrapper(::Dual{typeof(Core.kwcall)}, fargs::Vararg{Dual,N}) where {N}
    primals = map(primal, fargs)
    tangents = map(to_cr_tangent ∘ tangent, fargs[2:end])
    Ω, dΩ = Core.kwcall(primals[1], CRC.frule, tangents, primals[2:end]...)
    return Dual(Ω, mooncake_tangent(Ω, dΩ))
end

function construct_frule_wrapper_def(arg_names, arg_types, where_params)
    body = Expr(:call, frule_wrapper, arg_names...)
    return construct_frule_def(arg_names, arg_types, where_params, body)
end

"""
    rrule_wrapper(f::CoDual, args::CoDual...)

Used to implement `rrule!!`s via `ChainRulesCore.rrule`.

Given a function `foo`, argument types `arg_types`, and a method of `ChainRulesCore.rrule`
which applies to these, you can make use of this function as follows:
```julia
Mooncake.@is_primitive DefaultCtx Tuple{typeof(foo), arg_types...}
function Mooncake.rrule!!(f::CoDual{typeof(foo)}, args::CoDual...)
    return rrule_wrapper(f, args...)
end
```
Assumes that methods of `to_cr_tangent` and `to_mooncake_tangent` are defined such that you
can convert between the different representations of tangents that Mooncake and
ChainRulesCore expect.

Furthermore, it is _essential_ that
1. `f(args)` does not mutate `f` or `args`, and
2. the result of `f(args)` does not alias any data stored in `f` or `args`.

Subject to some constraints, you can use the [`@from_rrule`](@ref) macro to reduce the
amount of boilerplate code that you are required to write even further.
"""
function rrule_wrapper(fargs::Vararg{CoDual,N}) where {N}

    # Run forwards-pass.
    primals = tuple_map(primal, fargs)
    lazy_rdata = tuple_map(Mooncake.lazy_zero_rdata, primals)
    y_primal, cr_pb = CRC.rrule(primals...)
    y_fdata = fdata(zero_tangent(y_primal))

    function pb!!(y_rdata)

        # Construct tangent w.r.t. output.
        cr_tangent = to_cr_tangent(tangent(y_fdata, y_rdata))

        # Run reverse-pass using ChainRules.
        cr_dfargs = cr_pb(cr_tangent)

        # Increment fdata and get rdata.
        return map(fargs, lazy_rdata, cr_dfargs) do x, l_rdata, cr_dx
            return increment_and_get_rdata!(tangent(x), instantiate(l_rdata), cr_dx)
        end
    end
    return CoDual(y_primal, y_fdata), pb!!
end

function rrule_wrapper(::CoDual{typeof(Core.kwcall)}, fargs::Vararg{CoDual,N}) where {N}

    # Run forwards-pass.
    primals = tuple_map(primal, fargs)
    lazy_rdata = tuple_map(lazy_zero_rdata, primals)
    y_primal, cr_pb = Core.kwcall(primals[1], CRC.rrule, primals[2:end]...)
    y_fdata = fdata(zero_tangent(y_primal))

    function pb!!(y_rdata)

        # Construct tangent w.r.t. output.
        cr_tangent = to_cr_tangent(tangent(y_fdata, y_rdata))

        # Run reverse-pass using ChainRules.
        cr_dfargs = cr_pb(cr_tangent)

        # Increment fdata and compute rdata.
        kwargs_rdata = rdata(zero_tangent(primals[1]))
        args_rdata = map(fargs[2:end], lazy_rdata[2:end], cr_dfargs) do x, l_rdata, cr_dx
            return increment_and_get_rdata!(tangent(x), instantiate(l_rdata), cr_dx)
        end
        return NoRData(), kwargs_rdata, args_rdata...
    end
    return CoDual(y_primal, y_fdata), pb!!
end

function construct_rrule_wrapper_def(arg_names, arg_types, where_params)
    body = Expr(:call, rrule_wrapper, arg_names...)
    return construct_rrule_def(arg_names, arg_types, where_params, body)
end

"""
    @from_chainrules ctx sig [has_kwargs=false mode=nothing]

Convenience functionality to assist in using `ChainRuleCore.frule`s and
`ChainRulesCore.rrule`s to write `frule!!`s and `rrule!!`s.

# Arguments

- `ctx`: A Mooncake context type
- `sig`: the signature which you wish to assert should be a primitive in `Mooncake.jl`, and
    use an existing `ChainRulesCore.rrule` or `ChainRulesCore.frule` to implement this functionality.
- `has_kwargs=true`: a `Bool` stating whether or not the function has keyword arguments.
    This feature has the same limitations as `ChainRulesCore.frule` and
    `ChainRulesCore.rrule` and  -- the derivative w.r.t. all kwargs must be zero.
- `mode=nothing`: the mode to produce rules for. By default, produces rules for both forward
    and reverse mode. If `mode=ForwardMode` only rules for forward mode are produced. If
    `mode=ReverseMode` only rules for reverse mode are produced.

# Example Usage

## A Basic Example

```jldoctest
julia> using Mooncake: @from_chainrules, DefaultCtx, frule!!, rrule!!, Dual, zero_dual, zero_fcodual, TestUtils

julia> import ChainRulesCore

julia> foo(x::Real) = 5x;

julia> ChainRulesCore.frule((df, dx), ::typeof(foo), x::Real) = 5x, 5dx;

julia> function ChainRulesCore.rrule(::typeof(foo), x::Real)
           foo_pb(Ω::Real) = ChainRulesCore.NoTangent(), 5Ω
           return foo(x), foo_pb
       end;

julia> @from_chainrules DefaultCtx Tuple{typeof(foo), Base.IEEEFloat}

julia> frule!!(zero_dual(foo), Dual(5.0, 2.0))
Dual{Float64, Float64}(25.0, 10.0)

julia> rrule!!(zero_fcodual(foo), zero_fcodual(5.0))[2](1.0)
(NoRData(), 5.0)

julia> # Check that the rule works as intended. Put this in your test suite.
       TestUtils.test_rule(Xoshiro(123), foo, 5.0; is_primitive=true, print_results=false);
```

## An Example with Keyword Arguments and ReverseMode

```jldoctest
julia> using Mooncake: @from_chainrules, DefaultCtx, rrule!!, zero_fcodual, TestUtils, ReverseMode

julia> import ChainRulesCore

julia> foo(x::Real; cond::Bool) = cond ? 5x : 4x;

julia> function ChainRulesCore.rrule(::typeof(foo), x::Real; cond::Bool)
           foo_pb(Ω::Real) = ChainRulesCore.NoTangent(), cond ? 5Ω : 4Ω
           return foo(x; cond), foo_pb
       end;

julia> @from_chainrules DefaultCtx Tuple{typeof(foo), Base.IEEEFloat} true ReverseMode

julia> _, pb = rrule!!(
           zero_fcodual(Core.kwcall),
           zero_fcodual((cond=false, )),
           zero_fcodual(foo),
           zero_fcodual(5.0),
       );

julia> pb(3.0)
(NoRData(), NoRData(), NoRData(), 12.0)

julia> # Check that the rule works as intended. Put this in your test suite.
       TestUtils.test_rule(
           Xoshiro(123), Core.kwcall, (cond=false, ), foo, 5.0;
           is_primitive=true, print_results=false, mode=Mooncake.ReverseMode,
       );
```
Notice that, in order to access the kwarg method we must call the method of `Core.kwcall`,
as Mooncake's `rrule!!` does not itself permit the use of kwargs.

# Limitations

It is your responsibility to ensure that
1. calls with signature `sig` do not mutate their arguments,
2. the output of calls with signature `sig` does not alias any of the inputs.

As with all hand-written rules, you should definitely make use of
[`TestUtils.test_rule`](@ref) to verify correctness on some test cases.

# Argument Type Constraints

Many methods of `ChainRuleCore.rrule` are implemented with very loose type constraints.
For example, it would not be surprising to see a method of rrule with the signature
```julia
Tuple{typeof(rrule), typeof(foo), Real, AbstractVector{<:Real}}
```
There are a variety of reasons for this way of doing things, and whether it is a good idea
to write rules for such generic objects has been debated at length.

Suffice it to say, you should not write rules for Mooncake which are so generically
typed.
Rather, you should create rules for the subset of types for which you believe that the
`ChainRulesCore.rrule` will work correctly, and leave this package to derive rules for the
rest.
For example, it is quite common to be confident that a given rule will work correctly for
any `Base.IEEEFloat` argument, i.e. `Union{Float16, Float32, Float64}`, but it is usually
not possible to know that the rule is correct for all possible subtypes of `Real` that
someone might define.

# Conversions Between Different Tangent Type Systems

Under the hood, this functionality relies on three functions: `Mooncake.mooncake_tangent`,
`Mooncake.to_cr_tangent`, and `Mooncake.increment_and_get_rdata!`. These two functions
handle conversion to / from `Mooncake` tangent types and `ChainRulesCore` tangent types.
This functionality is known to work well for simple types, but has not been tested to a
great extent on complicated composite types. If `@from_chainrules` does not work in your
case because the required method of either of these functions does not exist, please open an
issue.
"""
macro from_chainrules(ctx, sig::Expr, has_kwargs::Bool=false, mode=Mode)
    return _from_chainrules_impl(ctx, sig, has_kwargs, mode)
end

function _from_chainrules_impl(ctx, sig::Expr, has_kwargs::Bool, mode)
    arg_type_syms, where_params = parse_signature_expr(sig)
    arg_names = map(n -> Symbol("x_$n"), eachindex(arg_type_syms))
    dual_arg_types = map(t -> :(Mooncake.Dual{<:$t}), arg_type_syms)
    codual_arg_types = map(t -> :(Mooncake.CoDual{<:$t}), arg_type_syms)
    frule_expr = construct_frule_wrapper_def(arg_names, dual_arg_types, where_params)
    rrule_expr = construct_rrule_wrapper_def(arg_names, codual_arg_types, where_params)

    if has_kwargs
        kw_sig = Expr(:curly, :Tuple, :(typeof(Core.kwcall)), :NamedTuple, arg_type_syms...)
        kw_sig = where_params === nothing ? kw_sig : Expr(:where, kw_sig, where_params...)
        # Type M will be available later on, and will be the mode type.
        kw_is_primitive = quote
            function Mooncake.is_primitive(
                ::Type{$(esc(ctx))}, ::Type{<:$mode}, ::Type{<:$kw_sig}
            )
                return true
            end
        end
        kwargs_frule_expr = construct_frule_wrapper_def(
            vcat(:_kwcall, :kwargs, arg_names),
            vcat(
                :(Mooncake.Dual{typeof(Core.kwcall)}),
                :(Mooncake.Dual{<:NamedTuple}),
                dual_arg_types,
            ),
            where_params,
        )
        kwargs_rrule_expr = construct_rrule_wrapper_def(
            vcat(:_kwcall, :kwargs, arg_names),
            vcat(
                :(Mooncake.CoDual{typeof(Core.kwcall)}),
                :(Mooncake.CoDual{<:NamedTuple}),
                codual_arg_types,
            ),
            where_params,
        )
    else
        kw_is_primitive = nothing
        kwargs_frule_expr = nothing
        kwargs_rrule_expr = nothing
    end

    return quote
        function Mooncake.is_primitive(
            ::Type{$(esc(ctx))}, ::Type{<:$mode}, ::Type{<:($(esc(sig)))}
        )
            return true
        end
        $frule_expr
        $rrule_expr
        $kw_is_primitive
        $kwargs_frule_expr
        $kwargs_rrule_expr
    end
end

"""
    @from_rrule ctx sig [has_kwargs=false]

Equivalent to `@from_chainrules ctx sig has_kwargs ReverseMode`. See
[`@from_chainrules`](@ref) for more information.
"""
macro from_rrule(ctx, sig::Expr, has_kwargs::Bool=false)
    return _from_chainrules_impl(ctx, sig, has_kwargs, ReverseMode)
end
