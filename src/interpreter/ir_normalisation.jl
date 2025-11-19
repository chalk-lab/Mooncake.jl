"""
    normalise!(ir::IRCode, spnames::Vector{Symbol})

Apply a sequence of standardising transformations to `ir` which leaves its semantics
unchanged, but makes AD more straightforward. In particular, replace
1. `:foreigncall` `Expr`s with `:call`s to `Mooncake._foreigncall_`,
2. `:new` `Expr`s with `:call`s to `Mooncake._new_`,
3. `:splatnew` Expr`s with `:call`s to `Mooncake._splat_new_`,
4. resolve unbound GlobalRefs (Julia 1.12+ only) to their actual values,
5. `Core.IntrinsicFunction`s with counterparts from `Mooncake.IntrinsicWrappers`,
6. `getfield(x, 1)` with `lgetfield(x, Val(1))`, and related transformations,
7. `memoryrefget` calls to `lmemoryrefget` calls, and related transformations,
8. `gc_preserve_begin` / `gc_preserve_end` exprs so that memory release is delayed.

`spnames` are the names associated to the static parameters of `ir`. These are needed when
handling `:foreigncall` expressions, in which it is not necessarily the case that all
static parameter names have been translated into either types, or `:static_parameter`
expressions.

Unfortunately, the static parameter names are not retained in `IRCode`, and the `Method`
from which the `IRCode` is derived must be consulted. `Mooncake.is_vararg_and_sparam_names`
provides a convenient way to do this.
"""
function normalise!(ir::IRCode, spnames::Vector{Symbol})
    sp_map = Dict{Symbol,CC.VarState}(zip(spnames, ir.sptypes))
    ir = interpolate_boundschecks!(ir)
    ir = fix_up_invoke_inference!(ir)
    for (n, inst) in enumerate(stmt(ir.stmts))
        inst = foreigncall_to_call(inst, sp_map)
        inst = new_to_call(inst)
        inst = splatnew_to_call(inst)
        @static if VERSION > v"1.12-"
            inst = resolve_unbound_globalrefs(ir, inst)
        end
        inst = intrinsic_to_function(inst)
        inst = lift_getfield_and_others(inst)
        inst = lift_memoryrefget_and_memoryrefset_builtins(inst)
        inst = lift_gc_preservation(inst)
        stmt(ir.stmts)[n] = inst
    end
    ir = const_prop_gotoifnots!(ir)

    # Dynamic error checks. Removing these would be like removing things from the test
    # suite. i.e. do not remove unless you're quite sure that they're redundant.
    CC.verify_ir(ir)
    verify_no_constant_gotoifnots(ir)

    return ir
end

"""
    interpolate_boundschecks!(ir::IRCode)

For every `x = Expr(:boundscheck, value)` in `ir`, interpolate `value` into all uses of `x`.
This is only required in order to ensure that literal versions of memoryrefget,
memoryrefset!, getfield, and setfield! work effectively. If they are removed through
improvements to the way that we handle constant propagation inside Mooncake, then this
functionality can be removed.
"""
function interpolate_boundschecks!(ir::IRCode)
    _interpolate_boundschecks!(stmt(ir.stmts))
    return ir
end

function _interpolate_boundschecks!(statements::Vector{Any})
    for (n, stmt) in enumerate(statements)
        if stmt isa Expr && stmt.head == :boundscheck && length(stmt.args) == 1
            def = SSAValue(n)
            val = only(stmt.args)
            for (m, stmt) in enumerate(statements)
                statements[m] = replace_uses_with!(stmt, def, val)
            end
            statements[n] = nothing
        end
    end
    return nothing
end

"""
    fix_up_invoke_inference!(ir::IRCode)

# The Problem

Consider the following:
```julia
@noinline function bar!(x)
    x .*= 2
end

function foo!(x)
    bar!(x)
    return nothing
end
```
In this case, the IR associated to `Tuple{typeof(foo), Vector{Float64}}` will be something
along the lines of
```julia
julia> Base.code_ircode_by_type(Tuple{typeof(foo), Vector{Float64}})
1-element Vector{Any}:
2 1 ─     invoke Main.bar!(_2::Vector{Float64})::Any
3 └──     return Main.nothing
   => Nothing
```
Observe that the type inferred for the first line is `Any`. Inference is at liberty to do
this without any risk of performance problems because the first line is not used anywhere
else in the function. Had this line been used elsewhere in the function, inference would
have inferred its type to be `Vector{Float64}`.

This causes performance problems for Mooncake, because it uses the return type to do
various things, including allocating storage for quantities required on the reverse-pass.
Consequently, inference infering `Any` rather than `Vector{Float64}` causes type
instabilities in the code that Mooncake generates, which can have catastrophic conseqeuences
for performance.

# The Solution

`:invoke` expressions contain the `Core.MethodInstance` associated to them, which contains
a `Core.CodeCache`, which contains the return type of the `:invoke`. This function looks
for `:invoke` statements whose return type is inferred to be `Any` in `ir`, and modifies it
to be the return type given by the code cache.
"""
function fix_up_invoke_inference!(ir::IRCode)::IRCode
    stmts = ir.stmts
    for n in 1:length(stmts)
        if Meta.isexpr(stmt(stmts)[n], :invoke) && CC.widenconst(stmts.type[n]) == Any
            mi = get_mi(stmt(stmts)[n].args[1])
            R = isdefined(mi, :cache) ? mi.cache.rettype : CC.return_type(mi.specTypes)
            stmts.type[n] = R
        end
    end
    return ir
end
function get_mi(ci::Core.CodeInstance)
    @static isdefined(CC, :get_ci_mi) ? CC.get_ci_mi(ci) : ci.def
end
get_mi(mi::Core.MethodInstance) = mi

"""
    foreigncall_to_call(inst, sp_map::Dict{Symbol, CC.VarState})

If `inst` is a `:foreigncall` expression translate it into an equivalent `:call` expression.
If anything else, just return `inst`. See `Mooncake._foreigncall_` for details.

`sp_map` maps the names of the static parameters to their values. This function is intended
to be called in the context of an `IRCode`, in which case the values of `sp_map` are given
by the `sptypes` field of said `IRCode`. The keys should generally be obtained from the
`Method` from which the `IRCode` is derived. See `Mooncake.normalise!` for more details.

The purpose of this transformation is to make it possible to differentiate `:foreigncall`
expressions in the same way as a primitive `:call` expression, i.e. via an `rrule!!`.
"""
function foreigncall_to_call(inst, sp_map::Dict{Symbol,CC.VarState})
    if Meta.isexpr(inst, :foreigncall)
        # See Julia's AST devdocs for info on `:foreigncall` expressions.
        args = inst.args
        name = __extract_foreigncall_name(args[1])
        RT = Val(interpolate_sparams(args[2], sp_map))
        AT = (map(x -> Val(interpolate_sparams(x, sp_map)), args[3])...,)
        nreq = Val(args[4])
        calling_convention = Val(args[5] isa QuoteNode ? args[5].value : args[5])
        x = args[6:end]
        return Expr(:call, _foreigncall_, name, RT, AT, nreq, calling_convention, x...)
    else
        return inst
    end
end

# Copied from Umlaut.jl.
__extract_foreigncall_name(x::Symbol) = Val(x)
function __extract_foreigncall_name(x::Expr)
    # Make sure that we're getting the expression that we're expecting.
    !Meta.isexpr(x, :call) && error("unexpected expr $x")
    !isa(x.args[1], GlobalRef) && error("unexpected expr $x")
    x.args[1].name != :tuple && error("unexpected expr $x")
    length(x.args) != 3 && error("unexpected expr $x")

    # Parse it into a name that can be passed as a type.
    v = eval(x)
    return Val((Symbol(v[1]), Symbol(v[2])))
end
__extract_foreigncall_name(v::Tuple) = Val((Symbol(v[1]), Symbol(v[2])))
__extract_foreigncall_name(x::QuoteNode) = __extract_foreigncall_name(x.value)
function __extract_foreigncall_name(x::GlobalRef)
    return __extract_foreigncall_name(getglobal(x.mod, x.name))
end

# Copied from Umlaut.jl. Originally, adapted from
# https://github.com/JuliaDebug/JuliaInterpreter.jl/blob/aefaa300746b95b75f99d944a61a07a8cb145ef3/src/optimize.jl#L239
function interpolate_sparams(@nospecialize(t::Type), sparams::Dict{Symbol,CC.VarState})
    t isa Core.TypeofBottom && return t
    while t isa UnionAll
        t = t.body
    end
    t = t::DataType
    if Base.isvarargtype(t)
        return Expr(:(...), t.parameters[1])
    end
    if Base.has_free_typevars(t)
        params = map(t.parameters) do @nospecialize(p)
            if isa(p, TypeVar)
                return sparams[p.name].typ.val
            elseif isa(p, DataType) && Base.has_free_typevars(p)
                return interpolate_sparams(p, sparams)
            elseif p isa CC.VarState
                @show "doing varstate"
                p.typ
            else
                return p
            end
        end
        T = t.name.Typeofwrapper.parameters[1]
        return T{params...}
    end
    return t
end

"""
    new_to_call(x)

If instruction `x` is a `:new` expression, replace it with a `:call` to `Mooncake._new_`.
Otherwise, return `x`.

The purpose of this transformation is to make it possible to differentiate `:new`
expressions in the same way as a primitive `:call` expression, i.e. via an `rrule!!`.
"""
new_to_call(x) = Meta.isexpr(x, :new) ? Expr(:call, _new_, x.args...) : x

"""
    splatnew_to_call(x)

If instruction `x` is a `:splatnew` expression, replace it with a `:call` to
`Mooncake._splat_new_`. Otherwise return `x`.

The purpose of this transformation is to make it possible to differentiate `:splatnew`
expressions in the same way as a primitive `:call` expression, i.e. via an `rrule!!`.
"""
splatnew_to_call(x) = Meta.isexpr(x, :splatnew) ? Expr(:call, _splat_new_, x.args...) : x

"""
    resolve_unbound_globalrefs(ir, ex)

Resolve GlobalRefs that are considered unbound in the given world range, but are actually
defined due to partitioned bindings. 

Only available on Julia 1.12+.
"""
# XXX: this is probably unsound, but the best workaround so far
function resolve_unbound_globalrefs(ir, @nospecialize ex)
    isa(ex, Expr) || return ex
    args = nothing
    for i in eachindex(ex.args)
        arg = ex.args[i]
        # extracted from `Compiler.check_op` (called by `Compiler.verify_ir`)
        if isa(arg, GlobalRef) && arg.mod !== Core && arg.mod !== Base
            (valid_worlds, alldef) = CC.scan_leaf_partitions(
                nothing,
                arg,
                CC.WorldWithRange(CC.min_world(ir.valid_worlds), ir.valid_worlds),
            ) do _, _, bpart
                CC.is_defined_const_binding(CC.binding_kind(bpart))
            end
            if !alldef ||
                CC.max_world(valid_worlds) < CC.max_world(ir.valid_worlds) ||
                CC.min_world(valid_worlds) > CC.min_world(ir.valid_worlds)
                # this is the potentially unsound bit, because we
                # resolve a binding that was otherwise thought to be
                # unbound during type inference.
                if isdefined(arg.mod, arg.name)
                    arg = getglobal(arg.mod, arg.name)
                end
            end
            if args === nothing
                args = ex.args[1:(i - 1)]
            end
        end
        args === nothing && continue
        push!(args, arg)
    end
    args === nothing && return ex
    resolved = Expr(ex.head)
    resolved.args = args
    return resolved
end

"""
    intrinsic_to_function(inst)

If `inst` is a `:call` expression to a `Core.IntrinsicFunction`, replace it with a call to
the corresponding `function` from `Mooncake.IntrinsicsWrappers`, else return `inst`.

`cglobal` is a special case -- it requires that its first argument be static in exactly the
same way as `:foreigncall`. See `IntrinsicsWrappers.__cglobal` for more info.

The purpose of this transformation is to make it possible to use dispatch to write rules for
intrinsic calls using dispatch in a type-stable way. See [`IntrinsicsWrappers`](@ref) for
more context.
"""
function intrinsic_to_function(inst)
    return Meta.isexpr(inst, :call) ? Expr(:call, lift_intrinsic(inst.args...)...) : inst
end

lift_intrinsic(x...) = x
lift_intrinsic(x::GlobalRef, args...) = lift_intrinsic(getglobal(x.mod, x.name), args...)
function lift_intrinsic(x::Core.IntrinsicFunction, v, args...)
    if x === cglobal
        return IntrinsicsWrappers.__cglobal, __extract_foreigncall_name(v), args...
    else
        return IntrinsicsWrappers.translate(Val(x)), v, args...
    end
end
lift_intrinsic(::typeof(Core._apply_iterate), args...) = _apply_iterate_equivalent, args...

"""
    lift_getfield_and_others(inst)

Converts expressions of the form `getfield(x, :a)` into `lgetfield(x, Val(:a))`. This has
identical semantics, but is performant in the absence of proper constant propagation.

Does the same for...
"""
function lift_getfield_and_others(inst)
    Meta.isexpr(inst, :call) || return inst
    f = __get_arg(inst.args[1])
    if f === getfield && length(inst.args) == 3 && inst.args[3] isa Union{QuoteNode,Int}
        field = inst.args[3]
        new_field = field isa Int ? Val(field) : Val(field.value)
        return Expr(:call, lgetfield, inst.args[2], new_field)
    elseif f === getfield &&
        length(inst.args) == 4 &&
        inst.args[3] isa Union{QuoteNode,Int} &&
        inst.args[4] isa Bool
        field = inst.args[3]
        new_field = field isa Int ? Val(field) : Val(field.value)
        return Expr(:call, lgetfield, inst.args[2], new_field, Val(inst.args[4]))
    elseif f === setfield! &&
        length(inst.args) == 4 &&
        inst.args[3] isa Union{QuoteNode,Int}
        name = inst.args[3]
        new_name = name isa Int ? Val(name) : Val(name.value)
        return Expr(:call, lsetfield!, inst.args[2], new_name, inst.args[4])
    else
        return inst
    end
end

__get_arg(x::GlobalRef) = getglobal(x.mod, x.name)
__get_arg(x::QuoteNode) = x.value
__get_arg(x) = x

# memoryrefget and memoryrefset! were introduced in 1.11.
@static if VERSION >= v"1.11-"
    """
        lift_memoryrefget_and_memoryrefset_builtins(inst)

    Replaces memoryrefget -> lmemoryrefget and memoryrefset! -> lmemoryrefset! if their final
    two arguments (`ordering` and `boundscheck`) are constants. See [`lmemoryrefget`] and
    [`lmemoryrefset!`](@ref) for more context.
    """
    function lift_memoryrefget_and_memoryrefset_builtins(inst)
        Meta.isexpr(inst, :call) || return inst
        f = __get_arg(inst.args[1])
        if f == Core.memoryrefget && length(inst.args) == 4
            ordering = inst.args[3]
            boundscheck = inst.args[4]
            if ordering isa QuoteNode && boundscheck isa Bool
                new_ordering = Val(ordering.value)
                return Expr(
                    :call, lmemoryrefget, inst.args[2], new_ordering, Val(boundscheck)
                )
            else
                return inst
            end
        elseif f == Core.memoryrefset! && length(inst.args) == 5
            ordering = inst.args[4]
            boundscheck = inst.args[5]
            if ordering isa QuoteNode && boundscheck isa Bool
                new_ordering = Val(ordering.value)
                bc = Val(boundscheck)
                return Expr(
                    :call, lmemoryrefset!, inst.args[2], inst.args[3], new_ordering, bc
                )
            else
                return inst
            end
        else
            return inst
        end
    end

else

    # memoryrefget and memoryrefset! do not exist before v1.11.
    lift_memoryrefget_and_memoryrefset_builtins(inst) = inst
end

"""
    gc_preserve(xs...)

A no-op function. Its `rrule!!` ensures that the memory associated to `xs` is not freed
until the pullback that it returns is run.
"""
@inline gc_preserve(xs...) = nothing

@is_primitive MinimalCtx Tuple{typeof(gc_preserve),Vararg{Any,N}} where {N}

frule!!(::Dual{typeof(gc_preserve)}, ::Vararg{Dual,N}) where {N} = zero_dual(nothing)
function rrule!!(f::CoDual{typeof(gc_preserve)}, xs::CoDual...)
    pb = NoPullback(f, xs...)
    gc_preserve_pb!!(::NoRData) = GC.@preserve xs pb(NoRData())
    return zero_fcodual(nothing), gc_preserve_pb!!
end

"""
    lift_gc_preserve(inst)

Expressions of the form
```julia
y = GC.@preserve x1 x2 foo(args...)
```
get lowered to
```julia
token = Expr(:gc_preserve_begin, x1, x2)
y = expr
Expr(:gc_preserve_end, token)
```
These expressions guarantee that any memory associated `x1` and `x2` not be freed until
the `:gc_preserve_end` expression is reached.

In the context of reverse-mode AD, we must ensure that the memory associated to `x1`, `x2`
and their fdata is available during the reverse pass code associated to `expr`.
We do this by preventing the memory from being freed until the `:gc_preserve_begin` is
reached on the reverse pass.

To achieve this, we replace the primal code with
```julia
# store `x` in `pb_gc_preserve` to prevent it from being freed.
_, pb_gc_preserve = rrule!!(zero_fcodual(gc_preserve), x1, x2)

# Differentiate the `:call` expression in the usual way.
y, foo_pb = rrule!!(zero_fcodual(foo), args...)

# Do not permit the GC to free `x` here.
nothing
```
The pullback should be something along the lines of
```julia
# no pullback associated to `nothing`.
nothing

# Run the pullback associated to `foo` in the usual manner. `x` must be available.
_, dargs... = foo_pb(dy)

# No-op pullback associated to `gc_preserve`.
pb_gc_preserve(NoRData())
```
"""
function lift_gc_preservation(inst)
    Meta.isexpr(inst, :gc_preserve_begin) && return Expr(:call, gc_preserve, inst.args...)
    Meta.isexpr(inst, :gc_preserve_end) && return nothing
    return inst
end

"""
    const_prop_gotoifnots(ir::IRCode)

Replace all occurences in `ir` of `goto %n if not true` in block `b` with a `goto b + 1`,
and all occurences of `goto %n if not false` with `goto n`, and make the adjustments to
`ir` that this necessitates.
"""
function const_prop_gotoifnots!(ir::IRCode)
    stmts = stmt(ir.stmts)
    for (n, stmt) in enumerate(stmts)
        if stmt isa GotoIfNot
            _current_blk = findfirst(i -> i > n, ir.cfg.index)
            current_blk = _current_blk === nothing ? length(ir.cfg.blocks) : _current_blk
            if stmt.cond === true
                stmts[n] = nothing
                remove_edge!(ir, current_blk, stmt.dest)
            elseif stmt.cond === false
                stmts[n] = GotoNode(stmt.dest)
                remove_edge!(ir, current_blk, current_blk + 1)
            end
        end
    end
    return ir
end

"""
    remove_edge!(ir::IRCode, from::Int, to::Int)

Removes an edge in `ir` from `from` to `to`. See implementation for what this entails.

Note: this is slightly different from `Core.Compiler.kill_edge!`, in that it also updates
`PhiNode`s in the `to` block. Moreover, the available methods of `remove_edge!` differ
between 1.10 and 1.11, so we need something which is stable across both.
"""
function remove_edge!(ir::IRCode, from::Int, to::Int)

    # Remove the `to` block from the `from` block's successor list.
    succs = ir.cfg.blocks[from].succs
    deleteat!(succs, findfirst(n -> n == to, succs))

    # Remove the `from` block from the `to` block's predecessor list.
    to_blk = ir.cfg.blocks[to]
    preds = to_blk.preds
    deleteat!(preds, findfirst(n -> n == from, preds))

    # Remove the `from` edge from any `PhiNode`s at the start of next blk.
    stmts = stmt(ir.stmts)
    for n in to_blk.stmts
        stmt = stmts[n]
        if stmt isa PhiNode
            edge_index = findfirst(i::Int32 -> i == from, stmt.edges)
            edge_index === nothing && continue
            deleteat!(stmt.edges, edge_index)
            deleteat!(stmt.values, edge_index)
        else
            break
        end
    end
    return nothing
end

"""
    verify_no_constant_gotoifnots(ir::IRCode)

Verify that we have successfully removed all instances of `goto %n if not true` and
`goto %n if not false`, as these can be reduced to simpler nodes (namely, `GotoNode`s or
"fallthrough"s). Moreover, removing them tends to yield performance improvements by reducing
the amount of information Mooncake must keep in its block stacks.

This is essentially just testing functionality for `const_prop_constant_gotoifnots`. This is
usually run each time a rule is compiled, as it is cheap, and because it is hard to
construct a convincing set of test cases which, if passed at test-time, would indicate we
were done.
"""
function verify_no_constant_gotoifnots(ir::IRCode)
    for (n, stmt) in enumerate(stmt(ir.stmts))
        if stmt isa GotoIfNot
            if stmt.cond isa Bool
                println("Constant GotoIfNot found at SSA $n in the following IRCode:")
                dislay(ir)
                println()
                throw(error("Bad IR, see above."))
            end
        end
    end
end
