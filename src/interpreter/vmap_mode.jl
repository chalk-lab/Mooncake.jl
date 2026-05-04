# ── vmap: batch lifting via the same IR machinery as forward mode ─────────────
#
# vmap is a second instantiation of generate_lifted_ir, parameterised by
# VmapMode / batch_type instead of ForwardMode / dual_type.
#
# Public API:
#   vmap(f)          - transform f into a cached VmappedFn (main API, JAX-style)
#   vmap_apply(f,xs) - one-shot: apply f to xs with no caching
#   build_vmap(...)  - low-level: compile a DerivedVmap for a specific sig + mode
#
# Branch strategy: error on divergent branches (batch elements must all take the
# same branch). Masking / dynamic splitting are documented future extensions.

# Forward declaration - methods are added in src/rules/vmap_rules.jl.
# Mirrors the frule!! / rrule!! pattern declared at the top of Mooncake.jl.
function vmap_rule!! end

# ── Struct SoA construction helper ───────────────────────────────────────────
# Called from the lifted IR at every %new(P, f1, f2, …) site where struct_batchable(P).
# Takes the already-batched field values and wraps them in a NamedTuple.
@inline function _construct_struct_batch(::Type{P}, batched_fields...) where P
    nt = NamedTuple{fieldnames(P)}(batched_fields)
    return ismutabletype(P) ? MutableTangent(nt) : Tangent(nt)
end

# Tuple construction: Core.tuple(a, b, …) in the lifted IR → Tangent with :_1,:_2,… names.
# N is fixed at compile time from the type parameters, so ntuple is fully inlined.
@generated function _construct_tuple_batch(batched_fields...)
    N     = length(batched_fields)
    names = ntuple(i -> Symbol(:_, i), N)
    return :(Tangent(NamedTuple{$names}(batched_fields)))
end

# NamedTuple construction: %new(NamedTuple{names,T}, f1, f2, …) in the lifted IR.
# Uses actual field names from the type parameter - unlike _construct_tuple_batch.
@inline function _construct_namedtuple_batch(
    ::Type{P}, batched_fields...
) where {names, T<:Tuple, P<:NamedTuple{names,T}}
    return Tangent(NamedTuple{names}(batched_fields))
end

# ── batch control-flow helper ─────────────────────────────────────────────────

# Extracts a single Bool from a batched condition, erroring if the batch diverges.
@inline function _batch_primal_of(x::AbstractVector{Bool})
    allequal(x) || error(
        "vmap: divergent branch - batch elements took different branches. " *
        "Masking is not yet implemented."
    )
    return x[1]
end
@inline _batch_primal_of(x::Bool) = x   # non-batched constant condition

# ── constant lifting ──────────────────────────────────────────────────────────

# Lift a constant primal value to a batch by replication.
# Returns the raw value if non-batched, or an index into captures otherwise.
function const_batch!(captures::Vector{Any}, stmt, mode::VmapMode)::Union{Any,Int}
    v = get_const_primal_value(stmt)
    batch_type(typeof(v)) == typeof(v) && return v
    x = _make_batch(v, mode.batch_size)
    push!(captures, x)
    return length(captures)
end

# ── DerivedVmap: compiled vmap rule ──────────────────────────────────────────

struct DerivedVmap{primal_sig,Tlifted_oc,isva,nargs}
    lifted_oc::Tlifted_oc
end

@inline function (fwd::DerivedVmap{P,T,isva,nargs})(
    args::Vararg{Any,N}
) where {P,T,N,isva,nargs}
    return fwd.lifted_oc(args...)
end

_copy(x::P) where {P<:DerivedVmap} =
    P(replace_captures(x.lifted_oc, _copy(x.lifted_oc.oc.captures)))

# ── LazyVmap / DynamicVmap: call handlers ────────────────────────────────────
#
# For each function call site in the lifted IR, the function f arrives as:
#   - A plain function object (from a GlobalRef in the primal IR) - NOT batched
#   - Or a Vector{typeof(f)} (if f itself was a batched argument) - batched
#
# In either case, we extract the scalar function and broadcast it over the
# remaining (possibly batched) arguments.  Non-batched scalars (Int, Bool, …)
# broadcast correctly as singletons.

struct LazyVmap
    debug_mode::Bool
    mode::VmapMode
    mi::Core.MethodInstance
end

_copy(x::LazyVmap) = LazyVmap(x.debug_mode, x.mode, x.mi)

@inline function (rule::LazyVmap)(args::Vararg{Any,N}) where {N}
    f = args[1] isa AbstractVector ? first(args[1]) : args[1]
    return _pack_batch(collect(broadcast(f, args[2:end]...)))
end

struct DynamicVmap
    debug_mode::Bool
    mode::VmapMode
end

_copy(x::DynamicVmap) = DynamicVmap(x.debug_mode, x.mode)

@inline function (dv::DynamicVmap)(args::Vararg{Any,N}) where {N}
    f = args[1] isa AbstractVector ? first(args[1]) : args[1]
    data_args = args[2:end]
    # Runtime fallback to vmap_rule!! for cases where compile-time dispatch missed
    # (e.g. getfield on Tangent via a :call node rather than :invoke in the primal IR).
    applicable(vmap_rule!!, f, data_args...) && return vmap_rule!!(f, data_args...)
    return _pack_batch(collect(broadcast(f, data_args...)))
end

# ── VmapInfo: analogue of PrimalRuleInfo ─────────────────────────────────────

struct VmapInfo
    isva::Bool
    nargs::Int
    lifted_ret_type::Type
end

# ── generate_vmap_ir ──────────────────────────────────────────────────────────

"""
    generate_vmap_ir(interp, sig_or_mi, mode::VmapMode; debug_mode=false)

Generate lifted IR for `sig_or_mi` under the vmap transform parameterised by `mode`.
Mirrors `generate_lifted_ir` but uses `batch_type` as the type functor and routes
primitive calls through broadcast rather than `frule!!`.
"""
function generate_vmap_ir(
    interp::MooncakeInterpreter,
    sig_or_mi,
    mode::VmapMode;
    debug_mode=false,
    do_inline=true,
    do_optimize=true,
)
    seed_id!()

    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    nargs = length(primal_ir.argtypes)

    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)

    lifted_ir = CC.copy(primal_ir)

    # Lift argument types via batch_type
    for (a, P) in enumerate(primal_ir.argtypes)
        lifted_ir.argtypes[a] = _lift_type(mode, CC.widenconst(P))
    end
    pushfirst!(lifted_ir.argtypes, Any)  # captures slot

    captures = Any[]
    is_used = characterised_used_ssas(stmt(primal_ir.stmts))
    vinfo = VmapLiftedInfo(primal_ir, interp, is_used, debug_mode, mode)

    for (n, inst) in enumerate(lifted_ir.stmts)
        ssa = SSAValue(n)
        modify_vmap_stmts!(stmt(inst), lifted_ir, ssa, captures, vinfo)
    end

    lifted_ir = CC.compact!(lifted_ir)
    CC.verify_ir(lifted_ir)

    captures_tuple = (captures...,)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)

    lifted_ir = do_optimize ? optimise_ir!(lifted_ir; do_inline) : lifted_ir

    ret_type = _lift_type(mode, compute_ir_rettype(primal_ir))
    return lifted_ir, captures_tuple, VmapInfo(isva, nargs, ret_type)
end

# ── VmapLiftedInfo: carries context through stmt modification ─────────────────

struct VmapLiftedInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
    is_used::Vector{Bool}
    debug_mode::Bool
    mode::VmapMode
end

# ── Statement modification - vmap IR rewrite ──────────────────────────────────

modify_vmap_stmts!(::Nothing, ::IRCode, ::SSAValue, ::Vector{Any}, ::VmapLiftedInfo) = nothing

function modify_vmap_stmts!(
    ::GotoNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::VmapLiftedInfo
)
    nothing
end

function modify_vmap_stmts!(
    stmt::GotoIfNot,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    ::VmapLiftedInfo,
)
    # Extract a single Bool from the (possibly batched) condition, erroring if divergent.
    replace_call!(lifted_ir, ssa, Expr(:call, _batch_primal_of, inc_args(stmt).cond))
    new_gotoifnot = new_inst(Core.GotoIfNot(ssa, stmt.dest))
    CC.insert_node!(lifted_ir, ssa, new_gotoifnot, ATTACH_AFTER)
    return nothing
end

function modify_vmap_stmts!(
    stmt::GlobalRef,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::VmapLiftedInfo,
)
    if isconst(stmt)
        v = get_const_primal_value(stmt)
        T = typeof(v)
        if batch_type(T) == T
            # Non-batched constant: leave as primal value
            replace_call!(lifted_ir, ssa, Expr(:call, identity, v))
        else
            b = _make_batch(v, info.mode.batch_size)
            push!(captures, b)
            replace_call!(
                lifted_ir, ssa, Expr(:call, get_capture, Argument(1), length(captures))
            )
        end
    else
        new_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(stmt), ATTACH_BEFORE)
        replace_call!(lifted_ir, ssa, Expr(:call, identity, new_ssa))
    end
    return nothing
end

function modify_vmap_stmts!(
    stmt::ReturnNode,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::VmapLiftedInfo,
)
    isdefined(stmt, :val) || return nothing
    if stmt.val isa Union{Argument,SSAValue}
        replace_call!(lifted_ir, ssa, ReturnNode(__inc(stmt.val)))
        return nothing
    end
    v = get_const_primal_value(stmt.val)
    T = typeof(v)
    if batch_type(T) == T
        replace_call!(lifted_ir, ssa, ReturnNode(v))
    else
        b = _make_batch(v, info.mode.batch_size)
        push!(captures, b)
        get_b = Expr(:call, get_capture, Argument(1), length(captures))
        get_b_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(get_b), ATTACH_BEFORE)
        replace_call!(lifted_ir, ssa, ReturnNode(get_b_ssa))
    end
    return nothing
end

function modify_vmap_stmts!(
    stmt::PhiNode,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::VmapLiftedInfo,
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        v = get_const_primal_value(stmt.values[n])
        T = typeof(v)
        stmt.values[n] = batch_type(T) == T ? v : _make_batch(v, info.mode.batch_size)
    end
    set_stmt!(lifted_ir, ssa, inc_args(stmt))
    set_ir!(
        lifted_ir, ssa, :type,
        _lift_type(info.mode, CC.widenconst(get_ir(lifted_ir, ssa, :type))),
    )
    return nothing
end

function modify_vmap_stmts!(
    stmt::PiNode,
    lifted_ir::IRCode,
    ssa::SSAValue,
    ::Vector{Any},
    info::VmapLiftedInfo,
)
    v = stmt.val isa Union{Argument,SSAValue} ? __inc(stmt.val) :
        let val = get_const_primal_value(stmt.val)
            batch_type(typeof(val)) == typeof(val) ? val : fill(val, info.mode.batch_size)
        end
    replace_call!(
        lifted_ir, ssa,
        PiNode(v, _lift_type(info.mode, CC.widenconst(stmt.typ))),
    )
    return nothing
end

function modify_vmap_stmts!(
    stmt::UpsilonNode,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::VmapLiftedInfo,
)
    if !(stmt.val isa Union{Argument,SSAValue})
        v = get_const_primal_value(stmt.val)
        T = typeof(v)
        stmt = UpsilonNode(batch_type(T) == T ? v : _make_batch(v, info.mode.batch_size))
    end
    set_stmt!(lifted_ir, ssa, inc_args(stmt))
    set_ir!(
        lifted_ir, ssa, :type,
        _lift_type(info.mode, CC.widenconst(get_ir(lifted_ir, ssa, :type))),
    )
    return nothing
end

function modify_vmap_stmts!(
    stmt::PhiCNode,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::VmapLiftedInfo,
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        v = get_const_primal_value(stmt.values[n])
        T = typeof(v)
        stmt.values[n] = batch_type(T) == T ? v : _make_batch(v, info.mode.batch_size)
    end
    set_stmt!(lifted_ir, ssa, inc_args(stmt))
    set_ir!(
        lifted_ir, ssa, :type,
        _lift_type(info.mode, CC.widenconst(get_ir(lifted_ir, ssa, :type))),
    )
    return nothing
end

@static if isdefined(Core, :EnterNode)
    function modify_vmap_stmts!(
        ::Core.EnterNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::VmapLiftedInfo
    )
        return nothing
    end
end

# ── Compile-time vmap_rule!! resolution ──────────────────────────────────────
#
# WHY COMPILE-TIME (NOT RUNTIME) DISPATCH:
#
# The analogue in forward mode is `is_primitive` + `build_primitive_frule`: at IR
# generation time, we check whether a primitive rule exists for each call site and
# bake the rule directly into the lifted IR's captures. At runtime the captured
# rule is called with zero dispatch overhead.
#
# A runtime `applicable(vmap_rule!!, f, args...)` check in LazyVmap would mean:
#  - One method lookup per call site per batch invocation.
#  - The rule function itself is not inlined into the lifted IR.
#  - No possibility of the compiler specializing on the rule's type.
#
# The compile-time path instead:
#  1. `_try_get_const_function`: extract f from the primal IR (GlobalRef / QuoteNode).
#     If f is not a compile-time constant (e.g. a closure stored in a variable), return
#     nothing - fall through to LazyVmap/DynamicVmap broadcast.
#  2. `_compile_time_vmap_rule`: compute the lifted argument types, run `hasmethod`
#     (compile-time method table lookup), and if found return Base.Fix1(vmap_rule!!, f).
#  3. The returned Fix1 is pushed into `captures` and referenced via `get_capture`.
#     The emitted IR calls it with only the batch data arguments - f is baked in.
#
# Result: zero runtime dispatch overhead; the compiler sees a fully specialized call.

# Try to extract a constant function value from a primal IR argument.
# Returns Some(f) if resolvable at IR-generation time, nothing otherwise.
function _try_get_const_function(arg)
    arg isa GlobalRef && isconst(arg) && return Some(getglobal(arg.mod, arg.name))
    arg isa QuoteNode  && return Some(arg.value)
    return nothing
end

# At IR-generation time, check whether vmap_rule!! has a method for (f, lifted_arg_types...).
# If yes, return Base.Fix1(vmap_rule!!, f) - the function is fixed at compile time, so the
# emitted IR only carries the batch args and calls the rule with zero runtime overhead.
# If no, return nothing - caller falls back to LazyVmap/DynamicVmap broadcast path.
function _compile_time_vmap_rule(f_maybe, raw_data_args, primal_ir, mode)
    f_maybe === nothing && return nothing
    f = something(f_maybe)
    primal_types = map(raw_data_args) do x
        CC.widenconst(get_forward_primal_type(primal_ir, x))
    end
    lifted_types = map(T -> _lift_type(mode, T), primal_types)
    hasmethod(vmap_rule!!, Tuple{typeof(f), lifted_types...}) || return nothing
    return Base.Fix1(vmap_rule!!, f)
end

# ── Expression handling ───────────────────────────────────────────────────────

function modify_vmap_stmts!(
    stmt::Expr,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::VmapLiftedInfo,
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        args = map(__inc, raw_args)
        mi = isexpr(stmt, :invoke) ? get_mi(stmt.args[1]) : missing

        # Attempt compile-time dispatch to vmap_rule!!.
        # raw_args[1] is the function; raw_args[2:end] are the data arguments.
        f_maybe = _try_get_const_function(raw_args[1])
        rule = _compile_time_vmap_rule(f_maybe, raw_args[2:end], info.primal_ir, info.mode)

        if rule !== nothing
            # Primitive path: bake the rule (with function fixed) into captures.
            # Emit a call with only the batch data args - no runtime function extraction.
            push!(captures, rule)
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            replace_call!(lifted_ir, ssa, Expr(:call, rule_ssa, args[2:end]...))
        else
            # Fallback path: LazyVmap (known invoke) or DynamicVmap (dynamic call).
            dm = info.debug_mode
            push!(
                captures,
                isexpr(stmt, :invoke) ? LazyVmap(dm, info.mode, mi) : DynamicVmap(dm, info.mode),
            )
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            replace_call!(lifted_ir, ssa, Expr(:call, rule_ssa, args...))
        end

    elseif isexpr(stmt, :new)
        # Struct construction: %new(P, field1, field2, …).
        # If struct_batchable(P): transform into _construct_struct_batch(P, batched_fields…)
        # which returns NamedTuple{fieldnames(P)}((batched_fields…)).
        # getfield on the resulting NamedTuple then returns each field's BatchContainer
        # directly - zero-copy, O(1), with no special IR handling needed.
        P_ref = stmt.args[1]
        P_val = if P_ref isa GlobalRef && isconst(P_ref)
            getglobal(P_ref.mod, P_ref.name)
        elseif P_ref isa Type
            P_ref
        else
            nothing
        end
        if P_val !== nothing && P_val <: Tuple
            # Tuple: Core.tuple(a,b,…) - positional names :_1,:_2,… baked by _construct_tuple_batch.
            field_args = map(__inc, stmt.args[2:end])
            replace_call!(lifted_ir, ssa, Expr(:call, _construct_tuple_batch, field_args...))
        elseif P_val !== nothing && P_val <: NamedTuple
            # NamedTuple: uses actual field names from the type parameter.
            field_args = map(__inc, stmt.args[2:end])
            push!(captures, P_val)
            get_P = Expr(:call, get_capture, Argument(1), length(captures))
            P_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(get_P), ATTACH_BEFORE)
            replace_call!(
                lifted_ir, ssa,
                Expr(:call, _construct_namedtuple_batch, P_ssa, field_args...),
            )
        elseif P_val !== nothing && struct_batchable(P_val)
            field_args = map(__inc, stmt.args[2:end])
            push!(captures, P_val)
            get_P   = Expr(:call, get_capture, Argument(1), length(captures))
            P_ssa   = CC.insert_node!(lifted_ir, ssa, new_inst(get_P), ATTACH_BEFORE)
            replace_call!(
                lifted_ir, ssa,
                Expr(:call, _construct_struct_batch, P_ssa, field_args...),
            )
        else
            throw(ArgumentError(
                "vmap: struct construction for $(P_val) is not supported. " *
                "Mark it with `@struct_batch $(P_val)` to enable SoA batching, " *
                "or add a vmap_rule!! for the containing function."
            ))
        end

    elseif isexpr(stmt, :boundscheck)
        inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        bc_ssa = CC.insert_node!(lifted_ir, ssa, inst, ATTACH_BEFORE)
        replace_call!(lifted_ir, ssa, Expr(:call, identity, bc_ssa))

    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(lifted_ir, ssa, nothing)

    elseif Meta.isexpr(stmt, :copyast)
        new_inst_ = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        new_ssa = CC.insert_node!(lifted_ir, ssa, new_inst_, ATTACH_BEFORE)
        replace_call!(lifted_ir, ssa, Expr(:call, identity, new_ssa))

    elseif Meta.isexpr(stmt, :loopinfo)
        # leave alone

    elseif isexpr(stmt, :throw_undef_if_not)
        primal_cond = Expr(:call, _batch_primal_of, inc_args(stmt).args[2])
        replace_call!(lifted_ir, ssa, primal_cond)
        CC.insert_node!(
            lifted_ir, ssa,
            new_inst(Expr(:throw_undef_if_not, stmt.args[1], ssa)),
            ATTACH_AFTER,
        )

    elseif isexpr(stmt, :enter) || isexpr(stmt, :leave) || isexpr(stmt, :pop_exception)
        # leave alone

    else
        throw(ArgumentError(
            "vmap: expressions of type `:$(stmt.head)` are not yet supported"
        ))
    end
    return nothing
end

# ── build_vmap ────────────────────────────────────────────────────────────────

"""
    build_vmap(interp, sig_or_mi, mode::VmapMode; debug_mode=false)

Compile a batched version of the function described by `sig_or_mi`.
Returns a `DerivedVmap` callable that accepts batched (`Vector`) arguments.
"""
function build_vmap(
    interp::MooncakeInterpreter,
    sig_or_mi,
    mode::VmapMode;
    debug_mode=false,
)
    vmap_ir, captures, info = generate_vmap_ir(interp, sig_or_mi, mode; debug_mode)
    vmap_oc = misty_closure(
        info.lifted_ret_type, vmap_ir, captures...; do_compile=true
    )
    sig = _get_sig(sig_or_mi)
    isva = info.isva
    nargs = info.nargs
    return DerivedVmap{sig,typeof(vmap_oc),isva,nargs}(vmap_oc)
end

# ── Public vmap API ───────────────────────────────────────────────────────────

"""
    VmappedFn{F}

A cached, callable wrapper produced by `vmap(f)`.

The first call for a given `(element_type, batch_size)` pair compiles a `DerivedVmap`
and stores it in an internal cache. All subsequent calls with the same combination reuse
the compiled rule - zero recompilation overhead.

Construct via `vmap(f)`, not directly.
"""
struct VmappedFn{F}
    f::F
    cache::Dict{Any, Any}
    lock::ReentrantLock
end

VmappedFn(f) = VmappedFn(f, Dict{Any, Any}(), ReentrantLock())

function (vf::VmappedFn{F})(xs::AbstractVector{T}) where {F, T}
    N = length(xs)
    N == 0 && return similar(xs, Any, 0)
    rule = lock(vf.lock) do
        get!(vf.cache, (T, N)) do
            build_vmap(get_interpreter(PrimalMode), Tuple{F, T}, VmapMode(N))
        end
    end
    batched_xs = batch_type(T) == T ? xs : _wrap_input(xs)
    return rule(fill(vf.f, N), batched_xs)
end

"""
    vmap(f) → VmappedFn

Transform `f` into a batched function that applies `f` independently to each element
of a `Vector` input. The compiled IR is cached by `(element_type, batch_size)` and
reused on every subsequent call - compile once, run many times.

Branches within `f` must be uniform across the batch; divergent branches throw an error.

# Example
```julia
sq = vmap(x -> x^2)
sq([1.0, 2.0, 3.0])   # [1.0, 4.0, 9.0]  - compiles on first call
sq([4.0, 5.0, 6.0])   # [16.0, 25.0, 36.0] - reuses compiled rule
sq([1.0f0, 2.0f0])    # recompiles once for Float32, then caches
```
"""
vmap(f) = VmappedFn(f)

"""
    vmap_apply(f, xs)

Apply `f` to each element of `xs` without caching the compiled rule.
Every call recompiles. Prefer `vmap(f)` when calling `f` more than once.
"""
function vmap_apply(f, xs::AbstractVector{T}) where {T}
    N = length(xs)
    N == 0 && return similar(xs, Any, 0)
    rule = build_vmap(get_interpreter(PrimalMode), Tuple{typeof(f), T}, VmapMode(N))
    batched_xs = batch_type(T) == T ? xs : _wrap_input(xs)
    return rule(fill(f, N), batched_xs)
end

"""
    _wrap_input(xs::AbstractVector) → BatchContainer or Tangent

Pack a user-supplied vector of inputs into the canonical SoA form expected by a compiled
`DerivedVmap`. Dispatches on the element type:

- `_VmapScalar` (Float32/64, Complex): contiguous `BatchContainer` backed by a dense array
- `AbstractArray{<:_VmapScalar}`: same-device `BatchContainer` via `similar` (GPU-safe)
- `Tuple` / `NamedTuple`: `Tangent` of per-field `BatchContainer`s (`@generated`)
- `@struct_batch` struct: `Tangent`/`MutableTangent` of per-field `BatchContainer`s
- Anything else: returned unchanged (treated as AoS; no batching)
"""
_wrap_input(xs::AbstractVector{T}) where {T<:_VmapScalar} =
    BatchContainer{T, Vector{T}}(Vector{T}(xs), 1)
_wrap_input(xs::AbstractVector{<:AbstractVector{T}}) where {T<:_VmapScalar} =
    BatchContainer{Vector{T}, Matrix{T}}(reduce(hcat, xs), 2)
_wrap_input(xs::AbstractVector{<:AbstractMatrix{T}}) where {T<:_VmapScalar} =
    BatchContainer{Matrix{T}, Array{T,3}}(cat(xs...; dims=3), 3)
_wrap_input(xs::AbstractVector{Array{T,N}}) where {T<:_VmapScalar, N} =
    BatchContainer{Array{T,N}, Array{T,N+1}}(cat(xs...; dims=N+1), N+1)
# Non-_VmapScalar dense arrays: same SoA layout, no BLAS rules. LazyVmap handles ops.
# The T<:_VmapScalar method above is more specific for float arrays.
_wrap_input(xs::AbstractVector{Array{T,N}}) where {T, N} =
    BatchContainer{Array{T,N}, Array{T,N+1}}(cat(xs...; dims=N+1), N+1)
# AbstractArray subtypes (sparse, static, GPU arrays, …): copy into a same-device
# backing via `similar`. Dense Array methods above are more specific and take precedence
# when eltype is Array. `similar` preserves the array type (CuArray stays on GPU).
# NOTE: full GPU correctness requires a package extension that also fixes batch_type
# for CuArray types; this only fixes the allocation path.
function _wrap_input(xs::AbstractVector{P}) where {E<:_VmapScalar, P<:AbstractArray{E}}
    isempty(xs) && return xs
    K = ndims(first(xs))
    data = similar(first(xs), E, size(first(xs))..., length(xs))
    for (i, x) in enumerate(xs)
        selectdim(data, K+1, i) .= x
    end
    return BatchContainer{P, typeof(data)}(data, K+1)
end
# Tuples: transpose N tuples into a Tangent of per-element BatchContainers.
# @generated so field count and positional names are resolved at code-generation time,
# making each recursive _wrap_input call type-stable.
@generated function _wrap_input(xs::AbstractVector{P}) where {P <: Tuple}
    !isconcretetype(P) && return :(xs)
    nfields = fieldcount(P)
    nfields == 0 && return :(Tangent(NamedTuple{()}(())))
    names = ntuple(i -> Symbol(:_, i), nfields)
    field_exprs = ntuple(nfields) do i
        FT = fieldtype(P, i)
        :(_wrap_input(_vmap_getfield(xs, Val($i))::Vector{$FT}))
    end
    return :(Tangent(NamedTuple{$names}(tuple($(field_exprs...)))))
end

# NamedTuples: same as Tuple but uses actual field names from the type parameter.
@generated function _wrap_input(xs::AbstractVector{P}) where {names, T<:Tuple, P<:NamedTuple{names,T}}
    nfields = length(names)
    nfields == 0 && return :(Tangent(NamedTuple{$names}(())))
    field_exprs = ntuple(nfields) do i
        FT = fieldtype(P, i)
        :(_wrap_input(_vmap_getfield(xs, Val($i))::Vector{$FT}))
    end
    return :(Tangent(NamedTuple{$names}(tuple($(field_exprs...)))))
end

# Struct SoA: transpose N structs into Tangent/MutableTangent of per-field BatchContainers.
# Must be a regular function - struct_batchable is user-extensible via @struct_batch and
# can be added after Mooncake loads. @generated bodies run at the world of the @generated
# definition and would never see user-registered structs.
function _wrap_input(xs::AbstractVector{P}) where P
    if struct_batchable(P)
        names   = fieldnames(P)
        batched = ntuple(i -> _wrap_input([getfield(x, i) for x in xs]), fieldcount(P))
        nt = NamedTuple{names}(batched)
        return ismutabletype(P) ? MutableTangent(nt) : Tangent(nt)
    end
    return xs
end
