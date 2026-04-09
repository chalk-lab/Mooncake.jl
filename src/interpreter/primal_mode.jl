#
# Roadmap note for `build_primal`.
#
# `build_primal` is the active IR-lifted forward path. The core plan is:
# 1. resolve and normalise primal IR for `f`
# 2. rebuild `f` as a primal/lifted executable rather than redispatching on AD wrappers
# 3. recursively use `build_primal(g, ...)` for resolved non-primitive callees so dispatch
#    is preserved through the resolved call graph
# 4. keep primitive / intrinsic / builtin leaves on explicit primitive rule paths
# 5. keep an explicit fallback only for genuinely unresolved dynamic calls
#
# The recursion / caching side of that problem is already the same one handled by
# `LazyPrimal`, `DynamicPrimal`, and `DerivedPrimal`; the interesting design work is
# representation choice and the lifted execution boundary.
#
# Routes we considered:
# - keep the lifted path fully on Mooncake `Dual` plus width-aware tangent storage
# - make the lifted path generic over `dual_type(::Val{N}, P)` so scalar / complex cases
#   can choose `NDual` directly
# - overload `tangent_type(::Val{N}, ...)` for packed layouts rather than changing dual type
# - use cached internal `Array{NDual}` work buffers behind callable rule objects for array
#   lowering, instead of trying to make array public dual types themselves be `Array{NDual}`
# - previously, lower some primitive sites through separate nfwd bridge types; that route was
#   removed in favour of Mooncake-owned primitive wrapper entrypoints
#
# Candidate width-aware tangent layouts we considered:
# - today's lane-outer `NTangent`, i.e. a tuple of ordinary tangents
# - a packed `NTangent` with lane-inner scalar leaves closer to `NDual`
# - overloading `tangent_type(::Val{N}, ...)` only for `Array{<:IEEEFloat}` /
#   `Array{<:Complex{<:IEEEFloat}}` so array leaves use NDual-like packed storage
# - overloading `tangent_type(::Val{N}, ...)` only for those arrays so tangents use one
#   extra lane dimension instead
#
# Current choices:
# - the active lifted IR path constructs and types values through
#   `dual_type(::Val{N}, P)` / `tangent_type(::Val{N}, P)`, so width-aware scalar and
#   complex paths can use `NDual` directly while structured values still default to
#   ordinary Mooncake `Dual` unless users choose a different `dual_type` overload
# - the public boundary is generic over width-aware dual types via `dual_type(::Val{N}, P)`
# - direct nfwd integration now lives behind Mooncake-owned primitive wrappers and
#   width-aware dual-type overloads, not a separate `NfwdMooncake` runtime include
# - primitive wrappers may accept top-level `NDual` directly where that path is explicit,
#   but the general lifted path does not proactively lower everything to `NDual`
#
# The remaining larger design direction is to make the lifted path itself generic over a
# width-aware dual-type protocol, rather than assuming concrete `Dual{P,T}` internally. A
# minimal construction / extraction protocol would be:
# - `primal(x)`
# - `tangent(x)`
# - `dual_type(::Val{N}, ::Type{P})`
# - `zero_dual(::Val{N}, x)`
# - `uninit_dual(::Val{N}, x)`
# - construction via `dual_type(Val(N), P)(x, dx)`, which may yield `Dual`, `NDual`, or
#   another width-aware dual type
# Public top-level forward entrypoints currently validate inputs with
# `verify_dual_type(x)`, but that check is auto-derived from the protocol above in the
# common case.
#
# That keeps width-aware dual-type choice type-driven and inference-friendly while
# separating dual-type selection from tangent-storage selection. The key type-stability
# constraint is that `dual_type(Val(N), P)` must remain foldable to a concrete dual type
# for concrete `P`; runtime dual-type switching would immediately weaken inference through
# the transformed IR.
#
# Still open:
# - broader array/container lowering if we want cached internal `Array{NDual}` execution
# - future reuse of this machinery for batched / vmap-style transforms
# - keeping the shared generated forward corpus green as the lifted path evolves

# Check if a type contains Union{} (bottom type) anywhere in its structure.
# This can happen with unreachable code or failed type inference.
@inline contains_bottom_type(T) = _contains_bottom_type(T, Base.IdSet{Any}())

function _contains_bottom_type(T, seen::Base.IdSet{Any})
    T === Union{} && return true
    if T isa Union
        return _contains_bottom_type(T.a, seen) || _contains_bottom_type(T.b, seen)
    elseif T isa TypeVar
        T in seen && return false
        push!(seen, T)
        return _contains_bottom_type(T.ub, seen)
    elseif T isa UnionAll
        T in seen && return false
        push!(seen, T)
        return _contains_bottom_type(T.body, seen)
    elseif T isa DataType
        T in seen && return false
        push!(seen, T)
        for p in T.parameters
            _contains_bottom_type(p, seen) && return true
        end
        return false
    else
        return false
    end
end

function _nfwd_primitive_frule_call end
function _nfwd_primitive_rrule_call end

@inline build_primitive_frule(
    ::Any, interp, sig; debug_mode=false, silence_debug_messages=true
) = nothing

@inline function _lifted_dual_ir_type(tangent_mode, P)
    T = CC.widenconst(P)
    return isconcretetype(T) ? dual_type(tangent_mode, T) : Any
end

function build_frule(
    args...;
    chunk_size=nothing,
    debug_mode=false,
    silence_debug_messages=true,
    skip_world_age_check=false,
)
    primals = map(x -> verify_dual_type(x) ? primal(x) : x, args)
    sig = _typeof(primals)
    interp = get_interpreter(ForwardMode)
    mode = Val(something(chunk_size, 1))
    return build_frule(
        interp,
        sig;
        call_target=first(primals),
        debug_mode,
        silence_debug_messages,
        skip_world_age_check,
        tangent_mode=mode,
    )
end

"""
    build_frule(
        interp::MooncakeInterpreter{C},
        sig_or_mi;
        call_target=nothing,
        debug_mode=false,
        silence_debug_messages=true,
        skip_world_age_check=false,
        tangent_mode=nothing,
    ) where {C}

Returns a callable which performs forward-mode AD for `sig_or_mi`.

`tangent_mode` selects the width-aware dual type used to derive the rule. The active
lifted-forward path keys on width, so `Val(N)` is the canonical form.
"""
function build_frule(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    call_target=nothing,
    debug_mode=false,
    silence_debug_messages=true,
    skip_world_age_check=false,
    tangent_mode=nothing,
) where {C}
    @nospecialize sig_or_mi

    tangent_mode === nothing && (tangent_mode = Val(1))

    if !skip_world_age_check && Base.get_world_counter() > interp.world
        throw(
            ArgumentError(
                "World age associated to interp is behind current world age. Please " *
                "create a new interpreter for the current world age.",
            ),
        )
    end

    if !silence_debug_messages && debug_mode
        @info "Compiling frule for $sig_or_mi in debug mode. Disable for best performance."
    end

    sig = _get_sig(sig_or_mi)
    if is_primitive(C, ForwardMode, sig, interp.world)
        rule = build_primitive_frule(
            tangent_mode, interp, sig; debug_mode, silence_debug_messages
        )
        isnothing(rule) && (rule = build_primitive_frule(sig))
        return debug_mode ? DebugFRule(rule) : rule
    end

    return build_primal(
        interp, sig_or_mi; call_target, debug_mode, skip_world_age_check=true, tangent_mode
    )
end

"""
    build_primal(
        interp::MooncakeInterpreter{C},
        sig_or_mi;
        skip_world_age_check=false,
        do_inline=true,
    ) where {C}

Build a callable from resolved, normalised primal IR. The result can be called directly on
ordinary primal inputs, or on forward `Dual` inputs as an IR-lifted alternative to a
derived forward rule.
"""
function build_primal(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    call_target=nothing,
    debug_mode=false,
    skip_world_age_check=false,
    do_inline=true,
    tangent_mode=Val(1),
) where {C}
    @nospecialize sig_or_mi

    if !skip_world_age_check && Base.get_world_counter() > interp.world
        throw(
            ArgumentError(
                "World age associated to interp is behind current world age. Please " *
                "create a new interpreter for the current world age.",
            ),
        )
    end

    lock(MOONCAKE_INFERENCE_LOCK)
    try
        oc_cache_key = ClosureCacheKey(
            interp.world, (sig_or_mi, :primal, do_inline, tangent_mode)
        )
        if haskey(interp.oc_cache, oc_cache_key)
            cached = _copy(interp.oc_cache[oc_cache_key])
            derived = DerivedPrimal(
                cached.primal_oc,
                cached.lifted_oc,
                cached.key,
                call_target,
                if isnothing(call_target)
                    nothing
                else
                    zero_dual(cached.tangent_mode, call_target)
                end,
                cached.isva,
                cached.nargs,
                cached.world,
                cached.tangent_mode,
            )
            return debug_mode ? DebugPrimal(derived) : derived
        end

        primal_ir, info, captures = generate_primal_ir(interp, sig_or_mi; do_inline)
        primal_oc = misty_closure(
            info.ret_type, primal_ir, captures...; isva=info.isva, do_compile=true
        )
        lifted_ir, lifted_captures, dual_ret_type = generate_lifted_primal_ir(
            interp, sig_or_mi; do_inline, tangent_mode
        )
        lifted_oc = misty_closure(
            dual_ret_type, lifted_ir, lifted_captures...; do_compile=true
        )
        derived = DerivedPrimal(
            primal_oc,
            lifted_oc,
            sig_or_mi,
            nothing,
            nothing,
            info.isva,
            info.nargs,
            interp.world,
            tangent_mode,
        )
        interp.oc_cache[oc_cache_key] = derived
        derived = DerivedPrimal(
            primal_oc,
            lifted_oc,
            sig_or_mi,
            call_target,
            isnothing(call_target) ? nothing : zero_dual(tangent_mode, call_target),
            info.isva,
            info.nargs,
            interp.world,
            tangent_mode,
        )
        return debug_mode ? DebugPrimal(derived) : derived
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

function build_primal(
    args...;
    debug_mode=false,
    skip_world_age_check=false,
    do_inline=true,
    tangent_mode=Val(1),
)
    primals = map(x -> verify_dual_type(x) ? primal(x) : x, args)
    sig = _typeof(primals)
    interp = get_interpreter(ForwardMode)
    return build_primal(
        interp,
        sig;
        call_target=first(primals),
        debug_mode,
        skip_world_age_check,
        do_inline,
        tangent_mode,
    )
end

function primal_type(interp::MooncakeInterpreter{C}, sig_or_mi; tangent_mode) where {C}
    primal_ir, info, captures = generate_primal_ir(interp, sig_or_mi)
    lifted_ir, lifted_captures, dual_ret_type = generate_lifted_primal_ir(
        interp, sig_or_mi; tangent_mode
    )
    Tprimal_oc = MistyClosure{
        Core.OpaqueClosure{
            compute_oc_signature(primal_ir, length(primal_ir.argtypes) - 1, info.isva),
            info.ret_type,
        },
    }
    Tlifted_oc = MistyClosure{
        Core.OpaqueClosure{
            compute_oc_signature(lifted_ir, length(lifted_ir.argtypes) - 1, false),
            dual_ret_type,
        },
    }
    return DerivedPrimal{
        Tprimal_oc,
        Tlifted_oc,
        typeof(sig_or_mi),
        Nothing,
        Nothing,
        info.isva,
        info.nargs,
        typeof(tangent_mode),
    }
end

struct PrimalIRInfo
    isva::Bool
    nargs::Int
    ret_type::Type
end

struct DualRuleInfo
    isva::Bool
    nargs::Int
    dual_ret_type::Type
end

struct PrimalInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
end

struct DualInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
    is_used::Vector{Bool}
    debug_mode::Bool
    tangent_mode
end

struct DerivedPrimal{
    Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode
}
    primal_oc::Tprimal_oc
    lifted_oc::Tlifted_oc
    key::Tkey
    call_target::Tcall_target
    lifted_call_target::Tlifted_call_target
    isva::Bool
    nargs::Int
    world::UInt
    tangent_mode::Tmode
end

function DerivedPrimal(
    primal_oc,
    lifted_oc,
    key,
    call_target,
    lifted_call_target,
    isva::Bool,
    nargs::Int,
    world::UInt,
    tangent_mode,
)
    return DerivedPrimal{
        typeof(primal_oc),
        typeof(lifted_oc),
        typeof(key),
        typeof(call_target),
        typeof(lifted_call_target),
        isva,
        nargs,
        typeof(tangent_mode),
    }(
        primal_oc,
        lifted_oc,
        key,
        call_target,
        lifted_call_target,
        isva,
        nargs,
        world,
        tangent_mode,
    )
end

const DerivedFRule = DerivedPrimal

"""
    DebugPrimal(primal)

Construct a callable equivalent to `primal` but with additional type checking on lifted
width-aware dual-type calls. Ordinary primal calls are forwarded unchanged.
"""
struct DebugPrimal{Tprimal}
    primal::Tprimal
end

_copy(x::P) where {P<:DebugPrimal} = P(_copy(x.primal))

@inline function (primal::DebugPrimal)(x::Vararg{Any,N}) where {N}
    any(verify_dual_type, x) &&
        !all(verify_dual_type, x) &&
        throw(
            ArgumentError(
                "DebugPrimal does not support mixing width-aware dual-type and primal inputs " *
                "in one call.",
            ),
        )
    return if all(verify_dual_type, x)
        verify_args(primal.primal, x)
        verify_dual_inputs(x)
        y = primal.primal(x...)
        verify_dual_output(x, y)
        y
    else
        primal.primal(x...)
    end
end

mutable struct LazyPrimal{primal_sig,Tprimal,Tmode}
    key::Core.MethodInstance
    world::UInt
    tangent_mode::Tmode
    primal::Tprimal
    function LazyPrimal(key::Core.MethodInstance, world::UInt, tangent_mode=Val(1))
        interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world)
        Tprimal = _lazy_primal_type(interp, key, tangent_mode)
        return new{key.specTypes,Tprimal,typeof(tangent_mode)}(key, world, tangent_mode)
    end
    function LazyPrimal{Tprimal_sig,Tprimal,Tmode}(
        key::Core.MethodInstance, world::UInt, tangent_mode::Tmode
    ) where {Tprimal_sig,Tprimal,Tmode}
        return new{Tprimal_sig,Tprimal,Tmode}(key, world, tangent_mode)
    end
end

const LAZY_PRIMAL_TYPE_LOCK = ReentrantLock()
const ACTIVE_LAZY_PRIMAL_TYPES = IdDict{Core.MethodInstance,Nothing}()

function _lazy_primal_type(
    interp::MooncakeInterpreter, key::Core.MethodInstance, tangent_mode
)
    recursive = lock(LAZY_PRIMAL_TYPE_LOCK) do
        haskey(ACTIVE_LAZY_PRIMAL_TYPES, key) && return true
        ACTIVE_LAZY_PRIMAL_TYPES[key] = nothing
        return false
    end
    recursive && return Any
    try
        return primal_type(interp, key; tangent_mode)
    catch
        return Any
    finally
        lock(LAZY_PRIMAL_TYPE_LOCK) do
            delete!(ACTIVE_LAZY_PRIMAL_TYPES, key)
        end
    end
end

function LazyPrimal(key::Type{<:Tuple}, world::UInt, tangent_mode=Val(1))
    min = Base.RefValue{UInt}(typemin(UInt))
    max = Base.RefValue{UInt}(typemax(UInt))
    ms = Base._methods_by_ftype(
        key, nothing, 1, world, true, min, max, Ptr{Int32}(C_NULL)
    )::Vector
    mm = first(ms)::Core.MethodMatch
    mi = CC.specialize_method(mm.method, mm.spec_types, mm.sparams)
    return LazyPrimal(mi, world, tangent_mode)
end

struct DynamicPrimal{V}
    cache::V
    world::UInt
    tangent_mode
end

function DynamicPrimal(world::UInt, tangent_mode=Val(1))
    DynamicPrimal(Dict{Any,Any}(), world, tangent_mode)
end

function generate_primal_ir(
    interp::MooncakeInterpreter, sig_or_mi; do_inline=true
)::Tuple{IRCode,PrimalIRInfo,Tuple}
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    nargs = length(primal_ir.argtypes)
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)
    lifted_ir = CC.copy(primal_ir)
    pushfirst!(lifted_ir.argtypes, Any)
    captures = Any[]
    info = PrimalInfo(primal_ir, interp)
    for (n, inst) in enumerate(lifted_ir.stmts)
        modify_primal_stmts!(stmt(inst), lifted_ir, SSAValue(n), captures, info)
    end
    lifted_ir = CC.compact!(lifted_ir)
    captures_tuple = (captures...,)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)
    lifted_ir = optimise_ir!(lifted_ir; do_inline)
    return (
        lifted_ir, PrimalIRInfo(isva, nargs, compute_ir_rettype(lifted_ir)), captures_tuple
    )
end

@inline get_capture(captures::T, n::Int) where {T} = captures[n]

"""
    const_dual!(captures::Vector{Any}, stmt)

Build a width-aware dual type from `stmt`, with zero / uninitialised tangent. If the
resulting dual is a bits type, return it directly. Otherwise push it into `captures` and
return its location.
"""
function const_dual!(captures::Vector{Any}, stmt, tangent_mode=Val(1))
    v = get_const_primal_value(stmt)
    x = uninit_dual(tangent_mode, v)
    if safe_for_literal(v)
        return x
    else
        push!(captures, x)
        return length(captures)
    end
end

const ATTACH_AFTER = true
const ATTACH_BEFORE = false

modify_fwd_ad_stmts!(::Nothing, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo) = nothing

modify_fwd_ad_stmts!(::GotoNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo) = nothing

function modify_fwd_ad_stmts!(
    stmt::GotoIfNot, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    replace_call!(dual_ir, ssa, Expr(:call, _primal, inc_args(stmt).cond))
    new_gotoifnot_inst = new_inst(Core.GotoIfNot(ssa, stmt.dest))
    CC.insert_node!(dual_ir, ssa, new_gotoifnot_inst, ATTACH_AFTER)
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::GlobalRef, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if isconst(stmt)
        d = const_dual!(captures, stmt, info.tangent_mode)
        if d isa Int
            replace_call!(dual_ir, ssa, Expr(:call, get_capture, Argument(1), d))
        else
            replace_call!(dual_ir, ssa, Expr(:call, identity, d))
        end
    else
        new_ssa = CC.insert_node!(dual_ir, ssa, new_inst(stmt), ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, Expr(:call, zero_dual, info.tangent_mode, new_ssa))
    end

    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::ReturnNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    isdefined(stmt, :val) || return nothing
    if stmt.val isa Union{Argument,SSAValue}
        replace_call!(dual_ir, ssa, ReturnNode(__inc(stmt.val)))
        return nothing
    end
    d = const_dual!(captures, stmt.val, info.tangent_mode)
    if d isa Int
        get_dual = Expr(:call, get_capture, Argument(1), d)
        get_dual_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_dual), ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, ReturnNode(get_dual_ssa))
    else
        replace_call!(dual_ir, ssa, ReturnNode(d))
    end
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = uninit_dual(
            info.tangent_mode, get_const_primal_value(stmt.values[n])
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        _lifted_dual_ir_type(info.tangent_mode, get_ir(dual_ir, ssa, :type)),
    )
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PiNode, dual_ir::IRCode, ssa::SSAValue, ::Vector{Any}, info::DualInfo
)
    stmt == PiNode(nothing, Union{}) && return replace_call!(dual_ir, ssa, nothing)
    if stmt.val isa Union{Argument,SSAValue}
        v = __inc(stmt.val)
        primal_ssa = CC.insert_node!(
            dual_ir,
            ssa,
            new_inst(Expr(:call, getfield, v, QuoteNode(:primal))),
            ATTACH_BEFORE,
        )
        tangent_ssa = CC.insert_node!(
            dual_ir,
            ssa,
            new_inst(Expr(:call, getfield, v, QuoteNode(:tangent))),
            ATTACH_BEFORE,
        )
        refined_primal_ssa = CC.insert_node!(
            dual_ir,
            ssa,
            new_inst(PiNode(primal_ssa, CC.widenconst(stmt.typ))),
            ATTACH_BEFORE,
        )
        refined_type = CC.widenconst(stmt.typ)
        if isconcretetype(refined_type)
            replace_call!(
                dual_ir,
                ssa,
                Expr(
                    :call,
                    dual_type(info.tangent_mode, refined_type),
                    refined_primal_ssa,
                    tangent_ssa,
                ),
            )
        else
            replace_call!(dual_ir, ssa, Expr(:call, Dual, refined_primal_ssa, tangent_ssa))
        end
    else
        v = uninit_dual(info.tangent_mode, get_const_primal_value(stmt.val))
        replace_call!(
            dual_ir, ssa, PiNode(v, _lifted_dual_ir_type(info.tangent_mode, stmt.typ))
        )
    end
    set_ir!(dual_ir, ssa, :type, _lifted_dual_ir_type(info.tangent_mode, stmt.typ))
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::UpsilonNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if !(stmt.val isa Union{Argument,SSAValue})
        stmt = UpsilonNode(uninit_dual(info.tangent_mode, get_const_primal_value(stmt.val)))
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        _lifted_dual_ir_type(info.tangent_mode, get_ir(dual_ir, ssa, :type)),
    )
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiCNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = uninit_dual(
            info.tangent_mode, get_const_primal_value(stmt.values[n])
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        _lifted_dual_ir_type(info.tangent_mode, get_ir(dual_ir, ssa, :type)),
    )
    return nothing
end

@static if isdefined(Core, :EnterNode)
    function modify_fwd_ad_stmts!(
        ::Core.EnterNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo
    )
        return nothing
    end
end

function modify_fwd_ad_stmts!(
    stmt::Expr, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        sig_types = map(raw_args) do x
            t = CC.widenconst(get_forward_primal_type(info.primal_ir, x))
            return contains_bottom_type(t) ? Any : t
        end
        sig = Tuple{sig_types...}
        args = map(__inc, raw_args)

        if !info.is_used[ssa.id] && get_const_primal_value(args[1]) == getfield
            fwds = new_inst(Expr(:call, __fwds_pass_no_ad!, args...))
            replace_call!(dual_ir, ssa, fwds)
            return nothing
        end

        if all(T -> tangent_type(T) === NoTangent, @view(sig_types[2:end])) &&
            !is_primitive(context_type(info.interp), ForwardMode, sig, info.interp.world)
            primal_ssa = CC.insert_node!(
                dual_ir,
                ssa,
                new_inst(Expr(:call, __fwds_pass_no_ad!, args...)),
                ATTACH_BEFORE,
            )
            replace_call!(
                dual_ir, ssa, Expr(:call, zero_dual, info.tangent_mode, primal_ssa)
            )
            return nothing
        end

        dual_args = map(args) do arg
            arg isa Union{Argument,SSAValue} && return arg
            return uninit_dual(info.tangent_mode, get_const_primal_value(arg))
        end

        if is_primitive(context_type(info.interp), ForwardMode, sig, info.interp.world)
            rule = build_primitive_frule(
                info.tangent_mode, info.interp, sig; debug_mode=info.debug_mode
            )
            isnothing(rule) && (rule = build_primitive_frule(sig))
            if safe_for_literal(rule)
                replace_call!(dual_ir, ssa, Expr(:call, rule, dual_args...))
            else
                push!(captures, rule)
                get_rule = Expr(:call, get_capture, Argument(1), length(captures))
                rule_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
                replace_call!(dual_ir, ssa, Expr(:call, rule_ssa, dual_args...))
            end
        else
            push!(
                captures,
                if isexpr(stmt, :invoke)
                    LazyPrimal(get_mi(stmt.args[1]), info.interp.world, info.tangent_mode)
                else
                    DynamicPrimal(info.interp.world, info.tangent_mode)
                end,
            )
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            replace_call!(dual_ir, ssa, Expr(:call, rule_ssa, dual_args...))
        end
    elseif isexpr(stmt, :boundscheck)
        inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        bc_ssa = CC.insert_node!(dual_ir, ssa, inst, ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, Expr(:call, zero_dual, info.tangent_mode, bc_ssa))
    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(dual_ir, ssa, nothing)
    elseif Meta.isexpr(stmt, :copyast)
        new_copyast_inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        new_copyast_ssa = CC.insert_node!(dual_ir, ssa, new_copyast_inst, ATTACH_BEFORE)
        replace_call!(
            dual_ir, ssa, Expr(:call, zero_dual, info.tangent_mode, new_copyast_ssa)
        )
    elseif Meta.isexpr(stmt, :loopinfo)
    elseif isexpr(stmt, :throw_undef_if_not)
        primal_cond = Expr(:call, _primal, inc_args(stmt).args[2])
        replace_call!(dual_ir, ssa, primal_cond)
        new_undef_inst = new_inst(Expr(:throw_undef_if_not, stmt.args[1], ssa))
        CC.insert_node!(dual_ir, ssa, new_undef_inst, ATTACH_AFTER)
    elseif isexpr(stmt, :enter) || isexpr(stmt, :leave) || isexpr(stmt, :pop_exception)
    else
        msg = "Expressions of type `:$(stmt.head)` are not yet supported in forward mode"
        throw(ArgumentError(msg))
    end
    return nothing
end

get_forward_primal_type(ir::CC.IRCode, a::Argument) = ir.argtypes[a.n]
get_forward_primal_type(ir::CC.IRCode, ssa::SSAValue) = get_ir(ir, ssa, :type)
get_forward_primal_type(::CC.IRCode, x::QuoteNode) = _typeof(x.value)
get_forward_primal_type(::CC.IRCode, x) = _typeof(x)
function get_forward_primal_type(::CC.IRCode, x::GlobalRef)
    return isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.binding.ty
end
function get_forward_primal_type(::CC.IRCode, x::Expr)
    x.head === :boundscheck && return Bool
    return error("Unrecognised expression $x found in argument slot.")
end

function modify_primal_stmts!(
    stmt, primal_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::PrimalInfo
)
    set_stmt!(primal_ir, ssa, inc_args(stmt))
end

@static if isdefined(Core, :EnterNode)
    function modify_primal_stmts!(
        stmt::Core.EnterNode,
        primal_ir::IRCode,
        ssa::SSAValue,
        captures::Vector{Any},
        info::PrimalInfo,
    )
        set_stmt!(primal_ir, ssa, stmt)
    end
end

function modify_primal_stmts!(
    stmt::Expr, primal_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::PrimalInfo
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        sig_types = map(raw_args) do x
            t = CC.widenconst(get_forward_primal_type(info.primal_ir, x))
            contains_bottom_type(t) ? Any : t
        end
        sig = Tuple{sig_types...}
        interp = info.interp
        if is_primitive(context_type(interp), ForwardMode, sig, interp.world)
            set_stmt!(primal_ir, ssa, inc_args(stmt))
        else
            push!(
                captures,
                if isexpr(stmt, :invoke)
                    LazyPrimal(get_mi(stmt.args[1]), interp.world)
                else
                    DynamicPrimal(interp.world, Val(1))
                end,
            )
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(primal_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            args = map(__inc, raw_args)
            replace_call!(primal_ir, ssa, Expr(:call, rule_ssa, args...))
        end
    else
        set_stmt!(primal_ir, ssa, inc_args(stmt))
    end
    return nothing
end

@inline function (primal::DerivedPrimal)(args::Vararg{Any,N}) where {N}
    any(verify_dual_type, args) &&
        !all(verify_dual_type, args) &&
        throw(
            ArgumentError(
                "DerivedPrimal does not support mixing width-aware dual-type and primal " *
                "inputs in one call.",
            ),
        )
    if all(verify_dual_type, args)
        internal_args = _convert_width_aware_dual_args(primal, args)
        lifted_args = if isnothing(primal.call_target)
            _prepare_lifted_runtime_args(primal, internal_args)
        elseif length(args) >= 2 && Mooncake.primal(first(args)) isa Function
            _prepare_lifted_runtime_args(primal, Base.tail(internal_args))
        else
            _prepare_lifted_runtime_args(primal, internal_args)
        end
        y = if isnothing(primal.call_target)
            _call_lifted_oc(primal, lifted_args...)
        else
            _call_lifted_oc(primal, primal.lifted_call_target, lifted_args...)
        end
        return _canonicalize_width_aware_dual(y)
    end
    return if isnothing(primal.call_target)
        _call_primal_oc(primal, args...)
    else
        _call_primal_oc(primal, primal.call_target, args...)
    end
end

@inline function _convert_width_aware_dual_arg(primal::DerivedPrimal, x)
    p = Mooncake.primal(x)
    D = dual_type(primal.tangent_mode, typeof(p))
    N = Val(typeof(primal.tangent_mode).parameters[1])
    tx = canonicalize_chunked_tangent(p, tangent(x), N)
    if typeof(x) === D && typeof(tangent(x)) === typeof(tx)
        return x
    end
    return D(p, tx)
end

@inline function _convert_width_aware_dual_args(primal::DerivedPrimal, args)
    return tuple_map(Base.Fix1(_convert_width_aware_dual_arg, primal), args)
end

@inline function _stored_primal_runtime_nargs(
    ::DerivedPrimal{
        Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode
    },
) where {Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode}
    return Nargs - 1
end

@generated function _prepare_lifted_runtime_args(
    primal::DerivedPrimal{
        Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode
    },
    args::NTuple{M,Any},
) where {Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode,M}
    Isva || return :(args)

    group_start = Tcall_target === Nothing ? Nargs : Nargs - 1
    prefix_args = [:(args[$n]) for n in 1:(group_start - 1)]
    grouped_primal = Expr(:tuple, [:(Mooncake.primal(args[$n])) for n in group_start:M]...)
    grouped_tangent = Expr(:tuple, [:(tangent(args[$n])) for n in group_start:M]...)
    width = Tmode.parameters[1]
    grouped_dual = quote
        grouped_primal = $grouped_primal
        grouped_tangent = $grouped_tangent
        dual_type(primal.tangent_mode, typeof(grouped_primal))(
            grouped_primal,
            canonicalize_chunked_tangent(grouped_primal, grouped_tangent, Val($width)),
        )
    end
    return Expr(:tuple, prefix_args..., grouped_dual)
end

@inline _call_primal_oc(primal::DerivedPrimal, args...) = primal.primal_oc.oc(args...)
@inline _call_lifted_oc(primal::DerivedPrimal, args...) = primal.lifted_oc.oc(args...)

@generated function _call_stored_dual_primal(
    primal::DerivedPrimal{
        Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode
    },
    x::Dual,
    args::Vararg{Dual,N},
) where {Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode,N}
    converted_args = Expr(
        :tuple,
        :(_convert_width_aware_dual_arg(primal, x)),
        [:(_convert_width_aware_dual_arg(primal, args[$n])) for n in 1:N]...,
    )
    direct_call = Expr(
        :call,
        :(primal.lifted_oc.oc),
        :(primal.lifted_call_target),
        :(_convert_width_aware_dual_arg(primal, x)),
        [:(_convert_width_aware_dual_arg(primal, args[$n])) for n in 1:N]...,
    )
    N == 0 &&
        Nargs == 1 &&
        return quote
            isnothing(primal.call_target) && throw(
                ArgumentError(
                    "DerivedPrimal without a stored call target cannot be called on Dual arguments directly. Use LazyPrimal or DynamicPrimal with the primal callable as the first argument.",
                ),
            )
            return _canonicalize_width_aware_dual(
                _call_lifted_oc(primal, primal.lifted_call_target)
            )
        end
    !Isva && return quote
        isnothing(primal.call_target) && throw(
            ArgumentError(
                "DerivedPrimal without a stored call target cannot be called on Dual arguments directly. Use LazyPrimal or DynamicPrimal with the primal callable as the first argument.",
            ),
        )
        return _canonicalize_width_aware_dual($direct_call)
    end
    return quote
        isnothing(primal.call_target) && throw(
            ArgumentError(
                "DerivedPrimal without a stored call target cannot be called on Dual arguments directly. Use LazyPrimal or DynamicPrimal with the primal callable as the first argument.",
            ),
        )
        if $N == 0 && _stored_primal_runtime_nargs(primal) == 0
            return _canonicalize_width_aware_dual(
                _call_lifted_oc(primal, primal.lifted_call_target)
            )
        end
        internal_args = _prepare_lifted_runtime_args(primal, $converted_args)
        return _canonicalize_width_aware_dual(
            _call_lifted_oc(primal, primal.lifted_call_target, internal_args...)
        )
    end
end

@generated function _call_stored_dual_primal(
    primal::DerivedPrimal{
        Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode
    },
    f::Dual,
    x::Dual,
    args::Vararg{Dual,N},
) where {Tprimal_oc,Tlifted_oc,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode,N}
    converted_tail = Expr(
        :tuple,
        :(_convert_width_aware_dual_arg(primal, x)),
        [:(_convert_width_aware_dual_arg(primal, args[$n])) for n in 1:N]...,
    )
    direct_tail_call = Expr(
        :call,
        :(primal.lifted_oc.oc),
        :(primal.lifted_call_target),
        :(_convert_width_aware_dual_arg(primal, x)),
        [:(_convert_width_aware_dual_arg(primal, args[$n])) for n in 1:N]...,
    )
    !Isva && return quote
        isnothing(primal.call_target) && throw(
            ArgumentError(
                "DerivedPrimal without a stored call target cannot be called on Dual arguments directly. Use LazyPrimal or DynamicPrimal with the primal callable as the first argument.",
            ),
        )
        return _canonicalize_width_aware_dual($direct_tail_call)
    end
    return quote
        isnothing(primal.call_target) && throw(
            ArgumentError(
                "DerivedPrimal without a stored call target cannot be called on Dual arguments directly. Use LazyPrimal or DynamicPrimal with the primal callable as the first argument.",
            ),
        )
        internal_args = _prepare_lifted_runtime_args(primal, $converted_tail)
        return _canonicalize_width_aware_dual(
            _call_lifted_oc(primal, primal.lifted_call_target, internal_args...)
        )
    end
end

@generated function _call_dynamic_dual_primal(
    primal::DerivedPrimal{Tprimal_oc,Tlifted_oc,Tkey,Nothing,Nothing,Isva,Nargs,Tmode},
    f::Dual,
    x::Dual,
    args::Vararg{Dual,N},
) where {Tprimal_oc,Tlifted_oc,Tkey,Isva,Nargs,Tmode,N}
    converted_args = Expr(
        :tuple,
        :(_convert_width_aware_dual_arg(primal, f)),
        :(_convert_width_aware_dual_arg(primal, x)),
        [:(_convert_width_aware_dual_arg(primal, args[$n])) for n in 1:N]...,
    )
    direct_call = Expr(
        :call,
        :(primal.lifted_oc.oc),
        :(_convert_width_aware_dual_arg(primal, f)),
        :(_convert_width_aware_dual_arg(primal, x)),
        [:(_convert_width_aware_dual_arg(primal, args[$n])) for n in 1:N]...,
    )
    !Isva && return quote
        return _canonicalize_width_aware_dual($direct_call)
    end
    return quote
        _canonicalize_width_aware_dual(
            _call_lifted_oc(
                primal, _prepare_lifted_runtime_args(primal, $converted_args)...
            ),
        )
    end
end

@inline function (primal::DerivedPrimal)(x::Dual, args::Vararg{Dual,N}) where {N}
    isnothing(primal.call_target) && throw(
        ArgumentError(
            "DerivedPrimal without a stored call target cannot be called on Dual arguments directly. Use LazyPrimal or DynamicPrimal with the primal callable as the first argument.",
        ),
    )
    return _call_stored_dual_primal(primal, x, args...)
end

@inline function (primal::DerivedPrimal)(f::Dual, x::Dual, args::Vararg{Dual,N}) where {N}
    return if isnothing(primal.call_target)
        _call_dynamic_dual_primal(primal, f, x, args...)
    else
        _call_stored_dual_primal(primal, f, x, args...)
    end
end

@static if VERSION < v"1.11-"
    @inline function __call_rule(
        primal::DerivedPrimal{
            Tprimal_oc,
            MistyClosure{OpaqueClosure{A,R}},
            Tkey,
            Tcall_target,
            Tlifted_call_target,
            Isva,
            Nargs,
            Tmode,
        },
        args,
    ) where {Tprimal_oc,A,R,Tkey,Tcall_target,Tlifted_call_target,Isva,Nargs,Tmode}
        return primal(args...)::R
    end

    @inline __call_rule(primal::DebugPrimal, args) = primal(args...)
    @inline __call_rule(primal::LazyPrimal, args) = primal(args...)
    @inline __call_rule(primal::DynamicPrimal, args) = primal(args...)
end

function _copy(x::P) where {P<:DerivedPrimal}
    return P(
        replace_captures(x.primal_oc, _copy(x.primal_oc.oc.captures)),
        replace_captures(x.lifted_oc, _copy(x.lifted_oc.oc.captures)),
        x.key,
        x.call_target,
        x.lifted_call_target,
        x.isva,
        x.nargs,
        x.world,
        x.tangent_mode,
    )
end

function verify_args(primal::DerivedPrimal, x)
    sig = primal.key isa Core.MethodInstance ? primal.key.specTypes : primal.key
    sig isa Type{<:Tuple} || return nothing
    Tx = if isnothing(primal.call_target)
        Tuple{map(_typeof ∘ Mooncake.primal, x)...}
    else
        Tuple{_typeof(primal.call_target),map(_typeof ∘ Mooncake.primal, x)...}
    end
    Tx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tx do not subtype primal signature, $sig"))
end

function __verify_sig(primal::DerivedPrimal, fx::Tuple)
    sig = primal.key isa Core.MethodInstance ? primal.key.specTypes : primal.key
    sig isa Type{<:Tuple} || return nothing
    Tx = Tuple{map(_typeof ∘ Mooncake.primal, fx)...}
    Tx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tx do not subtype primal signature, $sig"))
end

__verify_sig(primal::DebugPrimal, fx::Tuple) = __verify_sig(primal.primal, fx)

@inline function __value_and_gradient!!(
    rule::Union{DerivedPrimal,DebugPrimal,LazyPrimal,DynamicPrimal}, fx::Vararg{CoDual,N}
) where {N}
    return _irfwd_value_and_gradient!!(rule, 1, fx...)
end

@inline function __value_and_pullback!!(
    rule::Union{DerivedPrimal,DebugPrimal,LazyPrimal,DynamicPrimal},
    ȳ::T,
    fx::Vararg{CoDual,N};
    y_cache=nothing,
) where {T<:IEEEFloat,N}
    y, gradient = _irfwd_value_and_gradient!!(rule, 1, fx...)
    value = if y_cache === nothing
        _copy_output(y)
    else
        _copy_to_output!!(y_cache, y)
    end
    return value, tuple_map(g -> g isa NoTangent ? g : _scale(ȳ, g), gradient)
end

@inline function (primal::LazyPrimal)(args::Vararg{Any,N}) where {N}
    return if isdefined(primal, :primal)
        primal.primal(args...)
    else
        _build_primal!(primal, args)
    end
end

@noinline function _build_primal!(primal::LazyPrimal{sig,Tprimal}, args) where {sig,Tprimal}
    interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=primal.world)
    primal.primal = build_primal(
        interp, primal.key; skip_world_age_check=true, tangent_mode=primal.tangent_mode
    )::Tprimal
    return primal.primal(args...)
end

_copy(x::P) where {P<:LazyPrimal} = P(x.key, x.world, x.tangent_mode)

@inline function (dynamic::DynamicPrimal)(f, args::Vararg{Any,N}) where {N}
    verify_dual_type(f) &&
        !all(verify_dual_type, args) &&
        throw(
            ArgumentError(
                "DynamicPrimal does not support mixing width-aware dual-type and primal " *
                "inputs in one call.",
            ),
        )
    any(verify_dual_type, args) &&
        !all(verify_dual_type, args) &&
        throw(
            ArgumentError(
                "DynamicPrimal does not support mixing width-aware dual-type and primal " *
                "data inputs in one call.",
            ),
        )
    sig = Tuple{
        map(x -> verify_dual_type(x) ? _typeof(primal(x)) : _typeof(x), (f, args...))...
    }
    rule = get(dynamic.cache, sig, nothing)
    if rule === nothing
        interp = MooncakeInterpreter(DefaultCtx, ForwardMode; world=dynamic.world)
        rule = build_primal(
            interp, sig; skip_world_age_check=true, tangent_mode=dynamic.tangent_mode
        )
        dynamic.cache[sig] = rule
    end
    return rule(f, args...)
end

_copy(x::P) where {P<:DynamicPrimal} = P(Dict{Any,Any}(), x.world, x.tangent_mode)

function dual_ret_type(primal_ir::IRCode, tangent_mode=Val(1))
    return dual_type(tangent_mode, compute_ir_rettype(primal_ir))
end

function generate_dual_ir(
    interp::MooncakeInterpreter,
    sig_or_mi;
    debug_mode=false,
    do_inline=true,
    tangent_mode=Val(1),
)
    lifted_ir, captures, dual_ret = generate_lifted_primal_ir(
        interp, sig_or_mi; do_inline, tangent_mode
    )
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)
    return lifted_ir, captures, DualRuleInfo(isva, length(primal_ir.argtypes), dual_ret)
end

function build_primitive_frule(
    tangent_mode, interp, sig::Type{<:Tuple}; debug_mode=false, silence_debug_messages=true
)
    _ = tangent_mode
    _ = interp
    _ = sig
    _ = debug_mode
    _ = silence_debug_messages
    # The active lifted-primal path now keeps primitive execution on Mooncake's ordinary
    # `Dual` + width-aware tangent representation. Avoid lowering to the packed NDual path
    # here unless the execution is already operating on NDuals directly.
    return nothing
end

function generate_lifted_primal_ir(
    interp::MooncakeInterpreter, sig_or_mi; do_inline=true, tangent_mode=Val(1)
)
    seed_id!()
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)
    lifted_ir = CC.copy(primal_ir)
    for (a, P) in enumerate(primal_ir.argtypes)
        lifted_ir.argtypes[a] = _lifted_dual_ir_type(tangent_mode, P)
    end
    pushfirst!(lifted_ir.argtypes, Any)
    captures = Any[]
    info = DualInfo(
        primal_ir,
        interp,
        characterised_used_ssas(stmt(primal_ir.stmts)),
        false,
        tangent_mode,
    )
    for (n, inst) in enumerate(lifted_ir.stmts)
        modify_lifted_primal_stmts!(stmt(inst), lifted_ir, SSAValue(n), captures, info)
    end
    lifted_ir = CC.compact!(lifted_ir)
    CC.verify_ir(lifted_ir)
    captures_tuple = (captures...,)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)
    lifted_ir = optimise_ir!(lifted_ir; do_inline)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)
    for (a, P) in enumerate(primal_ir.argtypes)
        lifted_ir.argtypes[a + 1] = _lifted_dual_ir_type(tangent_mode, P)
    end
    return lifted_ir, captures_tuple, dual_ret_type(primal_ir, tangent_mode)
end

function modify_lifted_primal_stmts!(stmt, lifted_ir, ssa, captures, info)
    modify_fwd_ad_stmts!(stmt, lifted_ir, ssa, captures, info)
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:primal_mode})
    # Keep a tiny always-green helper-driven set here; the broader interpreter
    # corpus is exercised explicitly in `test/interpreter/primal_mode.jl`.
    f(x) = sin(x) + x
    g_complex(z) = real(z * z + cos(z))
    test_cases = Any[
        (false, :none, nothing, f, 1.0), (false, :none, nothing, g_complex, 1.0 + 2.0im)
    ]
    return test_cases, Any[]
end

function derived_rule_test_cases(rng_ctor, ::Val{:primal_mode})
    # The shared generated interpreter corpus is run directly from the primal-mode
    # test file, so this hook intentionally stays empty.
    return Any[], Any[]
end
