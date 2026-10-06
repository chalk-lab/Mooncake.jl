# AbstractInterpretation -- this is an instance of a Julia AbstractInterpreter. We use it
# in conjunction with the contexts above to decide what should be inlined and what should
# not be inlined. Similar strategies are employed by Enzyme and Diffractor.

# The most important bit of this code is `inlining_policy` (renamed to `src_inlining_policy` in Julia v1.12+) -- the rest is copy + pasted
# boiler plate, largely taken from https://github.com/JuliaLang/julia/blob/2fe4190b3d26b4eee52b2b1b1054ddd6e38a941e/test/compiler/newinterp.jl#L11
#
# Credit: much of the code in here is copied over from the main Julia repo, and from
# Enzyme.jl, which has a very similar set of concerns to Mooncake in terms of avoiding
# inlining primitive functions.
#

struct ClosureCacheKey
    world_age::UInt
    key::Any
end

# `roots` maps each key of an interpreter's `oc_cache` to the `CodeInstance` of the method that
# rule was derived from; Julia invalidates it through backedges when anything the rule inlined
# or called changes (see `successor_interpreter`).
struct MooncakeCache
    dict::IdDict{Core.MethodInstance,Core.CodeInstance}
    roots::Dict{ClosureCacheKey,Core.CodeInstance}
end

function MooncakeCache()
    return MooncakeCache(
        IdDict{Core.MethodInstance,Core.CodeInstance}(),
        Dict{ClosureCacheKey,Core.CodeInstance}(),
    )
end
Base.empty!(c::MooncakeCache) = (empty!(c.dict); empty!(c.roots); c)

# The method table used by `Mooncake.@mooncake_overlay`.
Base.Experimental.@MethodTable mooncake_method_table

struct MooncakeInterpreter{C,M<:Mode} <: CC.AbstractInterpreter
    meta # additional information
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}
    code_cache::MooncakeCache
    oc_cache::Dict{ClosureCacheKey,Any}
    # The world `oc_cache` is keyed on; interpreters sharing both caches share it.
    cache_world::UInt
    function MooncakeInterpreter(
        ::Type{C},
        ::Type{M};
        meta=nothing,
        world::UInt=Base.get_world_counter(),
        inf_params::CC.InferenceParams=CC.InferenceParams(),
        opt_params::CC.OptimizationParams=CC.OptimizationParams(),
        inf_cache::Vector{CC.InferenceResult}=CC.InferenceResult[],
        code_cache::MooncakeCache=MooncakeCache(),
        oc_cache::Dict{ClosureCacheKey,Any}=Dict{ClosureCacheKey,Any}(),
        cache_world::UInt=world,
    ) where {C,M<:Mode}
        ip = new{C,M}(
            meta, world, inf_params, opt_params, inf_cache, code_cache, oc_cache, cache_world
        )
        tts = Any[
            Tuple{typeof(sum),Tuple{Int}},
            Tuple{typeof(sum),Tuple{Int,Int}},
            Tuple{typeof(sum),Tuple{Int,Int,Int}},
            Tuple{typeof(sum),Tuple{Int,Int,Int,Int}},
            Tuple{typeof(sum),Tuple{Int,Int,Int,Int,Int}},
        ]
        for tt in tts
            for m in CC._methods_by_ftype(tt, 10, ip.world)::Vector
                m = m::CC.MethodMatch
                typ = Any[m.spec_types.parameters...]
                for i in 1:length(typ)
                    typ[i] = CC.unwraptv(typ[i])
                end
                CC.typeinf_type(ip, m.method, Tuple{typ...}, m.sparams)
            end
        end
        return ip
    end
end

# Don't print out the IRCode object, because this tends to pollute the REPL. Just make it
# clear that this is a MistyClosure, which contains an OpaqueClosure.
function Base.show(io::IO, mime::MIME"text/plain", mc::MooncakeInterpreter)
    return _show_interp(io, mime, mc)
end
Base.show(io::IO, mc::MooncakeInterpreter) = _show_interp(io, MIME"text/plain"(), mc)

function _show_interp(io::IO, ::MIME"text/plain", ::MooncakeInterpreter{C,M}) where {C,M}
    return print(io, "MooncakeInterpreter($M)")
end

MooncakeInterpreter(M::Type{<:Mode}) = MooncakeInterpreter(DefaultCtx, M)

context_type(::MooncakeInterpreter{C}) where {C} = C

CC.InferenceParams(interp::MooncakeInterpreter) = interp.inf_params
CC.OptimizationParams(interp::MooncakeInterpreter) = interp.opt_params
CC.get_inference_cache(interp::MooncakeInterpreter) = interp.inf_cache
function CC.code_cache(interp::MooncakeInterpreter)
    return CC.WorldView(interp.code_cache, CC.WorldRange(interp.world))
end
# A cached `CodeInstance` serves a lookup only if it is valid over the whole world range asked
# for, as Julia's own cache requires. Caches are shared across world moves, so entries inferred
# in another world (or invalidated since) must miss.
function valid_in_worlds(ci::Core.CodeInstance, worlds::CC.WorldRange)
    return ci.min_world <= first(worlds) && last(worlds) <= ci.max_world
end

function CC.get(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance, default)
    ci = get(wvc.cache.dict, mi, nothing)
    if ci === nothing || !valid_in_worlds(ci, wvc.worlds)
        return default
    end
    return ci
end
function CC.getindex(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance)
    ci = CC.get(wvc, mi, nothing)
    if ci === nothing
        throw(KeyError(mi))
    end
    return ci
end
function CC.haskey(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance)
    return CC.get(wvc, mi, nothing) !== nothing
end
function CC.setindex!(
    wvc::CC.WorldView{MooncakeCache}, ci::Core.CodeInstance, mi::Core.MethodInstance
)
    return setindex!(wvc.cache.dict, ci, mi)
end
function CC.method_table(interp::MooncakeInterpreter)
    return CC.OverlayMethodTable(interp.world, mooncake_method_table)
end

@static if VERSION < v"1.11.0"
    CC.get_world_counter(interp::MooncakeInterpreter) = interp.world
    get_inference_world(interp::CC.AbstractInterpreter) = CC.get_world_counter(interp)
else
    CC.get_inference_world(interp::MooncakeInterpreter) = interp.world
    CC.cache_owner(::MooncakeInterpreter) = nothing
    get_inference_world(interp::CC.AbstractInterpreter) = CC.get_inference_world(interp)
end

struct NoInlineCallInfo <: CC.CallInfo
    info::CC.CallInfo # wrapped call
    tt::Any # signature
end

CC.nsplit_impl(info::NoInlineCallInfo) = CC.nsplit(info.info)
CC.getsplit_impl(info::NoInlineCallInfo, idx::Int) = CC.getsplit(info.info, idx)
CC.getresult_impl(info::NoInlineCallInfo, idx::Int) = CC.getresult(info.info, idx)
@static if VERSION > v"1.12-"
    CC.add_edges_impl(edges::Vector{Any}, info::NoInlineCallInfo) = CC.add_edges!(
        edges, info.info
    )
end

function Core.Compiler.abstract_call_gf_by_type(
    interp::MooncakeInterpreter{C,M},
    @nospecialize(f),
    arginfo::CC.ArgInfo,
    si::CC.StmtInfo,
    @nospecialize(atype),
    sv::CC.AbsIntState,
    max_methods::Int,
) where {C,M}
    argtypes = arginfo.argtypes
    # Look up applicable methods for this call site without recursing into their bodies.
    # We need the method set to check for primitives before deciding how to proceed.
    if VERSION < v"1.12-"
        𝕃ᵢ = Core.Compiler.typeinf_lattice(interp)
        matches = Core.Compiler.find_matching_methods(
            𝕃ᵢ,
            argtypes,
            atype,
            Core.Compiler.method_table(interp),
            Core.Compiler.InferenceParams(interp).max_union_splitting,
            max_methods,
        )
    else
        matches = Core.Compiler.find_method_matches(interp, argtypes, atype; max_methods)
    end
    if !isa(matches, Core.Compiler.FailedMethodMatch)
        (; valid_worlds, applicable) = matches
        # For applicable method matches in IR, we need to check if any of them is a primitive.
        any_prim = any_matches_primitive(applicable, C, M, interp.world)
        if any_prim
            # A primitive already has a hand-written `rrule!!`, so Mooncake does not need
            # to inspect its body when differentiating. The only thing we need here is the
            # ordinary `CallMeta` for the call site, especially the inferred return type.
            #
            # We therefore ask `NativeInterpreter` for the `CallMeta`. This avoids recursing
            # through the callee IR using Mooncake's primitive-search logic:
            # `MooncakeInterpreter` would walk nested calls in that body, check them for
            # primitives, and continue that search down the callee tree. That extra work is
            # unnecessary for a primitive with a hand-written rule.
            #
            # `noinline_callmeta` below then blocks inlining/const-folding so the primitive
            # call stays in the caller IR and Mooncake can dispatch its `rrule!!` at runtime.
            # See PR #1115 for more discussion.
            native_interp = CC.NativeInterpreter(interp.world)
            ret = CC.abstract_call_gf_by_type(
                native_interp, f, arginfo, si, atype, sv, max_methods
            )
            @static if VERSION < v"1.12-"
                call = ret::CC.CallMeta
                # Keep primitives in caller IR by blocking const-folding and inlining
                _call = widen_rettype_callmeta(call, argtypes)
                return noinline_callmeta(_call, atype)
            else
                return CC.Future{CC.CallMeta}(
                    ret::CC.Future, interp, sv
                ) do call, interp, sv
                    _call = widen_rettype_callmeta(call, argtypes)
                    return noinline_callmeta(_call, atype)
                end
            end
        end
    end

    return @invoke CC.abstract_call_gf_by_type(
        interp::CC.AbstractInterpreter,
        f::Any,
        arginfo::CC.ArgInfo,
        si::CC.StmtInfo,
        atype::Any,
        sv::CC.AbsIntState,
        max_methods::Int,
    )
end

function any_matches_primitive(applicable, C, M, world)
    for app in applicable
        if VERSION < v"1.12-"
            sig = app.spec_types
        else
            sig = app.match.spec_types
        end
        if is_primitive(C, M, sig, world)
            return true
        end
    end
    false
end

"""
    widen_rettype_callmeta(call, argtypes)

Decide whether to widen a primitive call’s inferred return type from `CC.Const`
to its underlying Julia type (e.g. `Const(3.0)` → `Float64`).

`CC.Const(val)` represents an exact value in Julia’s extended type lattice
(see `Core.Const` and `Compiler/src/typelattice.jl`). If a call is inferred
as `Const`, later compiler passes may fold it away:

  - The inliner rewrites it to a `ConstantCase`.
  - `compact!` propagates the literal and removes the dead statement.

For Mooncake primitives, the call must remain in the final IR so that the
corresponding `rrule!!` executes during AD. Applying `CC.widenconst`
removes the `Const` wrapper and prevents folding.

Widening is performed only when:
  - the inferred return type is `Const`, and
  - at least one runtime argument (i.e. excluding the callee) is not `Const`.

If all runtime arguments are `Const`, the call is a genuine compile-time
constant (e.g. `sin(1.0)` with a literal argument), and folding is safe.

Arguments:
  - `call`: `CC.CallMeta` for the call site
  - `argtypes`: inferred argument types (1 = callee, 2:end = runtime args)
"""
function widen_rettype_callmeta(call::CC.CallMeta, argtypes::Vector{Any})
    # Check whether any runtime argument is not `Const`
    has_nonconst_runtime_arg = any(i -> !(argtypes[i] isa CC.Const), 2:length(argtypes))

    should_widen = call.rt isa CC.Const && has_nonconst_runtime_arg

    rt = should_widen ? CC.widenconst(call.rt) : call.rt

    @static if VERSION ≥ v"1.11-"
        return CC.CallMeta(rt, call.exct, call.effects, call.info)
    else
        return CC.CallMeta(rt, call.effects, call.info)
    end
end

function noinline_callmeta(call::CC.CallMeta, @nospecialize(atype))
    info = NoInlineCallInfo(call.info, atype)
    @static if VERSION ≥ v"1.11-"
        return CC.CallMeta(call.rt, call.exct, call.effects, info)
    else
        return CC.CallMeta(call.rt, call.effects, info)
    end
end

@static if VERSION < v"1.11-"
    function CC.inlining_policy(
        interp::MooncakeInterpreter{C},
        @nospecialize(src),
        @nospecialize(info::CC.CallInfo),
        stmt_flag::UInt8,
        mi::Core.MethodInstance,
        argtypes::Vector{Any},
    ) where {C}

        # Do not inline away primitives.
        info isa NoInlineCallInfo && return nothing

        # If not a primitive, AD doesn't care about it. Use the usual inlining strategy.
        return @invoke CC.inlining_policy(
            interp::CC.AbstractInterpreter,
            src::Any,
            info::CC.CallInfo,
            stmt_flag::UInt8,
            mi::Core.MethodInstance,
            argtypes::Vector{Any},
        )
    end

elseif VERSION < v"1.12-" # 1.11
    function CC.inlining_policy(
        interp::MooncakeInterpreter,
        @nospecialize(src),
        @nospecialize(info::CC.CallInfo),
        stmt_flag::UInt32,
    )
        # Do not inline away primitives.
        info isa NoInlineCallInfo && return nothing

        # If not a primitive, AD doesn't care about it. Use the usual inlining strategy.
        return @invoke CC.inlining_policy(
            interp::CC.AbstractInterpreter, src::Any, info::CC.CallInfo, stmt_flag::UInt32
        )
    end

else # 1.12 and up.
    function CC.src_inlining_policy(
        interp::MooncakeInterpreter,
        @nospecialize(src),
        @nospecialize(info::CC.CallInfo),
        stmt_flag::UInt32,
    )
        # Do not inline away primitives.
        info isa NoInlineCallInfo && return false

        # If not a primitive, AD doesn't care about it. Use the usual inlining strategy.
        return @invoke CC.src_inlining_policy(
            interp::CC.AbstractInterpreter, src::Any, info::CC.CallInfo, stmt_flag::UInt32
        )
    end
end

"""
    const GLOBAL_INTERPRETERS

Cached interpreters. Should only be accessed via `get_interpreter`.
"""
const GLOBAL_INTERPRETERS = Dict(
    ForwardMode => MooncakeInterpreter(DefaultCtx, ForwardMode),
    ReverseMode => MooncakeInterpreter(DefaultCtx, ReverseMode),
)

"""
    const GLOBAL_CACHE_STAMPS

The `rule_extension_stamp` each mode's cached rules were derived under.
"""
const GLOBAL_CACHE_STAMPS = Dict{Any,Any}(ForwardMode => nothing, ReverseMode => nothing)

# Visits every method of the global method table, recording the newest world in which a method
# of one of `function_types` was added. `methods(f)` costs 5x more on `_is_primitive`.
struct NewestMethodWorld{T<:Tuple}
    function_types::T
    world::Vector{UInt}
end

function (newest::NewestMethodWorld)(method::Method)
    head = Base.unwrap_unionall(method.sig).parameters[1]
    if any(type -> head === type, newest.function_types)
        newest.world[1] = max(newest.world[1], method.primary_world)
    end
    return nothing
end

# Visits every method of a method table, recording the newest world in which one was added.
struct NewestAnyMethodWorld
    world::Vector{UInt}
end

function (newest::NewestAnyMethodWorld)(method::Method)
    newest.world[1] = max(newest.world[1], method.primary_world)
    return nothing
end

"""
    rule_extension_stamp()

What rule derivation reads that Julia's backedges do not track: the newest method of each
function it dispatches through, the newest method of the `@mooncake_overlay` table (which
changes what inference resolves a call to), and the number of loaded packages (whose
extensions add such methods). Cached rules survive a world move only while the stamp is
unchanged.
"""
function rule_extension_stamp()
    functions = (
        frule!!, rrule!!, _is_primitive, tangent_type, build_primitive_frule, build_primitive_rrule
    )
    newest = NewestMethodWorld(map(typeof, functions), UInt[0])
    Base.visit(newest, Core.methodtable)
    newest_overlay = NewestAnyMethodWorld(UInt[0])
    Base.visit(newest_overlay, mooncake_method_table)
    return (only(newest.world), only(newest_overlay.world), length(Base.loaded_modules_array()))
end

"""
    successor_interpreter(old::MooncakeInterpreter)

The interpreter for the current world after `old`'s world has passed. From Julia 1.12, when no
method of an extension point of `rule_extension_stamp` was added, it shares `old`'s caches minus
every inferred method and rule that Julia has since invalidated; otherwise it starts empty.
A rule is valid exactly when the `CodeInstance` of its root method is, because Julia invalidates
that through backedges once anything the rule inlined or called is redefined or gains a method.
"""
function successor_interpreter(old::MooncakeInterpreter{C,M}) where {C,M}
    @static if VERSION >= v"1.12-"
        stamp = rule_extension_stamp()
        if stamp == GLOBAL_CACHE_STAMPS[M]
            cache = old.code_cache
            filter!(p -> p.second.max_world == typemax(UInt), cache.dict)
            filter!(p -> p.second.max_world == typemax(UInt), cache.roots)
            filter!(p -> haskey(cache.roots, p.first), old.oc_cache)
            return MooncakeInterpreter(
                C, M; code_cache=cache, oc_cache=old.oc_cache, cache_world=old.cache_world
            )
        end
        GLOBAL_CACHE_STAMPS[M] = stamp
    end
    return MooncakeInterpreter(C, M)
end

"""
    get_interpreter(mode::Type{<:Mode})

Returns a `MooncakeInterpreter` appropriate for the current world age. Will use a cached
interpreter if one already exists for the current world age, otherwise creates one with
`successor_interpreter`.

This should be prefered over constructing a `MooncakeInterpreter` directly.
"""
function get_interpreter(mode::Type{<:Mode})
    lock(MOONCAKE_INFERENCE_LOCK) do
        if GLOBAL_INTERPRETERS[mode].world != Base.get_world_counter()
            GLOBAL_INTERPRETERS[mode] = successor_interpreter(GLOBAL_INTERPRETERS[mode])
        end
        return GLOBAL_INTERPRETERS[mode]
    end
end

"""
    get_interpreter(mode::Type{<:Mode}, world::UInt)

Returns a `MooncakeInterpreter` for `mode` pinned to `world`. When `world` is the current
world age this is equivalent to `get_interpreter(mode)`. An older `world` inside the current
cache epoch (e.g. a Lazy/Dynamic rule rebuilding at its stored prediction world) shares the
current caches; one before it gets a fresh, uncached interpreter.
"""
function get_interpreter(mode::Type{<:Mode}, world::UInt)
    current = get_interpreter(mode)
    world == current.world && return current
    world < current.cache_world && return MooncakeInterpreter(DefaultCtx, mode; world)
    return MooncakeInterpreter(
        DefaultCtx,
        mode;
        world,
        code_cache=current.code_cache,
        oc_cache=current.oc_cache,
        cache_world=current.cache_world,
    )
end

"""
    root_call(f, args...)

Calls `f(args...)`. Inferring it for a rule's signature gives `register_rule_root!` a
`CodeInstance` whose backedges cover the method lookup of that signature, so a new more
specific method or a redefinition of the method itself invalidates it.
"""
root_call(f, args...) = f(args...)

"""
    rule_cache_key(interp::MooncakeInterpreter, sig_or_mi, debug_mode::Bool, direction::Symbol)

The key of a rule in `interp.oc_cache`. A rule for a dispatch tuple is keyed on the world the
cache began in, so it can be carried across world moves (see `cached_rule`); any other rule
is keyed on `interp.world` and never outlives its world.
"""
function rule_cache_key(
    interp::MooncakeInterpreter, sig_or_mi, debug_mode::Bool, direction::Symbol
)
    world = Base.isdispatchtuple(_get_sig(sig_or_mi)) ? interp.cache_world : interp.world
    return ClosureCacheKey(world, (sig_or_mi, debug_mode, direction))
end

"""
    cached_rule(interp::MooncakeInterpreter, key::ClosureCacheKey)

The rule cached under `key` if it is valid in `interp.world`, else `nothing`. A rule is valid
when its root `CodeInstance` (`register_rule_root!`) is valid in `interp.world`; Julia ends that
`CodeInstance`'s validity through backedges once anything the rule inlined or called changes.
Without a root, a rule is valid only in the world it was derived in.
"""
function cached_rule(interp::MooncakeInterpreter, key::ClosureCacheKey)
    rule = get(interp.oc_cache, key, nothing)
    root = get(interp.code_cache.roots, key, nothing)
    if rule === nothing
        return nothing
    elseif root === nothing
        return key.world_age == interp.world ? rule : nothing
    end
    return valid_in_worlds(root, CC.WorldRange(interp.world)) ? rule : nothing
end

"""
    register_rule_root!(interp::MooncakeInterpreter, key::ClosureCacheKey, sig_or_mi)

Records the root `CodeInstance` of the rule in `interp.oc_cache` under `key`, so that
`cached_rule` and `successor_interpreter` can tell whether the rule outlives a world move. A
rule without a root (Julia before 1.12, or a signature that is not a dispatch tuple) is valid
in its own world only.
"""
function register_rule_root!(interp::MooncakeInterpreter, key::ClosureCacheKey, sig_or_mi)
    delete!(interp.code_cache.roots, key)
    @static if VERSION >= v"1.12-"
        sig = _get_sig(sig_or_mi)
        if Base.isdispatchtuple(sig)
            atype = Tuple{typeof(root_call),sig.parameters...}
            mi = CC.specialize_method(only(methods(root_call)), atype, Core.svec())
            ci = CC.typeinf_ext(interp, mi, CC.SOURCE_MODE_NOT_REQUIRED)
            if ci isa Core.CodeInstance
                interp.code_cache.roots[key] = ci
            end
        end
    end
    return nothing
end

@static if VERSION >= v"1.11-"
    const REBASED_WORLD = Base.ScopedValues.ScopedValue(UInt(0))

    """
        pinned_world(world::UInt)

    The world a copied Lazy/Dynamic rule builds its rules at: the world of the interpreter
    that fetched the copy (`copy_rule_at`), else the world it was derived at.
    """
    pinned_world(world::UInt) = REBASED_WORLD[] == 0 ? world : REBASED_WORLD[]

    """
        copy_rule_at(interp::MooncakeInterpreter, rule)

    `_copy(rule)` whose Lazy/Dynamic rules build at `interp.world`. A rule carried across a
    world move behaves as one freshly derived there: dynamic dispatch sees the methods of the
    fetching world.
    """
    function copy_rule_at(interp::MooncakeInterpreter, rule)
        return Base.ScopedValues.with(() -> _copy(rule), REBASED_WORLD => interp.world)
    end
else
    pinned_world(world::UInt) = world
    copy_rule_at(::MooncakeInterpreter, rule) = _copy(rule)
end

"""
    empty_mooncake_caches!()

This is an internal function and not part of the public API. Called by `prepare_pullback_cache`,
`prepare_gradient_cache`, and `prepare_derivative_cache` when `Config(empty_cache=true)`
is passed.

Empties all three per-interpreter caches for both `ForwardMode` and `ReverseMode`:
- `oc_cache` : compiled `DerivedRule` / `OpaqueClosures`
- `code_cache` : `CodeInstance` objects (Julia IR per `MethodInstance`)
- `inf_cache` : `InferenceResult` objects from type inference

After clearing, Mooncake re-derives rules from scratch on the next use. Only Julia-level
(GC-managed) objects are freed; JIT-compiled native machine code allocated by LLVM
is held permanently by the Julia runtime.
"""
function empty_mooncake_caches!()
    for interp in values(GLOBAL_INTERPRETERS)
        empty!(interp.oc_cache)
        empty!(interp.code_cache)
        empty!(interp.inf_cache)
    end
    return nothing
end
