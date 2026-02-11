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

struct MooncakeCache
    dict::IdDict{Core.MethodInstance,Core.CodeInstance}
end

MooncakeCache() = MooncakeCache(IdDict{Core.MethodInstance,Core.CodeInstance}())

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
    ) where {C,M<:Mode}
        ip = new{C,M}(meta, world, inf_params, opt_params, inf_cache, code_cache, oc_cache)
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
function CC.get(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance, default)
    return get(wvc.cache.dict, mi, default)
end
function CC.getindex(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance)
    return getindex(wvc.cache.dict, mi)
end
function CC.haskey(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance)
    return haskey(wvc.cache.dict, mi)
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

    # invoke the default abstract call to get the default CC.CallMeta.
    ret = @invoke CC.abstract_call_gf_by_type(
        interp::CC.AbstractInterpreter,
        f::Any,
        arginfo::CC.ArgInfo,
        si::CC.StmtInfo,
        atype::Any,
        sv::CC.AbsIntState,
        max_methods::Int,
    )
    argtypes = arginfo.argtypes
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
        # For all applicable method matches, we need to check if any of them could hit a primitive
        any_prim = any_matches_primitive(applicable, C, M, interp.world)
        if any_prim
            @static if VERSION < v"1.12-"
                call = ret::CC.CallMeta
                info = NoInlineCallInfo(call.info, atype)
                # Primitives must remain visible in the caller IR so we can apply their
                # custom rules. Prevent inlining *and* disable const-prop of the return
                # value (otherwise `compact!` can fold the call away entirely, e.g. a
                # primitive whose inferred return is `Const`).
                widen_rt = should_widen_primitive_call_return_type(call, argtypes)
                return rewrap_callmeta(call, info, widen_rt)
            else
                return CC.Future{CC.CallMeta}(
                    ret::CC.Future, interp, sv
                ) do call, interp, sv
                    info = NoInlineCallInfo(call.info, atype)
                    # See comment in the non-Future branch above.
                    widen_rt = should_widen_primitive_call_return_type(call, argtypes)
                    return rewrap_callmeta(call, info, widen_rt)
                end
            end
        end
    end
    ret
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

# Decide whether to widen a primitive call's inferred return type from `CC.Const` to its
# underlying Julia type — e.g. `Const(3.0)` to `Float64`.
#
# `CC.Const(val)` is an element of Julia's extended type lattice that records the exact
# value a computation produces (see `Core.Const` and `Compiler/src/typelattice.jl`).  When
# the compiler sees a call whose return type is `Const`, several optimisation passes can
# replace the call with the literal value and delete the call site entirely:
#
#   - The inliner (`Compiler/src/ssair/inlining.jl`) turns such calls into a `ConstantCase`
#     via `handle_single_case!`, which rewrites `ir[SSAValue(idx)][:stmt]` to the constant.
#   - The `compact!` pass (`Compiler/src/ssair/ir.jl`) maps the SSA name to the literal in
#     its rename table, making the original statement dead.
#
# For Mooncake primitives this is problematic: the call must survive into the final IR so
# that the corresponding `rrule!!` runs during AD.  Widening the return type via
# `CC.widenconst` (defined in `Compiler/src/typelattice.jl` as
# `widenconst(c::Const) = typeof(c.val)`) strips the `Const` marker and prevents folding.
# This may over-widen when a primitive inherently returns a constant regardless of its
# inputs, but distinguishing that from genuine const-prop would require re-running inference.
#
# However, when every runtime argument is itself `Const`, the call is a genuine compile-time
# constant (e.g. `sin(1.0)` where `1.0` is a literal in the source).  Folding is safe here
# because the call can never see different values at runtime, so skipping the `rrule!!`
# loses no derivative information.
#
# `call`: the `CC.CallMeta` returned by Julia's abstract interpretation for this call site.
# `argtypes`: inferred types for all arguments.  Position 1 is the callee; positions 2:end
#   are the actual runtime arguments.
#
# Returns `true` (widen needed) when `call.rt` is `Const` and at least one runtime argument
# is not `Const`.  Returns `false` otherwise.
function should_widen_primitive_call_return_type(call::CC.CallMeta, argtypes::Vector{Any})
    call.rt isa CC.Const || return false
    for n in 2:length(argtypes)
        argtypes[n] isa CC.Const || return true
    end
    return false
end

function rewrap_callmeta(call::CC.CallMeta, info::CC.CallInfo, widen_rt::Bool)
    # If `widen_rt` is true, widen `call.rt` with `CC.widenconst`,  
    # discarding any `Const` information to prevent constant folding;  
    # otherwise, use `call.rt` unchanged.
    #
    # This drops value-level information tracked by inference  
    # (e.g. `Const(3)` → `Int`), ensuring the call is not folded away.  
    # For example, if `call.rt` is inferred as `Const(42)`, widening  
    # yields `Int`, preserving the call so downstream rules still apply.
    rt = widen_rt ? CC.widenconst(call.rt) : call.rt
    @static if VERSION ≥ v"1.11-"
        return CC.CallMeta(rt, call.exct, call.effects, info)
    else
        return CC.CallMeta(rt, call.effects, info)
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
    get_interpreter(mode::Type{<:Mode})

Returns a `MooncakeInterpreter` appropriate for the current world age. Will use a cached
interpreter if one already exists for the current world age, otherwise creates a new one.

This should be prefered over constructing a `MooncakeInterpreter` directly.
"""
function get_interpreter(mode::Type{<:Mode})
    if GLOBAL_INTERPRETERS[mode].world != Base.get_world_counter()
        GLOBAL_INTERPRETERS[mode] = MooncakeInterpreter(DefaultCtx, mode)
    end
    return GLOBAL_INTERPRETERS[mode]
end
