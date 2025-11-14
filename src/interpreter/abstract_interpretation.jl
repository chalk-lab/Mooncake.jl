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

function _show_interp(io::IO, ::MIME"text/plain", ::MooncakeInterpreter)
    return print(io, "MooncakeInterpreter()")
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
    is_primitive(C, M, atype) || return ret
    # Insert a `NoInlineCallInfo` to prevent any potential inlining.
    @static if VERSION < v"1.12-"
        call = ret::CC.CallMeta
        info = NoInlineCallInfo(call.info, atype)
        return rewrap_callmeta(call, info)
    else
        return CC.Future{CC.CallMeta}(ret::CC.Future, interp, sv) do call, interp, sv
            info = NoInlineCallInfo(call.info, atype)
            return rewrap_callmeta(call, info)
        end
    end
end

function rewrap_callmeta(call::CC.CallMeta, info::CC.CallInfo)
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
