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

function build_frule(
    args...; debug_mode=false, silence_debug_messages=true, chunk_size=1, nfwd::Bool=true
)
    sig = _typeof(TestUtils.__get_primals(args))
    interp = get_interpreter(ForwardMode)
    return build_frule(interp, sig; debug_mode, silence_debug_messages, chunk_size, nfwd)
end

struct DualRuleInfo
    isva::Bool
    nargs::Int
    dual_ret_type::Type
end

"""
    build_frule(
        interp::MooncakeInterpreter{C},
        sig_or_mi;
        debug_mode=false,
        silence_debug_messages=true,
        skip_world_age_check=false,
    ) where {C}

Returns a function which performs forward-mode AD for `sig_or_mi`. Will derive a rule if
`sig_or_mi` is not a primitive.

Set `skip_world_age_check=true` when the interpreter's world age is intentionally older
than the current world (e.g. when building rules for a MistyClosure, which uses its own
world, or when a Lazy/Dynamic rule rebuilds at its stored prediction world; see issue #1218).
"""
function build_frule(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    debug_mode=false,
    silence_debug_messages=true,
    skip_world_age_check=false,
    chunk_size::Int=1,
    nfwd::Bool=true,
) where {C}
    @nospecialize sig_or_mi

    # To avoid segfaults, ensure that we bail out if the interpreter's world age is greater
    # than the current world age.
    if !skip_world_age_check && Base.get_world_counter() > interp.world
        throw(
            ArgumentError(
                "World age associated to interp is behind current world age. Please " *
                "create a new interpreter for the current world age.",
            ),
        )
    end

    # If we're compiling in debug mode, let the user know by default.
    if !silence_debug_messages && debug_mode
        @info "Compiling frule for $sig_or_mi in debug mode. Disable for best performance."
    end

    # If we have a hand-coded rule, just use that.
    sig = _get_sig(sig_or_mi)
    if is_primitive(C, ForwardMode, sig, interp.world)
        rule = build_primitive_frule(sig)
        return debug_mode ? DebugFRule(rule) : rule
    end

    # If the whole function is nfwd-safe, run it directly on the inner dual values (bypassing the
    # per-op `Lifted`/frule transform envelope) rather than deriving a rule. This is the default;
    # `nfwd=false` or `debug_mode` keeps the fully-checked transform path.
    if nfwd && !debug_mode && _nfwd_safe(Any[sig.parameters...], chunk_size)
        return NfwdFRule{chunk_size}()
    end

    # We don't have a hand-coded rule, so derive one.
    lock(MOONCAKE_INFERENCE_LOCK)
    try
        # If we've already derived the OpaqueClosures and info, do not re-derive, just
        # create a copy and pass in new shared data.
        oc_cache_key = ClosureCacheKey(
            interp.world, (sig_or_mi, debug_mode, :forward, chunk_size)
        )
        if haskey(interp.oc_cache, oc_cache_key)
            # Mirror reverse-mode `build_derived_rrule`: return an independent copy so each
            # retrieval gets its own mutable OpaqueClosure-capture state (a `DynamicFRule`'s
            # `cache::Dict`, a `LazyFRule`'s rebuilt `rule`). Returning the shared cached
            # object would race under threads / nested AD, which is exactly what the forward
            # `_copy` machinery (and reverse's eager copy) exist to prevent.
            return _copy(interp.oc_cache[oc_cache_key])
        else
            # Derive forward-pass IR, and shove in a `MistyClosure`.
            dual_ir, captures, info = generate_dual_ir(
                interp, sig_or_mi; debug_mode, chunk_width=chunk_size
            )
            dual_oc = misty_closure(
                info.dual_ret_type, dual_ir, captures...; do_compile=true
            )
            sig = flatten_va_sig(sig, info.isva, info.nargs)
            raw_rule = DerivedFRule{sig,typeof(dual_oc),info.isva,info.nargs}(dual_oc)
            rule = debug_mode ? DebugFRule(raw_rule) : raw_rule
            interp.oc_cache[oc_cache_key] = rule
            return rule
        end
    catch e
        rethrow(e)
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

struct DerivedFRule{primal_sig,Tfwd_oc,isva,nargs}
    fwd_oc::Tfwd_oc
end

# Invoke the wrapped OpaqueClosure (`fwd_oc.oc`) directly rather than the `MistyClosure`
# wrapper — the wrapper carries tangent metadata used elsewhere, not on this call path. The
# call goes through `__call_rule`: on Julia 1.10 the `OpaqueClosure` method routes via
# `jl_apply_generic` (no specsig OC call, avoiding the julia#51016/#61368 codegen segfaults)
# behind an argument-type guard; on Julia 1.11+ it is a direct specsig call.
@inline function (fwd::DerivedFRule{P,sig,isva,nargs})(
    args::Vararg{Lifted,N}
) where {P,sig,N,isva,nargs}
    return __call_rule(fwd.fwd_oc.oc, __unflatten_dual_varargs(isva, args, Val(nargs)))
end

# On Julia 1.10 the call above goes through the dynamic `__call_rule` barrier and is inferred as
# `Any`; assert the rule's return type `R` (encoded in the MistyClosure type parameter) to
# restore type stability for callers.
@static if VERSION < v"1.11-"
    @inline function __call_rule(
        rule::DerivedFRule{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}, args
    ) where {P,A,R,isva,nargs}
        return rule(args...)::R
    end
end

# Copy forward rule with recursively copied captures
function _copy(x::P) where {P<:DerivedFRule}
    return P(replace_captures(x.fwd_oc, _copy(x.fwd_oc.oc.captures)))
end

_isva(::DerivedFRule{P,T,isva,nargs}) where {P,T,isva,nargs} = isva
_nargs(::DerivedFRule{P,T,isva,nargs}) where {P,T,isva,nargs} = nargs

# Extends functionality defined in debug_mode.jl.
function verify_args(r::DerivedFRule{sig}, x) where {sig}
    Tx = Tuple{
        map(_typeof ∘ primal, __unflatten_dual_varargs(_isva(r), x, Val(_nargs(r))))...
    }
    Tx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tx do not subtype rule signature, $sig"))
end

@inline _lifted_width(::Lifted{P,N}) where {P,N} = N

"""
    __unflatten_dual_varargs(isva::Bool, args, ::Val{nargs}) where {nargs}

If isva and nargs=2, then inputs `(lift(5.0, 0.0), lift(4.0, 0.0), lift(3.0, 0.0))`
are transformed into `(lift(5.0, 0.0), lift((4.0, 3.0), (0.0, 0.0)))` (each a `Lifted`).
"""
function __unflatten_dual_varargs(isva::Bool, args, ::Val{nargs}) where {nargs}
    isva || return args
    # The grouped vararg slot must carry the same chunk width as the incoming slots
    # (read from any arg's type parameter — all slots share the rule's build width).
    W = _lifted_width(first(args))
    group_primal = map(primal, args[nargs:end])
    # Plain `typeof`, not `_typeof`: `_typeof` per-element-sharpens, so a tuple of `Type` values
    # becomes `Tuple{Type{X},…}`, but the value's runtime type is `Tuple{DataType,…}` — not a
    # subtype (`isa` is `typeof <: T`), so that sharpened slot type is one no `Lifted` ctor can
    # build. `typeof` is always instance-valid and agrees with `_typeof` for non-Type-valued tuples.
    GP = typeof(group_primal)
    # An all-non-differentiable (or empty) vararg group has `dual_type === NoDual`; its slot
    # carries a single `NoDual`, not the element-wise `Tuple{NoDual,…}`. Collapse to match the
    # canonical V (otherwise the grouped `Lifted`'s V mismatches the rule's slot typeassert,
    # e.g. in debug-mode forward-over-reverse where the reverse args are non-diff CoDuals).
    group_v = dual_type(Val(W), GP) === NoDual ? NoDual() : map(tangent, args[nargs:end])
    grouped_args = Lifted{GP,W}(group_primal, group_v)
    return (args[1:(nargs - 1)]..., grouped_args)
end

struct DualInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
    is_used::Vector{Bool}
    debug_mode::Bool
    # Chunk width of the forward rule: every lifted slot / constant in the dual IR is
    # `Lifted{P, width, V}`. `width == 1` is the ordinary single-direction rule.
    width::Int
end

function generate_dual_ir(
    interp::MooncakeInterpreter,
    sig_or_mi;
    debug_mode=false,
    do_inline=true,
    do_optimize=true,
    chunk_width::Int=1,
)
    # Reset id count. This ensures that the IDs generated are the same each time this
    # function runs.
    seed_id!()

    # Grab code associated to the primal.
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        # Pin to one world so verify_ir's GlobalRef check passes; see `set_valid_world!`.
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    nargs = length(primal_ir.argtypes)

    # Reject before normalise! runs: Julia 1.12+ lowers non-const global writes
    # (`global x = y`) to Base.setglobal! on 1.12 and Core.setglobal! on 1.13+. CC.verify_ir
    # accepts these calls, so without this check the failure surfaces as a missing frule!!.
    @static if VERSION > v"1.12-"
        setglobal_calls = (GlobalRef(Base, :setglobal!), GlobalRef(Core, :setglobal!))
        for inst in stmt(primal_ir.stmts)
            if Meta.isexpr(inst, :call) && inst.args[1] in setglobal_calls
                unhandled_feature(
                    "Mooncake.jl does not support differentiating code that assigns to " *
                    "non-const global variables. Pass the state explicitly, return the " *
                    "updated value, or provide a custom frule!!. See the Known Limitations " *
                    "documentation for more context.",
                )
            end
        end
    end

    # Normalise the IR.
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)

    # Keep a copy of the primal IR with the insertions
    dual_ir = CC.copy(primal_ir)

    # Modify dual argument types:
    # - add one for the captures in the first position, with placeholder type for now
    # - convert the rest to lifted types (`Lifted{P, chunk_width, V}` per arg)
    for (a, P) in enumerate(primal_ir.argtypes)
        dual_ir.argtypes[a] = lifted_type(Val(chunk_width), CC.widenconst(P))
    end
    pushfirst!(dual_ir.argtypes, Any)

    # Data structure into which we can push any data which is to live in the captures field
    # of the OpaqueClosure used to implement this rule. The index at which a piece of data
    # lives in this data structure is equal to the index of the captures field of the
    # OpaqueClosure in which it will live. To write code which retrieves items from the
    # captures data structure, make use of `get_capture`.
    captures = Any[]

    is_used = characterised_used_ssas(stmt(primal_ir.stmts))
    info = DualInfo(primal_ir, interp, is_used, debug_mode, chunk_width)
    for (n, inst) in enumerate(dual_ir.stmts)
        ssa = SSAValue(n)
        modify_fwd_ad_stmts!(stmt(inst), dual_ir, ssa, captures, info)
    end

    # Process new nodes etc.
    dual_ir = CC.compact!(dual_ir)

    CC.verify_ir(dual_ir)

    # Now that the captured values are known, replace the placeholder value given for the
    # first argument type with the actual type.
    captures_tuple = (captures...,)
    dual_ir.argtypes[1] = _typeof(captures_tuple)

    # Inspection tools need the pre-optimization dual IR, while the AD pipeline still
    # wants the optimized form by default.
    dual_ir = do_optimize ? optimise_ir!(dual_ir; do_inline) : dual_ir
    return dual_ir,
    captures_tuple,
    DualRuleInfo(isva, nargs, dual_ret_type(primal_ir, Val(chunk_width)))
end

@inline get_capture(captures::T, n::Int) where {T} = captures[n]

"""
    const_dual!(captures::Vector{Any}, stmt, ::Val{N})::Union{Lifted,Int}

Build a width-`N` `Lifted` from `stmt` with a zero tangent — `stmt` is a constant, whose
derivative is zero, so its tangent must be zeroed (an uninitialised array tangent would leak
garbage into any op that reads the constant's tangent). `N` is the chunk width, threaded into
`zero_lifted(Val(N), v)` so the constant's V matches the surrounding chunked slots (`Val(1)`
for a standard forward rule). If the resulting `Lifted` is a bits type, then it is returned. If
it is not, then the `Lifted` is put into captures, and its location in `captures` returned.

Whether or not the value is a literal, or an index into the captures, can be determined from
the return type.
"""
function const_dual!(captures::Vector{Any}, stmt, ::Val{N})::Union{Lifted,Int} where {N}
    v = get_const_primal_value(stmt)
    x = zero_lifted(Val(N), v)
    if safe_for_literal(v)
        return x
    else
        push!(captures, x)
        return length(captures)
    end
end

## Modification of IR nodes

const ATTACH_AFTER = true
const ATTACH_BEFORE = false

modify_fwd_ad_stmts!(::Nothing, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo) = nothing

modify_fwd_ad_stmts!(::GotoNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::DualInfo) = nothing

function modify_fwd_ad_stmts!(
    stmt::GotoIfNot, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    # replace GotoIfNot with the call to primal
    Mooncake.replace_call!(dual_ir, ssa, Expr(:call, _primal, inc_args(stmt).cond))

    # reinsert the GotoIfNot right after the call to primal
    new_gotoifnot_inst = new_inst(Core.GotoIfNot(ssa, stmt.dest))
    CC.insert_node!(dual_ir, ssa, new_gotoifnot_inst, ATTACH_AFTER)
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::GlobalRef, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if isconst(stmt)
        d = const_dual!(captures, stmt, Val(info.width))
        if d isa Int
            Mooncake.replace_call!(dual_ir, ssa, Expr(:call, get_capture, Argument(1), d))
        else
            Mooncake.replace_call!(dual_ir, ssa, Expr(:call, identity, d))
        end
    else
        new_ssa = CC.insert_node!(dual_ir, ssa, new_inst(stmt), ATTACH_BEFORE)
        zero_lifted_call = Expr(:call, Mooncake.zero_lifted, Val(info.width), new_ssa)
        Mooncake.replace_call!(dual_ir, ssa, zero_lifted_call)
    end

    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::ReturnNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    # undefined `val` field means that stmt is unreachable.
    isdefined(stmt, :val) || return nothing

    # stmt is an Argument, then already a dual, and must just be incremented.
    if stmt.val isa Union{Argument,SSAValue}
        Mooncake.replace_call!(dual_ir, ssa, ReturnNode(__inc(stmt.val)))
        return nothing
    end

    # stmt is a const, so we have to turn it into a dual.
    d = const_dual!(captures, stmt.val, Val(info.width))
    if d isa Int
        get_dual = Expr(:call, get_capture, Argument(1), d)
        get_dual_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_dual), ATTACH_BEFORE)
        Mooncake.replace_call!(dual_ir, ssa, ReturnNode(get_dual_ssa))
    else
        Mooncake.replace_call!(dual_ir, ssa, ReturnNode(d))
    end
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = zero_lifted(
            Val(info.width), get_const_primal_value(stmt.values[n])
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        lifted_type(Val(info.width), CC.widenconst(get_ir(dual_ir, ssa, :type))),
    )
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PiNode, dual_ir::IRCode, ssa::SSAValue, ::Vector{Any}, info::DualInfo
)
    if stmt.val isa Union{Argument,SSAValue}
        v = __inc(stmt.val)
    else
        v = zero_lifted(Val(info.width), get_const_primal_value(stmt.val))
    end
    replace_call!(
        dual_ir, ssa, PiNode(v, lifted_type(Val(info.width), CC.widenconst(stmt.typ)))
    )
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::UpsilonNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if !(stmt.val isa Union{Argument,SSAValue})
        stmt = UpsilonNode(zero_lifted(Val(info.width), get_const_primal_value(stmt.val)))
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        lifted_type(Val(info.width), CC.widenconst(get_ir(dual_ir, ssa, :type))),
    )
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiCNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = zero_lifted(
            Val(info.width), get_const_primal_value(stmt.values[n])
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        lifted_type(Val(info.width), CC.widenconst(get_ir(dual_ir, ssa, :type))),
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

## Modification of IR nodes - expressions

__get_primal(x::Lifted) = primal(x)

#
# Nfwd forward path (the default; `debug_mode` forces the fully-checked transform).
#
# When a whole non-primitive function's NDual execution is provably free of dual-laundering (a
# reinterpret / bitcast / pointer op or a foreigncall handed a dual buffer), `build_frule` runs the
# primal directly on the inner dual values instead of deriving a per-op `Lifted`/frule transform.
# Soundness rests on the classifier below: it only vouches for code it can see through (recursable
# `:invoke`s and structural builtins) and rejects any call whose inferred graph hands a dual-typed
# argument to something opaque, or whose result is not NDual-coherent.

# Primal projection of an nfwd dual result (inner-value invariant: read, never recompute).
_nfwd_primal(y::Nfwd.NDual) = y.value
_nfwd_primal(y::Complex{<:Nfwd.NDual}) = Complex(y.re.value, y.im.value)
_nfwd_primal(y::Nfwd.NDualArray) = getfield(y, :primal)
@static if VERSION >= v"1.11-rc4"  # `NDualMemoryRef` wraps `MemoryRef`, absent on Julia 1.10
    _nfwd_primal(y::Nfwd.NDualMemoryRef) = getfield(y, :primal)
end
_nfwd_primal(y::Tuple) = map(_nfwd_primal, y)
_nfwd_primal(y::NamedTuple) = map(_nfwd_primal, y)
_nfwd_primal(@nospecialize(y)) = y

# The nfwd path is only taken when the result type is one whose primal `_nfwd_primal` can
# project exactly — a bare inner dual (`NDual`/`Complex{NDual}`/`NDualArray`/`NDualMemoryRef`) or a
# tuple/named-tuple of those. A result carrying NDuals *inside* a user struct (e.g. an nfwd-built
# `Normal{NDual}`) is rejected: the identity fallback would put a dual-typed value in the `Lifted`
# primal field, mismatching `lifted_type`. Rejected calls fall back to the frule transform.
function _nfwd_projectable(@nospecialize(T))
    T isa DataType || return false
    T.name === _TN_NDUAL && return true
    # An `NDualArray` is nfwd-projectable only when its backing array can be scalar-iterated
    # cheaply on the host. A GPU backing (`CuArray`) cannot: nfwd runs the primal element-wise
    # over the dual array, scalar-indexing the device array (disallowed) — such ops must route to
    # the transform's device frules. `T.parameters[4]` is the backing array type; the default is
    # `true`, and the CUDA extension overrides `_nfwd_backing_projectable` to `false` for `CuArray`.
    # This must precede the `_TN_NDMEMREF` check: on Julia 1.10 that sentinel aliases the
    # `NDualArray` TypeName, so an earlier unguarded return would admit GPU-backed dual arrays.
    T.name === _TN_NDARRAY && return _nfwd_backing_projectable(T.parameters[4])
    T.name === _TN_NDMEMREF && return true
    if T <: Complex
        p = T.parameters[1]
        return p isa DataType && p.name === _TN_NDUAL
    end
    # An empty tuple / named-tuple canonicalises to `NoDual` (its `tangent_type` is `NoTangent`),
    # but nfwd execution returns the bare `()` — a non-canonical dual the `Lifted` machinery
    # can't represent coherently — so it is not projectable and routes to the transform. `all`
    # over no parameters is vacuously true, hence the explicit non-empty guard.
    if T <: Tuple
        return !isempty(T.parameters) &&
               all(p -> p isa Type && _nfwd_projectable(p), T.parameters)
    end
    if T <: NamedTuple
        vt = T.parameters[2]
        return vt isa DataType &&
               vt <: Tuple &&
               !isempty(vt.parameters) &&
               all(p -> p isa Type && _nfwd_projectable(p), vt.parameters)
    end
    return false
end

# Backing-array trait for the `NDualArray` branch of `_nfwd_projectable`. Host arrays support the
# cheap scalar iteration nfwd needs; the CUDA extension overrides this to `false` for `CuArray`, so
# GPU-backed dual arrays route to the transform's device frules instead of scalar-indexing.
_nfwd_backing_projectable(@nospecialize(A)) = true

# Unwrap an nfwd-call operand into a value the primal function can be applied to. A differentiable
# slot yields its inner dual value (`NDual`/`NDualArray`/`NDualMemoryRef`, dispatch-compatible with
# the primal function); a non-differentiable slot (`NoDual` inner — how constants arrive) yields its
# primal value, exact because its derivative is zero. Every operand is a `Lifted` (the rule ABI).
@inline _nfwd_unwrap(a::Lifted{P,N,NoDual}) where {P,N} = primal(a)
@inline _nfwd_unwrap(a::Lifted) = tangent(a)

# Width-`N` forward rule for a whole function that is nfwd-safe (see `_nfwd_safe`): run
# the primal directly on the inner dual values instead of deriving a per-op transform rule. Built
# by `build_frule` when `nfwd` is set and the function classifies nfwd-safe.
struct NfwdFRule{N} end

@inline function (::NfwdFRule{N})(cf::Lifted, args::Vararg{Lifted,M}) where {N,M}
    y = primal(cf)(map(_nfwd_unwrap, args)...)
    p = _nfwd_primal(y)
    return Lifted{typeof(p),N}(p, y)
end

_copy(nf::NfwdFRule) = nf

const _TN_NDUAL = Base.unwrap_unionall(Nfwd.NDual).name
const _TN_NDARRAY = Base.unwrap_unionall(Nfwd.NDualArray).name
# Julia 1.10 has no `MemoryRef`, hence no `NDualMemoryRef`; alias the sentinel to the array
# TypeName (already tested alongside it) so the memref branches stay inert there.
@static if VERSION >= v"1.11-rc4"
    const _TN_NDMEMREF = Base.unwrap_unionall(Nfwd.NDualMemoryRef).name
else
    const _TN_NDMEMREF = _TN_NDARRAY
end

function _nfwd_has_ndual(@nospecialize(T), depth::Int=0)
    depth > 12 && return true
    T === Union{} && return false
    if T isa DataType
        (T.name === _TN_NDUAL || T.name === _TN_NDARRAY || T.name === _TN_NDMEMREF) &&
            return true
        for p in T.parameters
            p isa Type && _nfwd_has_ndual(p, depth + 1) && return true
        end
        return false
    elseif T isa Union
        return _nfwd_has_ndual(T.a, depth + 1) || _nfwd_has_ndual(T.b, depth + 1)
    elseif T isa UnionAll
        return _nfwd_has_ndual(T.body, depth + 1)
    end
    return false
end

function _nfwd_invoke_sig(@nospecialize(t))
    for _ in 1:4
        t isa Core.MethodInstance && return t.specTypes
        t isa Core.CodeInstance ? (t = t.def) : return nothing
    end
    return nothing
end

# Does any SSA/argument operand carry an inner dual? Only these two operand kinds can; a constant
# operand (`QuoteNode`/`GlobalRef`/literal) never holds a dual, so it is skipped.
function _nfwd_any_dual(ssatypes, @nospecialize(sig), args)
    for a in args
        if a isa Core.SSAValue
            t = ssatypes[a.id]
            t = if t isa Core.Const
                Core.Typeof(t.val)
            elseif t isa Core.PartialStruct
                t.typ
            else
                t
            end
            _nfwd_has_ndual(CC.widenconst(t)) && return true
        elseif a isa Core.Argument && 1 <= a.n <= length(sig.parameters)
            _nfwd_has_ndual(sig.parameters[a.n]) && return true
        end
    end
    return false
end

# The nfwd path can only vouch for code it can see through: statically-resolved `:invoke`s (whose
# bodies it recurses into) and structural `Core.Builtin`s that move or inspect values without ever
# reinterpreting their numeric bytes. Everything else is opaque to the scan — `:foreigncall`, any
# `Core.IntrinsicFunction`, and unresolved dynamic `:call`s — and an opaque op that receives a
# dual-typed argument could launder the derivative (read it back as a plain number), so the whole
# function is rejected and falls back to the frule transform. This is a whitelist posture: an
# unrecognised primitive is unsafe by default, so a newly-introduced dual-laundering op cannot
# silently pass. `_NFWD_SAFE_FOREIGN` and `_NFWD_SAFE_BUILTINS` are the only trusted opaque ops; the
# `nfwd primitive coverage` test asserts they stay in step with `Core` and fails loudly if
# a new builtin appears unclassified.

# Identity/hash foreigncalls that take a dual object but do not read its derivative: `objectid` of a
# dual is address-based (a dual is an immutable wrapper over mutable buffers, or an isbits scalar
# used only for dict/identity bookkeeping), so the hash never flows into the numeric gradient. This
# is what lets a real logdensity (whose VarInfo dict machinery calls `jl_object_id`) run under nfwd.
const _NFWD_SAFE_FOREIGN = Set{Symbol}([:jl_object_id])

# Structural builtins that only move, inspect, or select values (and type/metaprogramming
# machinery that never sees a numeric dual), so a dual passes through them intact — none reinterpret
# a value's bytes as a number. Any builtin NOT listed here is treated as opaque: a dual reaching it
# routes the function to the frule transform. `_NFWD_OPAQUE_BUILTINS` records the ones we deliberately
# exclude (indirections whose callee body the scan cannot recurse into); the `nfwd primitive
# coverage` test asserts every `Core.Builtin` is in one set or the other, so a new builtin fails
# loudly rather than being silently trusted.
const _NFWD_SAFE_BUILTINS = Set{Symbol}([
    # field / property / global access
    :getfield,
    :setfield!,
    :swapfield!,
    :modifyfield!,
    :replacefield!,
    :setfieldonce!,
    :getproperty,
    :setproperty!,
    :getglobal,
    :setglobal!,
    :swapglobal!,
    :modifyglobal!,
    :replaceglobal!,
    :setglobalonce!,
    :isdefinedglobal,
    :get_binding_type,
    :set_binding_type!,
    # typed memory / array element access (structural — returns/stores the element type). Julia
    # ≥1.11 lowers to `memoryref*`; 1.10 uses the `array*` builtins. Listing both keeps the
    # classifier version-robust — a name absent on a given version simply never matches.
    :memorynew,
    :memoryref_isassigned,
    :memoryrefget,
    :memoryrefset!,
    :memoryrefswap!,
    :memoryrefmodify!,
    :memoryrefreplace!,
    :memoryrefsetonce!,
    :memoryrefnew,
    :memoryrefoffset,
    :arrayref,
    :arrayset,
    :arraysize,
    :const_arrayref,
    # reflection / predicates / construction / select
    :isdefined,
    :nfields,
    :fieldtype,
    :typeof,
    :typeassert,
    :isa,
    :(<:),
    :(===),
    :applicable,
    :sizeof,
    :current_scope,
    :tuple,
    :ifelse,
    :throw,
    :throw_methoderror,
    :compilerbarrier,
    :donotdelete,
    # type / metaprogramming machinery (operates on types/exprs/svecs, never a numeric dual)
    :svec,
    :_svec_len,
    :_svec_ref,
    :_typevar,
    :_typebody!,
    :_structtype,
    :_abstracttype,
    :_primitivetype,
    :_setsuper!,
    :_defaultctors,
    :_equiv_typedef,
    :_compute_sparams,
    :_expr,
    :apply_type,
])

# Indirection builtins whose callee body the scan cannot see, so a dual flowing through them is
# opaque and routed to the frule transform (never trusted, never an error).
const _NFWD_OPAQUE_BUILTINS = Set{Symbol}([
    :invoke,
    :invokelatest,
    :invoke_in_world,
    :_call_in_world,
    :_call_in_world_total,
    :_call_latest,
    :_apply_iterate,
    :_apply_pure,
    :finalizer,
])

# Resolve a call target to a `Core.Builtin`/`Core.IntrinsicFunction` when it is statically known
# (a const global binding or an inlined primitive), else `nothing` (a dynamic/unresolved callee the
# scan must treat as opaque).
function _nfwd_callee(@nospecialize(a))
    if a isa GlobalRef
        return if (isdefined(a.mod, a.name) && isconst(a.mod, a.name))
            getglobal(a.mod, a.name)
        else
            nothing
        end
    elseif a isa Core.IntrinsicFunction || a isa Core.Builtin
        return a
    end
    return nothing
end

# Length-changing array operations. The `NDualArray` is a fixed-shape parallel structure (its
# partials are a matching-length block/arrays) with no grow/shrink, so a function that mutates a
# dual array's length cannot run natively on the duals. Reject such calls so nfwd fires only where
# every op has proven NDual coverage; the transform handles length mutation correctly. A dynamic
# `:call` to one of these on a dual is already rejected as an opaque call — the gap this closes is a
# resolved `:invoke` (e.g. `append!` lowers to `invoke push!(::NDualArray, …)`), which is otherwise
# recursed into rather than checked.
const _NFWD_ARRAY_MUTATORS = Set{Any}(
    Any[
        push!,
        pushfirst!,
        pop!,
        popfirst!,
        append!,
        prepend!,
        insert!,
        deleteat!,
        resize!,
        sizehint!,
        empty!,
        splice!,
        keepat!,
        filter!,
    ],
)
const _NFWD_ARRAY_MUTATOR_TYPES = Set{Any}(Any[typeof(f) for f in _NFWD_ARRAY_MUTATORS])

function _nfwd_scan_body!(work::Vector{Any}, ci, @nospecialize(sig))
    ssatypes = ci.ssavaluetypes
    for st in ci.code
        st isa Expr || continue
        if st.head === :foreigncall
            fn = st.args[1]  # foreigncall target: a `QuoteNode`/`Symbol` name, or a dynamic ccall
            name = if fn isa QuoteNode
                fn.value
            elseif fn isa Symbol
                fn
            else
                Symbol("")
            end
            name in _NFWD_SAFE_FOREIGN && continue
            _nfwd_any_dual(ssatypes, sig, st.args) && return true
        elseif st.head === :call && !isempty(st.args)
            cv = _nfwd_callee(st.args[1])
            # A structural builtin is dual-transparent; anything else (intrinsic, unknown builtin,
            # or unresolved dynamic call) is opaque and rejected if it touches a dual.
            if cv isa Core.Builtin && nameof(cv) in _NFWD_SAFE_BUILTINS
                continue
            end
            _nfwd_any_dual(ssatypes, sig, st.args) && return true
        elseif st.head === :invoke
            s = _nfwd_invoke_sig(st.args[1])
            if s !== nothing
                # A length-changing array op on a dual array cannot run natively (fixed-shape
                # NDualArray); reject so nfwd falls back to the transform.
                if s isa DataType &&
                    !isempty(s.parameters) &&
                    s.parameters[1] in _NFWD_ARRAY_MUTATOR_TYPES &&
                    _nfwd_any_dual(ssatypes, sig, st.args)
                    return true
                end
                push!(work, s)
            end
        end
    end
    return false
end

_nfwd_code_typed(@nospecialize(sig)) =
    try
        Base.code_typed_by_type(sig; optimize=true)
    catch
        nothing
    end

# Memoise the nfwd-safety verdict per dual signature. The verdict is a pure function of the
# signature and the inferred IR — fixed within a world — but computing it runs `code_typed` on the
# call and recursively on every reachable `:invoke`, so re-deriving it for every forward build of
# the same signature is wasted work. Cache keyed by signature and flushed whenever the world age
# advances, since new method definitions can change inference (and hence the verdict).
const _NFWD_SAFE_CACHE = Dict{Any,Bool}()
const _NFWD_SAFE_WORLD = Ref{UInt}(typemax(UInt))
const _NFWD_SAFE_LOCK = ReentrantLock()

function _nfwd_body_safe_cached(@nospecialize(sig), @nospecialize(expected))
    w = Base.get_world_counter()
    return Base.@lock _NFWD_SAFE_LOCK begin
        if w != _NFWD_SAFE_WORLD[]
            empty!(_NFWD_SAFE_CACHE)
            _NFWD_SAFE_WORLD[] = w
        end
        get!(() -> _nfwd_body_safe(sig, expected), _NFWD_SAFE_CACHE, (sig, expected))
    end
end

# `expected` is the canonical dual of the primal result, `dual_type(Val(N), primal_return_type)`.
# Nfwd execution must reproduce exactly this shape: a function can behave differently on inner
# duals than on its primal (e.g. `TwicePrecision` arithmetic collapses an `NDual` back to a bare
# scalar, so `_logrange_extra` returns `Tuple{NDual,NDual}` on duals but `Tuple{TwicePrecision,…}`
# on floats), which would give a wrong primal and tangent. Requiring `rt === expected` rejects
# those and routes them to the transform.
function _nfwd_body_safe(@nospecialize(sig), @nospecialize(expected); maxnodes::Int=600)
    cts = _nfwd_code_typed(sig)
    (cts === nothing || length(cts) != 1) && return false
    ci, rt = cts[1]
    (isconcretetype(rt) && _nfwd_projectable(rt) && rt === expected) || return false
    work = Any[]
    _nfwd_scan_body!(work, ci, sig) && return false
    visited = Set{Any}()
    nodes = 0
    while !isempty(work)
        s = pop!(work)
        s in visited && continue
        push!(visited, s)
        (nodes += 1) > maxnodes && return false
        cs = _nfwd_code_typed(s)
        (cs === nothing || length(cs) != 1) && return false
        _nfwd_scan_body!(work, cs[1][1], s) && return false
    end
    return true
end

# Principle: nfwd fires only when every differentiable leaf is
# `NDualEltype = Union{IEEEFloat, Complex{<:IEEEFloat}}` — represented as `NDual` (scalar),
# `NDualArray` (array), or `NDualMemoryRef` (memory). Anything else (e.g. `BFloat16`, whose
# `dual_type` is the generic `NTuple{N,·}`) is not projectable, so it routes to the frule transform.
# This gate enforces that per argument; NDual op coverage is completed only for IEEEFloat/Complex.
#
# The nfwd path handles inner-dual scalars/arrays only: an argument is admissible iff it is
# non-differentiable (dual `NoDual` → passed as its primal) or dual-lifts to a projectable inner
# dual (`NDual`/`Complex{NDual}`/`NDualArray`/`NDualMemoryRef`, or a tuple/named-tuple of those →
# passed as its dual). An argument that dual-lifts to a struct wrapper (`ImmutableDual`/
# `MutableDual`) is not dispatch-compatible with the primal function, so the call is rejected and
# falls back to the frule transform.
function _nfwd_safe(sig_types::Vector, width::Int)
    isempty(sig_types) && return false
    all(isconcretetype, sig_types) || return false
    # callee must be non-differentiable: nfwd call cannot carry a closure-field derivative.
    dual_type(Val(width), sig_types[1]) === NoDual || return false
    dsig_args = Any[]
    for i in 2:length(sig_types)
        dt = dual_type(Val(width), sig_types[i])
        if dt === NoDual
            push!(dsig_args, sig_types[i])          # non-differentiable → dispatch on the primal
        elseif _nfwd_projectable(dt)
            push!(dsig_args, dt)                     # inner dual → dispatch on the dual
        else
            return false                            # struct dual (ImmutableDual/…) → not nfwd
        end
    end
    # The nfwd dual result must equal the canonical dual of the primal result; infer the primal
    # return type and hand its `dual_type` to the safety check as the required shape.
    pcts = _nfwd_code_typed(Tuple{sig_types...})
    (pcts === nothing || length(pcts) != 1) && return false
    prt = pcts[1][2]
    isconcretetype(prt) || return false
    expected = try
        dual_type(Val(width), prt)
    catch
        return false
    end
    return _nfwd_body_safe_cached(Tuple{sig_types[1],dsig_args...}, expected)
end

function modify_fwd_ad_stmts!(
    stmt::Expr, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        sig_types = map(raw_args) do x
            t = CC.widenconst(get_forward_primal_type(info.primal_ir, x))
            # Replace types containing Union{} (unreachable code/failed inference)
            # with Any. This allows the code to proceed; is_primitive will return
            # false and we'll use dynamic rules that resolve types at runtime.
            return contains_bottom_type(t) ? Any : t
        end
        sig = Tuple{sig_types...}
        mi = isexpr(stmt, :invoke) ? get_mi(stmt.args[1]) : missing
        args = map(__inc, raw_args)

        # Special case: if the result of a call to getfield is un-used, then leave the
        # primal statement alone (just increment arguments as usual). This was causing
        # performance problems in a couple of situations where the field being requested is
        # not known at compile time. `getfield` cannot be dead-code eliminated, because it
        # can throw an error if the requested field does not exist. Everything _other_ than
        # the boundscheck is eliminated in LLVM codegen, so it's important that AD doesn't
        # get in the way of this.
        #
        # This might need to be generalised to more things than just `getfield`, but at the
        # time of writing this comment, it's unclear whether or not this is the case.
        if !info.is_used[ssa.id] && get_const_primal_value(args[1]) == getfield
            fwds = new_inst(Expr(:call, __fwds_pass_no_ad!, args...))
            replace_call!(dual_ir, ssa, fwds)
            return nothing
        end

        # Lift arguments.
        dual_args = map(args) do arg
            arg isa Union{Argument,SSAValue} && return arg
            return zero_lifted(Val(info.width), get_const_primal_value(arg))
        end

        interp = info.interp
        if is_primitive(context_type(interp), ForwardMode, sig, interp.world)
            rule = build_primitive_frule(sig)
            if safe_for_literal(rule)
                replace_call!(dual_ir, ssa, Expr(:call, rule, dual_args...))
            else
                push!(captures, rule)
                get_rule = Expr(:call, get_capture, Argument(1), length(captures))
                rule_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
                replace_call!(dual_ir, ssa, Expr(:call, rule_ssa, dual_args...))
            end
        else
            dm = info.debug_mode
            # Predict at the world this transform runs in: inside a pinned rebuild
            # (`_build_rule!`) the current world is later, which would reintroduce the
            # `Trule` mismatch of #1218 one level down.
            w = interp.world
            push!(
                captures,
                if isexpr(stmt, :invoke)
                    LazyFRule(mi, dm, info.width, w)
                else
                    DynamicFRule(dm, info.width, w)
                end,
            )
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            replace_call!(dual_ir, ssa, Expr(:call, rule_ssa, dual_args...))
        end
    elseif isexpr(stmt, :boundscheck)
        # Keep the boundscheck, but wrap it in a width-`info.width` Lifted.
        inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        bc_ssa = CC.insert_node!(dual_ir, ssa, inst, ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, Expr(:call, zero_lifted, Val(info.width), bc_ssa))
    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(dual_ir, ssa, nothing)
    elseif Meta.isexpr(stmt, :copyast)
        new_copyast_inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        new_copyast_ssa = CC.insert_node!(dual_ir, ssa, new_copyast_inst, ATTACH_BEFORE)
        replace_call!(
            dual_ir, ssa, Expr(:call, zero_lifted, Val(info.width), new_copyast_ssa)
        )
    elseif Meta.isexpr(stmt, :loopinfo)
        # Leave this node alone.
    elseif isexpr(stmt, :throw_undef_if_not)
        # args[1] is a Symbol, args[2] is the condition which must be primalized
        primal_cond = Expr(:call, _primal, inc_args(stmt).args[2])
        replace_call!(dual_ir, ssa, primal_cond)
        new_undef_inst = new_inst(Expr(:throw_undef_if_not, stmt.args[1], ssa))
        CC.insert_node!(dual_ir, ssa, new_undef_inst, ATTACH_AFTER)
    elseif isexpr(stmt, :enter)
        # Leave this node alone
    elseif isexpr(stmt, :leave)
        # Leave this node alone
    elseif isexpr(stmt, :pop_exception)
        # Leave this node alone
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

mutable struct LazyFRule{primal_sig,Trule}
    debug_mode::Bool
    mi::Core.MethodInstance
    width::Int
    world::UInt
    rule::Trule
    function LazyFRule(mi::Core.MethodInstance, debug_mode::Bool, width::Int, world::UInt)
        interp = get_interpreter(ForwardMode, world)
        return new{mi.specTypes,frule_type(interp, mi;debug_mode,chunk_size=width)}(
            debug_mode, mi, width, world
        )
    end
    function LazyFRule{Tprimal_sig,Trule}(
        mi::Core.MethodInstance, debug_mode::Bool, width::Int, world::UInt
    ) where {Tprimal_sig,Trule}
        return new{Tprimal_sig,Trule}(debug_mode, mi, width, world)
    end
end

# Create new lazy rule with same method instance, debug mode, chunk width, and prediction world
function _copy(x::P) where {P<:LazyFRule}
    return P(x.mi, x.debug_mode, x.width, x.world)
end

# On Julia 1.10, the generic __call_rule fallback is @stable-checked and returns Any for
# LazyFRule, triggering TypeInstabilityError when dispatch_doctor_mode = "error".
# Add type-asserting specialisations so callers in @stable contexts see a concrete type.
# LazyFRule doesn't contain an OpaqueClosure directly, so no dispatch barrier needed.
@static if VERSION < v"1.11-"
    @inline function __call_rule(
        rule::LazyFRule{sig,DerivedFRule{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}},
        args,
    ) where {sig,P,A,R,isva,nargs}
        return rule(args...)::R
    end
    @inline function __call_rule(
        rule::LazyFRule{
            sig,DebugFRule{DerivedFRule{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}}
        },
        args,
    ) where {sig,P,A,R,isva,nargs}
        return rule(args...)::R
    end
end

@inline function (rule::LazyFRule)(args::Vararg{Any,N}) where {N}
    return isdefined(rule, :rule) ? __call_rule(rule.rule, args) : _build_rule!(rule, args)
end

# Build at the world `Trule` was predicted at: a later world can re-tighten `mi`'s inferred
# return type, giving a rule that no longer matches `Trule` and fails to `convert` (#1218).
# Not covered: the inference-complexity-widening case in #1209's headline MWE.
@noinline function _build_rule!(rule::LazyFRule{sig,Trule}, args) where {sig,Trule}
    interp = get_interpreter(ForwardMode, rule.world)
    # `nfwd=false`: nfwd is a top-level whole-function decision. This is a sub-rule build, and its
    # result type must match `Trule` (the transform-rule type `frule_type` predicted at
    # construction); an `NfwdFRule` here would fail to `convert` into the `rule.rule` field.
    rule.rule = build_frule(
        interp,
        rule.mi;
        debug_mode=rule.debug_mode,
        chunk_size=rule.width,
        skip_world_age_check=true,
        nfwd=false,
    )
    return __call_rule(rule.rule, args)
end

function dual_ret_type(primal_ir::IRCode, ::Val{N}) where {N}
    return lifted_type(Val(N), compute_ir_rettype(primal_ir))
end

function frule_type(
    interp::MooncakeInterpreter{C}, mi::CC.MethodInstance; debug_mode, chunk_size::Int=1
) where {C}
    sig = _get_sig(mi)
    if is_primitive(C, ForwardMode, sig, interp.world)
        # Build the rule to obtain its concrete type. For non-singleton primitive rules
        # this allocates a throwaway instance; the cost is compile-time only and does not
        # affect hot-path performance.
        rule = build_primitive_frule(sig)
        return debug_mode ? DebugFRule{typeof(rule)} : typeof(rule)
    end
    ir, _ = lookup_ir(interp, mi)
    nargs = length(ir.argtypes)
    isva, _ = is_vararg_and_sparam_names(mi)
    arg_types = map(CC.widenconst, ir.argtypes)
    sig = Tuple{arg_types...}
    dual_args_type = Tuple{map(T -> lifted_type(Val(chunk_size), T), arg_types)...}
    closure_type = RuleMC{dual_args_type,dual_ret_type(ir, Val(chunk_size))}
    Tderived_rule = DerivedFRule{sig,closure_type,isva,nargs}
    return debug_mode ? DebugFRule{Tderived_rule} : Tderived_rule
end

struct DynamicFRule{V}
    cache::V
    debug_mode::Bool
    width::Int
    world::UInt
end

function DynamicFRule(debug_mode::Bool, width::Int, world::UInt)
    return DynamicFRule(Dict{Any,Any}(), debug_mode, width, world)
end

# Create new dynamic rule with empty cache, same debug mode, chunk width, and build world
function _copy(x::P) where {P<:DynamicFRule}
    return P(Dict{Any,Any}(), x.debug_mode, x.width, x.world)
end

function (dynamic_rule::DynamicFRule)(args::Vararg{Lifted,N}) where {N}
    # `Base._stable_typeof` must be used here, rather than `typeof` or `Mooncake._typeof`.
    # See DynamicDerivedRule for details, the same reasoning applies.
    sig = Tuple{map(Base._stable_typeof ∘ primal, args)...}
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        # Build at this rule's creation world, not the current one; see _build_rule! (#1218)
        interp = get_interpreter(ForwardMode, dynamic_rule.world)
        # `nfwd=false`: nfwd is a top-level whole-function decision, not a sub-rule one (see
        # `_build_rule!`).
        rule = build_frule(
            interp,
            sig;
            debug_mode=dynamic_rule.debug_mode,
            chunk_size=dynamic_rule.width,
            skip_world_age_check=true,
            nfwd=false,
        )
        dynamic_rule.cache[sig] = rule
    end
    return __call_rule(rule, args)
end
