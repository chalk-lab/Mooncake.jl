#
# Nfwd-native execution: the shared classifier and rule wrapper
#
# Nfwd runs a whole function's primal *directly on inner dual values* (`NDual`/`NDualArray`/
# `NDualMemoryRef`) instead of deriving a per-op `Lifted`/frule transform. That is only sound if
# nothing in the body can launder a derivative — read a dual's bytes back as a plain number via a
# reinterpret / bitcast / pointer op, or a foreigncall handed a dual buffer — so everything here
# exists to decide that question conservatively. `_nfwd_safe` is the trust boundary: it vouches only
# for code it can see through (recursable `:invoke`s and structural builtins) and rejects any call
# whose inferred graph hands a dual-typed argument to something opaque, or whose result is not
# NDual-coherent. A rejection falls back to the fully-checked transform — slower, always correct.
#
# Kept in its own file for two reasons: the soundness-critical classifier is easier to review and
# maintain in isolation (its whitelists must track new `Core` builtins and foreigncalls), and it is
# mode-agnostic — nothing below is specific to forward mode, so reverse mode can share it. The only
# nfwd footprint left in `forward_mode.jl` is `build_frule`'s `nfwd` kwarg and its dispatch to
# `NfwdFRule`.

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
struct NfwdFRule{N}
    # Differentiable constants the classifier saw anywhere in the natively-executed call graph.
    # The transform's guard is per-rule and nested rules re-check their own arguments, but nfwd
    # flattens the whole graph into this one call, so the set has to be transitive to match.
    consts::ConstAliasSet
end

@inline function (nf::NfwdFRule{N})(cf::Lifted, args::Vararg{Lifted,M}) where {N,M}
    _check_constant_aliasing(nf.consts, args)
    y = primal(cf)(map(_nfwd_unwrap, args)...)
    p = _nfwd_primal(y)
    return Lifted{typeof(p),N}(p, y)
end

_copy(nf::NfwdFRule) = nf

# On Julia 1.10 the generic `__call_rule` barrier infers `Any` and boxes its arguments (see
# `src/utils.jl` for why it exists). `NfwdFRule`'s call method holds no OpaqueClosure, so the
# world-age crash the barrier guards against cannot arise here: call it directly and keep the
# forward pass type-stable.
@static if VERSION < v"1.11-"
    @inline __call_rule(rule::NfwdFRule, args) = rule(args...)
end

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
    # construction / select
    :typeassert,
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

# Builtins whose result depends on how a value is REPRESENTED rather than on the value it stands
# for. A dual answers each of them differently from its primal — `NDual{Float64,1}` has a different
# size, field count, type and bit pattern than `Float64` — so running one on a dual computes a
# different function, not a differentiated one. Trusted only when no operand is dual-typed; a dual
# reaching one routes the function to the frule transform.
#
# This is the property the list above was reaching for; "does not reinterpret bytes as a number" is
# too weak, since `sizeof` reinterprets nothing. Testing the OPERAND rather than where the result
# flows is what makes it closed: `x === y` on two duals compares `partials`, and no analysis of the
# result's use would catch that, the result being a runtime `Bool` inference cannot fold.
#
# INCOMPLETE for the members inference CAN fold. `sizeof(NDual{Float64,1})` is the literal 16 before
# the scan runs, so no operand survives to test and `x * sizeof(x)` still returns 32.0 against a
# primal of 16.0. Catching those needs the primal and dual signatures compared statement by
# statement, which `_nfwd_branches_agree_one` does today only for branch conditions.
const _NFWD_REPR_BUILTINS = Set{Symbol}([
    :isdefined, :nfields, :fieldtype, :typeof, :isa, :(<:), :(===), :applicable, :sizeof
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

# Array-level products run element by element on dual arrays, so each scalar product becomes an
# `NDual` multiply, where the hand-written BLAS frules do a lane's whole product in one `gemm!` /
# `gemv!`. Measured at chunk width 8: matmul made `simple_mlp` 28x primal against 1.7x through the
# transform, and `dot` over 1600 elements is 8.4x slower native. Reject so the transform picks
# those rules up. Rejection additionally requires a dual *array* in the signature, which is what
# keeps scalar `dot`/`*` on `NDual` native. `sum` and the other reductions are only ~1.2x and stay
# native, so measure before extending this set.
# Built from whichever kernel names the running Julia defines: 1.12 renamed the inner kernels to
# `__generic_mat*mul!`, and matching only the 1.10/1.11 spellings left matvec-shaped code on the
# nfwd path there — `sum(A*x)` was still native at every width, keeping the whole dual-in-kernel
# cost. Any name absent on this version is simply skipped.
const _NFWD_ELEMENTWISE_LA_TYPES = Set{Any}(
    Any[(typeof(getfield(LinearAlgebra, n)) for n in (
        :_generic_matmatmul!,
        :__generic_matmatmul!,
        :generic_matmatmul!,
        :_generic_matvecmul!,
        :__generic_matvecmul!,
        :generic_matvecmul!,
        :dot,
    ) if isdefined(LinearAlgebra, n))...,],
)

# Does the invoke's own signature carry a dual array (as opposed to a bare `NDual` scalar)?
function _nfwd_sig_has_dual_array(s::DataType)
    for p in s.parameters
        p isa Type && p <: AbstractArray && _nfwd_has_ndual(p) && return true
    end
    return false
end

# Rejecting an op is only safe when a hand-written rule can actually take it: `dot`'s array rule is
# `Tuple{typeof(dot),Vector{P},Vector{P}} where {P<:BlasRealFloat}`, so a complex or
# higher-dimensional `dot` has no array-level primitive to land on. The transform then descends to
# the raw-pointer foreigncall, which cannot address a lane above chunk width 1, turning a working
# nfwd case into a throw. Reject only what the rule covers; the matmul kernels are unconditional
# because their frules span every operand shape that reaches them.
function _nfwd_la_rule_covers(s::DataType)
    s.parameters[1] === typeof(LinearAlgebra.dot) || return true
    for p in s.parameters
        p isa Type && p <: AbstractArray && _nfwd_has_ndual(p) || continue
        pu = Base.unwrap_unionall(p)
        pu isa DataType && length(pu.parameters) >= 3 || return false
        el, nd = pu.parameters[1], pu.parameters[3]
        (el isa Type && el <: Real && nd === 1) || return false
    end
    return true
end

# Record an operand that is a constant primal. `SSAValue`/`Argument`/`SlotNumber` name a computed
# value and `Symbol`/`Expr` are syntax, never a value; everything else is the constant itself.
function _nfwd_record_const!(consts::Vector{Any}, @nospecialize(a))
    v = if a isa GlobalRef
        (isdefined(a.mod, a.name) && isconst(a.mod, a.name)) || return nothing
        getglobal(a.mod, a.name)
    elseif a isa QuoteNode
        a.value
    elseif a isa Union{Core.SSAValue,Core.Argument,Core.SlotNumber,Expr,Symbol}
        return nothing
    else
        a
    end
    return record_const_alias!(consts, v)
end

# Constants of a statement of any kind. Which kind carries the constant is version-dependent: on
# 1.11 a `GlobalRef` stands as a statement of its own and the calls that use it refer to its
# `SSAValue`, while on 1.12 the same global is folded into the operand slots of those calls.
function _nfwd_record_stmt_consts!(consts::Vector{Any}, @nospecialize(st))
    if st isa Expr
        for a in st.args
            _nfwd_record_const!(consts, a)
        end
    elseif st isa Union{GlobalRef,QuoteNode}
        _nfwd_record_const!(consts, st)
    elseif st isa Union{Core.ReturnNode,Core.PiNode,Core.UpsilonNode}
        isdefined(st, :val) && _nfwd_record_const!(consts, st.val)
    elseif st isa Core.PhiNode
        for n in eachindex(st.values)
            isassigned(st.values, n) && _nfwd_record_const!(consts, st.values[n])
        end
    end
    return nothing
end

function _nfwd_scan_body!(work::Vector{Any}, consts::Vector{Any}, ci, @nospecialize(sig))
    ssatypes = ci.ssavaluetypes
    for st in ci.code
        _nfwd_record_stmt_consts!(consts, st)
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
                if s isa DataType && !isempty(s.parameters)
                    if s.parameters[1] in _NFWD_ARRAY_MUTATOR_TYPES &&
                        _nfwd_any_dual(ssatypes, sig, st.args)
                        return true
                    elseif s.parameters[1] in _NFWD_ELEMENTWISE_LA_TYPES &&
                        _nfwd_sig_has_dual_array(s) &&
                        _nfwd_la_rule_covers(s)
                        return true
                    end
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

function _nfwd_code_typed_unopt(@nospecialize(sig))
    try
        Base.code_typed_by_type(sig; optimize=false)
    catch
        nothing
    end
end

# Inverse of the dual map `_nfwd_safe` applies when it builds the dual signature. Only projectable
# duals are admitted there, so this is total over what can reach it; a non-differentiable argument
# is passed as its primal in BOTH signatures and maps to itself.
_nfwd_undual(@nospecialize(T)) = T
_nfwd_undual(::Type{Nfwd.NDual{T,N}}) where {T,N} = T
_nfwd_undual(::Type{Complex{Nfwd.NDual{T,N}}}) where {T,N} = Complex{T}
_nfwd_undual(::Type{<:Nfwd.NDualArray{T,N,D,A}}) where {T,N,D,A} = A
@static if VERSION >= v"1.11-rc4"  # `NDualMemoryRef` wraps `MemoryRef`, absent on Julia 1.10
    _nfwd_undual(::Type{<:Nfwd.NDualMemoryRef{T,N,M}}) where {T,N,M} = M
end
# Must invert every shape `_nfwd_projectable` admits, or the aggregates fall to the identity
# fallback above, `psig === dsig` holds, and the divergence check is skipped for exactly the
# arguments it should cover.
_nfwd_undual(::Type{T}) where {T<:Tuple} = Tuple{map(_nfwd_undual, T.parameters)...}
function _nfwd_undual(::Type{NamedTuple{names,T}}) where {names,T<:Tuple}
    return NamedTuple{names,_nfwd_undual(T)}
end

# Strip the lift from a constant inference folded at the DUAL signature, so it can be compared with
# the constant folded at the primal one: `sum(abs2, x)` folds `NDual{Float64,1}(0.0, (0.0,))`
# against `0.0`. Only the shapes that yield a NUMBER need a method, since the caller compares
# nothing else.
_nfwd_undual_const(@nospecialize(v)) = v
_nfwd_undual_const(v::Nfwd.NDual) = v.value
_nfwd_undual_const(v::Complex{<:Nfwd.NDual}) = Complex(v.re.value, v.im.value)

@inline function _nfwd_cond_type(ci, @nospecialize(cond))
    cond isa Core.SSAValue && return ci.ssavaluetypes[cond.id]
    cond isa Core.Argument && return ci.slottypes[cond.n]
    return cond
end

# Resolve a statement operand to a concrete TYPE in UNOPTIMISED IR, where operands are
# `SlotNumber`s and even the callee is an SSAValue holding a `Core.Const` of the function.
@inline function _nfwd_unopt_type(ci, @nospecialize(a))
    t = if a isa Core.SSAValue
        ci.ssavaluetypes[a.id]
    elseif a isa Core.SlotNumber
        ci.slottypes[a.id]
    elseif a isa Core.Argument
        ci.slottypes[a.n]
    elseif a isa GlobalRef
        isdefined(a.mod, a.name) ? Core.Typeof(getglobal(a.mod, a.name)) : nothing
    elseif a isa QuoteNode
        Core.Typeof(a.value)
    elseif a isa Expr
        nothing
    else
        Core.Typeof(a)
    end
    t === nothing && return nothing
    t isa Core.Const && return Core.Typeof(t.val)
    return Base.unwrapva(t)
end

# Callee signatures as they stand BEFORE inlining. The optimised work list cannot serve here: a
# small callee is inlined into its caller, so the branch it diverges on has already been folded
# away and the callee never appears as an `:invoke`. That is exactly the shape of the defect this
# check exists for -- `sz` inlines into `h1`, leaving `h1` with no branch of its own.
function _nfwd_unopt_callees!(out::Vector{Any}, ci)
    for st in ci.code
        (st isa Expr && st.head === :call && !isempty(st.args)) || continue
        ts = Any[]
        ok = true
        for a in st.args
            t = _nfwd_unopt_type(ci, a)
            if t === nothing || !isconcretetype(t)
                ok = false
                break
            end
            push!(ts, t)
        end
        ok || continue
        (ts[1] <: Core.Builtin || ts[1] <: Core.IntrinsicFunction) && continue
        push!(out, Tuple{ts...})
    end
    return out
end

# Returns false ONLY for a demonstrated divergence. Everything it cannot analyse is SKIPPED, not
# rejected: this check exists to add one specific rejection, and the whitelist scan it sits beside
# is unchanged, so refusing what it merely fails to understand would reject most of Base — measured,
# `sum(exp, x)`, broadcasts and `map` all fell out when unresolvable cases returned false.
# `sizeof`, `isa`, `typeof` and `nfields` answer questions about a value's TYPE, so a body that
# branches on one of them applied to a differentiated value takes the branch chosen for the `NDual`
# rather than for the primal, and returns a DIFFERENT FUNCTION's value with a matching derivative.
# Nothing downstream can detect that, so the classifier must: infer the same method at both
# specialisations and require every branch to agree.
#
# Compared UNOPTIMISED for two reasons. Optimisation destroys the statement correspondence the
# comparison needs, and dual arithmetic inlines differently everywhere, so optimised bodies always
# differ and comparing them would reject everything.
#
# Skipped when the two signatures resolve to DIFFERENT methods. A method written for duals
# (`sin(::NDual)`, and all of `low_level_maths.jl`) is the intended design rather than a divergence,
# and its statements have no correspondence with the primal method's.
function _nfwd_branches_agree_one(@nospecialize(dsig), work::Vector{Any})
    dsig isa DataType || return true
    psig = Tuple{map(_nfwd_undual, dsig.parameters)...}
    psig === dsig && return true                       # nothing differentiable to diverge on
    dm = _nfwd_which(dsig)
    pm = _nfwd_which(psig)
    (dm === nothing || pm === nothing) && return true
    dm === pm || return true
    dts, pts = _nfwd_code_typed_unopt(dsig), _nfwd_code_typed_unopt(psig)
    (dts === nothing || length(dts) != 1) && return true
    (pts === nothing || length(pts) != 1) && return true
    dci, pci = dts[1][1], pts[1][1]
    length(dci.code) == length(pci.code) || return true
    dtypes, ptypes = dci.ssavaluetypes, pci.ssavaluetypes
    for (i, st) in enumerate(dci.code)
        if dtypes isa Vector &&
            ptypes isa Vector &&
            i <= min(length(dtypes), length(ptypes))
            dt, pt = dtypes[i], ptypes[i]
            if dt isa Core.Const
                # Compare only NUMERIC folded constants. A number is what can flow into the
                # returned value, so a divergence there is a wrong answer: `sizeof(x)` folds to 16
                # for `NDual{Float64,1}` against 8 for `Float64` (`x * sizeof(x)` returns 32.0
                # where the primal is 16.0), and for an array only the DUAL side folds, since
                # `sizeof(::NDualArray)` is the wrapper's fixed size while
                # `sizeof(::Vector{Float64})` scales with length (`sum(x) * sizeof(x)` returned
                # 192.0 against 144.0). Type-level metadata is excluded because it SHOULD differ:
                # the dual genuinely has a different type, and on 1.12 `sum(sin.(x))` folds
                # `typename(NDualArray)` against `typename(Array)`.
                d = _nfwd_undual_const(dt.val)
                if d isa Number
                    (pt isa Core.Const && isequal(d, pt.val)) || return false
                end
            end
        end
        st isa Core.GotoIfNot || continue
        pst = pci.code[i]
        pst isa Core.GotoIfNot || return true
        _nfwd_cond_type(dci, st.cond) === _nfwd_cond_type(pci, pst.cond) || return false
    end
    _nfwd_unopt_callees!(work, dci)
    return true
end

_nfwd_which(@nospecialize(sig)) =
    try
        Base.which(sig)
    catch
        nothing
    end

function _nfwd_branches_agree(@nospecialize(dsig); maxnodes::Int=600)
    work = Any[dsig]
    visited = Set{Any}()
    nodes = 0
    while !isempty(work)
        s = pop!(work)
        s in visited && continue
        push!(visited, s)
        (nodes += 1) > maxnodes && return true
        _nfwd_branches_agree_one(s, work) || return false
    end
    return true
end

# Memoise the nfwd-safety verdict per dual signature. The verdict is a pure function of the
# signature and the inferred IR — fixed within a world — but computing it runs `code_typed` on the
# call and recursively on every reachable `:invoke`, so re-deriving it for every forward build of
# the same signature is wasted work. Cache keyed by signature and flushed whenever the world age
# advances, since new method definitions can change inference (and hence the verdict).
const _NFWD_SAFE_CACHE = Dict{Any,Tuple{Bool,ConstAliasSet}}()
const _NFWD_SAFE_WORLD = Ref{UInt}(typemax(UInt))
const _NFWD_SAFE_LOCK = ReentrantLock()

function _nfwd_body_safe_cached(@nospecialize(sig), @nospecialize(expected))
    w = Base.get_world_counter()
    return Base.@lock _NFWD_SAFE_LOCK begin
        if w != _NFWD_SAFE_WORLD[]
            empty!(_NFWD_SAFE_CACHE)
            _NFWD_SAFE_WORLD[] = w
        end
        get!(_NFWD_SAFE_CACHE, (sig, expected)) do
            consts = Any[]
            safe = _nfwd_body_safe(sig, expected, consts)
            # A rejected verdict abandons the walk part-way, so its constants are incomplete and
            # must not be handed to a rule; nothing takes that path anyway.
            (safe, ConstAliasSet(safe ? consts : Any[]))
        end
    end
end

# `expected` is the canonical dual of the primal result, `dual_type(Val(N), primal_return_type)`.
# Nfwd execution must reproduce exactly this shape: a function can behave differently on inner
# duals than on its primal (e.g. `TwicePrecision` arithmetic collapses an `NDual` back to a bare
# scalar, so `_logrange_extra` returns `Tuple{NDual,NDual}` on duals but `Tuple{TwicePrecision,…}`
# on floats), which would give a wrong primal and tangent. Requiring `rt === expected` rejects
# those and routes them to the transform.
function _nfwd_body_safe(
    @nospecialize(sig), @nospecialize(expected), consts::Vector{Any}; maxnodes::Int=600
)
    cts = _nfwd_code_typed(sig)
    (cts === nothing || length(cts) != 1) && return false
    ci, rt = cts[1]
    (isconcretetype(rt) && _nfwd_projectable(rt) && rt === expected) || return false
    _nfwd_branches_agree(sig) || return false
    work = Any[]
    _nfwd_scan_body!(work, consts, ci, sig) && return false
    visited = Set{Any}()
    nodes = 0
    while !isempty(work)
        s = pop!(work)
        s in visited && continue
        push!(visited, s)
        (nodes += 1) > maxnodes && return false
        cs = _nfwd_code_typed(s)
        (cs === nothing || length(cs) != 1) && return false
        _nfwd_branches_agree(s) || return false
        _nfwd_scan_body!(work, consts, cs[1][1], s) && return false
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
#
# The verdict is a property of `(sig_types, width)`, NOT of the function: the same function can be
# admitted at one chunk width and rejected at another. Rejection keys on callee identity, and
# inference inlines differently per width, so a callee on the reject list may be a visible `:invoke`
# at one width and inlined away — hence invisible — at the next. Never cache or reason about a
# verdict as "this function is nfwd-safe".
#
# Returns `(safe, consts)`: `consts` is the differentiable constants the call graph reads, which
# `NfwdFRule` carries so it can refuse an argument that is one of them. It is meaningful only when
# `safe` — every rejection path abandons the walk part-way.

# Shared rejection. No rule ever holds this set, so reusing the one empty vector is safe.
const _NFWD_UNSAFE = (false, ConstAliasSet())

function _nfwd_verdict(sig_types::Vector, width::Int)
    isempty(sig_types) && return _NFWD_UNSAFE
    all(isconcretetype, sig_types) || return _NFWD_UNSAFE
    # callee must be non-differentiable: nfwd call cannot carry a closure-field derivative.
    dual_type(Val(width), sig_types[1]) === NoDual || return _NFWD_UNSAFE
    dsig_args = Any[]
    for i in 2:length(sig_types)
        dt = dual_type(Val(width), sig_types[i])
        if dt === NoDual
            push!(dsig_args, sig_types[i])          # non-differentiable → dispatch on the primal
        elseif _nfwd_projectable(dt)
            push!(dsig_args, dt)                     # inner dual → dispatch on the dual
        else
            return _NFWD_UNSAFE                     # struct dual (ImmutableDual/…) → not nfwd
        end
    end
    # The nfwd dual result must equal the canonical dual of the primal result; infer the primal
    # return type and hand its `dual_type` to the safety check as the required shape.
    pcts = _nfwd_code_typed(Tuple{sig_types...})
    (pcts === nothing || length(pcts) != 1) && return _NFWD_UNSAFE
    prt = pcts[1][2]
    isconcretetype(prt) || return _NFWD_UNSAFE
    expected = try
        dual_type(Val(width), prt)
    catch
        return _NFWD_UNSAFE
    end
    return _nfwd_body_safe_cached(Tuple{sig_types[1],dsig_args...}, expected)
end

_nfwd_safe(sig_types::Vector, width::Int) = _nfwd_verdict(sig_types, width)[1]
