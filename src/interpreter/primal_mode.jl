# Primal-mode forward AD: single lifted OC for both primal (Val(0)) and dual (Val(N))
# execution. See ~/notes/mooncake/primal-mode.md for design rationale.

# Construct a PrimalMode interpreter with BugPatchInterpreter in meta, giving the
# inliner access to the native Julia code cache (where frule!! bodies live).
function _make_primal_mode_interp(::Type{C}) where {C}
    return MooncakeInterpreter(C, PrimalMode; meta=BugPatchInterpreter())
end

# Initialize PrimalMode interpreter (deferred from abstract_interpretation.jl because
# BugPatchInterpreter is defined in patch_for_319.jl, loaded after abstract_interpretation).
GLOBAL_INTERPRETERS[PrimalMode] = _make_primal_mode_interp(DefaultCtx)

# _prim_call dispatches primitive calls by tangent_mode:
# Val(0) → direct primal call, Val(N) → frule!! call.
@inline _prim_call(::Val{0}, f, args...) = f(args...)
@inline _prim_call(::Val{N}, f, args...) where {N} = frule!!(f, args...)

# _primal_of extracts the primal boolean for control flow.
@inline _primal_of(x::Bool) = x
@inline _primal_of(x) = primal(x)

# __get_primal for Dual values — reverse_mode.jl defines the CoDual overload and the
# generic fallback; this adds the Dual case needed by __fwds_pass_no_ad!.
__get_primal(x::Dual) = primal(x)

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

@inline get_capture(captures::T, n::Int) where {T} = captures[n]

"""
    const_dual!(captures::Vector{Any}, stmt)::Union{Dual,Int}

Build a `Dual` from `stmt`, with zero / uninitialised tangent. If the resulting `Dual` is
a bits type, then it is returned. If it is not, then the `Dual` is put into captures,
and its location in `captures` returned.

Whether or not the value is a literal, or an index into the captures, can be determined from
the return type.
"""
function const_dual!(captures::Vector{Any}, stmt)::Union{Dual,Int}
    v = get_const_primal_value(stmt)
    x = uninit_dual(v)
    if safe_for_literal(v)
        return x
    else
        push!(captures, x)
        return length(captures)
    end
end

const ATTACH_AFTER = true
const ATTACH_BEFORE = false

"""
    __unflatten_dual_varargs(isva::Bool, args, ::Val{nargs}) where {nargs}

If isva and nargs=2, then inputs `(Dual(5.0, 0.0), Dual(4.0, 0.0), Dual(3.0, 0.0))`
are transformed into `(Dual(5.0, 0.0), Dual((5.0, 4.0), (0.0, 0.0)))`.
"""
function __unflatten_dual_varargs(isva::Bool, args, ::Val{nargs}) where {nargs}
    isva || return args
    group_primal = map(primal, args[nargs:end])
    if tangent_type(_typeof(group_primal)) == NoTangent
        grouped_args = zero_dual(group_primal)
    else
        grouped_args = Dual(group_primal, map(tangent, args[nargs:end]))
    end
    return (args[1:(nargs - 1)]..., grouped_args)
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

# Public entry point

function build_frule(args...; debug_mode=false, silence_debug_messages=true)
    sig = _typeof(TestUtils.__get_primals(args))
    interp = get_interpreter(ForwardMode)
    return build_frule(interp, sig; debug_mode, silence_debug_messages)
end

struct PrimalRuleInfo
    isva::Bool
    nargs::Int
    lifted_ret_type::Type
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
than the current world (e.g., when building rules for MistyClosure which uses its own world).
"""
function build_frule(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    debug_mode=false,
    silence_debug_messages=true,
    skip_world_age_check=false,
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

    # We don't have a hand-coded rule, so derive one.
    lock(MOONCAKE_INFERENCE_LOCK)
    try
        # If we've already derived the OpaqueClosures and info, do not re-derive, just
        # create a copy and pass in new shared data.
        oc_cache_key = ClosureCacheKey(interp.world, (sig_or_mi, debug_mode, :forward))
        if haskey(interp.oc_cache, oc_cache_key)
            return interp.oc_cache[oc_cache_key]
        else
            # Derive forward-pass IR, and shove in a `MistyClosure`.
            lifted_ir, captures, info = generate_lifted_ir(interp, sig_or_mi; debug_mode)
            lifted_oc = misty_closure(
                info.lifted_ret_type, lifted_ir, captures...; do_compile=true
            )
            sig = flatten_va_sig(sig, info.isva, info.nargs)
            raw_rule = DerivedPrimal{sig,typeof(lifted_oc),info.isva,info.nargs}(lifted_oc)
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

# DerivedPrimal — compiled forward rule from lifted OC

struct DerivedPrimal{primal_sig,Tlifted_oc,isva,nargs}
    lifted_oc::Tlifted_oc
end

@inline function (fwd::DerivedPrimal{P,sig,isva,nargs})(
    args::Vararg{Dual,N}
) where {P,sig,N,isva,nargs}
    return fwd.lifted_oc(__unflatten_dual_varargs(isva, args, Val(nargs))...)
end

# On Julia 1.10, restore type stability lost to the inferencebarrier in __call_rule by
# asserting the return type, which is encoded in the MistyClosure type parameter.
@static if VERSION < v"1.11-"
    @inline function __call_rule(
        rule::DerivedPrimal{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}, args
    ) where {P,A,R,isva,nargs}
        return __call_rule_erased!(Base.inferencebarrier(rule), args)::R
    end
end

# Copy forward rule with recursively copied captures
function _copy(x::P) where {P<:DerivedPrimal}
    return P(replace_captures(x.lifted_oc, _copy(x.lifted_oc.oc.captures)))
end

_isva(::DerivedPrimal{P,T,isva,nargs}) where {P,T,isva,nargs} = isva
_nargs(::DerivedPrimal{P,T,isva,nargs}) where {P,T,isva,nargs} = nargs

# Extends functionality defined in debug_mode.jl.
function verify_args(r::DerivedPrimal{sig}, x) where {sig}
    Tx = Tuple{
        map(_typeof ∘ primal, __unflatten_dual_varargs(_isva(r), x, Val(_nargs(r))))...
    }
    Tx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tx do not subtype rule signature, $sig"))
end

# LazyPrimal — deferred rule for known invoke sites

mutable struct LazyPrimal{primal_sig,Trule}
    debug_mode::Bool
    mi::Core.MethodInstance
    rule::Trule
    function LazyPrimal(mi::Core.MethodInstance, debug_mode::Bool)
        interp = get_interpreter(ForwardMode)
        return new{mi.specTypes,primal_rule_type(interp, mi;debug_mode)}(debug_mode, mi)
    end
    function LazyPrimal{Tprimal_sig,Trule}(
        mi::Core.MethodInstance, debug_mode::Bool
    ) where {Tprimal_sig,Trule}
        return new{Tprimal_sig,Trule}(debug_mode, mi)
    end
end

# Create new lazy rule with same method instance and debug mode
_copy(x::P) where {P<:LazyPrimal} = P(x.mi, x.debug_mode)

# On Julia 1.10, the generic __call_rule fallback is @stable-checked and returns Any for
# LazyPrimal, triggering TypeInstabilityError when dispatch_doctor_mode = "error".
@static if VERSION < v"1.11-"
    @inline function __call_rule(
        rule::LazyPrimal{sig,DerivedPrimal{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}},
        args,
    ) where {sig,P,A,R,isva,nargs}
        return rule(args...)::R
    end
    @inline function __call_rule(
        rule::LazyPrimal{
            sig,DebugFRule{DerivedPrimal{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}}
        },
        args,
    ) where {sig,P,A,R,isva,nargs}
        return rule(args...)::R
    end
end

@inline function (rule::LazyPrimal)(args::Vararg{Any,N}) where {N}
    return isdefined(rule, :rule) ? __call_rule(rule.rule, args) : _build_rule!(rule, args)
end

@noinline function _build_rule!(rule::LazyPrimal{sig,Trule}, args) where {sig,Trule}
    interp = get_interpreter(ForwardMode)
    rule.rule = build_frule(interp, rule.mi; debug_mode=rule.debug_mode)
    return __call_rule(rule.rule, args)
end

# DynamicPrimal — dict-cached rule for dynamic callsites

struct DynamicPrimal{V}
    cache::V
    debug_mode::Bool
end

DynamicPrimal(debug_mode::Bool) = DynamicPrimal(Dict{Any,Any}(), debug_mode)

# Create new dynamic rule with empty cache and same debug mode  
_copy(x::P) where {P<:DynamicPrimal} = P(Dict{Any,Any}(), x.debug_mode)

function (dynamic_rule::DynamicPrimal)(args::Vararg{Dual,N}) where {N}
    sig = Tuple{map(Base._stable_typeof ∘ primal, args)...}
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        interp = get_interpreter(ForwardMode)
        rule = build_frule(interp, sig; debug_mode=dynamic_rule.debug_mode)
        dynamic_rule.cache[sig] = rule
    end
    return __call_rule(rule, args)
end

# IR generation — lifted primal IR

struct LiftedInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
    is_used::Vector{Bool}
    debug_mode::Bool
end

function generate_lifted_ir(
    interp::MooncakeInterpreter,
    sig_or_mi;
    debug_mode=false,
    do_inline=true,
    do_optimize=true,
)
    # Reset id count. This ensures that the IDs generated are the same each time this
    # function runs.
    seed_id!()

    # Grab code associated to the primal.
    primal_ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        primal_ir = set_valid_world!(primal_ir, interp.world)
    end
    nargs = length(primal_ir.argtypes)

    # Normalise the IR.
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)

    # Keep a copy of the primal IR with the insertions
    lifted_ir = CC.copy(primal_ir)

    # Modify lifted argument types:
    # - add one for the captures in the first position, with placeholder type for now
    # - convert the rest to dual types
    for (a, P) in enumerate(primal_ir.argtypes)
        lifted_ir.argtypes[a] = dual_type(CC.widenconst(P))
    end
    pushfirst!(lifted_ir.argtypes, Any)

    # Data structure for captures.
    captures = Any[]

    is_used = characterised_used_ssas(stmt(primal_ir.stmts))
    info = LiftedInfo(primal_ir, interp, is_used, debug_mode)
    for (n, inst) in enumerate(lifted_ir.stmts)
        ssa = SSAValue(n)
        modify_primal_stmts!(stmt(inst), lifted_ir, ssa, captures, info)
    end

    # Process new nodes etc.
    lifted_ir = CC.compact!(lifted_ir)

    CC.verify_ir(lifted_ir)

    # Now that the captured values are known, replace the placeholder value given for the
    # first argument type with the actual type.
    captures_tuple = (captures...,)
    lifted_ir.argtypes[1] = _typeof(captures_tuple)

    # Inspection tools need the pre-optimization lifted IR, while the AD pipeline still
    # wants the optimized form by default.
    lifted_ir = do_optimize ? optimise_ir!(lifted_ir; do_inline) : lifted_ir
    return lifted_ir,
    captures_tuple,
    PrimalRuleInfo(isva, nargs, lifted_ret_type(primal_ir))
end

function lifted_ret_type(primal_ir::IRCode)
    return dual_type(compute_ir_rettype(primal_ir))
end

function primal_rule_type(
    interp::MooncakeInterpreter{C}, mi::CC.MethodInstance; debug_mode
) where {C}
    sig = _get_sig(mi)
    if is_primitive(C, ForwardMode, sig, interp.world)
        rule = build_primitive_frule(sig)
        return debug_mode ? DebugFRule{typeof(rule)} : typeof(rule)
    end
    ir, _ = lookup_ir(interp, mi)
    nargs = length(ir.argtypes)
    isva, _ = is_vararg_and_sparam_names(mi)
    arg_types = map(CC.widenconst, ir.argtypes)
    sig = Tuple{arg_types...}
    dual_args_type = Tuple{map(dual_type, arg_types)...}
    closure_type = RuleMC{dual_args_type,lifted_ret_type(ir)}
    Tderived = DerivedPrimal{sig,closure_type,isva,nargs}
    return debug_mode ? DebugFRule{Tderived} : Tderived
end

# Statement modification — lifted primal IR rewrite

modify_primal_stmts!(::Nothing, ::IRCode, ::SSAValue, ::Vector{Any}, ::LiftedInfo) = nothing

function modify_primal_stmts!(::GotoNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::LiftedInfo)
    nothing
end

function modify_primal_stmts!(
    stmt::GotoIfNot,
    lifted_ir::IRCode,
    ssa::SSAValue,
    captures::Vector{Any},
    info::LiftedInfo,
)
    # Extract primal boolean via _primal_of for control flow.
    Mooncake.replace_call!(lifted_ir, ssa, Expr(:call, _primal_of, inc_args(stmt).cond))

    # reinsert the GotoIfNot right after the call to _primal_of
    new_gotoifnot_inst = new_inst(Core.GotoIfNot(ssa, stmt.dest))
    CC.insert_node!(lifted_ir, ssa, new_gotoifnot_inst, ATTACH_AFTER)
    return nothing
end

function modify_primal_stmts!(
    stmt::GlobalRef, lifted_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::LiftedInfo
)
    if isconst(stmt)
        d = const_dual!(captures, stmt)
        if d isa Int
            Mooncake.replace_call!(lifted_ir, ssa, Expr(:call, get_capture, Argument(1), d))
        else
            Mooncake.replace_call!(lifted_ir, ssa, Expr(:call, identity, d))
        end
    else
        new_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(stmt), ATTACH_BEFORE)
        zero_dual_call = Expr(:call, Mooncake.zero_dual, new_ssa)
        Mooncake.replace_call!(lifted_ir, ssa, zero_dual_call)
    end

    return nothing
end

function modify_primal_stmts!(
    stmt::ReturnNode, lifted_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::LiftedInfo
)
    # undefined `val` field means that stmt is unreachable.
    isdefined(stmt, :val) || return nothing

    # stmt is an Argument, then already a dual, and must just be incremented.
    if stmt.val isa Union{Argument,SSAValue}
        Mooncake.replace_call!(lifted_ir, ssa, ReturnNode(__inc(stmt.val)))
        return nothing
    end

    # stmt is a const, so we have to turn it into a dual.
    d = const_dual!(captures, stmt.val)
    if d isa Int
        get_dual = Expr(:call, get_capture, Argument(1), d)
        get_dual_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(get_dual), ATTACH_BEFORE)
        Mooncake.replace_call!(lifted_ir, ssa, ReturnNode(get_dual_ssa))
    else
        Mooncake.replace_call!(lifted_ir, ssa, ReturnNode(d))
    end
    return nothing
end

function modify_primal_stmts!(
    stmt::PhiNode, lifted_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::LiftedInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = uninit_dual(get_const_primal_value(stmt.values[n]))
    end
    set_stmt!(lifted_ir, ssa, inc_args(stmt))
    set_ir!(lifted_ir, ssa, :type, dual_type(CC.widenconst(get_ir(lifted_ir, ssa, :type))))
    return nothing
end

function modify_primal_stmts!(
    stmt::PiNode, lifted_ir::IRCode, ssa::SSAValue, ::Vector{Any}, ::LiftedInfo
)
    if stmt.val isa Union{Argument,SSAValue}
        v = __inc(stmt.val)
    else
        v = uninit_dual(get_const_primal_value(stmt.val))
    end
    replace_call!(lifted_ir, ssa, PiNode(v, dual_type(CC.widenconst(stmt.typ))))
    return nothing
end

function modify_primal_stmts!(
    stmt::UpsilonNode, lifted_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::LiftedInfo
)
    if !(stmt.val isa Union{Argument,SSAValue})
        stmt = UpsilonNode(uninit_dual(get_const_primal_value(stmt.val)))
    end
    set_stmt!(lifted_ir, ssa, inc_args(stmt))
    set_ir!(lifted_ir, ssa, :type, dual_type(CC.widenconst(get_ir(lifted_ir, ssa, :type))))
    return nothing
end

function modify_primal_stmts!(
    stmt::PhiCNode, lifted_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, ::LiftedInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = uninit_dual(get_const_primal_value(stmt.values[n]))
    end
    set_stmt!(lifted_ir, ssa, inc_args(stmt))
    set_ir!(lifted_ir, ssa, :type, dual_type(CC.widenconst(get_ir(lifted_ir, ssa, :type))))
    return nothing
end

@static if isdefined(Core, :EnterNode)
    function modify_primal_stmts!(
        ::Core.EnterNode, ::IRCode, ::SSAValue, ::Vector{Any}, ::LiftedInfo
    )
        return nothing
    end
end

## Modification of IR nodes - expressions

function modify_primal_stmts!(
    stmt::Expr, lifted_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::LiftedInfo
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        raw_args = isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args
        sig_types = map(raw_args) do x
            t = CC.widenconst(get_forward_primal_type(info.primal_ir, x))
            return contains_bottom_type(t) ? Any : t
        end
        sig = Tuple{sig_types...}
        mi = isexpr(stmt, :invoke) ? get_mi(stmt.args[1]) : missing
        args = map(__inc, raw_args)

        # Special case: if the result of a call to getfield is un-used, then leave the
        # primal statement alone (just increment arguments as usual).
        if !info.is_used[ssa.id] && get_const_primal_value(args[1]) == getfield
            fwds = new_inst(Expr(:call, __fwds_pass_no_ad!, args...))
            replace_call!(lifted_ir, ssa, fwds)
            return nothing
        end

        # Dual-ise arguments.
        dual_args = map(args) do arg
            arg isa Union{Argument,SSAValue} && return arg
            return uninit_dual(get_const_primal_value(arg))
        end

        interp = info.interp
        if is_primitive(context_type(interp), ForwardMode, sig, interp.world)
            rule = build_primitive_frule(sig)
            if safe_for_literal(rule)
                replace_call!(lifted_ir, ssa, Expr(:call, rule, dual_args...))
            else
                push!(captures, rule)
                get_rule = Expr(:call, get_capture, Argument(1), length(captures))
                rule_ssa = CC.insert_node!(
                    lifted_ir, ssa, new_inst(get_rule), ATTACH_BEFORE
                )
                replace_call!(lifted_ir, ssa, Expr(:call, rule_ssa, dual_args...))
            end
        else
            dm = info.debug_mode
            push!(captures, isexpr(stmt, :invoke) ? LazyPrimal(mi, dm) : DynamicPrimal(dm))
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(lifted_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            replace_call!(lifted_ir, ssa, Expr(:call, rule_ssa, dual_args...))
        end
    elseif isexpr(stmt, :boundscheck)
        # Keep the boundscheck, but put it in a Dual.
        inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        bc_ssa = CC.insert_node!(lifted_ir, ssa, inst, ATTACH_BEFORE)
        replace_call!(lifted_ir, ssa, Expr(:call, zero_dual, bc_ssa))
    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(lifted_ir, ssa, nothing)
    elseif Meta.isexpr(stmt, :copyast)
        new_copyast_inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        new_copyast_ssa = CC.insert_node!(lifted_ir, ssa, new_copyast_inst, ATTACH_BEFORE)
        replace_call!(lifted_ir, ssa, Expr(:call, zero_dual, new_copyast_ssa))
    elseif Meta.isexpr(stmt, :loopinfo)
        # Leave this node alone.
    elseif isexpr(stmt, :throw_undef_if_not)
        # args[1] is a Symbol, args[2] is the condition which must be primalized
        primal_cond = Expr(:call, _primal_of, inc_args(stmt).args[2])
        replace_call!(lifted_ir, ssa, primal_cond)
        new_undef_inst = new_inst(Expr(:throw_undef_if_not, stmt.args[1], ssa))
        CC.insert_node!(lifted_ir, ssa, new_undef_inst, ATTACH_AFTER)
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
