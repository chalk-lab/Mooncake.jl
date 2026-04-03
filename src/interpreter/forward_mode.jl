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

struct IRfwdMode{N} end

function _fwd_dual_type end
function _fwd_zero_dual end
function _fwd_uninit_dual end
@inline _fwd_primitive_rule(
    ::Any, interp, sig; debug_mode=false, silence_debug_messages=true
) = nothing

function build_frule end
function build_chunked_frule end

function _build_raw_frule(
    args...; debug_mode=false, silence_debug_messages=true, tangent_mode=IRfwdMode{1}()
)
    sig = _typeof(TestUtils.__get_primals(args))
    interp = get_interpreter(ForwardMode)
    return _build_raw_frule(interp, sig; debug_mode, silence_debug_messages, tangent_mode)
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
than the current world (e.g., when building rules for MistyClosure which uses its own world).
"""
function _build_raw_frule(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    debug_mode=false,
    silence_debug_messages=true,
    skip_world_age_check=false,
    tangent_mode=IRfwdMode{1}(),
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
        rule = _fwd_primitive_rule(
            tangent_mode, interp, sig; debug_mode, silence_debug_messages
        )
        isnothing(rule) && (rule = build_primitive_frule(sig))
        return debug_mode ? DebugFRule(rule) : rule
    end

    # We don't have a hand-coded rule, so derive one.
    lock(MOONCAKE_INFERENCE_LOCK)
    try
        # If we've already derived the OpaqueClosures and info, do not re-derive, just
        # create a copy and pass in new shared data.
        oc_cache_key = ClosureCacheKey(
            interp.world, (sig_or_mi, debug_mode, :forward, tangent_mode)
        )
        if haskey(interp.oc_cache, oc_cache_key)
            return interp.oc_cache[oc_cache_key]
        else
            # Derive forward-pass IR, and shove in a `MistyClosure`.
            dual_ir, captures, info = generate_dual_ir(
                interp, sig_or_mi; debug_mode, tangent_mode
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

@inline function (fwd::DerivedFRule{P,sig,isva,nargs})(
    args::Vararg{Dual,N}
) where {P,sig,N,isva,nargs}
    return fwd.fwd_oc(__unflatten_dual_varargs(isva, args, Val(nargs))...)
end

# On Julia 1.10, restore type stability lost to the inferencebarrier in __call_rule by
# asserting the return type, which is encoded in the MistyClosure type parameter.
@static if VERSION < v"1.11-"
    @inline function __call_rule(
        rule::DerivedFRule{P,MistyClosure{OpaqueClosure{A,R}},isva,nargs}, args
    ) where {P,A,R,isva,nargs}
        uf_args = __unflatten_dual_varargs(isva, args, Val(nargs))
        return __call_opaque_closure(rule.fwd_oc, uf_args)::R
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

"""
    __unflatten_dual_varargs(isva::Bool, args, ::Val{nargs}) where {nargs}

If isva and nargs=2, then inputs `(Dual(5.0, 0.0), Dual(4.0, 0.0), Dual(3.0, 0.0))`
are transformed into `(Dual(5.0, NTangent((0.0,))), Dual((5.0, 4.0), NTangent(((0.0, 0.0),))))`.
"""
function __unflatten_dual_varargs(
    isva::Bool, args::Tuple{Vararg{Any,N}}, ::Val{nargs}
) where {N,nargs}
    isva || return args
    group_primal = ntuple(i -> primal(args[nargs + i - 1]), Val(N - nargs + 1))
    if tangent_type(_typeof(group_primal)) == NoTangent
        grouped_args = zero_dual(group_primal)
    else
        grouped_args = Dual(
            group_primal,
            _canonical_forward_tangent(
                group_primal, ntuple(i -> tangent(args[nargs + i - 1]), Val(N - nargs + 1))
            ),
        )
    end
    return (ntuple(i -> args[i], Val(nargs - 1))..., grouped_args)
end

struct DualInfo
    primal_ir::IRCode
    interp::MooncakeInterpreter
    is_used::Vector{Bool}
    debug_mode::Bool
    tangent_mode
end

function generate_dual_ir(
    interp::MooncakeInterpreter,
    sig_or_mi;
    debug_mode=false,
    do_inline=true,
    tangent_mode=IRfwdMode{1}(),
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
    dual_ir = CC.copy(primal_ir)

    # Modify dual argument types:
    # - add one for the captures in the first position, with placeholder type for now
    # - convert the rest to dual types
    for (a, P) in enumerate(primal_ir.argtypes)
        dual_ir.argtypes[a] = _fwd_dual_type(tangent_mode, CC.widenconst(P))
    end
    pushfirst!(dual_ir.argtypes, Any)

    # Data structure into which we can push any data which is to live in the captures field
    # of the OpaqueClosure used to implement this rule. The index at which a piece of data
    # lives in this data structure is equal to the index of the captures field of the
    # OpaqueClosure in which it will live. To write code which retrieves items from the
    # captures data structure, make use of `get_capture`.
    captures = Any[]

    is_used = characterised_used_ssas(stmt(primal_ir.stmts))
    info = DualInfo(primal_ir, interp, is_used, debug_mode, tangent_mode)
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

    # Optimize dual IR
    dual_ir_opt = optimise_ir!(dual_ir; do_inline)
    return (
        dual_ir_opt,
        captures_tuple,
        DualRuleInfo(isva, nargs, dual_ret_type(primal_ir, tangent_mode)),
    )
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
function const_dual!(
    captures::Vector{Any}, stmt, tangent_mode=IRfwdMode{1}()
)::Union{Dual,Int}
    v = get_const_primal_value(stmt)
    x = _fwd_uninit_dual(tangent_mode, v)
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
        d = const_dual!(captures, stmt, info.tangent_mode)
        if d isa Int
            Mooncake.replace_call!(dual_ir, ssa, Expr(:call, get_capture, Argument(1), d))
        else
            Mooncake.replace_call!(dual_ir, ssa, Expr(:call, identity, d))
        end
    else
        new_ssa = CC.insert_node!(dual_ir, ssa, new_inst(stmt), ATTACH_BEFORE)
        zero_dual_call = Expr(:call, _fwd_zero_dual, info.tangent_mode, new_ssa)
        Mooncake.replace_call!(dual_ir, ssa, zero_dual_call)
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
    d = const_dual!(captures, stmt.val, info.tangent_mode)
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
        stmt.values[n] = _fwd_uninit_dual(
            info.tangent_mode, get_const_primal_value(stmt.values[n])
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        _fwd_dual_type(info.tangent_mode, CC.widenconst(get_ir(dual_ir, ssa, :type))),
    )
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PiNode, dual_ir::IRCode, ssa::SSAValue, ::Vector{Any}, info::DualInfo
)
    stmt == PiNode(nothing, Union{}) && return replace_call!(dual_ir, ssa, nothing)
    if stmt.val isa Union{Argument,SSAValue}
        v = __inc(stmt.val)
        # A PiNode narrows the primal value, not the whole Dual object. Rewriting
        #
        #     π(v, T)
        #
        # as
        #
        #     π(dual_v, Dual{T, ...})
        #
        # is unsound when the incoming dual was built from an abstract source such as
        # `Ref{Any}`. For example:
        #
        #     pi_node_tester(y::Ref{Any}) = isa(y[], Int) ? sin(y[]) : y[]
        #
        # can reach the `Int` branch with a value whose runtime primal is `5::Int` even
        # though the incoming dual was created from the abstract source type `Any`.
        # Pretending that the whole dual had already narrowed to `Dual{Int, ...}` caused
        # an invalid transformed program and eventually an illegal-instruction crash.
        # See the matching note in `src/tangents/dual.jl` for the constructor-side fix
        # that sharpens abstract source types to `typeof(x)` before this control-flow
        # refinement happens.
        #
        # The correct forward rewrite is:
        #   1. extract `primal(v)` and `tangent(v)`,
        #   2. apply the PiNode only to `primal(v)`,
        #   3. rebuild `Dual(refined_primal, tangent(v))`.
        #
        # This preserves the control-flow refinement on the primal while letting normal
        # dual construction recanonicalize the tangent against the refined runtime value.
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
        replace_call!(dual_ir, ssa, Expr(:call, Dual, refined_primal_ssa, tangent_ssa))
    else
        v = _fwd_uninit_dual(info.tangent_mode, get_const_primal_value(stmt.val))
        replace_call!(
            dual_ir,
            ssa,
            PiNode(v, _fwd_dual_type(info.tangent_mode, CC.widenconst(stmt.typ))),
        )
    end
    set_ir!(dual_ir, ssa, :type, _fwd_dual_type(info.tangent_mode, CC.widenconst(stmt.typ)))
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::UpsilonNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    if !(stmt.val isa Union{Argument,SSAValue})
        stmt = UpsilonNode(
            _fwd_uninit_dual(info.tangent_mode, get_const_primal_value(stmt.val))
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        _fwd_dual_type(info.tangent_mode, CC.widenconst(get_ir(dual_ir, ssa, :type))),
    )
    return nothing
end

function modify_fwd_ad_stmts!(
    stmt::PhiCNode, dual_ir::IRCode, ssa::SSAValue, captures::Vector{Any}, info::DualInfo
)
    for n in eachindex(stmt.values)
        isassigned(stmt.values, n) || continue
        stmt.values[n] isa Union{Argument,SSAValue} && continue
        stmt.values[n] = _fwd_uninit_dual(
            info.tangent_mode, get_const_primal_value(stmt.values[n])
        )
    end
    set_stmt!(dual_ir, ssa, inc_args(stmt))
    set_ir!(
        dual_ir,
        ssa,
        :type,
        _fwd_dual_type(info.tangent_mode, CC.widenconst(get_ir(dual_ir, ssa, :type))),
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

__get_primal(x::Dual) = primal(x)

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

        interp = info.interp

        # If every non-function input type is non-differentiable, there is usually no
        # forward work to do for this call. Evaluate the primal path and wrap the result in
        # a zero dual directly rather than routing through derived rule construction.
        #
        # Keep primitives on the normal rule path even in this case. Some primitives are
        # only valid through their dedicated frule!!, e.g. `_foreigncall_(Val(:jl_type_unionall), ...)`
        # which should stay on the zero-derivative primitive rule rather than replaying the
        # raw foreigncall through `__fwds_pass_no_ad!`.
        if all(T -> tangent_type(T) === NoTangent, @view(sig_types[2:end])) &&
            !is_primitive(context_type(info.interp), ForwardMode, sig, info.interp.world)
            primal_ssa = CC.insert_node!(
                dual_ir,
                ssa,
                new_inst(Expr(:call, __fwds_pass_no_ad!, args...)),
                ATTACH_BEFORE,
            )
            replace_call!(
                dual_ir, ssa, Expr(:call, _fwd_zero_dual, info.tangent_mode, primal_ssa)
            )
            return nothing
        end

        # Dual-ise arguments.
        dual_args = map(args) do arg
            arg isa Union{Argument,SSAValue} && return arg
            return _fwd_uninit_dual(info.tangent_mode, get_const_primal_value(arg))
        end

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
            push!(
                captures,
                if isexpr(stmt, :invoke)
                    LazyFRule(mi, dm, info.tangent_mode)
                else
                    DynamicFRule(dm, info.tangent_mode)
                end,
            )
            get_rule = Expr(:call, get_capture, Argument(1), length(captures))
            rule_ssa = CC.insert_node!(dual_ir, ssa, new_inst(get_rule), ATTACH_BEFORE)
            args_ssa = CC.insert_node!(
                dual_ir, ssa, new_inst(Expr(:call, tuple, dual_args...)), ATTACH_BEFORE
            )
            replace_call!(dual_ir, ssa, Expr(:call, __call_rule, rule_ssa, args_ssa))
        end
    elseif isexpr(stmt, :boundscheck)
        # Keep the boundscheck, but put it in a Dual.
        inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        bc_ssa = CC.insert_node!(dual_ir, ssa, inst, ATTACH_BEFORE)
        replace_call!(dual_ir, ssa, Expr(:call, _fwd_zero_dual, info.tangent_mode, bc_ssa))
    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(dual_ir, ssa, nothing)
    elseif Meta.isexpr(stmt, :copyast)
        new_copyast_inst = CC.NewInstruction(get_ir(info.primal_ir, ssa))
        new_copyast_ssa = CC.insert_node!(dual_ir, ssa, new_copyast_inst, ATTACH_BEFORE)
        replace_call!(
            dual_ir, ssa, Expr(:call, _fwd_zero_dual, info.tangent_mode, new_copyast_ssa)
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

mutable struct LazyFRule{primal_sig,Trule,Tmode}
    debug_mode::Bool
    tangent_mode::Tmode
    mi::Core.MethodInstance
    rule::Trule
    function LazyFRule(
        mi::Core.MethodInstance, debug_mode::Bool, tangent_mode=IRfwdMode{1}()
    )
        interp = get_interpreter(ForwardMode)
        return new{
            mi.specTypes,frule_type(interp, mi;debug_mode,tangent_mode),typeof(tangent_mode)
        }(
            debug_mode, tangent_mode, mi
        )
    end
    function LazyFRule{Tprimal_sig,Trule,Tmode}(
        mi::Core.MethodInstance, debug_mode::Bool, tangent_mode::Tmode
    ) where {Tprimal_sig,Trule,Tmode}
        return new{Tprimal_sig,Trule,Tmode}(debug_mode, tangent_mode, mi)
    end
end

# Create new lazy rule with same method instance and debug mode
_copy(x::P) where {P<:LazyFRule} = P(x.mi, x.debug_mode, x.tangent_mode)

@inline _canonicalise_fwd_arg(x) = x
@inline function _canonicalise_fwd_arg(x::Dual{P,<:NTangent}) where {P}
    tangent_type(typeof(primal(x))) == NoTangent || return x
    return Dual{P,NoTangent}(primal(x), NoTangent())
end

@inline function (rule::LazyFRule)(args::Vararg{Any,N}) where {N}
    canonical_args = map(_canonicalise_fwd_arg, args)
    return if isdefined(rule, :rule)
        __call_rule(rule.rule, canonical_args)
    else
        _build_rule!(rule, canonical_args)
    end
end

@noinline function _build_rule!(rule::LazyFRule{sig,Trule}, args) where {sig,Trule}
    interp = get_interpreter(ForwardMode)
    rule.rule = _build_raw_frule(
        interp, rule.mi; debug_mode=rule.debug_mode, tangent_mode=rule.tangent_mode
    )
    return __call_rule(rule.rule, args)
end

function dual_ret_type(primal_ir::IRCode, tangent_mode=IRfwdMode{1}())
    return _fwd_dual_type(tangent_mode, compute_ir_rettype(primal_ir))
end

function frule_type(
    interp::MooncakeInterpreter{C},
    mi::CC.MethodInstance;
    debug_mode,
    tangent_mode=IRfwdMode{1}(),
) where {C}
    sig = _get_sig(mi)
    if is_primitive(C, ForwardMode, sig, interp.world)
        # Build the rule to obtain its concrete type. For non-singleton primitive rules
        # (e.g. NfwdMooncake.Rule) this allocates a throwaway instance; the cost is compile-
        # time only and does not affect hot-path performance.
        rule = _fwd_primitive_rule(tangent_mode, interp, sig; debug_mode)
        isnothing(rule) && (rule = build_primitive_frule(sig))
        return debug_mode ? DebugFRule{typeof(rule)} : typeof(rule)
    end
    ir, _ = lookup_ir(interp, mi)
    nargs = length(ir.argtypes)
    isva, _ = is_vararg_and_sparam_names(mi)
    arg_types = map(CC.widenconst, ir.argtypes)
    sig = Tuple{arg_types...}
    dual_args_type = Tuple{map(Base.Fix1(_fwd_dual_type, tangent_mode), arg_types)...}
    closure_type = RuleMC{dual_args_type,dual_ret_type(ir, tangent_mode)}
    Tderived_rule = DerivedFRule{sig,closure_type,isva,nargs}
    return debug_mode ? DebugFRule{Tderived_rule} : Tderived_rule
end

struct DynamicFRule{V,Tmode}
    cache::V
    debug_mode::Bool
    tangent_mode::Tmode
end

function DynamicFRule(debug_mode::Bool, tangent_mode=IRfwdMode{1}())
    DynamicFRule(Dict{Any,Any}(), debug_mode, tangent_mode)
end

# Create new dynamic rule with empty cache and same debug mode  
_copy(x::P) where {P<:DynamicFRule} = P(Dict{Any,Any}(), x.debug_mode, x.tangent_mode)

function (dynamic_rule::DynamicFRule)(args::Vararg{Dual,N}) where {N}
    # `Base._stable_typeof` must be used here, rather than `typeof` or `Mooncake._typeof`.
    # See DynamicDerivedRule for details, the same reasoning applies.
    sig = Tuple{map(Base._stable_typeof ∘ primal, args)...}
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        interp = get_interpreter(ForwardMode)
        rule = _build_raw_frule(
            interp,
            sig;
            debug_mode=dynamic_rule.debug_mode,
            tangent_mode=dynamic_rule.tangent_mode,
        )
        dynamic_rule.cache[sig] = rule
    end
    canonical_args = map(_canonicalise_fwd_arg, args)
    return __call_rule(rule, canonical_args)
end
