@zero_derivative MinimalCtx Tuple{typeof(get_interpreter),Type{<:Mode}}
@zero_derivative MinimalCtx Tuple{typeof(build_rrule_checks),MooncakeInterpreter,Bool,Bool}
# TODO: is this still needed?
@zero_derivative MinimalCtx Tuple{typeof(is_primitive),Type,Type{<:Mode},Type,UInt}

@is_primitive MinimalCtx Tuple{
    typeof(build_derived_rrule),MooncakeInterpreter{C},Any,Any,Bool
} where {C}

function frule!!(
    ::Dual{typeof(build_derived_rrule)},
    _interp::Dual{<:MooncakeInterpreter},
    _sig_or_mi::Dual,
    _sig::Dual,
    _debug_mode::Dual{Bool},
)
    interp = primal(_interp)
    sig_or_mi = primal(_sig_or_mi)
    sig = primal(_sig)
    debug_mode = primal(_debug_mode)

    # Derive **unoptimized** forwards- and reverse-pass IR.
    dri = generate_ir(interp, sig_or_mi; debug_mode, do_optimize=false)

    # Compute the signature. Needs careful handling with varargs.
    nargs = num_args(dri.info)
    if dri.isva
        sig = Tuple{
            sig.parameters[1:(nargs - 1)]...,Tuple{sig.parameters[nargs:end]...}
        }
    end

    # Optimize as much as possible, then generate primal
    raw_rule = let
        optimized_fwd_ir = optimise_ir!(CC.copy(dri.fwd_ir))
        optimized_rvs_ir = optimise_ir!(CC.copy(dri.rvs_ir))
        fwd_oc = misty_closure(dri.fwd_ret_type, optimized_fwd_ir, dri.shared_data...)
        rvs_oc = misty_closure(dri.rvs_ret_type, optimized_rvs_ir, dri.shared_data...)

        DerivedRule(sig, fwd_oc, Ref(rvs_oc), dri.isva, Val(nargs))
    end

    # Generate dual rule
    raw_rule_tangent = let
        # Optimize as much as possible, but with a forward-mode interpreter
        # that will block inlining of frules
        # TODO: which ctx to use here?
        interp_forward = MooncakeInterpreter(DefaultCtx, ForwardMode; world=interp.world)

        optimized_fwd_ir = optimise_ir!(dri.fwd_ir; interp=interp_forward)
        optimized_rvs_ir = optimise_ir!(dri.rvs_ir; interp=interp_forward)
        fwd_oc = misty_closure(dri.fwd_ret_type, optimized_fwd_ir, dri.shared_data...)
        rvs_oc = misty_closure(dri.rvs_ret_type, optimized_rvs_ir, dri.shared_data...)

        # Build tangents at the same time to preserve aliasing (e.g. for comms)
        captures_tangent = zero_tangent((fwd_oc.oc.captures, rvs_oc.oc.captures))

        # TODO: debug_mode=true makes the rule fail somehow; something related to varargs
        fwd_oc_tangent = MistyClosureTangent(
            captures_tangent[1],
            build_frule(interp_forward, fwd_oc; skip_world_age_check=true, debug_mode=false)
        )
        rvs_oc_tangent = MistyClosureTangent(
            captures_tangent[2],
            build_frule(interp_forward, rvs_oc; skip_world_age_check=true, debug_mode=false)
        )

        Tangent((;
            fwds_oc=fwd_oc_tangent,
            pb_oc_ref=MutableTangent((; x=PossiblyUninitTangent(rvs_oc_tangent))),
            nargs=NoTangent(),
        ))
    end

    if debug_mode
        debug_rule_tangent = Tangent((; rule=raw_rule_tangent))
        return Dual(DebugRRule(raw_rule), debug_rule_tangent)
    else
        return Dual(raw_rule, raw_rule_tangent)
    end
end

function rrule!!(
    ::CoDual{typeof(build_derived_rrule)},
    _interp::CoDual{<:MooncakeInterpreter},
    _sig_or_mi::CoDual,
    _sig::CoDual,
    _debug_mode::CoDual{Bool},
)
    throw(
        ArgumentError(
            "Reverse-over-reverse differentiation is not supported. " *
            "Encountered attempt to differentiate build_derived_rrule in reverse mode.",
        ),
    )
end

# TODO: This is a workaround for forward-over-reverse. Primitives in reverse mode can get
# inlined when building the forward rule, exposing internal ccalls that lack an frule!!.
# For example, `dataids` is a reverse-mode primitive, but inlining it exposes
# `jl_genericmemory_owner`. The proper fix is to prevent primitive inlining during
# forward-over-reverse by forwarding `inlining_policy` through `BugPatchInterpreter` to
# `MooncakeInterpreter` during `optimise_ir!`, but this causes allocation regressions.
# See https://github.com/chalk-lab/Mooncake.jl/pull/878 for details.
# @static if VERSION >= v"1.11-"
#     function frule!!(
#         ::Dual{typeof(_foreigncall_)},
#         ::Dual{Val{:jl_genericmemory_owner}},
#         ::Dual{Val{Any}},
#         ::Dual{Tuple{Val{Any}}},
#         ::Dual{Val{0}},
#         ::Dual{Val{:ccall}},
#         a::Dual{<:Memory},
#     )
#         return zero_dual(ccall(:jl_genericmemory_owner, Any, (Any,), primal(a)))
#     end
#     function rrule!!(
#         ::CoDual{typeof(_foreigncall_)},
#         ::CoDual{Val{:jl_genericmemory_owner}},
#         ::CoDual{Val{Any}},
#         ::CoDual{Tuple{Val{Any}}},
#         ::CoDual{Val{0}},
#         ::CoDual{Val{:ccall}},
#         a::CoDual{<:Memory},
#     )
#         y = zero_fcodual(ccall(:jl_genericmemory_owner, Any, (Any,), primal(a)))
#         return y, NoPullback(ntuple(_ -> NoRData(), 7))
#     end
# end

# TODO: is this still needed?
@zero_derivative MinimalCtx Tuple{typeof(zero_tangent),Any}

# TODO: is this still needed?
@static if VERSION < v"1.11-"
    @generated function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_alloc_array_1d}},
        ::Dual{Val{Vector{P}}},
        ::Dual{Tuple{Val{Any},Val{Int}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        ::Dual{Type{Vector{P}}},
        n::Dual{Int},
        args::Vararg{Dual},
    ) where {P}
        T = tangent_type(P)
        return quote
            _n = primal(n)
            y = ccall(:jl_alloc_array_1d, Vector{$P}, (Any, Int), Vector{$P}, _n)
            dy = ccall(:jl_alloc_array_1d, Vector{$T}, (Any, Int), Vector{$T}, _n)
            return Dual(y, dy)
        end
    end
    @generated function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_alloc_array_2d}},
        ::Dual{Val{Matrix{P}}},
        ::Dual{Tuple{Val{Any},Val{Int},Val{Int}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        ::Dual{Type{Matrix{P}}},
        m::Dual{Int},
        n::Dual{Int},
        args::Vararg{Dual},
    ) where {P}
        T = tangent_type(P)
        return quote
            _m, _n = primal(m), primal(n)
            y = ccall(:jl_alloc_array_2d, Matrix{$P}, (Any, Int, Int), Matrix{$P}, _m, _n)
            dy = ccall(:jl_alloc_array_2d, Matrix{$T}, (Any, Int, Int), Matrix{$T}, _m, _n)
            return Dual(y, dy)
        end
    end
    @generated function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_alloc_array_3d}},
        ::Dual{Val{Array{P,3}}},
        ::Dual{Tuple{Val{Any},Val{Int},Val{Int},Val{Int}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        ::Dual{Type{Array{P,3}}},
        l::Dual{Int},
        m::Dual{Int},
        n::Dual{Int},
        args::Vararg{Dual},
    ) where {P}
        T = tangent_type(P)
        return quote
            _l, _m, _n = primal(l), primal(m), primal(n)
            y = ccall(
                :jl_alloc_array_3d,
                Array{$P,3},
                (Any, Int, Int, Int),
                Array{$P,3},
                _l,
                _m,
                _n,
            )
            dy = ccall(
                :jl_alloc_array_3d,
                Array{$T,3},
                (Any, Int, Int, Int),
                Array{$T,3},
                _l,
                _m,
                _n,
            )
            return Dual(y, dy)
        end
    end
end
