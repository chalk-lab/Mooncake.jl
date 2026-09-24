# Without this, zeroing the `@zero_derivative` return value below duals every cached rule.
tangent_type(::Type{<:MooncakeInterpreter}) = NoTangent

@zero_derivative MinimalCtx Tuple{typeof(get_interpreter),Type{<:Mode}}
@zero_derivative MinimalCtx Tuple{typeof(get_interpreter),Type{<:Mode},UInt}
@zero_derivative MinimalCtx Tuple{
    typeof(build_rrule_checks),MooncakeInterpreter,Any,Bool,Bool,Bool
}
@zero_derivative MinimalCtx Tuple{typeof(is_primitive),Type,Type{<:Mode},Any,UInt}

@is_primitive MinimalCtx Tuple{
    typeof(build_derived_rrule),MooncakeInterpreter{C},Any,Any,Bool
} where {C}

# `DerivedFoRRule` carries a pre-built `Lifted` extracted by `get_inner_rrule`;
# its Stacks self-reset after each forward+reverse pass. `Nothing` denotes a primitive.
# `LazyFoRRule` / `DynamicFoRRule` are frules for `build_derived_rrule`, not user rules:
# they clone Stacks per call because nested AD may re-enter before the previous pass resets.
# A concrete Trule permits a typed single-slot LazyFoRRule at one compiled call site.
# With @nospecialize, Trule can be Any and the site serves different inner signatures;
# DynamicFoRRule must distinguish those signatures to avoid returning the wrong rule.
mutable struct LazyFoRRule{Trule,Tfwd,Trvs}
    rule::Trule
    fwd_dual_callable::Tfwd
    rvs_dual_callable::Trvs
    LazyFoRRule{Trule,Tfwd,Trvs}() where {Trule,Tfwd,Trvs} = new()
end

# Key by signature, debug mode (different rule layouts), and compiled chunk width.
# Omitting sig_or_mi assumes each reachable MethodInstance has a unique signature;
# if two share a signature but select different IR, the key must include sig_or_mi.
# Neither this Dict nor LazyFoRRule's bare field assignments are thread-safe.
mutable struct DynamicFoRRule
    # (sig, debug_mode, chunk_width) => (rule, fwd_dc, rvs_dc)
    cache::Dict{Tuple{Any,Bool,Int},Tuple{Any,Any,Any}}
    DynamicFoRRule() = new(Dict{Tuple{Any,Bool,Int},Tuple{Any,Any,Any}}())
end

# build_frule copies captured constructor caches on a hit; each copy needs fresh
# mutable state. The generic copy fallback has no method for these types.
_copy(::DynamicFoRRule) = DynamicFoRRule()
_copy(::P) where {P<:LazyFoRRule} = P()

@generated function __build_primitive_frule(
    sig::Type{<:Tuple{typeof(build_derived_rrule),MooncakeInterpreter{C},SMI,S,Bool}}
) where {C,SMI,S}
    Trule = Core.Compiler.return_type(
        build_derived_rrule, Tuple{MooncakeInterpreter{C},SMI,S,Bool}
    )
    # @nospecialize can leave Trule=Any at a site shared by different signatures;
    # fieldtype would fail, and a single-slot cache would return the wrong rule.
    if !isconcretetype(Trule)
        return :(DynamicFoRRule())
    end
    # Unwrap debug rules; unexpected layouts cannot use the typed cache.
    inner = Trule <: DebugRRule ? fieldtype(Trule, :rule) : Trule
    if !hasfield(inner, :fwds_oc) || !hasfield(inner, :pb_oc_ref)
        return :(DynamicFoRRule())
    end
    fwds_oc_T = fieldtype(inner, :fwds_oc)
    rvs_oc_T = fieldtype(fieldtype(inner, :pb_oc_ref), :x)
    interp_fwd_T = MooncakeInterpreter{C,ForwardMode}
    Tfwd = Core.Compiler.return_type(build_frule, Tuple{interp_fwd_T,fwds_oc_T})
    Trvs = Core.Compiler.return_type(build_frule, Tuple{interp_fwd_T,rvs_oc_T})
    # Fall back to DynamicFoRRule if inference cannot pin down the dual callable types;
    # LazyFoRRule{Trule,Any,Any} would defeat its own purpose (typed single-slot cache).
    if !isconcretetype(Tfwd) || !isconcretetype(Trvs)
        return :(DynamicFoRRule())
    end
    return :(LazyFoRRule{$Trule,$Tfwd,$Trvs}())
end

function build_primitive_frule(
    sig::Type{<:Tuple{typeof(build_derived_rrule),MooncakeInterpreter{C},SMI,S,Bool}}
) where {C,SMI,S}
    return __build_primitive_frule(sig)
end

# Custom factories build dual-callables via _dual_mc and share comms captures via
# the MistyClosure cache; the generic structural walker cannot construct this V.
for (f, internal) in
    ((:zero_dual, :_zero_dual_internal), (:uninit_dual, :_uninit_dual_internal))
    @eval @inline $f(w::Val{N}, x::Union{DerivedRule,MistyClosure}) where {N} = $internal(
        w, x, IdDict{Any,Any}()
    )
end
@inline randn_dual(w::Val{N}, rng::AbstractRNG, x::Union{DerivedRule,MistyClosure}) where {N} = _randn_dual_internal(
    w, rng, x, IdDict{Any,Any}()
)

# Compiler state carries no user derivatives; seeding it must not lift its caches.
@foldable @inline dual_type(::Val{N}, ::Type{<:MooncakeInterpreter}) where {N} = NoDual
@inline zero_dual(::Val{N}, ::MooncakeInterpreter) where {N} = NoDual()
@inline uninit_dual(::Val{N}, ::MooncakeInterpreter) where {N} = NoDual()
# Match both tangent kinds specifically to avoid ambiguity with generic lift methods.
@inline lift(x::MooncakeInterpreter, ::NoTangent) = Lifted{typeof(x),1,NoDual}(x, NoDual())
@inline lift(x::MooncakeInterpreter, ::NoDual) = Lifted{typeof(x),1,NoDual}(x, NoDual())

# Copy captures once into both closures, preserving their shared comms Stacks.
# _for_rule_dual's shared IdDict preserves the same aliasing in tangent Stacks.
# _copy(Stack) resets primal Stacks; fresh tangent Stacks must match their size.
function _for_rule_cached_dual(rule, fwd_dc, rvs_dc, ::Val{N}, debug_mode::Bool) where {N}
    return _for_rule_dual(_copy(rule), _copy(fwd_dc), _copy(rvs_dc), Val(N), debug_mode)
end

# Captures have zero seeds; directions come from the outer x. Reuse the dual-callables
# compiled over forward-optimised IR: zero_dual(rule) would re-derive over reverse IR.
# The shared cache preserves fwds/pb Stack aliasing, and V must match dual_type exactly.
function _for_rule_dual(rule, fwd_dc, rvs_dc, ::Val{N}, debug_mode::Bool) where {N}
    inner = debug_mode ? rule.rule : rule
    fwd_caps = inner.fwds_oc.oc.captures
    rvs_caps = inner.pb_oc_ref[].oc.captures
    d = IdDict()
    fwd_capsV = Lifted{typeof(fwd_caps),N}(
        fwd_caps, _zero_dual_internal(Val(N), fwd_caps, d)
    )
    rvs_capsV = Lifted{typeof(rvs_caps),N}(
        rvs_caps, _zero_dual_internal(Val(N), rvs_caps, d)
    )
    innerV = ImmutableDual((;
        fwds_oc=MistyClosureTangent(fwd_capsV, fwd_dc),
        pb_oc_ref=MutableDual((;
            x=PossiblyUninitTangent(MistyClosureTangent(rvs_capsV, rvs_dc))
        )),
        nargs=NoDual(),
        consts=NoDual(),
    ))
    V = debug_mode ? ImmutableDual((; rule=innerV)) : innerV
    # Diagnose stale hand-written fields here, before an opaque closure type error.
    expected = dual_type(Val(N), typeof(rule))
    typeof(V) === expected ||
        _throw_for_rule_dual_mismatch(typeof(rule), typeof(V), expected)
    return Lifted{typeof(rule),N,typeof(V)}(rule, V)
end

@noinline function _throw_for_rule_dual_mismatch(
    @nospecialize(Trule), @nospecialize(got), @nospecialize(expected)
)
    return error(
        "`_for_rule_dual` assembled a forward value for $Trule that does not match " *
        "`dual_type`. Got\n  $got\nexpected\n  $expected\nThis usually means $Trule gained or " *
        "lost a field and the hand-written `ImmutableDual` above was not updated to match.",
    )
end

function _compile_for_rule(
    interp::MooncakeInterpreter{C}, sig_or_mi, sig, debug_mode::Bool; chunk_size::Int=1
) where {C}
    @nospecialize sig_or_mi sig

    # Keep primitive rules as static call boundaries until after the forward transform.
    dri = generate_ir(
        interp, sig_or_mi; debug_mode, do_optimize=false, noinline_primitive_rules=true
    )

    # noinline_primitive_rules protects primitive-rule calls. Protect increment!! too:
    # inlining can erase its arithmetic before forward AD selects the rules. Apply
    # this protection to other internals if their rule boundaries are lost likewise.
    # Global do_inline=false prevents inlining needed to eliminate temporary allocations.
    for inst in dri.rvs_ir.stmts
        ex = stmt(inst)
        if Meta.isexpr(ex, :call) && ex.args[1] === increment!!
            CC.setindex!(inst, CC.getindex(inst, :flag) | CC.IR_FLAG_NOINLINE, :flag)
        end
    end

    # Optimize and build the primal DerivedRule.
    raw_rule = let
        optimized_fwd_ir = optimise_ir!(CC.copy(dri.fwd_ir))
        optimized_rvs_ir = optimise_ir!(CC.copy(dri.rvs_ir))
        fwd_oc = misty_closure(dri.fwd_ret_type, optimized_fwd_ir, dri.shared_data...)
        rvs_oc = misty_closure(dri.rvs_ret_type, optimized_rvs_ir, dri.shared_data...)
        nargs = num_args(dri.info)
        sig_flat = flatten_va_sig(sig, dri.isva, nargs)
        DerivedRule(
            sig_flat,
            fwd_oc,
            Ref(rvs_oc),
            dri.isva,
            Val(nargs),
            _aliasable_constants(dri.shared_data, dri.info.global_bindings),
        )
    end

    # A forward interpreter preserves frule boundaries during optimisation.
    # The closures share comms captures; cached callables need fresh _copy state on reuse.
    fwd_dc, rvs_dc = let
        interp_forward = MooncakeInterpreter(C, ForwardMode; world=interp.world)
        optimized_fwd_ir = optimise_ir!(dri.fwd_ir; interp=interp_forward)
        optimized_rvs_ir = optimise_ir!(dri.rvs_ir; interp=interp_forward)
        fwd_oc = misty_closure(dri.fwd_ret_type, optimized_fwd_ir, dri.shared_data...)
        rvs_oc = misty_closure(dri.rvs_ret_type, optimized_rvs_ir, dri.shared_data...)
        fwd_dc = build_frule(
            interp_forward, fwd_oc; skip_world_age_check=true, debug_mode, chunk_size
        )
        rvs_dc = build_frule(
            interp_forward, rvs_oc; skip_world_age_check=true, debug_mode, chunk_size
        )
        fwd_dc, rvs_dc
    end

    rule = debug_mode ? DebugRRule(raw_rule) : raw_rule
    return rule, fwd_dc, rvs_dc
end

function (cache::LazyFoRRule{Trule,Tfwd,Trvs})(
    ::Lifted{typeof(build_derived_rrule),Nw},
    _interp::Lifted{<:MooncakeInterpreter{C}},
    _sig_or_mi::Lifted,
    _sig::Lifted,
    _debug_mode::Lifted{Bool},
) where {Trule,Tfwd,Trvs,C,Nw}
    @nospecialize _sig_or_mi _sig

    debug_mode = primal(_debug_mode)

    # Cache hit: reuse compiled artifacts with fresh empty Stacks. sig is not
    # re-checked because each LazyFoRRule lives at exactly one call site in the
    # compiled IR (inside a fixed-grad_f closure), so the inner signature — and the
    # outer chunk width `Nw` — are invariant for its lifetime. debug_mode is checked
    # below because the cached rule layout differs between DebugRRule and plain DerivedRule.
    if isdefined(cache, :rule)
        if debug_mode != (cache.rule isa DebugRRule)
            error(
                "LazyFoRRule cache hit with debug_mode=$debug_mode but cached rule is " *
                "$(typeof(cache.rule)); debug_mode must be consistent across calls.",
            )
        end
        return _for_rule_cached_dual(
            cache.rule,
            cache.fwd_dual_callable,
            cache.rvs_dual_callable,
            Val(Nw),
            debug_mode,
        )
    end

    rule, fwd_dc, rvs_dc = _compile_for_rule(
        primal(_interp), primal(_sig_or_mi), primal(_sig), debug_mode; chunk_size=Nw
    )
    cache.rule = rule
    cache.fwd_dual_callable = fwd_dc
    cache.rvs_dual_callable = rvs_dc
    return _for_rule_dual(rule, fwd_dc, rvs_dc, Val(Nw), debug_mode)
end

function (cache::DynamicFoRRule)(
    ::Lifted{typeof(build_derived_rrule),Nw},
    _interp::Lifted{<:MooncakeInterpreter{C}},
    _sig_or_mi::Lifted,
    _sig::Lifted,
    _debug_mode::Lifted{Bool},
) where {C,Nw}
    @nospecialize _sig_or_mi _sig

    debug_mode = primal(_debug_mode)

    dict_key = (primal(_sig), debug_mode, Nw)

    entry = get(cache.cache, dict_key, nothing)
    if entry !== nothing
        rule, fwd_dc, rvs_dc = entry
        return _for_rule_cached_dual(rule, fwd_dc, rvs_dc, Val(Nw), debug_mode)
    end

    rule, fwd_dc, rvs_dc = _compile_for_rule(
        primal(_interp), primal(_sig_or_mi), primal(_sig), debug_mode; chunk_size=Nw
    )
    cache.cache[dict_key] = (rule, fwd_dc, rvs_dc)
    return _for_rule_dual(rule, fwd_dc, rvs_dc, Val(Nw), debug_mode)
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

# The tangent lives inside the cached Lifted, so this carrier is non-differentiable.
# Preserve forward-compiled callables instead of re-deriving over reverse-optimised IR.
# The cached rule is pinned to preparation's world age; rebuild if methods change.
struct DerivedFoRRule{D}
    rule_dual::D
end
function compile_for_rule(f, x...; debug_mode::Bool=false, chunk_size::Int=1)
    sig = _typeof((f, x...))
    interp = get_interpreter(ReverseMode)
    if is_primitive(DefaultCtx, ReverseMode, sig, interp.world)
        return DerivedFoRRule{Nothing}(nothing)
    end
    rule, fwd_dc, rvs_dc = _compile_for_rule(interp, sig, sig, debug_mode; chunk_size)
    return DerivedFoRRule(_for_rule_dual(rule, fwd_dc, rvs_dc, Val(chunk_size), debug_mode))
end
tangent_type(::Type{<:DerivedFoRRule}) = NoTangent
dual_type(::Val{N}, ::Type{<:DerivedFoRRule}) where {N} = NoDual

get_inner_rrule(r::DerivedFoRRule{<:Lifted}) = primal(r.rule_dual)
@is_primitive MinimalCtx Tuple{typeof(get_inner_rrule),<:DerivedFoRRule{<:Lifted}}
function frule!!(
    ::Lifted{typeof(get_inner_rrule),Nw}, r::Lifted{<:DerivedFoRRule{<:Lifted},Nw}
) where {Nw}
    return primal(r).rule_dual
end
function rrule!!(::CoDual{typeof(get_inner_rrule)}, ::CoDual{<:DerivedFoRRule{<:Lifted}})
    throw(
        ArgumentError(
            "DerivedFoRRule is forward-over-reverse only; reverse-mode " *
            "differentiation through it is not supported.",
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
# TODO: can be removed once we improve the performance of differentiating through building
# rules, such that the DI test will pass with no inner prep without this workaround.
@static if VERSION >= v"1.11-"
    function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{:jl_genericmemory_owner},Nw},
        ::Lifted{Val{Any},Nw},
        ::Lifted{Tuple{Val{Any}},Nw},
        ::Lifted{Val{0},Nw},
        ::Lifted{Val{:ccall},Nw},
        a::Lifted{<:Memory},
    ) where {Nw}
        y = ccall(:jl_genericmemory_owner, Any, (Any,), primal(a))
        # Memory owners can be differentiable, so the zero must have canonical V.
        return zero_lifted(Val(Nw), y)
    end
    function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{:jl_genericmemory_owner}},
        ::CoDual{Val{Any}},
        ::CoDual{Tuple{Val{Any}}},
        ::CoDual{Val{0}},
        ::CoDual{Val{:ccall}},
        a::CoDual{<:Memory},
    )
        y = zero_fcodual(ccall(:jl_genericmemory_owner, Any, (Any,), primal(a)))
        return y, NoPullback(ntuple(_ -> NoRData(), 7))
    end
end

# This rule is potentially unnecessary if fixes are made elsewhere,
# but currently fixes differentiating through zero_tangent_internal for Arrays.
@zero_derivative MinimalCtx Tuple{typeof(zero_tangent),Any}

# The cached seed constructors call `zero_tangent_internal` directly, so the `zero_tangent` rule
# above does not cover them: forward-over-reverse would otherwise differentiate through their
# `IdDict` construction and hit `_new_(Vector{Float64}, ...)`, which forward mode refuses.
@zero_derivative MinimalCtx Tuple{typeof(_zero_tangents),Any}
@zero_derivative MinimalCtx Tuple{typeof(_zero_codual_cached),Any,Any}

@static if VERSION < v"1.11-"
    function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{:jl_alloc_array_1d},Nw},
        ::Lifted{Val{Vector{P}},Nw},
        ::Lifted{Tuple{Val{Any},Val{Int}},Nw},
        ::Lifted{Val{0},Nw},
        ::Lifted{Val{:ccall},Nw},
        ::Lifted{Type{Vector{P}},Nw},
        n::Lifted{Int},
        args::Vararg{Lifted},
    ) where {Nw,P}
        _n = primal(n)
        y = ccall(:jl_alloc_array_1d, Vector{P}, (Any, Int), Vector{P}, _n)
        return Lifted{Vector{P},Nw}(y, uninit_dual(Val(Nw), y))
    end
    function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{:jl_alloc_array_2d},Nw},
        ::Lifted{Val{Matrix{P}},Nw},
        ::Lifted{Tuple{Val{Any},Val{Int},Val{Int}},Nw},
        ::Lifted{Val{0},Nw},
        ::Lifted{Val{:ccall},Nw},
        ::Lifted{Type{Matrix{P}},Nw},
        m::Lifted{Int},
        n::Lifted{Int},
        args::Vararg{Lifted},
    ) where {Nw,P}
        _m, _n = primal(m), primal(n)
        y = ccall(:jl_alloc_array_2d, Matrix{P}, (Any, Int, Int), Matrix{P}, _m, _n)
        return Lifted{Matrix{P},Nw}(y, uninit_dual(Val(Nw), y))
    end
    function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{:jl_alloc_array_3d},Nw},
        ::Lifted{Val{Array{P,3}},Nw},
        ::Lifted{Tuple{Val{Any},Val{Int},Val{Int},Val{Int}},Nw},
        ::Lifted{Val{0},Nw},
        ::Lifted{Val{:ccall},Nw},
        ::Lifted{Type{Array{P,3}},Nw},
        l::Lifted{Int},
        m::Lifted{Int},
        n::Lifted{Int},
        args::Vararg{Lifted},
    ) where {Nw,P}
        _l, _m, _n = primal(l), primal(m), primal(n)
        y = ccall(
            :jl_alloc_array_3d, Array{P,3}, (Any, Int, Int, Int), Array{P,3}, _l, _m, _n
        )
        return Lifted{Array{P,3},Nw}(y, uninit_dual(Val(Nw), y))
    end
end
