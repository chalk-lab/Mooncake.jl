#
# Reverse-mode source-to-source transform roadmap
#
# This file is in roughly the following order:
# 1. Local reverse-mode IR types and ID utilities.
# 2. Shared closure-capture management and global AD state.
# 3. Statement translation from primal IR to forward/reverse fragments.
# 4. Callable wrapper types used by derived rules at runtime.
# 5. Deferred rule wrappers for dynamic dispatch and recursive :invoke handling.
# 6. Rule derivation entry points and IR generation.
# 7. Forward-pass assembly, CFGBlock-based lowering, and pullback assembly.
#
# The implementation starts with low-level types because later sections share them heavily,
# but the main transform entry points are `build_rrule`, `build_derived_rrule`, and
# `generate_ir`.
#

#
# Reverse-mode local IR: IDs and CFG-local node types
#

const _id_count::Dict{Int,Int32} = Dict{Int,Int32}()
# `seed_id!` resets per-thread counters for deterministic IR generation, so updates to the
# shared thread-id map must be serialized when rules are derived concurrently.
const _id_count_lock = ReentrantLock()

struct ID
    id::Int32
    function ID()
        lock(_id_count_lock)
        try
            current_thread_id = Threads.threadid()
            id_count = get(_id_count, current_thread_id, Int32(0))
            _id_count[current_thread_id] = id_count + Int32(1)
            return new(id_count)
        finally
            unlock(_id_count_lock)
        end
    end
end

Base.copy(id::ID) = id

function seed_id!()
    lock(_id_count_lock)
    try
        return global _id_count[Threads.threadid()] = 0
    finally
        unlock(_id_count_lock)
    end
end

struct IDPhiNode
    edges::Vector{ID}
    values::Vector{Any}
end

Base.:(==)(x::IDPhiNode, y::IDPhiNode) = x.edges == y.edges && x.values == y.values
Base.copy(node::IDPhiNode) = IDPhiNode(copy(node.edges), copy(node.values))

struct IDGotoNode
    label::ID
end

Base.copy(node::IDGotoNode) = IDGotoNode(copy(node.label))

struct IDGotoIfNot
    cond::Any
    dest::ID
end

Base.copy(node::IDGotoIfNot) = IDGotoIfNot(copy(node.cond), copy(node.dest))

struct Switch
    conds::Vector{Any}
    dests::Vector{ID}
    fallthrough_dest::ID
    function Switch(conds::Vector{Any}, dests::Vector{ID}, fallthrough_dest::ID)
        @assert length(conds) == length(dests)
        return new(conds, dests, fallthrough_dest)
    end
end

const IDInstPair = Tuple{ID,NewInstruction}
const InstVector = Vector{NewInstruction}
const SSAToIdDict = Dict{SSAValue,ID}
const BlockNumToIdDict = Dict{Integer,ID}

function characterise_used_ids(stmts::Vector{IDInstPair})::Dict{ID,Bool}
    is_used = Dict{ID,Bool}()
    for (id, _) in stmts
        @assert !haskey(is_used, id)
        is_used[id] = false
    end
    for (_, inst) in stmts
        _find_id_uses!(is_used, inst.stmt)
    end
    return is_used
end

function _find_id_uses!(d::Dict{ID,Bool}, x::Expr)
    foreach(a -> _find_id_uses!(d, a), x.args)
    return nothing
end
function _find_id_uses!(d::Dict{ID,Bool}, x::IDGotoIfNot)
    return _find_id_uses!(d, x.cond)
end
_find_id_uses!(::Dict{ID,Bool}, ::IDGotoNode) = nothing
function _find_id_uses!(d::Dict{ID,Bool}, x::PiNode)
    return _find_id_uses!(d, x.val)
end
function _find_id_uses!(d::Dict{ID,Bool}, x::IDPhiNode)
    for n in eachindex(x.values)
        # Normalized compiler phi nodes can leave incoming values undefined on dead edges.
        isassigned(x.values, n) || continue
        _find_id_uses!(d, x.values[n])
    end
    return nothing
end
function _find_id_uses!(d::Dict{ID,Bool}, x::ReturnNode)
    return isdefined(x, :val) ? _find_id_uses!(d, x.val) : nothing
end
_find_id_uses!(::Dict{ID,Bool}, ::QuoteNode) = nothing
_find_id_uses!(d::Dict{ID,Bool}, x::ID) = d[x] = true
_find_id_uses!(::Dict{ID,Bool}, x) = nothing

#
# Shared closure captures and reverse-mode global state
#

"""
    SharedDataPairs()

A data structure used to manage the captured data in the `OpaqueClosures` which implement
the bulk of the forwards- and reverse-passes of AD. An entry `(id, data)` at element `n`
of the `pairs` field of this data structure means that `data` will be available at register
`id` during the forwards- and reverse-passes of `AD`.

This is achieved by storing all of the data in the `pairs` field in the captured tuple which
is passed to an `OpaqueClosure`, and extracting this data into registers associated to the
corresponding `ID`s.
"""
struct SharedDataPairs
    pairs::Vector{Tuple{ID,Any}}
    SharedDataPairs() = new(Tuple{ID,Any}[])
end

"""
    add_data!(p::SharedDataPairs, data)::ID

Puts `data` into `p`, and returns the `id` associated to it. This `id` should be assumed to
be available during the forwards- and reverse-passes of AD, and it should further be assumed
that the value associated to this `id` is always `data`.
"""
function add_data!(p::SharedDataPairs, data::Any)::ID
    id = ID()
    push!(p.pairs, (id, data))
    return id
end

"""
    shared_data_tuple(p::SharedDataPairs)::Tuple

Create the tuple that will constitute the captured variables in the forwards- and reverse-
pass `OpaqueClosure`s.

For example, if `p.pairs` is
```julia
[(ID(5), 5.0), (ID(3), "hello")]
```
then the output of this function is
```julia
(5.0, "hello")
```
"""
shared_data_tuple(p::SharedDataPairs)::Tuple = tuple(map(last, p.pairs)...)

"""
    shared_data_stmts(p::SharedDataPairs)::Vector{IDInstPair}

Produce a sequence of id-statement pairs which will extract the data from
`shared_data_tuple(p)` such that the correct value is associated to the correct `ID`.

For example, if `p.pairs` is
```julia
[(ID(5), 5.0), (ID(3), "hello")]
```
then the output of this function is
```julia
IDInstPair[
    (ID(5), new_inst(:(getfield(_1, 1)))),
    (ID(3), new_inst(:(getfield(_1, 2)))),
]
```
"""
function shared_data_stmts(p::SharedDataPairs)::Vector{IDInstPair}
    return map(enumerate(p.pairs)) do (n, p)
        return (p[1], new_inst(Expr(:call, get_shared_data_field, Argument(1), n)))
    end
end
# maybe manually inline this
@inline get_shared_data_field(shared_data, n) = getfield(shared_data, n)

"""
The block stack is the stack used to keep track of which basic blocks are visited on the
forwards pass, and therefore which blocks need to be visited on the reverse pass. There is
one block stack per derived rule.
By using Int32, we assume that there aren't more than `typemax(Int32)` unique basic blocks
in a given function, which ought to be reasonable.
"""
const BlockStack = Stack{Int32}

"""
    ADInfo

This data structure is used to hold "global" information associated to a particular call to
`build_rrule`. It is used as a means of communication between `make_ad_stmts!` and the
codegen which produces the forwards- and reverse-passes.

At a high level, the most important fields are the shared captures, the block-stack state used
to replay control flow, the reverse-data refs for arguments and SSA values, and the static
primal type information used while translating statements.

- `interp`: a `MooncakeInterpreter`.
- `block_stack_id`: the ID associated to the block stack -- the stack which keeps track of
    which blocks we visited during the forwards-pass, and which is used on the reverse-pass
    to determine which blocks to visit.
- `block_stack`: the block stack. Can always be found at `block_stack_id` in the forwards-
    and reverse-passes.
- `entry_id`: ID associated to the block inserted at the start of execution in the
    forwards-pass, and the end of execution in the pullback.
- `shared_data_pairs`: the `SharedDataPairs` used to define the captured variables passed
    to both the forwards- and reverse-passes.
- `arg_types`: a map from `Argument` to its static type.
- `ssa_insts`: a map from `ID` associated to lines to the primal `NewInstruction`. This
    contains the line of code, its static / inferred type, and some other details. See
    `Core.Compiler.NewInstruction` for a full list of fields.
- `arg_rdata_ref_ids`: the dict mapping from arguments to the `ID` which creates and
    initialises the `Ref` which contains the reverse data associated to that argument.
    Recall that the heap allocations associated to this `Ref` are always optimised away in
    the final programme.
- `ssa_rdata_ref_ids`: the same as `arg_rdata_ref_ids`, but for each `ID` associated to an
    ssa rather than each argument.
- `debug_mode`: if `true`, run in "debug mode" -- wraps all rule calls in `DebugRRule`. This
    is applied recursively, so that debug mode is also switched on in derived rules.
- `is_used_dict`: for each `ID` associated to a line of code, is `false` if line is not used
    anywhere in any other line of code.
- `lazy_zero_rdata_ref_id`: for any arguments whose type doesn't permit the construction of
    a zero-valued rdata directly from the type alone (e.g. a struct with an abstractly-
    typed field), we need to have a zero-valued rdata available on the reverse-pass so that
    this zero-valued rdata can be returned if the argument (or a part of it) is never used
    during the forwards-pass and consequently doesn't obtain a value on the reverse-pass.
    To achieve this, we construct a `LazyZeroRData` for each of the arguments on the
    forwards-pass, and make use of it on the reverse-pass. This field is the ID that will be
    associated to this information.
"""
struct ADInfo
    interp::MooncakeInterpreter
    block_stack_id::ID
    block_stack::BlockStack
    entry_id::ID
    shared_data_pairs::SharedDataPairs
    arg_types::Dict{Argument,Any}
    ssa_insts::Dict{ID,NewInstruction}
    arg_rdata_ref_ids::Dict{Argument,ID}
    ssa_rdata_ref_ids::Dict{ID,ID}
    debug_mode::Bool
    is_used_dict::Dict{ID,Bool}
    lazy_zero_rdata_ref_id::ID
    fwd_ret_type::Type
    rvs_ret_type::Type
end

# See the definition of the ADInfo struct for info on the arguments.
function ADInfo(
    interp::MooncakeInterpreter,
    arg_types::Dict{Argument,Any},
    ssa_insts::Dict{ID,NewInstruction},
    is_used_dict::Dict{ID,Bool},
    debug_mode::Bool,
    zero_lazy_rdata_ref::Ref{<:Tuple},
    fwd_ret_type::Type,
    rvs_ret_type::Type,
)
    shared_data_pairs = SharedDataPairs()
    block_stack = BlockStack()
    return ADInfo(
        interp,
        add_data!(shared_data_pairs, block_stack),
        block_stack,
        ID(),
        shared_data_pairs,
        arg_types,
        ssa_insts,
        Dict((k, ID()) for k in keys(arg_types)),
        Dict((k, ID()) for k in keys(ssa_insts)),
        debug_mode,
        is_used_dict,
        add_data!(shared_data_pairs, zero_lazy_rdata_ref),
        fwd_ret_type,
        rvs_ret_type,
    )
end

"""
    add_data!(info::ADInfo, data)::ID

Equivalent to `add_data!(info.shared_data_pairs, data)`.
"""
add_data!(info::ADInfo, @nospecialize(data))::ID = add_data!(info.shared_data_pairs, data)

"""
    add_data_if_not_singleton!(p::Union{ADInfo, SharedDataPairs}, x)

Returns `x` if it is a singleton, or the `ID` of the ssa which will contain it on the
forwards- and reverse-passes. The reason for this is that if something is a singleton, it
can be inserted directly into the IR.
"""
function add_data_if_not_singleton!(p::Union{ADInfo,SharedDataPairs}, @nospecialize(x))
    return Base.issingletontype(_typeof(x)) ? x : add_data!(p, x)
end

"""
    is_used(info::ADInfo, id::ID)::Bool

Returns `true` if `id` is used by any of the lines in the ir, false otherwise.
"""
is_used(info::ADInfo, id::ID)::Bool = info.is_used_dict[id]

"""
    get_primal_type(info::ADInfo, x)

Returns the static / inferred type associated to `x`.
"""
get_primal_type(info::ADInfo, x::Argument) = info.arg_types[x]
get_primal_type(info::ADInfo, x::ID) = CC.widenconst(info.ssa_insts[x].type)
get_primal_type(::ADInfo, x::QuoteNode) = _typeof(x.value)
get_primal_type(::ADInfo, @nospecialize(x)) = _typeof(x)
@static if VERSION > v"1.12-"
    function get_primal_type(info::ADInfo, x::GlobalRef)
        return get_primal_type(info.interp.world, x.binding)
    end
    # The comments for the `jl_partition_kind` enum are a good reference
    function get_primal_type(world::UInt, x::Core.Binding)
        partition = Base.lookup_binding_partition(world, x)
        # no restriction available
        isdefined(partition, :restriction) || return Any
        kind = Base.binding_kind(partition)
        # for a constant, the restriction is the value, return its type
        if Base.is_defined_const_binding(kind)
            return _typeof(partition.restriction)
        end
        # otherwise for an imported global the restriction is the imported binding
        if Base.is_some_imported(kind)
            return get_primal_type(world, partition.restriction::Core.Binding)
        end
        # otherwise we have a mutable global, the restriction is the type
        return partition.restriction::Type
    end
else
    function get_primal_type(::ADInfo, x::GlobalRef)
        isconst(x) && return _typeof(getglobal(x.mod, x.name))
        return isdefined(x.binding, :ty) ? x.binding.ty : x.binding.owner.ty
    end
end # @static
function get_primal_type(::ADInfo, x::Expr)
    x.head === :boundscheck && return Bool
    return error("Unrecognised expression $x found in argument slot.")
end

"""
    get_rev_data_id(info::ADInfo, x)

Returns the `ID` associated to the line in the reverse pass which will contain the
reverse data for `x`. If `x` is not an `Argument` or `ID`, then `nothing` is returned.
"""
get_rev_data_id(info::ADInfo, x::Argument) = info.arg_rdata_ref_ids[x]
get_rev_data_id(info::ADInfo, x::ID) = info.ssa_rdata_ref_ids[x]
get_rev_data_id(::ADInfo, ::Any) = nothing

"""
    reverse_data_ref_stmts(info::ADInfo)

Create the `:new` statements which initialise the reverse-data `Ref`s. Interpolates the
initial rdata directly into the statement, which is safe because it is always a bits type.
"""
function reverse_data_ref_stmts(info::ADInfo)
    function make_ref_stmt(id::ID, P::Type)
        ref_type = Base.RefValue{P<:Type ? NoRData : zero_like_rdata_type(P)}
        init_ref_val = P <: Type ? NoRData() : Mooncake.zero_like_rdata_from_type(P)
        return (id, new_inst(Expr(:new, ref_type, QuoteNode(init_ref_val))))
    end
    return vcat(
        map(collect(info.arg_rdata_ref_ids)) do (k, id)
            return make_ref_stmt(id, CC.widenconst(info.arg_types[k]))
        end,
        map(collect(info.ssa_rdata_ref_ids)) do (k, id)
            return make_ref_stmt(id, CC.widenconst(info.ssa_insts[k].type))
        end,
    )
end

# Returns the number of arguments that the primal function has.
num_args(info::ADInfo) = length(info.arg_types)

"""
    RRuleZeroWrapper(rule)

This struct is used to ensure that `ZeroRData`s, which are used as placeholder zero
elements whenever an actual instance of a zero rdata for a particular primal type cannot
be constructed without also having an instance of said type, never reach rules.
On the pullback, we increment the cotangent dy by an amount equal to zero. This ensures
that if it is a `ZeroRData`, we instead get an actual zero of the correct type. If it is
not a zero rdata, the computation _should_ be elided via inlining + constant prop.
"""
struct RRuleZeroWrapper{Trule}
    rule::Trule
end

# Recursively copy the wrapped rule
_copy(x::P) where {P<:RRuleZeroWrapper} = P(_copy(x.rule))

struct RRuleWrapperPb{Tpb!!,Tl}
    pb!!::Tpb!!
    l::Tl
end

(rule::RRuleWrapperPb)(dy) = rule.pb!!(increment!!(dy, instantiate(rule.l)))

@inline function (rule::RRuleZeroWrapper{R})(f::F, args::Vararg{CoDual,N}) where {R,F,N}
    y, pb!! = __call_rule(rule.rule, (f, args...))
    l = lazy_zero_rdata(primal(y))
    return y::CoDual, (pb!! isa NoPullback ? pb!! : RRuleWrapperPb(pb!!, l))
end

#
# Statement translation bookkeeping
#

"""
    ADStmtInfo

Data structure which contains the result of `make_ad_stmts!`. Fields are
- `line`: the ID associated to the primal line from which this is derived
- `comms_id`: an `ID` from one of the lines in `fwds`, whose value will be made
    available on the reverse-pass in the same `ID`. Nothing is asserted about _how_ this
    value is made available on the reverse-pass of AD, so this package is free to do this in
    whichever way is most efficient, in particular to group these communication `ID` on a
    per-block basis.
- `fwds`: the instructions which run the forwards-pass of AD
- `rvs`: the instructions which run the reverse-pass of AD / the pullback
"""
struct ADStmtInfo
    line::ID
    comms_id::Union{ID,Nothing}
    fwds::Vector{IDInstPair}
    rvs::Vector{IDInstPair}
end

struct RuleSelection
    args::Tuple
    rule_ref
    T_pb!!::Type
    output_type::Type
end

struct BlockCommsInsts
    fwds_suffix::Vector{IDInstPair}
    rvs_prefix::Vector{IDInstPair}
end

"""
    ad_stmt_info(line::ID, comms_id::Union{ID, Nothing}, fwds, rvs)

Convenient constructor for `ADStmtInfo`. If either `fwds` or `rvs` is not a vector,
`__vec` promotes it to a single-element `Vector`.
"""
function ad_stmt_info(line::ID, comms_id::Union{ID,Nothing}, fwds, rvs)
    if !(comms_id === nothing || in(comms_id, map(first, __vec(line, fwds))))
        throw(ArgumentError("comms_id not found in IDs of `fwds` instructions."))
    end
    return ADStmtInfo(line, comms_id, __vec(line, fwds), __vec(line, rvs))
end

function _select_rule(stmt::Expr, line::ID, info::ADInfo, is_invoke::Bool)
    args = ((is_invoke ? stmt.args[2:end] : stmt.args)...,)
    arg_types = map(arg -> get_primal_type(info, arg), args)

    sig = Tuple{arg_types...}
    interp = info.interp
    raw_rule = if is_primitive(context_type(interp), ReverseMode, sig, interp.world)
        build_primitive_rrule(sig)
    elseif is_invoke
        LazyDerivedRule(get_mi(stmt.args[1]), info.debug_mode)
    else
        DynamicDerivedRule(info.debug_mode)
    end

    output_type = get_primal_type(info, line)
    is_no_pullback = pullback_type(_typeof(raw_rule), arg_types) <: NoPullback
    strip_zero_rdata = can_produce_zero_rdata_from_type(output_type) || is_no_pullback
    wrapped_rule = strip_zero_rdata ? raw_rule : RRuleZeroWrapper(raw_rule)
    rule = info.debug_mode ? DebugRRule(wrapped_rule) : wrapped_rule

    return RuleSelection(
        args,
        add_data_if_not_singleton!(info, rule),
        pullback_type(_typeof(rule), arg_types),
        output_type,
    )
end

function _pullback_increment_stmts(
    info::ADInfo, args, call_pullback_id::ID
)::Vector{IDInstPair}
    increments = IDInstPair[]
    for (n, arg) in enumerate(args)
        rev_data_id = get_rev_data_id(info, arg)
        rev_data_id === nothing && continue

        rdata_inc_id = ID()
        push!(
            increments, (rdata_inc_id, new_inst(Expr(:call, getfield, call_pullback_id, n)))
        )
        append!(increments, increment_ref_stmts(rev_data_id, rdata_inc_id))
    end
    return increments
end

__vec(line::ID, x::Any) = __vec(line, new_inst(x))
__vec(line::ID, x::NewInstruction) = IDInstPair[(line, x)]
function __vec(::ID, x::Vector{Tuple{ID,Any}})
    throw(
        ArgumentError(
            "Expected `Vector{IDInstPair}` but found a plain `Vector{Tuple{ID,Any}}`."
        ),
    )
end
__vec(line::ID, x::Vector{IDInstPair}) = x

"""
    comms_channel(info::ADStmtInfo)

Return the element of `fwds` whose `ID` is the communication `ID`. Returns `Nothing` if
`comms_id` is `nothing`.
"""
function comms_channel(info::ADStmtInfo)
    info.comms_id === nothing && return nothing
    return only(filter(x -> x[1] == info.comms_id, info.fwds))
end

"""
    inc_args(stmt)

Increment by `1` the `n` field of any `Argument`s present in `stmt`.
Used in `make_ad_stmts!`.
"""
inc_args(x::Expr) = Expr(x.head, map(__inc, x.args)...)
inc_args(x::ReturnNode) = isdefined(x, :val) ? ReturnNode(__inc(x.val)) : x
inc_args(x::Union{GotoIfNot,IDGotoIfNot}) = typeof(x)(__inc(x.cond), x.dest)
inc_args(x::Union{GotoNode,IDGotoNode}) = x
function inc_args(x::T) where {T<:Union{IDPhiNode,PhiNode}}
    new_values = Vector{Any}(undef, length(x.values))
    for n in eachindex(x.values)
        if isassigned(x.values, n)
            new_values[n] = __inc(x.values[n])
        end
    end
    return T(x.edges, new_values)
end
inc_args(::Nothing) = nothing
inc_args(x::GlobalRef) = x
inc_args(x::PiNode) = PiNode(__inc(x.val), x.typ)
function inc_args(x::PhiCNode)
    new_values = Vector{Any}(undef, length(x.values))
    for n in eachindex(x.values)
        if isassigned(x.values, n)
            new_values[n] = __inc(x.values[n])
        end
    end
    return PhiCNode(new_values)
end
inc_args(x::UpsilonNode) = UpsilonNode(__inc(x.val))

__inc(x::Argument) = Argument(x.n + 1)
__inc(x) = x

"""
    make_ad_stmts!(inst::NewInstruction, line::ID, info::ADInfo)::ADStmtInfo

Every line in the primal code is associated to one or more lines in the forwards-pass of AD,
and one or more lines in the pullback. This function has method specific to every
node type in the Julia SSAIR.

Translates the instruction `inst`, associated to `line` in the primal, into a specification
of what should happen for this instruction in the forwards- and reverse-passes of AD, and
what data should be shared between the forwards- and reverse-passes. Returns this in the
form of an `ADStmtInfo`.

`info` is a data structure containing various bits of global information that certain types
of nodes need access to.
"""
function make_ad_stmts! end

#=
    make_ad_stmts!(::Nothing, line::ID, ::ADInfo)

`nothing` as a statement in Julia IR indicates the presence of a line which will later be
removed. We emit a no-op on both the forwards- and reverse-passes. No shared data.
=#
function make_ad_stmts!(::Nothing, line::ID, ::ADInfo)
    return ad_stmt_info(line, nothing, nothing, nothing)
end

#=
    make_ad_stmts!(stmt::ReturnNode, line::ID, info::ADInfo)

`ReturnNode`s have a single field, `val`, for which there are three cases to consider:

1. `val` is undefined: this `ReturnNode` is unreachable. Consequently, we'll never hit the
  associated statements on the forwards-pass or pullback. We just return the original
  statement on the forwards-pass, and `nothing` on the reverse-pass.
2. `val isa Union{Argument, ID}`: this is an active piece of data. Consequently, we know
  that it will be a `CoDual`, and can just return it. Therefore `stmt` is returned as the
  forwards-pass (with any `Argument`s incremented). On the reverse-pass the associated rdata
  ref should be incremented with the rdata passed to the pullback, residing in argument 2.
3. `val` is defined, but not a `Union{Argument, ID}`: in this case we are returning a
  constant -- build a constant CoDual and return that. There is nothing to do on the
  reverse pass.

For cases 2 and 3, we also insert a call to `typeassert` to ensure that `info.fwd_ret_type`
is respected. A similar check for `info.rvs_ret_type` is handled elsewhere.
=#
function make_ad_stmts!(stmt::ReturnNode, line::ID, info::ADInfo)
    if !is_reachable_return_node(stmt)
        return ad_stmt_info(line, nothing, inc_args(stmt), nothing)
    end
    if is_active(stmt.val)
        rdata_id = get_rev_data_id(info, stmt.val)
        rvs = increment_ref_stmts(rdata_id, Argument(2))
        assert_id = ID()
        val = __inc(stmt.val)
        fwds = [
            (assert_id, new_inst(Expr(:call, typeassert, val, info.fwd_ret_type))),
            (ID(), new_inst(ReturnNode(assert_id))),
        ]
        return ad_stmt_info(line, nothing, fwds, rvs)
    else
        const_id = ID()
        assert_id = ID()
        fwds = [
            (const_id, new_inst(const_codual_stmt(stmt.val, info))),
            (assert_id, new_inst(Expr(:call, typeassert, const_id, info.fwd_ret_type))),
            (ID(), new_inst(ReturnNode(assert_id))),
        ]
        return ad_stmt_info(line, nothing, fwds, nothing)
    end
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDGotoNode, line::ID, ::ADInfo)
    return ad_stmt_info(line, nothing, inc_args(stmt), nothing)
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDGotoIfNot, line::ID, ::ADInfo)
    stmt = inc_args(stmt)

    # If cond is not going to be wrapped in a `CoDual`, so just return the stmt.
    is_active(stmt.cond) || return ad_stmt_info(line, nothing, stmt, nothing)

    # stmt.cond is active, so primal must be extracted from `CoDual`.
    cond_id = ID()
    fwds = [
        (cond_id, new_inst(Expr(:call, primal, stmt.cond))),
        (line, new_inst(IDGotoIfNot(cond_id, stmt.dest), Any)),
    ]
    return ad_stmt_info(line, nothing, fwds, nothing)
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDPhiNode, line::ID, info::ADInfo)
    vals = stmt.values
    new_vals = Vector{Any}(undef, length(vals))
    for n in eachindex(vals)
        isassigned(vals, n) || continue
        new_vals[n] = inc_or_const(vals[n], info)
    end

    # It turns out to be really very important to do type inference correctly for PhiNodes.
    # For some reason, type inference really doesn't like it when you encounter mutually-
    # dependent PhiNodes whose types are unknown and for which you set the flag to
    # CC.IR_FLAG_REFINED. To avoid this we directly tell the compiler what the type is.
    new_type = fcodual_type(get_primal_type(info, line))
    _inst = new_inst(IDPhiNode(stmt.edges, new_vals), new_type, info.ssa_insts[line].flag)
    return ad_stmt_info(line, nothing, _inst, nothing)
end

function make_ad_stmts!(stmt::PiNode, line::ID, info::ADInfo)

    # PiNodes of the form `π (nothing, Union{})` have started appearing in 1.11. These nodes
    # appear in unreachable sections of code, and appear to serve no purpose. Consequently,
    # we mark them for removal (replace them with `nothing`). We do not currently have a
    # unit test for this, but integration testing seems to catch it in multiple places.
    stmt == PiNode(nothing, Union{}) && return ad_stmt_info(line, nothing, stmt, nothing)

    if is_active(stmt.val)
        # Get the primal type of this line, and the rdata refs for the `val` of this
        # `PiNode` and this line itself.
        P = get_primal_type(info, line)
        val_rdata_ref_id = get_rev_data_id(info, stmt.val)
        output_rdata_ref_id = get_rev_data_id(info, line)
        fwds = inc_args(PiNode(stmt.val, fcodual_type(CC.widenconst(stmt.typ))))

        # Get the rdata from the output_rdata_ref, and set its new value to zero, and
        # increment the output ref.
        output_rdata_id = ID()
        deref_stmts = deref_and_zero_stmts(P, output_rdata_ref_id, output_rdata_id)
        inc_exprs = increment_ref_stmts(val_rdata_ref_id, output_rdata_id)
        rvs = vcat(deref_stmts, inc_exprs)
    else
        # If the value of the PiNode is a constant / QuoteNode etc, then there is nothing to
        # do on the reverse-pass.
        const_id = ID()
        fwds = [
            (const_id, new_inst(const_codual_stmt(stmt.val, info))),
            (line, new_inst(PiNode(const_id, fcodual_type(CC.widenconst(stmt.typ))))),
        ]
        rvs = nothing
    end

    return ad_stmt_info(line, nothing, fwds, rvs)
end

# Constant GlobalRefs are handled. See const_codual. Non-constant GlobalRefs are handled by
# assuming that they are constant, and creating a CoDual with the value. We then check at
# run-time that the value has not changed.
function make_ad_stmts!(stmt::GlobalRef, line::ID, info::ADInfo)
    isconst(stmt) && return const_ad_stmt(stmt, line, info)

    const_id, globalref_id = ID(), ID()
    fwds = [
        (globalref_id, new_inst(stmt)),
        (const_id, new_inst(const_codual_stmt(getglobal(stmt.mod, stmt.name), info))),
        (line, new_inst(Expr(:call, __verify_const, globalref_id, const_id))),
    ]
    return ad_stmt_info(line, nothing, fwds, nothing)
end

# Helper used by `make_ad_stmts! ` for `GlobalRef`. Noinline to avoid IR bloat.
@noinline function __verify_const(global_ref, stored_value)
    @assert global_ref == primal(stored_value)
    return uninit_fcodual(global_ref)
end

# QuoteNodes are constant.
make_ad_stmts!(stmt::QuoteNode, line::ID, info::ADInfo) = const_ad_stmt(stmt, line, info)

# Literal constant.
make_ad_stmts!(stmt, line::ID, info::ADInfo) = const_ad_stmt(stmt, line, info)

"""
    const_ad_stmt(stmt, line::ID, info::ADInfo)

Implementation of `make_ad_stmts!` used for constants.
"""
function const_ad_stmt(stmt, line::ID, info::ADInfo)
    return ad_stmt_info(line, nothing, const_codual_stmt(stmt, info), nothing)
end

"""
    const_codual_stmt(stmt, info::ADInfo)

Returns a `:call` expression which will return a `CoDual` whose primal is `stmt`, and whose
tangent is whatever `uninit_tangent` returns.
"""
function const_codual_stmt(stmt, info::ADInfo)
    v = get_const_primal_value(stmt)
    if safe_for_literal(v)
        return Expr(:call, uninit_fcodual, v)
    else
        return Expr(:call, identity, add_data!(info, uninit_fcodual(v)))
    end
end

"""
    const_codual(stmt, info::ADInfo)

Build a `CoDual` from `stmt`, with zero / uninitialised fdata. If the resulting CoDual is
a bits type, then it is returned. If it is not, then the CoDual is put into shared data,
and the ID associated to it in the forwards- and reverse-passes returned.
"""
function const_codual(stmt, info::ADInfo)
    v = get_const_primal_value(stmt)
    x = uninit_fcodual(v)
    return safe_for_literal(v) ? x : add_data!(info, x)
end

function safe_for_literal(v)
    v isa Expr && v.head === :boundscheck && return true
    v isa String && return true
    v isa Type && return true
    v isa Tuple && all(safe_for_literal, v) && return true
    isbitstype(_typeof(v)) && return true
    return false
end

inc_or_const(stmt, info::ADInfo) = is_active(stmt) ? __inc(stmt) : const_codual(stmt, info)

function inc_or_const_stmt(stmt, info::ADInfo)
    return if is_active(stmt)
        Expr(:call, identity, __inc(stmt))
    else
        const_codual_stmt(stmt, info)
    end
end

"""
    get_const_primal_value(x::GlobalRef)

Get the value associated to `x`. For `GlobalRef`s, verify that `x` is indeed a constant.
"""
function get_const_primal_value(x::GlobalRef)
    isconst(x) || unhandled_feature("Non-constant GlobalRef not supported: $x")
    return getglobal(x.mod, x.name)
end
get_const_primal_value(x::QuoteNode) = x.value
get_const_primal_value(x::Expr) = eval(x)
get_const_primal_value(x) = x

# Mooncake does not yet handle `PhiCNode`s. Throw an error if one is encountered.
function make_ad_stmts!(stmt::Core.PhiCNode, ::ID, ::ADInfo)
    return unhandled_feature("Encountered PhiCNode: $stmt")
end

# Mooncake does not yet handle `UpsilonNode`s. Throw an error if one is encountered.
function make_ad_stmts!(stmt::Core.UpsilonNode, ::ID, ::ADInfo)
    return unhandled_feature(
        "Encountered UpsilonNode: $stmt. These are generated as part of some try / catch " *
        "/ finally blocks. At the present time, Mooncake.jl cannot differentiate through " *
        "these, so they must be avoided. Strategies for resolving this error include " *
        "re-writing code such that it avoids generating any UpsilonNodes, or writing a " *
        "rule to differentiate the code by hand. If you are in any doubt as to what to " *
        "do, please request assistance by opening an issue at " *
        "github.com/chalk-lab/Mooncake.jl.",
    )
end

# There are quite a number of possible `Expr`s that can be encountered. This `:call` /
# `:invoke` path stays mostly linear on purpose: keeping rule selection, forward emission,
# and reverse emission together makes the translated dataflow easier to inspect and avoids
# introducing helper boundaries that can interfere with inlining in unstable cases.
function make_ad_stmts!(stmt::Expr, line::ID, info::ADInfo)
    is_invoke = Meta.isexpr(stmt, :invoke)
    if Meta.isexpr(stmt, :call) || is_invoke

        #
        # Step 1: classify the call site and choose the rule object.
        #

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
        args = ((is_invoke ? stmt.args[2:end] : stmt.args)...,)
        if !is_used(info, line) && get_const_primal_value(args[1]) == getfield
            fwds = new_inst(Expr(:call, __fwds_pass_no_ad!, map(__inc, args)...))
            return ad_stmt_info(line, nothing, fwds, nothing)
        end

        selection = _select_rule(stmt, line, info, is_invoke)

        #
        # Step 2: write the forward fragment.
        #
        # These statements are written out manually because routing them through a helper can
        # prevent inlining in type-unstable situations.
        #

        # Make arguments to rrule call. Things which are not already CoDual must be made so.
        codual_args = IDInstPair[]
        codual_arg_ids = map(selection.args) do arg
            is_active(arg) && return __inc(arg)
            id = ID()
            push!(codual_args, (id, new_inst(inc_or_const_stmt(arg, info))))
            return id
        end

        # Make call to rule.
        rule_call_id = ID()
        rule_call = Expr(:call, selection.rule_ref, codual_arg_ids...)

        # Extract the output-codual from the returned tuple.
        raw_output_id = ID()
        raw_output = Expr(:call, getfield, rule_call_id, 1)

        # Extract the pullback from the returned tuple. Specialise on the case that the
        # pullback is provably a singleton type.
        if Base.issingletontype(selection.T_pb!!)
            pb = selection.T_pb!!.instance
            pb_stmt = (ID(), new_inst(nothing))
            comms_id = nothing
        else
            pb = ID()
            pb_stmt = (
                pb, new_inst(Expr(:call, getfield, rule_call_id, 2), selection.T_pb!!)
            )
            comms_id = pb
        end

        # Provide a type assertion to help the compiler out. Doing it this way, rather than
        # directly changing the inferred type of the instruction associated to raw_output,
        # has the advantage of not introducing the possibility of segfaults. It will still
        # be optimised away in situations where the compiler is able to successfully infer
        # the type, so performance in performance-critical situations is unaffected.
        output_id = line
        F = fcodual_type(selection.output_type)
        output = Expr(:call, Core.typeassert, raw_output_id, F)

        # Create statements associated to forwards-pass.
        fwds = vcat(
            codual_args,
            IDInstPair[
                (rule_call_id, new_inst(rule_call)),
                (raw_output_id, new_inst(raw_output)),
                pb_stmt,
                (output_id, new_inst(output)),
            ],
        )

        #
        # Step 3: write the reverse fragment.
        #
        # If the reverse pass is provably `NoPullback`, there is nothing to emit.
        rvs_pass = if selection.T_pb!! <: NoPullback
            nothing
        else
            # Get the rdata which we pass into the pullback from its rdata ref.
            rdata_ref_id = get_rev_data_id(info, line)
            rdata_output_id = ID()
            rdata_output_expr = Expr(:call, getfield, rdata_ref_id, QuoteNode(:x))
            rdata_output = (rdata_output_id, new_inst(rdata_output_expr))

            # Zero out the value stored in this rdata ref now that we have its current
            # value. The new value is rdata, so must be an instance of a bits type, so is
            # safe to interpolate straight into instruction.
            zero_val = zero_like_rdata_from_type(selection.output_type)
            zero_rdata_expr = Expr(:call, setfield!, rdata_ref_id, QuoteNode(:x), zero_val)
            zero_rdata_ref = (ID(), new_inst(zero_rdata_expr))

            # Run the pullback. The result is a tuple comprising `length(args)` elements.
            call_pullback_id = ID()
            call_pullback = (call_pullback_id, new_inst(Expr(:call, pb, rdata_output_id)))
            pullback_increments = _pullback_increment_stmts(
                info, selection.args, call_pullback_id
            )
            vcat(
                IDInstPair[rdata_output, zero_rdata_ref, call_pullback],
                pullback_increments,
            )
        end
        return ad_stmt_info(line, comms_id, fwds, rvs_pass)

    elseif Meta.isexpr(stmt, :boundscheck)
        # For some reason the compiler cannot handle boundscheck statements when we run it
        # again. Consequently, emit `true` to be safe. Ideally we would handle this in a
        # more natural way, but I'm not sure how to do that.
        return ad_stmt_info(line, nothing, zero_fcodual(true), nothing)

    elseif Meta.isexpr(stmt, :code_coverage_effect)
        # Code coverage irrelevant for derived code, and really inflates it in some
        # situations. Since code coverage is usually only requrested during CI, including
        # these effects also creates differences between the code generated when developing
        # and the code generated in CI, which occassionally creates hard-to-debug issues.
        return ad_stmt_info(line, nothing, nothing, nothing)

    elseif Meta.isexpr(stmt, :copyast)
        # Get constant out and shove it in shared storage.
        return ad_stmt_info(line, nothing, const_codual_stmt(stmt.args[1], info), nothing)

    elseif Meta.isexpr(stmt, :loopinfo)
        # Cannot pass loopinfo back through the optimiser for some reason.
        # At the time of writing, I am unclear why this is not possible.
        return ad_stmt_info(line, nothing, nothing, nothing)

    elseif stmt.head in [
        :enter,
        :gc_preserve_begin,
        :gc_preserve_end,
        :leave,
        :pop_exception,
        :throw_undef_if_not,
        :meta,
    ]
        # Expressions which do not require any special treatment.
        return ad_stmt_info(line, nothing, stmt, nothing)

    elseif stmt.head == :(=) && stmt.args[1] isa GlobalRef
        msg =
            "Encountered assignment to global variable: $(stmt.args[1]). " *
            "Cannot differentiate through assignments to globals. " *
            "Please refactor your code to avoid assigning to a global, for example by " *
            "passing the variable in to the function as an argument."
        unhandled_feature(msg)
    else
        # Encountered an expression that we've not seen before.
        throw(error("Unrecognised expression $stmt"))
    end
end

"""
    increment_ref_stmts(ref_id::ID, inc_data)::Vector{IDInstPair}

Equivalent to `ref[] = increment!!(ref[], inc_data)`, where `ref` and `inc_data` are the
values associated to `ref_id` and `inc_data` respectively.
"""
function increment_ref_stmts(ref_id::ID, inc_data)::Vector{IDInstPair}

    # Get the value stored in the `Base.RefValue`.
    ref_val_id = ID()
    ref_val = (ref_val_id, new_inst(Expr(:call, getfield, ref_id, QuoteNode(:x))))

    # Increment the value by inc_data.
    new_val_id = ID()
    new_val = (new_val_id, new_inst(Expr(:call, increment!!, ref_val_id, inc_data)))

    # Update the value stored in the rdata reference.
    set_ref_expr = Expr(:call, setfield!, ref_id, QuoteNode(:x), new_val_id)
    set_ref = (ID(), new_inst(set_ref_expr))

    return IDInstPair[ref_val, new_val, set_ref]
end

is_active(::Union{Argument,ID}) = true
is_active(::Any) = false

"""
    pullback_type(Trule, arg_types)

Get a bound on the pullback type, given a rule and associated primal types.
"""
function pullback_type(Trule, arg_types)
    T = Core.Compiler.return_type(Tuple{Trule,map(fcodual_type, arg_types)...})
    return T <: Tuple ? _pullback_type(T) : Any
end

_pullback_type(::Core.TypeofBottom) = Any
_pullback_type(T::DataType) = length(T.parameters) >= 2 ? T.parameters[2] : Any
_pullback_type(T::Union) = Union{_pullback_type(T.a),_pullback_type(T.b)}

# Used by the getfield special-case in call / invoke statments.
@inline function __fwds_pass_no_ad!(f::F, raw_args::Vararg{Any,N}) where {F,N}
    return tuple_splat(__get_primal(f), tuple_map(__get_primal, raw_args))
end

__get_primal(x::CoDual) = primal(x)
__get_primal(x) = x

const RuleMC{A,R} = MistyClosure{OpaqueClosure{A,R}}

#
# Runtime wrapper types for generated rules.
#
# These wrappers sit on the hot path once a rule has already been derived. Their main job is
# to hide closure/capture details and translate between differing varargs conventions.
#

struct Pullback{Tprimal,Tpb_args,Tpb_ret,isva,nargs}
    pb_oc::Base.RefValue{RuleMC{Tpb_args,Tpb_ret}}
end

function Pullback(sig, pb_oc::Ref{<:RuleMC{A,R}}, isva::Bool, nargs::Int) where {A,R}
    return Pullback{sig,A,R,isva,nargs}(pb_oc)
end

_isva(::Pullback{<:Any,<:Any,<:Any,isva}) where {isva} = isva
_nargs(::Pullback{<:Any,<:Any,<:Any,<:Any,nargs}) where {nargs} = nargs
function nvargs(pb::Pullback{sig}) where {sig}
    return Val{_isva(pb) ? _nargs(pb) - length(sig.parameters) + 1 : 0}
end

@inline (pb::Pullback)(dy) = __flatten_varargs(_isva(pb), pb.pb_oc[](dy), nvargs(pb)())

struct DerivedRule{Tprimal,Tfwd_args,Tfwd_ret,Tpb_args,Tpb_ret,isva,Tnargs<:Val}
    fwds_oc::RuleMC{Tfwd_args,Tfwd_ret}
    pb_oc_ref::Base.RefValue{RuleMC{Tpb_args,Tpb_ret}}
    nargs::Tnargs
end

_isva(::DerivedRule{A,B,C,D,E,isva}) where {A,B,C,D,E,isva} = isva

function DerivedRule(
    sig, fwds_oc::RuleMC{FA,FR}, pb_oc::Base.RefValue{RuleMC{RA,RR}}, isva::Bool, nargs::W
) where {FA,FR,RA,RR,W}
    return DerivedRule{sig,FA,FR,RA,RR,isva,W}(fwds_oc, pb_oc, nargs)
end

# Extends functionality defined for debug_mode.
function verify_args(r::DerivedRule{sig}, x) where {sig}
    Tx = Tuple{map(_typeof ∘ primal, __unflatten_codual_varargs(_isva(r), x, r.nargs))...}
    Tx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tx do not subtype rule signature, $sig"))
end

function _copy(x::P) where {P<:DerivedRule}
    new_captures = _copy(x.fwds_oc.oc.captures)
    new_fwds_oc = replace_captures(x.fwds_oc, new_captures)
    new_pb_oc_ref = Ref(replace_captures(x.pb_oc_ref[], new_captures))
    return P(new_fwds_oc, new_pb_oc_ref, x.nargs)
end

@inline function (fwds::DerivedRule{sig})(args::Vararg{CoDual,N}) where {sig,N}
    uf_args = __unflatten_codual_varargs(_isva(fwds), args, fwds.nargs)
    pb = Pullback(sig, fwds.pb_oc_ref, _isva(fwds), N)
    return fwds.fwds_oc(uf_args...)::CoDual, pb
end

# On Julia 1.10, restore type stability lost to the inferencebarrier in __call_rule by
# asserting the return type. Both the CoDual and Pullback types are encoded in DerivedRule's
# type parameters; the Pullback's nargs comes from the number of args at the call site.
@static if VERSION < v"1.11-"
    @inline function __call_rule(
        rule::DerivedRule{Tp,FA,FR,RA,RR,isva,Val{pnargs}}, args::A
    ) where {Tp,FA,FR,RA,RR,isva,pnargs,A<:Tuple}
        return __call_rule_erased!(
            Base.inferencebarrier(rule), args
        )::Tuple{FR,Pullback{Tp,RA,RR,isva,fieldcount(A)}}
    end
end

"""
    __flatten_varargs(isva::Bool, args, ::Val{nvargs}) where {nvargs}

If isva, inputs (5.0, (4.0, 3.0)) are transformed into (5.0, 4.0, 3.0).
"""
function __flatten_varargs(isva::Bool, args, ::Val{nvargs}) where {nvargs}
    isva || return args
    last_el = isa(args[end], NoRData) ? ntuple(n -> NoRData(), nvargs) : args[end]
    return (args[1:(end - 1)]..., last_el...)
end

"""
    __unflatten_codual_varargs(isva::Bool, args, ::Val{nargs}) where {nargs}

If isva and nargs=2, then inputs `(CoDual(5.0, 0.0), CoDual(4.0, 0.0), CoDual(3.0, 0.0))`
are transformed into `(CoDual(5.0, 0.0), CoDual((5.0, 4.0), (0.0, 0.0)))`.
"""
function __unflatten_codual_varargs(isva::Bool, args, ::Val{nargs}) where {nargs}
    isva || return args
    group_primal = map(primal, args[nargs:end])
    if fdata_type(tangent_type(_typeof(group_primal))) == NoFData
        grouped_args = zero_fcodual(group_primal)
    else
        grouped_args = CoDual(group_primal, map(tangent, args[nargs:end]))
    end
    return (args[1:(nargs - 1)]..., grouped_args)
end

#
# Deferred runtime rule wrappers for dynamic dispatch and recursive `:invoke`
#
# These wrappers live next to the other callable rule wrappers above because they are also
# part of the runtime surface seen by generated reverse-mode code. Their constructors depend
# on compilation helpers such as `build_rrule` and `rule_type`, which are defined later.
#

"""
    DynamicDerivedRule(interp::MooncakeInterpreter, debug_mode::Bool)

For internal use only.

A callable data structure which, when invoked, calls an rrule specific to the dynamic types
of its arguments. Stores rules in an internal cache to avoid re-deriving.

This is used to implement dynamic dispatch.
"""
struct DynamicDerivedRule{V}
    cache::V
    debug_mode::Bool
end

DynamicDerivedRule(debug_mode::Bool) = DynamicDerivedRule(Dict{Any,Any}(), debug_mode)

# Create new dynamic rule with empty cache and same debug mode
_copy(x::P) where {P<:DynamicDerivedRule} = P(Dict{Any,Any}(), x.debug_mode)

function (dynamic_rule::DynamicDerivedRule)(args::Vararg{Any,N}) where {N}

    # `Base._stable_typeof` is used here, rather than `typeof` or `Mooncake._typeof`. Its
    # precise behaviour (equivalent to `typeof` for everything except `Type`s, for which it
    # returns `Type{P}` rather than `typeof(P)`) is needed to ensure that this signature
    # matches the types that `rule` sees when `rule(args...)` is called below. If you get
    # this wrong, an assertion is violated, causing a hard-to-debug error (see issue 660).
    sig = Tuple{map(Base._stable_typeof ∘ primal, args)...}

    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        interp = get_interpreter(ReverseMode)
        rule = build_rrule(interp, sig; debug_mode=dynamic_rule.debug_mode)
        dynamic_rule.cache[sig] = rule
    end
    return __call_rule(rule, args)
end

"""
    LazyDerivedRule(interp, mi::Core.MethodInstance, debug_mode::Bool)

For internal use only.

A type-stable wrapper around a `DerivedRule`, which only instantiates the `DerivedRule`
when it is first called. This is useful, as it means that if a rule does not get run, it
does not have to be derived.

If `debug_mode` is `true`, then the rule constructed will be a `DebugRRule`. This is useful
when debugging, but should usually be switched off for production code as it (in general)
incurs some runtime overhead.

Note: the signature of the primal for which this is a rule is stored in the type. The only
reason to keep this around is for debugging -- it is very helpful to have this type visible
in the stack trace when something goes wrong, as it allows you to trivially determine which
bit of your code is the culprit.

# Extended Help

There are two main reasons why deferring the construction of a `DerivedRule` until we need
to use it is crucial.

The first is to do with recursion. Consider the following function:
```julia
f(x) = x > 0 ? f(x - 1) : x
```
If we generate the `IRCode` for this function, we will see something like the following:
```julia
julia> Base.code_ircode_by_type(Tuple{typeof(f), Float64})[1][1]
1 1 ─ %1  = Base.lt_float(0.0, _2)::Bool
  │   %2  = Base.or_int(%1, false)::Bool
  └──       goto #6 if not %2
  2 ─ %4  = Base.sub_float(_2, 1.0)::Float64
  │   %5  = Base.lt_float(0.0, %4)::Bool
  │   %6  = Base.or_int(%5, false)::Bool
  └──       goto #4 if not %6
  3 ─ %8  = Base.sub_float(%4, 1.0)::Float64
  │   %9  = invoke Main.f(%8::Float64)::Float64
  └──       goto #5
  4 ─       goto #5
  5 ┄ %12 = φ (#3 => %9, #4 => %4)::Float64
  └──       return %12
  6 ─       return _2
```
Suppose that we decide to construct a `DerivedRule` immediately whenever we find an
`:invoke` statement in a rule that we're currently building a `DerivedRule` for.
In the above example, we produce an infinite recursion when we attempt to produce a
`DerivedRule` for %9, because it has the same signature as the call which generates this IR.
By instead adopting a policy of constructing a `LazyDerivedRule` whenever we encounter an
`:invoke` statement, we avoid this problem.

The second reason that delaying the construction of a `DerivedRule`, is essential is that it
ensures that we don't derive rules for method instances which aren't run. Suppose that
function B contains code for which we can't derive a rule -- perhaps it contains an
unsupported language feature like a `PhiCNode` or an `UpsilonNode`. Suppose that function A
contains an `:invoke` which refers to function `B`, but that this call is on a branch which
deals with error handling, and doesn't get run run unless something goes wrong. By deferring
the derivation of the rule for B, we only ever attempt to derive it if we land on this
error handling branch. Conversely, if we attempted to derive the rule for B when we derive
the rule for A, we would be unable to complete the derivation of the rule for A.
"""
mutable struct LazyDerivedRule{primal_sig,Trule}
    debug_mode::Bool
    mi::Core.MethodInstance
    rule::Trule
    function LazyDerivedRule(mi::Core.MethodInstance, debug_mode::Bool)
        interp = get_interpreter(ReverseMode)
        return new{mi.specTypes,rule_type(interp, mi;debug_mode)}(debug_mode, mi)
    end
    function LazyDerivedRule{Tprimal_sig,Trule}(
        mi::Core.MethodInstance, debug_mode::Bool
    ) where {Tprimal_sig,Trule}
        return new{Tprimal_sig,Trule}(debug_mode, mi)
    end
end

# Create new lazy rule with same method instance and debug mode
_copy(x::P) where {P<:LazyDerivedRule} = P(x.mi, x.debug_mode)

# On Julia 1.10, the generic __call_rule fallback is @stable-checked and returns Any for
# LazyDerivedRule, triggering TypeInstabilityError when dispatch_doctor_mode = "error".
# Add type-asserting specialisations so callers in @stable contexts see a concrete type.
# LazyDerivedRule doesn't contain an OpaqueClosure directly, so no inferencebarrier needed.
@static if VERSION < v"1.11-"
    @inline function __call_rule(
        rule::LazyDerivedRule{sig,DerivedRule{Tp,FA,FR,RA,RR,isva,Val{pnargs}}}, args::A
    ) where {sig,Tp,FA,FR,RA,RR,isva,pnargs,A<:Tuple}
        return rule(args...)::Tuple{FR,Pullback{Tp,RA,RR,isva,fieldcount(A)}}
    end
    @inline function __call_rule(
        rule::LazyDerivedRule{
            sig,DebugRRule{DerivedRule{Tp,FA,CoDual{P,FD},RA,RR,isva,Val{pnargs}}}
        },
        args::A,
    ) where {sig,Tp,FA,P,FD,RA,RR,isva,pnargs,A<:Tuple}
        return rule(
            args...
        )::Tuple{CoDual{P,FD},DebugPullback{Pullback{Tp,RA,RR,isva,fieldcount(A)},P}}
    end
end

@inline function (rule::LazyDerivedRule)(args::Vararg{Any,N}) where {N}
    return isdefined(rule, :rule) ? __call_rule(rule.rule, args) : _build_rule!(rule, args)
end

@noinline function _build_rule!(rule::LazyDerivedRule{sig,Trule}, args) where {sig,Trule}
    interp = get_interpreter(ReverseMode)
    rule.rule = build_rrule(interp, rule.mi; debug_mode=rule.debug_mode)
    return __call_rule(rule.rule, args)
end

#
# Rule derivation entry points and compile-time helpers
#

_get_sig(sig::Type) = sig
_get_sig(mi::Core.MethodInstance) = mi.specTypes
_get_sig(mc::MistyClosure) = Tuple{map(CC.widenconst, mc.ir[].argtypes)...}

"""
Flatten the signature of a vararg method to group the
possibly multiple vararg arguments (what users pass to the function)
into a single tuple argument matching `ir.argtypes`.
"""
function flatten_va_sig(sig, isva, nargs)
    @nospecialize sig
    return if isva
        Tuple{sig.parameters[1:(nargs - 1)]...,Tuple{sig.parameters[nargs:end]...}}
    else
        sig
    end
end

function forwards_ret_type(primal_ir::IRCode)
    return fcodual_type(compute_ir_rettype(primal_ir))
end

function pullback_ret_type(primal_ir::IRCode)
    return Tuple{map(rdata_type ∘ tangent_type ∘ CC.widenconst, primal_ir.argtypes)...}
end

struct MooncakeRuleCompilationError <: Exception
    interp::MooncakeInterpreter
    sig
    debug_mode::Bool
    cause::Exception
end

function Base.showerror(io::IO, err::MooncakeRuleCompilationError)
    msg_lines = (
        "MooncakeRuleCompilationError: an error occurred while Mooncake was compiling a",
        "rule to differentiate something. If the `caused by` error message below does",
        "not make it clear to you how the problem can be fixed, please open an issue",
        "at github.com/chalk-lab/Mooncake.jl describing your problem.",
    )
    cause_width = min(_boxed_message_width(io, "│ "), 78)
    cause_lines = let lines = if hasfield(typeof(err.cause), :msg)
            msg = getfield(err.cause, :msg)
            if msg isa AbstractString
                split(msg, '\n')
            else
                split(sprint(showerror, err.cause), '\n')
            end
        else
            split(sprint(showerror, err.cause), '\n')
        end
        while !isempty(lines) && isempty(last(lines))
            pop!(lines)
        end
        wrapped_lines = String[]
        for line in lines
            append!(wrapped_lines, _wrap_boxed_line(line, cause_width))
        end
        wrapped_lines
    end
    detail_lines = ("Caused by:", cause_lines..., "", msg_lines...)

    # Print the source location of the method being differentiated, if available.
    try
        m = lookup_method(err.sig)
        if m !== nothing
            mstr = sprint(show, m)
            header, location = let parts = split(mstr, " @ "; limit=2)
                length(parts) == 2 ? (parts[1], parts[2]) : (mstr, nothing)
            end
            _print_boxed_error(
                io,
                (
                    "Mooncake failed to differentiate the following method:",
                    header,
                    "",
                    detail_lines...,
                );
                footer=isnothing(location) ? nothing : "@ $location",
            )
            println(io)  # blank line before the main error body
        else
            _print_boxed_error(io, detail_lines)
            println(io)
        end
    catch e
        # If method lookup fails for any reason, skip gracefully.
        @debug "MooncakeRuleCompilationError: method lookup failed" exception = e
        _print_boxed_error(io, detail_lines)
        println(io)
    end
    println(io, "To replicate this error run the following:\n")
    println(
        io,
        "Mooncake.build_rrule(Mooncake.$(err.interp), $(err.sig); debug_mode=$(err.debug_mode))",
    )
    return println(
        io,
        "\nNote that you may need to `using` some additional packages if not all of the " *
        "names printed in the above signature are available currently in your environment.",
    )
end

"""
    build_rrule(args...; kwargs...)

Helper method: equivalent to extracting the signature from `args` and calling
`build_rrule(sig; kwargs...)`.
"""
function build_rrule(args...; kwargs...)
    interp = get_interpreter(ReverseMode)
    return build_rrule(interp, _typeof(TestUtils.__get_primals(args)); kwargs...)
end

"""
    build_rrule(sig::Type{<:Tuple}; kwargs...)

Helper method: Equivalent to
`build_rrule(Mooncake.get_interpreter(ReverseMode), sig; kwargs...)`.
"""
function build_rrule(sig::Type{<:Tuple}; kwargs...)
    return build_rrule(get_interpreter(ReverseMode), sig; kwargs...)
end

const MOONCAKE_INFERENCE_LOCK = ReentrantLock()

struct DerivedRuleInfo
    primal_ir::IRCode
    fwd_ir::IRCode
    fwd_ret_type::Type
    rvs_ir::IRCode
    rvs_ret_type::Type
    shared_data::Tuple
    info::ADInfo
    isva::Bool
end

"""
    build_rrule(interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode=false) where {C}

Returns a `DerivedRule` which is an `rrule!!` for `sig_or_mi` in context `C`. See the
docstring for `rrule!!` for more info.

If `debug_mode` is `true`, then all calls to rules are replaced with calls to `DebugRRule`s.
"""
function build_rrule(
    interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode=false, silence_debug_messages=true
) where {C}
    @nospecialize sig_or_mi

    build_rrule_checks(interp, sig_or_mi, debug_mode, silence_debug_messages)

    # If we have a hand-coded rule, just use that.
    sig = _get_sig(sig_or_mi)
    if is_primitive(C, ReverseMode, sig, interp.world)
        rule = build_primitive_rrule(sig)
        return (debug_mode ? DebugRRule(rule) : rule)
    end

    return build_derived_rrule(interp, sig_or_mi, sig, debug_mode)
end

# Separated out so we can make an frule!! for it, for forward-over-reverse.
function build_rrule_checks(
    interp::MooncakeInterpreter, sig_or_mi, debug_mode::Bool, silence_debug_messages::Bool
)
    @nospecialize sig_or_mi

    # To avoid segfaults, ensure that we bail out if the interpreter's world age is greater
    # than the current world age.
    if Base.get_world_counter() > interp.world
        throw(
            ArgumentError(
                "World age associated to interp is behind current world age. Please " *
                "create a new interpreter for the current world age.",
            ),
        )
    end

    # If we're compiling in debug mode, let the user know by default.
    if !silence_debug_messages && debug_mode
        @info "Compiling rule for $sig_or_mi in debug mode. Disable for best performance."
    end
end

function build_derived_rrule(
    interp::MooncakeInterpreter{C}, sig_or_mi, sig, debug_mode::Bool
) where {C}
    @nospecialize sig_or_mi sig

    # No hand-coded rule exists, so derive one from compiler IR.
    lock(MOONCAKE_INFERENCE_LOCK)
    try
        # If we've already derived the OpaqueClosures and info, do not re-derive, just
        # create a copy and pass in new shared data.
        oc_cache_key = ClosureCacheKey(interp.world, (sig_or_mi, debug_mode, :reverse))
        if haskey(interp.oc_cache, oc_cache_key)
            return _copy(interp.oc_cache[oc_cache_key])
        else
            # Derive the forward and reverse IR, then package them into `MistyClosure`s.
            dri = try
                generate_ir(interp, sig_or_mi; debug_mode)
            catch err
                # Julia 1.10 can hit this IR-interpreter limitation during optimization on
                # otherwise valid derived reverse rules. Retry without optimize_ir! so rule
                # construction remains robust for those methods.
                if err isa AssertionError && occursin(
                    "irinterp is unable to handle heavy recursion",
                    sprint(showerror, err),
                )
                    generate_ir(interp, sig_or_mi; debug_mode, do_optimize=false)
                else
                    rethrow()
                end
            end
            fwd_oc = misty_closure(dri.fwd_ret_type, dri.fwd_ir, dri.shared_data...)
            rvs_oc = misty_closure(dri.rvs_ret_type, dri.rvs_ir, dri.shared_data...)

            # Compute the signature. Needs careful handling with varargs.
            nargs = num_args(dri.info)
            sig = flatten_va_sig(sig, dri.isva, nargs)
            raw_rule = DerivedRule(sig, fwd_oc, Ref(rvs_oc), dri.isva, Val(nargs))
            rule = debug_mode ? DebugRRule(raw_rule) : raw_rule
            interp.oc_cache[oc_cache_key] = rule
            return rule
        end
    catch e
        if e isa MooncakeRuleCompilationError
            rethrow(e)
        else
            sig = sig_or_mi isa Core.MethodInstance ? sig_or_mi.specTypes : sig_or_mi
            throw(MooncakeRuleCompilationError(interp, sig, debug_mode, e))
        end
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

"""
    generate_ir(
        interp::MooncakeInterpreter, sig_or_mi; debug_mode=false, do_inline=true
)
Used by `build_rrule`, and the various debugging tools: primal_ir, fwds_ir, adjoint_ir.
"""
function generate_ir(
    interp::MooncakeInterpreter,
    sig_or_mi;
    debug_mode=false,
    do_inline=true,
    do_optimize=true,
)
    # Reset id count. This ensures that the IDs generated are the same each time this
    # function runs.
    seed_id!()

    # Look up the inferred primal IR.
    ir, _ = lookup_ir(interp, sig_or_mi)
    @static if VERSION > v"1.12-"
        ir = set_valid_world!(ir, interp.world)
    end
    Treturn = compute_ir_rettype(ir)
    fwd_ret_type = forwards_ret_type(ir)
    rvs_ret_type = pullback_ret_type(ir)

    # Check before normalise! to avoid a cryptic CC.verify_ir failure downstream.
    for inst in stmt(ir.stmts)
        is_enter = Meta.isexpr(inst, :enter)
        @static if isdefined(Core, :EnterNode)
            is_enter = is_enter || inst isa Core.EnterNode
        end
        if is_enter
            unhandled_feature(
                "try/catch/finally blocks are not supported by Mooncake.jl in reverse " *
                "mode. The code being differentiated contains a try/catch or try/finally " *
                "construct. Strategies for resolving this error include re-writing code " *
                "to avoid try/catch blocks (e.g. by replacing them with explicit " *
                "conditional checks), or providing a custom rrule!! for the relevant " *
                "function. See the known limitations documentation for more context.",
            )
        end
    end

    # Reverse mode now starts from normalized IRCode and uses the local CFG builder directly.
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    ir = normalise!(ir, spnames)
    primal_blocks = _remove_unreachable_cfg_blocks!(_ircode_to_cfg_blocks(ir))

    # Compute global info.
    arg_types = Dict{Argument,Any}(
        map(((n, t),) -> (Argument(n) => CC.widenconst(t)), enumerate(ir.argtypes))
    )
    _, ssa_insts, is_used_dict = _primal_stmt_metadata(primal_blocks)
    Tlazy_rdata_ref = Tuple{map(lazy_zero_rdata_type ∘ CC.widenconst, ir.argtypes)...}
    zero_lazy_rdata_ref = Ref{Tlazy_rdata_ref}()
    info = ADInfo(
        interp,
        arg_types,
        ssa_insts,
        is_used_dict,
        debug_mode,
        zero_lazy_rdata_ref,
        fwd_ret_type,
        rvs_ret_type,
    )

    # For each block in the primal CFG, translate all statements. Running this will, in
    # general, push items to `info.shared_data_pairs`.
    ad_stmts_blocks = map(primal_blocks) do primal_blk
        ids = first.(primal_blk.insts)
        stmts = map(x -> x[2].stmt, primal_blk.insts)
        return (primal_blk.id, make_ad_stmts!.(stmts, ids, Ref(info)))
    end

    # Make shared data, and construct IR for the forwards-pass and pullback.
    block_comms = create_comms_insts!(ad_stmts_blocks, info)
    shared_data = shared_data_tuple(info.shared_data_pairs)

    fwd_ir = forwards_pass_ir(
        ir, primal_blocks, ad_stmts_blocks, block_comms, info, _typeof(shared_data)
    )
    rvs_ir = pullback_ir(
        ir, primal_blocks, Treturn, ad_stmts_blocks, block_comms, info, _typeof(shared_data)
    )
    opt_fwd_ir = do_optimize ? optimise_ir!(fwd_ir; do_inline) : fwd_ir
    opt_rvs_ir = do_optimize ? optimise_ir!(rvs_ir; do_inline) : rvs_ir
    return DerivedRuleInfo(
        ir, opt_fwd_ir, fwd_ret_type, opt_rvs_ir, rvs_ret_type, shared_data, info, isva
    )
end

"""
    replace_captures(oc::Toc, new_captures) where {Toc<:OpaqueClosure}

Given an `OpaqueClosure` `oc`, create a new `OpaqueClosure` of the same type, but with new
captured variables. This is needed for efficiency reasons -- if `build_rrule` is called
repeatedly with the same signature and intepreter, it is important to avoid recompiling
the `OpaqueClosure`s that it produces multiple times, because it can be quite expensive to
do so.
"""
function replace_captures(oc::Toc, new_captures) where {Toc<:OpaqueClosure}
    return __replace_captures_internal(oc, new_captures)
end

@eval function __replace_captures_internal(oc::Toc, new_captures) where {Toc<:OpaqueClosure}
    return $(Expr(
        :new, :(Toc), :new_captures, :(oc.world), :(oc.source), :(oc.invoke), :(oc.specptr)
    ))
end

"""
    replace_captures(mc::Tmc, new_captures) where {Tmc<:MistyClosure}

Same as `replace_captures` for `Core.OpaqueClosure`s, but returns a new `MistyClosure`.
"""
function replace_captures(mc::Tmc, new_captures) where {Tmc<:MistyClosure}
    return Tmc(replace_captures(mc.oc, new_captures), mc.ir)
end

const ADStmts = Vector{Tuple{ID,Vector{ADStmtInfo}}}

function _flatten_cfg_insts(blocks)::Vector{IDInstPair}
    return [inst for block in blocks for inst in block.insts]
end

function _primal_stmt_metadata(blocks)
    primal_stmts = _flatten_cfg_insts(blocks)
    return primal_stmts,
    Dict{ID,NewInstruction}(primal_stmts),
    characterise_used_ids(primal_stmts)
end

"""
    create_comms_insts!(ad_stmts_blocks::ADStmts, info::ADInfo)

This function produces code which can be inserted into the forwards-pass and reverse-pass at
specific locations to implement the promise associated to the `comms_id` field of the
`ADStmtInfo` type -- namely that if you assign a value to `comms_id` on the forwards-pass,
the same value will be available at `comms_id` on the reverse-pass.

For each basic block represented in `ADStmts`:
1. create a stack containing a `Tuple` which can hold all of the values associated to the
    `comms_id`s for each statement. Put this stack in shared data.
2. create instructions which can be inserted at the _end_ of the block generated to perform
    the forwards-pass (in `forwards_pass_ir`) which will put all of the data associated to
    the `comms_id`s into shared data, and
3. create instruction which can be inserted at the _start_ of the block generated to perform
    the reverse-pass (in `pullback_ir`), which will extract all of the data put into
    shared data by the instructions generated by the previous point, and assigned them to
    the `comms_id`s.

Returns a `Vector{BlockCommsInsts}`. The nth element contains the forward-pass suffix and
reverse-pass prefix associated to the nth block in `ad_stmts_blocks`.
"""
#
# Forward-pass communication and CFG assembly
#

function create_comms_insts!(ad_stmts_blocks::ADStmts, info::ADInfo)
    return map(ad_stmts_blocks) do (_, ad_stmts)

        # Get the communication channel for each stmt which has one.
        comms_channels = filter(!=(nothing), map(comms_channel, ad_stmts))
        comms_ids = map(first, comms_channels)

        # Determine the type of the tuple which will contain these, and create a stack which
        # can hold such tuples. Put this stack in shared data, and get its `ID`.
        # Optimise for the case that `TT` is a singleton type -- the result of this
        # optimisation is to directly insert the stack into the code if `TT` is a singleton
        # type, and avoid adding anything to shared data. One notable case in which this
        # will hold is when comms_ids is empty.
        TT = Tuple{map(x -> x[2].type, comms_channels)...}
        stack = Base.issingletontype(TT) ? SingletonStack{TT}() : Stack{TT}()
        comms_stack_id = add_data_if_not_singleton!(info, stack)

        # Create instructions for forwards-pass to create tuple + push onto comms stack.
        tuple_id = ID()
        fwds_insts = IDInstPair[
            (tuple_id, new_inst(Expr(:call, tuple, comms_ids...))),
            (ID(), new_inst(Expr(:call, push!, comms_stack_id, tuple_id))),
        ]

        # Create instructions for reverse-pass to pop comms stack and extract elements of
        # tuple into comms ids.
        rvs_insts = IDInstPair[
            (tuple_id, new_inst(Expr(:call, pop!, comms_stack_id))),
            map(enumerate(comms_ids)) do (n, id)
                (id, new_inst(Expr(:call, getfield, tuple_id, n)))
            end...,
        ]

        return BlockCommsInsts(fwds_insts, rvs_insts)
    end
end

"""
    forwards_pass_ir(
        ir::IRCode,
        primal_blocks,
        ad_stmts_blocks::ADStmts,
        info::ADInfo,
        Tshared_data,
    )

Produce the IR associated to the `OpaqueClosure` which runs most of the forwards-pass.
"""
function forwards_pass_ir(
    ir::IRCode,
    primal_blocks,
    ad_stmts_blocks::ADStmts,
    block_comms,
    info::ADInfo,
    Tshared_data,
)
    is_unique_pred, pred_is_unique_pred = _characterise_unique_predecessor_blocks(
        primal_blocks
    )

    # Insert a block at the start which extracts all items from the captures field of the
    # `OpaqueClosure`, which contains all of the data shared between the forwards- and
    # reverse-passes. These are assigned to the `ID`s given by the `SharedDataPairs`.
    # Push the entry id onto the block stack if needed. Create `LazyZeroRData` for each
    # argument, and put it in the `Ref` for use on the reverse-pass.
    sds = shared_data_stmts(info.shared_data_pairs)
    if pred_is_unique_pred[primal_blocks[1].id]
        push_block_stack_insts = IDInstPair[]
    else
        push_block_stack_stmt = Expr(
            :call, __push_blk_stack!, info.block_stack_id, info.entry_id.id
        )
        push_block_stack_insts = [(ID(), new_inst(push_block_stack_stmt))]
    end
    lazy_zero_rdata_stmt = Expr(
        :call,
        __assemble_lazy_zero_rdata,
        info.lazy_zero_rdata_ref_id,
        map(n -> Argument(n + 1), 1:num_args(info))...,
    )
    lazy_zero_rdata_insts = [(ID(), new_inst(lazy_zero_rdata_stmt))]
    entry_stmts = vcat(sds, lazy_zero_rdata_insts, push_block_stack_insts)
    entry_block = CFGBlock(info.entry_id, entry_stmts)

    # Construct augmented version of each basic block from the primal. For each block:
    # 1. pull the translated basic block statements from ad_stmts_blocks,
    # 2. insert a series of statements to log the contents of the `comms_id`s -- see
    #   the `comms_id` field of `ADStmtInfo`,
    # 3. insert a statement which logs the ID of the current block (if necessary to know
    #   how to perform the reverse-pass),
    # 4. return the CFG block.
    blocks = map(ad_stmts_blocks, block_comms) do (block_id, ad_stmts), comms

        # Extract the `fwds` fields from the stmts, and create the block for the fwds pass.
        insts = reduce(vcat, map(x -> x.fwds, ad_stmts))

        # Insert communication instructions. See `create_comms_insts!` for an explanation.
        for stack_inst in comms.fwds_suffix
            _insert_before_terminator!(insts, stack_inst)
        end

        # Log the ID of the current basic block. This is needed to know which basic block to
        # jump to during the reverse-pass if the current block is not the unique predecessor
        # of each of its successors (in which case there is no need to log that control
        # passed through this block as opposed to any other).
        if !is_unique_pred[block_id]
            ins_stmt = Expr(:call, __push_blk_stack!, info.block_stack_id, block_id.id)
            _insert_before_terminator!(insts, (ID(), new_inst(ins_stmt)))
        end

        return CFGBlock(block_id, insts)
    end

    # Lower the forwards-pass CFG directly to `IRCode`.
    arg_types = vcat(Tshared_data, map(fcodual_type ∘ CC.widenconst, ir.argtypes))
    return lower_cfg_blocks_to_ir(
        ir, arg_types, vcat([entry_block], blocks); sort_cfg=false
    )
end

"""
    __push_blk_stack!(block_stack::BlockStack, id::Int32)

Equivalent to `push!(block_stack, id)`. Going via this function, rather than just calling
push! directly, is helpful for debugging and performance analysis -- it makes it very
straightforward to figure out much time is spent pushing to the block stack when profiling.
"""
@inline __push_blk_stack!(block_stack::BlockStack, id::Int32) = push!(block_stack, id)

__lazy_zero_rdata_primal(T, x) = lazy_zero_rdata(T, primal(x))

@inline @generated function __assemble_lazy_zero_rdata(
    r::Ref{T}, args::Vararg{CoDual,N}
) where {T<:Tuple,N}
    return :(r[] = tuple_map(__lazy_zero_rdata_primal, $(fieldtypes(T)), args))
end

#
# CFGBlock working IR
#
# Reverse mode assembles new control flow in this local representation first, then lowers the
# finished CFG back to compiler IR in one step.
#

"""
    CFGBlock(id::ID, insts::Vector{IDInstPair})

Reverse-mode-local basic block representation used while assembling reverse-mode CFGs before
lowering to compiler IR.
"""
struct CFGBlock
    id::ID
    insts::Vector{IDInstPair}
end

function _remap_assigned_phi_values(f, values::Vector{Any})::Vector{Any}
    # Keep dead-edge phi slots undefined while remapping the assigned incoming values.
    new_values = Vector{Any}(undef, length(values))
    for n in eachindex(values)
        isassigned(values, n) && (new_values[n] = f(values[n]))
    end
    return new_values
end

#
# `IRCode` -> `CFGBlock` conversion
#

function _ssa_to_ids(d::SSAToIdDict, inst::NewInstruction)
    return NewInstruction(inst; stmt=_ssa_to_ids(d, inst.stmt))
end
function _ssa_to_ids(d::SSAToIdDict, x::ReturnNode)
    return isdefined(x, :val) ? ReturnNode(get(d, x.val, x.val)) : x
end
_ssa_to_ids(d::SSAToIdDict, x::Expr) = Expr(x.head, map(a -> get(d, a, a), x.args)...)
_ssa_to_ids(d::SSAToIdDict, x::PiNode) = PiNode(get(d, x.val, x.val), get(d, x.typ, x.typ))
_ssa_to_ids(::SSAToIdDict, x::QuoteNode) = x
_ssa_to_ids(::SSAToIdDict, x) = x
function _ssa_to_ids(d::SSAToIdDict, x::PhiNode)
    return PhiNode(x.edges, _remap_assigned_phi_values(v -> get(d, v, v), x.values))
end
_ssa_to_ids(::SSAToIdDict, x::GotoNode) = x
_ssa_to_ids(d::SSAToIdDict, x::GotoIfNot) = GotoIfNot(get(d, x.cond, x.cond), x.dest)

function _ssas_to_ids(insts::InstVector)::Tuple{Vector{ID},InstVector}
    ids = map(_ -> ID(), insts)
    val_id_map = SSAToIdDict(zip(SSAValue.(eachindex(insts)), ids))
    return ids, map(Base.Fix1(_ssa_to_ids, val_id_map), insts)
end

function _block_num_to_ids(d::BlockNumToIdDict, x::NewInstruction)
    return NewInstruction(x; stmt=_block_num_to_ids(d, x.stmt))
end
function _block_num_to_ids(d::BlockNumToIdDict, x::PhiNode)
    return IDPhiNode(ID[d[e] for e in x.edges], x.values)
end
_block_num_to_ids(d::BlockNumToIdDict, x::GotoNode) = IDGotoNode(d[x.label])
_block_num_to_ids(d::BlockNumToIdDict, x::GotoIfNot) = IDGotoIfNot(x.cond, d[x.dest])
_block_num_to_ids(::BlockNumToIdDict, x) = x

function _block_nums_to_ids(insts::InstVector, cfg::CC.CFG)::Tuple{Vector{ID},InstVector}
    ids = map(_ -> ID(), cfg.blocks)
    block_num_id_map = BlockNumToIdDict(zip(eachindex(cfg.blocks), ids))
    return ids, map(Base.Fix1(_block_num_to_ids, block_num_id_map), insts)
end

function _ircode_to_cfg_blocks(ir::IRCode)::Vector{CFGBlock}
    # Reuse the shared cross-version stmt accessor rather than branching on field names here.
    stmts = map(
        (stmt, type, info, line, flag) -> NewInstruction(stmt, type, info, line, flag),
        stmt(ir.stmts),
        ir.stmts.type,
        ir.stmts.info,
        ir.stmts.line,
        ir.stmts.flag,
    )
    ssa_ids, stmts = _ssas_to_ids(stmts)
    block_ids, stmts = _block_nums_to_ids(stmts, ir.cfg)
    return map(zip(block_ids, ir.cfg.blocks)) do (block_id, bb)
        CFGBlock(block_id, collect(zip(ssa_ids[bb.stmts], stmts[bb.stmts])))
    end
end

_cfg_terminator(stmt) = stmt isa Union{Switch,IDGotoIfNot,IDGotoNode,ReturnNode}
function _cfg_terminator(block::CFGBlock)
    isempty(block.insts) && return nothing
    stmt = last(block.insts)[2].stmt
    return _cfg_terminator(stmt) ? stmt : nothing
end

function _cfg_phi_nodes(block::CFGBlock)
    # Phi nodes are only valid at the start of a block, so stop at the first non-phi.
    n_phi_nodes = findfirst(x -> !(x[2].stmt isa IDPhiNode), block.insts)
    n_phi_nodes = isnothing(n_phi_nodes) ? length(block.insts) : n_phi_nodes - 1
    return first.(block.insts[1:n_phi_nodes]), last.(block.insts[1:n_phi_nodes])
end

#
# CFG analysis and canonicalization helpers
#

function _compute_cfg_successors(blocks::Vector{CFGBlock})::Dict{ID,Vector{ID}}
    succs = Dict{ID,Vector{ID}}()
    for (n, block) in enumerate(blocks)
        is_final_block = n == length(blocks)
        t = _cfg_terminator(block)
        if t === nothing
            succs[block.id] = is_final_block ? ID[] : ID[blocks[n + 1].id]
        elseif t isa IDGotoNode
            succs[block.id] = ID[t.label]
        elseif t isa IDGotoIfNot
            succs[block.id] = is_final_block ? ID[t.dest] : ID[t.dest, blocks[n + 1].id]
        elseif t isa ReturnNode
            succs[block.id] = ID[]
        elseif t isa Switch
            succs[block.id] = vcat(t.dests, t.fallthrough_dest)
        else
            error("Unhandled terminator $t")
        end
    end
    return succs
end

function _compute_cfg_predecessors(blocks::Vector{CFGBlock})::Dict{ID,Vector{ID}}
    successor_map = _compute_cfg_successors(blocks)
    predecessor_map = Dict{ID,Vector{ID}}(block.id => ID[] for block in blocks)
    for (k, succs) in successor_map
        for succ in succs
            push!(predecessor_map[succ], k)
        end
    end
    return predecessor_map
end

function _cfg_distance_to_entry(blocks::Vector{CFGBlock})::Vector{Int}
    id_to_int = Dict(zip(map(block -> block.id, blocks), eachindex(blocks)))
    successors = _compute_cfg_successors(blocks)
    distances = fill(typemax(Int), length(blocks))
    distances[1] = 0
    queue = [blocks[1].id]
    head = 1
    while head <= length(queue)
        block_id = queue[head]
        head += 1
        dist = distances[id_to_int[block_id]]
        for successor in successors[block_id]
            successor_idx = id_to_int[successor]
            if distances[successor_idx] == typemax(Int)
                distances[successor_idx] = dist + 1
                push!(queue, successor)
            end
        end
    end
    return distances
end

function _sort_cfg_blocks!(blocks::Vector{CFGBlock})::Vector{CFGBlock}
    I = sortperm(_cfg_distance_to_entry(blocks))
    blocks .= blocks[I]
    return blocks
end

function _remove_unreachable_cfg_blocks!(blocks::Vector{CFGBlock})::Vector{CFGBlock}
    is_reachable = _cfg_distance_to_entry(blocks) .< typemax(Int)
    remaining_blocks = blocks[is_reachable]
    removed_block_ids = map(idx -> blocks[idx].id, findall(!, is_reachable))
    for block in remaining_blocks, (_, inst) in block.insts
        stmt = inst.stmt
        stmt isa IDPhiNode || continue
        for n in reverse(1:length(stmt.edges))
            if stmt.edges[n] in removed_block_ids
                deleteat!(stmt.edges, n)
                deleteat!(stmt.values, n)
            end
        end
    end
    return remaining_blocks
end

function _characterise_unique_predecessor_blocks(
    blocks::Vector{CFGBlock}
)::Tuple{Dict{ID,Bool},Dict{ID,Bool}}
    block_ids = ID[block.id for block in blocks]
    preds = _compute_cfg_predecessors(blocks)
    succs = _compute_cfg_successors(blocks)

    is_unique_pred = Dict{ID,Bool}()
    for id in block_ids
        ss = succs[id]
        is_unique_pred[id] = !isempty(ss) && all(s -> length(preds[s]) == 1, ss)
    end

    reachable_return_blocks = filter(blocks) do block
        is_reachable_return_node(_cfg_terminator(block))
    end
    if length(reachable_return_blocks) == 1
        is_unique_pred[only(reachable_return_blocks).id] = true
    end

    pred_is_unique_pred = Dict{ID,Bool}()
    for id in block_ids
        pred_is_unique_pred[id] = length(preds[id]) == 1 && is_unique_pred[only(preds[id])]
    end

    entry_id = block_ids[1]
    pred_is_unique_pred[entry_id] = isempty(preds[entry_id])
    return is_unique_pred, pred_is_unique_pred
end

function _insert_before_terminator!(insts::Vector{IDInstPair}, inst::IDInstPair)
    if !isempty(insts) && _cfg_terminator(last(insts)[2].stmt)
        insert!(insts, length(insts), inst)
    else
        push!(insts, inst)
    end
    return insts
end

function _canonicalise_cfg_blocks(blocks::Vector{CFGBlock}; sort_cfg::Bool=true)
    blocks = copy(blocks)
    # Canonicalization is "sort first, then prune" so phi-edge cleanup sees final block order.
    sort_cfg && _sort_cfg_blocks!(blocks)
    return _remove_unreachable_cfg_blocks!(blocks)
end

function _cfg_lower_switch_statements(blocks::Vector{CFGBlock})::Vector{CFGBlock}
    new_blocks = CFGBlock[]
    for block in blocks
        t = _cfg_terminator(block)
        if t isa Switch
            push!(new_blocks, CFGBlock(block.id, block.insts[1:(end - 1)]))
            foreach(t.conds, t.dests) do cond, dest
                push!(
                    new_blocks,
                    CFGBlock(ID(), [(ID(), new_inst(IDGotoIfNot(cond, dest), Any))]),
                )
            end
            push!(
                new_blocks,
                CFGBlock(ID(), [(ID(), new_inst(IDGotoNode(t.fallthrough_dest), Any))]),
            )
        else
            push!(new_blocks, block)
        end
    end
    return new_blocks
end

function _cfg_remove_double_edges(blocks::Vector{CFGBlock})::Vector{CFGBlock}
    return map(enumerate(blocks)) do (n, block)
        t = _cfg_terminator(block)
        if n < length(blocks) && t isa IDGotoIfNot && t.dest == blocks[n + 1].id
            term_id, term_inst = last(block.insts)
            new_insts = copy(block.insts)
            new_insts[end] = (term_id, NewInstruction(term_inst; stmt=IDGotoNode(t.dest)))
            return CFGBlock(block.id, new_insts)
        else
            return block
        end
    end
end

function _cfg_control_flow_graph(blocks::Vector{CFGBlock})::CC.CFG
    preds_ids = _compute_cfg_predecessors(blocks)
    succs_ids = _compute_cfg_successors(blocks)
    block_ids = map(block -> block.id, blocks)
    id_to_num = Dict{ID,Int}(zip(block_ids, eachindex(block_ids)))
    preds = map(id -> sort(map(p -> id_to_num[p], preds_ids[id])), block_ids)
    succs = map(id -> sort(map(s -> id_to_num[s], succs_ids[id])), block_ids)
    @static if VERSION >= v"1.11"
        push!(preds[1], 0)
    end
    index = vcat(0, cumsum(map(block -> length(block.insts), blocks))) .+ 1
    basic_blocks = map(eachindex(blocks)) do n
        stmt_range = CC.StmtRange(index[n], index[n + 1] - 1)
        return CC.BasicBlock(stmt_range, preds[n], succs[n])
    end
    return CC.CFG(basic_blocks, index[2:(end - 1)])
end

function _cfg_to_ssas(d::Dict, inst::NewInstruction)
    return NewInstruction(inst; stmt=_cfg_to_ssas(d, inst.stmt))
end
function _cfg_to_ssas(d::Dict, x::ReturnNode)
    isdefined(x, :val) ? ReturnNode(get(d, x.val, x.val)) : x
end
_cfg_to_ssas(d::Dict, x::Expr) = Expr(x.head, map(a -> get(d, a, a), x.args)...)
_cfg_to_ssas(d::Dict, x::PiNode) = PiNode(get(d, x.val, x.val), get(d, x.typ, x.typ))
_cfg_to_ssas(d::Dict, x::QuoteNode) = x
_cfg_to_ssas(d::Dict, x) = x
function _cfg_to_ssas(d::Dict, x::IDPhiNode)
    return PhiNode(
        map(edge -> Int32(getindex(d, edge).id), x.edges),
        _remap_assigned_phi_values(v -> get(d, v, v), x.values),
    )
end
_cfg_to_ssas(d::Dict, x::IDGotoNode) = GotoNode(d[x.label].id)
_cfg_to_ssas(d::Dict, x::IDGotoIfNot) = GotoIfNot(get(d, x.cond, x.cond), d[x.dest].id)

function _cfg_ids_to_line_numbers(blocks::Vector{CFGBlock})::InstVector
    block_ids = map(block -> block.id, blocks)
    block_lengths = map(block -> length(block.insts), blocks)
    block_start_ssas = SSAValue.(vcat(1, cumsum(block_lengths)[1:(end - 1)] .+ 1))
    lines = [inst for block in blocks for inst in block.insts]
    line_ids = first.(lines)
    line_ssas = SSAValue.(eachindex(line_ids))
    id_to_ssa_map = Dict(zip(vcat(block_ids, line_ids), vcat(block_start_ssas, line_ssas)))
    return [_cfg_to_ssas(id_to_ssa_map, inst) for (_, inst) in lines]
end

function _cfg_lines_to_blocks(insts::InstVector, cfg::CC.CFG)::InstVector
    stmts = __line_numbers_to_block_numbers!(Any[x.stmt for x in insts], cfg)
    return map((inst, stmt) -> NewInstruction(inst; stmt), insts, stmts)
end

#
# CFG line/block numbering and compiler-IR reconstruction
#

function _cfg_instruction_stream(ir::IRCode, insts::InstVector)
    @static if VERSION > v"1.12-"
        lines = CC.copy(ir.debuginfo.codelocs)
        n = length(insts)
        if length(lines) > 3n
            resize!(lines, 3n)
        elseif length(lines) < 3n
            for _ in (length(lines) + 1):3n
                push!(lines, 0)
            end
        end
        return CC.InstructionStream(
            Any[x.stmt for x in insts],
            Any[x.type for x in insts],
            CC.CallInfo[x.info for x in insts],
            lines,
            UInt32[x.flag for x in insts],
        )
    else
        return CC.InstructionStream(
            Any[x.stmt for x in insts],
            Any[x.type for x in insts],
            CC.CallInfo[x.info for x in insts],
            Int32[x.line for x in insts],
            UInt32[x.flag for x in insts],
        )
    end
end

function _rebuild_ircode(ir::IRCode, arg_types, cfg::CC.CFG, insts::InstVector)::IRCode
    inst_stream = _cfg_instruction_stream(ir, insts)
    @static if VERSION > v"1.12-"
        return IRCode(
            inst_stream,
            cfg,
            CC.copy(ir.debuginfo),
            Any[arg_types...],
            CC.copy(ir.meta),
            CC.copy(ir.sptypes),
            ir.valid_worlds,
        )
    else
        return IRCode(
            inst_stream,
            cfg,
            CC.copy(ir.linetable),
            Any[arg_types...],
            CC.copy(ir.meta),
            CC.copy(ir.sptypes),
        )
    end
end

"""
    lower_cfg_blocks_to_ir(ir::IRCode, arg_types, blocks::Vector{CFGBlock}; sort_cfg=true)

Lower reverse-mode-local CFG blocks directly to `IRCode`.
"""
#
# `CFGBlock` -> `IRCode` lowering
#

function lower_cfg_blocks_to_ir(
    ir::IRCode, arg_types, blocks::Vector{CFGBlock}; sort_cfg::Bool=true
)::IRCode
    blocks = _canonicalise_cfg_blocks(blocks; sort_cfg)
    blocks = _cfg_remove_double_edges(_cfg_lower_switch_statements(blocks))
    insts = _cfg_ids_to_line_numbers(blocks)
    cfg = _cfg_control_flow_graph(blocks)
    insts = _cfg_lines_to_blocks(insts, cfg)
    return _rebuild_ircode(ir, arg_types, cfg, insts)
end

"""
    pullback_ir(
        ir::IRCode,
        primal_blocks,
        Tret,
        ad_stmts_blocks::ADStmts,
        info::ADInfo,
        Tshared_data,
    )

Produce the IR associated to the `OpaqueClosure` which runs most of the pullback.
"""
#
# Pullback CFG assembly
#

function pullback_ir(
    ir::IRCode,
    primal_blocks,
    Tret,
    ad_stmts_blocks::ADStmts,
    block_comms,
    info::ADInfo,
    Tshared_data,
)
    # Compute the blocks which return in the primal.
    primal_exit_blocks_inds = findall(
        is_reachable_return_node ∘ _cfg_terminator, primal_blocks
    )

    #
    # Short-circuit for non-terminating primals -- applies to a tiny fraction of primals:
    #

    # If there are no blocks which successfully return in the primal, then the primal never
    # terminates without throwing, meaning that if AD hits this function, it definitely
    # won't succeed on the forwards-pass. As such, the reverse-pass can just be a no-op.
    if isempty(primal_exit_blocks_inds)
        blocks = [CFGBlock(ID(), [(ID(), new_inst(ReturnNode(nothing)))])]
        return lower_cfg_blocks_to_ir(ir, Any[Any], blocks)
    end

    #
    # Standard path pullback generation -- applies to 99% of primals:
    #

    # Compute the argument types associated to the reverse-pass.
    arg_types = vcat(Tshared_data, rdata_type(tangent_type(Tret)))

    # Create entry block which:
    # 1. extracts items from shared data to the correct IDs,
    # 2. creates `Ref`s (which will be optimised away later) to hold rdata for all ssas,
    # 3. create switch statement to block which terminated the forwards pass. If there is
    #   only a single block in the primal containing a reachable ReturnNode, then there is
    #   no need to pop the block stack.
    data_stmts = shared_data_stmts(info.shared_data_pairs)
    rev_data_ref_stmts = reverse_data_ref_stmts(info)
    exit_blocks_ids = map(n -> primal_blocks[n].id, primal_exit_blocks_inds)
    switch_stmts = make_switch_stmts(exit_blocks_ids, length(exit_blocks_ids) == 1, info)
    entry_block = CFGBlock(ID(), vcat(data_stmts, rev_data_ref_stmts, switch_stmts))

    # For each basic block in the primal:
    # 1. if the block is reachable on the reverse-pass, the bulk of its statements are the
    #   translated basic block statements, in reverse.
    # 2. if, on the other hand, the block is provably not reachable on the reverse-pass,
    #   return a block with nothing in it. At present we only assert that a block is not
    #   reachable if it ends with an unreachable `Core.ReturnNode`.
    # 3. if we need to pop the predecessor stack, pop it. We don't need to pop it if there
    #   is only a single predecessor to this block, and said predecessor is a _unique_
    #   _predecessor_ (see characterise_unique_predecessor_blocks for more info), as its
    #   ID is uniquely determined, and nothing will have been put on to the block stack
    #   during the forwards-pass (see how the output of
    #   characterise_unique_predecessor_blocks is used in forwards_pass_ir).
    # 4. if the block began with one or more PhiNodes, then handle their rdata.
    # 5. jump to the predecessor block.
    ps = _compute_cfg_predecessors(primal_blocks)
    _, pred_is_unique_pred = _characterise_unique_predecessor_blocks(primal_blocks)
    main_blocks = map(
        ad_stmts_blocks, enumerate(primal_blocks), block_comms
    ) do (blk_id, ad_stmts), (n, blk), comms

        # Short-circuit if we know that this block cannot be reached on the reverse-pass.
        if is_unreachable_return_node(_cfg_terminator(blk))
            return CFGBlock(blk_id, [(ID(), new_inst(nothing))])
        end

        # Extract reverse-stmts from ad_stmts.
        rvs_ad_stmts = reduce(vcat, [x.rvs for x in reverse(ad_stmts)])

        # Conclude the block.
        pred_ids = vcat(ps[blk.id], n == 1 ? [info.entry_id] : ID[])
        tmp = pred_is_unique_pred[blk_id]
        additional_stmts, new_blocks = conclude_rvs_block(blk, pred_ids, tmp, info)

        # Combine all blocks and return. See `create_comms_insts!` for more info regarding
        # `comms`.
        rvs_block = CFGBlock(blk_id, vcat(comms.rvs_prefix, rvs_ad_stmts, additional_stmts))
        return vcat(rvs_block, new_blocks)
    end
    main_blocks = reduce(vcat, main_blocks)

    # Create an exit block. Dereferences reverse-data for arguments, increments a zero rdata
    # against it to ensure that it is of the correct type, and returns it.
    arg_rdata_ref_ids = map(n -> info.arg_rdata_ref_ids[Argument(n)], 1:num_args(info))

    # De-reference the lazy zero rdata.
    lazy_zero_rdata_tuple_id = ID()
    lazy_zero_rdata_tuple = new_inst(Expr(:call, getindex, info.lazy_zero_rdata_ref_id))

    # For each argument, dereference its rdata, and increment said rdata against the zero
    # rdata element.
    final_ids = Vector{ID}()
    rdata_extraction_stmts = map(eachindex(arg_rdata_ref_ids)) do n

        # De-reference the nth rdata.
        rdata_id = ID()
        rdata = new_inst(Expr(:call, getfield, arg_rdata_ref_ids[n], QuoteNode(:x)))

        # Get the nth lazy zero rdata.
        lazy_zero_rdata_id = ID()
        lazy_zero_rdata = new_inst(Expr(:call, getfield, lazy_zero_rdata_tuple_id, n))

        # Instantiate the nth zero rdata.
        zero_rdata_id = ID()
        zero_rdata = new_inst(Expr(:call, instantiate, lazy_zero_rdata_id))

        # Increment the rdata.
        final_rdata_id = ID()
        final_rdata = new_inst(Expr(:call, increment!!, rdata_id, zero_rdata_id))

        # Log the ID of the rdata to return.
        push!(final_ids, final_rdata_id)

        return IDInstPair[
            (rdata_id, rdata),
            (lazy_zero_rdata_id, lazy_zero_rdata),
            (zero_rdata_id, zero_rdata),
            (final_rdata_id, final_rdata),
        ]
    end

    # Construct a tuple containing all of the rdata.
    deref_id = ID()
    deref = new_inst(Expr(:call, tuple, final_ids...))

    # Assert the type of the return value subtypes info.rvs_ret_type.
    assert_id = ID()
    assert = new_inst(Expr(:call, typeassert, deref_id, info.rvs_ret_type))

    # Construct return node and assemble final basic block.
    ret = new_inst(ReturnNode(assert_id))
    exit_block = CFGBlock(
        info.entry_id,
        vcat(
            (lazy_zero_rdata_tuple_id, lazy_zero_rdata_tuple),
            rdata_extraction_stmts...,
            [(deref_id, deref), (assert_id, assert), (ID(), ret)],
        ),
    )

    # Lower the pullback CFG directly to `IRCode`.
    return lower_cfg_blocks_to_ir(
        ir, arg_types, vcat([entry_block], main_blocks, [exit_block])
    )
end

"""
    conclude_rvs_block(
        blk::CFGBlock, pred_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo
    )

Generates code which is inserted at the end of each counterpart block in the reverse-pass.
Handles phi nodes, and choosing the correct next block to switch to.
"""
function conclude_rvs_block(
    blk::CFGBlock, pred_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo
)
    # Get the PhiNodes and their IDs.
    phi_ids, phis = _cfg_phi_nodes(blk)

    # If there are no PhiNodes in this block, switch directly to the predecessor.
    if length(phi_ids) == 0
        return make_switch_stmts(pred_ids, pred_is_unique_pred, info), CFGBlock[]
    end

    # Create statements which extract + zero the rdata refs associated to them.
    rdata_ids = map(_ -> ID(), phi_ids)
    tmp = map(phi_ids, rdata_ids) do phi_id, deref_id
        P = get_primal_type(info, phi_id)
        r = get_rev_data_id(info, phi_id)
        return deref_and_zero_stmts(P, r, deref_id)
    end
    deref_stmts = reduce(vcat, tmp; init=IDInstPair[])

    # For each predecessor, create a `CFGBlock` which processes its corresponding edge in
    # each of the `PhiNode`s.
    new_blocks = map(pred_ids) do pred_id
        values = Any[__get_value(pred_id, p.stmt) for p in phis]
        return rvs_phi_block(pred_id, rdata_ids, values, info)
    end
    new_pred_ids = map(blk -> blk.id, new_blocks)
    switch = make_switch_stmts(pred_ids, new_pred_ids, pred_is_unique_pred, info)
    return vcat(deref_stmts, switch), new_blocks
end

"""
    __get_value(edge::ID, x::IDPhiNode)

Helper functionality for conclude_rvs_block.
"""
function __get_value(edge::ID, x::IDPhiNode)
    edge in x.edges || return nothing
    n = only(findall(==(edge), x.edges))
    return isassigned(x.values, n) ? x.values[n] : nothing
end

"""
    deref_and_zero_stmts(P, ref_id, val_id)

Equivalent to something like
```julia
val = ref[]
ref[] = zero_rdata_from_type(P)
```
"""
function deref_and_zero_stmts(P, ref_id, val_id)
    val = (val_id, new_inst(Expr(:call, getfield, ref_id, QuoteNode(:x))))
    r = Mooncake.zero_like_rdata_from_type(P)
    set_ref = (ID(), new_inst(Expr(:call, setfield!, ref_id, QuoteNode(:x), r)))
    return IDInstPair[val, set_ref]
end

"""
    rvs_phi_block(pred_id::ID, rdata_ids::Vector{ID}, values::Vector{Any}, info::ADInfo)

Produces a `CFGBlock` which runs the reverse-pass for the edge associated to `pred_id` in a
collection of `IDPhiNode`s, and then goes to the block associated to `pred_id`.

For example, suppose that we encounter the following collection of `PhiNode`s at the start
of some block:
```julia
%6 = φ (#2 => _1, #3 => %5)
%7 = φ (#2 => 5., #3 => _2)
```
Let the rdata refs associated to `%6`, `%7`, and `_1`` be denoted `r%6`, `r%7`, and `r_1`
resp., and let `pred_id` be `#2`, and `increment_ref!` be the following function,
```julia
increment_ref!(ref, x) = ref[] = increment!!(ref[], x)
```
then this `rvs_phi_block` will produce a basic block of the form
```julia
increment_ref!(r_1, r%6)
nothing
goto #2
```
The call to `increment_ref!` appears because `_1` is the value associated to`%6` when the
primal code comes from `#2`. Similarly, the `goto #2` statement appears because we came from
`#2` on the forwards-pass. There is no `increment_ref!` associated to `%7` because `5.` is a
constant. We emit a `nothing` statement, which the compiler will happily optimise away later
on.

The same ideas apply if `pred_id` were `#3`. The block would end with `#3`, and there would
be two `increment_ref!` calls because both `%5` and `_2` are not constants.

In practice, code which is equivalent to `increment_ref!` is created directly, rather than
inserting a call to a generic Julia function. This is because we need to be certain that
the getfield and setfield! calls applied to any references are visible to the SROA
optimisation pass. If we insert a call to a function like `increment_ref!`, it might not be
inlined away, making such references opaque.
"""
function rvs_phi_block(
    pred_id::ID, rdata_ids::Vector{ID}, values::Vector{Any}, info::ADInfo
)
    @assert length(rdata_ids) == length(values)
    tmp = map(rdata_ids, values) do id, val
        rev_data_id = get_rev_data_id(info, val)
        rev_data_id === nothing && return nothing
        return increment_ref_stmts(rev_data_id, id)
    end
    inc_stmts = reduce(vcat, filter(x -> !(x === nothing), tmp); init=IDInstPair[])
    goto_stmt = (ID(), new_inst(IDGotoNode(pred_id)))
    return CFGBlock(ID(), vcat(inc_stmts, goto_stmt))
end

"""
    make_switch_stmts(
        pred_ids::Vector{ID}, target_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo
    )

`preds_ids` comprises the `ID`s associated to all possible predecessor blocks to the primal
block under consideration. Suppose its value is `[ID(1), ID(2), ID(3)]`, then
`make_switch_stmts` emits code along the lines of

```julia
prev_block = pop!(block_stack)
not_pred_was_1 = !(prev_block == ID(1))
not_pred_was_2 = !(prev_block == ID(2))
switch(
    not_pred_was_1 => ID(1),
    not_pred_was_2 => ID(2),
    ID(3)
)
```

In words: `make_switch_stmts` emits code which jumps to whichever block preceded the current
block during the forwards-pass.
"""
function make_switch_stmts(
    pred_ids::Vector{ID}, target_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo
)
    # If there are no predecessors, then we can't possibly have hit this block. This can
    # happen when all of the statements in a block have been eliminated, but the Julia
    # optimiser has not removed the block entirely from the `IRCode`. This often presents as
    # a block containing only a single `nothing` statement.
    # Consequently, we just direct this block back towards the entry node. This is safe, as
    # this block will never get hit, and ensures that the block is safe under re-ordering.
    isempty(pred_ids) && return [(ID(), new_inst(IDGotoNode(info.entry_id)))]

    # Get the predecessor that we actually had in the primal.
    prev_blk_id = ID()
    if pred_is_unique_pred
        prev_blk = new_inst(QuoteNode(only(pred_ids)))
    else
        prev_blk = new_inst(Expr(:call, __pop_blk_stack!, info.block_stack_id))
    end

    # Compare predecessor from primal with all possible predecessors.
    conds = map(pred_ids[1:(end - 1)]) do id
        return (ID(), new_inst(Expr(:call, __switch_case, id.id, prev_blk_id)))
    end

    # Switch statement to change to the predecessor.
    switch_stmt = Switch(Any[c[1] for c in conds], target_ids[1:(end - 1)], target_ids[end])
    switch = (ID(), new_inst(switch_stmt))

    return vcat((prev_blk_id, prev_blk), conds, switch)
end

function make_switch_stmts(pred_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo)
    return make_switch_stmts(pred_ids, pred_ids, pred_is_unique_pred, info)
end
"""
    __pop_blk_stack!(block_stack::BlockStack)

Equivalent to `pop!(block_stack)`. Going via this function, rather than just calling `pop!`
directly, makes it easy to figure out how much time is spent popping the block stack when
profiling performance, and to know that this function was hit when debugging.
"""
@inline __pop_blk_stack!(block_stack::BlockStack) = pop!(block_stack)

"""
    __switch_case(id::Int32, predecessor_id::Int32)

Helper function emitted by `make_switch_stmts`.
"""
__switch_case(id::Int32, predecessor_id::Int32) = !(id === predecessor_id)

"""
    rule_type(interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode) where {C}

Compute the concrete type of the rule that will be returned from `build_rrule`. This is
important for performance in dynamic dispatch, and to ensure that recursion works
properly.
"""
function rule_type(interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode) where {C}
    sig = _get_sig(sig_or_mi)
    if is_primitive(C, ReverseMode, sig, interp.world)
        # Build the rule to obtain its concrete type. For non-singleton primitive rules
        # (e.g. NfwdMooncake.RRule) this allocates a throwaway instance; the cost is compile-
        # time only and does not affect hot-path performance.
        rule = build_primitive_rrule(sig)
        return debug_mode ? DebugRRule{typeof(rule)} : typeof(rule)
    end

    ir, _ = lookup_ir(interp, sig_or_mi)
    Treturn = compute_ir_rettype(ir)
    isva, _ = is_vararg_and_sparam_names(sig_or_mi)

    arg_types = map(CC.widenconst, ir.argtypes)
    sig = Tuple{arg_types...}
    fwd_args_type = Tuple{map(fcodual_type, arg_types)...}
    fwd_return_type = forwards_ret_type(ir)
    Trdata_return = rdata_type(tangent_type(Treturn))
    # For non-returning primals, Tuple{} means a zero-argument pullback; Union{} would
    # instead mean there is no possible argument value.
    pb_args_type = Trdata_return === Union{} ? Tuple{} : Tuple{Trdata_return}
    pb_return_type = pullback_ret_type(ir)
    nargs = Val{length(ir.argtypes)}

    Tderived_rule = DerivedRule{
        sig,fwd_args_type,fwd_return_type,pb_args_type,pb_return_type,isva,nargs
    }
    return debug_mode ? DebugRRule{Tderived_rule} : Tderived_rule
end
