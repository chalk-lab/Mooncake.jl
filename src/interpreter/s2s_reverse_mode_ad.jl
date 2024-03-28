#=
    SharedDataPairs()

A data structure used to manage the captured data in the `OpaqueClosures` which implement
the bulk of the forwards- and reverse-passes of AD. An entry `(id, data)` at element `n`
of the `pairs` field of this data structure means that `data` will be available at register
`id` during the forwards- and reverse-passes of `AD`.

This is achieved by storing all of the data in the `pairs` field in the captured tuple which
is passed to an `OpaqueClosure`, and extracting this data into registers associated to the
corresponding `ID`s.
=#
struct SharedDataPairs
    pairs::Vector{Tuple{ID, Any}}
    SharedDataPairs() = new(Tuple{ID, Any}[])
end

#=
    add_data!(p::SharedDataPairs, data)::ID

Puts `data` into `p`, and returns the `id` associated to it. This `id` should be assumed to
be available during the forwards- and reverse-passes of AD, and it should further be assumed
that the value associated to this `id` is always `data`.
=#
function add_data!(p::SharedDataPairs, data)::ID
    id = ID()
    push!(p.pairs, (id, data))
    return id
end

#=
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
=#
shared_data_tuple(p::SharedDataPairs)::Tuple = tuple(map(last, p.pairs)...)

#=
    shared_data_stmts(p::SharedDataPairs)::Vector{Tuple{ID, NewInstruction}}

Produce a sequence of id-statment pairs which will extract the data from
`shared_data_tuple(p)` such that the correct value is associated to the correct `ID`.

For example, if `p.pairs` is
```julia
[(ID(5), 5.0), (ID(3), "hello")]
```
then the output of this function is
```julia
Tuple{ID, NewInstruction}[
    (ID(5), new_inst(:(getfield(_1, 1)))),
    (ID(3), new_inst(:(getfield(_1, 2)))),
]
```
=#
function shared_data_stmts(p::SharedDataPairs)::Vector{Tuple{ID, NewInstruction}}
    return map(enumerate(p.pairs)) do (n, p)
        return (p[1], new_inst(Expr(:call, getfield, Argument(1), n)))
    end
end

#=
    ADInfo

This data structure is used to hold "global" information associated to a particular call to
`build_rrule`. It is used as a means of communication between `make_ad_stmts!` and the
codegen which produces the forwards- and reverse-passes.

- `interp`: a `TapirInterpreter`.
- `block_stack_id`: the ID associated to the block stack -- the stack which keeps track of
    which blocks we visited during the forwards-pass, and which is used on the reverse-pass
    to determine which blocks to visit.
- `block_stack`: the block stack. Can always be found at `block_stack_id` in the forwards-
    and reverse-passes.
- `entry_id`: special ID associated to the block inserted at the start of execution in the
    the forwards-pass, and the end of execution in the pullback.
- `shared_data_pairs`: the `SharedDataPairs` used to define the captured variables passed
    to both the forwards- and reverse-passes..
- `arg_types`: a map from `Argument` to its static type.
- `ssa_insts`: a map from `ID` associated to lines to the primal `NewInstruction`.
- `arg_tangent_stacks`: a map from primal `Argument`s to their tangent stacks. If the stack
    associated to an `Argument` is a bits type then this will just be the tangent stack.
    Otherwise, it will be the `ID` associated to the stack, and the stack itself will be put
    in the `shared_data_pairs`.
- `tangent_stacks`: a map from `ID` to tangent stacks. If the tangent stack associated to
    the `ID` is a bits type, then this will actually be the tangent stack. Otherwise it will
    be the `ID` associated to the stack, and the stack itself will be put in the
    `shared_data_pairs`.
=#
struct ADInfo
    interp::PInterp
    block_stack_id::ID
    block_stack::Stack{Int32}
    entry_id::ID
    shared_data_pairs::SharedDataPairs
    arg_types::Dict{Argument, Any}
    ssa_insts::Dict{ID, NewInstruction}
    arg_tangent_stacks::Dict{Argument, Any}
    tangent_stacks::Dict{ID, Any}
end

# The constructor that you should use for ADInfo.
function ADInfo(
    interp::PInterp,
    arg_types::Dict{Argument, Any},
    ssa_insts::Dict{ID, NewInstruction},
    arg_tangent_stacks,
)
    shared_data_pairs = SharedDataPairs()
    block_stack = Stack{Int32}()
    return ADInfo(
        interp,
        add_data!(shared_data_pairs, block_stack),
        block_stack,
        ID(),
        shared_data_pairs,
        arg_types,
        ssa_insts,
        make_arg_tangent_stacks!(shared_data_pairs, arg_tangent_stacks),
        make_tangent_stacks!(shared_data_pairs, ssa_insts),
    )
end

function __log_data(p::Union{ADInfo, SharedDataPairs}, x)
    return Base.issingletontype(_typeof(x)) ? x : add_data!(p, x)
end

# Construct a map from primal `Argument`s to the location of its tangent stack in the
# forwards-pass and pullback. If tangent stack is a singleton, just yields the tangent
# stack itself.
function make_arg_tangent_stacks!(p::SharedDataPairs, arg_tangent_stacks)
    arguments = Argument.(eachindex(arg_tangent_stacks))
    stack_ids = map(Base.Fix1(__log_data, p), arg_tangent_stacks)
    return Dict{Argument, Any}(zip(arguments, stack_ids))
end

# Construct a map from primal `ID`s corresponding to lines in the IR, to the location of
# their tangent stacks in the forwards-pass and pullback. If tangent stacks is a singleton,
# just yields the tangent stack itself.
function make_tangent_stacks!(p::SharedDataPairs, ssa_insts::Dict{ID, NewInstruction})
    tangent_stacks = Dict{ID, Any}()
    for (k, inst) in ssa_insts
        Meta.isexpr(inst.stmt, :call) || Meta.isexpr(inst.stmt, :invoke) || continue
        tangent_stacks[k] = __log_data(p, make_tangent_stack(_get_type(inst.type)))
    end
    return tangent_stacks
end

# Shortcut for `add_data!(info.shared_data_pairs, data)`.
add_data!(info::ADInfo, data) = add_data!(info.shared_data_pairs, data)

# Returns the static / inferred type associated to `x`.
get_primal_type(info::ADInfo, x::Argument) = info.arg_types[x]
get_primal_type(info::ADInfo, x::ID) = _get_type(info.ssa_insts[x].type)
get_primal_type(::ADInfo, x::QuoteNode) = _typeof(x.value)
get_primal_type(::ADInfo, x) = _typeof(x)
function get_primal_type(::ADInfo, x::GlobalRef)
    return isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.binding.ty
end

#=
    ADStmtInfo

Data structure which contains the result of `make_ad_stmts!`. Fields are
- `line`: the ID associated to the primal line from which this is derived
- `fwds`: the instructions which run the forwards-pass of AD
- `rvs`: the instructions which run the reverse-pass of AD / the pullback
=#
struct ADStmtInfo
    line::ID
    fwds::Vector{Tuple{ID, NewInstruction}}
    rvs::Vector{Tuple{ID, NewInstruction}}
end

# Convenient constructor for `ADStmtInfo`. If either `fwds` or `rvs` is not a vector,
# `__vec` promotes it to a single-element `Vector`.
ad_stmt_info(line::ID, fwds, rvs) = ADStmtInfo(line, __vec(line, fwds), __vec(line, rvs))

__vec(line::ID, x::Any) = __vec(line, new_inst(x))
__vec(line::ID, x::NewInstruction) = Tuple{ID, NewInstruction}[(line, x)]
__vec(line::ID, x::Vector{Tuple{ID, Any}}) = throw(error("boooo"))
__vec(line::ID, x::Vector{Tuple{ID, NewInstruction}}) = x

#=
    make_ad_stmts(inst::NewInstruction, line::ID, info::ADInfo)::ADStmtInfo

Every line in the primal code is associated to one or more lines in the forwards-pass of AD,
and one or more lines in the pullback. This function has method specific to every
node type in the Julia SSAIR.

Translates the instruction `inst`, associated to `line` in the primal, into a specification
of what should happen for this instruction in the forwards- and reverse-passes of AD, and
what data should be shared between the forwards- and reverse-passes. Returns this in the
form of an `ADStmtInfo`.

`info` is a data structure containing various bits of global information that certain types
of nodes need access to.
=#
function make_ad_stmts! end

# `nothing` as a statement in Julia IR indicates the presence of a line which will later be
# removed. We emit a no-op on both the forwards- and reverse-passes. No shared data.
make_ad_stmts!(::Nothing, line::ID, ::ADInfo) = ad_stmt_info(line, nothing, nothing)

# `ReturnNode`s have a single field, `val`, for which there are three cases to consider:
#
# 1. `val isa Union{Argument, ID}`: this is an active bit of data. Consequently, we know
#   that it will be an `AugmentedRegister` already, and can just return it. Therefore `stmt`
#   is returned as the forwards-pass (with any `Argument`s incremented), and nothing happens
#   in the pullback.
# 2. `val` is undefined: this `ReturnNode` is unreachable. Consequently, we'll never hit the
#   associated statements on the forwards-pass of pullback. We just return the original
#   statement on the forwards-pass, and `nothing` on the reverse-pass.
# 3. `val` is defined, but not a `Union{Argument, ID}`: in this case we're returning a
#   constant -- build a constant register and return that.
function make_ad_stmts!(stmt::ReturnNode, line::ID, info::ADInfo)
    if !isdefined(stmt, :val) || is_active(stmt.val)
        return ad_stmt_info(line, inc_args(stmt), nothing)
    else
        return ad_stmt_info(line, ReturnNode(const_register(stmt.val, info)), nothing)
    end
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDGotoNode, line::ID, ::ADInfo)
    return ad_stmt_info(line, inc_args(stmt), nothing)
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDGotoIfNot, line::ID, ::ADInfo)
    stmt = inc_args(stmt)
    if stmt.cond isa Union{Argument, ID}
        # If cond refers to a register, then the primal must be extracted.
        cond_id = ID()
        fwds = [
            (cond_id, new_inst(Expr(:call, primal, stmt.cond))),
            (line, new_inst(IDGotoIfNot(cond_id, stmt.dest), Any)),
        ]
        return ad_stmt_info(line, fwds, nothing)
    else
        # If something other than a register, then there is nothing to do.
        return ad_stmt_info(line, stmt, nothing)
    end
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDPhiNode, line::ID, info::ADInfo)
    vals = stmt.values
    new_vals = Vector{Any}(undef, length(vals))
    for n in eachindex(vals)
        isassigned(vals, n) || continue
        new_vals[n] = is_active(vals[n]) ? __inc(vals[n]) : const_register(vals[n], info)
    end

    # It turns out to be really very important to do type inference correctly for PhiNodes.
    # For some reason, type inference really doesn't like it when you encounter mutually-
    # dependent PhiNodes whose types are unknown and for which you set the flag to
    # CC.IR_FLAG_REFINED.
    new_type = register_type(get_primal_type(info, line))
    _inst = new_inst(IDPhiNode(stmt.edges, new_vals), new_type, info.ssa_insts[line].flag)
    return ad_stmt_info(line, _inst, nothing)
end

function make_ad_stmts!(stmt::PiNode, line::ID, info::ADInfo)
    isa(stmt.val, Union{Argument, ID}) || unhandled_feature("PiNode: $stmt")

    # Create line which sharpens the register type as much as possible.
    sharp_primal_type = _get_type(stmt.typ)
    sharpened_register_type = AugmentedRegister{codual_type(_get_type(sharp_primal_type))}
    new_pi_line = ID()
    new_pi = PiNode(__inc(stmt.val), sharpened_register_type)

    # Create a statement which moves data from the loosely-typed register to a more
    # strictly typed one, which is possible because of the `PiNode`.
    tangent_stack = make_tangent_stack(sharp_primal_type)
    tangent_stack_id = add_data!(info, tangent_stack)
    val_type = get_primal_type(info, stmt.val)
    tangent_ref_stack = make_tangent_ref_stack(tangent_ref_type_ub(val_type))
    tangent_ref_stack_id = add_data!(info, tangent_ref_stack)
    new_line = Expr(:call, __pi_fwds!, tangent_stack_id, tangent_ref_stack_id, new_pi_line)

    # Assemble the above lines and construct reverse-pass.
    return ad_stmt_info(
        line,
        [(new_pi_line, new_inst(new_pi)), (line, new_inst(new_line))],
        Expr(:call, __pi_rvs!, tangent_stack_id, tangent_ref_stack_id),
    )
end

@inline function __pi_fwds!(tangent_stack, tangent_ref_stack, reg::AugmentedRegister)
    push!(tangent_ref_stack, reg.tangent_ref)
    push!(tangent_stack, tangent(reg.codual))
    return AugmentedRegister(reg.codual, top_ref(tangent_stack))
end

@inline function __pi_rvs!(tangent_stack, tangent_ref_stack)
    increment_ref!(pop!(tangent_ref_stack), pop!(tangent_stack))
    return nothing
end

# Constant GlobalRefs are handled. See const_register. Non-constant
# GlobalRefs are handled by assuming that they are constant, and creating a register with
# the value. We then check at run-time that the value has not changed.
function make_ad_stmts!(stmt::GlobalRef, line::ID, info::ADInfo)
    if isconst(stmt)
        return const_ad_stmt(stmt, line, info)
    else
        reg = const_register(getglobal(stmt.mod, stmt.name), info)
        gref_id = ID()
        fwds = [
            (gref_id, new_inst(stmt)),
            (line, new_inst(Expr(:call, __verify_const, gref_id, reg))),
        ]
        return ad_stmt_info(line, fwds, nothing)
    end
end

# Helper used by `make_ad_stmts! ` for `GlobalRef`.
@noinline function __verify_const(global_ref, stored_value)
    @assert global_ref == primal(stored_value)
    return stored_value
end

# QuoteNodes are constant. See make_const_register for details.
make_ad_stmts!(stmt::QuoteNode, line::ID, info::ADInfo) = const_ad_stmt(stmt, line, info)

# Literal constant. See const_register for details.
make_ad_stmts!(stmt, line::ID, info::ADInfo) = const_ad_stmt(stmt, line, info)

# `make_ad_stmts!` for constants.
function const_ad_stmt(stmt, line::ID, info::ADInfo)
    reg = const_register(stmt, info)
    return ad_stmt_info(line, reg isa ID ? Expr(:call, identity, reg) : reg, nothing)
end

# Build an `AugmentedRegister` from `stmt`, which will be checked to ensure that its value
# is constant. If the resulting register is a bits type, then it is returned. If it is not,
# then the register is put into shared data, and the ID associated to it in the forwards-
# and reverse-passes returned.
function const_register(stmt, info::ADInfo)
    reg = build_const_reg(stmt)
    return isbitstype(_typeof(reg)) ? reg : add_data!(info, reg)
end

# Create a constant augmented register which lives in the shared data. Returns the `ID`
# which will be associated to this data in the forwards- and reverse-passes.
shared_data_const_reg(stmt, info::ADInfo) = add_data!(info, build_const_reg(stmt))

# Create an `AugmentedRegister` containing the values associated to `stmt`, a zero tangent.
# Pushes a single element onto the stack, and puts a reference to that stack in the
# register.
function build_const_reg(stmt)
    primal_value = get_const_primal_value(stmt)
    tangent_stack = make_tangent_stack(_typeof(primal_value))
    tangent = uninit_tangent(primal_value)
    push!(tangent_stack, tangent)
    return AugmentedRegister(CoDual(primal_value, tangent), top_ref(tangent_stack))
end

# Get the value associated to `x`. For `GlobalRef`s, verify that `x` is indeed a constant,
# and error if it is not.
function get_const_primal_value(x::GlobalRef)
    isconst(x) || unhandled_feature("Non-constant GlobalRef not supported: $x")
    return getglobal(x.mod, x.name)
end
get_const_primal_value(x::QuoteNode) = x.value
get_const_primal_value(x) = x

# Tapir does not yet handle `PhiCNode`s. Throw an error if one is encountered.
function make_ad_stmts!(stmt::Core.PhiCNode, ::ID, ::ADInfo)
    unhandled_feature("Encountered PhiCNode: $stmt")
end

# Tapir does not yet handle `UpsilonNode`s. Throw an error if one is encountered.
function make_ad_stmts!(stmt::Core.UpsilonNode, ::ID, ::ADInfo)
    unhandled_feature("Encountered UpsilonNode: $stmt")
end

# There are quite a number of possible `Expr`s that can be encountered. Each case has its
# own comment, explaining what is going on.
function make_ad_stmts!(stmt::Expr, line::ID, info::ADInfo)
    is_invoke = Meta.isexpr(stmt, :invoke)
    if Meta.isexpr(stmt, :call) || is_invoke

        # Find the types of all arguments to this call / invoke.
        args = ((is_invoke ? stmt.args[2:end] : stmt.args)..., )
        arg_types = map(arg -> get_primal_type(info, arg), args)

        # Construct signature, and determine how the rrule is to be computed.
        sig = Tuple{arg_types...}
        rule = if is_primitive(context_type(info.interp), sig)
            rrule!! # intrinsic / builtin / thing we provably have rule for
        elseif is_invoke
            LazyDerivedRule(info.interp, sig) # Static dispatch
        else
            DynamicDerivedRule(info.interp)  # Dynamic dispatch
        end

        # If the rule is `rrule!!` (i.e. `sig` is primitive), then don't bother putting
        # the rule into shared data, because it's safe to put it directly into the code.
        rule_ref = __log_data(info, rule)

        # Tangent stacks are allocated in build_rrule, and stored in the `info`. Just
        # retrieve the stack associated to the tangent returned from this line.
        ret_tangent_stack_id = info.tangent_stacks[line]

        # If the type of the pullback is a singleton type, then there is no need to store it
        # in the shared data, it can be interpolated directly into the generated IR.
        pb_stack = build_pb_stack(_typeof(rule), arg_types)
        pb_stack_id = __log_data(info, pb_stack)

        # if the pullback is a `NoPullback`, then there is no need to log the references to
        # the tangent stacks associated to the inputs to this call, because there will never
        # need to be any incrementing done. There are functions called within
        # `__fwds_pass!` and `__rvs_pass!` that specialise on the type of the pullback to
        # avoid ever using the arg tangent ref stacks, so we just need to create a default
        # value here (`nothing`), as it will never be used.
        arg_tangent_ref_stacks_id = ID()
        if pb_stack isa SingletonStack{NoPullback}
            arg_tangent_ref_stacks = nothing
        else
            ref_stacks = map(arg_types, args) do arg_type, arg
                stack = __make_arg_tangent_ref_stack(arg_type, arg)
                if Base.issingletontype(_typeof(stack))
                    return stack
                elseif haskey(info.tangent_stacks, arg)
                    return info.tangent_stacks[arg]
                elseif arg isa Argument
                    return info.arg_tangent_stacks[arg]
                else
                    return add_data!(info, stack)
                end
            end
            arg_tangent_ref_stacks = Expr(:call, __tangent_ref_stacks, ref_stacks...)
        end

        # Create calls to `__fwds_pass!` and `__rvs_pass!`, which run the forwards pass and
        # pullback associated to a call / invoke.
        fwds_pass_call = Expr(
            :call,
            __fwds_pass!,
            arg_tangent_ref_stacks_id,
            rule_ref,
            ret_tangent_stack_id,
            pb_stack_id,
            register_type(get_primal_type(info, line)),
            map(__inc, args)...,
        )

        rvs_pass_call = Expr(
            :call, __rvs_pass!, arg_tangent_ref_stacks_id, ret_tangent_stack_id, pb_stack_id
        )

        fwds = [
            (arg_tangent_ref_stacks_id, new_inst(arg_tangent_ref_stacks)),
            (line, new_inst(fwds_pass_call)),
        ]
        rvs = Tuple{ID, NewInstruction}[
            (arg_tangent_ref_stacks_id, new_inst(arg_tangent_ref_stacks)),
            (line, new_inst(rvs_pass_call)),
        ]
        return ad_stmt_info(line, fwds, rvs)

    elseif Meta.isexpr(stmt, :boundscheck)
        # For some reason the compiler cannot handle boundscheck statements when we run it
        # again. Consequently, emit `true` to be safe. Ideally we would handle this in a
        # more natural way, but I'm not sure how to do that.
        tmp = AugmentedRegister(zero_codual(true), NoTangentStack())
        return ad_stmt_info(line, tmp, nothing)

    elseif Meta.isexpr(stmt, :code_coverage_effect)
        # Code coverage irrelevant for derived code.
        return ad_stmt_info(line, nothing, nothing)

    elseif Meta.isexpr(stmt, :copyast)
        # Get constant out and shove it in shared storage.
        reg = const_register(stmt.args[1], info)
        return ad_stmt_info(line, Expr(:call, identity, reg), nothing)

    elseif Meta.isexpr(stmt, :loopinfo)
        # Cannot pass loopinfo back through the optimiser for some reason.
        # At the time of writing, I am unclear why this is not possible.
        return ad_stmt_info(line, nothing, nothing)

    elseif stmt.head in [
        :enter,
        :gc_preserve_begin,
        :gc_preserve_end,
        :leave,
        :pop_exception,
        :throw_undef_if_not,
    ]
        # Expressions which do not require any special treatment.
        return ad_stmt_info(line, stmt, nothing)
    else
        # Encountered an expression that we've not seen before.
        throw(error("Unrecognised expression $stmt"))
    end
end

# Used in `make_ad_stmts!` for call and invoke exprs. If an argument to the stmt is active,
# then we grab its tangnet ref stack. If it's inactive (a constant of some kind -- really
# anything that's not an `Argument` or an `ID`), then we create a dummy stack that will get
# optimised away.
function __make_arg_tangent_ref_stack(arg_type, arg)
    is_active(arg) || return InactiveStack(InactiveRef(__zero_tangent(arg)))
    return make_tangent_ref_stack(tangent_ref_type_ub(arg_type))
end

@inline function __tangent_ref_stacks(args::Vararg{Any, N}) where {N}
    return tuple_map(___tangent_ref_stacks_helper, args)
end

# Distinguish between tangent stacks and tangent ref stacks based on their type. If we see
# a type which looks like a tangent ref stack, just return it. If we see any other type,
# assume it is a tangent stack, meaning that the tangent stack is fixed.
# This is bit of a hack -- ideally we would get the code construction in `make_ad_stmts!` to
# determine this, as doing this based on type is potentially flakey. It will have to do for
# now though.
@inline @generated function ___tangent_ref_stacks_helper(arg::P) where {P}
    if P <: Union{InactiveStack, Stack{<:Ref}, NoTangentRefStack} && !(P <: Stack{<:Ptr})
        return :(arg)
    else
        return :(FixedStackTangentRefStack(arg))
    end
end

is_active(::Union{Argument, ID}) = true
is_active(::Any) = false

__zero_tangent(arg) = zero_tangent(arg)
__zero_tangent(arg::GlobalRef) = zero_tangent(getglobal(arg.mod, arg.name))
__zero_tangent(arg::QuoteNode) = zero_tangent(arg.value)

# Build a stack to contain the pullback. Specialises on whether the pullback is a singleton,
# and whether we get to know the concrete type of the pullback or not.
function build_pb_stack(Trule, arg_types)
    T_pb!! = Core.Compiler.return_type(Tuple{Trule, map(codual_type, arg_types)...})
    if T_pb!! <: Tuple && T_pb!! !== Union{} && !(T_pb!! isa Union)
        F = T_pb!!.parameters[2]
        return Base.issingletontype(F) ? SingletonStack{F}() : Stack{F}()
    else
        return Stack{Any}()
    end
end

# Used in `make_ad_stmts!` method for `Expr(:call, ...)` and `Expr(:invoke, ...)`.
@inline function __fwds_pass!(
    arg_tangent_ref_stacks,
    rule,
    ret_tangent_stack,
    pb_stack,
    ::Type{R},
    f::F,
    raw_args::Vararg{Any, N},
) where {R, F, N}

    raw_args = (f, raw_args...)
    __log_tangent_refs!(pb_stack, raw_args, arg_tangent_ref_stacks)

    # Run the rule.
    args = tuple_map(x -> isa(x, AugmentedRegister) ? x.codual : uninit_codual(x), raw_args)
    out, pb!! = rule(args...)

    # Log the results and return.
    __push_tangent_stack!(ret_tangent_stack, tangent(out))
    __push_pb_stack!(pb_stack, pb!!)
    return AugmentedRegister(out, top_ref(ret_tangent_stack))::R
end

@inline function __log_tangent_refs!(::Any, raw_args, arg_tangent_ref_stacks)
    tangent_refs = map(x -> isa(x, AugmentedRegister) ? x.tangent_ref : nothing, raw_args)
    tuple_map(__push_ref_stack, arg_tangent_ref_stacks, tangent_refs)
end

@inline __log_tangent_refs!(::SingletonStack{NoPullback}, ::Any, ::Any) = nothing

@inline __push_ref_stack(tangent_ref_stack, ref) = push!(tangent_ref_stack, ref)
@inline __push_ref_stack(::InactiveStack, ref) = nothing
@inline __push_ref_stack(::NoTangentRefStack, ref) = nothing

@inline __push_tangent_stack!(stack, t) = push!(stack, t)
@inline __push_pb_stack!(stack, pb!!) = push!(stack, pb!!)

# Used in `make_ad_stmts!` method for `Expr(:call, ...)` and `Expr(:invoke, ...)`.
@inline function __rvs_pass!(arg_tangent_ref_stacks, ret_tangent_stack, pb_stack)::Nothing
    pb = __pop_pb_stack!(pb_stack)
    tngt = __pop_tangent_stack!(ret_tangent_stack)
    __execute_reverse_pass!(pb, tngt, arg_tangent_ref_stacks)
end

@inline __execute_reverse_pass!(::NoPullback, ::Any, ::Any) = nothing

@inline function __execute_reverse_pass!(pb!!, dout, arg_tangent_ref_stacks)
    # Get the tangent w.r.t. each argument of the primal.
    tangent_refs = tuple_map(pop!, arg_tangent_ref_stacks)

    # Run the pullback and increment the argument tangents.
    dargs = tuple_map(set_immutable_to_zero ∘ getindex, tangent_refs)
    new_dargs = pb!!(dout, dargs...)
    tuple_map(increment_ref!, tangent_refs, new_dargs)
    return nothing
end

@inline __pop_pb_stack!(stack) = pop!(stack)
@inline __pop_tangent_stack!(tangent_stack) = pop!(tangent_stack)

#
# Runners for generated code.
#

struct Pullback{Tpb, Tret_ref, Targ_tangent_stacks, Tisva, Tnargs}
    pb_oc::Tpb
    ret_ref::Tret_ref
    arg_tangent_stacks::Targ_tangent_stacks
    isva::Tisva
    nargs::Tnargs
end

@inline function (pb::Pullback{P, Q})(dy, dargs::Vararg{Any, N}) where {P, Q, N}
    unflattened_dargs = __unflatten_varargs(pb.isva, dargs, pb.nargs)
    map(setindex!, map(top_ref, pb.arg_tangent_stacks), unflattened_dargs)
    increment_ref!(pb.ret_ref, dy)
    pb.pb_oc(dy, unflattened_dargs...)
    out = __flatten_varargs(pb.isva, map(pop!, pb.arg_tangent_stacks), nvargs(length(dargs), pb.nargs))
    return out::_typeof(dargs)
end

@inline nvargs(n_flat, ::Val{nargs}) where {nargs} = Val(n_flat - nargs + 1)

struct DerivedRule{Tfwds_oc, Targ_tangent_stacks, Tpb_oc, Tisva<:Val, Tnargs<:Val}
    fwds_oc::Tfwds_oc
    pb_oc::Tpb_oc
    arg_tangent_stacks::Targ_tangent_stacks
    block_stack::Stack{Int32}
    isva::Tisva
    nargs::Tnargs
end

@inline function (fwds::DerivedRule{P, Q, S})(args::Vararg{CoDual, N}) where {P, Q, S, N}

    # Load arguments in to stacks, and create tuples.
    args = __unflatten_codual_varargs(fwds.isva, args, fwds.nargs)
    args_with_tangent_stacks = map(args, fwds.arg_tangent_stacks) do arg, arg_tangent_stack
        push!(arg_tangent_stack, tangent(arg))
        return AugmentedRegister(arg, top_ref(arg_tangent_stack))
    end

    # Run forwards-pass.
    reg = fwds.fwds_oc(args_with_tangent_stacks...)::AugmentedRegister

    # Extract result and assemble pullback.
    pb!! = Pullback(fwds.pb_oc, reg.tangent_ref, fwds.arg_tangent_stacks, fwds.isva, fwds.nargs)
    return reg.codual, pb!!
end

# Compute the concrete type of the rule that will be returned from `build_rrule`. This is
# important for performance in dynamic dispatch, and to ensure that recursion works
# properly.
function rule_type(interp::TapirInterpreter{C}, ::Type{sig}) where {C, sig}
    is_primitive(C, sig) && return typeof(rrule!!)

    ir, _ = lookup_ir(interp, sig)
    Treturn = Base.Experimental.compute_ir_rettype(ir)
    isva, _ = is_vararg_sig_and_sparam_names(sig)

    arg_types = map(_get_type, ir.argtypes)
    arg_tangent_types = map(tangent_type, arg_types)
    Targ_registers = Tuple{map(tangent_stack_type ∘ _get_type, ir.argtypes)...}
    Treturn_register = register_type(Treturn)
    if isconcretetype(Treturn_register)
        return DerivedRule{
            Core.OpaqueClosure{Tuple{map(register_type, arg_types)...}, Treturn_register},
            Targ_registers,
            Core.OpaqueClosure{Tuple{tangent_type(Treturn), arg_tangent_types...}, Nothing},
            Val{isva},
            Val{length(ir.argtypes)},
        }
    else
        return DerivedRule{
            Core.OpaqueClosure{Tuple{map(register_type, arg_types)...}, T} where {T<:Treturn_register},
            Targ_registers,
            Core.OpaqueClosure{Tuple{tangent_type(Treturn), arg_tangent_types...}, Nothing},
            Val{isva},
            Val{length(ir.argtypes)},
        }
    end
end

# if isva and nargs=2, then inputs (5.0, 4.0, 3.0) are transformed into (5.0, (4.0, 3.0)).
function __unflatten_varargs(::Val{isva}, args, ::Val{nargs}) where {isva, nargs}
    isva || return args
    if all(t -> t isa NoTangent, args[nargs:end])
        return (args[1:nargs-1]..., NoTangent())
    else
        return (args[1:nargs-1]..., args[nargs:end])
    end
end

# If isva, inputs (5.0, (4.0, 3.0)) are transformed into (5.0, 4.0, 3.0).
function __flatten_varargs(::Val{isva}, args, ::Val{nvargs}) where {isva, nvargs}
    isva || return args
    if args[end] isa NoTangent
        return (args[1:end-1]..., ntuple(n -> NoTangent(), nvargs)...)
    else
        return (args[1:end-1]..., args[end]...)
    end
end

# If isva and nargs=2, then inputs `(CoDual(5.0, 0.0), CoDual(4.0, 0.0), CoDual(3.0, 0.0))`
# are transformed into `(CoDual(5.0, 0.0), CoDual((5.0, 4.0), (0.0, 0.0)))`.
function __unflatten_codual_varargs(::Val{isva}, args, ::Val{nargs}) where {isva, nargs}
    isva || return args
    group_primal = map(primal, args[nargs:end])
    if tangent_type(_typeof(group_primal)) == NoTangent
        grouped_args = zero_codual(group_primal)
    else
        grouped_args = CoDual(group_primal, map(tangent, args[nargs:end]))
    end
    return (args[1:nargs-1]..., grouped_args)
end

"""
    build_rrule(args...)

Helper method. Only uses static information from `args`.
"""
function build_rrule(args...)
    return build_rrule(TapirInterpreter(), _typeof(TestUtils.__get_primals(args)))
end

"""
    build_rrule(interp::PInterp{C}, sig::Type{<:Tuple}) where {C}

Returns a `DerivedRule` which is an `rrule!!` for `sig` in context `C`. See the docstring
for `rrule!!` for more info.
"""
function build_rrule(interp::PInterp{C}, sig::Type{<:Tuple}) where {C}

    # Reset id count. This ensures that everything in this function is deterministic.
    seed_id!()

    # If we have a hand-coded rule, just use that.
    is_primitive(C, sig) && return rrule!!

    # Grab code associated to the primal.
    ir, _ = lookup_ir(interp, sig)
    Treturn = Base.Experimental.compute_ir_rettype(ir)

    # Normalise the IR, and generated BBCode version of it.
    isva, spnames = is_vararg_sig_and_sparam_names(sig)
    ir = normalise!(ir, spnames)
    primal_ir = BBCode(ir)

    # Compute global info.
    arg_types = Dict{Argument, Any}(
        map(((n, t),) -> (Argument(n) => _get_type(t)), enumerate(ir.argtypes))
    )
    insts = new_inst_vec(ir.stmts)
    ssa_types = Dict{ID, NewInstruction}(zip(concatenate_ids(primal_ir), insts))
    arg_tangent_stacks = (map(make_tangent_stack ∘ _get_type, primal_ir.argtypes)..., )
    info = ADInfo(interp, arg_types, ssa_types, arg_tangent_stacks)

    # For each block in the fwds and pullback BBCode, translate all statements.
    ad_stmts_blocks = map(primal_ir.blocks) do primal_blk
        ids = primal_blk.inst_ids
        primal_stmts = map(x -> x.stmt, primal_blk.insts)
        return (primal_blk.id, make_ad_stmts!.(primal_stmts, ids, Ref(info)))
    end

    # Make shared data, and construct BBCode for forwards-pass and pullback.
    shared_data = shared_data_tuple(info.shared_data_pairs)
    # display(sig)
    # @show length(shared_data)
    # @show length(ir.stmts.inst)
    # display(collect(_typeof(shared_data).parameters))

    # If we've already derived the OpaqueClosures and info, do not re-derive, just create a
    # copy and pass in new shared data.
    if !haskey(interp.oc_cache, sig)
        fwds_ir = forwards_pass_ir(primal_ir, ad_stmts_blocks, info, _typeof(shared_data))
        pb_ir = pullback_ir(primal_ir, Treturn, ad_stmts_blocks, info, _typeof(shared_data))
        optimised_fwds_ir = optimise_ir!(IRCode(fwds_ir); do_inline=true)
        optimised_pb_ir = optimise_ir!(IRCode(pb_ir); do_inline=true)
        # @show length(optimised_fwds_ir.stmts.inst)
        # @show length(optimised_pb_ir.stmts.inst)
        # display(ir)
        # display(optimised_fwds_ir)
        # display(optimised_pb_ir)
        fwds_oc = OpaqueClosure(optimised_fwds_ir, shared_data...; do_compile=true)
        pb_oc = OpaqueClosure(optimised_pb_ir, shared_data...; do_compile=true)
        interp.oc_cache[sig] = (fwds_oc, pb_oc)
    else
        existing_fwds_oc, existing_pb_oc = interp.oc_cache[sig]
        fwds_oc = replace_captures(existing_fwds_oc, shared_data)
        pb_oc = replace_captures(existing_pb_oc, shared_data)
    end

    return rule_type(interp, sig)(
        fwds_oc,
        pb_oc,
        arg_tangent_stacks,
        info.block_stack,
        Val(isva),
        Val(length(ir.argtypes)),
    )
end

# Given an `OpaqueClosure` `oc`, create a new `OpaqueClosure` of the same type, but with new
# captured variables. This is needed for efficiency reasons -- if `build_rrule` is called
# repeatedly with the same signature and intepreter, it is important to avoid recompiling
# the `OpaqueClosure`s that it produces multiple times, because it can be quite expensive to
# do so.
@eval function replace_captures(oc::Toc, new_captures) where {Toc<:Core.OpaqueClosure}
    return $(Expr(
        :new, :(Toc), :new_captures, :(oc.world), :(oc.source), :(oc.invoke), :(oc.specptr)
    ))
end

const ADStmts = Vector{Tuple{ID, Vector{ADStmtInfo}}}

#=
    forwards_pass_ir(ir::BBCode, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

Produce the IR associated to the `OpaqueClosure` which runs most of the forwards-pass.
=#
function forwards_pass_ir(ir::BBCode, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

    # Insert a block at the start which extracts all items from the captures field of the
    # `OpaqueClosure`, which contains all of the data shared between the forwards- and
    # reverse-passes. These are assigned to the `ID`s given by the `SharedDataPairs`.
    # Additionally, push the entry id onto the block stack.
    push_block_stack_stmt = Expr(:call, push!, info.block_stack_id, info.entry_id.id)
    entry_stmts = vcat(
        shared_data_stmts(info.shared_data_pairs),
        (ID(), new_inst(push_block_stack_stmt)),
    )
    entry_block = BBlock(info.entry_id, entry_stmts)

    # Construct augmented version of each basic block from the primal. For each block:
    # 1. pull the translated basic block statements from ad_stmts_blocks.
    # 2. insert a statement which logs the ID of the current block to the block stack.
    # 3. construct and return a BBlock.
    blocks = map(ad_stmts_blocks) do (block_id, ad_stmts)
        fwds_stmts = reduce(vcat, map(x -> x.fwds, ad_stmts))
        ins_loc = length(fwds_stmts) + (isa(fwds_stmts[end][2].stmt, Terminator) ? 0 : 1)
        ins_stmt = Expr(:call, __push_blk_stack!, info.block_stack_id, block_id.id)
        ins_inst = (ID(), new_inst(ins_stmt))
        return BBlock(block_id, insert!(fwds_stmts, ins_loc, ins_inst))
    end

    # Create and return the `BBCode` for the forwards-pass.
    arg_types = vcat(Tshared_data, map(register_type ∘ _get_type, ir.argtypes))
    return BBCode(vcat(entry_block, blocks), arg_types, ir.sptypes, ir.linetable, ir.meta)
end

@noinline __push_blk_stack!(block_stack::Stack{Int32}, id::Int32) = push!(block_stack, id)

#=
    pullback_ir(ir::BBCode, Tret, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

Produce the IR associated to the `OpaqueClosure` which runs most of the pullback.
=#
function pullback_ir(ir::BBCode, Tret, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

    # Compute the argument types associated to the reverse-pass.
    darg_types = map(tangent_type ∘ _get_type, ir.argtypes)
    arg_types = vcat(Tshared_data, tangent_type(Tret), darg_types)

    # Compute the blocks which return in the primal.
    primal_exit_blocks_inds = findall(is_reachable_return_node ∘ terminator, ir.blocks)

    #
    # Short-circuit for non-terminating primals -- applies to a tiny fraction of primals:
    #

    # If there are no blocks which successfully return in the primal, then the primal never
    # terminates without throwing, meaning that if AD hits this function, it definitely
    # won't succeed on the forwards-pass. As such, the reverse-pass can just be a no-op.
    if isempty(primal_exit_blocks_inds)
        blocks = [BBlock(ID(), [(ID(), new_inst(ReturnNode(nothing)))])]
        return BBCode(blocks, arg_types, ir.sptypes, ir.linetable, ir.meta)
    end

    #
    # Standard path pullback generation -- applied to 99% of primals:
    #

    # Create entry block, which pops the block_stack, and switches to whichever block we
    # were in at the end of the forwards-pass.
    exit_blocks_ids = map(n -> ir.blocks[n].id, primal_exit_blocks_inds)
    data_stmts = shared_data_stmts(info.shared_data_pairs)
    switch_stmts = make_switch_stmts(exit_blocks_ids, info)
    entry_block = BBlock(ID(), vcat(data_stmts, switch_stmts))

    # For each basic block in the primal:
    # 1. pull the translated basic block statements from ad_stmts_blocks
    # 2. reverse the statements
    # 3. pop block stack to get the predecessor block
    # 4. insert a switch statement to determine which block to jump to. Restrict blocks
    #   considered to only those which are predecessors of this one. If in the first block,
    #   check whether or not the block stack is empty. If empty, jump to the exit block.
    ps = compute_all_predecessors(ir)
    main_blocks = map(ad_stmts_blocks, enumerate(ir.blocks)) do (blk_id, ad_stmts), (n, blk)
        rvs_stmts = reduce(vcat, [x.rvs for x in reverse(ad_stmts)])
        pred_ids = vcat(ps[blk.id], n == 1 ? [info.entry_id] : ID[])
        switch_stmts = make_switch_stmts(pred_ids, info)
        return BBlock(blk_id, vcat(rvs_stmts, switch_stmts))
    end

    # Create an exit block. Simply returns nothing.
    exit_block = BBlock(info.entry_id, [(ID(), new_inst(ReturnNode(nothing)))])

    # Create and return `BBCode` for the pullback.
    blks = vcat(entry_block, main_blocks, exit_block)
    return _sort_blocks!(BBCode(blks, arg_types, ir.sptypes, ir.linetable, ir.meta))
end

#=
    make_switch_stmts(pred_ids::Vector{ID}, info::ADInfo)

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
=#
function make_switch_stmts(pred_ids::Vector{ID}, info::ADInfo)

    # If there are no predecessors, then we can't possible have hit this block. This can
    # happen when all of the statements in a block have been eliminated, but the Julia
    # optimiser has not removed the block entirely from the `IRCode`. This often presents as
    # a block containing only a single `nothing` statement.
    # Consequently, we just direct this block back towards the entry node. This is safe, as
    # this block will never get hit, and ensures that the block is safe under re-ordering.
    isempty(pred_ids) && return [(ID(), new_inst(IDGotoNode(info.entry_id)))]

    # Get the predecessor that we actually had in the primal.
    prev_blk_id = ID()
    prev_blk = new_inst(Expr(:call, __pop_blk_stack!, info.block_stack_id))

    # Compare predecessor from primal with all possible predecessors.
    conds = map(pred_ids[1:end-1]) do id
        return (ID(), new_inst(Expr(:call, __switch_case, id.id, prev_blk_id)))
    end

    # Switch statement to change to the predecessor.
    switch_stmt = Switch(Any[c[1] for c in conds], pred_ids[1:end-1], pred_ids[end])
    switch = (ID(), new_inst(switch_stmt))

    return vcat((prev_blk_id, prev_blk), conds, switch)
end

@noinline __pop_blk_stack!(block_stack::Stack{Int32}) = pop!(block_stack)

# Helper function emitted by `make_switch_stmts`.
__switch_case(id::Int32, predecessor_id::Int32) = !(id === predecessor_id)


#=
    DynamicDerivedRule(interp::TapirInterpreter)

For internal use only.

A callable data structure which, when invoked, calls an rrule specific to the dynamic types
of its arguments. Stores rules in an internal cache to avoid re-deriving.

This is used to implement dynamic dispatch.
=#
struct DynamicDerivedRule{T, V}
    interp::T
    cache::V
end

DynamicDerivedRule(interp::TapirInterpreter) = DynamicDerivedRule(interp, Dict{Any, Any}())

function (dynamic_rule::DynamicDerivedRule)(args::Vararg{Any, N}) where {N}
    sig = Tuple{map(_typeof, map(primal, args))...}
    is_primitive(context_type(dynamic_rule.interp), sig) && return rrule!!(args...)
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        rule = build_rrule(dynamic_rule.interp, sig)
        dynamic_rule.cache[sig] = rule
    end
    return rule(args...)
end

#=
    LazyDerivedRule(interp, sig)

For internal use only.

A type-stable wrapper around a `DerivedRule`, which only instantiates the `DerivedRule`
when it is first called. This is useful, as it means that if a rule does not get run, it
does not have to be derived.
=#
mutable struct LazyDerivedRule{Trule, T, V}
    interp::T
    sig::V
    rule::Trule
    function LazyDerivedRule(interp::T, sig::V) where {T<:PInterp, V<:Type{<:Tuple}}
        return new{rule_type(interp, sig), T, V}(interp, sig)
    end
end

function (rule::LazyDerivedRule)(args::Vararg{Any, N}) where {N}
    if !isdefined(rule, :rule)
        rule.rule = build_rrule(rule.interp, rule.sig)
    end
    return rule.rule(args...)
end
