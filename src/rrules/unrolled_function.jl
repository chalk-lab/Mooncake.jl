using Base: RefValue

struct CoInstruction{Tinputs<:Tuple{Vararg{RefValue}}, Toutput<:RefValue, Tpb}
    inputs::Tinputs
    output::Toutput
    pb::Tpb
end

const_coinstruction(x::CoDual) = CoInstruction((), Ref(x), nothing)

input_primals(x::CoInstruction) = map(primal ∘ getindex, x.inputs)
input_shadows(x::CoInstruction) = map(shadow ∘ getindex, x.inputs)

output_primal(x::CoInstruction) = primal(x.output[])
output_shadow(x::CoInstruction) = shadow(x.output[])

function seed_output_shadow!(x::CoInstruction{T, V}, x̄) where {T, V}
    x.output[] = set_shadow!!(x.output[], x̄)
    return nothing
end

function optimised_rrule!!(args...)
    primals = map(primal, args)
    any(might_be_active ∘ typeof, primals) && return rrule!!(args...)
    y = primals[1](primals[2:end]...)
    return CoDual(y, uninit_tangent(y)), NoPullback()
end

function build_coinstruction(inputs::CoInstruction...)
    input_refs = map(x -> x.output, inputs)
    input_values = map(getindex, input_refs)
    output_value, pb!! = optimised_rrule!!(input_values...)
    output_ref = Ref(output_value)
    pb_ref = Ref(pb!!)
    return CoInstruction(input_refs, output_ref, pb_ref)
end

function (instruction::CoInstruction)(inputs::CoInstruction...)
    input_refs = map(x -> x.output, inputs)
    input_values = map(getindex, input_refs)
    foreach(verify_codual_type, input_values)
    output_value, pb!! = optimised_rrule!!(input_values...)
    verify_codual_type(output_value)
    output_ref = instruction.output
    output_ref[] = output_value
    pb_ref = instruction.pb
    pb_ref[] = pb!!
    return CoInstruction(input_refs, output_ref, pb_ref)
end

# pullback for "constant" CoInstruction.
pullback!(::CoInstruction{Tuple{}, <:Ref, Nothing}) = nothing

# pullback for general case CoInstruction.
function pullback!(instruction::CoInstruction)
    input_shadows = map(shadow ∘ getindex, instruction.inputs)
    output_shadow = shadow(instruction.output[])
    new_input_shadows = instruction.pb[](output_shadow, input_shadows...)
    foreach(replace_shadow!, instruction.inputs, new_input_shadows)
    return nothing
end

function replace_shadow!(x::Ref{<:CoDual{Tx, Tdx}}, new_shadow::Tdx) where {Tx, Tdx}
    x_val = x[]
    x[] = CoDual(primal(x_val), new_shadow)
    return nothing
end

function to_reverse_mode_ad(tape::Tape{RMC}, ȳ, inputs::CoInstruction...)
    inputs!(tape, inputs...)

    new_tape = Tape(tape.c)

    # Transform forwards pass, replacing ops with associated rrule calls.
    for op in tape.ops
        push!(new_tape, to_reverse_mode_ad(op, new_tape))
    end
    new_tape.result = unbind(tape.result)

    # Seed reverse-pass and create operations to execute it.
    seed_op = mkcall(seed_output_shadow!, new_tape.result, ȳ)
    push!(new_tape, seed_op)

    Umlaut.exec!(new_tape, seed_op)
    for op in reverse(new_tape.ops[1:end-1])
        pb_op = mkcall(pullback!, Variable(op.id))
        push!(new_tape, pb_op)
        Umlaut.exec!(new_tape, pb_op)
    end

    return new_tape
end

is_umlaut_type(x::Union{Variable, Constant, Input}) = true
is_umlaut_type(x) = false

to_reverse_mode_ad(x::Input, new_tape) = Input(x.val)
function to_reverse_mode_ad(x::Constant, new_tape)
    return Constant(const_coinstruction(CoDual(x.val, uninit_tangent(x.val))))
end
function to_reverse_mode_ad(x::Call, new_tape)
    f = x.fn isa Variable ? new_tape[x.fn].val : x.fn
    f = f isa CoInstruction ? f : const_coinstruction(CoDual(f, uninit_tangent(f)))
    raw_args = map(x -> x isa Variable ? new_tape[x].val : x, x.args)
    args = map(raw_args) do x
        x isa CoInstruction ? x : const_coinstruction(CoDual(x, uninit_tangent(x)))
    end
    v = build_coinstruction(f, args...)
    return mkcall(v, f, args...; val=v)
end

struct UnrolledFunction{Ttape}
    tape::Ttape
end


tangent_type(::Type{<:UnrolledFunction}) = NoTangent
randn_tangent(::AbstractRNG, ::UnrolledFunction) = NoTangent()
zero_tangent(::UnrolledFunction) = NoTangent()

(f::UnrolledFunction)(args...) = play!(f.tape, args...)

function seed_variable!(tape, var, ȳ)
    y_ref = tape[var].val.output
    dy = shadow(y_ref[])
    dy_new = increment!!(dy, ȳ)
    y_ref[] = CoDual(primal(y_ref[]), dy_new)
    return nothing
end

function rebinding_pass!(tape)
    new_tape = tape
    result = unbind(tape.result)
    num_ops = length(tape)
    rebind_ops = Any[]
    for (i, op) in enumerate(reverse(tape.ops))
        op_num = num_ops - i + 1
        if op isa Umlaut.Call
            f_args = [op.fn, op.args...]
            new_args = map(enumerate(op.args)) do (n, arg)
                !(arg isa Variable) && return arg
                if findfirst(Base.Fix1(===, arg), f_args[1:n]) === nothing
                    return arg
                else
                    new_op = mkcall(rebind, arg; val=tape[arg].val)
                    push!(rebind_ops, new_op)
                    return Variable(new_op)
                end
            end
            if !isempty(rebind_ops)
                push!(rebind_ops, mkcall(op.fn, new_args...; val=op.val))
                replace!(new_tape, op_num => rebind_ops)
                empty!(rebind_ops)
            end
        end
    end
    new_tape.result = result
    return new_tape
end

remake(op::Input) = Input(op.id, op.val, op.tape, op.line)
remake(op::Constant) = Constant(op.id, op.typ, op.val, op.tape, op.line)

function rrule!!(f::CoDual{<:UnrolledFunction}, args...)
    tape = rebinding_pass!(primal(f).tape)
    wrapped_args = map(const_coinstruction, args)
    inputs!(tape, wrapped_args...)

    new_tape = Tape(tape.c)
    # Transform forwards pass, replacing ops with associated rrule calls.
    for op in tape.ops
        new_op = to_reverse_mode_ad(op, new_tape)
        new_op_val = new_op.val.output[]
        if tangent_type(typeof(primal(new_op_val))) != typeof(shadow(new_op_val))
            inputs = map(getindex, new_op.val.inputs)
            display(inputs)
            println()
            display(new_op_val)
            println()
            display(which(rrule!!, map(Core.Typeof, inputs)))
            println()
            display("expected shadow type $(tangent_type(typeof(primal(new_op_val))))")
            println()
            throw(error("bad output types found in practice for op"))
        end
        push!(new_tape, new_op)
    end
    new_tape.result = unbind(tape.result)
    y_ref = new_tape[new_tape.result].val.output

    # Run the reverse-pass.
    function unrolled_function_pb!!(ȳ, ::NoTangent, dargs...)

        # Initialise values on the tape.
        seed_variable!(new_tape, new_tape.result, ȳ)
        foreach((v, x̄) -> seed_variable!(new_tape, v, x̄), inputs(new_tape), dargs)

        # Run the tape backwards.
        for op in reverse(new_tape.ops)
            pullback!(new_tape[Variable(op.id)].val)
        end

        # Extract the results from the tape.
        return NoTangent(), map(v -> shadow(new_tape[v].val.output[]), inputs(new_tape))...
    end

    return y_ref[], unrolled_function_pb!!
end

tangent_type(::Type{<:Umlaut.Variable}) = NoTangent

function value_and_gradient(tape::Tape, f, x...)
    f_ur = UnrolledFunction(tape)
    args = (f_ur, f, x...)
    dargs = map(zero_tangent, args)
    y, pb!! = rrule!!(map(CoDual, args, dargs)...)
    @assert primal(y) isa Float64
    dargs = pb!!(1.0, dargs...)
    return y, dargs
end

function value_and_gradient(f, x...)
    tape = last(trace(f, x...; ctx=RMC()))
    return value_and_gradient(tape, f, x...)
end

function construct_accel_tape(ȳ, f::CoDual, args::CoDual...)
    tape = primal(f).tape

    wrapped_args = map(const_coinstruction, args)
    inputs!(tape, wrapped_args...)

    new_tape = Tape(tape.c)
    # Transform forwards pass, replacing ops with associated rrule calls.
    for (n, op) in enumerate(tape.ops)
        new_op = to_reverse_mode_ad(op, new_tape)
        new_op_val = new_op.val.output[]
        if tangent_type(typeof(primal(new_op_val))) != typeof(shadow(new_op_val))
            inputs = map(getindex, new_op.val.inputs)
            display(inputs)
            println()
            display(new_op_val)
            println()
            display(which(rrule!!, map(Core.Typeof, inputs)))
            println()
            display("expected shadow type $(tangent_type(typeof(primal(new_op_val))))")
            println()
            throw(error("bad output types found in practice for op"))
        end
        push!(new_tape, new_op)
    end
    new_tape.result = unbind(tape.result)
    y_ref = new_tape[new_tape.result].val.output

    # Run the reverse-pass to ensure that we don't get state wrong.
    dargs = map(shadow, args)
    seed_variable!(new_tape, new_tape.result, ȳ)
    foreach((v, x̄) -> seed_variable!(new_tape, v, x̄), inputs(new_tape), dargs)

    # Push operations onto the tape to run the reverse-pass.
    for op in reverse(new_tape.ops)
        pb_op = mkcall(pullback!, Variable(op.id))
        push!(new_tape, pb_op)
        Umlaut.exec!(new_tape, pb_op)
    end

    # Accelerate the forwards-tape.
    return accelerate(new_tape)
end
