# Reverse-Mode Design

The purpose of this document is to explain how reverse-mode AD in Mooncake.jl is implemented.
It should do so to a sufficient level of depth to enable the interested reader to read the reverse-mode AD code in Mooncake.jl and understand what is going on.

This document
1. specifies the semantics of a "rule" for reverse-mode AD,
1. specifies how to implement rules by-hand for primitives,
1. specifies how to derive rules from `IRCode` algorithmically in general,
1. discusses the compilation process from IR to executable pullbacks,
1. discusses key data structures and their roles in the compilation pipeline,
1. explains the forward and reverse pass generation,
1. discusses some notable technical differences from forwards-mode AD implementation details, and
1. concludes with implementation notes and debugging guidance.

## Reverse-Rule Interface

Loosely, a rule for reverse-mode AD simultaneously
1. performs the same computation as the original function (the "forward pass"), and
1. returns a pullback function that can compute VJPs (Vector-Jacobian Products).

This is best explained through a worked example.
Consider a function call
```julia
z = f(x, y)
```
where `f` itself may contain data / state which is modified by executing `f`.
`rule_for_f` is _some_ callable which claims to be a reverse-rule for `f`.
For `rule_for_f` to be a valid reverse-rule for `f`, it must be applicable to `CoDual`s as follows:
```julia
z_codual, pb!! = rule_for_f(CoDual(f, df), CoDual(x, dx), CoDual(y, dy))
```
where:
1. `rule_for_f` is a callable. It might be written by-hand, or derived algorithmically.
1. `df`, `dx`, and `dy` are reverse-data (rdata) for `f`, `x`, and `y` respectively. These will accumulate gradient information during the reverse pass.
1. `z_codual` is a `CoDual` containing the primal result `z` and its associated reverse-data.
1. `pb!!` is the pullback function that, when called with a cotangent for `z`, will accumulate gradients into the reverse-data of the inputs.
1. running `rule_for_f` leaves `f`, `x`, and `y` in the same state that running `f` does.

The pullback `pb!!` can then be called as:
```julia
pb!!(dz)
```
where `dz` is a cotangent (gradient) with respect to `z`. This will accumulate gradients into `df`, `dx`, and `dy`.

We refer readers to [Algorithmic Differentiation](@ref) to explain what we mean when we talk about the "derivative" above.

Note that `rule_for_f` is an as-yet-unspecified callable which we introduced purely to specify the interface that a reverse-rule must satisfy.
In [Hand-Written Reverse Rules](#hand-written-reverse-rules) and [Derived Reverse Rules](#derived-reverse-rules) below, we introduce two concrete ways to produce rules for `f`.

### FData and RData Types

Reverse-mode AD in Mooncake.jl uses a sophisticated type system to manage gradient information:

- **FData (Forward Data)**: Represents the "tangent" or "differential" of a value during the forward pass
- **RData (Reverse Data)**: Represents the accumulated gradient information during the reverse pass

The key functions for working with these types are:
- `fdata_type(T)`: Returns the FData type for primal type `T`
- `rdata_type(T)`: Returns the RData type for primal type `T`
- `zero_fdata(x)`: Creates zero FData for value `x`
- `zero_rdata(x)`: Creates zero RData for value `x`

### Testing

Suppose that we have (somehow) produced a supposed reverse-rule. To check that it is correctly implemented, we must ensure that
1. all primal state after running the rule is approximately the same as all primal state after running the primal, and
2. the gradient computed by the reverse pass matches the gradient computed by finite differences or other AD tools.

We already have the functionality to do this in a very general way (see [`Mooncake.TestUtils.test_rule`](@ref)).

## Hand-Written Reverse Rules

Hand-written rules are implemented by writing methods of two functions: `is_primitive` and `rrule!!`.

### `is_primitive`

`is_primitive(::Type{<:Union{MinimalCtx, DefaultCtx}}, signature::Type{<:Tuple})` should return `true` if AD must attempt to differentiate a call by passing the arguments to `rrule!!`, and `false` otherwise.
The [`Mooncake.@is_primitive`](@ref) macro helps makes implementing this very easy.

### `rrule!!`

Methods of `rrule!!` do the actual differentiation, and must satisfy the [Reverse-Rule Interface](#reverse-rule-interface) discussed above.

In what follows, we will refer to `rrule!!`s for signatures.
For example, the `rrule!!` for signature `Tuple{typeof(sin), Float64}` is the rule which would differentiate calls like `sin(5.0)`.

#### Simple Scalar Function

For the function ``y = \sin(x)``, the reverse-mode rule computes the forward pass and returns a pullback:
```julia
function rrule!!(::CoDual{typeof(sin)}, x::CoDual{Float64})
    x_primal = primal(x)
    y_primal = sin(x_primal)
    
    # Return primal result and pullback function
    y_codual = CoDual(y_primal, zero_rdata(y_primal))
    
    function sin_pullback(dy)
        dx = cos(x_primal) * dy
        increment_rdata!(rdata(x), dx)
        return NoRData()
    end
    
    return y_codual, sin_pullback
end
```

#### Matrix-Matrix Multiply

For ``Z = X Y``, the reverse-mode rule is more complex due to the need to handle matrix operations:
```julia
function rrule!!(
    ::CoDual{typeof(LinearAlgebra.mul!)}, 
    Z::CoDual{P}, 
    X::CoDual{P}, 
    Y::CoDual{P}
) where {P<:Matrix{Float64}}
    
    # Forward pass
    Z_primal = primal(Z)
    X_primal = primal(X)
    Y_primal = primal(Y)
    mul!(Z_primal, X_primal, Y_primal)
    
    function mul_pullback(dZ)
        # For Z = XY, we have:
        # dX = dZ * Y^T
        # dY = X^T * dZ
        increment_rdata!(rdata(X), dZ * Y_primal')
        increment_rdata!(rdata(Y), X_primal' * dZ)
        return NoRData()
    end
    
    return Z, mul_pullback
end
```

## Derived Reverse Rules

This is the "automatic" / "algorithmic" bit of AD!
This is the second way of producing concrete callable objects which satisfy the [Reverse-Rule Interface](#reverse-rule-interface) discussed above.
The object which we will ultimately construct is an instance of `Mooncake.DerivedRule`.

### Compilation Pipeline Overview

The compilation process for reverse-mode AD involves several key steps:

1. **IR Lookup**: Obtain the optimized `IRCode` for the target function
2. **Normalization**: Apply standardizing transformations to the IR
3. **BBCode Generation**: Convert to Basic Block Code representation for easier manipulation
4. **AD Statement Translation**: Transform each statement into forward and reverse operations
5. **Communication Channel Setup**: Establish data flow between forward and reverse passes
6. **Shared Data Management**: Handle data that needs to be preserved between passes
7. **IR Generation**: Create separate IR for forward and reverse passes
8. **Optimization**: Apply Julia compiler optimizations
9. **MistyClosure Creation**: Package the optimized IR into executable closures

Let's examine each step in detail.

### IR Lookup and Normalization

The process begins with `lookup_ir(interp, sig_or_mi)`, which retrieves the optimized `IRCode` for a given function signature or method instance. This IR is then normalized using `normalise!(ir, spnames)` to:

- Convert `:foreigncall` expressions to `:call`s to `Mooncake._foreigncall_`
- Convert `:new` expressions to `:call`s to `Mooncake._new_`
- Convert `:splatnew` expressions to `:call`s to `Mooncake._splat_new_`
- Replace `Core.IntrinsicFunction`s with wrappers from `Mooncake.IntrinsicWrappers`
- Lift getfield operations and memory operations for better AD handling

### BBCode: Basic Block Code Representation

The normalized IR is converted to `BBCode` (Basic Block Code), a representation that makes it easier to manipulate control flow during AD transformation. Key features:

- Each basic block has a unique `ID`
- Instructions are represented as `IDInstPair`s (ID and instruction pairs)
- Control flow is explicitly tracked through predecessors and successors
- Unreachable blocks are automatically removed

### AD Statement Translation

The core of the compilation process is the `make_ad_stmts!` function, which transforms each statement in the primal IR into:

1. **Forward Pass Instructions**: Operations needed during the forward pass
2. **Reverse Pass Instructions**: Operations needed during the pullback
3. **Communication Data**: Information that needs to flow from forward to reverse pass

Each `ADStmtInfo` contains:
- `line::ID`: The original statement's ID
- `comms_id::Union{ID,Nothing}`: ID for data shared between passes
- `fwds::Vector{IDInstPair}`: Forward pass instructions
- `rvs::Vector{IDInstPair}`: Reverse pass instructions

### Key Data Structures

#### ADInfo

The `ADInfo` structure serves as the central coordinator for the compilation process:

```julia
struct ADInfo
    interp::MooncakeInterpreter
    block_stack_id::ID              # Stack tracking visited blocks
    block_stack::BlockStack         # The actual block stack
    entry_id::ID                    # Entry block ID
    shared_data_pairs::SharedDataPairs  # Data shared between passes
    arg_types::Dict{Argument,Type}   # Argument type information
    # ... additional fields for tracking IDs and references
end
```

#### SharedDataPairs

Manages data that needs to be available in both forward and reverse passes:

- Stores `(ID, data)` pairs in a vector
- Generates statements to extract data from captured tuples
- Handles the lifetime of shared objects between passes

#### BlockStack

A stack data structure that tracks which basic blocks are visited during the forward pass, enabling the reverse pass to visit them in the correct (reverse) order.

### Forward and Reverse Pass Generation

#### Forward Pass IR Generation

The forward pass IR (`forwards_pass_ir`) contains:
- All original computations
- Instructions to save necessary intermediate values
- Block stack management (pushing visited blocks)
- Communication channel setup (saving data for reverse pass)

#### Reverse Pass IR Generation  

The reverse pass IR (`pullback_ir`) contains:
- Block stack management (popping blocks in reverse order)
- Communication channel extraction (retrieving saved data)
- Gradient propagation through all operations
- RData accumulation into argument gradients

### Communication Channels

One of the most sophisticated aspects of Mooncake's reverse-mode compilation is the communication system between forward and reverse passes. The system works as follows:

1. **Identification**: During statement translation, identify what data the reverse pass needs
2. **Storage**: In the forward pass, save this data in block-specific stacks
3. **Retrieval**: In the reverse pass, extract data from stacks in LIFO order
4. **Optimization**: Group communications per block to minimize overhead

### Memory Management and Optimization

The compilation process includes several optimization strategies:

- **Shared Data Deduplication**: Identical data is stored only once
- **Singleton Handling**: Singleton types are inlined directly rather than stored
- **Block-Level Optimization**: Communications are batched per basic block
- **Standard Compiler Optimizations**: The generated IR undergoes full Julia optimization

### Example: Derived Rule Generation

Consider a simple function:
```julia
function example_func(x, y)
    z = x * y
    w = sin(z)
    return w
end
```

The compilation process would generate:

**Forward Pass**:
```julia
function forward_pass(shared_data, f, x, y)
    # Push entry block
    push!(block_stack, 1)
    
    # Original computation
    z = x * y
    w = sin(z)
    
    # Save data for reverse pass
    save_for_reverse!(shared_data, z)  # sin needs input for derivative
    
    return w
end
```

**Reverse Pass**:
```julia
function reverse_pass(shared_data, dw)
    # Pop blocks in reverse order
    block_id = pop!(block_stack)
    
    # Retrieve saved data
    z = retrieve_saved_data!(shared_data)
    
    # Propagate gradients
    dz = cos(z) * dw  # ∂w/∂z = cos(z)
    dx = y * dz       # ∂z/∂x = y  
    dy = x * dz       # ∂z/∂y = x
    
    # Accumulate into argument gradients
    increment_rdata!(rdata(x), dx)
    increment_rdata!(rdata(y), dy)
    
    return NoRData()
end
```

## Technical Differences from Forward-Mode

The reverse-mode implementation has several key differences from forward-mode:

1. **Two-Pass Structure**: Forward-mode is single-pass, reverse-mode requires separate forward and reverse passes
2. **Memory Management**: Reverse-mode must save intermediate values during forward pass for use in reverse pass
3. **Control Flow Tracking**: Block stack mechanism to ensure correct reverse traversal of control flow
4. **Data Communication**: Sophisticated system for passing data between forward and reverse passes
5. **RData System**: Additional type system for managing reverse-mode gradient accumulation
6. **Optimization Complexity**: More complex optimization due to the two-pass nature

## Implementation Notes

### Debugging and Development

When working with reverse-mode compilation:

1. **Use Debug Mode**: Set `debug_mode=true` in `build_rrule` for detailed error information
2. **IR Inspection**: Use `primal_ir`, `fwds_ir`, and `adjoint_ir` functions to examine generated IR
3. **Test Incrementally**: Use `Mooncake.TestUtils.test_rule` to verify correctness
4. **Check Communications**: Ensure data saved in forward pass matches what reverse pass expects

### Performance Considerations

- **Memory Usage**: Reverse-mode can use significant memory for saving intermediate values
- **Compilation Time**: Complex functions may have long compilation times due to IR transformation complexity
- **Runtime Overhead**: Block stack and communication overhead, though optimized, still has cost

### Common Pitfalls

1. **Missing Communications**: Forgetting to save necessary data in forward pass
2. **Type Instability**: Incorrect type annotations in generated IR
3. **Control Flow Errors**: Incorrect block stack management for complex control flow
4. **RData Mismanagement**: Incorrect handling of reverse-data accumulation

## Conclusion

Mooncake's reverse-mode AD compilation is a sophisticated system that transforms Julia IR into efficient forward and reverse passes. The key innovations include:

- **BBCode representation** for easier IR manipulation
- **Communication channels** for efficient data flow between passes  
- **Block stack mechanism** for correct control flow reversal
- **Comprehensive type system** for gradient management
- **Extensive optimization** for performance

Understanding these components is essential for contributing to Mooncake's reverse-mode implementation or debugging complex differentiation scenarios.