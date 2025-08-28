# GitHub Copilot Instructions for Mooncake.jl

This document provides context and guidelines for GitHub Copilot when working with the Mooncake.jl codebase.

## Project Overview

Mooncake.jl is a Julia automatic differentiation (AD) library that provides both forward-mode and reverse-mode differentiation. It aims to improve upon existing Julia AD packages like ForwardDiff.jl, ReverseDiff.jl, and Zygote.jl while being competitive with Enzyme.jl. The library is written entirely in Julia and focuses on performance and correctness.

### Key Features
- Forward-mode and reverse-mode automatic differentiation
- Integration with DifferentiationInterface.jl 
- Extensive rule system for built-in functions
- Support for complex data structures and custom types
- Performance optimizations using Julia's type system

## Code Organization

### Main Directories
- `src/`: Core library implementation
  - `Mooncake.jl`: Main module file
  - `interpreter/`: IR transformation and interpretation
  - `rrules/`: Reverse-mode differentiation rules for various functions
  - `interface.jl`: Public API definitions
  - `tools_for_rules.jl`: Utilities for defining differentiation rules
- `test/`: Comprehensive test suite organized by functionality
- `docs/`: Documentation including developer guides
- `ext/`: Package extensions for optional dependencies

### Key Files
- `src/interface.jl`: Primary user-facing API (`value_and_gradient!!`, etc.)
- `src/tangents.jl`: Tangent space types and operations
- `src/codual.jl`: Coupled primal-tangent values
- `src/dual.jl`: Forward-mode dual numbers
- `src/config.jl`: Configuration options

## Coding Standards

### Julia Style
- Follow the Blue style guidelines (configured in `.JuliaFormatter.toml`)
- Code can be formatted automatically by running `julia -e 'using JuliaFormatter; format(".")'` from the root of the repo
- Use meaningful variable names, especially for mathematical concepts
- Prefer explicit type annotations for public APIs
- Use `@stable` and `@unstable` annotations from DispatchDoctor

### Naming Conventions
- Use snake_case for variable names and functions
- Use CamelCase for types and modules
- Prefix internal functions with underscore: `_internal_function`
- Use `!!` suffix for in-place/mutating operations: `increment!!`
- Use descriptive names for mathematical concepts (e.g., `tangent`, `primal`, `pullback`)

### Performance Considerations
- Leverage Julia's type system for performance
- Use `@inferred` in tests to ensure type stability
- Minimize allocations in hot paths
- Use `@opaque` for closures when appropriate

## Core Concepts and Patterns

### Tangent Types
```julia
# Pattern for tangent space operations
tangent_type(::Type{T}) where T = # implementation
zero_tangent(x::T) where T = # zero element in tangent space
```

### Rule Definitions
```julia
# Reverse-mode rule pattern
function rrule!!(::CoDual{T}, args::CoDual...) where T
    # Forward pass computation
    y = # computation
    
    # Return value and pullback function
    function pullback!!(dy)
        # Reverse pass computation
        return # tangents for inputs
    end
    
    return CoDual(y, NoFData()), pullback!!
end

# Forward-mode rule pattern  
function frule!!(::CoDual{T}, args::CoDual...) where T
    # Combined forward and tangent computation
    return CoDual(y, dy)  # primal and tangent
end
```

### Testing Patterns
```julia
# Use test_utils.jl utilities for comprehensive testing
@testset "function_name" begin
    test_rule(rng, rule_type, function_call, args...)
    test_data(rng, test_input)  # For tangent type testing
end
```

## Domain-Specific Knowledge

### Automatic Differentiation Concepts
- **Primal**: The original computation/value
- **Tangent**: Directional derivatives in tangent space
- **Pullback**: Reverse-mode computation that propagates gradients backwards
- **CoDual**: Coupled primal-tangent value representation
- **Rule**: Function that defines how to differentiate a specific operation

### Mathematical Operations
- Differentiation rules follow standard mathematical conventions
- Chain rule application is fundamental to the design
- Handle special cases (NaN, Inf) appropriately
- Consider numerical stability in implementations

### Memory Management
- Use `increment!!` for in-place tangent accumulation
- Minimize allocations in reverse-mode pullbacks
- Handle mutable data structures correctly
- Zero tangents when necessary to avoid accumulation errors

## Error Handling

### Common Error Types
- `ArgumentError`: For invalid function signatures or arguments
- Custom exceptions like `ValueAndGradientReturnTypeError`
- Use descriptive error messages that help users understand the issue

### Debugging Support
- Use `Config().debug_mode` for additional runtime checks
- Provide helpful error messages with context
- Support for debugging tools in `developer_tools.jl`

## Testing Guidelines

### Test Organization
- Tests are organized by functionality in separate files
- Use descriptive test names that explain what's being tested
- Group related tests in `@testset` blocks

### Test Patterns
```julia
# Basic correctness testing
@test result ≈ expected_result

# Type stability testing
@test @inferred function_call(args...)

# Error condition testing
@test_throws ErrorType function_call(invalid_args...)
```

### Integration with CI
- Tests run in multiple groups for parallelization
- Quality checks include formatting and static analysis
- Use appropriate floating-point tolerances for comparisons

## Extension Patterns

### Package Extensions
- Located in `ext/` directory for optional dependencies
- Follow naming convention: `MooncakePackageNameExt`
- Conditionally load based on package availability

### Custom Rules
- Use `@mooncake_overlay` for method overrides visible only to Mooncake
- Implement both forward and reverse rules when possible
- Test thoroughly with `test_rule` utilities

## Performance Considerations

### Type Stability
- Ensure all hot-path functions are type-stable
- Use concrete types in function signatures where possible
- Leverage Julia's type inference effectively

### Allocations
- Minimize allocations in differentiation rules
- Use in-place operations where safe
- Consider memory layout for performance-critical code

### Compilation
- Be aware of compilation overhead for complex rules
- Use appropriate abstractions that don't harm performance
- Consider using `@generated` functions sparingly and correctly

## Common Anti-Patterns to Avoid

- Don't ignore type stability in performance-critical code
- Avoid unnecessary allocations in pullback functions
- Don't forget to handle edge cases (empty arrays, special values)
- Avoid breaking mathematical conventions without clear justification
- Don't add rules without proper testing coverage

## Integration Points

### DifferentiationInterface.jl
- Primary integration point for users
- Provides standardized AD interface
- Support both forward and reverse modes

### ChainRules.jl
- Use `@from_chainrules` to leverage existing rules
- Convert between tangent representations when necessary
- Maintain compatibility where possible

This context should help GitHub Copilot understand the codebase structure, conventions, and domain-specific patterns when suggesting code completions and modifications.