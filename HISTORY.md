# 0.5.0

## Breaking Changes
- The tangent type of a `Complex{P<:IEEEFloat}` is now `Complex{P}` instead of `Tangent{@NamedTuple{re::P, im::P}}`.
- The `prepare_pullback_cache`, `prepare_gradient_cache` and `prepare_derivative_cache` interface functions now accept a `Mooncake.Config` directly.

# 0.4.147

## Public Interface
- Mooncake offers forward mode AD.
- Two new functions added to the public interface: `prepare_derivative_cache` and `value_and_derivative!!`.
- One new type added to the public interface: `Dual`.

## Internals
- `get_interpreter` was previously a zero-arg function. Is now a unary function, called with a "mode" argument: `get_interpreter(ForwardMode)`, `get_interpreter(ReverseMode)`.
- `@zero_derivative` should now be preferred to `@zero_adjoint`. `@zero_adjoint` will be removed in 0.5.
- `@from_chainrules` should now be preferred to `@from_rrule`. `@from_rrule` will be removed in 0.5.
