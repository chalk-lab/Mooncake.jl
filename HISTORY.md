# 0.4.143

## Public Interface
- Mooncake offers forward mode AD.
- Two new functions added to the public interface: `prepare_derivative_cache` and `value_and_derivative!!`.
- One new type added to the public interface: `Dual`.

## Internals
- `get_interpreter` was previously a zero-arg function. Is now a unary function, called with a "mode" argument: `get_interpreter(ForwardMode)`, `get_interpreter(ReverseMode)`.
- `@zero_derivative` should now be preferred to `@zero_adjoint`. `@zero_adjoint` will be removed in 0.5.
- `@from_chainrules` should now be preferred to `@from_rrule`. `@from_rrule` will be removed in 0.5.
