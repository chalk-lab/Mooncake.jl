# 0.4.143

## Public Interface
N/A

## Internals
- `get_interpreter` was previously a zero-arg function. Is now a unary function, called with a "mode" argument: `get_interpreter(ForwardMode)`, `get_interpreter(ReverseMode)`.
- 