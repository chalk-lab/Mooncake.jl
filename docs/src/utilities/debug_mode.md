# Debug Mode

```@meta
DocTestSetup = quote
    using Mooncake
end
```


_**The Problem**_

A major source of potential problems in AD systems is rules returning the wrong type of tangent / fdata / rdata for a given primal value.
For example, if someone writes a rule like
```julia
function rrule!!(::CoDual{typeof(+)}, x::CoDual{<:Real}, y::CoDual{<:Real})
    plus_reverse_pass(dz::Real) = NoRData(), dz, dz
    return zero_fcodual(primal(x) + primal(y))
end
```
and calls
```julia
rrule(zero_fcodual(+), zero_fcodual(5.0), zero_fcodual(4f0))
```
then the type of `dz` on the reverse pass will be `Float64` (assuming everything happens correctly), and this rule will return a `Float64` as the rdata for `y`.
However, the primal value of `y` is a `Float32`, and `rdata_type(Float32)` is `Float32`, so returning a `Float64` is incorrect.
This error might cause the reverse pass to fail loudly immediately, but it might also fail silently.
It might cause an error much later in the reverse pass, making it hard to determine that the source of the error was the above rule.
Worst of all, in some cases it could plausibly cause a segfault, which is more-or-less the worst kind of outcome possible.


_**The Solution**_

Check that the types of the fdata / rdata associated to arguments are exactly what `tangent_type` / `fdata_type` / `rdata_type` require upon entry to / exit from rules and pullbacks.

This is implemented via `DebugRRule`:
```@docs; canonical=false
Mooncake.DebugRRule
```

You can straightforwardly enable it when building a rule via the `debug_mode` kwarg in the following:
```@docs; canonical=false
Mooncake.build_rrule
```

You can also enable it through [`Mooncake.Config`](@ref), which is the configuration
struct accepted by the prepare-cache functions:
```jldoctest
julia> config = Mooncake.Config(; debug_mode=true);

julia> config.debug_mode
true
```

When using Mooncake via `ADTypes.AutoMooncake`, the same `Config` is forwarded through the
`config` keyword.

### When Should You Use Debug Mode?

Only use `debug_mode` when debugging a problem.
This is because is has substantial performance implications.


```@meta
DocTestSetup = nothing
```
