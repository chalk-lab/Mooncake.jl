"""
    Config(; debug_mode::Bool=false, silence_debug_messages::Bool=false, friendly_tangents::Bool=false)

Configuration struct for use with `ADTypes.AutoMooncake`.

# Keyword Arguments
- `debug_mode::Bool=false`: whether or not to run additional type checks when
    differentiating a function. This has considerable runtime overhead, and should only be
    switched on if you are trying to debug something that has gone wrong in Mooncake.
- `silence_debug_messages::Bool=false`: if `false` and `debug_mode` is `true`, Mooncake will
    display some warnings that debug mode is enabled, in order to help prevent accidentally
    leaving debug mode on. If you wish to disable these messages, set this to `true`.
- `friendly_tangents::Bool=true`: if `true`, Mooncake will represent tangents using the
    primal type at the interface level: the tangent type of a primal type `P` will be `P`
    when using friendly tangents, and `tangent_type(P)` otherwise (e.g. the friendly tangent of a
    custom struct will be of the same type as the struct instead of Mooncake's `Tangent` type).
    The tangent is converted from/to the friendly representation at the interface level,
    so all Mooncake internal computations and rule implementations always use the
    [`tangent_type`](@ref) representation.
"""
@kwdef struct Config
    debug_mode::Bool = false
    silence_debug_messages::Bool = false
    friendly_tangents::Bool = true
end
