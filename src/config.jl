"""
    Config(;
        debug_mode::Bool=false,
        silence_debug_messages::Bool=false,
        friendly_tangents::Bool=false,
        chunk_size::Union{Nothing,Int}=nothing,
        empty_cache::Bool=false,
    )

Configuration struct for use with `ADTypes.AutoMooncake`.

# Keyword Arguments
- `debug_mode::Bool=false`: whether or not to run additional type checks when
    differentiating a function. This has considerable runtime overhead, and should only be
    switched on if you are trying to debug something that has gone wrong in Mooncake.
- `silence_debug_messages::Bool=false`: if `false` and `debug_mode` is `true`, Mooncake will
    display some warnings that debug mode is enabled, in order to help prevent accidentally
    leaving debug mode on. If you wish to disable these messages, set this to `true`.
- `friendly_tangents::Bool=false`: if `true`, Mooncake will represent tangents using the
    primal type at the interface level: the tangent type of a primal type `P` will be `P`
    when using friendly tangents, and `tangent_type(P)` otherwise (e.g. the friendly tangent of a
    custom struct will be of the same type as the struct instead of Mooncake's `Tangent` type).
    The tangent is converted from/to the friendly representation at the interface level,
    so all Mooncake internal computations and rule implementations always use the
    [`tangent_type`](@ref) representation.
- `chunk_size::Union{Nothing,Int}=nothing`: reserved for future use. Currently ignored.
- `empty_cache::Bool=false`: if `true`, all internal Mooncake caches (compiled OpaqueClosures,
    CodeInstances, and type-inference results) are cleared before building the new rule. This
    allows the garbage collector to reclaim memory held by previously compiled rules, and is
    useful in long-running sessions where many distinct functions have been differentiated.
    Note that only Julia-level (GC-managed) objects are freed; JIT-compiled native machine
    code is held permanently by the Julia runtime and cannot be reclaimed.
"""
struct Config
    debug_mode::Bool
    silence_debug_messages::Bool
    friendly_tangents::Bool
    chunk_size::Union{Nothing,Int}
    empty_cache::Bool
end

function Config(;
    debug_mode::Bool=false,
    silence_debug_messages::Bool=false,
    friendly_tangents::Bool=false,
    chunk_size::Union{Nothing,Int}=nothing,
    enable_nfwd::Bool=true,
    empty_cache::Bool=false,
)
    if !enable_nfwd
        Base.depwarn(
            "The `enable_nfwd` keyword argument is deprecated and has no effect.", :Config
        )
    end
    return Config(debug_mode, silence_debug_messages, friendly_tangents, chunk_size, empty_cache)
end
