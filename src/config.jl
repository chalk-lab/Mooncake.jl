"""
    Config(;
        debug_mode::Bool=false,
        silence_debug_messages::Bool=false,
        friendly_tangents::Bool=false,
        chunk_size::Union{Nothing,Int}=nothing,
        empty_cache::Bool=false,
        second_order_mode::Symbol=:forward_over_reverse,
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
- `chunk_size::Union{Nothing,Int}=nothing`: optional forward chunk width for the public
    [`prepare_derivative_cache`](@ref) path and APIs layered on top of it. `nothing` uses
    Mooncake's default width-1 path; an explicit integer compiles a width-`N` forward rule
    and uses chunked evaluation in [`value_and_derivative!!`](@ref) /
    [`value_and_gradient!!`](@ref). This does not affect reverse-mode caches.
- `empty_cache::Bool=false`: if `true`, all internal Mooncake caches (compiled OpaqueClosures,
    CodeInstances, and type-inference results) are cleared before building the new rule. This
    allows the garbage collector to reclaim memory held by previously compiled rules, and is
    useful in long-running sessions where many distinct functions have been differentiated.
    Note that only Julia-level (GC-managed) objects are freed; JIT-compiled native machine
    code is held permanently by the Julia runtime and cannot be reclaimed.
- `second_order_mode::Symbol=:forward_over_reverse`: controls the nesting strategy used by
    [`prepare_hvp_cache`](@ref) and [`prepare_hessian_cache`](@ref).
    `:forward_over_reverse` differentiates a gradient closure with forward-mode AD.
    `:reverse_over_forward` compiles a reverse-mode rule over `NDual` inputs so that a
    single forward+backward pass yields both the gradient and the Hessian-vector product.
"""
struct Config
    debug_mode::Bool
    silence_debug_messages::Bool
    friendly_tangents::Bool
    chunk_size::Union{Nothing,Int}
    empty_cache::Bool
    second_order_mode::Symbol
    function Config(
        debug_mode,
        silence_debug_messages,
        friendly_tangents,
        chunk_size,
        empty_cache,
        second_order_mode,
    )
        second_order_mode in (:forward_over_reverse, :reverse_over_forward) || throw(
            ArgumentError(
                "`second_order_mode` must be `:forward_over_reverse` or " *
                "`:reverse_over_forward`, got `:$second_order_mode`.",
            ),
        )
        return new(
            debug_mode,
            silence_debug_messages,
            friendly_tangents,
            chunk_size,
            empty_cache,
            second_order_mode,
        )
    end
end

function Config(;
    debug_mode::Bool=false,
    silence_debug_messages::Bool=false,
    friendly_tangents::Bool=false,
    chunk_size::Union{Nothing,Int}=nothing,
    enable_nfwd::Bool=true,
    empty_cache::Bool=false,
    second_order_mode::Symbol=:forward_over_reverse,
)
    if !enable_nfwd
        Base.depwarn(
            "The `enable_nfwd` keyword argument is deprecated and has no effect.", :Config
        )
    end
    return Config(
        debug_mode,
        silence_debug_messages,
        friendly_tangents,
        chunk_size,
        empty_cache,
        second_order_mode,
    )
end

# Backward-compatible 6-arg positional overload for the old field order:
#   Config(debug_mode, silence_debug_messages, friendly_tangents, chunk_size,
#          enable_nfwd::Bool, empty_cache::Bool)
# The new inner constructor has `second_order_mode::Symbol` in the 6th slot, so this
# overload catches the old (Bool, Bool) tail and maps it to the new layout.
function Config(
    debug_mode::Bool,
    silence_debug_messages::Bool,
    friendly_tangents::Bool,
    chunk_size::Union{Nothing,Int},
    enable_nfwd::Bool,
    empty_cache::Bool,
)
    if !enable_nfwd
        Base.depwarn(
            "The `enable_nfwd` positional argument is deprecated and has no effect.",
            :Config,
        )
    end
    return Config(
        debug_mode,
        silence_debug_messages,
        friendly_tangents,
        chunk_size,
        empty_cache,
        :forward_over_reverse,
    )
end

# Copy constructor: derive a new Config from an existing one with selected fields
# overridden. Avoids hand-listing every field at call sites — new Config fields
# propagate automatically.
function Config(base::Config; kwargs...)
    overrides = NamedTuple(kwargs)
    fields = ntuple(
        i -> begin
            n = fieldname(Config, i)
            haskey(overrides, n) ? overrides[n] : getfield(base, n)
        end, fieldcount(Config)
    )
    return Config(fields...)
end
