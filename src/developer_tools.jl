"""
    primal_ir(sig::Type{<:Tuple}; interp=get_interpreter())::IRCode

!!! warning: this is not part of the public interface of Mooncake. As such, it may change as
part of a non-breaking release of the package.

Get the `Core.Compiler.IRCode` associated to `sig` from which the a rule can be derived.
Roughly equivalent to `Base.code_ircode_by_type(sig; interp)`.

For example, if you wanted to get the IR associated to the call `map(sin, randn(10))`, you
could do one of the following calls:
```jldoctest
julia> Mooncake.primal_ir(Tuple{typeof(map), typeof(sin), Vector{Float64}}) isa Core.Compiler.IRCode
true
julia> Mooncake.primal_ir(typeof((map, sin, randn(10)))) isa Core.Compiler.IRCode
true
```
"""
function primal_ir(sig::Type{<:Tuple}; interp=get_interpreter())::IRCode
    return generate_ir(interp, sig).primal_ir
end

"""
    fwd_ir(
        sig::Type{<:Tuple};
        interp=get_interpreter(), debug_mode::Bool=false, do_inline::Bool=true
    )::IRCode

!!! warning: this is not part of the public interface of Mooncake. As such, it may change as
part of a non-breaking release of the package.

Generate the `Core.Compiler.IRCode` used to construct the forwards-pass of AD. Take a look
at how `build_rrule` makes use of `generate_ir` to see exactly how this is used in practice.

For example, if you wanted to get the IR associated to the forwards pass for the call
`map(sin, randn(10))`, you could do either of the following:
```jldoctest
julia> Mooncake.fwd_ir(Tuple{typeof(map), typeof(sin), Vector{Float64}}) isa Core.Compiler.IRCode
true
julia> Mooncake.fwd_ir(typeof((map, sin, randn(10)))) isa Core.Compiler.IRCode
true
```

# Arguments
- `sig::Type{<:Tuple}`: the signature of the call to be differentiated.

# Keyword Arguments
- `interp`: the interpreter to use to obtain the primal IR.
- `debug_mode::Bool`: whether the generated IR should make use of Mooncake's debug mode.
- `do_inline::Bool`: whether to apply an inlining pass prior to returning the ir generated
    by this function. This is `true` by default, but setting it to `false` can sometimes be
    helpful if you need to understand what function calls are generated in order to perform
    AD, before lots of it gets inlined away.
"""
function fwd_ir(
    sig::Type{<:Tuple};
    interp=get_interpreter(),
    debug_mode::Bool=false,
    do_inline::Bool=true,
)::IRCode
    return generate_ir(interp, sig; debug_mode, do_inline).fwd_ir
end

"""
    rvs_ir(
        sig::Type{<:Tuple};
        interp=get_interpreter(), debug_mode::Bool=false, do_inline::Bool=true
    )::IRCode

!!! warning: this is not part of the public interface of Mooncake. As such, it may change as
part of a non-breaking release of the package.

Generate the `Core.Compiler.IRCode` used to construct the reverse-pass of AD. Take a look
at how `build_rrule` makes use of `generate_ir` to see exactly how this is used in practice.

For example, if you wanted to get the IR associated to the reverse pass for the call
`map(sin, randn(10))`, you could do either of the following:
```jldoctest
julia> Mooncake.rvs_ir(Tuple{typeof(map), typeof(sin), Vector{Float64}}) isa Core.Compiler.IRCode
true
julia> Mooncake.rvs_ir(typeof((map, sin, randn(10)))) isa Core.Compiler.IRCode
true
```

# Arguments
- `sig::Type{<:Tuple}`: the signature of the call to be differentiated.

# Keyword Arguments
- `interp`: the interpreter to use to obtain the primal IR.
- `debug_mode::Bool`: whether the generated IR should make use of Mooncake's debug mode.
- `do_inline::Bool`: whether to apply an inlining pass prior to returning the ir generated
    by this function. This is `true` by default, but setting it to `false` can sometimes be
    helpful if you need to understand what function calls are generated in order to perform
    AD, before lots of it gets inlined away.
"""
function rvs_ir(
    sig::Type{<:Tuple};
    interp=get_interpreter(),
    debug_mode::Bool=false,
    do_inline::Bool=true,
)::IRCode
    return generate_ir(interp, sig; debug_mode, do_inline).rvs_ir
end

"""

"""
function dual_ir(
    sig::Type{<:Tuple};
    interp=get_interpreter(),
    debug_mode::Bool=false,
    do_inline::Bool=true,
)
    return generate_dual_ir(interp, sig; debug_mode, do_inline)
end
