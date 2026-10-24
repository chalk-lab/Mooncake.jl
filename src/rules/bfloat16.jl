# Type-level tangent infrastructure for Core.BFloat16 (Julia >= 1.11).
# Arithmetic infrastructure and rules live in ext/MooncakeBFloat16sExt.jl.

#! format: off
@static if isdefined(Core, :BFloat16)

@foldable tangent_type(::Type{NoFData}, ::Type{Core.BFloat16}) = Core.BFloat16

end # @static if isdefined(Core, :BFloat16)
#! format: on
