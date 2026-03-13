# Type-level tangent infrastructure for Core.BFloat16 (Julia >= 1.11).
# Arithmetic infrastructure and rules live in ext/MooncakeBFloat16sExt.jl.

#! format: off
@static if isdefined(Core, :BFloat16)

@foldable tangent_type(::Type{Core.BFloat16}) = Core.BFloat16

fdata_type(::Type{Core.BFloat16}) = NoFData

rdata_type(::Type{Core.BFloat16}) = Core.BFloat16

@foldable tangent_type(::Type{NoFData}, ::Type{Core.BFloat16}) = Core.BFloat16

tangent(::NoFData, r::Core.BFloat16) = r

__verify_fdata_value(::IdDict{Any,Nothing}, ::Core.BFloat16, ::NoFData) = nothing

_verify_rdata_value(::Core.BFloat16, ::Core.BFloat16) = nothing

end # @static if isdefined(Core, :BFloat16)
#! format: on
