module MooncakeDynamicPPLExt

__precompile__(false)

using DynamicPPL
using Mooncake
import Mooncake: set_to_zero!!
using Mooncake: NoTangent, Tangent, MutableTangent, NoCache, set_to_zero_internal!!

"""
Check if a tangent corresponds to a DynamicPPL.LogDensityFunction
"""
function is_dppl_ldf_tangent(x)
    x isa Tangent || return false
    hasfield(typeof(x), :fields) || return false

    fields = x.fields
    propertynames(fields) == (:model, :varinfo, :context, :adtype, :prep) || return false

    is_dppl_varinfo_tangent(fields.varinfo) || return false
    is_dppl_model_tangent(fields.model) || return false

    return true
end

"""
Check if a tangent corresponds to a DynamicPPL.VarInfo
"""
function is_dppl_varinfo_tangent(x)
    x isa Tangent || return false
    hasfield(typeof(x), :fields) || return false

    fields = x.fields
    propertynames(fields) == (:metadata, :logp, :num_produce) || return false

    # Additional validation could be added here
    return true
end

"""
Check if a tangent corresponds to a DynamicPPL.Model
"""
function is_dppl_model_tangent(x)
    x isa Tangent || return false
    hasfield(typeof(x), :fields) || return false

    fields = x.fields
    all(f in propertynames(fields) for f in (:f, :args, :defaults, :context)) ||
        return false

    return true
end

"""
Check if a MutableTangent corresponds to DynamicPPL.Metadata
"""
function is_dppl_metadata_tangent(x)
    x isa MutableTangent || return false
    hasfield(typeof(x), :fields) || return false

    fields = x.fields
    # Check for the expected fields in Metadata
    expected_fields = (:idcs, :vns, :ranges, :vals, :dists, :orders, :flags)
    all(f in propertynames(fields) for f in expected_fields) || return false

    return true
end

function Mooncake.set_to_zero!!(x)
    if is_dppl_ldf_tangent(x)
        model_f_tangent = x.fields.model.fields.f
        is_closure = false
        if model_f_tangent isa MutableTangent
            is_closure = true
        elseif model_f_tangent isa Tangent && hasfield(typeof(model_f_tangent), :fields)
            # Check if any field is a MutableTangent with PossiblyUninitTangent{Any}
            for (_, fval) in pairs(model_f_tangent.fields)
                if fval isa MutableTangent &&
                    hasfield(typeof(fval), :fields) &&
                    hasfield(typeof(fval.fields), :contents) &&
                    fval.fields.contents isa Mooncake.PossiblyUninitTangent{Any}
                    is_closure = true
                    break
                end
            end
        end

        if is_closure
            return set_to_zero_internal!!(IdDict{Any,Bool}(), x)
        else
            return set_to_zero_internal!!(NoCache(), x)
        end
    elseif x isa Tangent && (is_dppl_varinfo_tangent(x) || is_dppl_model_tangent(x))
        # VarInfo and Model
        return set_to_zero_internal!!(NoCache(), x)
    elseif x isa MutableTangent && is_dppl_metadata_tangent(x)
        return set_to_zero_internal!!(NoCache(), x)
    else
        # Use the original implementation with IdDict for all other types
        return set_to_zero_internal!!(IdDict{Any,Bool}(), x)
    end
end

end # module
