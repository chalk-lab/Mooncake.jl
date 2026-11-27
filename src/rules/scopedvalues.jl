@static if isdefined(Base.ScopedValues, :AbstractScopedValue)
    using Base.ScopedValues: AbstractScopedValue
else
    using Base.ScopedValues: ScopedValue as AbstractScopedValue
end
using Base.ScopedValues: Scope

function throw_scopedvalue_error()
    unhandled_feature(
        "ScopedValues are not supported in code differentiated by Mooncake.jl, whether " *
        "reading one or entering a scope with `with` or `@with`. Read the value before " *
        "differentiating and pass it in as an argument, or move the `with` call outside " *
        "the differentiated function. A rule for the enclosing function also works; " *
        "@zero_derivative is right only when its result neither depends on nor aliases " *
        "the differentiated inputs.",
    )
end

# Scope is non-public and Base builds one only to enter it, which is refused. Direct
# construction or a `.values` read still overflows in `tangent_type(PersistentDict)`.
tangent_type(::Type{Scope}) = NoTangent

# getindex delegates to get; isassigned consults the scope independently.
for f in (Base.ScopedValues.get, isassigned)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),AbstractScopedValue}
        function frule!!(::Dual{typeof($f)}, ::Dual{<:AbstractScopedValue})
            throw_scopedvalue_error()
        end
        function rrule!!(::CoDual{typeof($f)}, ::CoDual{<:AbstractScopedValue})
            throw_scopedvalue_error()
        end
    end
end

frule!!(::Dual{typeof(Core.current_scope)}) = throw_scopedvalue_error()
rrule!!(::CoDual{typeof(Core.current_scope)}) = throw_scopedvalue_error()
