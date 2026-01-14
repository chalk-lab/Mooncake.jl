using Mooncake
using Mooncake: tangent_type, frule!!, rrule!!

struct ChunkedPrimitiveTangent{N,P}
    tangents::NTuple{N,P}
end

function (::ChunkedPrimitiveTangent{N,P})(val::P) where {N,P}
return ChunkedPrimitiveTangent{N, P}(ntuple(_ -> val, Val(N)))
end

# Do this for all primitives.
function Mooncake.tangent_type(::Type{Float64}, ::Val{N}) where {N}
    return ChunkedPrimitiveTangent{tangent_type(T),N}
end

# Compositions of primtives become treessssss
# with trees fields can be acces nrmally with chunks etc.

# how frule!! and rrule!! acts upon ChunkedPrimitiveTangent?
Mooncake.frule!!()
end
Mooncake.rrule!!()
end