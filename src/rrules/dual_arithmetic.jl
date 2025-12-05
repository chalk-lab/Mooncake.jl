@inline function _dual_add_pullback(dy::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return NoRData(), dy, dy
end

@inline function _dual_sub_pullback(dy::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return NoRData(), dy, Dual(-primal(dy), -tangent(dy))
end

@inline function _dual_neg_pullback(dy::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return NoRData(), Dual(-primal(dy), -tangent(dy))
end

@is_primitive MinimalCtx Tuple{
    typeof(+),Dual{P,T},Dual{P,T}
} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(+)}, x::CoDual{Dual{P,T}}, y::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) + primal(y)
    return CoDual(z, NoFData()), _dual_add_pullback
end

@is_primitive MinimalCtx Tuple{typeof(+),Dual{P,T},P} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(+)}, x::CoDual{Dual{P,T}}, y::CoDual{P}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) + primal(y)
    pb!! = dy -> (NoRData(), dy, NoRData())
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{typeof(+),P,Dual{P,T}} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(+)}, x::CoDual{P}, y::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) + primal(y)
    pb!! = dy -> (NoRData(), NoRData(), dy)
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{
    typeof(+),Dual{P,T},Integer
} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(+)}, x::CoDual{Dual{P,T}}, y::CoDual{<:Integer}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) + primal(y)
    pb!! = dy -> (NoRData(), dy, NoRData())
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{
    typeof(+),Integer,Dual{P,T}
} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(+)}, x::CoDual{<:Integer}, y::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) + primal(y)
    pb!! = dy -> (NoRData(), NoRData(), dy)
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{typeof(-),Dual{P,T}} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(-)}, x::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = -primal(x)
    return CoDual(z, NoFData()), _dual_neg_pullback
end

@is_primitive MinimalCtx Tuple{
    typeof(-),Dual{P,T},Dual{P,T}
} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(-)}, x::CoDual{Dual{P,T}}, y::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) - primal(y)
    return CoDual(z, NoFData()), _dual_sub_pullback
end

@is_primitive MinimalCtx Tuple{typeof(-),Dual{P,T},P} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(-)}, x::CoDual{Dual{P,T}}, y::CoDual{P}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) - primal(y)
    pb!! = dy -> (NoRData(), dy, NoRData())
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{typeof(-),P,Dual{P,T}} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(-)}, x::CoDual{P}, y::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) - primal(y)
    pb!! = dy -> (NoRData(), NoRData(), Dual(-primal(dy), -tangent(dy)))
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{
    typeof(-),Dual{P,T},Integer
} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(-)}, x::CoDual{Dual{P,T}}, y::CoDual{<:Integer}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) - primal(y)
    pb!! = dy -> (NoRData(), dy, NoRData())
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{
    typeof(-),Integer,Dual{P,T}
} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(-)}, x::CoDual{<:Integer}, y::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    z = primal(x) - primal(y)
    pb!! = dy -> (NoRData(), NoRData(), Dual(-primal(dy), -tangent(dy)))
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{
    typeof(*),Dual{P,T},Dual{P,T}
} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(*)}, x::CoDual{Dual{P,T}}, y::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    px, py = primal(x), primal(y)
    z = px * py
    function mul_dual_dual_pb!!(dy::Dual{P,T})
        dx = py * dy
        dy_out = px * dy
        return NoRData(), dx, dy_out
    end
    return CoDual(z, NoFData()), mul_dual_dual_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(*),Dual{P,T},P} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(*)}, x::CoDual{Dual{P,T}}, y::CoDual{P}
) where {P<:IEEEFloat,T<:IEEEFloat}
    px, py = primal(x), primal(y)
    z = px * py
    pb!! = dy -> (NoRData(), py * dy, NoRData())
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{typeof(*),P,Dual{P,T}} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(*)}, x::CoDual{P}, y::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    px, py = primal(x), primal(y)
    z = px * py
    pb!! = dy -> (NoRData(), NoRData(), px * dy)
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{typeof(/),Dual{P,T},P} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(/)}, x::CoDual{Dual{P,T}}, y::CoDual{P}
) where {P<:IEEEFloat,T<:IEEEFloat}
    px, py = primal(x), primal(y)
    z = px / py
    pb!! = dy -> (NoRData(), dy / py, NoRData())
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{
    typeof(*),Integer,Dual{P,T}
} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(*)}, x::CoDual{<:Integer}, y::CoDual{Dual{P,T}}
) where {P<:IEEEFloat,T<:IEEEFloat}
    px, py = primal(x), primal(y)
    z = px * py
    pb!! = dy -> (NoRData(), NoRData(), px * dy)
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{
    typeof(*),Dual{P,T},Integer
} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(*)}, x::CoDual{Dual{P,T}}, y::CoDual{<:Integer}
) where {P<:IEEEFloat,T<:IEEEFloat}
    px, py = primal(x), primal(y)
    z = px * py
    pb!! = dy -> (NoRData(), py * dy, NoRData())
    return CoDual(z, NoFData()), pb!!
end

@is_primitive MinimalCtx Tuple{typeof(^),Dual{P,T},Int} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(^)}, x::CoDual{Dual{P,T}}, n::CoDual{Int}
) where {P<:IEEEFloat,T<:IEEEFloat}
    px = primal(x)
    pn = primal(n)
    z = px^pn
    function pow_dual_int_pb!!(dy::Dual{P,T})
        dx = pn * px^(pn - 1) * dy
        return NoRData(), dx, NoRData()
    end
    return CoDual(z, NoFData()), pow_dual_int_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(^),Dual{P,T},P} where {P<:IEEEFloat,T<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(^)}, x::CoDual{Dual{P,T}}, y::CoDual{P}
) where {P<:IEEEFloat,T<:IEEEFloat}
    px, py = primal(x), primal(y)
    z = px^py
    function pow_dual_float_pb!!(dy::Dual{P,T})
        dx = py * px^(py - one(P)) * dy
        return NoRData(), dx, NoRData()
    end
    return CoDual(z, NoFData()), pow_dual_float_pb!!
end
