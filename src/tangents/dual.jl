"""
    Dual(primal::P, tangent::T)

Internal forward-mode `(primal, tangent)` pair retained for the
forward-over-reverse `LazyFoRRule` / `DynamicFoRRule` HVP path in
`src/rules/high_order_derivative_patches.jl`, which threads a
`Dual(rule, rule_tangent)` through the compiled OpaqueClosure layer.

External forward-mode AD uses `Lifted{P, N, V}` (see `src/lifted.jl`);
the public `frule!!` interface dispatches on `Lifted`.
"""
struct Dual{P,T}
    primal::P
    tangent::T
end

primal(x::Dual) = x.primal
tangent(x::Dual) = x.tangent
Base.copy(x::Dual) = Dual(copy(primal(x)), copy(tangent(x)))
_copy(x::P) where {P<:Dual} = x
extract(x::Dual) = primal(x), tangent(x)

# Sharpening at the boundary: a `Type{P}` primal becomes `Dual{Type{P}, NoTangent}`
# so downstream static dispatch can resolve the parameter rather than seeing the
# widened `DataType` slot.
function Dual(x::Type{P}, dx::NoTangent) where {P}
    return Dual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(x, dx)
end

_primal(x) = x
_primal(x::Dual) = primal(x)
