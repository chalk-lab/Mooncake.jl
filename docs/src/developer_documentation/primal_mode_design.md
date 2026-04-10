# Primal Mode Design

`build_primal` is the active IR-lifted forward path. The core plan is:

1. resolve and normalise primal IR for `f`
2. rebuild `f` as a primal/lifted executable rather than redispatching on AD wrappers
3. recursively use `build_primal(g, ...)` for resolved non-primitive callees so dispatch is
   preserved through the resolved call graph
4. keep primitive / intrinsic / builtin leaves on explicit primitive rule paths
5. keep an explicit fallback only for genuinely unresolved dynamic calls

The recursion / caching side of that problem is already the same one handled by
`LazyPrimal`, `DynamicPrimal`, and `DerivedPrimal`; the interesting design work is
representation choice and the lifted execution boundary.

## Routes Considered

- keep the lifted path fully on Mooncake `Dual` plus width-aware tangent storage
- make the lifted path generic over `dual_type(::Val{N}, P)` so scalar / complex cases can
  choose `NDual` directly
- overload `tangent_type(::Val{N}, ...)` for packed layouts rather than changing dual type
- use cached internal `Array{NDual}` work buffers behind callable rule objects for array
  lowering, instead of trying to make array public dual types themselves be `Array{NDual}`
- previously, lower some primitive sites through separate nfwd bridge types; that route was
  removed in favour of Mooncake-owned primitive wrapper entrypoints

## Candidate Chunked Tangent Layouts

- today's basis-dir-outer `NTangent`, i.e. a tuple of ordinary tangents
- a packed `NTangent` with basis-dir-inner scalar leaves closer to `NDual`
- overloading `tangent_type(::Val{N}, ...)` only for `Array{<:IEEEFloat}` /
  `Array{<:Complex{<:IEEEFloat}}` so array leaves use NDual-like packed storage
- overloading `tangent_type(::Val{N}, ...)` only for those arrays so tangents use one extra
  basis-dir dimension instead

## Current Choices

- the active lifted IR path constructs and types values through
  `dual_type(::Val{N}, P)` / `tangent_type(::Val{N}, P)`, so width-aware scalar and
  complex paths can use `NDual` directly while structured values still default to ordinary
  Mooncake `Dual` unless users choose a different `dual_type` overload
- the public boundary is generic over width-aware dual types via `dual_type(::Val{N}, P)`
- direct nfwd integration now lives behind Mooncake-owned primitive wrappers and
  width-aware dual-type overloads, not a separate `NfwdMooncake` runtime include
- primitive wrappers may accept top-level `NDual` directly where that path is explicit,
  but the general lifted path does not proactively lower everything to `NDual`

## Width-Aware Dual-Type Protocol

The remaining larger design point is to make the lifted path itself generic over a
width-aware dual-type protocol, rather than assuming concrete `Dual{P,T}` internally. A
minimal construction / extraction protocol would be:

- `primal(x)`
- `tangent(x)`
- `dual_type(::Val{N}, ::Type{P})`
- `zero_dual(::Val{N}, x)`
- `uninit_dual(::Val{N}, x)`
- construction via `dual_type(Val(N), P)(x, dx)`, which may yield `Dual`, `NDual`, or
  another width-aware dual type

Public top-level forward entrypoints currently validate inputs with `verify_dual_type(x)`,
but that check is auto-derived from the protocol above in the common case.

That keeps width-aware dual-type choice type-driven and inference-friendly while
separating dual-type selection from tangent-storage selection. The key type-stability
constraint is that `dual_type(Val(N), P)` must remain foldable to a concrete dual type for
concrete `P`; runtime dual-type switching would immediately weaken inference through the
transformed IR.

## Still Open

- future reuse of this machinery for batched / vmap-style transforms
- internal lifted-path support for arbitrary width-aware dual types, not just public-entry
  support through `dual_type(::Val{N}, P)`
