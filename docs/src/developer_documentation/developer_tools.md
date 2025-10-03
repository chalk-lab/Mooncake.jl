# Developer Tools

Mooncake.jl offers developers to a few convenience functions which give access to the IR
that it generates in order to perform AD. These are lightweight wrappers around internals
which save you from having to dig in to the objects created by `build_rrule`.

Since these provide access to internals, they do not follow the usual rules of semver, and
may change without notice!
```@docs; canonical=false
Mooncake.primal_ir
Mooncake.dual_ir
Mooncake.fwd_ir
Mooncake.rvs_ir
```
