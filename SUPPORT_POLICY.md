# Summary

At any given point in time, `Mooncake.jl` supports the current Long Term Support (LTS) release of Julia, and the latest release version of Julia 1.
Consequently, the versions of Julia which are officially supported by `Mooncake.jl` will change (almost) _immediately_ whenever a new Julia LTS version is declared, or a minor release of Julia is made.

We may also run CI against a prerelease of the next Julia minor version to find compatibility problems before release. These lanes are preparatory: a Julia version becomes officially supported only when its stable release is available.

For example, the LTS is 1.10 and the latest release is 1.12, at the time of writing. When 1.13 is released, we will
1. bump the Julia compat bounds in `Mooncake.jl` to require either 1.10 or 1.13,
1. cease to run CI on 1.12,
1. cease to provide bug fixes for 1.12,
1. cease to accept 1.12-specific bug fixes, as we will not be running CI for 1.12 and therefore will not be able to test that they have worked.

In short: as far as `Mooncake.jl`'s future releases are concerned, 1.12 ceases to exist the moment 1.13 is released.

Note that these changes are not applied retrospectively to existing releases of `Mooncake.jl`.
Suppose that `Mooncake.jl` is at `v0.4.50` when 1.12 is released.
Then the above changes would be relevant to `Mooncake.jl` versions `v0.4.51` and higher.

# Patch Versions

The above only discussed minor versions of Julia (1.10, 1.11, 1.12, etc).
However, it also applies to patch versions of Julia.
For example, at the time of writing, Julia version 1.10.12 is _actually_ the LTS, and 1.12.7 the current release of Julia.
The moment that 1.10.13 is released, we will cease to run any CI on 1.10.12, and will not accept fixes for it.
The same is true of 1.12.8.

Since patch releases of Julia are less invasive than minor releases, this should generally not cause users problems.

# Context

In order to support a particular version of Julia, we must
1. always run CI for that version,
1. accept and proactively produce fixes for that version,
1. maintain version-specific code in the `Mooncake.jl` codebase.

All of this adds a surprising amount of overhead to the development of `Mooncake.jl`, and tends to substantially increase the complexity of the codebase.
All of this makes it harder to improve `Mooncake.jl`.
Consequently, this policy represents a decision to tradeoff support for a range of minor Julia versions in exchange for easing the development burden associated to `Mooncake.jl`.

## Why not gently drop support?

In the JuliaGaussianProcesses ecosystem, we had a loosely-defined policy of keeping support for an older version until we ran into a large problem which could not be fixed easily, at which point we would drop support.
While this sounds appealing, in practice it makes it hard to know exactly when to drop support for a particular version of Julia, increases the burden for maintainers, and makes it hard for users to know exactly what to expect.
