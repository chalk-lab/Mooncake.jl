# Summary

At any given point in time, `Mooncake.jl` supports the current Long Term Support (LTS) release of Julia, and the latest release version of Julia 1.
Consequently, the versions of Julia which are officially supported by `Mooncake.jl` will change (almost) _immediately_ whenever a new Julia LTS version is declared, or a minor release of Julia is made.

We may also run CI against a prerelease of the next Julia minor version to find compatibility problems before release. These lanes are preparatory: a Julia version becomes officially supported only when its stable release is available.

The LTS is 1.10 and the latest stable release is 1.13, at the time of writing. With the release of 1.13, we
1. run required CI on 1.10 and stable 1.13, including extension and integration tests,
1. cease to run CI or guarantee bug fixes for 1.12,
1. cease to accept 1.12-specific bug fixes, as we no longer run CI for 1.12 and therefore cannot test that they have worked.

Official support is distinct from installability. Julia 1.11 and 1.12 remain allowed by the Julia compat bounds on a best-effort basis, with minimum versions 1.11.6 and 1.12.1 respectively. They are not covered by CI, and correctness or continued compatibility is not guaranteed. Dropping official support does not by itself require excluding a version from compat. Untested future minor versions are not admitted automatically.

Note that these changes are not applied retrospectively to existing releases of `Mooncake.jl`.
Users staying on Julia 1.11 or 1.12 can install new Mooncake releases while their compat bounds permit it, or retain an older compatible release if they encounter problems.

# Patch Versions

The above only discussed minor versions of Julia (1.10, 1.11, 1.12, etc).
However, it also applies to patch versions of Julia.
For example, at the time of writing, Julia version 1.10.12 is _actually_ the LTS, and 1.13.0 the current release of Julia.
The moment that 1.10.13 is released, we will cease to run any CI on 1.10.12, and will not accept fixes for it.
The same is true of 1.13.1 replacing 1.13.0.

Since patch releases of Julia are less invasive than minor releases, this should generally not cause users problems.

# Context

In order to officially support a particular version of Julia, we must
1. always run CI for that version,
1. accept and proactively produce fixes for that version,
1. maintain version-specific code in the `Mooncake.jl` codebase.

All of this adds a surprising amount of overhead to the development of `Mooncake.jl`, and tends to substantially increase the complexity of the codebase.
All of this makes it harder to improve `Mooncake.jl`.
Consequently, this policy represents a decision to tradeoff support for a range of minor Julia versions in exchange for easing the development burden associated to `Mooncake.jl`.

## Why not gently drop official support?

In the JuliaGaussianProcesses ecosystem, we had a loosely-defined policy of keeping support for an older version until we ran into a large problem which could not be fixed easily, at which point we would drop support.
While this sounds appealing, in practice it makes it hard to know exactly when to drop support for a particular version of Julia, increases the burden for maintainers, and makes it hard for users to know exactly what to expect.
Allowing installation on a best-effort basis does not extend this official support commitment.
