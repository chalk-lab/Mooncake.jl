# Running Tests Locally

Mooncake.jl’s test suite is extensive. While you *can* run it with `Pkg.test`, this is usually suboptimal and will not run all tests. During development, you typically want to run only the tests relevant to the code you’re editing, not the entire suite.

Two workflows for running tests are described below.

## Core Test Structure

Tests for code in `src` are organized as follows:

1. Shared setup code for most test suites lives in `test/front_matter.jl`.
2. Tests for a file in `src` are located in a file with the same relative path under `test`.  
   For example, tests for `src/rules/new.jl` are in `test/rules/new.jl`.

From the repository root, you can run a specific test group with:

```bash
julia --project=. -e 'import Pkg; Pkg.test(; test_args=ARGS)' -- rules/random
```

This command runs the `rules/random` test group defined in `test/runtests.jl`.  
A complete list of test groups is available [here](https://github.com/chalk-lab/Mooncake.jl/blob/main/test/runtests.jl).

For debugging or verifying a specific rule, see [Debugging and MWEs](@ref Debugging-and-MWEs).

## Recommended Development Workflow

A workflow that I (Will) find works very well is the following:
1. Ensure that you have Revise.jl and TestEnv.jl installed in your default environment.
1. start the REPL, `dev` Mooncake.jl, and navigate to the top level of the Mooncake.jl directory.
1. `using TestEnv, Revise`. Better still, load both of these in your `.julia/config/startup.jl` file so that you don't ever forget to load them.
1. Run the following: `using Pkg; Pkg.activate("."); TestEnv.activate(); include("test/front_matter.jl");` to set up your environment.
1. `include` whichever test file you want to run the tests from.
1. Modify code, and re-`include` tests to check it has done was you need. Loop this until done.
1. Make a PR. This runs the entire test suite -- I find that I almost _never_ run the entire test suite locally.

The purpose of this approach is to:
1. Avoid restarting the REPL each time you make a change, and
2. Run the smallest bit of the test suite possible when making changes, in order to make development a fast and enjoyable process.

If you find that this strategy leaves you running more of the test suite than you would like, consider copy + pasting specific tests into the REPL, or commenting out a chunk of tests in the file that you are editing during development (try not to commit this).
I find this is rather crude strategy effective in practice.

## Extension and Integration Testing

Mooncake now has quite a lot of package extensions, and a large number of integration tests.
Unfortunately, these come with a lot of additional dependencies.
To avoid these dependencies causing CI to take much longer to run, we locate all tests for extensions and integration testing in their own environments. These can be found in the `test/ext` and `test/integration_testing` directories respectively.

These directories comprise a single `.jl` file, and a `Project.toml`.
You should run these tests by simply `include`ing the `.jl` file. Doing so will activate the environemnt, ensure that the correct version of Mooncake is used, and run the tests.
