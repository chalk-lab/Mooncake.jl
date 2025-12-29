# Running Tests Locally

Mooncake.jl's test suite is fairly extensive. While you can use `Pkg.test` to run the test suite in the standard manner, this is not usually optimal in Mooncake.jl, and will not run all of the tests. When editing some code, you typically only want to run the tests associated with it, not the entire test suite.

There are two workflows for running tests, discussed below.

## Main Testing Functionality

To run specific test groups, use a command like

```bash
julia --project=. -e 'import Pkg; Pkg.test(; test_args=ARGS)' -- rules/random
```

which targets the `rules/random` test group defined in `test/runtests.jl`. The complete list of test groups can be found [here](https://github.com/chalk-lab/Mooncake.jl/blob/main/test/runtests.jl).

For all code in `src`, Mooncake's tests are organised as follows:
1. Things that are required for most / all test suites are loaded up in `test/front_matter.jl`.
1. The tests for something in `src` are located in an identically-named file in `test`. e.g. the unit tests for `src/rules/new.jl` are located in `test/rules/new.jl`.

Thus, a workflow that I (Will) find works very well is the following:
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
