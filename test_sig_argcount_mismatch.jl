#!/usr/bin/env julia

# Minimal reproducer for sig_argcount_mismatch test failure
# Run with: julia test_sig_argcount_mismatch.jl

using Pkg
Pkg.activate(; temp=true)
Pkg.add("TestEnv")
using TestEnv
Pkg.activate("./")
TestEnv.activate()

using Mooncake
using Mooncake.TestResources: sig_argcount_mismatch
using Mooncake.TestUtils
using StableRNGs

# The failing test case
println("Testing sig_argcount_mismatch...")
rng = StableRNG(123456)
x = ones(4)

println("\n=== Testing Forward Mode ===")
try
    TestUtils.test_rule(
        rng,
        sig_argcount_mismatch,
        x;
        is_primitive=false,
        perf_flag=:none,
        interface_only=false,
        mode=Mooncake.ForwardMode,
    )
    println("✓ Forward mode passed")
catch e
    println("✗ Forward mode failed:")
    showerror(stdout, e, catch_backtrace())
    println()
end

println("\n=== Testing Reverse Mode ===")
try
    TestUtils.test_rule(
        rng,
        sig_argcount_mismatch,
        x;
        is_primitive=false,
        perf_flag=:none,
        interface_only=false,
        mode=Mooncake.ReverseMode,
    )
    println("✓ Reverse mode passed")
catch e
    println("✗ Reverse mode failed:")
    showerror(stdout, e, catch_backtrace())
    println()
end

println("\n=== Finished ===")
