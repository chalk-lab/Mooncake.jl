module MooncakeJETExt

using JET, Mooncake

Mooncake.TestUtils.test_opt_internal(::Mooncake.TestUtils.Shim, x...) = JET.test_opt(x...)
Mooncake.TestUtils.report_opt_internal(::Mooncake.TestUtils.Shim, tt) = JET.report_opt(tt)

end
