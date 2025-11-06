module MooncakeJETExt

using JET, Mooncake

@static if VERSION < v"1.12-"
  Mooncake.TestUtils.test_opt_internal(::Mooncake.TestUtils.Shim, x...) = JET.test_opt(x...)
  Mooncake.TestUtils.report_opt_internal(::Mooncake.TestUtils.Shim, tt) = JET.report_opt(tt)
else
  Mooncake.TestUtils.test_opt_internal(::Mooncake.TestUtils.Shim, x...) = nothing
  Mooncake.TestUtils.report_opt_internal(::Mooncake.TestUtils.Shim, tt) = nothing
end

end
