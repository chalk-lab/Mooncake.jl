# Tangents

As discussed in [Representing Gradients](@ref), Mooncake requires that each "primal" type be associated to a unique "tangent" type, given by the function [tangent_type](@ref).
Moreover, we must be able to "split" a given tangent into its _fdata_ ("forwards-data") and _rdata_ ("reverse-data"), whose types are given by [`Mooncake.fdata_type`](@ref) and [`Mooncake.rdata_type`](@ref) respectively.
Furthermore, we (at the very least) require methods of `rrule!!` for a few core functions in order to be able to differentiate through construction and the getting / setting of fields.

_Very_ occasionally it may be necessary to specify your own tangent type.
This is not an entirely trivial undertaking, as there is quite a lot of functionality that must be added to make it work properly.
So, before diving into adding your own custom type, seriously consider whether it is worth the effort, and whether the default definition given by Mooncake is really inadequate for your use-case.

## Testing Functionality

The interface is given in the form of three functions, each of which specify which functions you must implement methods for when creating a custom tangent type:
```@docs
Mooncake.TestUtils.test_tangent_interface
Mooncake.TestUtils.test_tangent_splitting
Mooncake.TestUtils.test_rule_and_type_interactions
```

You can call all three of these functions at once using
```@docs
Mooncake.TestUtils.test_data
```

If all the tests in these functions pass, then you have satisfied the interface.

## Forward-mode representation interface

The functions above define the *reverse-mode* tangent interface. The forward-mode (`Lifted` /
`NDual`) representation has a parallel, rule-free contract checked by
[`Mooncake.TestUtils.test_lifted`](@ref) (with [`Mooncake.TestUtils.test_lifted_type`](@ref) for the
type-level part) — the forward counterpart of [`Mooncake.TestUtils.test_tangent`](@ref). It verifies,
at chunk widths 1, 2 and 3, that the forward seed factories produce a coherent slot whose primal aliases
the input, that every inner dual's `.value` tracks the primal it shadows (the inner-value invariant
`test_rule` does not check), and that a reverse tangent round-trips through `unlift`/`lift`.
