"""
    module TestTypes

A module containing types and associated utility functionality against which standardised
functionality can be tested. The goal is to provide sufficiently broad coverage of different
kinds of types to provide a high degree of confidence that correctness and performance
of tangent operations generalises to new types found in the wild.
"""
module TestTypes

using Base.Iterators: product
using Core: svec
using ExprTools: combinedef
using ..Mooncake:
    NoTangent,
    tangent_type,
    _typeof,
    set_to_zero!!,
    increment!!,
    is_primitive,
    randn_tangent,
    _scale,
    _add_to_primal,
    _diff,
    _dot

const PRIMALS = Tuple{Bool,Any,Tuple}[]

# Generate all of the composite types against which we might wish to test.
function generate_primals()
    empty!(PRIMALS)
    for n_fields in [0, 1, 2], is_mutable in [true, false]

        # Generate all possible permutations of primitive fields types.
        fields = (
            (type=Float64, primal=1.0, tangent=1.0),
            (type=Int64, primal=1, tangent=NoTangent()),
            (type=Vector{Float64}, primal=ones(2), tangent=ones(2)),
            (type=Vector{Int64}, primal=Int64[1, 1], tangent=fill(NoTangent(), 2)),
        )
        field_combinations = vec(collect(product(fill(fields, n_fields)...)))

        ns_always_def = 0:n_fields

        for fields in field_combinations, n_always_def in ns_always_def
            mutable_str = is_mutable ? "Mutable" : ""
            field_types = map(x -> x.type, fields)
            type_string = join(map(string, field_types), "_")
            name = Symbol("$(mutable_str)Struct_$(type_string)_$(n_always_def)")
            field_names = map(n -> Symbol("x$n"), 1:n_fields)

            # Create the specified type.
            struct_expr = Expr(
                :struct,
                is_mutable,
                name,
                Expr(
                    :block,

                    # Specify fields.
                    map(n -> Expr(:(::), field_names[n], field_types[n]), 1:n_fields)...,

                    # Specify inner constructors.
                    map(n_always_def:n_fields) do n
                        return combinedef(
                            Dict(
                                :head => :function,
                                :name => name,
                                :args => field_names[1:n],
                                :body => Expr(:call, :new, field_names[1:n]...),
                            ),
                        )
                    end...,
                ),
            )
            @eval $(struct_expr)

            t = @eval $name
            for n in n_always_def:n_fields
                interface_only = any(x -> isbitstype(x.type), fields[(n + 1):end])
                fields_copies = map(x -> deepcopy(x.primal), fields[1:n])
                push!(PRIMALS, (interface_only, t, fields_copies))
            end
        end
    end
    return nothing
end

instantiate(test_case) = (test_case[1], test_case[2](test_case[3]...))

end

"""
    module TestUtils

A collection of functions comprising collections of unit tests which check to see if the
interfaces that this package defines have been implemented correctly.
"""
module TestUtils

using Random, Mooncake, Test, InteractiveUtils
using Mooncake:
    CoDual,
    NoTangent,
    PossiblyUninitTangent,
    Tangent,
    MutableTangent,
    rrule!!,
    build_rrule,
    tangent_type,
    zero_tangent,
    primal,
    tangent,
    is_init,
    zero_codual,
    DefaultCtx,
    @is_primitive,
    val,
    is_always_fully_initialised,
    get_tangent_field,
    set_tangent_field!,
    MutableTangent,
    Tangent,
    _typeof,
    rdata,
    NoFData,
    to_fwds,
    uninit_fdata,
    zero_rdata,
    zero_rdata_from_type,
    CannotProduceZeroRDataFromType,
    lazy_zero_rdata,
    instantiate,
    can_produce_zero_rdata_from_type,
    increment_rdata!!,
    fcodual_type,
    verify_fdata_type,
    verify_rdata_type,
    verify_fdata_value,
    verify_rdata_value,
    InvalidFDataException,
    InvalidRDataException,
    uninit_codual,
    lgetfield,
    lsetfield!,
    increment_internal!!,
    set_to_zero_internal!!,
    CC,
    set_to_zero!!,
    increment!!,
    is_primitive,
    randn_tangent,
    _scale,
    _add_to_primal,
    _diff,
    _dot,
    NoFData,
    fdata_type,
    fdata,
    NoRData,
    rdata_type,
    rdata

struct Shim end

"""
    test_hook(f, caller, ::Any...; kws...)

A function which can alter the behavior of a given test. `f` is the function
that is normally executed, `caller` is the function this is being called from,
and `args` and `kws` are the arguments that were passed to `caller`.

Normally, `f()` is executed. However, this can be overridden by an integration test
defining custom behavior.
"""
test_hook(f, caller, args...; kws...) = f()

function test_opt(x...)
    test_hook(test_opt, x...) do
        test_opt_internal(Shim(), x...)
    end
end
test_opt_internal(::Any, x...) = throw(error("Load JET to use this function."))

function report_opt(tt)
    test_hook(report_opt, tt) do
        report_opt_internal(Shim(), tt)
    end
end
report_opt_internal(::Any, tt) = throw(error("Load JET to use this function."))

"""
    has_equal_data(x, y; equal_undefs=true)

Determine if two objects `x` and `y` have equivalent data. If `equal_undefs` 
is `true`, undefined elements in arrays or unassigned fields in structs are 
considered equal.

The main logic is implemented in `has_equal_data_internal`, which is a recursive function
that takes an additional `visited` dictionary to track visited objects and avoid infinite
recursion in cases of circular references.
"""
function has_equal_data(x, y; equal_undefs=true)
    return has_equal_data_internal(x, y, equal_undefs, Dict{Tuple{UInt,UInt},Bool}())
end

function has_equal_data_internal(
    x::Type, y::Type, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
)
    return x == y
end
function has_equal_data_internal(
    x::T, y::T, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T<:String}
    return x == y
end
function has_equal_data_internal(
    x::Core.TypeName, y::Core.TypeName, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
)
    return x == y
end
function has_equal_data_internal(
    x::P, y::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P<:Base.IEEEFloat}
    return (isapprox(x, y) && !isnan(x)) || (isnan(x) && isnan(y))
end
function has_equal_data_internal(
    x::Module, y::Module, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
)
    return x == y
end
function has_equal_data_internal(
    x::GlobalRef, y::GlobalRef; equal_undefs=true, d::Dict{Tuple{UInt,UInt},Bool}
)
    return x.mod == y.mod && x.name == y.name
end
function has_equal_data_internal(
    x::T, y::T, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T<:Array}
    size(x) != size(y) && return false

    # The dictionary is used to detect circular references in the data structures.
    # For example, if x.a.a === x and y.a.a === y, we want to consider them to have equal data.
    #
    # When we first encounter a pair of objects:
    # 1. We add them to the dictionary, marking that we've seen them.
    # 2. This doesn't guarantee they're equal, just that we've encountered them.
    #
    # As we recursively compare x and y:
    # - If we see a pair we've seen before, it indicates circular references.
    # - We consider "circular references to itself" as equal data for this subcomponent.
    # - However, other parts of x and y may still differ, so we continue checking.

    id_pair = (objectid(x), objectid(y))
    if haskey(d, id_pair)
        return d[id_pair]
    end

    d[id_pair] = true
    equality = map(1:length(x)) do n
        if isassigned(x, n) != isassigned(y, n)
            return !equal_undefs
        elseif !isassigned(x, n)
            return true
        else
            return has_equal_data_internal(x[n], y[n], equal_undefs, d)
        end
    end
    return all(equality)
end
function has_equal_data_internal(
    x::T, y::T, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T<:Core.SimpleVector}
    return all(map((a, b) -> has_equal_data_internal(a, b, equal_undefs, d), x, y))
end
function has_equal_data_internal(
    x::T, y::T, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T}
    isprimitivetype(T) && return isequal(x, y)

    id_pair = (objectid(x), objectid(y))
    if haskey(d, id_pair)
        return d[id_pair]
    end

    d[id_pair] = true

    if ismutabletype(x)
        return all(
            map(fieldnames(T)) do n
                if isdefined(x, n)
                    has_equal_data_internal(
                        getfield(x, n), getfield(y, n), equal_undefs, d
                    )
                else
                    true
                end
            end,
        )
    else
        for n in fieldnames(T)
            if !isdefined(x, n) && !isdefined(y, n)
                continue # consider undefined fields as equal
            elseif isdefined(x, n) && isdefined(y, n)
                if has_equal_data_internal(getfield(x, n), getfield(y, n), equal_undefs, d)
                    continue
                else
                    return false
                end
            else # one is defined and the other is not
                return false
            end
        end
        return true
    end
end
function has_equal_data_internal(
    x::T, y::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T,P}
    return false
end
function has_equal_data_internal(
    x::T, y::T, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T<:Dict}
    f(x, y) = has_equal_data_internal(x, y, equal_undefs, d)
    return length(x) == length(y) &&
           all(map(f, keys(x), keys(y))) &&
           all(map(f, values(x), values(y)))
end

has_equal_data_up_to_undefs(x::T, y::T) where {T} = has_equal_data(x, y; equal_undefs=false)

const AddressMap = Dict{Ptr{Nothing},Ptr{Nothing}}

"""
    populate_address_map(primal, tangent)

Constructs an empty `AddressMap` and calls `populate_address_map_internal`.
"""
function populate_address_map(primal, tangent)
    return populate_address_map_internal(AddressMap(), primal, tangent)
end

"""
    populate_address_map_internal(m::AddressMap, primal, tangent)

Fills `m` with pairs mapping from memory addresses in `primal` to corresponding memory
addresses in `tangent`. If the same memory address appears multiple times in `primal`,
throws an `AssertionError` if the same address is not mapped to in `tangent` each time.
"""
function populate_address_map_internal(m::AddressMap, primal::P, tangent::T) where {P,T}
    isprimitivetype(P) && return m
    T === NoTangent && return m
    T === NoFData && return m
    if ismutabletype(P)
        @assert T <: MutableTangent
        k = pointer_from_objref(primal)
        v = pointer_from_objref(tangent)
        if haskey(m, k)
            @assert m[k] == v
            return m
        end
        m[k] = v
    end
    foreach(fieldnames(P)) do n
        t_field = __get_data_field(tangent, n)
        if isdefined(primal, n) && is_init(t_field)
            populate_address_map_internal(m, getfield(primal, n), val(t_field))
        elseif isdefined(primal, n) && !is_init(t_field)
            throw(error("unhandled defined-ness"))
        elseif !isdefined(primal, n) && is_init(t_field)
            throw(error("unhandled defined-ness"))
        end
    end
    return m
end

__get_data_field(t::Union{Tangent,MutableTangent}, n) = getfield(t.fields, n)
__get_data_field(t::Union{Mooncake.FData,Mooncake.RData}, n) = getfield(t.data, n)

function populate_address_map_internal(
    m::AddressMap, p::P, t
) where {P<:Union{Tuple,NamedTuple}}
    t isa NoFData && return m
    t isa NoTangent && return m
    foreach(
        n -> populate_address_map_internal(m, getfield(p, n), getfield(t, n)), fieldnames(P)
    )
    return m
end

function populate_address_map_internal(m::AddressMap, p::Array, t::Array)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    if haskey(m, k)
        @assert m[k] == v
        return m
    end
    m[k] = v
    foreach(
        n -> isassigned(p, n) && populate_address_map_internal(m, p[n], t[n]), eachindex(p)
    )
    return m
end

function populate_address_map_internal(m::AddressMap, p::Core.SimpleVector, t::Vector{Any})
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    if haskey(m, k)
        @assert m[k] == v
        return m
    end
    m[k] = v
    foreach(n -> populate_address_map_internal(m, p[n], t[n]), eachindex(p))
    return m
end

function populate_address_map_internal(
    m::AddressMap, p::Union{Core.TypeName,Type,Symbol,String}, t
)
    return m
end

"""
    address_maps_are_consistent(x::AddressMap, y::AddressMap)

`true` if all keys in both `x` and `y` map to the same values. i.e. if any key appears in
both `x` and `y`, and the corresponding value is not the same in both `x` and `y`, return
`false`.
"""
function address_maps_are_consistent(x::AddressMap, y::AddressMap)
    return all(map(k -> x[k] == y[k], collect(intersect(keys(x), keys(y)))))
end

# Assumes that the interface has been tested, and we can simply check for numerical issues.
function test_rule_correctness(rng::AbstractRNG, x_x̄...; rule, unsafe_perturb::Bool)
    @nospecialize rng x_x̄

    x_x̄ = map(_deepcopy, x_x̄) # defensive copy

    # Run original function on deep-copies of inputs.
    x = map(primal, x_x̄)
    x̄ = map(tangent, x_x̄)

    # Run primal, and ensure that we still have access to mutated inputs afterwards.
    x_primal = _deepcopy(x)
    y_primal = x_primal[1](x_primal[2:end]...)

    # Use finite differences to estimate vjps. Compute the estimate at a range of different
    # step sizes. We'll just require that one of them ends up being close to what AD gives.
    ẋ = map(_x -> randn_tangent(rng, _x), x)
    ε_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    fd_results = Vector{Any}(undef, length(ε_list))
    for (n, ε) in enumerate(ε_list)
        x′_l = _add_to_primal(x, _scale(ε, ẋ), unsafe_perturb)
        y′_l = x′_l[1](x′_l[2:end]...)
        x′_r = _add_to_primal(x, _scale(-ε, ẋ), unsafe_perturb)
        y′_r = x′_r[1](x′_r[2:end]...)
        fd_results[n] = (
            ẏ=_scale(1 / 2ε, _diff(y′_l, y′_r)),
            ẋ_post=map((_x′, _x_p) -> _scale(1 / 2ε, _diff(_x′, _x_p)), x′_l, x′_r),
        )
    end

    # Run rule on copies of `f` and `x`. We use randomly generated tangents so that we
    # can later verify that non-zero values do not get propagated by the rule.
    x̄_zero = map(zero_tangent, x)
    x̄_fwds = map(Mooncake.fdata, x̄_zero)
    x_x̄_rule = map((x, x̄_f) -> fcodual_type(_typeof(x))(_deepcopy(x), x̄_f), x, x̄_fwds)
    inputs_address_map = populate_address_map(
        map(primal, x_x̄_rule), map(tangent, x_x̄_rule)
    )
    y_ȳ_rule, pb!! = rule(x_x̄_rule...)

    # Verify that inputs / outputs are the same under `f` and its rrule.
    @test has_equal_data(x_primal, map(primal, x_x̄_rule))
    @test has_equal_data(y_primal, primal(y_ȳ_rule))

    # Query both `x_x̄` and `y`, because `x_x̄` may have been mutated by `f`.
    outputs_address_map = populate_address_map(
        (map(primal, x_x̄_rule)..., primal(y_ȳ_rule)),
        (map(tangent, x_x̄_rule)..., tangent(y_ȳ_rule)),
    )
    @test address_maps_are_consistent(inputs_address_map, outputs_address_map)

    # Run reverse-pass.
    ȳ_delta = randn_tangent(rng, primal(y_ȳ_rule))
    x̄_delta = map(Base.Fix1(randn_tangent, rng) ∘ primal, x_x̄_rule)

    ȳ_init = set_to_zero!!(zero_tangent(primal(y_ȳ_rule), tangent(y_ȳ_rule)))
    x̄_init = map(set_to_zero!!, x̄_zero)
    ȳ = increment!!(ȳ_init, ȳ_delta)
    map(increment!!, x̄_init, x̄_delta)
    x̄_rvs_inc = pb!!(Mooncake.rdata(ȳ))
    x̄_rvs = increment!!(map(rdata, x̄_delta), x̄_rvs_inc)
    x̄ = map(tangent, x̄_fwds, x̄_rvs)

    # Check that inputs have been returned to their original value.
    @test all(map(has_equal_data_up_to_undefs, x, map(primal, x_x̄_rule)))

    # Pullbacks increment, so have to compare to the incremented quantity. Require only one
    # precision to be close to the answer AD gives. i.e. prove that there exists a step size
    # such that AD and central differences agree on the answer.
    isapprox_results = map(fd_results) do result
        ẏ, ẋ_post = result
        return isapprox(
            _dot(ȳ_delta, ẏ) + _dot(x̄_delta, ẋ_post),
            _dot(x̄, ẋ);
            rtol=1e-3,
            atol=1e-3,
        )
    end
    @test any(isapprox_results)
end

get_address(x) = ismutable(x) ? pointer_from_objref(x) : nothing

_deepcopy(x) = deepcopy(x)
_deepcopy(x::Module) = x

rrule_output_type(::Type{Ty}) where {Ty} = Tuple{Mooncake.fcodual_type(Ty),Any}

function test_rrule_interface(f_f̄, x_x̄...; rule)
    @nospecialize f_f̄ x_x̄

    # Pull out primals and run primal computation.
    f = primal(f_f̄)
    f̄ = tangent(f_f̄)
    x_x̄ = map(_deepcopy, x_x̄)
    x = map(primal, x_x̄)
    x̄ = map(tangent, x_x̄)

    # Run the primal programme. Bail out early if this doesn't work.
    y = try
        f(deepcopy(x)...)
    catch e
        display(e)
        println()
        throw(ArgumentError("Primal evaluation does not work."))
    end

    # Check that input types are valid.
    @test _typeof(tangent(f_f̄)) == tangent_type(_typeof(primal(f_f̄)))
    for x_x̄ in x_x̄
        @test _typeof(tangent(x_x̄)) == tangent_type(_typeof(primal(x_x̄)))
    end

    # Extract the forwards-data from the tangents.
    f_fwds = to_fwds(f_f̄)
    x_fwds = map(to_fwds, x_x̄)

    # Run the rrule, check it has output a thing of the correct type, and extract results.
    # Throw a meaningful exception if the rrule doesn't run at all.
    x_addresses = map(get_address, x)
    rrule_ret = try
        rule(f_fwds, x_fwds...)
    catch e
        display(e)
        println()
        throw(
            ArgumentError(
                "rule for $(_typeof(f_fwds)) with argument types $(_typeof(x_fwds)) does not run.",
            ),
        )
    end
    @test rrule_ret isa rrule_output_type(_typeof(y))
    y_ȳ, pb!! = rrule_ret

    # Check that returned fdata type is correct.
    @test typeof(y_ȳ.dx) == fdata_type(tangent_type(typeof(y_ȳ.x)))
    @test Mooncake._verify_fdata_value(IdDict{Any,Nothing}(), y_ȳ.x, y_ȳ.dx) === nothing

    # Run the reverse-pass. Throw a meaningful exception if it doesn't run at all.
    ȳ = Mooncake.rdata(zero_tangent(primal(y_ȳ), tangent(y_ȳ)))
    f̄_new, x̄_new... = try
        pb!!(ȳ)
    catch e
        display(e)
        println()
        throw(
            ArgumentError(
                "pullback for $(_typeof(f_f̄)) with argument types $(_typeof(x_x̄)) does not run.",
            ),
        )
    end

    # Check that the pullback returns the correct number of things.
    @test length(x̄_new) == length(x_fwds)

    # Check that memory addresses have remained constant under pb!!.
    new_x_addresses = map(get_address, x)
    @test all(map(==, x_addresses, new_x_addresses))

    # Check the tangent types output by the reverse-pass, and that memory addresses of
    # mutable objects have remained constant.
    @test _typeof(f̄_new) == _typeof(rdata(f̄))
    @test all(map((a, b) -> _typeof(a) == _typeof(rdata(b)), x̄_new, x̄))
end

@noinline function __forwards_and_backwards(rule::R, x_x̄::Vararg{Any,N}) where {R,N}
    out, pb!! = rule(x_x̄...)
    return pb!!(Mooncake.zero_rdata(primal(out)))
end

function test_rrule_performance(
    performance_checks_flag::Symbol, rule::R, f_f̄::F, x_x̄::Vararg{Any,N}
) where {R,F,N}

    # Verify that a valid performance flag has been passed.
    valid_flags = (:none, :stability, :allocs, :stability_and_allocs)
    if !in(performance_checks_flag, valid_flags)
        throw(
            ArgumentError(
                "performance_checks=$performance_checks_flag. Must be one of $valid_flags"
            ),
        )
    end
    performance_checks_flag == :none && return nothing

    if performance_checks_flag in (:stability, :stability_and_allocs)

        # Test primal stability.
        test_opt(primal(f_f̄), map(_typeof ∘ primal, x_x̄))

        # Test forwards-pass stability.
        test_opt(rule, (_typeof(to_fwds(f_f̄)), map(_typeof ∘ to_fwds, x_x̄)...))

        # Test reverse-pass stability.
        y_ȳ, pb!! = rule(to_fwds(f_f̄), map(to_fwds, _deepcopy(x_x̄))...)
        rvs_data = Mooncake.rdata(zero_tangent(primal(y_ȳ), tangent(y_ȳ)))
        test_opt(pb!!, (_typeof(rvs_data),))
    end

    if performance_checks_flag in (:allocs, :stability_and_allocs)
        f = primal(f_f̄)
        x = map(primal, x_x̄)

        # Test allocations in primal.
        f(x...)
        @test (@allocations f(x...)) == 0

        # Test allocations in round-trip.
        f_f̄_fwds = to_fwds(f_f̄)
        x_x̄_fwds = map(to_fwds, x_x̄)
        __forwards_and_backwards(rule, f_f̄_fwds, x_x̄_fwds...)
        count_allocs(__forwards_and_backwards, rule, f_f̄_fwds, x_x̄_fwds...)
        @test count_allocs(__forwards_and_backwards, rule, f_f̄_fwds, x_x̄_fwds...) == 0
    end
end

__get_primals(xs) = map(x -> x isa CoDual ? primal(x) : x, xs)

"""
    test_rule(
        rng, x...;
        interface_only=false,
        is_primitive::Bool=true,
        perf_flag::Symbol=:none,
        interp::Mooncake.MooncakeInterpreter=Mooncake.get_interpreter(),
        debug_mode::Bool=false,
        unsafe_perturb::Bool=false,
    )

Run standardised tests on the `rule` for `x`.
The first element of `x` should be the primal function to test, and each other element a
positional argument.
In most cases, elements of `x` can just be the primal values, and `randn_tangent` can be
relied upon to generate an appropriate tangent to test. Some notable exceptions exist
though, in partcular `Ptr`s. In this case, the argument for which `randn_tangent` cannot be
readily defined should be a `CoDual` containing the primal, and a _manually_ constructed
tangent field.

This function uses [`Mooncake.build_rrule`](@ref) to construct a rule. This will use an
`rrule!!` if one exists, and derive a rule otherwise.

# Arguments
- `rng::AbstractRNG`: a random number generator
- `x...`: the function (first element) and its arguments (the remainder)

# Keyword Arguments
- `interface_only::Bool=false`: test only that the interface is satisfied, without testing
    correctness. This should generally be set to `false` (the default value), and only
    enabled if the testing infrastructure is unable to test correctness for some reason
    e.g. the returned value of the function is a `Ptr`, and appropriate tangents cannot,
    therefore, be generated for it automatically.
- `is_primitive::Bool=true`: check whether the thing that you are testing has a hand-written
    `rrule!!`. This option is helpful if you are testing a new `rrule!!`, as it enables you
    to verify that your method of `is_primitive` has returned the correct value, and that
    you are actually testing a method of the `rrule!!` function -- a common mistake when
    authoring a new `rrule!!` is to implement `is_primitive` incorrectly and to accidentally
    wind up testing a rule which Mooncake has derived, as opposed to the one that you have
    written. If you are testing something for which you have not
    hand-written an `rrule!!`, or which you do not care whether it has a hand-written
    `rrule!!` or not, you should set it to `false`.
- `perf_flag::Symbol=:none`: the value of this symbol determines what kind of performance
    tests should be performed. By default, none are performed. If you believe that a rule
    should be allocation-free (iff the primal is allocation free), set this to `:allocs`. If
    you hand-write an `rrule!!` and believe that your test case should be type stable, set
    this to `:stability` (at present we cannot verify whether a derived rule is type stable
    for technical reasons). If you believe that a hand-written rule should be _both_
    allocation-free and type-stable, set this to `:stability_and_allocs`.
- `interp::Mooncake.MooncakeInterpreter=Mooncake.get_interpreter()`: the abstract
    interpreter to be used when testing this rule. The default should generally be used.
- `debug_mode::Bool=false`: whether or not the rule should be tested in debug mode.
    Typically this should be left at its default `false` value, but if you are finding that
    the tests are failing for a given rule, you may wish to temporarily set it to `true` in
    order to get access to additional information and automated testing.
- `unsafe_perturb::Bool=false`: value passed as the third argument to `_add_to_primal`.
    Should usually be left `false` -- consult the docstring for `_add_to_primal` for more
    info on when you might wish to set it to `true`.
"""
function test_rule(
    rng::AbstractRNG,
    x...;
    interface_only::Bool=false,
    is_primitive::Bool=true,
    perf_flag::Symbol=:none,
    interp::Mooncake.MooncakeInterpreter=Mooncake.get_interpreter(),
    debug_mode::Bool=false,
    unsafe_perturb::Bool=false,
)
    # Construct the rule.
    sig = _typeof(__get_primals(x))
    rule = Mooncake.build_rrule(interp, sig; debug_mode)

    # If something is primitive, then the rule should be `rrule!!`.
    is_primitive && @test rule == (debug_mode ? Mooncake.DebugRRule(rrule!!) : rrule!!)

    # Generate random tangents for anything that is not already a CoDual.
    x_x̄ = map(x -> if x isa CoDual
        x
    elseif interface_only
        uninit_codual(x)
    else
        zero_codual(x)
    end, x)

    # Test that the interface is basically satisfied (checks types / memory addresses).
    test_rrule_interface(x_x̄...; rule)

    # Test that answers are numerically correct / consistent.
    interface_only || test_rule_correctness(rng, x_x̄...; rule, unsafe_perturb)

    # Test the performance of the rule.
    test_rrule_performance(perf_flag, rule, x_x̄...)

    # Test the interface again, in order to verify that caching is working correctly.
    return test_rrule_interface(x_x̄...; rule=Mooncake.build_rrule(interp, sig; debug_mode))
end

function run_hand_written_rrule!!_test_cases(rng_ctor, v::Val)
    test_cases, memory =
        test_hook(Mooncake.generate_hand_written_rrule!!_test_cases, rng_ctor, v) do
            Mooncake.generate_hand_written_rrule!!_test_cases(rng_ctor, v)
        end
    GC.@preserve memory @testset "$f, $(_typeof(x))" for (
        interface_only, perf_flag, _, f, x...
    ) in test_cases

        test_rule(rng_ctor(123), f, x...; interface_only, perf_flag)
    end
end

function run_derived_rrule!!_test_cases(rng_ctor, v::Val)
    test_cases, memory =
        test_hook(Mooncake.generate_derived_rrule!!_test_cases, rng_ctor, v) do
            Mooncake.generate_derived_rrule!!_test_cases(rng_ctor, v)
        end
    GC.@preserve memory @testset "$f, $(typeof(x))" for (
        interface_only, perf_flag, _, f, x...
    ) in test_cases

        test_rule(rng_ctor(123), f, x...; interface_only, perf_flag, is_primitive=false)
    end
end

function run_rrule!!_test_cases(rng_ctor, v::Val)
    run_hand_written_rrule!!_test_cases(rng_ctor, v)
    return run_derived_rrule!!_test_cases(rng_ctor, v)
end

"""
    test_tangent(rng::AbstractRNG, p, T; interface_only=false, perf=true)

Like `test_tangent(rng, p)`, but also checks that `tangent_type(typeof(p)) == T`.
"""
function test_tangent(rng::AbstractRNG, p, T; interface_only=false, perf=true)
    test_tangent_type(typeof(p), T)
    test_tangent(rng, p; interface_only, perf)
    return nothing
end

"""
    test_tangent(rng::AbstractRNG, p; interface_only=false, perf=true)

Test that standard tangent-related functionality works for `p`.
"""
function test_tangent(rng::AbstractRNG, p; interface_only=false, perf=true)
    @nospecialize rng p
    test_tangent_interface(rng, p; interface_only)
    return perf && test_tangent_performance(rng, p)
end

"""
    is_foldable(f, types)::Bool

`true` if the effects inferred for the application of `f` to arguments of type `types`
indicate that the compiler believes such a call can be constant-folded.

See the docstrings for `Base.@infer_effects` and `Base.infer_effects` for more information
on the effects system in Julia.
"""
function is_foldable(f, types)::Bool
    effects = Base.infer_effects(f, types)
    tmp = VERSION > v"1.11" ? effects.noub == CC.ALWAYS_TRUE && effects.nortcall : true
    return effects.consistent == CC.ALWAYS_TRUE &&
           effects.effect_free == CC.ALWAYS_TRUE &&
           effects.terminates &&
           tmp
end

"""
    test_tangent_type(primal_type, expected_tangent_type)

Checks that `tangent_type(primal_type)` yields `expected_tangent_type`, and that everything
infers / optimises away, and that the effects are as expected.
"""
function test_tangent_type(primal_type::Type, expected_tangent_type::Type)

    # Verify tangent type returns the expected type.
    @test tangent_type(primal_type) == expected_tangent_type
    @test is_foldable(tangent_type, (Type{expected_tangent_type},))
    test_opt(tangent_type, Tuple{_typeof(primal_type)})
    return nothing
end

"""
    test_tangent_interface(rng::AbstractRNG, p; interface_only=false)

Verify that standard functionality for tangents runs, and is consistent. This function is
the defacto formal definition of the "tangent interface" -- if this function runs without
error for a given value of `p`, then that `p` satisfies the tangent interface.

# Extended Help

Verifies that the following functions are implemented correctly (as far as possible) for
`p` / its type, and its tangents / their type:
- [`Mooncake.tangent_type`](@ref)
- [`Mooncake.zero_tangent_internal`](@ref)
- [`Mooncake.randn_tangent_internal`](@ref)
- [`Mooncake.TestUtils.has_equal_data`](@ref)
- [`Mooncake.increment_internal!!`](@ref)
- [`Mooncake.set_to_zero_internal!!`](@ref)
- [`Mooncake._add_to_primal_internal`](@ref)
- [`Mooncake._diff_internal`](@ref)
- [`Mooncake._dot_internal`](@ref)
- [`Mooncake._scale_internal`](@ref)
- [`Mooncake.TestUtils.populate_address_map_internal`](@ref)

In conjunction with the functions tested by [`test_tangent_splitting`](@ref), these functions
constitute a complete set of functions which must be applicable to `p` in order to ensure
that it operates correctly in the context of reverse-mode AD. This list should be up to date
at any given point in time, but the best way to verify that you've implemented everything is
simply to run this function, and see whether it errors / produces a failing test.
"""
function test_tangent_interface(rng::AbstractRNG, p::P; interface_only=false) where {P}
    @nospecialize rng p
    return test_hook(test_tangent_interface, rng, p; interface_only) do
        _test_tangent_interface(rng, p; interface_only)
    end
end

function _test_tangent_interface(rng::AbstractRNG, p::P; interface_only=false) where {P}
    @nospecialize rng p

    # Define helpers which call internal methods directly. Doing this ensures that we know
    # that methods of the internal function have been implemented for the type we're
    # testing, rather than the user-facing versions. e.g. that a method of
    # `zero_tangent_internal` exists for `p`, rather than just `zero_tangent`.
    _zero_tangent(p) = Mooncake.zero_tangent_internal(p, IdDict())
    _randn_tangent(rng, p) = Mooncake.randn_tangent_internal(rng, p, IdDict())
    _increment!!(x, y) = Mooncake.increment_internal!!(IdDict{Any,Bool}(), x, y)
    _set_to_zero!!(t) = Mooncake.set_to_zero_internal!!(Vector{UInt}(), t)
    function __add_to_primal(p, t, unsafe::Bool)
        return Mooncake._add_to_primal_internal(IdDict{Any,Any}(), p, t, unsafe)
    end
    __diff(p, t) = Mooncake._diff_internal(IdDict{Any,Any}(), p, t)
    __dot(t, s) = Mooncake._dot_internal(IdDict{Any,Any}(), t, s)
    __scale(a::Float64, t) = Mooncake._scale_internal(IdDict{Any,Any}(), a, t)
    _populate_address_map(p, t) = populate_address_map_internal(AddressMap(), p, t)

    # Check that tangent_type returns a `Type`.
    T = tangent_type(P)
    @test T isa Type

    # Check that `zero_tangent_internal` runs and produces something of the correct type.
    z = _zero_tangent(p)
    @test z isa T

    # Check that `randn_tangent_internal` runs and produces something of the correct type.
    t = _randn_tangent(rng, p)
    @test t isa T

    # Check that we can compare the values of primals and tangents.
    function test_equality_comparison(x)
        @nospecialize x
        @test has_equal_data(x, x) isa Bool
        @test has_equal_data_up_to_undefs(x, x) isa Bool
        @test has_equal_data(x, x)
        @test has_equal_data_up_to_undefs(x, x)
    end
    test_equality_comparison(p)
    test_equality_comparison(t)

    # Check that `tangent_type` is performant.
    test_tangent_type(P, T)

    # Check that zero_tangent isn't obviously non-deterministic.
    @test has_equal_data(z, _zero_tangent(p))

    # Check that ismutabletype(P) => ismutabletype(T).
    if ismutabletype(P) && !(T == NoTangent)
        @test ismutabletype(T)
    end

    # Verify z is zero via its action on t.
    zc = deepcopy([z])[1]
    tc = deepcopy([t])[1]
    @test has_equal_data(@inferred(_increment!!(zc, zc)), zc)
    @test has_equal_data(_increment!!(zc, tc), tc)
    @test has_equal_data(_increment!!(tc, zc), tc)

    # increment!! preserves types.
    @test _increment!!(zc, zc) isa T
    @test _increment!!(zc, tc) isa T
    @test _increment!!(tc, zc) isa T

    # The output of `increment!!` for a mutable type must have the property that the first
    # argument === the returned value.
    if ismutabletype(P)
        @test _increment!!(zc, zc) === zc
        @test _increment!!(tc, zc) === tc
        @test _increment!!(zc, tc) === zc
        @test _increment!!(tc, tc) === tc
    end

    # If t isn't the zero element, then adding it to itself must change its value.
    if !has_equal_data(t, z) && !ismutabletype(P)
        tc′ = _increment!!(tc, tc)
        @test tc === tc′ || !has_equal_data(tc′, tc)
    end

    # Setting to zero equals zero.
    @test has_equal_data(_set_to_zero!!(tc), z)
    @test has_equal_data(_set_to_zero!!(tc), z)
    if ismutabletype(P)
        @test _set_to_zero!!(tc) === tc
    end

    z = _zero_tangent(p)
    r = _randn_tangent(rng, p)

    # Check set_tangent_field if mutable.
    t isa MutableTangent && test_set_tangent_field!_correctness(deepcopy(t), deepcopy(z))

    # Verify that operations required for finite difference testing to run, and produce the
    # correct output type.
    @test __add_to_primal(p, t, true) isa P
    @test __diff(p, p) isa T
    @test __dot(t, t) isa Float64
    @test __scale(11.0, t) isa T
    @test _populate_address_map(p, t) isa AddressMap

    # Run some basic numerical sanity checks on the output the functions required for finite
    # difference testing. These are necessary but insufficient conditions.
    if !interface_only
        @test has_equal_data(__add_to_primal(p, z, true), p)
        if !has_equal_data(z, r)
            @test !has_equal_data(__add_to_primal(p, r, true), p)
        end
        @test has_equal_data(__diff(p, p), _zero_tangent(p))
    end
    @test __dot(t, t) >= 0.0
    @test __dot(t, _zero_tangent(p)) == 0.0
    @test __dot(t, _increment!!(deepcopy(t), t)) ≈ 2 * __dot(t, t)
    @test has_equal_data(__scale(1.0, t), t)
    @test has_equal_data(__scale(2.0, t), _increment!!(deepcopy(t), t))
end

# Helper used in `test_tangent_interface`.
function test_set_tangent_field!_correctness(t1::T, t2::T) where {T<:MutableTangent}
    Tfields = _typeof(t1.fields)
    for n in 1:fieldcount(Tfields)
        !Mooncake.is_init(t2.fields[n]) && continue
        v = get_tangent_field(t2, n)

        # Int form.
        v′ = Mooncake.set_tangent_field!(t1, n, v)
        @test v′ === v
        @test Mooncake.get_tangent_field(t1, n) === v

        # Symbol form.
        s = fieldname(Tfields, n)
        g = Mooncake.set_tangent_field!(t1, s, v)
        @test g === v
        @test Mooncake.get_tangent_field(t1, n) === v
    end
end

function check_allocs(f, x...)
    test_hook(check_allocs, f, x...) do
        check_allocs_internal(Shim(), f, x...)
    end
end
function check_allocs_internal(::Any, f::F, x::Vararg{Any,N}) where {F,N}
    throw(error("Load AllocCheck.jl to use this functionality."))
end

"""
    test_tangent_performance(rng::AbstractRNG, p::P) where {P}

Runs a variety of performance-related tests on tangents. These tests constitute a set of
necessary conditions for good overall performance.

The performance model in a few cases is a little bit complicated, because it depends on
various properties of the type in question (is it mutable, are its fields mutable, are all
of its fields necessarily defined, etc), so the source code should be consulted for precise
details.

*Note:* this function assumes that the tangent interface is implemented correctly for `p`.
To verify that this is the case, ensure that all tests in `test_tangent_interface` pass.
"""
function test_tangent_performance(rng::AbstractRNG, p::P) where {P}
    return test_hook(test_tangent_performance, rng, p) do
        _test_tangent_performance(rng, p)
    end
end

function _test_tangent_performance(rng::AbstractRNG, p::P) where {P}
    # Should definitely infer, because tangent type must be known statically from primal.
    z = @inferred zero_tangent(p)
    t = @inferred randn_tangent(rng, p)

    # Computing the tangent type must always be type stable and allocation-free.
    @inferred tangent_type(P)
    @test count_allocs(tangent_type, P) == 0

    # Check there are no allocations when there ought not to be.
    if !__tangent_generation_should_allocate(P)
        test_opt(Tuple{typeof(zero_tangent),P})
        check_allocs(Mooncake.zero_tangent, p)
        test_opt(Tuple{typeof(randn_tangent),Xoshiro,P})
        check_allocs(Mooncake.randn_tangent, rng, p)
    end

    # `increment!!` should always infer.
    @inferred increment!!(t, z)
    @inferred increment!!(z, t)
    @inferred increment!!(t, t)
    @inferred increment!!(z, z)

    # Unfortunately, `increment!!` does occassionally allocate at the minute due to the
    # way we're handling partial initialisation. Hopefully this will change in the future.
    __increment_should_allocate(P) || test_allocations(t, z)

    # set_tangent_field! should never allocate.
    t isa MutableTangent && test_set_tangent_field!_performance(t, z)
    return t isa Union{MutableTangent,Tangent} && test_get_tangent_field_performance(t)
end

function test_allocations(t::T, z::T) where {T}
    check_allocs(increment_internal!!, Mooncake.NoCache(), t, t)
    check_allocs(increment_internal!!, Mooncake.NoCache(), t, z)
    check_allocs(increment_internal!!, Mooncake.NoCache(), z, t)
    check_allocs(increment_internal!!, Mooncake.NoCache(), z, z)
    return nothing
end

_set_tangent_field!(x, ::Val{i}, v) where {i} = set_tangent_field!(x, i, v)
_get_tangent_field(x, ::Val{i}) where {i} = get_tangent_field(x, i)

function test_set_tangent_field!_performance(t1::T, t2::T) where {V,T<:MutableTangent{V}}
    for n in 1:fieldcount(V)
        !is_init(t2.fields[n]) && continue
        v = get_tangent_field(t2, n)

        # Int mode.
        _set_tangent_field!(t1, Val(n), v)
        report_opt(Tuple{typeof(_set_tangent_field!),typeof(t1),Val{n},typeof(v)})

        if all(n -> !(fieldtype(V, n) <: Mooncake.PossiblyUninitTangent), 1:fieldcount(V))
            i = Val(n)
            _set_tangent_field!(t1, i, v)
            @test count_allocs(_set_tangent_field!, t1, i, v) == 0
        end

        # Symbol mode.
        s = Val(fieldname(V, n))
        @inferred _set_tangent_field!(t1, s, v)
        report_opt(Tuple{typeof(_set_tangent_field!),typeof(t1),typeof(s),typeof(v)})

        if all(n -> !(fieldtype(V, n) <: Mooncake.PossiblyUninitTangent), 1:fieldcount(V))
            _set_tangent_field!(t1, s, v)
            @test count_allocs(_set_tangent_field!, t1, s, v) == 0
        end
    end
end

function test_get_tangent_field_performance(t::Union{MutableTangent,Tangent})
    V = Mooncake._typeof(t.fields)
    for n in 1:fieldcount(V)
        !is_init(t.fields[n]) && continue
        Tfield = fieldtype(Mooncake.fields_type(Mooncake._typeof(t)), n)
        !__is_completely_stable_type(Tfield) && continue

        # Int mode.
        i = Val(n)
        report_opt(Tuple{typeof(_get_tangent_field),typeof(t),typeof(i)})
        @inferred _get_tangent_field(t, i)
        @test count_allocs(_get_tangent_field, t, i) == 0

        # Symbol mode.
        s = Val(fieldname(V, n))
        report_opt(Tuple{typeof(_get_tangent_field),typeof(t),typeof(s)})
        @inferred _get_tangent_field(t, s)
        @test count_allocs(_get_tangent_field, t, s) == 0
    end
end

# Function barrier to ensure inference in value types.
function count_allocs(f::F, x::Vararg{Any,N}) where {F,N}
    test_hook(count_allocs, f, x...) do
        @allocations f(x...)
    end
end

# Returns true if both `zero_tangent` and `randn_tangent` should allocate when run on
# an object of type `P`.
function __tangent_generation_should_allocate(::Type{P}) where {P}
    (!isconcretetype(P) || isabstracttype(P)) && return true
    (fieldcount(P) == 0 && !ismutabletype(P)) && return false
    return ismutabletype(P) || any(__tangent_generation_should_allocate, fieldtypes(P))
end

__tangent_generation_should_allocate(::Type{P}) where {P<:Array} = true

function __increment_should_allocate(::Type{P}) where {P}
    return any(tt -> tt <: PossiblyUninitTangent, Mooncake.tangent_field_types(P))
end
__increment_should_allocate(::Type{Core.SimpleVector}) = true
__increment_should_allocate(::Type{<:Array{Any}}) = true

function __is_completely_stable_type(::Type{P}) where {P}
    (!isconcretetype(P) || isabstracttype(P)) && return false
    isprimitivetype(P) && return true
    return all(__is_completely_stable_type, fieldtypes(P))
end

function test_equality_comparison(x)
    @nospecialize x

    # Check that the internal methods have been implemented.
    function _has_equal_data(x, y)
        return has_equal_data_internal(x, y, true, Dict{Tuple{UInt,UInt},Bool}())
    end
    function _has_equal_data_up_to_undefs(x, y)
        return has_equal_data_internal(x, y, false, Dict{Tuple{UInt,UInt},Bool}())
    end

    @test _has_equal_data(x, x) isa Bool
    @test _has_equal_data_up_to_undefs(x, x) isa Bool
    @test _has_equal_data(x, x)
    @test _has_equal_data_up_to_undefs(x, x)
end

"""
    test_tangent_splitting(rng::AbstractRNG, p::P; test_opt_flag=true) where {P}

Verify that tangent splitting functionality associated to primal `p` works correctly.
Ensure that [`test_tangent_interface`](@ref) runs for `p` before running these tests.
`test_opt_flag` controls whether to run JET-based checks. 

# Extended Help

 Verifies that the following functionality work correctly for `p` / its type / tangents:
- [`Mooncake.fdata_type`](@ref)
- [`Mooncake.rdata_type`](@ref)
- [`Mooncake.fdata`](@ref)
- [`Mooncake.rdata`](@ref)
- [`Mooncake.uninit_fdata`](@ref)
- [`Mooncake.tangent_type`](@ref) (binary method)
- [`Mooncake.tangent`](@ref) (binary method)
"""
function test_tangent_splitting(rng::AbstractRNG, p::P; test_opt_flag=true) where {P}
    return test_hook(test_tangent_splitting, rng, p; test_opt_flag) do
        _test_tangent_splitting_internal(rng, p; test_opt_flag)
    end
end

function _test_tangent_splitting_internal(
    rng::AbstractRNG, p::P; test_opt_flag=true
) where {P}
    # Check that fdata_type and rdata_type run and produce types.
    T = tangent_type(P)
    F = Mooncake.fdata_type(T)
    @test F isa Type
    check_allocs(Mooncake.fdata_type, T)
    R = Mooncake.rdata_type(T)
    @test R isa Type
    check_allocs(Mooncake.rdata_type, T)

    # Check that fdata and rdata produce the correct types.
    t = randn_tangent(rng, p)
    f = Mooncake.fdata(t)
    @test f isa F
    r = Mooncake.rdata(t)
    @test r isa R

    # Check that fdata / rdata validation functionality doesn't error on valid fdata / rdata
    # and does error on obviously wrong fdata / rdata.
    @test verify_fdata_type(P, F) === nothing
    @test verify_rdata_type(P, R) === nothing
    @test verify_fdata_value(p, f) === nothing
    @test verify_rdata_value(p, r) === nothing
    @test_throws InvalidFDataException verify_fdata_type(P, Int)
    @test_throws InvalidRDataException verify_rdata_type(P, Int)
    @test_throws InvalidFDataException verify_fdata_value(p, 0)
    @test_throws InvalidRDataException verify_rdata_value(p, 0)

    # Check that uninit_fdata yields data of the correct type.
    @test uninit_fdata(p) isa F

    # Compute the tangent type associated to `F` and `R`, and check it is equal to `T`.
    @test tangent_type(F, R) == T
    @test is_foldable(tangent_type, (Type{F}, Type{R}))

    # Check that combining f and r yields a tangent of the correct type and value.
    t_combined = Mooncake.tangent(f, r)
    @test t_combined isa T
    @test t_combined === t

    # Check that pulling out `f` and `r` from `t_combined` yields the correct values.
    @test Mooncake.fdata(t_combined) === f
    @test Mooncake.rdata(t_combined) === r

    # Test that `zero_rdata` produces valid reverse data.
    @test zero_rdata(p) isa R

    # Check that constructing a zero tangent from reverse data yields the original tangent.
    z = zero_tangent(p)
    f_z = Mooncake.fdata(z)
    @test f_z isa Mooncake.fdata_type(T)
    z_new = zero_tangent(p, f_z)
    @test z_new isa tangent_type(P)
    @test z_new === z

    # Query whether or not the rdata type can be built given only the primal type.
    can_make_zero = @inferred can_produce_zero_rdata_from_type(P)

    # Check that when the zero element is asked from the primal type alone, the result is
    # either an instance of R _or_ a `CannotProduceZeroRDataFromType`.
    test_opt_flag && test_opt(zero_rdata_from_type, Tuple{Type{P}})
    rzero_from_type = @inferred zero_rdata_from_type(P)
    @test rzero_from_type isa R || rzero_from_type isa CannotProduceZeroRDataFromType
    @test can_make_zero != isa(rzero_from_type, CannotProduceZeroRDataFromType)

    # Check that we can produce a lazy zero rdata, and that it has the correct type.
    test_opt_flag && test_opt(lazy_zero_rdata, Tuple{P})
    lazy_rzero = @inferred lazy_zero_rdata(p)
    @test instantiate(lazy_rzero) isa R

    # Check incrementing the fdata component of a tangnet yields the correct type.
    @test increment!!(f, f) isa F

    # Check incrementing the rdata component of a tangent yields the correct type.
    @test increment_rdata!!(t, r) isa T
end

"""
    test_rule_and_type_interactions(rng::AbstractRNG, p)

Check that a collection of standard functions for which we _ought_ to have a working rrule
for `p` work, and produce the correct answer. For example, the `rrule!!` for `typeof` should
work correctly on any type, we should have a working rule for `getfield` for any
struct-type, and we should have a rule for `setfield!` for any mutable struct type.
See extended help for more info.

# Extended Help

The purpose of this test is to ensure that, for any given `p`, the full range of primitive
functions that _ought_ to work on it, do indeed work on it.

This is one part of the interface where some care _might_ be required. If, for some reason,
it should _never_ be the case that e.g. for a particular `p`, `getfield` should be called,
then it may make no sense at all to run these tests. In such cases, the author of the type
is responsible for knowing what they are doing. Please open an issue to discuss for your
type if you are at all unsure what to do.

When defining a custom tangent type for `P`, the functions that you will need to pay
attention to writing rules for are
- [`Mooncake._new_`](@ref)
- [`Mooncake.lgetfield`](@ref)
- [`Mooncake.lsetfield!`](@ref)

In all cases, you may wish to consult the current implementations of `rrule!!` for these
functions for inspiration regarding how you might implement them for your type.
"""
function test_rule_and_type_interactions(rng::AbstractRNG, p::P) where {P}
    @nospecialize rng p

    # Generate standard test cases.
    fs = if ismutabletype(P)
        functions_for_mutable_structs()
    elseif isstructtype(P)
        functions_for_structs()
    else
        functions_for_all_types()
    end

    # Run standardised tests for all functions.
    @testset "$f" for f in fs
        arg_sets = generate_args(f, p)
        @testset for args in arg_sets
            test_rule(
                rng,
                f,
                args...;
                interface_only=true,
                is_primitive=true,
                perf_flag=:none,
                interp=Mooncake.get_interpreter(),
            )
        end
    end
end

#
# Test that some basic operations work on a given type.
#

generate_args(::typeof(===), x) = [(x, 0.0), (1.0, x)]
function generate_args(::typeof(Core.ifelse), x)
    return [(true, x, 0.0), (false, x, 0.0), (true, 0.0, x), (false, 0.0, x)]
end
generate_args(::typeof(Core.sizeof), x) = [(x,)]
generate_args(::typeof(Core.svec), x) = [(x,), (x, x)]
function generate_args(::typeof(getfield), x)
    syms = filter(f -> isdefined(x, f), fieldnames(_typeof(x)))
    fs = vcat(syms..., eachindex(syms)...)
    return vcat(map(n -> (x, n), fs), map(n -> (x, n, :not_atomic), fs))
end
function generate_args(::typeof(lgetfield), x)
    syms = filter(f -> isdefined(x, f), fieldnames(_typeof(x)))
    fs = vcat(syms..., eachindex(syms)...)
    return vcat(map(n -> (x, Val(n)), fs), map(n -> (x, Val(n), Val(:not_atomic)), fs))
end

_new_excluded(::Type) = false
_new_excluded(::Type{<:Union{String}}) = true

@static if VERSION < v"1.11-"
    # Prior to 1.11, Arrays are special objects, with special constructors that don't
    # involve calling the `:new` instruction. From 1.11 onwards, they behave more like
    # regular mutable composite types, so calling `_new_` becomes meaningful.
    _new_excluded(::Type{<:Array}) = true
else
    # Memory and MemoryRef appeared in 1.11. Neither are constructed in the usual manner
    # via the `:new` instruction, but rather by a variety of built-ins and `ccall`s.
    # Consequently, it does not make sense to call `_new_` on them -- while this _can_ be
    # made to work, it typically yields segfaults in very short order, and I _believe_ it
    # should never occur in practice.
    _new_excluded(::Type{<:Union{Memory,MemoryRef}}) = true
end

function generate_args(::typeof(Mooncake._new_), x)
    _new_excluded(_typeof(x)) && return []
    syms = filter(f -> isdefined(x, f), fieldnames(_typeof(x)))
    field_values = map(sym -> getfield(x, sym), syms)
    return [(_typeof(x), field_values...)]
end
generate_args(::typeof(isa), x) = [(x, Float64), (x, Int), (x, _typeof(x))]
function generate_args(::typeof(setfield!), x)
    names = filter(fieldnames(_typeof(x))) do f
        return !isconst(_typeof(x), f) && isdefined(x, f)
    end
    return map(n -> (x, n, getfield(x, n)), vcat(names..., eachindex(names)...))
end
function generate_args(::typeof(lsetfield!), x)
    names = filter(fieldnames(_typeof(x))) do f
        return !isconst(_typeof(x), f) && isdefined(x, f)
    end
    return map(n -> (x, Val(n), getfield(x, n)), vcat(names..., eachindex(names)...))
end
generate_args(::typeof(tuple), x) = [(x,), (x, x), (x, x, x)]
generate_args(::typeof(typeassert), x) = [(x, _typeof(x))]
generate_args(::typeof(typeof), x) = [(x,)]

function functions_for_all_types()
    return [===, Core.ifelse, Core.sizeof, isa, tuple, typeassert, typeof]
end

function functions_for_structs()
    return vcat(functions_for_all_types(), [getfield, lgetfield, Mooncake._new_])
end

function functions_for_mutable_structs()
    return vcat(
        functions_for_structs(),
        [setfield!, lsetfield!],# modifyfield!, replacefield!, swapfield!],
    )
end

"""
    test_data(rng::AbstractRNG, p::P)

Verify that all tangent / fdata / rdata functionality work properly for `p`. Furthermore,
verify that all primitives listed in `TestUtils.test_rule_and_type_interactions` work
correctly on `p`. This functionality is particularly useful if you are writing your own
custom tangent / fdata / rdata types and want to be confident that you have implemented the
functionality that you need in order to make these custom types work with all the rules
written in Mooncake itself.

You should consult the docstrings for [`test_tangent_interface`](@ref),
[`test_tangent_splitting`](@ref), and [`test_rule_and_type_interactions`](@ref), in order to
see what is required to satisfy the full tangent interface for `p`.
"""
function test_data(rng::AbstractRNG, p::P; interface_only=false) where {P}
    test_tangent_interface(rng, p; interface_only)
    test_tangent_splitting(rng, p)
    test_rule_and_type_interactions(rng, p)
    return nothing
end

end
