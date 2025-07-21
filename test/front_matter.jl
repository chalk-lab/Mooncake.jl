using Aqua,
    BenchmarkTools,
    DiffRules,
    JET,
    JuliaFormatter,
    LinearAlgebra,
    Random,
    StableRNGs,
    Mooncake,
    Test

using AllocCheck: AllocCheck # load to enable testing functionality

using ChainRulesCore: ChainRulesCore

using Base: unsafe_load, pointer_from_objref, IEEEFloat, TwicePrecision
using Base.Iterators: product
using Core:
    bitcast, svec, ReturnNode, PhiNode, PiNode, GotoIfNot, GotoNode, SSAValue, Argument
using Core.Intrinsics: pointerref, pointerset
using FunctionWrappers: FunctionWrapper

using Mooncake

using Mooncake:
    primal,
    tangent,
    randn_tangent,
    increment!!,
    NoTangent,
    Tangent,
    MutableTangent,
    PossiblyUninitTangent,
    set_to_zero!!,
    tangent_type,
    zero_tangent,
    _scale,
    _add_to_primal,
    _diff,
    _dot,
    zero_dual,
    zero_codual,
    codual_type,
    rrule!!,
    build_rrule,
    value_and_gradient!!,
    value_and_pullback!!,
    NoFData,
    NoRData,
    fdata_type,
    rdata_type,
    fdata,
    rdata,
    get_interpreter,
    Mode,
    ForwardMode,
    ReverseMode

using Mooncake:
    CC,
    IntrinsicsWrappers,
    TestUtils,
    TestResources,
    CoDual,
    DefaultCtx,
    rrule!!,
    lgetfield,
    lsetfield!,
    Stack,
    _typeof,
    BBCode,
    ID,
    IDPhiNode,
    IDGotoNode,
    IDGotoIfNot,
    BBlock,
    make_ad_stmts!,
    ADStmtInfo,
    ad_stmt_info,
    ADInfo,
    SharedDataPairs,
    increment_field!!,
    NoFData,
    NoRData,
    zero_fcodual,
    zero_like_rdata_from_type,
    zero_rdata,
    instantiate,
    LazyZeroRData,
    lazy_zero_rdata,
    new_inst,
    characterise_unique_predecessor_blocks,
    NoPullback,
    characterise_used_ids,
    InvalidFDataException,
    InvalidRDataException,
    verify_fdata_value,
    verify_rdata_value,
    is_primitive,
    MinimalCtx,
    stmt,
    can_produce_zero_rdata_from_type,
    zero_rdata_from_type,
    CannotProduceZeroRDataFromType

using .TestUtils:
    test_rule,
    has_equal_data,
    AddressMap,
    populate_address_map_internal,
    populate_address_map,
    test_tangent,
    check_allocs

using .TestResources:
    TypeStableMutableStruct,
    StructFoo,
    MutableFoo,
    TypeUnstableStruct,
    TypeUnstableStruct2,
    TypeUnstableMutableStruct,
    TypeUnstableMutableStruct2,
    make_circular_reference_struct,
    make_indirect_circular_reference_struct,
    make_circular_reference_array,
    make_indirect_circular_reference_array

# The integration tests take ages to run, so we split them up. CI sets up two jobs -- the
# "basic" group runs test that, when passed, _ought_ to imply correctness of the entire
# scheme. The "extended" group runs a large battery of tests that should pick up on anything
# that has been missed in the "basic" group. As a rule, if the "basic" group passes, but the
# "extended" group fails, there are clearly new tests that need to be added to the "basic"
# group.
const test_group = get(ENV, "TEST_GROUP", "basic")

sr(n::Int) = StableRNG(n)

# This is annoying and hacky and should be improved.
if isempty(Mooncake.TestTypes.PRIMALS)
    Mooncake.TestTypes.generate_primals()
end
