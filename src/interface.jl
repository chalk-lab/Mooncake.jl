#
# Shared types, constants, and errors
#

struct ValueAndGradientReturnTypeError <: Exception
    msg::String
end

@noinline function throw_val_and_grad_ret_type_error(y; caller, cache, eltypes::Type)
    throw(
        ValueAndGradientReturnTypeError(
            "$(_caller_label(caller, cache)) requires the primal `f(x...)` to return a " *
            "subtype of $eltypes. Instead, found a value of type $(typeof(y)).",
        ),
    )
end

@inline function _check_scalar_output(
    y; caller=nothing, cache=nothing, eltypes::Type=IEEEFloat
)
    y isa eltypes ||
        throw_val_and_grad_ret_type_error(y; caller=caller, cache=cache, eltypes=eltypes)
    return nothing
end

struct ValueAndPullbackReturnTypeError <: Exception
    msg::String
end

function Base.showerror(io::IO, err::ValueAndGradientReturnTypeError)
    _print_boxed_error(io, split("ValueAndGradientReturnTypeError: $(err.msg)", '\n'))
end

function Base.showerror(io::IO, err::ValueAndPullbackReturnTypeError)
    _print_boxed_error(io, split("ValueAndPullbackReturnTypeError: $(err.msg)", '\n'))
end

function throw_ptr_in_output_error(y)
    throw(
        ValueAndPullbackReturnTypeError(
            "Found a value of type $(typeof(y)) in output, but output is not permitted " *
            "to be or contain a pointer. This is because the amount of memory to which " *
            "it refers is unknown, therefore Mooncake.jl is unable to allocate " *
            "appropriate memory for its gradients.",
        ),
    )
end

function throw_circular_reference_or_alias_error(y)
    throw(
        ValueAndPullbackReturnTypeError(
            "Object with address $(objectid(y)) and type $(typeof(y)) appears more than " *
            "once. Output cannot contain Circular references or aliases",
        ),
    )
end

# Cache parameters keep the rule, reusable buffers, and input/output signatures concrete.
# HVPCache is shared by HVP and Hessian calls.
struct Cache{Trule,Ty_cache,Ttangents<:Tuple,Tdests,Tȳ_cache,TIS<:Tuple,TOS}
    rule::Trule
    # Cache for function output; **primal** type for y.
    y_cache::Ty_cache
    # Cache for internal gradient representation; **tangent** type for (f, x...)
    tangents::Ttangents
    # Pre-allocated friendly-tangent dest tree for (f, x...), built by
    # map(friendly_tangent_cache, fx).  `nothing` when friendly_tangents=false.
    dests::Tdests
    # Cache to convert from friendly to internal representation of ȳ.
    # Tangent type for y, i.e. this is a **tangent** type for y.
    ȳ_cache::Tȳ_cache
    # Top-level type/size signature for (f, x...), used to reject cache misuse early.
    input_types_and_sizes::TIS
    # Top-level type/size signature for y = f(x...).
    output_type_and_size::TOS
end

# `input_types_and_sizes` leads with the entry for `f`; the displayed inputs are `x...`.
_cache_input_types_and_sizes(cache) = Base.tail(getfield(cache, :input_types_and_sizes))

# `sizestr` distinguishes a known array size from the type-only summary used by `show`.
function _cache_summary(::Type{T}, sizestr) where {T}
    T === Any && return "unknown"
    kind = if T <: IEEEFloat || T <: Complex{<:IEEEFloat}
        "scalar"
    elseif T <: AbstractArray
        sizestr
    elseif T <: NamedTuple
        "named tuple"
    elseif T <: Tuple
        "tuple"
    elseif T <: Function
        "function"
    elseif fieldcount(T) > 0 || Base.ismutabletype(T)
        "struct"
    else
        "value"
    end
    return "$(T) ($kind)"
end

function _cache_show_fields(cache::Cache)
    return (;
        mode=:reverse,
        friendly_tangents=(!isnothing(getfield(cache, :dests))),
        inputs=length(_cache_input_types_and_sizes(cache)),
    )
end

const _BuiltinArrays = @static VERSION >= v"1.11" ? Union{Array,Memory} : Array

struct GradientChunkSize
    width::Int
    auto::Bool
end

struct FCache{R,IT<:Union{Nothing,Tuple},FG,GW,CF,S<:Tuple,IS,GS,JB}
    single_rule::R
    input_tangents::IT
    friendly_gradients::FG
    gradient_workspace::GW
    gradient_chunk_size::GradientChunkSize
    chunk_rule::CF
    input_types_and_sizes::S
    # Snapshot x... (not the potentially uncopyable f) once per call and restore in place:
    # forward slots alias user storage, which a mutating rule must leave unchanged.
    input_snapshot::IS
    # Cache-owned gradient seeds: a tuple for same-eltype float vectors,
    # StructuredGradSeed for array-backed leaves, or IsbitsGradSeed for real scalars.
    # `nothing` selects the generic chunked sweep; structured seeds admit real/complex
    # arrays and require nondifferentiable leaves to be isbits.
    gradient_seed::GS
    # Whether the prepared inputs share differentiable storage across positions (`f.v === x`,
    # say), which the gradient sweeps cannot represent and so refuse. See `_inputs_share_storage`.
    inputs_share_storage::Bool
    # Packable single-float-vector Jacobian buffer, sized on first use (output size is lazy).
    # Returned storage is reused and overwritten; other shapes use `nothing`.
    jacobian_buffer::JB
end

function _cache_show_fields(cache::FCache)
    chunk = getfield(cache, :gradient_chunk_size)
    chunk_size = chunk.width
    return (;
        mode=:forward,
        friendly_tangents=(!isnothing(getfield(cache, :input_tangents))),
        chunk=(!isnothing(getfield(cache, :chunk_rule))),
        chunk_size=if chunk.auto
            "$(chunk_size) (auto)"
        else
            chunk_size
        end,
        inputs=length(_cache_input_types_and_sizes(cache)),
    )
end

# Encode `T` in the type so generated cache checks avoid runtime `jl_types_equal`.
struct TypeAndSize{T,S}
    size::S
end

TypeAndSize(::Type{T}, s::S) where {T,S} = TypeAndSize{T,S}(s)

function _cache_summary(type_and_size::TypeAndSize{T}) where {T}
    _cache_summary(T, "size $(type_and_size.size)")
end

const _MAX_CHUNK_WIDTH = 8

struct PreparedCacheError <: Exception
    msg::String
end

function Base.showerror(io::IO, err::PreparedCacheError)
    _print_boxed_error(io, split("PreparedCacheError:\n$(err.msg)", '\n'))
end

function _throw_prepared_cache_mismatch_error(kind::Symbol, i::Int, expected, got)
    label = i == 1 ? "`f`" : "`x$(i - 1)`"
    msg = if kind === :arity
        "Cached autodiff call expected $(expected) total arguments `(f, x...)`, got " *
        "$(got).\nPrepared pullback, gradient, derivative, HVP, and Hessian caches must " *
        "be reused with the same top-level argument structure they were prepared with."
    elseif kind === :type
        "Cached autodiff call has a type mismatch for $label.\nExpected top-level type: " *
        "$expected\nGot top-level type: $got\nPrepared pullback, gradient, derivative, " *
        "HVP, and Hessian caches must be reused with the same top-level argument types " *
        "they were prepared with."
    else
        "Cached autodiff call has a size mismatch for $label.\nExpected top-level size: " *
        "$expected\nGot top-level size: $got\nPrepared pullback, gradient, derivative, " *
        "HVP, and Hessian caches must be reused with the same top-level array sizes they " *
        "were prepared with."
    end
    throw(PreparedCacheError(msg))
end

function _throw_prepared_cache_aliasing_error(li::String, lj::String, aliased_now::Bool)
    # "one storage" rather than "the same object": an `Array` and its backing `Memory` are never
    # `===` yet are one storage, and that pair is exactly what `_shares_storage` added.
    what = if aliased_now
        "share one storage now but were separate"
    else
        "are separate now but shared one storage"
    end
    throw(
        PreparedCacheError(
            "Cached autodiff call has an aliasing mismatch: $li and $lj $what when the cache " *
            "was prepared.\nA prepared cache holds one tangent buffer per argument, so the " *
            "aliasing among arguments is part of the shape it was prepared for: reusing it with " *
            "different aliasing writes into the wrong buffers and silently returns the wrong " *
            "answer. Prepare a separate cache for this argument aliasing.",
        ),
    )
end
"""
    _mutable_tangent_paths(T::Type)

Field paths from a tangent of type `T` to the mutable objects reachable through immutable
fixed-arity containers, as tuples of field indices (`()` when `T` is itself mutable). Only `Tuple`
and `NamedTuple` are walked, because their tangents mirror the primal element-wise, so one path
indexes both. The walk is complete: a truncated one silently accepts calls whose aliasing differs
from the prepared shape past the cut, which is a wrong gradient rather than a missed diagnostic.
"""
function _mutable_tangent_paths(@nospecialize(T::Type), path=(), out=Vector{Any}())
    if Base.ismutabletype(T)
        push!(out, path)
    elseif (T <: Tuple || T <: NamedTuple) && isconcretetype(T)
        for (k, FT) in enumerate(fieldtypes(T))
            _mutable_tangent_paths(FT, (path..., k), out)
        end
    end
    return out
end

# `tangents[i][p1][p2]...` as an expression, and the same position as a label for the error.
_path_expr(base::Symbol, i::Int, path) = foldl((e, k) -> :($e[$k]), path; init=:($base[$i]))
function _alias_label(i::Int, path)
    return (i == 1 ? "`f" : "`x$(i - 1)") * prod(k -> "[$k]", path; init="") * "`"
end

# Use the same storage identity for primals and tangents: an Array and its backing
# Memory are distinct objects but share accumulation storage, as do reshaped arrays.
@inline _storage_id(@nospecialize(x)) = x
@static if VERSION >= v"1.11-rc4"  # 1.10 has no `Memory`, and its tangents do not share one.
    @inline _storage_id(x::Array) = getfield(x, :ref).mem
    @inline _storage_id(x::Memory) = x
end
@inline _shares_storage(@nospecialize(x), @nospecialize(y)) =
    _storage_id(x) === _storage_id(y)

# The positions both aliasing emitters compare, as `(argument index, path)`. `leafwise` walks into
# immutable containers for the tangent check; the primal check takes top-level mutables only.
# Shared so the two cannot drift apart in what they enumerate.
function _aliasable_positions(@nospecialize(T::Type), leafwise::Bool)
    leaves = Tuple{Int,Any}[]
    for (i, P) in enumerate(T.parameters)
        if leafwise
            for path in _mutable_tangent_paths(P)
                push!(leaves, (i, path))
            end
        elseif Base.ismutabletype(P)
            push!(leaves, (i, ()))
        end
    end
    return leaves
end

# Compare mutable tangent leaves only: immutable `===` is value equality, not aliasing.
# Find nested Tuple/NamedTuple leaves from their types to avoid a per-call graph walk.
# Variable-arity containers and struct fields remain unchecked; see `known_limitations.md`.
# `bidirectional` requires both partitions to agree (reverse); forward only refuses
# shared tangent storage for distinct primals.
function _alias_checks(@nospecialize(tangents::Type), bidirectional::Bool)
    # Compare leaves across the whole signature, including two leaves of one argument.
    leaves = _aliasable_positions(tangents, true)
    n = length(leaves)
    # Limit quadratic codegen to 256 pairs; larger signatures use a linear runtime pass.
    # The runtime pass allocates two IdDicts, so small signatures retain the unrolled path.
    if n * (n - 1) ÷ 2 > 256
        ts = Expr(:tuple, (_path_expr(:tangents, i, p) for (i, p) in leaves)...)
        fs = Expr(:tuple, (_path_expr(:fx, i, p) for (i, p) in leaves)...)
        labels = Expr(:tuple, (_alias_label(i, p) for (i, p) in leaves)...)
        return :(_check_alias_partition($ts, $fs, $labels, $bidirectional))
    end
    bad = bidirectional ? :(same_primal != same_tangent) : :(same_tangent && !same_primal)
    checks = Expr(:block)
    for a in 1:n, b in (a + 1):n
        (i, pi), (j, pj) = leaves[a], leaves[b]
        ti, tj = _path_expr(:tangents, i, pi), _path_expr(:tangents, j, pj)
        fi, fj = _path_expr(:fx, i, pi), _path_expr(:fx, j, pj)
        li, lj = _alias_label(i, pi), _alias_label(j, pj)
        push!(
            checks.args,
            quote
                let same_primal = _shares_storage($fi, $fj),
                    same_tangent = _shares_storage($ti, $tj)

                    $bad && _throw_prepared_cache_aliasing_error($li, $lj, same_primal)
                end
            end,
        )
    end
    return checks
end

# Runtime form of the same pairwise contract for signatures too wide to unroll. Each
# repeated tangent must still name its first primal; reverse mode also checks the inverse.
@noinline function _check_alias_partition(
    tangents::Tuple, primals::Tuple, labels::Tuple, bidirectional::Bool
)
    first_tangent = IdDict{Any,Int}()
    first_primal = IdDict{Any,Int}()
    for k in eachindex(tangents)
        t = get!(first_tangent, _storage_id(tangents[k]), k)
        _shares_storage(primals[t], primals[k]) ||
            _throw_prepared_cache_aliasing_error(labels[t], labels[k], false)
        if bidirectional
            f = get!(first_primal, _storage_id(primals[k]), k)
            _shares_storage(tangents[f], tangents[k]) ||
                _throw_prepared_cache_aliasing_error(labels[f], labels[k], true)
        end
    end
    return nothing
end

# Reverse requires the prepared tangent partition to match the call's primal partition;
# types and sizes alone cannot establish this.
@generated function _check_tangent_aliasing(tangents::Tuple, fx::Tuple)
    return Expr(:block, _alias_checks(tangents, true), :(return nothing))
end

# A friendly forward cache shares prepared tangent buffers. Distinct call-time primals
# cannot write different seeds into one buffer; separate buffers for aliased primals are valid.
@generated function _check_prepared_forward_aliasing(tangents::Tuple, fx::Tuple)
    return Expr(:block, _alias_checks(tangents, false), :(return nothing))
end

# Repeated mutable primals must receive the same tangent object. Otherwise cached and
# rule-direct tuple paths silently keep different seeds (first or last), with no unique JVP.
# Sharing one tangent remains supported, including through mutation.
@generated function _check_repeated_arg_tangents(fx::Tuple)
    n = length(fx.parameters)
    checks = Expr(:block)
    for i in 1:n, j in (i + 1):n
        Base.ismutabletype(fx.parameters[i].parameters[1]) || continue
        push!(checks.args, quote
            if fx[$i][1] === fx[$j][1] && !(fx[$i][2] === fx[$j][2])
                _throw_repeated_arg_tangent_error($i, $j)
            end
        end)
    end
    push!(checks.args, :(return nothing))
    return checks
end

# `fx` carries the callable at position 1, so report the positions as the caller passed them.
@noinline function _throw_repeated_arg_tangent_error(i::Int, j::Int)
    name(k) = k == 1 ? "the callable" : "argument $(k - 1)"
    throw(
        ArgumentError(
            "$(name(i)) and $(name(j)) are the same mutable object but were given different " *
            "tangents. They share one storage, so they share one tangent, and only one direction " *
            "can be carried through it. Pass the same tangent at both positions to differentiate " *
            "along that direction.",
        ),
    )
end

# A callable capturing an argument's array shares storage without top-level identity.
# Compare supplied tangents against one canonically seeded set, only when the prepared
# `inputs_share_storage` flag indicates sharing.
@inline function _check_shared_input_tangents(
    cache, input_primals::Tuple, input_tangents::Tuple
)
    # The prepared flag is only a pre-filter: call-time primals may be distinct, which is valid.
    getfield(cache, :inputs_share_storage) || return nothing
    ts = _zero_tangents(input_primals)
    shared_dim = tangent_dim(ts)
    _inputs_share_storage(shared_dim, ts, input_primals) || return nothing
    @static if VERSION >= v"1.11-rc4"
        # Canonical tangents share the primals' backing Memory. Compare the actual mapping:
        # equal dimension counts and unrelated sharing in reverse scratch cannot certify it.
        # Coherent shared seeds remain supported; constructing them for HVP is separate work.
        return _check_tangent_storage!(IdDict{Any,Any}(), ts, input_tangents)
    end
    # Canonical tangents count shared storage once. Extra supplied dimensions indicate
    # conflicting seeds; fewer are valid when distinct primals receive the same direction.
    shared = tangent_dim(input_tangents, IdDict{Any,Any}())
    shared > shared_dim && _throw_shared_input_tangent_error()
    # One buffer under two array containers (`da` and `reshape(da)`, or an `Array` beside its
    # backing `Memory`) leaves the counts equal, so `_any_shared_storage` is what recognises it.
    _any_shared_storage(input_tangents) && return nothing
    summed = sum(t -> tangent_dim(t, IdDict{Any,Any}()), input_tangents; init=0)
    shared == summed && _throw_shared_input_tangent_error()
    return nothing
end

# Compare canonical and supplied tangent storage, not scalar values. This is one-directional:
# distinct primals may use the same direction, but a shared primal cannot carry two directions.
# Julia 1.10 keeps the count check above because its reshape tangents do not share backing storage.
function _check_tangent_storage!(seen::IdDict, expected::T, supplied) where {T}
    isbitstype(T) && return nothing
    if expected isa MistyClosureTangent
        # The compiled dual callable is code, not derivative storage (as in `tangent_dim`).
        return _check_tangent_storage!(
            seen, expected.captures_tangent, supplied.captures_tangent
        )
    end
    if expected isa AbstractArray
        (isempty(expected) || eltype(expected) === NoTangent) && return nothing
    end
    if ismutable(expected)
        key, value = _storage_id(expected), _storage_id(supplied)
        if haskey(seen, key)
            # A wrapper of integer or empty arrays can be mutable yet carry no dimensions.
            seen[key] === value ||
                tangent_dim(expected) == 0 ||
                _throw_shared_input_tangent_error()
            return nothing
        end
        seen[key] = value
    end
    if expected isa AbstractArray
        isbitstype(eltype(expected)) && return nothing
        for i in eachindex(expected)
            isassigned(expected, i) &&
                _check_tangent_storage!(seen, expected[i], supplied[i])
        end
    else
        for i in 1:fieldcount(T)
            isdefined(expected, i) &&
                _check_tangent_storage!(seen, getfield(expected, i), getfield(supplied, i))
        end
    end
    return nothing
end

@noinline function _throw_shared_input_tangent_error()
    throw(
        ArgumentError(
            "These inputs share differentiable storage across positions (`f` holding the same " *
            "array that is also passed as an argument, say), but the supplied tangents do not " *
            "share it. One storage carries one tangent, so the lift would keep one of them and " *
            "silently drop the other. Pass the same tangent object for the shared storage at " *
            "every position it occupies, or use reverse mode, which accumulates into it instead.",
        ),
    )
end

# A forward gradient's per-argument basis ranges cannot represent shared storage;
# `value_and_derivative!!` can, using a caller-supplied shared seed.
# Check mutable top-level arguments per call using `_shares_storage`, including Array/Memory
# pairs. Immutable identity is value equality. Nested sharing and sharing with `f` are
# checked at preparation by `_inputs_share_storage` to avoid per-call traversal.
# Generate literal indices: dynamic indexing of the heterogeneous tuple is type-unstable.
@generated function _check_primal_aliasing(x::Tuple)
    checks = Expr(:block)
    leaves = _aliasable_positions(x, false)
    for a in eachindex(leaves), b in (a + 1):length(leaves)
        i, j = leaves[a][1], leaves[b][1]
        push!(
            checks.args,
            :(
                _shares_storage(x[$i], x[$j]) &&
                _holds_derivatives(x[$i]) &&
                _throw_gradient_arg_alias_error($i, $j)
            ),
        )
    end
    return quote
        $checks
        return nothing
    end
end

# Shared storage matters only if it carries dimensions. Empty arrays share Julia's global
# empty Memory, and integer buffers carry no derivatives; emptiness requires a runtime check.
@inline _holds_derivatives(x) = tangent_dim(zero_tangent(x)) != 0

function _throw_gradient_arg_alias_error(i::Int, j::Int)
    throw(
        ArgumentError(
            "Forward-mode `value_and_gradient!!` does not support arguments $i and $j sharing " *
            "one storage — the same mutable object, or an `Array` passed alongside its backing " *
            "`Memory`: the gradient is assembled from one standard-basis tangent_dim range per " *
            "argument, which cannot represent a storage that occupies two of them. Use " *
            "`value_and_derivative!!` with one tangent shared across those positions, or use " *
            "reverse mode.",
        ),
    )
end

# Count the SAME tangents jointly and per argument: only cross-position sharing changes
# the count; repeated leaves within one argument and equal scalars do not.
# This catches nested sharing and sharing with `f`, unlike `_check_primal_aliasing`.
# The full traversal runs only at preparation, so newly introduced nested sharing is not
# caught. The Jacobian needs no check: its single input range covers aliased captures too.
function _inputs_share_storage(shared_dim::Int, ts::Tuple, fx::Tuple)
    shared_dim != sum(tangent_dim, ts; init=0) && return true
    # 1.11+ reads the sharing off the TANGENTS, where aliased primals share a `Memory`. On 1.10
    # they do not, so that version reads it off the primals instead; see `_any_shared_storage!`.
    return @static if VERSION >= v"1.11-rc4"
        _any_shared_storage(ts)
    else
        _any_shared_storage(fx)
    end
end

# Track visited objects separately from claimed backing storage: repeated objects already
# share tangents, but distinct containers over one buffer evade identity-based dimension counts.
# Memory can be both a visited object and another container's backing storage.
struct _StorageSeen
    objs::IdDict{Any,Nothing}
    backing::IdDict{Any,Nothing}
end
function _any_shared_storage(x)
    return _any_shared_storage!(
        _StorageSeen(IdDict{Any,Nothing}(), IdDict{Any,Nothing}()), x
    )
end

# Empty arrays share Julia's global empty Memory; nondifferentiable buffers have no
# contributions to double-count. Neither should trigger the sharing guard.
function _any_shared_storage!(s::_StorageSeen, x::_BuiltinArrays)
    (isempty(x) || tangent_type(eltype(x)) === NoTangent) && return false
    haskey(s.objs, x) && return false
    s.objs[x] = nothing
    # Julia 1.10 has no `Memory`, so the data address stands in for it: stable while the array is
    # live, which it is for the whole traversal, and equal for two arrays over one buffer.
    store = @static VERSION >= v"1.11-rc4" ? _storage_id(x) : UInt(pointer(x))
    haskey(s.backing, store) && return true
    s.backing[store] = nothing
    # The elements can share storage while their containers do not. Guarded on the visited check
    # above, so an array holding itself terminates. A bits element holds no storage of its own,
    # which is also what keeps a large float array off this path.
    isbitstype(eltype(x)) && return false
    for i in eachindex(x)
        isassigned(x, i) && _any_shared_storage!(s, x[i]) && return true
    end
    return false
end
function _any_shared_storage!(s::_StorageSeen, x::Union{Tuple,NamedTuple})
    return any(v -> _any_shared_storage!(s, v), x)
end

@static if VERSION >= v"1.11-rc4"
    _any_shared_storage!(::_StorageSeen, ::Any) = false
    function _any_shared_storage!(s::_StorageSeen, x::PossiblyUninitTangent)
        return is_init(x) && _any_shared_storage!(s, val(x))
    end
    function _any_shared_storage!(s::_StorageSeen, x::Union{Tangent,MutableTangent})
        # Register before descending: a self-referential struct (`node.next === node`) otherwise
        # recurses forever, as the other tangent traversals guard against for the same reason.
        haskey(s.objs, x) && return false
        s.objs[x] = nothing
        return _any_shared_storage!(s, x.fields)
    end
else
    # On 1.10, reshape tangents do not share storage, so walk primals instead.
    # Admit struct fields by tangent type, matching the 1.11+ tangent walk and avoiding
    # unrelated graphs such as a module captured by f.
    function _any_shared_storage!(s::_StorageSeen, x)
        tangent_type(_typeof(x)) <: Union{Tangent,MutableTangent} || return false
        # Only a mutable can be cyclic, and registering an immutable would key an `IdDict` on its
        # CONTENTS, so two distinct-but-equal structs would look visited and the second go unwalked.
        if ismutable(x)
            haskey(s.objs, x) && return false
            s.objs[x] = nothing
        end
        return any(
            i -> isdefined(x, i) && _any_shared_storage!(s, getfield(x, i)),
            1:fieldcount(_typeof(x)),
        )
    end
end

# A concrete input needs a concrete derivative representation: invariant CoDual/Lifted
# slots otherwise fail annotation, potentially crashing reverse codegen. Abstract-field
# NamedTuples are one example. Only inputs are checked; internal values can still work.
# Ask each mode separately: extensions may support only one representation, e.g.
# DynamicExpressions' recursive nodes have a custom reverse tangent but no forward dual.
@inline function _check_representable_input(mode::Mode, @nospecialize(P::Type), i::Int)
    isconcretetype(P) || return nothing
    forward = mode isa ForwardMode
    # NDualArray partials use `zero(T)`, requiring a concrete element type even when the
    # array type itself is concrete. Other forward array representations and reverse
    # plain-array tangents support abstract element types.
    if forward
        D = dual_type(Val(1), P)
        D <: Nfwd.NDualArray &&
            !isconcretetype(D.parameters[1]) &&
            throw(
                ArgumentError(
                    "Argument $i has type `$P`, whose element type `$(eltype(P))` is not " *
                    "concrete. Forward mode stores each element's partials as an " *
                    "`NTuple{N,T}`, which needs a concrete `T` to stay unboxed — that is what " *
                    "makes chunked forward mode fast — so Mooncake restricts forward inputs " *
                    "to concrete element types. Use one, e.g. `Vector{Float64}` or " *
                    "`Vector{Float32}`.",
                ),
            )
    end
    isconcretetype(forward ? dual_type(Val(1), P) : tangent_type(P)) && return nothing
    return throw(
        ArgumentError(
            "Argument $i has type `$P`, whose derivative representation cannot be annotated: " *
            "$(forward ? "`dual_type`" : "`tangent_type`") widens it to a non-concrete type, and " *
            "the slot wrappers are invariant in that parameter. A `NamedTuple` with an abstract " *
            "field type (`@NamedTuple{a}`) is the usual case. Use a struct with the same field " *
            "instead — its representation keeps the declared field type.",
        ),
    )
end

# Refused here rather than at construction because `value_and_derivative!!` shares this cache
# and handles aliased inputs correctly: the caller supplies the seeds, so one tangent can cover
# every position the shared leaf occupies.
@inline function _check_gradient_input_aliasing(cache::FCache)
    getfield(cache, :inputs_share_storage) && _throw_gradient_input_alias_error()
    return nothing
end

function _throw_gradient_input_alias_error()
    throw(
        ArgumentError(
            "Forward-mode `value_and_gradient!!` does not support two arguments over one " *
            "storage — `f` holding the same array that is also passed as an argument, say. " *
            "Each argument carries its own tangent storage, so the sweep gives the shared " *
            "leaf one standard-basis tangent_dim range per position and its gradient comes back " *
            "scaled by that count. Repeated leaves within a single argument are supported " *
            "and agree with reverse mode; sharing that is not object identity (a `reshape`, " *
            "a `view`, an `Array` beside its backing `Memory`) is not, wherever it appears. " *
            "Use `value_and_derivative!!` with one tangent shared across the positions, or " *
            "reverse mode, which accumulates into the shared storage.",
        ),
    )
end

# Generic gradient and Jacobian sweeps snapshot/restore arguments before each chunk and
# at exit, preventing compounded mutations. Zero-allocation paths refresh cache-owned seeds
# instead. The callable is not snapshotted; it may be uncopyable.

# ── Zero-allocation gradient for array-backed structured inputs ───────────────

#
# Preallocate width-W seeds, gradients, and dimension-ordered array leaf pairs.
# Real/complex arrays with mixed eltypes qualify; scalar dimensions, unsupported shapes,
# repeated leaves, or non-isbits NoDual state require generic snapshot/restore.
struct StructuredGradSeed{Ff,As,Gs,Ls,Bs}
    f_seed::Ff
    arg_seeds::As
    grad_bufs::Gs
    leaves::Ls
    # Mutable primal/dual objects and their saved fields; see `_save_seed_bindings`.
    bindings::Bs
end

# Isbits seeds rebuild on the stack behind a concrete width-W barrier. Stored template
# types keep Lifted reconstruction concrete; precomputed dimensions avoid an IdDict.
struct IsbitsGradSeed{W,Tmpls}
    templates::Tmpls
    total_dim::Int
end

# `fwd_cache` is the derivative cache for `grad_f`. For non-primitive `f`, the compiled
# inner rrule lives in the `DerivedFoRRule` captured by `grad_f` (built once at prep);
# `get_inner_rrule`'s frule serves its `Lifted` to the forward pass on every
# `value_and_hvp!!` call.
"""
    HVPCache

Cache type used by [`prepare_hvp_cache`](@ref) and [`prepare_hessian_cache`](@ref) for
repeated Hessian-vector product and Hessian evaluations.
"""
struct HVPCache{Tf,Tgrad_f,Tgrad_tangent,Tfwd_cache,TOS,THB}
    f::Tf
    grad_f::Tgrad_f
    # Pre-computed zero tangent for grad_f; the function is never perturbed, only x is.
    # Safe to reuse because grad_f's closure environment is shape-stable for the lifetime
    # of the cache: grad_cache mutates stored values between calls but does not change the
    # closure/capture structure that zero_tangent depends on.
    grad_tangent::Tgrad_tangent
    fwd_cache::Tfwd_cache
    output_type_and_size::TOS
    # Hessian-assembly buffers populated by `prepare_hessian_cache`, `nothing` for caches
    # built via `prepare_hvp_cache`. `value_gradient_and_hessian!!` writes into these.
    # Layout: `((; H::Matrix, grad::Vector, v::Vector), chunked)`.
    hess_buffers::THB
end

# Include the cache's mode in errors: forward and reverse have different admission rules.
# HVP/Hessian use the bare name because they combine both modes.
_caller_label(::Nothing, ::Nothing) = "this function"
_caller_label(::Nothing, ::Cache) = "reverse-mode differentiation"
_caller_label(::Nothing, ::FCache) = "forward-mode differentiation"
_caller_label(::Nothing, ::HVPCache) = "this function"
_caller_label(caller, ::Nothing) = string(caller)
_caller_label(caller, ::Cache) = "reverse-mode $caller"
_caller_label(caller, ::FCache) = "forward-mode $caller"
_caller_label(caller, ::HVPCache) = string(caller)

function _cache_show_fields(cache::HVPCache)
    fwd_cache = getfield(cache, :fwd_cache)
    return (;
        mode=:forward_over_reverse,
        chunk=(!isnothing(getfield(fwd_cache, :chunk_rule))),
        inputs=length(_cache_input_types_and_sizes(fwd_cache)),
    )
end

function Base.show(io::IO, cache::Union{Cache,FCache,HVPCache})
    print(io, "Mooncake.", nameof(typeof(cache)), "(")
    for (i, (name, value)) in enumerate(pairs(_cache_show_fields(cache)))
        i == 1 || print(io, ", ")
        print(io, name, "=", value isa Symbol ? ":" : "", value)
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cache::Union{Cache,FCache,HVPCache})
    print(io, "Mooncake.", nameof(typeof(cache)))
    for (name, value) in pairs(_cache_show_fields(cache))
        print(io, "\n  ", name, ": ", value)
    end
    inputs = cache isa HVPCache ? getfield(cache, :fwd_cache) : cache
    for (i, type_and_size) in enumerate(_cache_input_types_and_sizes(inputs))
        print(io, "\n  input_", i, ": ", _cache_summary(type_and_size))
    end
    print(io, "\n  output: ", _cache_summary(cache))
end

#
# Forward mode — derivative and Jacobian
#

"""
    value_and_derivative!!(rule, f::Lifted, x::Lifted...)
    value_and_derivative!!(rule, (f, df), (x, dx), ...)

Run a forward rule directly, without first constructing a `FCache`.

The tuple interface lifts each input to a width-1 slot and returns `(y, dy)` for a single
directional derivative. The `Lifted` interface returns the rule output (a `Lifted`) directly
and computes one derivative per lane of the supplied `Lifted` width — width-1 unless the
caller built wider (chunked) slots.

Positions holding the same object share one tangent, so the tuple interface refuses two
different tangents for them: only one direction can be carried and choosing silently is worse
than refusing. Passing the same tangent object at every such position is well-posed and
supported. The `Lifted` interface does not check — there the caller has already built the
slots, and two slots over one primal carrying independent directions is a deliberate part of
the forward representation.
"""
@inline function value_and_derivative!!(rule::R) where {R}
    throw(
        ArgumentError(
            "`value_and_derivative!!(rule, ...)` expects at least the function input, " *
            "either as `f::Lifted` or `(f, df)`.",
        ),
    )
end

@inline function value_and_derivative!!(rule::R, fx::Vararg{Lifted,N}) where {R,N}
    return __call_rule(rule, fx)
end

@inline function value_and_derivative!!(rule::R, fx::Vararg{Tuple{Any,Any},N}) where {R,N}
    _check_repeated_arg_tangents(fx)
    input_primals = tuple_map(first, fx)
    input_tangents = tuple_map(last, fx)
    # Share the lift cache across arguments: fresh partial blocks for aliased primals would
    # make mutations visible through the primal but not through its derivative.
    c = IdDict()
    input_lifteds = tuple_map((p, t) -> lift(p, t, c), input_primals, input_tangents)
    output = __call_rule(rule, input_lifteds)
    return primal(output), last(unlift(output))
end

@inline _dual_primal_type(::Type) = Any
@inline _dual_primal_type(::Type{<:Lifted{Y}}) where {Y} = Y

function _cache_summary(cache::Union{Cache,HVPCache})
    _cache_summary(getfield(cache, :output_type_and_size))
end

function _cache_summary(cache::FCache)
    # The forward output shape is unknown at prepare time, so it is always inferred from the
    # rule's return type.
    lifted_arg_types = Tuple{
        map(
            type_and_size -> lifted_type(Val(1), typeof(type_and_size).parameters[1]),
            getfield(cache, :input_types_and_sizes),
        )...,
    }
    output_type = Core.Compiler.return_type(getfield(cache, :single_rule), lifted_arg_types)
    return _cache_summary(_dual_primal_type(output_type), "size unknown")
end

"""
    prepare_derivative_cache(fx...; config=Mooncake.Config())

Returns a cache used with [`value_and_derivative!!`](@ref). See that function for more info.

`config.chunk_size` sets the forward chunk width baked into this cache (a default heuristic
when `nothing`), capped at the total differentiable degrees of freedom. It governs the
chunked [`value_and_gradient!!`](@ref) / [`value_and_jacobian!!`](@ref) paths (every input
shape chunks), not the single-direction [`value_and_derivative!!`](@ref); the resolved width
is shown by the cache.

!!! note
    Cache construction stays lazy and does not execute `f(x...)`. Unlike the reverse
    pullback cache, the forward output is therefore not pre-validated: an output that is or
    contains a `Ptr` (or aliases/cycles) surfaces inside the rule at evaluation time rather
    than as a `ValueAndPullbackReturnTypeError` at preparation.
"""
@unstable @inline function prepare_derivative_cache(
    f, x::Vararg{Any,N}; config=Config()
) where {N}
    config.empty_cache && empty_mooncake_caches!()
    fx = (f, x...)
    # `_stable_typeof`, not `_typeof`: the latter sharpens NamedTuple elements, narrowing
    # `@NamedTuple{a}` to `@NamedTuple{a::Float64}`, whose `dual_type` IS concrete — the check would
    # then miss exactly the shape it exists for.
    ntuple(
        i -> _check_representable_input(ForwardMode(), Base._stable_typeof(fx[i]), i - 1),
        Val(N + 1),
    )
    requested_chunk_size = getfield(config, :chunk_size)
    requested_chunk_size = if isnothing(requested_chunk_size)
        0
    else
        Nfwd._nfwd_check_chunk_size(requested_chunk_size)
    end
    gradient_chunk_size_auto = requested_chunk_size == 0
    rule = build_frule(fx...; config.debug_mode, config.silence_debug_messages)
    input_types_and_sizes = map(_type_and_size, fx)
    # All shapes chunk; only zero-allocation seeds below are shape-restricted.
    input_ts = _zero_tangents(fx)
    total_dim = tangent_dim(input_ts)
    inputs_share_storage = _inputs_share_storage(total_dim, input_ts, fx)
    gradient_chunk_size = let
        requested = gradient_chunk_size_auto ? _MAX_CHUNK_WIDTH : requested_chunk_size
        min(total_dim, requested)
    end
    # Width 1 reuses `single_rule`; wider rules batch directional derivatives.
    chunk_rule = if gradient_chunk_size > 1
        build_frule(
            fx...;
            chunk_size=gradient_chunk_size,
            config.debug_mode,
            config.silence_debug_messages,
        )
    else
        nothing
    end
    # Preallocated seed for the zero-allocation single-float-vector gradient path: a width-W
    # `x_seed` over a cache-owned primal buffer (partials mutated in place per chunk) plus
    # the inert `f_seed`. `nothing` for every other shape, which uses the generic gradient
    # path.
    gradient_seed = let args = Base.tail(fx)
        # The flat seed assumes nondifferentiable `f` and same-eltype float vectors, matching
        # its sweep's dispatch. `similar` must preserve each argument's type: a plain Vector
        # seed for a SubArray mismatches the cache and rule. Such wrappers use structured seeds.
        if gradient_chunk_size >= 1 &&
            tangent_type(typeof(first(fx))) === NoTangent &&
            !isempty(args) &&
            first(args) isa AbstractVector{<:IEEEFloat} &&
            all(a -> a isa AbstractVector{eltype(first(args))}, args) &&
            # The flat sweep counts elements; extra differentiable fields need the generic path.
            all(i -> tangent_dim(input_ts[i + 1]) == length(args[i]), eachindex(args)) &&
            all(a -> typeof(similar(a)) == typeof(a), args)
            W = gradient_chunk_size
            (
                zero_lifted(Val(W), fx[1]),
                map(a -> zero_lifted(Val(W), similar(a)), args),
                map(similar, args),
            )
        else
            nothing
        end
    end
    # Try structured array-backed seeds next; unsupported dimensions or differentiable `f`
    # fall back to the generic sweep.
    if gradient_seed === nothing &&
        gradient_chunk_size >= 1 &&
        tangent_type(typeof(first(fx))) === NoTangent &&
        !isempty(Base.tail(fx))
        W = gradient_chunk_size
        _args = Base.tail(fx)
        # Seed cache-owned copies so in-place rules cannot mutate prepare-time arguments.
        # Preserve intra-argument aliases so `_tangent_layout` can reject them.
        _arg_seeds = map(a -> zero_lifted(Val(W), deepcopy(a)), _args)
        _grad_bufs = _zero_tangents(_args)
        _leaves = _tangent_layout(_arg_seeds, _grad_bufs)
        _bindings = if _leaves === nothing
            nothing
        else
            _cat_leaves(map(s -> _save_seed_bindings(tangent(s), primal(s)), _arg_seeds))
        end
        if _leaves !== nothing && _bindings !== nothing
            gradient_seed = StructuredGradSeed(
                zero_lifted(Val(W), fx[1]), _arg_seeds, _grad_bufs, _leaves, _bindings
            )
        elseif isbitstype(typeof(fx)) &&
            _all_real_scalars(typeof(tangent(zero_lifted(Val(W), fx))))
            # Both primal and V must be isbits to avoid per-chunk allocation; templates keep
            # Lifted reconstruction type-stable.
            templates = map(a -> zero_lifted(Val(W), a), fx)
            gradient_seed = IsbitsGradSeed{W,typeof(templates)}(templates, total_dim)
        end
    end
    # Jacobian output buffer for the zero-allocation packable path: a single same-eltype
    # float-vector input qualifies. The output shape is unknown here, so hold a `Ref` that
    # the first `value_and_jacobian!!` call sizes and fills once the output vector is known.
    jacobian_buffer = let args = Base.tail(fx)
        if gradient_seed isa Tuple && length(args) == 1 && eltype(only(args)) <: IEEEFloat
            Base.RefValue{Union{Nothing,Matrix{eltype(only(args))}}}(nothing)
        else
            nothing
        end
    end
    input_tangents, friendly_gradients, gradient_workspace = if config.friendly_tangents
        # `input_ts` is already built for the dimension count and has not been written to.
        (input_ts, _copy_output(fx), Ref{Union{Nothing,typeof(input_ts)}}(nothing))
    else
        # Keep the lazy workspace concretely typed without evaluating more tangents.
        (
            nothing,
            nothing,
            Ref{Union{Nothing,Tuple{map(tangent_type, fieldtypes(typeof(fx)))...}}}(
                nothing
            ),
        )
    end
    return FCache(
        rule,
        input_tangents,
        friendly_gradients,
        gradient_workspace,
        GradientChunkSize(gradient_chunk_size, gradient_chunk_size_auto),
        chunk_rule,
        input_types_and_sizes,
        map(_copy_output, Base.tail(fx)),
        gradient_seed,
        inputs_share_storage,
        jacobian_buffer,
    )
end

"""
    value_and_derivative!!(cache::FCache, f::Lifted, x::Vararg{Lifted,N})

Returns a `Lifted` containing the result of applying forward-mode AD to compute the
(Fréchet) derivative of `primal(f)` at the primal values in `x` in the direction of the
tangent values in `f` and `x`.
"""
# Width-1 slots select `single_rule`; wider slots must match the prepared chunk width.
# The tuple interface below always supplies width 1.
function value_and_derivative!!(cache::FCache, fx::Vararg{Lifted{<:Any,1},N}) where {N}
    input_primals = map(primal, fx)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), input_primals)
    return __call_rule(cache.single_rule, fx)
end
function value_and_derivative!!(cache::FCache, fx::Vararg{Lifted,N}) where {N}
    input_primals = map(primal, fx)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), input_primals)
    rule = cache.chunk_rule
    rule === nothing && throw(
        PreparedCacheError(
            "This cache holds no chunk rule: width-N Lifted inputs require the cache to " *
            "have been prepared with chunk_size > 1 (resolved chunk width " *
            "$(getfield(cache, :gradient_chunk_size).width)).",
        ),
    )
    W = getfield(cache, :gradient_chunk_size).width
    # Every slot must share the cache's chunk width: the chunk rule's OpaqueClosure is built
    # at width `W` and would otherwise type-mismatch on a trailing slot opaquely. Checking
    # only `first(fx)` would let mixed-width inputs (e.g. width-W `f` with a width-W'
    # argument) past.
    widths = map(_lifted_width, fx)
    all(==(W), widths) || throw(
        PreparedCacheError(
            "Lifted inputs have chunk widths $widths, but this cache's chunk rule was " *
            "built at width $W; all inputs must share the cache's chunk width.",
        ),
    )
    return __call_rule(rule, fx)
end

"""
    value_and_derivative!!(cache::FCache, (f, df), (x, dx), ...)

Returns a tuple `(y, dy)` containing the result of applying forward-mode AD to compute the
(Fréchet) derivative of `primal(f)` at the primal values in `x` in the direction of the
tangent values contained in `df` and `dx`.

Tuples are used as inputs and outputs instead of a combined value/tangent wrapper to
accommodate the case where internal Mooncake tangent types do not coincide with tangents
provided by the user (in which case we translate between "friendly tangents" and internal
tangents using cache storage).
With `friendly_tangents=false`, each direction must have the internal tangent type for its
primal and matching axes (or length for sized collections without axes).

The arguments in `x` are returned to their original state, whether the rule returns or
raises: if `f` mutates them in place, they are restored from a cache-owned snapshot, so the
inputs are not mutated. `f` itself is not snapshotted — a callable that mutates its own
fields is not restored.

!!! info
    `cache` must be the output of [`prepare_derivative_cache`](@ref), and (fields of) `f`
    and `x` must be of the same size and shape as those used to construct the `cache`. This
    is to ensure that the gradient can be written to the memory allocated when the `cache`
    was built.

!!! warning
    `cache` owns any mutable state returned by this function, meaning that mutable
    components of values returned by it will be mutated if you run this function again with
    different arguments. Therefore, if you need to keep the values returned by this function
    around over multiple calls to this function with the same `cache`, you should take a
    copy (using `copy` or `deepcopy`) of them before calling again.
"""
@inline function value_and_derivative!!(
    cache::FCache{R,IT,FG,GW,CF,S}, fx::Vararg{Tuple{Any,Any},M}
) where {R,IT<:Tuple,FG,GW,CF,S,M}
    input_primals = tuple_map(first, fx)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), input_primals)
    # Types and sizes match when only the aliasing differs, so the check above cannot see it.
    _check_prepared_forward_aliasing(cache.input_tangents, input_primals)
    _check_repeated_arg_tangents(fx)
    input_friendly_tangents = tuple_map(last, fx)
    input_tangents = tuple_map(
        primal_to_tangent!!, cache.input_tangents, input_friendly_tangents
    )

    # Restore in `finally` so a mutating rule that throws also leaves inputs unchanged.
    input_snapshot, restore_contexts = _snapshot_inputs!!(
        cache.input_snapshot, Base.tail(input_primals)
    )
    try
        # Shared aliasing cache across the argument tuple; see the `FCache{R,Nothing,…}`
        # method.
        c = IdDict()
        output = __call_rule(
            cache.single_rule,
            tuple_map((p, t) -> lift(p, t, c), input_primals, input_tangents),
        )
        output_primal = primal(output)
        _, output_internal_tangent = unlift(output)
        output_friendly_tangent = tangent_to_friendly!!(
            friendly_tangent_cache(output_primal),
            output_primal,
            output_internal_tangent,
            _friendly_cache((output_primal,)),
        )
        # Copy before restoring: the result may alias a mutated input. Isbits copies are free.
        return _copy_output(output_primal), output_friendly_tangent
    finally
        _restore_inputs!!(Base.tail(input_primals), input_snapshot, restore_contexts)
    end
end

@inline function value_and_derivative!!(
    cache::FCache{R,Nothing,FG,GW,CF,S}, fx::Vararg{Tuple{Any,Any},M}
) where {R,FG,GW,CF,S<:Tuple,M}
    input_primals = tuple_map(first, fx)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), input_primals)
    _check_repeated_arg_tangents(fx)
    input_tangents = tuple_map(last, fx)
    # Only this method can check nested shared-input seeds: the friendly path has already
    # merged them into prepared buffers, and the rule-direct path has no canonical tangent
    # set. Building one there would defeat its allocation savings; see `known_limitations.md`.
    _check_shared_input_tangents(cache, input_primals, input_tangents)

    tuple_map(_check_tangent_for_primal, input_primals, input_tangents)

    # One aliasing cache scoped to this input lift: a reverse rule captured in
    # `grad_f` shares its `fwds_oc`/`pb_oc` captures, so the forward tangent of
    # that shared mutable state must be shared too (see `lift(::MistyClosure)`).
    c = IdDict()
    input_lifted = tuple_map((p, t) -> lift(p, t, c), input_primals, input_tangents)
    # Snapshot/restore around the rule so an in-place `f` does not mutate the user's inputs,
    # on the throwing path as well; see the friendly method above.
    input_snapshot, restore_contexts = _snapshot_inputs!!(
        cache.input_snapshot, Base.tail(input_primals)
    )
    try
        output = __call_rule(cache.single_rule, input_lifted)
        # Copy the output primal out before the restore: it may alias an in-place-mutated
        # input, which the restore would otherwise overwrite with the original.
        return _copy_output(primal(output)), last(unlift(output))
    finally
        _restore_inputs!!(Base.tail(input_primals), input_snapshot, restore_contexts)
    end
end

function _check_vector_argument(
    x; caller=nothing, cache=nothing, eltypes::Type=IEEEFloat, dense::Bool=false
)
    x isa AbstractVector || throw(
        ArgumentError(
            "$(_caller_label(caller, cache)) only supports AbstractVector inputs; got $(typeof(x))",
        ),
    )
    T = eltype(x)
    # Concreteness excludes Union float eltypes that cannot seed NDualArray and makes
    # output `eltype(y) <: T` an equality check.
    isconcretetype(T) || throw(
        ArgumentError(
            "$(_caller_label(caller, cache)) requires a concrete element type; got eltype $T",
        ),
    )
    T <: eltypes || throw(
        ArgumentError(
            "$(_caller_label(caller, cache)) only supports AbstractVector inputs with $eltypes " *
            "element types; got eltype $T",
        ),
    )
    # The sweep seeds offsets `1:n`; strided views would seed the wrong parent elements.
    !dense ||
        x isa DenseVector ||
        throw(
            ArgumentError(
                "$(_caller_label(caller, cache)) only supports dense vector inputs; got $(typeof(x))",
            ),
        )
    return T
end

function _check_vector_output(
    y; caller=nothing, cache=nothing, eltypes::Type, dense::Bool=false
)
    y isa AbstractVector || throw(
        ArgumentError(
            "$(_caller_label(caller, cache)) only supports AbstractVector outputs; got $(typeof(y))",
        ),
    )
    # Same reason as the input side, one step later: a wrapper output's derivative is a struct
    # lift over its parent, so the Jacobian has no flat array to read its columns from.
    !dense ||
        y isa DenseVector ||
        throw(
            ArgumentError(
                "$(_caller_label(caller, cache)) only supports dense vector outputs; got " *
                "$(typeof(y)). Materialise the output first (e.g. `collect`).",
            ),
        )
    T = eltype(y)
    T <: eltypes || throw(
        ArgumentError(
            if isconcretetype(eltypes)
                "$(_caller_label(caller, cache)) requires input and output AbstractVector element " *
                "types to match; got input eltype $eltypes and output eltype $T"
            else
                "$(_caller_label(caller, cache)) only supports AbstractVector outputs with " *
                "$eltypes element types; got eltype $T"
            end,
        ),
    )
    return T
end

function _check_jacobian_output(y, Tx)
    # One call, not two: `Tx` is the eltype of an input already checked against `IEEEFloat`,
    # so `Ty <: Tx` carries the float property with it.
    Ty = _check_vector_output(y; caller=(value_and_jacobian!!), eltypes=Tx, dense=true)
    # Wrapper derivatives may be struct lifts, not flat arrays, in either mode.
    # Width 1 suffices to check the representation. Using `dual_type` for reverse is bounded
    # by the AbstractVector guard, but a recursive AbstractVector could still overflow;
    # a reverse-specific check needs verification across the wrapper family.
    dual_type(Val(1), typeof(y)) <: NDualArray || throw(
        ArgumentError(
            "value_and_jacobian!! does not support a $(typeof(y)) output: its derivative " *
            "representation is a struct lift over the parent rather than a flat array. " *
            "Materialise the output first (e.g. `collect`).",
        ),
    )
    return Ty
end

# A concrete barrier keeps the packable Jacobian sweep type-stable. Jref owns the output
# matrix, sized on the first call and overwritten on reuse.
function _fcache_jacobian_packable!!(
    cache::FCache, Jref, f_seed, arg_seed, W::Int, total_dim::Int, x::AbstractVector{T}
) where {T}
    nda = arg_seed.rep
    z = zero(T)
    local y, J
    s = 1
    while s <= total_dim
        _reset_seed_extent!(nda, total_dim)
        copyto!(nda.primal, x)
        # Zero every lane, then poke this chunk's standard-basis entries (element `slot`, lane
        # `lane`). Storage layout is version-specific (element-major block on 1.11+, per-lane
        # arrays on 1.10); `Nfwd._zero_seed!`/`_set_partial!` hide it.
        Nfwd._zero_seed!(nda)
        @inbounds for lane in 1:W
            slot = s + lane - 1
            slot <= total_dim && Nfwd._set_partial!(nda, slot, lane, one(T))
        end
        output = value_and_derivative!!(cache, f_seed, arg_seed)
        if s == 1
            # Not copied: this path is zero-allocation by contract, so the returned value aliases
            # cache-owned storage exactly as `J` does, and the next call on this cache overwrites
            # both. The docstring says so. Copying here costs the guarantee the allocation test pins.
            y = primal(output)
            _check_jacobian_output(y, T)
            cached = Jref[]
            J = if cached === nothing || size(cached) != (length(y), total_dim)
                Jref[] = zeros(T, length(y), total_dim)
            else
                fill!(cached, z)
            end
        end
        # Read each lane's directional derivative straight out of the output's block, which
        # `_check_jacobian_output` has established is an `NDualArray`. Going through
        # `tangent(output, lane)` instead would materialize a fresh per-lane copy out of the
        # element-major block — an allocation per lane.
        ov = tangent(output)::NDualArray
        @inbounds for lane in 1:W
            col = s + lane - 1
            col <= total_dim || break
            for r in 1:length(getfield(ov, :primal))
                J[r, col] = Nfwd._get_partial(ov, r, lane)
            end
        end
        s += W
    end
    return y, J
end

"""
    value_and_jacobian!!(cache::FCache, f, x)
    value_and_jacobian!!(cache::Cache, f, x)

Using a pre-built cache, compute and return `(value, jacobian)` for a vector-valued function
`f` of a single vector input.

The current implementation supports a single non-empty dense vector input and an
`AbstractVector` output, both with the same `IEEEFloat` element type. (Note the input must
be dense even though [`value_and_gradient!!`](@ref) on the same cache also accepts strided
views.) The returned Jacobian is a dense matrix whose columns correspond to input
coordinates.

As with all functionality in Mooncake, `x` is returned to its original state: if `f` mutates
`x` in place, it is restored, so the input is not mutated.

!!! info
    `cache` must be the output of [`prepare_derivative_cache`](@ref) or
    [`prepare_pullback_cache`](@ref), and `f` and `x` must match the types and shapes used
    to construct the cache. A [`prepare_gradient_cache`](@ref) cache is for a scalar output
    and is not usable here; it is rejected at call time with an "only supports
    AbstractVector outputs" error.

!!! warning
    With a forward [`prepare_derivative_cache`](@ref) cache, the returned Jacobian *and value*
    alias buffers owned by `cache` *only on the zero-allocation path*, taken when `f` is
    non-differentiable (carries no parameters of its own). On that path those buffers (reused
    as the gradient and Hessian paths do) are overwritten on the next call with the same
    cache, so `copy` them first if you need to retain them. A differentiable `f` (e.g. a closure
    capturing parameters) instead returns a freshly allocated Jacobian each call, as does a
    reverse [`prepare_pullback_cache`](@ref) cache.
"""
@unstable @inline function value_and_jacobian!!(
    cache::FCache, f::F, x::AbstractVector{<:IEEEFloat}
) where {F}
    _check_vector_argument(x; caller=(value_and_jacobian!!), cache=cache, dense=true)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), (f, x))
    total_dim = length(x)
    # No input dimensions: the chunk width resolves to zero, so there is no seed to sweep and no
    # chunk rule to call. The Jacobian is `length(f(x)) x 0`; evaluate the primal directly for
    # the value and the output length. Even an empty vector can be grown by `f`.
    if total_dim == 0
        snapshots, contexts = _snapshot_inputs!!(cache.input_snapshot, (x,))
        try
            y = _copy_output(f(x))
            Ty = _check_jacobian_output(y, eltype(x))
            return y, zeros(Ty, length(y), 0)
        finally
            _restore_inputs!!((x,), snapshots, contexts)
        end
    end
    # Packable seeds and J are cache-owned and reused; refreshing seeds leaves x untouched.
    seed = cache.gradient_seed
    Jref = cache.jacobian_buffer
    # Gate on `seed isa Tuple`, not just `!== nothing`: the packable seed is a 3-tuple, but
    # a `StructuredGradSeed` (e.g. a custom `DenseVector` whose `similar` returns a plain
    # `Vector`) can also reach here — destructuring it below would MethodError. A structured
    # seed instead falls through to the generic non-packable sweep.
    if seed isa Tuple && Jref !== nothing
        f_seed_stored, arg_seeds, _ = seed
        # Re-wrap the call-time `f` (the stored seed holds the prepare-time instance); `V
        # === NoDual` is guaranteed by packability, so this is a free isbits rewrap. The
        # sweep runs in a concretely-typed function barrier (this `@unstable` method would
        # otherwise box the seed's per-lane partials each iteration).
        f_seed = typeof(f_seed_stored)(f, tangent(f_seed_stored))
        return _fcache_jacobian_packable!!(
            cache, Jref, f_seed, arg_seeds[1], cache.gradient_chunk_size.width, total_dim, x
        )
    end
    # A differentiable `f` contributes to the prepared width W, so W can exceed length(x).
    # Guard every Jacobian write, including the first chunk's, against that smaller dimension.
    W = cache.gradient_chunk_size.width
    # One lift cache lets x's basis directions reach aliased captures in f too.
    seed_vs = tangent(zero_lifted(Val(W), (f, x)))
    f_seed = Lifted{typeof(f),W}(f, seed_vs[1])
    x_seed = Lifted{typeof(x),W}(x, seed_vs[2])  # partials reseeded in place per chunk
    cols(start_col) = ntuple(lane -> let slot = start_col + lane - 1
        slot <= total_dim ? slot : 0
    end, W)
    # Snapshot `x` into the cache buffer (the args copy, so `x` is element 1) before any
    # chunk runs `f`; restore before each subsequent chunk (so an in-place `f` does not
    # compound) and once at the end, leaving `x` unchanged.
    snapshots, contexts = _snapshot_inputs!!(cache.input_snapshot, (x,))
    try
        output = value_and_derivative!!(cache, f_seed, basis_lifted!!(x_seed, cols(1)))
        # Copy before the restores below: `x_seed` aliases the caller's `x`, so for an `f` that
        # returns its mutated argument `y === x`, the restore would rewrite the value we return.
        # Same reason the `value_and_derivative!!` methods copy their output.
        y = _copy_output(primal(output))
        Ty = _check_jacobian_output(y, eltype(x))
        J = zeros(Ty, length(y), total_dim)
        # Guard the first chunk too: `W` can exceed `total_dim` (it includes `f`'s dimensions), so
        # lanes past `total_dim` would write out of bounds of `J`'s `total_dim` columns.
        @inbounds for lane in 1:W
            lane <= total_dim || break
            J[:, lane] .= tangent(output, lane)
        end
        for start_col in (W + 1):W:total_dim
            _restore_inputs!!((x,), snapshots, contexts)
            seed_vs = tangent(zero_lifted(Val(W), (f, x)))
            f_seed = Lifted{typeof(f),W}(f, seed_vs[1])
            x_seed = Lifted{typeof(x),W}(x, seed_vs[2])
            output = value_and_derivative!!(
                cache, f_seed, basis_lifted!!(x_seed, cols(start_col))
            )
            @inbounds for lane in 1:W
                col = start_col + lane - 1
                col <= total_dim || break
                J[:, col] .= tangent(output, lane)
            end
        end
        return y, J
    finally
        # Leave `x` unchanged whether or not a chunk threw (each chunk ran on the original).
        _restore_inputs!!((x,), snapshots, contexts)
    end
end

@unstable @inline function value_and_jacobian!!(
    cache::Cache, f::F, x::AbstractVector{<:IEEEFloat}
) where {F}
    _check_vector_argument(x; caller=(value_and_jacobian!!), cache=cache, dense=true)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), (f, x))
    total_dim = length(x)
    y_cache = cache.y_cache
    Ty = _check_jacobian_output(y_cache, eltype(x))
    ȳ = zeros(Ty, length(y_cache))
    J = zeros(Ty, length(ȳ), total_dim)
    # Reverse mode restores any in-place mutation of `x` on the pullback, so — unlike the
    # forward `(::FCache)` method above, which snapshots `x` explicitly — each
    # `value_and_pullback!!` call below leaves `x` unchanged with no snapshot here.
    # Also covers an empty INPUT: `J` is already `length(y) x total_dim`, so a
    # zero-dimension input gives the `n x 0` Jacobian with no sweep to run.
    if isempty(ȳ) || total_dim == 0
        y, _ = value_and_pullback!!(cache, ȳ, f, x)
        return y, J
    end

    local y
    @inbounds for row in 1:length(ȳ)
        ȳ[row] = one(Ty)
        val, pb = value_and_pullback!!(cache, ȳ, f, x)
        row == 1 && (y = val)
        J[row, :] .= pb[2]
        ȳ[row] = zero(Ty)
    end

    return y, J
end

@unstable function value_and_jacobian!!(cache::Union{Cache,FCache}, f, x)
    # Reached only for inputs the methods above reject (`x` is not a dense
    # `AbstractVector{<:IEEEFloat}`). `_check_vector_argument` always throws
    # a specific message here; the explicit throw documents that this fallback
    # never returns a value.
    _check_vector_argument(x; caller=(value_and_jacobian!!), cache=cache, dense=true)
    return throw(
        ArgumentError(
            "value_and_jacobian!! only supports dense AbstractVector{<:IEEEFloat} " *
            "inputs; got $(typeof(x))",
        ),
    )
end

@unstable function value_and_jacobian!!(cache, f, x)
    throw(ArgumentError("value_and_jacobian!! only supports cache types Cache and FCache"))
end

# Multi-argument calls match no 3-arg method above; give a clear error rather than a raw,
# many-line MethodError (sibling `value_and_*!!` accept `f, x...`, so users may try it
# here).
@unstable function value_and_jacobian!!(cache, f, x, xs...)
    throw(
        ArgumentError(
            "value_and_jacobian!! supports only a single AbstractVector input; got " *
            "$(length(xs) + 1) arguments. Concatenate the inputs into one vector, or use " *
            "value_and_gradient!! / value_and_hvp!! for multi-argument functions.",
        ),
    )
end

# IT=Nothing specialisation: disambiguates against the Lifted-vararg and Tuple-vararg
# zero-arg overloads (Aqua detects the ambiguity without this more-specific method). The
# validate always throws an arity `PreparedCacheError` (`input_types_and_sizes` has the `f`
# entry, no args given).
function value_and_derivative!!(
    cache::FCache{R,Nothing,FG,GW,CF,S}
) where {R,FG,GW,CF,S<:Tuple}
    return _check_prepared_cache(cache.input_types_and_sizes, ())
end

function value_and_derivative!!(cache::FCache)
    return _check_prepared_cache(cache.input_types_and_sizes, ())
end

#
# Reverse mode — gradient and pullback
#

"""
    __value_and_pullback!!(rule, ȳ, f::CoDual, x::CoDual...; y_cache=nothing)

*Note:* this is not part of the public Mooncake.jl interface, and may change without
warning.

In-place version of `value_and_pullback!!` in which the arguments have been wrapped in
`CoDual`s. Note that any mutable data in `f` and `x` will be incremented in-place. As such,
if calling this function multiple times with different values of `x`, should be careful to
ensure that you zero-out the tangent fields of `x` each time.
"""
function __value_and_pullback!!(
    rule::R, ȳ::T, fx::Vararg{CoDual,N}; y_cache=nothing
) where {R,N,T}
    fx_fwds = tuple_map(to_fwds, fx)
    __verify_sig(rule, fx_fwds)
    out, pb!! = __call_rule(rule, fx_fwds)
    @assert _typeof(tangent(out)) == fdata_type(T)
    increment!!(tangent(out), fdata(ȳ))
    v = if y_cache === nothing
        _copy_output(primal(out))
    else
        _copy_to_output!!(y_cache, primal(out))
    end
    return v, tuple_map((f, r) -> tangent(fdata(tangent(f)), r), fx, pb!!(rdata(ȳ)))
end

function __verify_sig(rule::DerivedRule{<:Any,sig}, fx) where {sig}
    Pfx = typeof(__unflatten_codual_varargs(_isva(rule), fx, rule.nargs))
    if sig != Pfx
        msg = "signature of arguments, $Pfx, not equal to signature required by rule, $sig."
        throw(ArgumentError(msg))
    end
end

__verify_sig(rule::DebugRRule, fx) = __verify_sig(rule.rule, fx)

# rrule!! doesn't specify specific argument types which must be used, so there's nothing to
# check here.
__verify_sig(::typeof(rrule!!), fx::Tuple) = nothing

@static if VERSION < v"1.11-"
    # rrule!! is a plain Julia function (not an OpaqueClosure), so calling it directly is
    # safe on Julia 1.10; the `(rule::Any)` dispatch barrier is not needed here.
    @inline __call_rule(rule::typeof(rrule!!), args) = rule(args...)
end

"""
    __value_and_gradient!!(rule, f::CoDual, x::CoDual...)

*Note:* this is not part of the public Mooncake.jl interface, and may change without
warning.

Equivalent to `__value_and_pullback!!(rule, 1.0, f, x...)` -- assumes `f` returns a
`Float64`.

```jldoctest; setup = :(using Mooncake; import Mooncake: build_rrule, zero_tangent)
# Set up the problem.
f(x, y) = sum(x .* y)
x = [2.0, 2.0]
y = [1.0, 1.0]
rule = build_rrule(f, x, y)

# Allocate tangents. These will be written to in-place. You are free to re-use these if you
# compute gradients multiple times.
tf = zero_tangent(f)
tx = zero_tangent(x)
ty = zero_tangent(y)

# Do AD.
Mooncake.__value_and_gradient!!(
    rule, Mooncake.CoDual(f, tf), Mooncake.CoDual(x, tx), Mooncake.CoDual(y, ty)
)
# output

(4.0, (NoTangent(), [1.0, 1.0], [2.0, 2.0]))
```
"""
function __value_and_gradient!!(rule::R, fx::Vararg{CoDual,N}) where {R,N}
    fx_fwds = tuple_map(to_fwds, fx)
    __verify_sig(rule, fx_fwds)
    out, pb!! = __call_rule(rule, fx_fwds)
    y = primal(out)
    _check_scalar_output(y; caller=(value_and_gradient!!))
    return y, tuple_map((f, r) -> tangent(fdata(tangent(f)), r), fx, pb!!(one(y)))
end

"""
    value_and_pullback!!(rule, ȳ, f, x...; friendly_tangents=false)

Compute the value and pullback of `f(x...)`. If `friendly_tangents=false`, `ȳ` must be a
valid tangent for the primal return by `f(x...)`. If `friendly_tangents=true`, `ȳ` must be
of the same type as the primal returned by `f(x...)`.

`rule` should be constructed using `build_rrule`.

*Note:* There are lots of subtle ways to mis-use `value_and_pullback!!`, so we generally
recommend using `value_and_gradient!!` where possible.

*Note:* If calling `value_and_pullback!!` multiple times for various values of `x`, you
should use the same instance of `rule` each time.

Aliased inputs share tangent storage through one seeding cache. Each aliased position
reports the accumulated pullback for that storage. For example,
```julia
X = randn(5, 5)
rule = build_rrule(dot, X, X)
value_and_pullback!!(rule, 1.0, dot, X, X)
```
returns `2X` at both input positions.

*Note:* This method of `value_and_pullback!!` has to first call `zero_codual` on all of its
arguments. This may cause some additional allocations. If this is a problem in your
use-case, consider pre-allocating the `CoDual`s and calling the other method of this
function. The `CoDual`s should be primal-tangent pairs (as opposed to primal-fdata pairs).
There are lots of ways to get this wrong though, so we generally advise against doing this.
"""
# Returns NoCache when all primals are bits types (no mutable aliasing possible).
# Otherwise returns IdDict to handle aliased mutable buffers across the tuple of tangents.
_friendly_cache(fx::Tuple) = isbitstype(typeof(fx)) ? NoCache() : IdDict{Any,Any}()

# Convert the internal tangents `native` back to friendly (primal-shaped) tangents, sharing
# one aliasing cache across the tuple. Shared by the four (pullback/gradient) ×
# (rule/cached) friendly branches below.
@inline function _to_friendly(dests, fx, native)
    c = _friendly_cache(fx)
    return tuple_map((d, p, t) -> tangent_to_friendly!!(d, p, t, c), dests, fx, native)
end

# Share one seeding cache across arguments to preserve reverse fdata aliasing and count
# repeated storage once in `tangent_dim`. `_to_friendly` shares the conversion cache too.
@inline function _zero_tangents(fx::Tuple)
    c = _friendly_cache(fx)
    return tuple_map(x -> zero_tangent_internal(x, c), fx)
end

# @inline forces specialisation on Vararg with function-valued arguments, avoiding severe
# perf regressions. See https://github.com/chalk-lab/Mooncake.jl/issues/1020.
@inline function value_and_pullback!!(
    rule::R, ȳ, fx::Vararg{Any,N}; friendly_tangents=false
) where {R,N}
    if friendly_tangents
        ȳ_tangent = primal_to_tangent!!(zero_tangent(ȳ), ȳ)
        value, pb = __value_and_pullback!!(rule, ȳ_tangent, __create_coduals(fx)...)
        friendly_pb = _to_friendly(map(friendly_tangent_cache, fx), fx, pb)
        return value, friendly_pb
    end
    return __value_and_pullback!!(rule, ȳ, __create_coduals(fx)...)
end

"""
    value_and_gradient!!(rule, f, x...; friendly_tangents=false)

Equivalent to `value_and_pullback!!(rule, 1.0, f, x...)`, and assumes `f` returns a
`Union{Float16,Float32,Float64}`.

*Note:* There are lots of subtle ways to mis-use [`value_and_pullback!!`](@ref), so we
generally recommend using `Mooncake.value_and_gradient!!` (this function) where possible.
The docstring for [`value_and_pullback!!`](@ref) is useful for understanding this function
though.

An example:
```jldoctest; setup = :(using Mooncake; import Mooncake: build_rrule)
f(x, y) = sum(x .* y)
x = [2.0, 2.0]
y = [1.0, 1.0]
rule = build_rrule(f, x, y)
value_and_gradient!!(rule, f, x, y)

# output

(4.0, (NoTangent(), [1.0, 1.0], [2.0, 2.0]))
```
"""
@inline function value_and_gradient!!(
    rule::R, fx::Vararg{Any,N}; friendly_tangents=false
) where {R,N}
    if friendly_tangents
        value, gradient = __value_and_gradient!!(rule, __create_coduals(fx)...)
        friendly_gradient = _to_friendly(map(friendly_tangent_cache, fx), fx, gradient)
        return value, friendly_gradient
    end
    return __value_and_gradient!!(rule, __create_coduals(fx)...)
end

# `zero_tangent_internal(::Ptr)` already returns the bitcast placeholder `uninit_tangent` does, so
# pointers need no method of their own here (the single-argument `zero_tangent(::Ptr)` throws, which
# is why `zero_codual` does need one).
@inline _zero_codual_cached(x, c::MaybeCache) = CoDual(x, zero_tangent_internal(x, c))

function __create_coduals(args)
    try
        return tuple_map(CoDual, args, _zero_tangents(args))
    catch e
        if e isa StackOverflowError
            error(
                "Found a StackOverFlow error when trying to wrap inputs. This often " *
                "means that Mooncake.jl has encountered a self-referential type. " *
                "Mooncake.jl is not presently able to handle self-referential types, so " *
                "if you are indeed using a self-referential type somewhere, you will " *
                "need to refactor to avoid it if you wish to use Mooncake.jl.",
            )
        else
            rethrow(e)
        end
    end
end

"""
    prepare_pullback_cache(f, x...; config=Mooncake.Config())

Returns a cache used with [`value_and_pullback!!`](@ref). See that function for more info,
including the `config.friendly_tangents` output-tangent contract and how arguments sharing
one storage are handled.

The API guarantees that tangents are initialized at zero before the first autodiff pass.

!!! note
    Evaluates `f(x...)` twice during cache preparation: once on a deepcopy of the arguments
    to validate the output, and once during the differentiated pass. Non-reversible side
    effects (e.g. I/O such as printing) therefore occur twice; in-place mutations of the
    arguments are restored by the reverse pass and net to a single observable change.
"""
@unstable function prepare_pullback_cache(fx...; config=Config())

    # Clear global caches if requested.
    config.empty_cache && empty_mooncake_caches!()
    foreach(
        i -> _check_representable_input(ReverseMode(), Base._stable_typeof(fx[i]), i - 1),
        eachindex(fx),
    )

    # Check that the output of `fx` is supported.
    __exclude_func_with_unsupported_output(fx)

    # Construct rule and tangents.
    interp = get_interpreter(ReverseMode)
    rule = build_rrule(
        interp, Tuple{map(_typeof, fx)...}; config.debug_mode, config.silence_debug_messages
    )
    tangents = _zero_tangents(fx)
    y, rvs!! = __call_rule(rule, map((x, dx) -> CoDual(x, fdata(dx)), fx, tangents))

    # Snapshot before the pullback restores input mutations: an output aliasing a grown
    # input must retain its output size and contents in the cache.
    y_cache = _copy_output(primal(y))
    y_cache = _copy_to_output!!(y_cache, primal(y))
    output_type_and_size = _type_and_size(y_cache)

    # Run reverse-pass in order to reset stacks + state.
    rvs!!(zero_rdata(primal(y)))

    input_types_and_sizes = map(_type_and_size, fx)
    dests = config.friendly_tangents ? map(friendly_tangent_cache, fx) : nothing
    ȳ_cache = config.friendly_tangents ? zero_tangent(y_cache) : nothing
    return Cache(
        rule,
        y_cache,
        tangents,
        dests,
        ȳ_cache,
        input_types_and_sizes,
        output_type_and_size,
    )
end

"""
    value_and_pullback!!(cache::Cache, ȳ, f, x...; args_to_zero=(true, ...))

!!! info
    If `f(x...)` returns a scalar, you should use [`value_and_gradient!!`](@ref), not this
    function.

Computes a 2-tuple. The first element is `f(x...)`, and the second is a tuple containing the
pullback of `f` applied to `ȳ`. The first element is the component of the pullback
associated to any fields of `f`, the second w.r.t the first element of `x`, etc. If the
cache was prepared with `config.friendly_tangents=true`, the pullback uses the same types as
those of `f` and `x`. Otherwise, it uses the tangent types associated to `f` and `x`.

There are no restrictions on what `y = f(x...)` is permitted to return. However, `ȳ` must
be an acceptable tangent for `y`. If the cache was prepared with
`config.friendly_tangents=false`, this means that, for example, it must be true that
`tangent_type(typeof(y)) == typeof(ȳ)`. If the cache was prepared with
`config.friendly_tangents=true`, then `typeof(y) == typeof(ȳ)`.

As with all functionality in Mooncake, if `f` modifies itself or `x`, `value_and_pullback!!`
will return both to their original state as part of the process of computing the pullback.

!!! info
    `cache` must be the output of [`prepare_pullback_cache`](@ref), and (fields of) `f` and
    `x` must be of the same size and shape as those used to construct the `cache`. This is
    to ensure that the gradient can be written to the memory allocated when the `cache` was
    built.

!!! warning
    `cache` owns any mutable state returned by this function, meaning that mutable
    components of values returned by it will be mutated if you run this function again with
    different arguments. Therefore, if you need to keep the values returned by this function
    around over multiple calls to this function with the same `cache`, you should take a
    copy (using `copy` or `deepcopy`) of them before calling again.

!!! warning
    Arguments that share one storage — the same array in two positions (`dot(X, X)`), or an
    `Array` alongside its backing `Memory` — share one cotangent buffer, so each of those
    positions reports the whole accumulated cotangent rather than its own share of it. Do
    not add them together. Which arguments share a storage is part of the shape the cache
    was prepared for: where Mooncake can see the sharing (top-level arguments, and mutables
    nested in `Tuple`s and `NamedTuple`s), reusing the cache with a different pattern raises
    a `PreparedCacheError`.

The keyword argument `args_to_zero` is a tuple of boolean values specifying which cotangents
should be reset to zero before differentiation. It contains one boolean for each element of
`(f, x...)`. It is used for performance optimizations if you can guarantee that the initial
cotangent allocated in `cache` (created by `zero_tangent`) never needs to be zeroed out
again.

!!! danger
    Setting an entry to `false` skips resetting that argument's cotangent, so it keeps stale
    values across calls and can silently corrupt gradients — including those of *other*
    arguments, since reverse-mode rules propagate cotangents between them (the pullback of
    `A \\ b` derives `A`'s from `b`'s). It is guaranteed safe only when the argument holds
    no differentiable data (`tangent_type(typeof(arg)) === NoTangent`); a closure over data
    or a "constant" array does not qualify. See [issue
    #1238](https://github.com/chalk-lab/Mooncake.jl/issues/1238).

# Example Usage
```jldoctest; setup = :(using Mooncake)
f(x, y) = sum(x .* y)
x = [2.0, 2.0]
y = [1.0, 1.0]
cache = Mooncake.prepare_pullback_cache(f, x, y)
Mooncake.value_and_pullback!!(cache, 1.0, f, x, y)

# output

(4.0, (NoTangent(), [1.0, 1.0], [2.0, 2.0]))
```
"""
@inline function value_and_pullback!!(
    cache::Cache,
    ȳ,
    f::F,
    x::Vararg{Any,N};
    # A `false` entry is an unsafe optimization unless the arg holds no differentiable data;
    # see docstring / #1238.
    args_to_zero::NTuple=ntuple(Returns(true), Val(N + 1)),
) where {F,N}
    fx = (f, x...)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), fx)
    _check_tangent_aliasing(getfield(cache, :tangents), fx)
    tangents = tuple_map(set_to_zero_maybe!!, getfield(cache, :tangents), args_to_zero)
    coduals = tuple_map(CoDual, fx, tangents)
    if isnothing(cache.dests)
        return __value_and_pullback!!(cache.rule, ȳ, coduals...; y_cache=cache.y_cache)
    end
    ȳ_tangent = primal_to_tangent!!(cache.ȳ_cache, ȳ)
    value, pb = __value_and_pullback!!(
        cache.rule, ȳ_tangent, coduals...; y_cache=cache.y_cache
    )
    friendly_pb = _to_friendly(getfield(cache, :dests), fx, pb)
    return value, friendly_pb
end

"""
    prepare_gradient_cache(f, x...; config=Mooncake.Config())

Returns a cache used with [`value_and_gradient!!`](@ref). See that function for more info,
including the `config.friendly_tangents` output-tangent contract and how arguments sharing
one storage are handled. The same cache also serves [`value_and_pullback!!`](@ref) with a
scalar output tangent.

The API guarantees that tangents are initialized at zero before the first autodiff pass.

!!! note
    Calls `f(x...)` once during cache preparation.
"""
@unstable function prepare_gradient_cache(fx...; config=Config())
    config.empty_cache && empty_mooncake_caches!()
    foreach(
        i -> _check_representable_input(ReverseMode(), Base._stable_typeof(fx[i]), i - 1),
        eachindex(fx),
    )
    rule = build_rrule(fx...; config.debug_mode, config.silence_debug_messages)
    tangents = _zero_tangents(fx)
    y, rvs!! = __call_rule(rule, map((x, dx) -> CoDual(x, fdata(dx)), fx, tangents))
    _check_scalar_output(primal(y); caller=prepare_gradient_cache)
    rvs!!(zero_tangent(primal(y))) # run reverse-pass to reset stacks + state
    input_types_and_sizes = map(_type_and_size, fx)
    output_type_and_size = _type_and_size(primal(y))
    # Snapshot the (scalar) output into y_cache like prepare_pullback_cache, so a gradient
    # Cache is also a well-formed pullback Cache. `_copy_output` suffices for the isbits
    # scalar output — no `_copy_to_output!!` fill needed (and the gradient run path never
    # reads it anyway).
    y_cache = _copy_output(primal(y))
    dests = config.friendly_tangents ? map(friendly_tangent_cache, fx) : nothing
    # A friendly gradient cache also serves pullbacks, which require this output-tangent buffer.
    ȳ_cache = config.friendly_tangents ? zero_tangent(y_cache) : nothing
    return Cache(
        rule,
        y_cache,
        tangents,
        dests,
        ȳ_cache,
        input_types_and_sizes,
        output_type_and_size,
    )
end

"""
    value_and_gradient!!(cache::Cache, f, x...; args_to_zero=(true, ...))

Computes a 2-tuple. The first element is `f(x...)`, and the second is a tuple containing the
gradient of `f` w.r.t. each argument. The first element is the gradient w.r.t any
differentiable fields of `f`, the second w.r.t the first element of `x`, etc. If the cache
was prepared with `config.friendly_tangents=true`, the gradient uses the same types as those
of `f` and `x`. Otherwise, it uses the tangent types associated to `f` and `x`.

Assumes that `f` returns a `Union{Float16, Float32, Float64}`.

As with all functionality in Mooncake, if `f` modifies itself or `x`, `value_and_gradient!!`
will return both to their original state as part of the process of computing the gradient.

!!! info
    `cache` must be the output of [`prepare_gradient_cache`](@ref), and (fields of) `f` and
    `x` must be of the same size and shape as those used to construct the `cache`. This is
    to ensure that the gradient can be written to the memory allocated when the `cache` was
    built.

!!! warning
    `cache` owns any mutable state returned by this function, meaning that mutable
    components of values returned by it will be mutated if you run this function again with
    different arguments. Therefore, if you need to keep the values returned by this function
    around over multiple calls to this function with the same `cache`, you should take a
    copy (using `copy` or `deepcopy`) of them before calling again.

!!! warning
    Arguments that share one storage — the same array in two positions (`dot(X, X)`), or an
    `Array` alongside its backing `Memory` — share one cotangent buffer, so each of those
    positions reports the whole accumulated cotangent rather than its own share of it. Do
    not add them together. Which arguments share a storage is part of the shape the cache
    was prepared for: where Mooncake can see the sharing (top-level arguments, and mutables
    nested in `Tuple`s and `NamedTuple`s), reusing the cache with a different pattern raises
    a `PreparedCacheError`.

The keyword argument `args_to_zero` is a tuple of boolean values specifying which cotangents
should be reset to zero before differentiation. It contains one boolean for each element of
`(f, x...)`. It is used for performance optimizations if you can guarantee that the initial
cotangent allocated in `cache` (created by `zero_tangent`) never needs to be zeroed out
again.

!!! danger
    Setting an entry to `false` skips resetting that argument's cotangent, so it keeps stale
    values across calls and can silently corrupt gradients — including those of *other*
    arguments, since reverse-mode rules propagate cotangents between them (the pullback of
    `A \\ b` derives `A`'s from `b`'s). It is guaranteed safe only when the argument holds
    no differentiable data (`tangent_type(typeof(arg)) === NoTangent`); a closure over data
    or a "constant" array does not qualify. See [issue
    #1238](https://github.com/chalk-lab/Mooncake.jl/issues/1238).

# Example Usage
```jldoctest; setup = :(using Mooncake)
f(x, y) = sum(x .* y)
x = [2.0, 2.0]
y = [1.0, 1.0]
cache = prepare_gradient_cache(f, x, y)
value_and_gradient!!(cache, f, x, y)

# output

(4.0, (NoTangent(), [1.0, 1.0], [2.0, 2.0]))
```
"""
@inline function value_and_gradient!!(
    cache::Cache,
    f::F,
    x::Vararg{Any,N};
    # A `false` entry is an unsafe optimization unless the arg holds no differentiable data;
    # see docstring / #1238.
    args_to_zero::NTuple=ntuple(Returns(true), Val(N + 1)),
) where {F,N}
    fx = (f, x...)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), fx)
    _check_tangent_aliasing(getfield(cache, :tangents), fx)
    tangents = tuple_map(set_to_zero_maybe!!, getfield(cache, :tangents), args_to_zero)
    coduals = tuple_map(CoDual, fx, tangents)
    if isnothing(cache.dests)
        return __value_and_gradient!!(cache.rule, coduals...)
    end
    value, gradient = __value_and_gradient!!(cache.rule, coduals...)
    friendly_gradient = _to_friendly(getfield(cache, :dests), fx, gradient)
    return value, friendly_gradient
end

# IsbitsGradSeed's cursor handles only real scalar dimensions nested in tuples,
# NamedTuples or ImmutableDual. Complex, uninitialised and other shapes must fall back.
_all_real_scalars(::Type{NoDual}) = true
_all_real_scalars(::Type{<:Nfwd.NDual{T}}) where {T<:IEEEFloat} = true
function _all_real_scalars(::Type{T}) where {T<:Tuple}
    all(_all_real_scalars, fieldtypes(T))
end
function _all_real_scalars(::Type{NamedTuple{names,T}}) where {names,T}
    _all_real_scalars(T)
end
_all_real_scalars(::Type{<:ImmutableDual{T}}) where {T} = _all_real_scalars(T)
_all_real_scalars(::Type) = false

# Gather `(NDualArray, Array)` leaf pairs from a forward V `v` and the parallel reverse
# tangent `g`, in dimension order. Returns a flat tuple of pairs, or `nothing` if any dimension is not
# array-backed. `dict` guards against aliasing/cycles: a revisited array or mutable wrapper
# means the flat leaf table would mis-order or double-count dimensions (the generic path dedups
# instead), so bail to `nothing` and let that path handle it.
_grad_leaves(::NoDual, @nospecialize(g), dict) = ()
function _grad_leaves(
    v::Nfwd.NDualArray{T}, g::AbstractArray{T}, dict
) where {T<:Nfwd.NDualEltype}
    haskey(dict, v) && return nothing
    dict[v] = nothing
    return ((v, g),)
end

# A complex element owns TWO consecutive dimensions, the real direction then the imaginary one,
# matching the order `tangent_dim` counts them and `basis_lifted!!` seeds them.
@inline _leaf_dims(v::Nfwd.NDualArray{<:IEEEFloat}) = length(v.primal)
@inline _leaf_dims(v::Nfwd.NDualArray{<:Complex{<:IEEEFloat}}) = 2 * length(v.primal)
function _grad_leaves(v::Tuple, g::Tuple, dict)
    return if length(v) == length(g)
        _cat_leaves(map((a, b) -> _grad_leaves(a, b, dict), v, g))
    else
        nothing
    end
end
function _grad_leaves(v::NamedTuple{ns}, g::NamedTuple{ns}, dict) where {ns}
    return _grad_leaves(values(v), values(g), dict)
end
_grad_leaves(v::ImmutableDual, g::Tangent, dict) = _grad_leaves(v.fields, g.fields, dict)
function _grad_leaves(v::MutableDual, g::MutableTangent, dict)
    haskey(dict, v) && return nothing
    dict[v] = nothing
    return _grad_leaves(v.fields, g.fields, dict)
end
_grad_leaves(@nospecialize(v), @nospecialize(g), dict) = nothing  # scalar/complex/abstract/uninit/mismatch

# Restore primal and dual field bindings each chunk: rebinding would orphan the cached
# leaf table. Re-gathering would allocate. `_grad_leaves` already excludes repeats and
# cycles, so no cycle guard is needed here.
_save_seed_bindings(::Nfwd.NDualArray, p) = ()
# Composite NoDual leaves can own mutable descendants outside the dual leaf table.
# Use generic snapshot/restore for these inputs, including changed extents and bindings.
_save_seed_bindings(::NoDual, p) = isbitstype(typeof(p)) ? () : nothing
_save_seed_bindings(v::ImmutableDual, p) = _save_seed_bindings(v.fields, p)
function _save_seed_bindings(v::MutableDual, p)
    fields = ntuple(i -> getfield(p, i), fieldcount(typeof(p)))
    return _cat_leaves((((v, (v.fields,)), (p, fields)), _save_seed_bindings(v.fields, p)))
end
_save_seed_bindings(v::Tuple, p) = _cat_leaves(map(_save_seed_bindings, v, p))
function _save_seed_bindings(v::NamedTuple{ns}, p) where {ns}
    return _cat_leaves(map(n -> _save_seed_bindings(getfield(v, n), getfield(p, n)), ns))
end

@inline _restore_seed_bindings!(::Tuple{}) = nothing
@inline function _restore_seed_bindings!(rs::Tuple)
    obj, fields = first(rs)
    for i in eachindex(fields)
        getfield(obj, i) === fields[i] ||
            ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), obj, i - 1, fields[i])
    end
    return _restore_seed_bindings!(Base.tail(rs))
end

# One row per differentiable leaf: the cache-owned seed, the gradient buffer it fills, and the
# range of derivative dimensions it owns. Assigning that range once is what lets the sweeps index
# by it instead of threading a running offset through every walk, and makes the total a field read
# rather than a second recursion over the same rows.
function _tangent_layout(arg_seeds::Tuple, grad_bufs::Tuple)
    dict = IdDict{Any,Any}()
    leaves = _cat_leaves(
        map((s, g) -> _grad_leaves(tangent(s), g, dict), arg_seeds, grad_bufs)
    )
    return leaves === nothing ? nothing : _tangent_layout(leaves, 1)
end

@inline _tangent_layout(::Tuple{}, ::Int) = ()
@inline function _tangent_layout(ls::Tuple, first_dim::Int)
    v, g = first(ls)
    n = _leaf_dims(v)
    dims = first_dim:(first_dim + n - 1)
    return ((v, g, dims), _tangent_layout(Base.tail(ls), first_dim + n)...)
end

#
# `value_and_gradient!!` generic chunked path
#

"""
    value_and_gradient!!(cache::FCache, f, x...)

Compute the value and gradient of the scalar-returning `f` at `x...` using forward-mode AD
(the gradient is assembled from standard-basis directional derivatives, evaluated in chunks
of the cache's resolved chunk width). This overload exists so callers can prepare a forward
cache once, then use it either for directional derivatives via
[`value_and_derivative!!`](@ref) or for full gradients.

All differentiable input shapes are chunked (a zero-dimension input is evaluated once). Four shape
families take a zero-allocation path that reuses cache-owned seeds: (0) a single scalar
`x::IEEEFloat`; and, with a non-differentiable isbits `f`, (1) one or more same-element-type
dense float vectors; (2) array-backed tuples/NamedTuples/structs with real or complex
IEEEFloat array leaves, including mixed leaf eltypes, and isbits nondifferentiable state;
(3) tuples/NamedTuples/immutable structs of real float scalars. (A non-isbits `f` on paths
1–2 costs one `Lifted` allocation per call, not per chunk.) Unsupported leaf shapes,
possibly-uninitialised fields, non-isbits NoDual state, or a differentiable `f` use the
generic chunked path, which allocates a fresh seed per chunk.

Arguments that share one storage are refused. Forward mode gives each argument its own
tangent storage, so two arguments over one array are two independent directions rather than
one shared derivative, and the standard-basis sweep cannot represent that. Repeated leaves
*within* one argument are fine — `f((a, a))` lifts through a single aliasing cache and gives
the same gradient reverse mode does — but sharing that is not object identity (a `reshape`, a
`view`, an `Array` beside its backing `Memory`) is refused wherever it appears, because the
lift keys on the object. For those, use [`value_and_derivative!!`](@ref) with one tangent
covering every position the storage occupies, or reverse mode, which accumulates into it.

The arguments in `x` are left unchanged: an in-place `f` mutates only cache-owned buffers
(the zero-allocation paths copy `x` into them each chunk) or a cache-owned snapshot that is
restored (the generic path). `f` itself is not snapshotted — a callable that mutates its own
fields is not restored.

!!! warning
    `cache` owns any mutable state returned by this function: mutable components of the
    returned gradients will be overwritten if you call this function again with the same
    `cache`. Take a copy (`copy` / `deepcopy`) of anything you need to keep across calls.
"""
# Shared finalisation for every value_and_gradient!! path: return the native gradient
# directly, or (when the cache was prepared with `friendly_tangents=true`) convert it to a
# primal-shaped one. Type-stable — each caller's types are concrete at the call site.
@inline function _finalize_gradient(cache::FCache, y, native_gradients, input_primals)
    isnothing(cache.input_tangents) && return y, native_gradients
    friendly_gradients = _copy_to_output!!(cache.friendly_gradients, input_primals)
    return y,
    tangent_to_primal_internal!!(
        friendly_gradients, native_gradients, _friendly_cache(friendly_gradients)
    )
end

# The generic fallback uses runtime width in seed type parameters, deliberately trading
# type stability and fresh per-chunk allocations for shape generality. Concrete fast
# paths retain type stability and zero allocation.
@unstable function value_and_gradient!!(cache::FCache, f::F, x::Vararg{Any,N}) where {F,N}
    # Array-backed structured inputs take the zero-allocation leaf-table path; scalar-only
    # structured inputs take the isbits concrete-barrier path.
    _check_primal_aliasing(x)
    _check_gradient_input_aliasing(cache)
    seed = cache.gradient_seed
    seed isa StructuredGradSeed && return _structured_gradient!!(cache, f, x, seed)
    seed isa IsbitsGradSeed && return _isbits_gradient!!(cache, f, x, seed)
    input_primals = (f, x...)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), input_primals)
    native_gradients = let workspace = cache.gradient_workspace[]
        if isnothing(workspace)
            workspace = _zero_tangents(input_primals)
            cache.gradient_workspace[] = workspace
            workspace
        else
            zeroed = tuple_map(set_to_zero!!, workspace)
            cache.gradient_workspace[] = zeroed
            zeroed
        end
    end
    # `tangent_dim` walks the tangent; reuse the freshly-built/zeroed workspace tangent.
    total_dim = tangent_dim(native_gradients)

    if total_dim == 0
        # Snapshot/restore like the chunked loop below: forward slots alias the user's
        # storage, so an in-place `f` over a zero-dimension input (e.g. `Vector{Int}`) would
        # otherwise mutate it.
        input_snapshot, restore_contexts = _snapshot_inputs!!(
            cache.input_snapshot, Base.tail(input_primals)
        )
        # `_finalize_gradient` reads the input primals, so it runs after the restore.
        y = try
            primal(
                __call_rule(
                    cache.single_rule, tuple_map(lift, input_primals, native_gradients)
                ),
            )
        finally
            _restore_inputs!!(Base.tail(input_primals), input_snapshot, restore_contexts)
        end
        _check_scalar_output(y; caller=(value_and_gradient!!), cache=cache)
        return _finalize_gradient(cache, y, native_gradients, input_primals)
    end

    # Seed the whole input tuple through one cache, split into width-W slots, and scatter
    # lane coefficients against width-1 reverse basis tangents. Guard the final short chunk:
    # out-of-range seeds are zero lanes, not gradient entries.
    W = cache.gradient_chunk_size.width
    nfields = Val(fieldcount(typeof(input_primals)))
    P = typeof(input_primals)
    # Snapshot the inputs into the cache buffer before any chunk runs `f`; restore from it
    # before each subsequent chunk (so an in-place `f` does not compound) and once at the
    # end.
    input_snapshot, restore_contexts = _snapshot_inputs!!(
        cache.input_snapshot, Base.tail(input_primals)
    )
    # Single sweep over all chunks. `total_dim >= 1` here (the zero-dimension case returned
    # above), so the first iteration always runs and assigns `y`; its leading input restore
    # is a no-op (the snapshot was just taken with no intervening `f`).
    local y
    try
        for start_slot in 1:W:total_dim
            _restore_inputs!!(Base.tail(input_primals), input_snapshot, restore_contexts)
            slots = ntuple(lane -> start_slot + lane - 1, W)
            lanes = ntuple(
                lane -> last(
                    unlift(
                        basis_lifted!!(zero_lifted(Val(1), input_primals), (slots[lane],)),
                    ),
                ),
                W,
            )
            vs = tangent(basis_lifted!!(zero_lifted(Val(W), input_primals), slots))
            lifted = ntuple(
                i -> Lifted{fieldtype(P, i),W}(input_primals[i], vs[i]), nfields
            )
            output = value_and_derivative!!(cache, lifted...)
            if start_slot == 1
                y = primal(output)
                _check_scalar_output(y; caller=(value_and_gradient!!), cache=cache)
            end
            for lane in 1:W
                slot = start_slot + lane - 1
                slot <= total_dim || break
                coeff = Float64(tangent(output, lane))
                # `lanes[lane]` is the per-input-field reverse-tangent tuple for this lane,
                # parallel to `native_gradients`; scatter each field's contribution directly (no
                # input-major transpose).
                native_gradients = tuple_map(
                    (g, lt) -> if lt isa NoTangent
                        g
                    else
                        increment!!(g, _scale_gradient_basis(coeff, lt))
                    end,
                    native_gradients,
                    lanes[lane],
                )
            end
        end
    finally
        # Leave the inputs unchanged whether or not a chunk threw (each chunk ran on the
        # original).
        _restore_inputs!!(Base.tail(input_primals), input_snapshot, restore_contexts)
    end

    return _finalize_gradient(cache, y, native_gradients, input_primals)
end

# This assembly scales a standard-basis tangent, so zero entries are inactive
# coordinates, even when the active derivative is infinite. Keep this policy local:
# ordinary tangent scaling must still implement IEEE multiplication.
_scale_gradient_basis(a, t) = _scale_gradient_basis(a, t, IdDict{Any,Any}())
_scale_gradient_basis(a, t, c) = _scale_internal(c, a, t)
_scale_gradient_basis(a, t::IEEEFloat, c) = iszero(t) ? t : oftype(t, a * t)
function _scale_gradient_basis(a, t::Complex{<:IEEEFloat}, c)
    return complex(
        _scale_gradient_basis(a, real(t), c), _scale_gradient_basis(a, imag(t), c)
    )
end
function _scale_gradient_basis(a, t::Union{Tuple,NamedTuple}, c)
    return map(x -> _scale_gradient_basis(a, x, c), t)
end
function _scale_gradient_basis(a, t::T, c) where {T<:PossiblyUninitTangent}
    return is_init(t) ? T(_scale_gradient_basis(a, val(t), c)) : T()
end
_scale_gradient_basis(a, t::Tangent, c) = typeof(t)(_scale_gradient_basis(a, t.fields, c))
function _scale_gradient_basis(a, t::MutableTangent, c)
    haskey(c, t) && return c[t]
    out = typeof(t)()
    c[t] = out
    out.fields = _scale_gradient_basis(a, t.fields, c)
    return out
end
function _scale_gradient_basis(a, t::_BuiltinArrays, c)
    haskey(c, t) && return c[t]
    out = similar(t)
    c[t] = out
    for i in eachindex(t)
        isassigned(t, i) && (out[i] = _scale_gradient_basis(a, t[i], c))
    end
    return out
end
@static if VERSION >= v"1.11.0-rc4"
    function _scale_gradient_basis(a, t::MemoryRef, c)
        mem = _scale_gradient_basis(a, t.mem, c)
        return Core.memoryrefnew(Core.memoryrefnew(mem), Core.memoryrefoffset(t), false)
    end
end
function _scale_gradient_basis(a, t::IdDict, c)
    haskey(c, t) && return c[t]
    out = empty(t)
    c[t] = out
    for (k, v) in t
        out[k] = _scale_gradient_basis(a, v, c)
    end
    return out
end

#
# `value_and_gradient!!` fast paths
#

# See `value_and_gradient!!(::FCache, ...)` for fast-path admission and fallback shapes.

# Scalar `value_and_gradient!!` fast path: a single width-1 forward evaluation through
# `cache.single_rule`. A scalar input has one degree of freedom, so there is nothing to
# chunk; this avoids the generic path's standard-basis seeding and lane accumulation.
@inline function value_and_gradient!!(cache::FCache, f::F, x::T) where {F,T<:IEEEFloat}
    # A differentiable `f` carries its own degrees of freedom: the width-1 single-seed run
    # below cannot represent them (and `lift(f, NoTangent())` would seed uninitialised
    # tangent storage), so fall back to the generic chunked path, which sweeps `f`'s dimensions
    # too.
    tangent_type(F) === NoTangent ||
        return invoke(value_and_gradient!!, Tuple{FCache,Any,Vararg{Any}}, cache, f, x)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), (f, x))
    output = __call_rule(cache.single_rule, (lift(f, NoTangent()), lift(x, one(x))))
    y = primal(output)
    _check_scalar_output(y; caller=(value_and_gradient!!), cache=cache)
    native_gradients = (NoTangent(), last(unlift(output)))
    return _finalize_gradient(cache, y, native_gradients, (f, x))
end

# The flat sweep reuses same-eltype vector seeds. Restore primals and zero partials per
# chunk to prevent compounded mutation; structured seeds and differentiable callables
# are delegated to the generic entry point.
function value_and_gradient!!(
    cache::FCache, f::F, x1::AbstractVector{T}, xs_rest::Vararg{AbstractVector{T},Nm1}
) where {F,T<:IEEEFloat,Nm1}
    # Leading `x1` binds T; a bare Vararg leaves it unbound at N=0 (Aqua).
    xs = (x1, xs_rest...)
    N = Nm1 + 1
    _check_primal_aliasing(xs)
    _check_gradient_input_aliasing(cache)
    seed = cache.gradient_seed
    # Only destructure flat tuple seeds; other shapes use the generic method's dispatch.
    # `invoke` avoids re-entering this float-vector method.
    seed isa Tuple ||
        return invoke(value_and_gradient!!, Tuple{FCache,Any,Vararg{Any}}, cache, f, xs...)
    # Validate once, on the packable path only (the fallback validates in the generic
    # method).
    input_primals = (f, xs...)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), input_primals)
    f_seed_stored, arg_seeds, grad_bufs = seed
    # Re-wrap the CALL-time `f` (the stored seed holds the prepare-time instance, and a
    # non-differentiable callable can still carry primal-visible state). `V === NoDual` is
    # guaranteed by the packability gate, so this is a free isbits rewrap.
    f_seed = typeof(f_seed_stored)(f, tangent(f_seed_stored))
    total_dim = sum(length, xs)
    # `prepare_derivative_cache` built the seeds at exactly this width, so the cache field
    # is the authoritative source (kept in lockstep with `_lifted_width(f_seed)`).
    W = cache.gradient_chunk_size.width
    # Bind `y` from the primal below (not a fabricated `zero(T)`), so the return type stays
    # concrete even when `f`'s output float type differs from the input eltype. The scatter
    # writes every gradient position exactly once per sweep, so `grad_bufs` needs no
    # pre-zeroing.
    local y
    s = 1
    while s <= total_dim
        # Re-seed every chunk: an in-place `f` mutates the seed primals (and its rule scales
        # the partials) during the previous chunk's run, so restore the primals from the
        # user's arrays and zero the partials before setting this chunk's basis directions —
        # otherwise the mutation compounds across chunks and the gradient is silently wrong.
        off = 0
        @inbounds for i in 1:N
            nda = arg_seeds[i].rep
            # Single-leaf inline of the structured path's `_refresh_seed!` (restore seed
            # primal).
            _reset_seed_extent!(nda, length(grad_bufs[i]))
            copyto!(nda.primal, xs[i])
            len = length(xs[i])
            Nfwd._zero_seed!(nda)
            for lane in 1:W
                slot = s + lane - 1
                (slot <= total_dim && off < slot <= off + len) &&
                    Nfwd._set_partial!(nda, slot - off, lane, one(T))
            end
            off += len
        end
        output = value_and_derivative!!(cache, f_seed, arg_seeds...)
        yv = primal(output)
        _check_scalar_output(yv; caller=(value_and_gradient!!), cache=cache)
        y = yv
        off = 0
        @inbounds for i in 1:N
            nda = arg_seeds[i].rep
            gb = grad_bufs[i]
            len = length(xs[i])
            for lane in 1:W
                slot = s + lane - 1
                if slot <= total_dim && off < slot <= off + len
                    gb[slot - off] = tangent(output, lane)
                end
            end
            off += len
        end
        s += W
    end
    native_gradients = (NoTangent(), grad_bufs...)
    return _finalize_gradient(cache, y, native_gradients, input_primals)
end

# Reset cache-owned vector storage to the prepared extent BEFORE strict input refresh.
# The gradient buffer/leaf table retains that extent even when a rule resizes its seed.
_reset_seed_extent!(v, len) = nothing
function _reset_seed_extent!(v::Nfwd.NDualArray{T,N,1,<:Vector}, len) where {T,N}
    length(v.primal) == len || resize!(v.primal, len)
    block = getfield(v, :partials_block)
    storage = getfield(block, :parent)
    length(storage) == N * len || resize!(storage, N * len)
    return nothing
end

# Refresh a preallocated seed's `NDualArray.primal` leaves from the current call's input `x`
# (a parallel walk of the forward V and the primal; the inner duals read `.primal`, so this
# is all the per-call primal state the rule needs). Type-stable, allocation-free.
_refresh_seed!(::NoDual, @nospecialize(x)) = nothing
function _refresh_seed!(v::Nfwd.NDualArray{T}, x::AbstractArray) where {T<:Nfwd.NDualEltype}
    # Validate nested sizes, not just lengths: a reshaped input must not silently run on
    # the stale cache-owned shape. Top-level cache checks cannot see this.
    size(v.primal) == size(x) || throw(
        PreparedCacheError(
            "Prepared cache mismatch: a nested array argument has size $(size(x)) but " *
            "the cache was built for size $(size(v.primal)). Rebuild the cache for the " *
            "new shape.",
        ),
    )
    copyto!(v.primal, x)
    return nothing
end
_refresh_seed!(v::Union{ImmutableDual,MutableDual}, x) = _refresh_seed!(v.fields, x)
@generated function _refresh_seed!(v::Tuple, x::Tuple)
    return Expr(
        :block,
        (:(_refresh_seed!(v[$i], x[$i])) for i in 1:length(v.parameters))...,
        :(nothing),
    )
end
@generated function _refresh_seed!(v::NamedTuple{ns}, x) where {ns}
    return Expr(
        :block,
        (
            :(_refresh_seed!(getfield(v, $(QuoteNode(n))), getfield(x, $(QuoteNode(n)))))
            for n in ns
        )...,
        :(nothing),
    )
end
# Refresh call-time nondifferentiable state too: types and top-level sizes do not detect
# changed struct fields or view indices. Preserve cache-owned differentiable buffers and
# return unchanged objects to retain zero allocation. Unsupported V shapes must error.
_refresh_nondiff(::Nfwd.NDualArray, p, _x) = p

# A NoDual slot must remain nondifferentiable at call time. Refuse an abstract field
# changing from Int to Float before an incoherent slot reaches the dual IR.
function _refresh_nondiff(::NoDual, p, x)
    if tangent_type(_typeof(x)) !== NoTangent
        throw(
            PreparedCacheError(
                "Prepared cache mismatch: an argument position holding a non-differentiable " *
                "$(_typeof(p)) at preparation time now holds a differentiable $(_typeof(x)). The " *
                "cache has no derivative storage for it. Rebuild the cache for the new types.",
            ),
        )
    end
    # Copy same-type non-isbits state into cache storage so in-place mutation cannot reach
    # user storage or compound across chunks. `_copy_to_output!!` preserves cycles, aliases,
    # and const fields. Bits values and changed types pass through for the caller to rebuild.
    P = typeof(x)
    return typeof(p) === P && !isbitstype(P) ? _copy_to_output!!(p, x) : x
end

# An ARRAY has no fields to copy, so it needs its own method for the same rule: the call's contents
# go into the cache's own buffer, which additionally lets the size mismatch be reported here.
function _refresh_nondiff(::NoDual, p::A, x::A) where {A<:Array}
    size(p) == size(x) || throw(
        PreparedCacheError(
            "Prepared cache mismatch: a non-differentiable array argument has size $(size(x)) " *
            "but the cache was built for size $(size(p)). Rebuild the cache for the new shape.",
        ),
    )
    copyto!(p, x)
    return p
end

@generated function _refresh_nondiff(v::ImmutableDual{NT}, p::P, x) where {NT<:NamedTuple,P}
    ns = fieldnames(NT)
    isempty(ns) && return :p
    syms = [Symbol(:f_, i) for i in eachindex(ns)]
    body = [
        :(
            $(syms[i]) = _refresh_nondiff(
                getfield(vv, $(QuoteNode(ns[i]))),
                getfield(p, $(QuoteNode(ns[i]))),
                getfield(x, $(QuoteNode(ns[i]))),
            )
        ) for i in eachindex(ns)
    ]
    unchanged = foldl(
        (a, b) -> :($a && $b),
        [:($(syms[i]) === getfield(p, $(QuoteNode(ns[i])))) for i in eachindex(ns)],
    )
    return quote
        vv = getfield(v, :fields)
        $(body...)
        $unchanged && return p
        return _new_($P, $(syms...))
    end
end

# A mutable primal is refreshed in place, so the seed keeps its identity. Writing only fields that
# actually changed also keeps a `const` field silent unless the reuse genuinely conflicts with it.
@generated function _refresh_nondiff(v::MutableDual{NT}, p, x) where {NT<:NamedTuple}
    ns = fieldnames(NT)
    body = [
        quote
            let f = _refresh_nondiff(
                    getfield(vv, $(QuoteNode(n))),
                    getfield(p, $(QuoteNode(n))),
                    getfield(x, $(QuoteNode(n))),
                )
                f === getfield(p, $(QuoteNode(n))) || setfield!(p, $(QuoteNode(n)), f)
            end
        end for n in ns
    ]
    return quote
        vv = getfield(v, :fields)
        $(body...)
        return p
    end
end

@generated function _refresh_nondiff(v::Tuple, p::Tuple, x::Tuple)
    n = length(v.parameters)
    n == 0 && return :p
    syms = [Symbol(:f_, i) for i in 1:n]
    body = [:($(syms[i]) = _refresh_nondiff(v[$i], p[$i], x[$i])) for i in 1:n]
    unchanged = foldl((a, b) -> :($a && $b), [:($(syms[i]) === p[$i]) for i in 1:n])
    return quote
        $(body...)
        $unchanged && return p
        return tuple($(syms...))
    end
end

@generated function _refresh_nondiff(v::NamedTuple{ns}, p, x) where {ns}
    isempty(ns) && return :p
    syms = [Symbol(:f_, i) for i in eachindex(ns)]
    body = [
        :(
            $(syms[i]) = _refresh_nondiff(
                getfield(v, $(QuoteNode(ns[i]))),
                getfield(p, $(QuoteNode(ns[i]))),
                getfield(x, $(QuoteNode(ns[i]))),
            )
        ) for i in eachindex(ns)
    ]
    unchanged = foldl(
        (a, b) -> :($a && $b),
        [:($(syms[i]) === getfield(p, $(QuoteNode(ns[i])))) for i in eachindex(ns)],
    )
    return quote
        $(body...)
        $unchanged && return p
        return typeof(p)(tuple($(syms...)))
    end
end

# Refresh nondifferentiable state and differentiable buffers. Return the stored tuple
# when unchanged: rebuilding non-isbits Lifteds would break zero allocation.
@generated function _refresh_all!(arg_seeds::Tuple, xs::Tuple)
    n = length(arg_seeds.parameters)
    n == 0 && return :arg_seeds
    syms = [Symbol(:p_, i) for i in 1:n]
    rebuild = [
        :(
            $(syms[i]) = _refresh_nondiff(
                tangent(arg_seeds[$i]), primal(arg_seeds[$i]), xs[$i]
            )
        ) for i in 1:n
    ]
    refresh = [:(_refresh_seed!(tangent(arg_seeds[$i]), xs[$i])) for i in 1:n]
    unchanged = foldl(
        (a, b) -> :($a && $b), [:($(syms[i]) === primal(arg_seeds[$i])) for i in 1:n]
    )
    rebuilt = [:(typeof(arg_seeds[$i])($(syms[i]), tangent(arg_seeds[$i]))) for i in 1:n]
    return quote
        $(rebuild...)
        $(refresh...)
        $unchanged && return arg_seeds
        return tuple($(rebuilt...))
    end
end

# Unroll layout rows for type stability and zero allocation. Clear ALL partials each chunk:
# in-place f can dirty inactive entries too (O(total_dim²) work over the sweep).
# Keep zeroing separate from seeding so shared blocks cannot erase an earlier row's seed.
@inline _zero_seeds!(::Tuple{}) = nothing
@inline function _zero_seeds!(ls::Tuple)
    Nfwd._zero_seed!(first(ls)[1])
    return _zero_seeds!(Base.tail(ls))
end
@inline _seed_chunk!(::Tuple{}, s, W) = nothing
@inline function _seed_chunk!(ls::Tuple, s, W)
    nda, _, dims = first(ls)
    _seed_leaf!(nda, dims, s, W)
    return _seed_chunk!(Base.tail(ls), s, W)
end

# Element i's lane k. Storage layout (element-major block on 1.11+, per-lane arrays on 1.10) is
# hidden by `Nfwd._set_partial!`; rank-agnostic, so matrix leaves work too.
@inline function _seed_leaf!(nda::Nfwd.NDualArray{T}, dims, s, W) where {T<:IEEEFloat}
    o = one(T)
    @inbounds for (i, d) in enumerate(dims)
        s <= d <= s + W - 1 && Nfwd._set_partial!(nda, i, d - s + 1, o)
    end
    return nothing
end
# The two dimensions of one complex element can fall in DIFFERENT chunks, so each is tested on
# its own rather than seeding both when the element is in range.
@inline function _seed_leaf!(
    nda::Nfwd.NDualArray{Complex{R}}, dims, s, W
) where {R<:IEEEFloat}
    re, im = Complex(one(R), zero(R)), Complex(zero(R), one(R))
    @inbounds for i in eachindex(nda.primal)
        dr, di = dims[2i - 1], dims[2i]
        s <= dr <= s + W - 1 && Nfwd._set_partial!(nda, i, dr - s + 1, re)
        s <= di <= s + W - 1 && Nfwd._set_partial!(nda, i, di - s + 1, im)
    end
    return nothing
end

@inline _scatter_chunk!(::Tuple{}, out, s, W) = nothing
@inline function _scatter_chunk!(ls::Tuple, out, s, W)
    _, g, dims = first(ls)
    _scatter_leaf!(g, dims, out, s, W)
    return _scatter_chunk!(Base.tail(ls), out, s, W)
end

# `dims` covers 1..total_dim across all rows, so a `min(·, total_dim)` upper clamp would be a
# no-op; the `d <= s + W - 1` bound alone excludes a short final chunk's empty lanes.
@inline function _scatter_leaf!(g::AbstractArray{T}, dims, out, s, W) where {T<:IEEEFloat}
    @inbounds for (i, d) in enumerate(dims)
        s <= d <= s + W - 1 && (g[i] = tangent(out, d - s + 1))
    end
    return nothing
end
# The real direction lands in `real(g[i])` and the imaginary one in `imag(g[i])`. Each half is
# written exactly once per call but possibly in different chunks, so each write preserves the
# other half rather than assuming both are in hand.
@inline function _scatter_leaf!(
    g::AbstractArray{Complex{R}}, dims, out, s, W
) where {R<:IEEEFloat}
    @inbounds for i in eachindex(g)
        dr, di = dims[2i - 1], dims[2i]
        s <= dr <= s + W - 1 && (g[i] = Complex(tangent(out, dr - s + 1), imag(g[i])))
        s <= di <= s + W - 1 && (g[i] = Complex(real(g[i]), tangent(out, di - s + 1)))
    end
    return nothing
end

# Structured sweep: restore seed primals and partials each chunk to prevent compounded
# mutation. Each gradient dimension is written once, so grad_bufs needs no zeroing.
function _structured_gradient!!(
    cache::FCache, f::F, xs::Tuple, seed::StructuredGradSeed
) where {F}
    input_primals = (f, xs...)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), input_primals)
    _check_tangent_aliasing((NoTangent(), seed.grad_bufs...), input_primals)
    f_stored = seed.f_seed
    # Rewrap the call-time `f` (the stored seed holds the prepare-time instance); `V ===
    # NoDual` is guaranteed by the non-differentiable-`f` gate, so this is a free isbits
    # rewrap.
    f_seed = typeof(f_stored)(f, tangent(f_stored))
    arg_seeds = seed.arg_seeds
    grad_bufs = seed.grad_bufs
    leaves = seed.leaves
    W = cache.gradient_chunk_size.width
    total_dim = isempty(leaves) ? 0 : last(leaves)[3].stop
    local y
    s = 1
    while s <= total_dim
        # Undo any field rebinding by `f`, which would otherwise orphan `leaves`.
        _restore_seed_bindings!(seed.bindings)
        foreach(row -> _reset_seed_extent!(row[1], length(row[2])), leaves)
        # Refresh call-time nondifferentiable state every chunk too, so mutation cannot
        # compound. Unchanged state rebuilds nothing.
        arg_seeds = _refresh_all!(arg_seeds, xs)
        _zero_seeds!(leaves)
        _seed_chunk!(leaves, s, W)
        out = value_and_derivative!!(cache, f_seed, arg_seeds...)
        y = primal(out)
        _check_scalar_output(y; caller=(value_and_gradient!!), cache=cache)
        _scatter_chunk!(leaves, out, s, W)
        s += W
    end
    native_gradients = (NoTangent(), grad_bufs...)
    return _finalize_gradient(cache, y, native_gradients, input_primals)
end

# A concrete barrier keeps isbits seed reconstruction on the stack without an IdDict.
@inline function _isbits_chunk(cache, input_primals, templates, ::Val{W}, s) where {W}
    seed_w = zero_lifted(Val(W), input_primals)
    vs = tangent(basis_lifted!!(seed_w, ntuple(k -> s + k - 1, Val(W))))
    lifteds = map((t, p, v) -> typeof(t)(p, v), templates, input_primals, vs)
    return value_and_derivative!!(cache, lifteds...)
end

# Write the chunk's `W` directional derivatives into the gradient. `out`'s lane `k` is the
# derivative w.r.t. dimension `s + k - 1`; the gradient mirrors the input structure (recursive
# coherence), so a single dimension-ordered walk sets each leaf scalar directly. Pure isbits
# rebuild with a threaded `Int` cursor — no `basis_lifted!!`/`increment!!`, allocation-free.
@inline function _isbits_scatter(ng, out, ::Val{W}, s) where {W}
    coeffs = ntuple(lane -> tangent(out, lane), Val(W))
    g, _ = _scatter_isbits_leaves(ng, coeffs, s, 0)
    return g
end
@inline _scatter_isbits_leaves(g::NoTangent, _coeffs, _s, c::Int) = (g, c)
@inline function _scatter_isbits_leaves(
    g::T, coeffs::NTuple{W}, s, c::Int
) where {T<:IEEEFloat,W}
    c += 1
    # A scalar leaf consumes one dimension; write the active lane's coefficient at this cursor
    # position when it falls in the current chunk `[s, s+W-1]` (matches the array path
    # `_scatter_chunk!`).
    return (s <= c <= s + W - 1 ? T(coeffs[c - s + 1]) : g, c)
end
@inline _scatter_isbits_leaves(::Tuple{}, _coeffs, _s, c::Int) = ((), c)
@inline function _scatter_isbits_leaves(g::Tuple, coeffs, s, c::Int)
    h, c = _scatter_isbits_leaves(first(g), coeffs, s, c)
    t, c = _scatter_isbits_leaves(Base.tail(g), coeffs, s, c)
    return ((h, t...), c)
end
@inline function _scatter_isbits_leaves(
    g::NamedTuple{names}, coeffs, s, c::Int
) where {names}
    t, c = _scatter_isbits_leaves(values(g), coeffs, s, c)
    return (NamedTuple{names}(t), c)
end
@inline function _scatter_isbits_leaves(g::Tangent, coeffs, s, c::Int)
    inner, c = _scatter_isbits_leaves(g.fields, coeffs, s, c)
    return (typeof(g)(inner), c)
end

# Zero-allocation gradient for scalar-only structured inputs (isbits V); see
# `IsbitsGradSeed`.
function _isbits_gradient!!(
    cache::FCache, f::F, xs::Tuple, gs::IsbitsGradSeed{W}
) where {F,W}
    input_primals = (f, xs...)
    _check_prepared_cache(getfield(cache, :input_types_and_sizes), input_primals)
    total_dim = gs.total_dim
    templates = gs.templates
    native_gradients = _zero_tangents(input_primals)
    # Peel the first (always full-width) chunk to keep the scalar `y` concretely typed.
    first_out = _isbits_chunk(cache, input_primals, templates, Val(W), 1)
    y = primal(first_out)
    _check_scalar_output(y; caller=(value_and_gradient!!), cache=cache)
    native_gradients = _isbits_scatter(native_gradients, first_out, Val(W), 1)
    s = 1 + W
    while s <= total_dim
        out = _isbits_chunk(cache, input_primals, templates, Val(W), s)
        native_gradients = _isbits_scatter(native_gradients, out, Val(W), s)
        s += W
    end
    return _finalize_gradient(cache, y, native_gradients, input_primals)
end

#
# Forward-over-reverse — Hessian-vector products (HVP)
#

@inline function _check_tangent_for_primal(primal, tangent)
    typeof(tangent) <: tangent_type(typeof(primal)) || throw(
        ArgumentError(
            "Tangent types do not match primal types: tangent $(typeof(tangent)) is not a " *
            "$(tangent_type(typeof(primal))) for primal $(typeof(primal)). " *
            "Supply an internal tangent (e.g. `Mooncake.zero_tangent(x)`).",
        ),
    )
    # Base's generic `axes(x) = map(oneto, size(x))` makes `applicable(axes, x)` true even
    # for a struct-shaped `Tangent` with no `size` method, so check `size` instead.
    if applicable(size, primal) && applicable(size, tangent)
        axes(primal) == axes(tangent) || throw(
            ArgumentError(
                "Tangent direction must match the primal axes; got axes " *
                "$(axes(tangent)) for tangent vs $(axes(primal)) for primal",
            ),
        )
    elseif applicable(length, primal) && applicable(length, tangent)
        length(primal) == length(tangent) || throw(
            ArgumentError(
                "Tangent direction must match the primal length; got " *
                "length $(length(tangent)) for tangent vs $(length(primal)) for primal",
            ),
        )
    end
    return nothing
end

"""
    prepare_hvp_cache(f, x; config=Mooncake.Config())

Prepare a cache for computing Hessian-vector products (HVPs) of `f`. Returns an `HVPCache`
for use with [`value_and_hvp!!`](@ref).

`f` must map a single `AbstractVector` `x` to a scalar. Like [`value_and_jacobian!!`](@ref),
only a single input is supported; concatenate the inputs of a multi-argument function into
one vector.

The cache compiles an outer forward-mode rule over an inner reverse-mode gradient. The inner
rule is compiled only once regardless of how many HVPs are subsequently evaluated.

*Note:* `cache` is tied to the type and shape of `x`. Evaluating at a different point is
fine, but changing the shape requires a new cache.

!!! note
    Calls `f(x...)` during cache preparation (via inner gradient and derivative caches).

```jldoctest; setup = :(using Mooncake)
f(x) = sum(x .* x)
x = [1.0, 2.0]
cache = Mooncake.prepare_hvp_cache(f, x)
f_val, gradient, hvp = Mooncake.value_and_hvp!!(cache, f, [1.0, 0.0], x)
f_val ≈ 5.0 && gradient ≈ [2.0, 4.0] && hvp ≈ [2.0, 0.0]

# output

true
```
"""
@unstable @inline function prepare_hvp_cache(
    f::F, x::Vararg{Any,N}; config=Config(), _chunk::Int=1
) where {F,N}
    N == 0 && throw(ArgumentError("prepare_hvp_cache requires at least one x argument"))
    N > 1 && _throw_hvp_multiarg("prepare_hvp_cache", N)
    # Validates that `f` returns an `IEEEFloat` (running `f` once), allocates the tangent
    # buffers reused by `grad_f`, and supplies `output_type_and_size`. The primitive
    # (`DerivedFoRRule{Nothing}`) branches below also evaluate gradients through
    # `grad_cache.rule`.
    grad_cache = prepare_gradient_cache(f, x...; config)

    # Reuse DerivedFoRRule's precompiled Lifted dual callables; re-deriving would leak
    # reverse-optimised IR into forward AD. Primitive rules have no MistyClosure IR and use
    # grad_cache directly. Keep dual and outer forward-rule widths equal; standalone HVP
    # stays width 1, while Hessians request a width capped at the input dimension.
    fwd_chunk_size = _chunk == 1 ? 1 : min(_chunk, tangent_dim(_zero_tangents(x)))
    # Use the FoR width, not config.chunk_size, to avoid incompatible chunk rules.
    # Do not clear caches again: prepare_gradient_cache already honoured empty_cache.
    # Keep the inner cache non-friendly: grad_tangent and v are internal tangents, not
    # primal-shaped directions. HVP/Hessian take no user output tangent, so friendly_tangents
    # does not affect their result shape.
    fwd_config = Config(;
        config.debug_mode,
        config.silence_debug_messages,
        chunk_size=fwd_chunk_size,
        empty_cache=false,
    )
    for_rule = compile_for_rule(
        f, x...; debug_mode=config.debug_mode, chunk_size=fwd_chunk_size
    )

    # The derived branches capture only these buffers: capturing `grad_cache` itself
    # would make `zero_tangent(grad_f)` traverse `grad_cache.rule`'s MistyClosures and
    # eagerly build dual callables over reverse-mode-optimised IR that the derived path
    # never invokes.
    tangents = grad_cache.tangents
    grad_f = if for_rule isa DerivedFoRRule{Nothing}
        y -> begin
            val_and_grad = value_and_gradient!!(grad_cache, f, y)
            (val_and_grad[1], val_and_grad[2][2])
        end
    else
        y -> begin
            inner_rule = get_inner_rrule(for_rule)
            t_f, t_y = tangents
            t_f = set_to_zero!!(t_f)
            t_y = set_to_zero!!(t_y)
            val, grad = __value_and_gradient!!(inner_rule, CoDual(f, t_f), CoDual(y, t_y))
            (val, grad[2])
        end
    end
    fwd_cache = prepare_derivative_cache(grad_f, x...; config=fwd_config)
    return HVPCache(
        f,
        grad_f,
        zero_tangent(grad_f),
        fwd_cache,
        getfield(grad_cache, :output_type_and_size),
        nothing,
    )
end

# HVP and Hessian, like `value_and_jacobian!!`, take a single vector input; multi-argument
# functions must concatenate their inputs into one vector.
@noinline _throw_hvp_multiarg(fname::String, got::Int) = throw(
    ArgumentError(
        "$fname supports only a single AbstractVector input; got $got arguments. " *
        "Concatenate the inputs into one vector.",
    ),
)

"""
    value_and_hvp!!(cache::HVPCache, f, v, x)

Given a cache prepared by [`prepare_hvp_cache`](@ref), compute the gradient of `f` at `x`
and the Hessian-vector product `H v`. `v` must have the internal tangent type for `x` and
matching axes (or length for sized collections without axes); returns `(f(x), ∇f(x),
H(x)v)`. For `f: Rⁿ → R` with `x::Vector{Float64}`, the gradient and HVP are
`Vector{Float64}`. Like [`value_and_jacobian!!`](@ref), only a single `AbstractVector` input
is supported; concatenate the inputs of a multi-argument function into one vector.

As with all functionality in Mooncake, `x` is returned to its original state: if `f` mutates
`x` in place, it is restored, so the input is not mutated.

!!! warning
    `cache` must be the output of [`prepare_hvp_cache`](@ref), and `f` must be the same
    function object used to construct `cache`. All `x` arguments must have the same sizes
    and element types as used to construct the cache.

!!! warning
    `cache` owns the mutable state in the returned values. Take a copy before calling again
    if you need to retain previous results.

!!! warning
    `HVPCache` is not safe for concurrent reuse across threads. Use a separate cache per
    task/thread if calls may overlap in time.

```jldoctest; setup = :(using Mooncake)
f(x) = sum(x .* x)
x = [1.0, 2.0]
cache = Mooncake.prepare_hvp_cache(f, x)
f_val, gradient, hvp = Mooncake.value_and_hvp!!(cache, f, [1.0, 0.0], x)
f_val ≈ 5.0 && gradient ≈ [2.0, 4.0] && hvp ≈ [2.0, 0.0]

# output

true
```
"""
@inline function value_and_hvp!!(cache::HVPCache, f::F, v, x1::T1) where {F,T1}
    cache.f === f || throw(
        ArgumentError("`f` must be the same function object used to construct `cache`")
    )
    _check_prepared_cache(
        getfield(cache.fwd_cache, :input_types_and_sizes), (cache.grad_f, x1)
    )
    _check_tangent_for_primal(x1, v)
    (f_val, grad), (_, hvp) = value_and_derivative!!(
        cache.fwd_cache, (cache.grad_f, cache.grad_tangent), (x1, v)
    )
    return f_val, grad, hvp
end

# Multi-argument calls match no single-arg method above; give a clear error rather than a
# raw MethodError (mirrors `value_and_jacobian!!(cache, f, x, xs...)`).
@inline function value_and_hvp!!(cache::HVPCache, f, v, x1, x2, xs::Vararg{Any,N}) where {N}
    return _throw_hvp_multiarg("value_and_hvp!!", N + 2)
end

#
# Forward-over-reverse — Hessian
#

function _make_hessian_buffers(x::AbstractVector)
    # Match x's array type so GPU inputs get device-resident H/grad/v buffers;
    # both sweeps write via broadcast and copyto! without scalar device indexing.
    T = eltype(x)
    n = length(x)
    return (;
        H=fill!(similar(x, T, n, n), zero(T)), grad=zero_tangent(x), v=zero_tangent(x)
    )
end

@noinline _throw_not_hessian_cache() = throw(
    ArgumentError(
        "`cache` was not built with `prepare_hessian_cache`; rebuild via " *
        "`prepare_hessian_cache(f, x)` to use `value_gradient_and_hessian!!`",
    ),
)

"""
    prepare_hessian_cache(f, x; config=Mooncake.Config())

Return a cache for computing `f(x)`, the gradient `∇f`, and the Hessian of `f` via
[`value_gradient_and_hessian!!`](@ref). Returns an [`HVPCache`](@ref), which is also
accepted by [`value_and_hvp!!`](@ref).

Like [`value_and_jacobian!!`](@ref), only a single `AbstractVector` input of an IEEE-float
element type is supported; concatenate the inputs of a multi-argument function into one
vector. Validation is eager and raises `ArgumentError` here rather than at evaluation time.
The cache pre-allocates the Hessian, gradient, and basis-direction buffers that
[`value_gradient_and_hessian!!`](@ref) writes into, so subsequent calls do not allocate
fresh outputs. The returned `gradient` and Hessian alias cache storage; copy them if you
need to retain previous results. Buffers are allocated to match the input's array type:
GPU-array inputs (e.g. `CuArray`) produce a device-resident gradient and Hessian (use
`Array(H)` to move the result to the host).

Hessian computation uses forward-over-reverse AD over the reverse-mode gradient function.
The Hessian is chunked: it sweeps `W = config.chunk_size` basis directions per forward pass
(≈`ceil(n/W)` passes for `n` input dimensions). `chunk_size` defaults to automatic (up to an
internal maximum, capped at `n`); pass `config=Config(; chunk_size=W)` to set it.

!!! note
    This path uses Mooncake's generic public forward cache over the captured reverse-mode
    gradient closure. A chunked Hessian prepares two forward variants — width-1 (shared with
    [`value_and_hvp!!`](@ref)) and width-`W` for the column sweep — so `f` is evaluated and
    the forward-over-reverse rule compiled twice during preparation; the width-1 (scalar
    1-dimension or `chunk_size=1`) case prepares a single variant.

```jldoctest; setup = :(using Mooncake)
f(x) = sum(x .^ 2)
x = [1.0, 2.0, 3.0]
cache = Mooncake.prepare_hessian_cache(f, x)
Mooncake.value_gradient_and_hessian!!(cache, f, x)

# output

(14.0, [2.0, 4.0, 6.0], [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0])
```
"""
@unstable @inline function prepare_hessian_cache(
    f::F, x::Vararg{Any,N}; config=Config()
) where {F,N}
    N == 0 && throw(ArgumentError("prepare_hessian_cache requires at least one x argument"))
    N > 1 && _throw_hvp_multiarg("prepare_hessian_cache", N)
    x1 = only(x)
    _check_vector_argument(x1; caller=prepare_hessian_cache)
    base = prepare_hvp_cache(f, x1; config)
    # Chunked forward-over-reverse Hessian sweep: build a width-W variant of `grad_f` whose
    # FoR rule's dual callables are width W, alongside the width-1 `base` used by
    # `value_and_hvp!!`. `chunked === nothing` keeps the width-1 column loop (scalar 1-dimension,
    # or chunk_size 1). Auto (`chunk_size` nothing/0) chunks the full matrix.
    dim_x = tangent_dim(zero_tangent(x1))
    W = let c = getfield(config, :chunk_size)
        req = if (c === nothing || c == 0)
            _MAX_CHUNK_WIDTH
        else
            Nfwd._nfwd_check_chunk_size(c)
        end
        min(req, dim_x)
    end
    # `empty_cache=false`: the `base` prepare above already honoured `config.empty_cache`;
    # re-clearing for the width-W variant would invalidate what `base` just compiled.
    chunked = if W > 1
        ch_config = Config(;
            config.debug_mode,
            config.silence_debug_messages,
            config.friendly_tangents,
            config.chunk_size,
            empty_cache=false,
        )
        ch = prepare_hvp_cache(f, x1; config=ch_config, _chunk=W)
        (ch.grad_f, ch.fwd_cache, W)
    else
        nothing
    end
    return HVPCache(
        base.f,
        base.grad_f,
        base.grad_tangent,
        base.fwd_cache,
        base.output_type_and_size,
        (_make_hessian_buffers(x1), chunked),
    )
end

# Hessian columns are lanes of grad_f's gradient output (tuple index 2).
# Val(W) keeps partials unboxed across the unstable caller's barrier. Snapshot/restore
# x1 because seed primals alias it and mutations must not compound across chunks.
function _chunked_hessian_sweep!(grad_f, fwd, H, g, x1, n::Int, ::Val{W}) where {W}
    f_seed = zero_lifted(Val(W), grad_f)
    x_seed = zero_lifted(Val(W), x1)
    cols(start_col) = ntuple(lane -> let slot = start_col + lane - 1
        slot <= n ? slot : 0
    end, Val(W))
    x_snapshot = copy(x1)
    try
        output = value_and_derivative!!(fwd, f_seed, basis_lifted!!(x_seed, cols(1)))
        po = primal(output)
        value = po[1]
        g .= po[2]
        @inbounds for lane in 1:W
            lane <= n || break
            H[:, lane] .= tangent(output, lane)[2]
        end
        for start_col in (W + 1):W:n
            copyto!(x1, x_snapshot)
            output = value_and_derivative!!(
                fwd, f_seed, basis_lifted!!(x_seed, cols(start_col))
            )
            @inbounds for lane in 1:W
                col = start_col + lane - 1
                col <= n || break
                H[:, col] .= tangent(output, lane)[2]
            end
        end
        return value
    finally
        copyto!(x1, x_snapshot)
    end
end

# Checked at the entry point, not in the sweep: the width-1 sweep reaches
# `_check_shared_input_tangents` through `value_and_hvp!!`, but `_chunked_hessian_sweep!` calls the
# pre-lifted `value_and_derivative!!` method, which that guard never sees.
@inline function _check_hessian_input_aliasing(cache::HVPCache)
    getfield(getfield(cache, :fwd_cache), :inputs_share_storage) &&
        _throw_hessian_input_alias_error()
    return nothing
end

function _throw_hessian_input_alias_error()
    throw(
        ArgumentError(
            "`value_gradient_and_hessian!!` does not support inputs that share differentiable " *
            "storage across positions — `f` holding the same array that is also passed as an " *
            "argument, say. Each Hessian column is one standard-basis direction, which reaches " *
            "the shared leaf at a single position, so the columns come back missing the " *
            "contributions from the others. Concatenate the shared storage into the single " *
            "input vector the Hessian is taken with respect to.",
        ),
    )
end

"""
    value_gradient_and_hessian!!(cache::HVPCache, f, x)

Using a pre-built `cache` from [`prepare_hessian_cache`](@ref), compute and return `(f(x),
∇f(x), ∇²f(x))` — value, gradient vector, and Hessian matrix of `f`.

The input `x` is restored to its original state even if a sweep throws.

Uses forward-over-reverse AD; the Hessian is chunked, sweeping `config.chunk_size` basis
directions per forward pass. Like [`value_and_jacobian!!`](@ref), only a single
`AbstractVector` input is supported; concatenate the inputs of a multi-argument function
into one vector.

!!! info
    `cache` must be the output of [`prepare_hessian_cache`](@ref), and `f` must be the same
    function object used to construct `cache`. `x` must have the same size and element type
    as used to construct the cache. The implementation supports only `AbstractVector`s of
    IEEE floats. For non-vector inputs, use [`value_and_hvp!!`](@ref) to obtain second-order
    directional derivatives without forming a full Hessian.

!!! warning
    The returned `gradient` and Hessian alias buffers owned by `cache` and are overwritten
    on the next call with the same cache. Copy them (`copy`/`deepcopy`) before mutating or
    if you need to retain previous results.

!!! warning
    `HVPCache` is not safe for concurrent reuse across threads. Use a separate cache per
    task/thread if calls may overlap in time.

# Example
```jldoctest; setup = :(using Mooncake)
f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x = [1.2, 1.2]
cache = Mooncake.prepare_hessian_cache(f, x)
_, _, H = Mooncake.value_gradient_and_hessian!!(cache, f, x)
H

# output

2×2 Matrix{Float64}:
 1250.0  -480.0
 -480.0   200.0
```
"""
@unstable @inline function value_gradient_and_hessian!!(
    cache::HVPCache, f::F, x1::T1
) where {F,T1}
    cache.f === f || throw(
        ArgumentError("`f` must be the same function object used to construct `cache`")
    )
    hb = cache.hess_buffers
    hb === nothing && _throw_not_hessian_cache()
    _check_hessian_input_aliasing(cache)
    buf, chunked = hb
    T = _check_vector_argument(x1; caller=(value_gradient_and_hessian!!), cache=cache)
    H = buf.H
    g = buf.grad
    v = buf.v
    n = length(x1)
    # Buffer sizes are fixed at cache build time; reject mismatched inputs before
    # indexing `v`/`H`, otherwise the sweep below raises a raw `BoundsError`.
    n == length(v) || throw(
        ArgumentError(
            "input vector has length $n but cache was prepared for length $(length(v)); " *
            "rebuild via `prepare_hessian_cache`",
        ),
    )
    if n == 0
        fval, _, _ = value_and_hvp!!(cache, f, v, x1)
        return fval, g, H
    end
    # Chunked forward-over-reverse: when `prepare_hessian_cache` built a width-W variant of
    # `grad_f` (its FoR dual callables are width W), sweep W Hessian columns per pass. Falls
    # back to the width-1 column loop otherwise (scalar 1-dimension, or chunk_size 1).
    if chunked !== nothing
        grad_f_c, fwd_c, W = chunked
        value = _chunked_hessian_sweep!(grad_f_c, fwd_c, H, g, x1, n, Val(W))
        return value, g, H
    end
    local value
    # Reset `v` in case a prior call threw between `v[i] = one(T)` and `v[i] = zero(T)`.
    # Only the width-1 sweep touches `v`; the chunked path builds its own seed, so this
    # stays out of it.
    fill!(v, zero(T))
    # One-hot writes via a reused host buffer + `copyto!`: `v[i] = x` is GPU scalar
    # indexing (errors on CuArray), while `copyto!` serves both Vector and CuArray.
    e = Vector{T}(undef, 1)
    for i in 1:n
        e[1] = one(T)
        copyto!(v, i, e, 1, 1)
        fval, grad_alias, hvp = value_and_hvp!!(cache, f, v, x1)
        if i == 1
            value = fval
            g .= grad_alias
        end
        @inbounds @views H[:, i] .= hvp
        e[1] = zero(T)
        copyto!(v, i, e, 1, 1)
    end
    return value, g, H
end

# Multi-argument calls match no single-arg method above; give a clear error rather than a
# raw MethodError (mirrors `value_and_jacobian!!(cache, f, x, xs...)`).
@unstable @inline function value_gradient_and_hessian!!(
    cache::HVPCache, f, x1, x2, xs::Vararg{Any,N}
) where {N}
    return _throw_hvp_multiarg("value_gradient_and_hessian!!", N + 2)
end

#
# Shared cross-mode utilities
#

"""
    __exclude_unsupported_output(y)
    __exclude_func_with_unsupported_output(fx)

Required for the robust design of [`value_and_pullback!!`](@ref),
[`prepare_pullback_cache`](@ref). Ensures that `y` or returned value of `fx::Tuple{Tf,
Targs...}` contains no aliasing, circular references, `Ptr`s or non differentiable
datatypes. In the forward pass f(args...) output can only return a "Tree" like datastructure
with leaf nodes as primitive types. Refer
https://github.com/chalk-lab/Mooncake.jl/issues/517#issuecomment-2715202789 and related
issue for details. Internally calls [`__exclude_unsupported_output_internal!`](@ref). The
design is modelled after `zero_tangent`.
"""
function __exclude_unsupported_output(y::T) where {T}
    __exclude_unsupported_output_internal!(y, Set{UInt}())
    return nothing
end

function __exclude_func_with_unsupported_output(fx)
    _fx = deepcopy(fx)
    _func, _args = _fx[1], _fx[2:end]
    _y = _func(_args...)
    return __exclude_unsupported_output(_y)
end

# Isbits recursion terminates, but Ptr and isbits structs containing Ptr must still
# reach the unsupported-output guard rather than the pointer-free fast path.
_isbits_contains_ptr(::Type{<:Ptr}) = true
_isbits_contains_ptr(::Type{T}) where {T} = any(_isbits_contains_ptr, fieldtypes(T))

"""
    __exclude_unsupported_output_internal!(y::T, address_set::Set{UInt}) where {T}

For checking if output`y` is a valid Mutable/immutable composite or a primitive type.
Performs a recursive depth first search over the function output `y` with an `isbitstype()`
check base case. The visited memory addresses are stored inside `address_set`. If the set
already contains a newly visited address, it errors out indicating an Alias or Circular
reference. Also errors out if `y` is or contains a Pointer. It is called internally by
[`__exclude_unsupported_output(y)`](@ref).
"""
function __exclude_unsupported_output_internal!(y::T, address_set::Set{UInt}) where {T}
    isbitstype(T) && !_isbits_contains_ptr(T) && return nothing
    if objectid(y) in address_set
        throw_circular_reference_or_alias_error(y)
    end

    # immutable types are copied on the stack.
    ismutable(y) && push!(address_set, objectid(y))

    # recurse over a composite type's fields.
    for y_sub in fieldnames(T)
        # isdefined() is valid for Mutable Structs, Structs.
        !isdefined(y, y_sub) && continue
        __exclude_unsupported_output_internal!(getfield(y, y_sub), address_set)
    end

    return nothing
end

# Retain snapshot-node => original-call-object correspondence during population, before the
# rule can rebind any fields. Each argument validates its own prepared partition; sharing across
# arguments may change for a directional derivative. Restore visits are independent of this map.
struct _RestoreContext
    originals::IdDict{Any,Any}
    seen::IdDict{Any,Any}
    vector_refs::IdDict{Any,Any}
end

function _snapshot_inputs!!(dst::Tuple, src::Tuple)
    pairs = map(dst, src) do d, s
        c = IdDict{Any,Any}()
        snapshot = _copy_to_output!!(d, s, c)
        refs = IdDict{Any,Any}()
        @static if VERSION >= v"1.11.0-rc4"
            for obj in keys(c)
                obj isa Vector && (refs[obj] = obj.ref)
            end
        end
        (snapshot, _RestoreContext(c, IdDict{Any,Any}(), refs))
    end
    return map(first, pairs), map(last, pairs)
end

function _restore_inputs!!(dst::Tuple, src::Tuple, contexts::Tuple)
    return map(dst, src, contexts) do d, s, c
        empty!(c.seen)
        _copy_to_output!!(d, s, c.seen, c)
    end
end

# Source and destination graphs must share nodes at the same positions, or restoration
# would re-point the caller's graph. Register both directions while populating the snapshot
# to reject mismatches before running the rule. Matching repeats and cycles cost one `===`.
@inline function _same_destination(previous::P, dst::P) where {P}
    previous === dst && return previous
    return _throw_copy_sharing_error(P)
end

@noinline function _throw_copy_sharing_error(@nospecialize(P::Type))
    throw(
        PreparedCacheError(
            "Prepared cache mismatch: a $P shared between two positions of the prepared storage " *
            "is not shared between the corresponding positions of the value being copied (or the " *
            "other way round). Aliasing inside an argument is part of the shape a cache is " *
            "prepared for. Prepare a separate cache for this argument aliasing.",
        ),
    )
end

# Register mutable nodes and opaque custom leaves before descent, returning repeats immediately.
@inline function _copy_context_destination!!(dst, src::P, c::IdDict, r) where {P}
    if r isa _RestoreContext
        dst = get(r.originals, src, dst)
        haskey(c, src) && return c[src]::P, true
    else
        haskey(c, src) && return _same_destination(c[src]::P, dst), true
        haskey(c, dst) && _throw_copy_sharing_error(P)
        c[dst] = src
    end
    c[src] = dst
    return dst::P, false
end

# Check extents before inbounds copies: isassigned(dst, i) is false out of range, so it
# cannot prevent an overrun or a truncated result when the cached shape is stale.
@inline function _check_copy_extent(dst, src)
    size(dst) == size(src) && return nothing
    throw(
        PreparedCacheError(
            "Prepared cache mismatch: cached storage has size $(size(dst)), but the value to " *
            "copy into it has size $(size(src)). Rebuild the cache for the new shape.",
        ),
    )
end

"""
    _copy_to_output!!(dst::T, src::T)

Copy the contents of `src` to `dst`, with zero or minimal new memory allocation. The type of
`dst` and `src` must be the same. Required as Base.copy!() does not work for all supported
primal types. For example, `Base.copy!` does not work for `Core.svec`. A defined source field
initialises an undefined destination field. Retain the returned value: an immutable `dst` is
rebuilt rather than updated. For types with custom copy semantics, overload this function
(see `Core.SimpleVector` for an example).
"""
_copy_to_output!!(dst::Number, src::Number, c=nothing, r=nothing) = src
_copy_to_output!!(::Type, src::Type, c=nothing, r=nothing) = src
_copy_to_output!!(::Core.TypeName, src::Core.TypeName, c=nothing, r=nothing) = src
_copy_to_output!!(::Module, src::Module, c=nothing, r=nothing) = src
@static if VERSION >= v"1.11.0-rc4"
    function _copy_to_output!!(dst::P, src::P, c=nothing, r=nothing) where {P<:MemoryRef}
        mem = _copy_to_output!!(dst.mem, src.mem, c, r)
        ref = Core.memoryrefnew(mem)
        offset = Core.memoryrefoffset(src)
        return offset == 1 ? ref : Core.memoryrefnew(ref, offset, false)
    end
end

function _copy_to_output!!(dst::SimpleVector, src::SimpleVector, c=nothing, r=nothing)
    return Core.svec(map((d, s) -> _copy_to_output!!(d, s, c, r), dst, src)...)
end
function _copy_to_output!!(dst::P, src::P, c=nothing, r=nothing) where {P<:_BuiltinArrays}
    c === nothing && !isbitstype(eltype(P)) && (c = IdDict{Any,Any}())
    if c !== nothing
        dst, seen = _copy_context_destination!!(dst, src, c, r)
        seen && return dst
    end
    if r isa _RestoreContext && dst isa Vector
        @static if VERSION >= v"1.11.0-rc4"
            # Growth may replace the backing Memory. Restore the original owner as well
            # as the length, preserving refs/views and the next call's seed coverage.
            if haskey(r.vector_refs, dst)
                setfield!(dst, :ref, r.vector_refs[dst])
                setfield!(dst, :size, size(src))
            else
                resize!(dst, length(src))
            end
        else
            resize!(dst, length(src))
        end
    end
    _check_copy_extent(dst, src)
    @inbounds for i in eachindex(src)
        if isassigned(src, i)
            dst[i] = if isassigned(dst, i)
                _copy_to_output!!(dst[i], src[i], c, r)
            else
                _copy_missing(src[i], c, r)
            end
        elseif isassigned(dst, i)
            Base._unsetindex!(dst, i)
        end
    end
    return dst
end
function _copy_to_output!!(dst::P, src::P, c=nothing, r=nothing) where {P<:Tuple}
    isbitstype(P) && return src
    return map((d, s) -> _copy_to_output!!(d, s, c, r), dst, src)
end

function _copy_to_output!!(dst::P, src::P, c=nothing, r=nothing) where {P<:NamedTuple}
    isbitstype(P) && return src
    return P(map((d, s) -> _copy_to_output!!(d, s, c, r), values(dst), values(src)))
end
function _copy_to_output!!(dst::P, src::P, c=nothing, r=nothing) where {P}
    # Only the generic struct fallback needs to detect an opaque two-argument extension:
    # all specialised bodies dispatch above, so no list of built-in types is required.
    custom =
        which(_copy_to_output!!, Tuple{P,P}) !==
        which(_copy_to_output!!, Tuple{Base.RefValue{Nothing},Base.RefValue{Nothing}})
    if custom
        c === nothing && return _copy_to_output!!(dst, src)
        dst, seen = _copy_context_destination!!(dst, src, c, r)
        return seen ? dst : _copy_to_output!!(dst, src)
    end
    isbitstype(P) && return src
    nf = nfields(src)
    nf == 0 && return src
    if ismutable(src)
        c === nothing && (c = IdDict{Any,Any}())
        dst, seen = _copy_context_destination!!(dst, src, c, r)
        seen && return dst
        for src_sub in 1:nf
            if isdefined(src, src_sub)
                # using ccall as setfield! fails for const fields of a mutable struct.
                ccall(
                    :jl_set_nth_field,
                    Cvoid,
                    (Any, Csize_t, Any),
                    dst,
                    src_sub - 1,
                    if isdefined(dst, src_sub)
                        _copy_to_output!!(
                            getfield(dst, src_sub), getfield(src, src_sub), c, r
                        )
                    else
                        _copy_missing(getfield(src, src_sub), c, r)
                    end,
                )
            elseif isdefined(dst, src_sub)
                _unset_copy_field!(dst, src_sub)
            end
        end
        return dst
    else
        flds = Vector{Any}(undef, nf)
        for src_sub in 1:nf
            if isdefined(src, src_sub)
                flds[src_sub] = if isdefined(dst, src_sub)
                    _copy_to_output!!(getfield(dst, src_sub), getfield(src, src_sub), c, r)
                else
                    _copy_missing(getfield(src, src_sub), c, r)
                end
            else
                nf = src_sub - 1
                break
            end
        end
        !isassigned(flds, 1) && return src
        return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), P, flds, nf)::P
    end
end

# A missing destination during restore must recover original mutable descendants, not
# install snapshot-owned storage. Walking an immutable snapshot against itself rebuilds
# its fields through the same restore context, including cycles through mutable nodes.
function _copy_missing(src, c, r)
    if r isa _RestoreContext
        _copy_to_output!!(get(r.originals, src, src), src, c, r)
    else
        _copy_output(src, c)
    end
end

# Julia has no unsetfield! counterpart to Base._unsetindex!. Clear the field's storage,
# including inline reference-bearing immutable fields. jl_set_nth_field with NULL is a
# no-op. This runs only for a field that is undefined in the saved object; bits fields
# cannot have that state. Clearing references needs no GC write barrier.
function _unset_copy_field!(dst::P, i::Int) where {P}
    offset = fieldoffset(P, i)
    stop = i == fieldcount(P) ? sizeof(P) : fieldoffset(P, i + 1)
    GC.@preserve dst ccall(
        :memset,
        Ptr{Cvoid},
        (Ptr{Cvoid}, Cint, Csize_t),
        pointer_from_objref(dst) + offset,
        0,
        stop - offset,
    )
    return dst
end

# fallback for invalid type combinations
function _copy_to_output!!(dst::T, src::P, c=nothing, r=nothing) where {T,P}
    if r isa _RestoreContext
        return _copy_to_output!!(get(r.originals, src, src), src, c, r)
    end
    throw(
        ArgumentError(
            "Mooncake.jl does not currently have a method `_copy_to_output!!` to handle " *
            "this type combination: dst passed is of type $T, while src is a $P. This " *
            "often happens when differentiating over non-differentiable types (e.g. " *
            "integers or booleans).",
        ),
    )
end

"""
    _copy_output(x::T)

Returns a copy of `x`, of the same type `T`. Allocates new memory for the copy. Required as
Base.copy() does not work for all supported primal types. For example, `Base.copy` does not
work for `Core.svec`. For types with custom copy semantics, overload this function (see
`Core.SimpleVector` for an example).
"""
# Every aggregate shares one cache across its children. Mutable nodes are registered before
# descent for cycles and repeated references, including arrays with isbits elements. A lone
# isbits-element array can keep the cache-free allocation path.

# Type values (DataType, UnionAll, Union), Core.TypeName, and Modules
# cannot be deep-copied; return x as-is.
@unstable _copy_output(x::Type, c::C=nothing) where {C<:Union{Nothing,IdDict}} = x
_copy_output(x::Core.TypeName, c::C=nothing) where {C<:Union{Nothing,IdDict}} = x
_copy_output(x::Module, c::C=nothing) where {C<:Union{Nothing,IdDict}} = x

# Compiled callables are nondifferentiable and retain cyclic reflection/IR graphs;
# return them unchanged so copying a captured reverse rule terminates.
_copy_output(x::Core.OpaqueClosure, c::C=nothing) where {C<:Union{Nothing,IdDict}} = x
_copy_output(x::MistyClosure, c::C=nothing) where {C<:Union{Nothing,IdDict}} = x

@static if VERSION >= v"1.11.0-rc4"
    function _copy_output(x::MemoryRef, c::C=nothing) where {C<:Union{Nothing,IdDict}}
        mem = _copy_output(x.mem, c)
        ref = Core.memoryrefnew(mem)
        offset = Core.memoryrefoffset(x)
        return offset == 1 ? ref : Core.memoryrefnew(ref, offset, false)
    end
end

function _copy_output(x::SimpleVector, c::C=nothing) where {C<:Union{Nothing,IdDict}}
    c === nothing && (c = IdDict{Any,Any}())
    # Copy each element via its own `_copy_output` dispatch (arrays, structs, type values,
    # …); the sibling `_copy_to_output!!(::SimpleVector)` copies element-wise the same way.
    return Core.svec([_copy_output(x_sub, c) for x_sub in x]...)
end

@static if VERSION >= v"1.11.0-rc4"
    function _copy_output(x::P, c::C=nothing) where {P<:Array,C<:Union{Nothing,IdDict}}
        c === nothing && isbitstype(eltype(P)) && return copy(x)
        c === nothing && (c = IdDict{Any,Any}())
        haskey(c, x) && return c[x]::P
        # Register the header before its owner: an Any Memory may refer back to this array.
        temp = P(undef, ntuple(_ -> 0, ndims(x)))
        c[x] = temp
        c[temp] = x
        setfield!(temp, :ref, _copy_output(x.ref, c))
        setfield!(temp, :size, size(x))
        return temp
    end
end

# Array and Memory identities matter even when their elements cannot participate in a cycle.
function _copy_output(x::P, c::C=nothing) where {P<:_BuiltinArrays,C<:Union{Nothing,IdDict}}
    Tx = eltype(P)
    if c === nothing && isbitstype(Tx)
        temp = similar(x)
        @inbounds for i in eachindex(temp)
            isassigned(x, i) && (temp[i] = _copy_output(x[i])::Tx)
        end
        return temp::P
    end
    c === nothing && (c = IdDict{Any,Any}())
    haskey(c, x) && return c[x]::P
    @static if VERSION < v"1.11.0-rc4"
        # Julia 1.10's jl_array_t places flags after data and length. Only how == 3
        # has an owner slot; calling jl_array_data_owner for another how is invalid.
        flags = GC.@preserve x unsafe_load(
            Ptr{UInt16}(pointer_from_objref(x) + 2sizeof(Int))
        )
        if flags & 3 == 3
            owner = ccall(:jl_array_data_owner, Any, (Any,), x)
            if owner isa Array{Tx}
                copied = _copy_output(owner, c)
                # Copying the owner may already have reached x through an element cycle.
                haskey(c, x) && return c[x]::P
                # A same-shape reshape returns its input; use a different header first.
                header = ndims(copied) == 1 ? reshape(copied, 1, :) : vec(copied)
                temp = reshape(header, size(x))::P
                c[x] = temp
                c[temp] = x
                return temp
            end
        end
    end
    temp = similar(x)
    c[x] = temp
    c[temp] = x
    @inbounds for i in eachindex(temp)
        if isassigned(x, i)
            temp[i] = _copy_output(x[i], c)::Tx
        end
    end
    return temp::P
end

# Tuple, NamedTuple
function _copy_output(x::Tuple, c::C=nothing) where {C<:Union{Nothing,IdDict}}
    isbitstype(typeof(x)) && return x
    c === nothing && (c = IdDict{Any,Any}())
    return map(s -> _copy_output(s, c), x)::typeof(x)
end

function _copy_output(x::NamedTuple, c::C=nothing) where {C<:Union{Nothing,IdDict}}
    isbitstype(typeof(x)) && return x
    c === nothing && (c = IdDict{Any,Any}())
    return typeof(x)(map(s -> _copy_output(s, c), values(x)))
end

# Generic fallback: bitstypes, zero-field opaque types (e.g. Symbol/String), and mutable or
# immutable non-bits structs.
function _copy_output(x::P, c::C=nothing) where {P,C<:Union{Nothing,IdDict}}
    if c !== nothing && which(_copy_output, Tuple{P}) !== which(_copy_output, Tuple{Any})
        haskey(c, x) && return c[x]::P
        temp = _copy_output(x)
        c[x] = temp
        c[temp] = x
        return temp
    end
    isbitstype(P) && return x
    # nfields(x) not nfields(P): the latter counts fields of the
    # DataType object itself.
    nf = nfields(x)

    # No Julia-visible fields (e.g. Symbol, String): nothing to copy.
    # Overload _copy_output to customise.
    nf == 0 && return x

    if ismutable(x)
        c === nothing && return _copy_output(x, IdDict{Any,Any}())
        haskey(c, x) && return c[x]::P
        _copy_output_mutable_struct(x, c)
    else
        # Immutable fields share one cache; every cycle crosses a memoized mutable node.
        c === nothing && return _copy_output(x, IdDict{Any,Any}())
        _copy_output_immutable_struct(x, c)
    end
end

# Both struct copies unroll over the fields so each `getfield` has a literal index and infers.
@generated function _copy_output_mutable_struct(x::P, c::IdDict) where {P}
    sets = [
        :(
            isdefined(x, $i) && ccall(
                :jl_set_nth_field,
                Cvoid,
                (Any, Csize_t, Any),
                temp,
                $(i - 1),
                _copy_output(getfield(x, $i), c),
            )
        ) for i in 1:fieldcount(P)
    ]
    return quote
        temp = ccall(:jl_new_struct_uninit, Any, (Any,), P)::P
        # Register before copying fields so a self-reference resolves to `temp`.
        c[x] = temp
        c[temp] = x
        $(sets...)
        return temp::P
    end
end

# Copies the leading defined fields and rebuilds through `jl_new_structv`, so an object built by
# a non-initialising inner constructor keeps its trailing undefined fields (`Base.deepcopy` misses
# this). Assumes that once a field is undefined, all later ones are.
@generated function _copy_output_immutable_struct(x::P, c::IdDict) where {P}
    nf = fieldcount(P)
    copies = [:(_copy_output(getfield(x, $i), c)) for i in 1:nf]
    upto(k) =
        if k == 0
            :x
        else
            :(ccall(
                :jl_new_structv, Any, (Any, Ptr{Any}, UInt32), P, Any[$(copies[1:k]...)], $k
            )::P)
        end
    ex = upto(nf)
    for i in nf:-1:1
        ex = :(isdefined(x, $i) ? $ex : $(upto(i - 1)))
    end
    return ex
end

function __exclude_unsupported_output_internal!(
    y::T, address_set::Set{UInt}
) where {T<:_BuiltinArrays}
    if objectid(y) in address_set
        throw_circular_reference_or_alias_error(y)
    end

    # mutable types are always stored on the heap.
    push!(address_set, objectid(y))

    # recurse over iterable collections.
    for i in eachindex(y)
        # isassigned() is valid for Arrays, Memory.
        !isassigned(y, i) && continue
        __exclude_unsupported_output_internal!(y[i], address_set)
    end

    return nothing
end

function __exclude_unsupported_output_internal!(
    y::Union{Tuple,NamedTuple}, address_set::Set{UInt}
)
    foreach(Base.Fix2(__exclude_unsupported_output_internal!, address_set), y)
    return nothing
end

# in case f(args...) directly outputs a Ptr{T} or it contains a nested Ptr{T}.
function __exclude_unsupported_output_internal!(y::Ptr, ::Set{UInt})
    return throw_ptr_in_output_error(y)
end

# Prepared-cache type and size for one primal: array inputs record their size, everything
# else records `()`.
@inline _type_and_size(x) =
    if x isa AbstractArray
        TypeAndSize(typeof(x), size(x))
    else
        TypeAndSize(typeof(x), ())
    end

# Shared prepared-cache input validation for Cache, FCache, and HVPCache entry points.
# The expected type T_i is extracted from the TypeAndSize{T_i,S_i} type parameter
# at @generated specialisation time, so the emitted `typeof(x_i) == T_i` comparison uses a
# compile-time constant type — eliminating the runtime jl_types_equal call.
@generated function _check_prepared_cache(types_and_sizes::Tuple, fx::Tuple)
    n = length(types_and_sizes.parameters)
    m = length(fx.parameters)
    n == m || return :(_throw_prepared_cache_mismatch_error(:arity, 0, $n, $m))
    checks = Expr(:block)
    for i in 1:n
        T_i = types_and_sizes.parameters[i].parameters[1]
        push!(
            checks.args,
            quote
                let x_i = fx[$i]
                    typeof(x_i) == $T_i ||
                        _throw_prepared_cache_mismatch_error(:type, $i, $T_i, typeof(x_i))
                    if x_i isa AbstractArray
                        size(x_i) == types_and_sizes[$i].size ||
                            _throw_prepared_cache_mismatch_error(
                                :size, $i, types_and_sizes[$i].size, size(x_i)
                            )
                    end
                end
            end,
        )
    end
    return quote
        $checks
        return fx
    end
end

# Count tangent dimensions with mutable-node deduplication for aliases and cycles.
# Dense leaves reuse Nfwd.primal_dim; IEEEFloat/Complex elements are always assigned.
@inline tangent_dim(t) = tangent_dim(t, IdDict{Any,Any}())
@inline tangent_dim(::NoTangent, ::IdDict{Any,Any}) = 0
@inline function tangent_dim(t::Union{IEEEFloat,Complex{<:IEEEFloat}}, ::IdDict{Any,Any})
    return Nfwd.primal_dim(t)
end
# `Union{}`-eltype arrays (e.g. an empty `Memory{Union{}}` reached while walking a closure
# tangent like the HVP `grad_f`'s `MistyClosureTangent`) carry no differentiable content; 0.
# More specific than the float/complex-array and generic-array methods, so no ambiguity.
@inline tangent_dim(::AbstractArray{Union{}}, ::IdDict{Any,Any}) = 0
@inline function tangent_dim(
    t::AbstractArray{<:Union{IEEEFloat,Complex{<:IEEEFloat}}}, seen::IdDict{Any,Any}
)
    haskey(seen, t) && return 0
    seen[t] = nothing
    return Nfwd.primal_dim(t)
end
@inline function tangent_dim(t::AbstractArray, seen::IdDict{Any,Any})
    haskey(seen, t) && return 0
    seen[t] = nothing
    total = 0
    if t isa _BuiltinArrays
        for i in eachindex(t)
            isassigned(t, i) || continue
            total += tangent_dim(t[i], seen)
        end
    else
        for ti in t
            total += tangent_dim(ti, seen)
        end
    end
    return total
end
@inline function tangent_dim(t::PossiblyUninitTangent, seen::IdDict{Any,Any})
    return is_init(t) ? tangent_dim(val(t), seen) : 0
end
# Walk fields of tuple/named-tuple and custom/closure tangents, deduplicating mutable
# nodes. This also covers Tangent/MutableTangent through their `fields` NamedTuple.
@inline function tangent_dim(t::P, seen::IdDict{Any,Any}) where {P}
    if Base.ismutabletype(P)
        haskey(seen, t) && return 0
        seen[t] = nothing
    end
    total = 0
    for n in 1:fieldcount(P)
        isdefined(t, n) && (total += tangent_dim(getfield(t, n), seen))
    end
    return total
end

@inline _cat_leaves(t::Tuple) = _any_nothing(t) ? nothing : _flatten_leaves(t)
@inline _any_nothing(::Tuple{}) = false
@inline _any_nothing(t::Tuple) = first(t) === nothing || _any_nothing(Base.tail(t))
@inline _flatten_leaves(::Tuple{}) = ()
@inline _flatten_leaves(t::Tuple) = (first(t)..., _flatten_leaves(Base.tail(t))...)
