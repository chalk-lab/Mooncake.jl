# Cross-call scratch reuse via a single process-global pool.
#
# A primitive rule allocates internal workspace (`similar(A)` temporaries) on every call; when a rule
# is prepared once and run many times (training / MCMC loops) that is pure GC pressure. A rule opts
# into reuse by drawing workspace from the global `ScratchPool` via `buffered_similar!(proto)`
# instead of `similar` — filled by an *in-place* op. The pool is reset once per top-level AD call:
# the public entry points (`value_and_gradient!!` / `value_and_pullback!!` / `value_and_derivative!!`)
# wrap their runner call in `with_scratch_pool`, so slots are reused across calls but never within
# one. The wrap lives at the public entry, NOT in the internal `__value_and_*` runners: forward-over-
# reverse forward-differentiates the reverse runner, and a lock there is not differentiable (it hits
# a `jl_gc_disable_finalizers_internal` foreigncall), so the runners must stay lock-free.
#
# Two locks make it concurrency-safe:
#   - the OUTER lock (`with_scratch_pool`) is held for a whole top-level AD call, serializing them so
#     two never share the pool, and resets the cursor at entry;
#   - the INNER lock (`buffered_similar!`) makes the cursor push atomic, so worker tasks inside a
#     differentiated `Threads.@threads` region each get a distinct slot (top-level AD calls are
#     already serialized by the outer lock, so the only concurrency is intra-call workers).
#
# There is NO per-call reset of forward scratch — each `buffered_similar!` pushes a fresh slot and
# the pool grows across a call, reclaimed only by the next call's reset. So a forward `frule!!` does
# not reuse workspace within one AD call (a loop calling a buffered frule holds one buffer per
# iteration); this is the deliberate cost of a single shared pool with no per-rule cursor. Reverse
# `rrule!!`s likewise just push: the pullback-captured buffers stay live through the reverse sweep
# (the whole forward sweep precedes the whole reverse sweep, so every buffer is live at the peak
# regardless) and are reclaimed at the next call's reset.
#
# The pool removes only the buffer alloc; allocations inside opaque calls (`BLAS`/`LAPACK`
# internals) or from non-in-place ops (`A*B`, unfused broadcast) still need the op itself made
# in-place. A pooled buffer is reused memory, not a freshly-allocated tracked dual, so a buffered
# reverse restore-copy does not compose with forward-over-reverse (HVP/Hessian) — it fails loudly (a
# `:memmove` ccall error), never silently; a reverse rule needing nested-AD support keeps
# `Base.@noinline copy` (as `trtrs!` does) instead of pooling.

mutable struct ScratchPool
    backings::Vector{Any}
    pos::Int
end
ScratchPool() = ScratchPool(Any[], 0)

const _SCRATCH_POOL = ScratchPool()
# Outer: held for a whole top-level AD call, serializing them and resetting the cursor at entry.
const _SCRATCH_LOCK = ReentrantLock()
# Inner: makes the cursor push atomic so concurrent worker tasks (a buffered rule inside a
# differentiated `Threads.@threads`) each get a distinct slot.
const _SCRATCH_PUSH_LOCK = ReentrantLock()
# Separate exact-size pool for transient intermediate tangent leaves (`Memory{T}`): a tangent must be
# exactly `n` long, so it can't ride the grow-keep-max `_SCRATCH_POOL` (which serves sub-length wraps).
const _TANGENT_POOL = ScratchPool()

"""
    with_scratch_pool(f)

Run `f()` as one top-level AD call under the scratch-pool discipline: take the outer lock (so
top-level AD calls serialize and never share the pool concurrently), reset the pool cursor, run,
release. The public `value_and_gradient!!` / `value_and_pullback!!` / `value_and_derivative!!` entry
points wrap their runner call in this — never the internal `__value_and_*` runners, which must stay
lock-free so forward-over-reverse can forward-differentiate them.
"""
@inline function with_scratch_pool(f::F) where {F}
    lock(_SCRATCH_LOCK)
    try
        _SCRATCH_POOL.pos = 0
        _TANGENT_POOL.pos = 0
        return f()
    finally
        unlock(_SCRATCH_LOCK)
    end
end

# Backing store for a pool slot, and how a request is served from it. Version-dependent:
@static if VERSION >= v"1.11-"
    # Julia ≥1.11: a `Memory{T}` wrapped as a real `Array` via `Base.wrap`, which roots the `Memory`
    # (GC-safe — no dangling) and supports sub-length wraps, so grow-only-keep-max still holds. The
    # result is a *canonical* `Array`, so it can also back canonical outputs (e.g. an `NDualArray`).
    _pool_backing(::Type{T}, n::Int) where {T} = Memory{T}(undef, n)
    _is_pool_backing(x, ::Type{T}) where {T} = x isa Memory{T}
    function _pool_array(m::Memory{T}, dims::NTuple{N,Int}) where {T,N}
        return Base.wrap(Array, m, dims)::Array{T,N}
    end
else
    # Julia <1.11: a `Vector{T}` backing wrapped as a canonical `Array` via `unsafe_wrap` (there is
    # no `Memory` yet). The wrap does not itself root the backing, but the pool holds it alive for
    # the whole computation and never regrows a slot while a wrap of it is live (one request per slot
    # per call, reset only between calls), so it cannot dangle — and returning a canonical `Array`
    # keeps `buffered_similar!` coherent with the ≥1.11 branch.
    _pool_backing(::Type{T}, n::Int) where {T} = Vector{T}(undef, n)
    _is_pool_backing(x, ::Type{T}) where {T} = x isa Vector{T}
    function _pool_array(v::Vector{T}, dims::NTuple{N,Int}) where {T,N}
        return unsafe_wrap(Array, pointer(v), dims)::Array{T,N}
    end
end

"""
    buffered_similar!(proto::AbstractArray) -> AbstractArray

Return a workspace array with the same element type and shape as `proto`, drawn from the global
scratch pool and reused across AD calls — a pooled replacement for `similar(proto)`.

Intended for **dense arrays of `isbits` (bits-type) elements**: the numeric workspace of
BLAS/LAPACK-style rules (`Float32`/`Float64`/`Complex{…}` and other isbits scalars). Each pool slot
keeps a backing that grows only to the largest length ever requested and is shared into a
contiguous, strided **canonical `Array`** of `size(proto)` — so it feeds in-place
`BLAS`/`LAPACK`/broadcast like a fresh `Array` and reuses with essentially no allocation once the
backing has reached size. The push is taken under the inner lock so concurrent worker tasks (a
buffered rule inside a differentiated `Threads.@threads`) each get a distinct slot.

Semantics match `similar`, not `zeros`: the returned buffer holds **uninitialised / stale** contents,
so the body must fully write it before reading — `copyto!` it to replace a `copy`, `fill!` it to
replace a `zeros`. It is **fixed-shape** workspace aliasing pooled storage — fill it in place, do not
`resize!`/`push!` it, do not let it escape the rule (the next AD call reclaims its slot). A rule must
only call this while running under `with_scratch_pool` (i.e. from a real AD call).

NOT intended for (keep a fresh allocation instead):
- arrays with **non-isbits / boxed** elements (`Array{Any}`, `Array{<:AbstractArray}`,
  `Array{BigInt}`) — a shared flat backing would retain boxed references across calls;
- **composite AD values** themselves (`CoDual`/`Lifted`/`NDualArray`/`Tangent`) — buffer a
  composite's individual dense array leaves, not the composite;
- non-array or dynamically-grown workspace (`Dict`, `push!`-grown vectors) and scalars/tuples that
  do not heap-allocate — no applicable benefit.
"""
@inline function buffered_similar!(proto::AbstractArray{T,N}) where {T,N}
    # `isbitstype(T)` folds at compile time (`T` is a type parameter), so this guards the intended
    # domain — dense isbits-element arrays — with no cost on the valid path. A shared flat backing of
    # boxed elements would retain references across calls, so those must keep a fresh allocation.
    isbitstype(T) || throw(
        ArgumentError(
            "buffered_similar! is scoped to dense arrays of isbits elements; got eltype $T. " *
            "Use a fresh allocation (`similar`) for boxed / non-isbits element types.",
        ),
    )
    n = length(proto)
    # Atomic push (distinct slot per concurrent worker task); grow-only-keep-max per slot.
    backing = @lock _SCRATCH_PUSH_LOCK begin
        i = (_SCRATCH_POOL.pos += 1)
        if i > length(_SCRATCH_POOL.backings)
            b = _pool_backing(T, n)
            push!(_SCRATCH_POOL.backings, b)
            b
        else
            cached = _SCRATCH_POOL.backings[i]
            if _is_pool_backing(cached, T) && length(cached) >= n
                cached
            else
                b = _pool_backing(T, n)   # grow (keep-max) or adapt to a new element type
                _SCRATCH_POOL.backings[i] = b
                b
            end
        end
    end
    return _pool_array(backing, size(proto))
end

@static if VERSION >= v"1.11-"
    """
        buffered_zero_memory!(::Type{T}, n::Int) -> Memory{T}

    Return a zero-filled `Memory{T}` of exactly length `n`, drawn from the tangent pool and reused
    across AD calls — a pooled replacement for `zero_tangent`'s leaf `Memory{T}(undef, n)` + zero.
    For **transient intermediate tangents only** (e.g. the `Memory{P}(undef, n)` rrule's `dx`): the
    slot is reclaimed at the next AD call's reset, so it must never back a persistent (cache) tangent
    or one that escapes the call. Exact-size (not grow-keep-max): a tangent `Memory` cannot be a
    sub-length view, so a slot reallocates when its length or eltype changes.
    """
    @inline function buffered_zero_memory!(::Type{T}, n::Int) where {T}
        m = @lock _SCRATCH_PUSH_LOCK begin
            i = (_TANGENT_POOL.pos += 1)
            if i > length(_TANGENT_POOL.backings)
                b = Memory{T}(undef, n)
                push!(_TANGENT_POOL.backings, b)
                b
            else
                cached = _TANGENT_POOL.backings[i]
                if cached isa Memory{T} && length(cached) == n
                    cached
                else
                    b = Memory{T}(undef, n)
                    _TANGENT_POOL.backings[i] = b
                    b
                end
            end
        end::Memory{T}
        fill!(m, zero(T))
        return m
    end
end
