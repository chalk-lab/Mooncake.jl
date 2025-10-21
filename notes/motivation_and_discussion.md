# [Excerpt Document Loop Optimisation Opportunities](https://github.com/chalk-lab/Mooncake.jl/issues/156)

- `map`, `broadcast`, `mapreduce`, and any other higher-order functions I’ve forgotten about, all lower to loops in the CFG. Tapir.jl doesn’t have rules for them, so Tapir.jl sees these loops.
- Loops in Tapir.jl are reasonably performant. For example, they are completely type-stable and are (usually) allocation-free. So if you’re doing a bit of work at each iteration (e.g. `sin(cos(exp(x[n])))` ), the time spent managing “overhead” associated to looping (e.g. logging stuff on the forwards pass at each iteration which you need on the reverse-pass) is small in comparison to the time spent doing the work that you care about (e.g. computing `sin(cos(exp(x[n])))` and doing AD on each operation in it etc)
- If you’re doing a very small amount of work at each iteration of a loop, then your computation is (currently) dominated by “overhead”. `sum` is an extreme case of this, because adding two `Float64`s together at each iteration is about the cheapest differentiable operation that you could imagine doing. Moreover, the current way that we handle looping in Tapir.jl “gets in the way” of vectorisation (on the forwards-pass and reverse-pass).
- The loop optimisations that I will discuss in the issue will largely target this overhead. They will therefore improve the performance of every single example in this table, but the largest improvements will be seen for `kron` and `sum`. I imagine they’ll be especially great on `sum` as they should be able to “get out of the way” of vectorisation (i.e. things should vectorise nicely) in many cases.

That we just rely on everything boiling down to the same kind of looping structure in the CFG is a great advantage of this approach -- basically everything CPU-based that’s performant gets reduced to a loop in the CFG (specifically, a thing called a “Natural Loop” in compiler optimisation terminology). There are well-established optimisation strategies for loops, so we don’t need to implement separate rules for all the different higher-order functions to get good performance, nor do we need to tell people to steer clear of writing for or while loops.
Rather, we just optimise these so-called “natural loop” structures which appear in the CFG, and then everything (or, rather, most things) will (should) be fast.

(The situation in which this strategy breaks down is if people use `@goto` to produce certain kinds of “weird” looping structures. Such structures will only ever be as performant as they are currently. Frankly, it’s not bad, but we should probably discourage people from using `@goto` , which is definitely something that I can live with)

Mooncake.jl does not perform as well as it could on functions like the following:
```julia
function foo!(y::Vector{Float64}, x::Vector{Float64})
    @inbounds @simd for n in eachindex(x)
        y[n] = y[n] + x[n]
    end
    return y
end
```
For example, on my computer:
```julia
y = randn(4096)
x = randn(4096)

julia> @benchmark foo!($y, $x)
BenchmarkTools.Trial: 10000 samples with 173 evaluations.
 Range (min … max):  547.150 ns …   3.138 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     646.633 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   682.488 ns ± 116.548 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

       ▄██▂                                                      
  ▁▁▂▄▇████▇▇▇▆▆▅▅▅▅▄▃▃▃▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  547 ns           Histogram: frequency by time         1.18 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.

rule = Tapir.build_rrule(foo!, y, x);
foo!_d = zero_fcodual(foo!)
y_d = zero_fcodual(y)
x_d = zero_fcodual(x)
out, pb!! = rule(foo!_d, y_d, x_d);

julia> @benchmark ($rule)($foo!_d, $y_d, $x_d)[2](NoRData())
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  64.042 μs … 202.237 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     78.675 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   75.763 μs ±  10.175 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▇ ▇ ▇ ▄▂       ▅ ▃  ▆ ▅▆ █▄ ▄  ▁▂         ▂       ▁        ▂ ▃
  █▃█▃█▄██▁▄▁▆▄▁▁█▄█▆██████████▇▄██▃▅█▃▃█▆▆▇█▆▆▆▇▆▄▁█▃▃▃▅█▅▃▁█ █
  64 μs         Histogram: log(frequency) by time       108 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
```
So the performance ratio is roughly `64 / 0.5` which is  `128`.

Note that this is not due to type-instabilities. One way to convince yourself of this is that there are no allocations required to run AD, which would most certainly not be the case were there type instabilities.
Rather, the problems are to do with the overhead associated to our implementation of reverse-mode AD.

To see this, take a look at the optimised IR for `foo!`:
```julia
2 1 ── %1  = Base.arraysize(_3, 1)::Int64                                    │╻╷╷╷╷    macro expansion
  │    %2  = Base.slt_int(%1, 0)::Bool                                       ││╻╷╷╷╷    eachindex
  │    %3  = Core.ifelse(%2, 0, %1)::Int64                                   │││╻        axes1
  │    %4  = %new(Base.OneTo{Int64}, %3)::Base.OneTo{Int64}                  ││││┃││││    axes
  └───       goto #14 if not true                                            │╻        macro expansion
  2 ── %6  = Base.slt_int(0, %3)::Bool                                       ││╻        <
  └───       goto #12 if not %6                                              ││       
  3 ──       nothing::Nothing                                                │        
  4 ┄─ %9  = φ (#3 => 0, #11 => %27)::Int64                                  ││       
  │    %10 = Base.slt_int(%9, %3)::Bool                                      ││╻        <
  └───       goto #12 if not %10                                             ││       
  5 ── %12 = Base.add_int(%9, 1)::Int64                                      ││╻╷       simd_index
  └───       goto #9 if not false                                            │││╻        getindex
  6 ── %14 = Base.slt_int(0, %12)::Bool                                      ││││╻        >
  │    %15 = Base.sle_int(%12, %3)::Bool                                     ││││╻        <=
  │    %16 = Base.and_int(%14, %15)::Bool                                    ││││╻        &
  └───       goto #8 if not %16                                              ││││     
  7 ──       goto #9                                                         │        
  8 ──       invoke Base.throw_boundserror(%4::Base.OneTo{Int64}, %12::Int64)::Union{}
  └───       unreachable                                                     ││││     
  9 ┄─       goto #10                                                        │        
  10 ─       goto #11                                                        │        
  11 ─ %23 = Base.arrayref(false, _2, %12)::Float64                          ││╻╷       macro expansion
  │    %24 = Base.arrayref(false, _3, %12)::Float64                          │││┃        getindex
  │    %25 = Base.add_float(%23, %24)::Float64                               │││╻        +
  │          Base.arrayset(false, _2, %25, %12)::Vector{Float64}             │││╻        setindex!
  │    %27 = Base.add_int(%9, 1)::Int64                                      ││╻        +
  │          $(Expr(:loopinfo, Symbol("julia.simdloop"), nothing))::Nothing  │╻        macro expansion
  └───       goto #4                                                         ││       
  12 ┄       goto #14 if not false                                           ││       
  13 ─       nothing::Nothing                                                │        
5 14 ┄       return _2                                                       │        
```
The performance-critical chunk of the loop happens between `%23` and `%27`. Tapir.jl does basically the same kind of thing for each of these lines, so we just look at `%23`:
```julia
%23_ = rrule!!(zero_fcodual(Base.arrayref), zero_fcodual(false), _2, %12)
%23 = %23[1]
push!(%23_pb_stack, %23[2])
```
In short, we run the rule, pull out the first element of the result, and push the pullback to the stack for use on the reverse-pass.

So there is at least one _really_ large obvious source of overhead here: pushing to / popping from the stacks. If you take a look at the pullbacks for the `arrayref` calls, you'll see that they contain:
1. (a reference to) the shadow of the array being referenced, and
2. a copy of the index at which the forwards-pass references the array.

This information is necessary for AD, but 
1. the array being referenced and its shadow are loop invariants -- their value does not change at each iteration of the loop -- meaning that we're just pushing `4096` references to the same array to a stack and popping them, which is wasteful, and
2. the index is an _induction_ _variable_ -- its value changes by a fixed known amount at each loop iteration, meaning that (in principle) we can just recompute it on the reverse-pass rather than storing it.

What's not obvious here, but is also important, is that the call to `push!` tends to get inlined and contains a branch. This prevents LLVM from vectorising the loop, thus prohibiting quite a lot of optimisation.

Now, Tapir.jl is implemented in such a way that, if the pullback for a particular function is a singleton / doesn't carry around any information, the associated pullback stack is eliminated entirely. Moreover, just reducing the amount of memory stored at each iteration should reduce memory pressure. Consequently, a good strategy for making progress is to figure out how to reduce the amount of stuff which gets stored in the pullback stacks. The two points noted above provide obvious starting points.

# Making use of loop invariants

In short: ammend the rule interface such that the arguments to the forwards pass are also made available on the reverse pass.

For example, the `arrayref` rule is presently something along the lines of
```julia
function rrule!!(::CoDual{typeof(arrayref)}, inbounds::CoDual{Bool}, x::CoDual{Vector{Float64}}, ind::CoDual{Int})
    _ind = primal(ind)
    dx = tangent(x)
    function arrayref_pullback(dy)
        dx[_ind] += dy
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(primal(x)[_ind], tangent(x)[_ind]), arrayref_pullback
end
```
This skips some details, but the important point is that `_ind` and `dx` are closed over, and are therefore stored in `arrayref_pullback`.

Under the new interface, this would look something like
```julia
function rrule!!(::CoDual{typeof(arrayref)}, inbounds::CoDual{Bool}, x::CoDual{Vector{Float64}}, ind::CoDual{Int})
    function arrayref_pullback(dy, ::CoDual{typeof(arrayref)}, ::CoDual{Bool}, x::CoDual{Vector{Float64}}, ind::CoDual{Int})
        _ind = primal(ind)
        dx = tangent(x)
        dx[_ind] += dy
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(primal(x)[_ind], tangent(x)[_ind]), arrayref_pullback
end
```
In this version of the rule, `arrayref_pullback` is a singleton because it does not close over any data from the enclosing `rrule!!`.

So this interface change frees up Tapir.jl to provide the arguments on the reverse-pass in whichever way it pleases. In this particular example, both `x` and `y` are arguments to `foo!`, so applying this new interface recursively would give us direct access to them on the reverse pass by construction.
A similar strategy could be employed for variables which aren't arguments by putting them in the storage shared by the forwards and reverse passes.

It's impossible to know for sure how much of an effect this would have, but doing this alone would more than halve the memory requirement for `arrayref` (a `Vector{Float64}` knows its address in memory and its length, which requires 16B of memory, vs an index which is just an `Int` which takes 8B of memory), and do even more for `arrayset` (it requires references to the primal array _and_ to the shadow). Since the pullback for `+` is already a singleton in both the `Float64` and `Int` case, this would more than halve the memory footprint of the loop.

# Induction Variable Analysis

I won't address how we could make use of induction variable analysis here because I'm still trying to get my head around exactly how is easiest to go about it.
Rather, just note that the above interface change is necessary in order to make use of the results of induction variable analysis -- the purpose of induction variable analysis would be to avoid having to store the index on each iteration of the loop, and to just re-compute it on the reverse pass, and give it to the pullbacks. The above change to the interface would permit this.

Another obvious optimisation is to analyse the trip count, and pre-allocate the (necessary) pullback stacks in order to avoid branching during execution (i.e. checking that they're long enough to store the next pullback, and allocating more memory if not).

This is related to induction variable analysis, so we'd probably want to do that first.

Doing this kind of optimisation would enable vectorisation to happen more effectively in AD, as would could completely eliminate branching from a number of tight loops.

# A Compiler-Friendly Implementation of `getfield` et al

`getfield(x, f)` is only fast if `f` is a constant. For example, the following is fast:
```julia
julia> foo(x) = x.f

julia> @code_warntype optimize=true foo((f=5.0, a=4))
MethodInstance for foo(::@NamedTuple{f::Float64, a::Int64})
  from foo(x) @ Main REPL[55]:1
Arguments
  #self#::Core.Const(foo)
  x::@NamedTuple{f::Float64, a::Int64}
Body::Float64
1 ─ %1 = Base.getfield(x, :f)::Float64
└──      return %1
```
You can see that everything infers, because `:f` appears directly in the code. Conversely, the following has inference failures:
```julia
julia> bar(x, f) = getfield(x, f)
bar (generic function with 1 method)

julia> @code_warntype optimize=true bar((f=5.0, a=4), :f)
MethodInstance for bar(::@NamedTuple{f::Float64, a::Int64}, ::Symbol)
  from bar(x, f) @ Main REPL[58]:1
Arguments
  #self#::Core.Const(bar)
  x::@NamedTuple{f::Float64, a::Int64}
  f::Symbol
Body::Union{Float64, Int64}
1 ─ %1 = Main.getfield(x, f)::Union{Float64, Int64}
└──      return %1
```
Type inference / constant propagation is unable to determine which field of `x` will be accessed at compile time, so it must assume that either could be returned. In this case you get union-splitting, but in general you'll fall back to `Any` rather quickly.

## The Problem That This Presents For Mooncake

The `rrule!!` for `getfield` needs to know which field to refer to in the pullback, and therefore needs to know `f` on the reverse-pass. A line of IR of the form
```julia
getfield(%5, :f)
```
in the primal IR is translated to
```julia
rrule!!(CoDual(getfield, NoFData()), %5, CoDual(:f, NoFData()))
```
on the forwards-pass. A schematic implementation of this method of `rrule!!` might be something like
```julia
function rrule!!(::CoDual{typeof(getfield)}, x::CoDual, _f::CoDual{Symbol})
    ...
    function pb!!(dy)
        <use-of-primal-field-of-_f>
    end
    return ..., pb!!
end
```
Crucially, while constant propagation can typically make all uses of `_f` on the forwards-pass performant, constant-ness will _definitely_ be lost on the reverse-pass, because the closure constructed doesn't preserve the constantness of `_f`. In Mooncake, we presently translate each call to `getfield(x, :f)` into `lgetfield(x, Val(:f))`. This encodes `f` as a type, which can be successfully passed into the reverse-pass.

However, this is well-known to yield poor performance, and over-specialisation. In the new design, a rule for `getfield` might look something like the following:
```julia
function rrule!!(::CoDual{typeof(getfield)}, x::CoDual, _f::CoDual{Symbol})
    ...
    function pb!!(dy, ::CoDual, ::CoDual, f::CoDual{Symbol})
        <use-of-primal-field-of-f>
    end
    return ..., pb!!
end
```
In this context, if the argument `f` to the call to `pb!!` on the reverse-pass is a constant, then constant propagation should work correctly. An example of such a call in the reverse-pass IR might be something like
```julia
Expr(:invoke, pb!!, ..., CoDual(:f, NoFData()))
```
Mooncake could trivially emit such code.

## Others

The above argument applies to basically any call where type stability depends upon knowing that certain arguments are constant. Examples of such calls are best found by looking at the passes [here](https://github.com/compintell/Mooncake.jl/blob/06834aa0e061ed46b790b7281019fae133950f28/src/interpreter/ir_normalisation.jl#L252) and [here](https://github.com/compintell/Mooncake.jl/blob/06834aa0e061ed46b790b7281019fae133950f28/src/interpreter/ir_normalisation.jl#L290) in Mooncake.jl -- these are all of the examples of things which require the same kind of mapping as `getfield` -> `lgetfield`.

All of these passes, and the associated function definitions, could be removed. I anticipate that the result would be very substantially improved compile times.

here are some benchmarking numbers to give you a rough sense of how much time is spent doing "book-keeping" vs "interesting work". I'll leave you to extrapolate my comments in the `sum` example to the other examples regarding exactly what each piece of overhead should be attributed to.

## Setup

To replicate, first activate the `bench` directory, and `include` the `run_benchmarks.jl` file. Then run the following:
```julia
function prep_test_case(fargs...)
    rule = Mooncake.build_rrule(fargs...)
    coduals = map(zero_codual, fargs)
    return rule, coduals
end

function run_many_times(N, f, args::Vararg{Any, P}; kwargs...) where {P}
    @inbounds for _ in 1:N
        f(args...; kwargs...)
    end
    return nothing
end
```

You may need to run the `@profview` commands twice to replicate my results -- the first run sometimes includes some compilation stuff.

## `sum`

Simply benchmarking `sum(x)` where `x = randn(1_000)`. Running
```julia
rule, coduals = prep_test_case(sum, randn(1_000));
@profview run_many_times(1_000_000, to_benchmark, rule, coduals...)
```
should yield something like
<img width="907" alt="Image" src="https://github.com/user-attachments/assets/8e24a838-38e3-4cdb-82d4-e7ae399f6d6a" />

Observe that
1. the cost is almost 100% overhead associated with book-keeping, as shown by the fact that all of the time is spent inside `push!` / `pop!` calls.
2. Roughly 30-40% of the time is spent doing book-keeping associated to keeping track of which basic blocks we're visit. You can see this by seeing how much time is spent in calls to `__push_blk_stack` and `__pop_blk_stack!`.
3. The remaining 60-70% of the time is spent performing other book-keeping activities. Having taken a look at the IR generated for the forwards-pass, it looks to be entirely pushing / popping the pullbacks for `memoryref_get` (the new primitive in 1.11 to get the value of an `Array` at a particular location). This makes sense -- the adjoint for `Float64 + Float64` is a singleton, so the only other thing which could possible appear is for `memoryref_get`. This _doesn't_ currently have a singleton pullback because it needs to know the location of the tangent memory on the reverse-pass.

Note: the result of profiling functions like `_kron_sum` and `_kron_view_sum` are very similar to this.

## `map-sin-cos-exp`

To replicate, run:
```julia
rule, coduals = prep_test_case(_map_sin_cos_exp, randn(10, 10));
@profview run_many_times(1_000_000, to_benchmark, rule, coduals...)
```

<img width="2311" alt="Image" src="https://github.com/user-attachments/assets/4102a7e9-673a-4c88-9a59-9a4831c8a4a0" />

For reference, the definition of `_map_sin_cos_exp` is
```julia
_map_sin_cos_exp(x::AbstractArray{<:Real}) = sum(map(x -> sin(cos(exp(x))), x))
```

Observe that:
1. Roughly 50% of the time is spent doing "useful" work in calls to `rrule!!`.
2. The reverse-pass occupies only about 18% of the overall time, and is almost entirely book-keeping.
3. The remaining 30% of the time is spent doing book-keeping in the forwards-pass.
4. As before, the book-keeping time comprising a mix of keeping track of which blocks we're visiting, and the pullbacks for rules.

Either way, if we can get rid of roughly half of the book-keeping overheads, we're looking at a 25% improvement in performance. The AD / primal ratio is currently roughly 2.7, so this improvement would get us down to around 2.0.

## Simple Turing Model

To replicate, run
```julia
rule, coduals = prep_test_case(build_turing_problem()...);
```

<img width="2153" alt="Image" src="https://github.com/user-attachments/assets/c8f8daad-fe5f-441d-a164-57470ee916b4" />

Observe that
1. overall, roughly 43% of the time is spent doing book-keeping related stuff,
2. only 28% of the time is spent doing non-book-keeping work inside `rrule!!` calls,
3. a good chunk of time is spent doing book-keeping, but exactly what is being done is unclear to me. This is indicated by large sections of the flamegraph in which there are calls to generic functions for which not all of the run time is accounted (see e.g. the large overhanging sections of the graph for e.g. `opaque closure` and `DerivedRule`).

All of this is to say that performance could probably be improved quite noticeably by reducing overhead.

Per our discussions, here's a simple hand-derived example in which handling loop-invariants + induction variables correctly should be enough to get us good performance.

Consider the following simple implementation of sum:
```julia
function _sum(f::F, x::AbstractArray{<:Real}) where {F}
    y = 0.0
    n = 0
    while n < length(x)
        n += 1
        y += f(x[n])
    end
    return y
end
```
On Julia LTS, it has the following IRCode:
```julia
julia> Base.code_ircode_by_type(Tuple{typeof(_sum), typeof(identity), Vector{Float64}})
1-element Vector{Any}:
   1 ─      nothing::Nothing                                      │ 
7  2 ┄ %2 = φ (#1 => 0, #3 => %7)::Int64                          │ 
   │   %3 = φ (#1 => 0.0, #3 => %9)::Float64                      │ 
   │   %4 = Base.arraylen(_3)::Int64                              │╻ length
   │   %5 = Base.slt_int(%2, %4)::Bool                            │╻ <
   └──      goto #4 if not %5                                     │ 
8  3 ─ %7 = Base.add_int(%2, 1)::Int64                            │╻ +
9  │   %8 = Base.arrayref(true, _3, %7)::Float64                  │╻ getindex
   │   %9 = Base.add_float(%3, %8)::Float64                       │╻ +
10 └──      goto #2                                               │ 
11 4 ─      return %3                                             │ 
    => Float64
```
It cycles between blocks 2 and 3 to perform the loop, exiting via block 4. In block 2, SSA values `%4` and `%5`, and the `goto if not` that follows them determine whether to continue looping. In block 3, `%7` increments the induction variable, `%8` pulls a value out of the array, `%9` adds it to the current running total, and then we return to block 2.

To understand what book-keeping is happening, it is enough to understand the forwards-pass IR generated by Mooncake. Our goal will be to eliminate the need for various book-keeping data structures in the forwards-pass -- once they're gone from here, they'll disappear from the reverse-pass also.

The fowards-pass IR generated by Mooncake is
```julia
5 1 ─ %1  = (Mooncake.get_shared_data_field)(_1, 1)::Mooncake.Stack{Int32}                                                               │
  │   %2  = (Mooncake.get_shared_data_field)(_1, 2)::Base.RefValue{Tuple{Mooncake.LazyZeroRData{typeof(_sum), Nothing}, Mooncake.LazyZeroRData{typeof(identity), Nothing}, Mooncake.LazyZeroRData{Vector{Float64}, Nothing}}}
  │   %3  = (Mooncake.get_shared_data_field)(_1, 3)::Mooncake.Stack{Tuple{Mooncake.var"#arrayref_pullback!!#635"{1, Vector{Float64}, Int64}}}
  └──       (Mooncake.__assemble_lazy_zero_rdata)(%2, _2, _3, _4)::Core.Const((Mooncake.LazyZeroRData{typeof(_sum), Nothing}(nothing), Mooncake.LazyZeroRData{typeof(identity), Nothing}(nothing), Mooncake.LazyZeroRData{Vector{Float64}, Nothing}(nothing)))
  2 ─       (Mooncake.__push_blk_stack!)(%1, 11)::Core.Const(nothing)                                                                    │
  3 ┄ %6  = φ (#2 => Mooncake.CoDual{Int64, NoFData}(0, NoFData()), #4 => %19)::Mooncake.CoDual{Int64, NoFData}                          │
  │   %7  = φ (#2 => Mooncake.CoDual{Float64, NoFData}(0.0, NoFData()), #4 => %28)::Mooncake.CoDual{Float64, NoFData}                    │
  │   %8  = (identity)(_4)::Mooncake.CoDual{Vector{Float64}, Vector{Float64}}                                                            │
  │   %9  = (Mooncake.rrule!!)($(QuoteNode(Mooncake.CoDual{typeof(Mooncake.IntrinsicsWrappers.arraylen), NoFData}(Mooncake.IntrinsicsWrappers.arraylen, NoFData()))), %8)::Tuple{Mooncake.CoDual{Int64, NoFData}, Mooncake.NoPullback{Tuple{Mooncake.LazyZeroRData{typeof(Mooncake.IntrinsicsWrappers.arraylen), Nothing}, Mooncake.LazyZeroRData{Vector{Float64}, Nothing}}}}
  │   %10 = (getfield)(%9, 1)::Mooncake.CoDual{Int64, NoFData}                                                                           │
  │   %11 = (identity)(%6)::Mooncake.CoDual{Int64, NoFData}                                                                              │
  │   %12 = (identity)(%10)::Mooncake.CoDual{Int64, NoFData}                                                                             │
  │   %13 = (Mooncake.rrule!!)($(QuoteNode(Mooncake.CoDual{typeof(Mooncake.IntrinsicsWrappers.slt_int), NoFData}(Mooncake.IntrinsicsWrappers.slt_int, NoFData()))), %11, %12)::Tuple{Mooncake.CoDual{Bool, NoFData}, Mooncake.NoPullback{Tuple{Mooncake.LazyZeroRData{typeof(Mooncake.IntrinsicsWrappers.slt_int), Nothing}, Mooncake.LazyZeroRData{Int64, Nothing}, Mooncake.LazyZeroRData{Int64, Nothing}}}}
  │   %14 = (getfield)(%13, 1)::Mooncake.CoDual{Bool, NoFData}                                                                           │
  │   %15 = (Mooncake.primal)(%14)::Bool                                                                                                 │
  └──       goto #5 if not %15                                                                                                           │
  4 ─ %17 = (identity)(%6)::Mooncake.CoDual{Int64, NoFData}                                                                              │
  │   %18 = (Mooncake.rrule!!)($(QuoteNode(Mooncake.CoDual{typeof(Mooncake.IntrinsicsWrappers.add_int), NoFData}(Mooncake.IntrinsicsWrappers.add_int, NoFData()))), %17, $(QuoteNode(Mooncake.CoDual{Int64, NoFData}(1, NoFData()))))::Tuple{Mooncake.CoDual{Int64, NoFData}, Mooncake.NoPullback{Tuple{Mooncake.LazyZeroRData{typeof(Mooncake.IntrinsicsWrappers.add_int), Nothing}, Mooncake.LazyZeroRData{Int64, Nothing}, Mooncake.LazyZeroRData{Int64, Nothing}}}}
  │   %19 = (getfield)(%18, 1)::Mooncake.CoDual{Int64, NoFData}                                                                          │
  │   %20 = (identity)(_4)::Mooncake.CoDual{Vector{Float64}, Vector{Float64}}                                                            │
  │   %21 = (identity)(%19)::Mooncake.CoDual{Int64, NoFData}                                                                             │
  │   %22 = (Mooncake.rrule!!)($(QuoteNode(Mooncake.CoDual{typeof(Core.arrayref), NoFData}(Core.arrayref, NoFData()))), $(QuoteNode(Mooncake.CoDual{Bool, NoFData}(true, NoFData()))), %20, %21)::Tuple{Mooncake.CoDual{Float64, NoFData}, Mooncake.var"#arrayref_pullback!!#635"{1, Vector{Float64}, Int64}}
  │   %23 = (getfield)(%22, 1)::Mooncake.CoDual{Float64, NoFData}                                                                        │
  │   %24 = (getfield)(%22, 2)::Mooncake.var"#arrayref_pullback!!#635"{1, Vector{Float64}, Int64}                                        │
  │   %25 = (identity)(%7)::Mooncake.CoDual{Float64, NoFData}                                                                            │
  │   %26 = (identity)(%23)::Mooncake.CoDual{Float64, NoFData}                                                                           │
  │   %27 = (Mooncake.rrule!!)($(QuoteNode(Mooncake.CoDual{typeof(Mooncake.IntrinsicsWrappers.add_float), NoFData}(Mooncake.IntrinsicsWrappers.add_float, NoFData()))), %25, %26)::Tuple{Mooncake.CoDual{Float64, NoFData}, Mooncake.IntrinsicsWrappers.var"#add_float_pb!!#2"}
  │   %28 = (getfield)(%27, 1)::Mooncake.CoDual{Float64, NoFData}                                                                        │
  │   %29 = (tuple)(%24)::Tuple{Mooncake.var"#arrayref_pullback!!#635"{1, Vector{Float64}, Int64}}                                       │
  │         (push!)(%3, %29)::Core.Const(nothing)                                                                                        │
  │         (Mooncake.__push_blk_stack!)(%1, 13)::Core.Const(nothing)                                                                    │
  └──       goto #3                                                                                                                      │
  5 ─       return %7 
```
(before inlining). To orient yourself, observe that
1. for each basic block in the original code, there is one basic block in the forwards-pass, with a single additional block inserted at the top to handle some book-keeping. i.e. the counterpart of block 1 in the primal IR is block 2 in the forwards-pass IR.
2. each phi node in the primal has a corresponding phi node in the forwads-pass. e.g. `%2` -> `%6` and `%3` - `%7`.
3. each goto node and goto-if-not node in the primal has a corresponding node in the forwards-pass.
4. each call / invoke expression in the primal corresponds to a collection of lines in the forwards-pass. There should be one call to `rrule!!` for each call to a "primitive" function (`add_int`, `slt_int`, etc).

The thing that we're interested in optimising away is the call to `push!` at the end of block `#4`. If you follow the chain of SSA values back up, you'll see that the the thing we push onto it is a `Tuple` containing only the pullback returned by the call to `rrule!!` for `Core.arrayref` (i.e. `the low-level implementation of `getindex` for `Array`s).

To understand what needs to happen here, observe that the type of the pullback for `arrayref` is `Mooncake.var"#arrayref_pullback!!#635"{1, Vector{Float64}, Int64}` -- it contains a `Vector{Float64}` and an `Int64`. If you inspect the implementation you will see that the `Vector{Float64}` is the tangent vector and the `Int64` is element of the primal vector that we grab on the forwards-pass. This means that in order to make things more efficient, we need to find a different way to provide these quantities on the reverse-pass.

Happily, this is quite straightforward in principle. We can see in the primal code that the array being passed in to `arrayref` is`_3` i.e. the third positional argument to the function. This array is _definitely_ a loop invariant, so we need not store it at each iteration. Similarly, `%7` is just the result of adding `1` to an induction variable -- this is also straightforward to handle once we've performed induction variable analysis (any loop-invariant affine transformation of an induction variable is straightforward to handle).

Once we've made it so that `arrayref_pullback!!` doesn't have to store these quantities for the reverse-pass itself, it can become a singleton. This will cause the complete removal of the stack associated to line %29, and deal with roughly half of all of the overhead in this function.

The remainder of the overhead is a discussion for later, but I believe that the majority of it just requires induction variable analysis.

Let's divide the book-keeping overhead into two categories:
1. keeping track of which blocks you are visiting, and
2. storing pullbacks for the reverse-pass in ``pullback stacks''.

I agree that your proposal to do some loop unrolling ought to reduce the first source of book-keeping overhead -- I agree that it ought to roughly reduce this overhead by half if you have two statements per block, by a third if you have three statements, etc.

I also agree that it is likely to have _some_ effect on the overhead associated to pushing / popping data to / from the pullback stacks (you'll have halve the pushes / pops if you double the number of statements per block), but it won't change the amount of data actually being stored in these stacks. This makes it hard to reason about its impact in practice without measuring what's going on without benchmarking -- I'll do some benchmarking with your manually-unrolled example, and see what happens.

Implementation: I'm fairly certain that in order to unroll a loop we'd have to
1. identify natural loops,
2. identify induction variables,
3. modify the primal IR.

(The first two steps are required in order to identify the loop, and which variable we're iterating over). So it seems to me that all of the ideas discussed here require that we identify the natural loops, and both loop unrolling / exploiting structure in induction variables require that we identify induction variables. This makes me think that it is unlikely that loop unrolling is likely to be any easier than any of the other things that we have discussed. I actually think it might be a bit more awkward, because we would have to modify the primal IR. Do you think I'm missing something?

In terms of how well understood the techniques are, my impression is that natural loop identification, loop unrolling, induction variable analysis, and the identification of loop invariants (and associated code motion) are all very well understood.

Additional benchmark for `_sum`. To reproduce, follow instructions above, and run:
```julia
rule, coduals = prep_test_case(_sum, identity, randn(1_000));
@profview run_many_times(1_000_000, to_benchmark, rule, coduals...)
```

<img width="1232" alt="Image" src="https://github.com/user-attachments/assets/fdd64140-755a-4f29-aaf8-de0a3fe97028" />

As with `sum`, largely book-keeping. Looks like there's a little bit of time spent in `rrule!!`s on the forwards pass though.
