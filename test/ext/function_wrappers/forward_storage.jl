# These test-only tripwires fail immediately if a walker leaves derivative storage.
function Mooncake.tangent_dim(::Union{Core.OpaqueClosure,Method}, ::IdDict{Any,Any})
    error("visited executable code")
end
function Mooncake._check_tangent_storage!(::IdDict, ::Core.OpaqueClosure, supplied)
    error("visited executable code")
end
Mooncake.friendly_tangent_cache(::Type) = error("visited type metadata")

struct WrapperGain{M}
    a::Float64
    method::M
end
WrapperGain(a) = WrapperGain(a, nothing)
(g::WrapperGain)(x) = g.a * x

struct WrapperHold{W}
    w::W
end
(h::WrapperHold)(x) = h.w(x)
mutable struct MutableWrapperHold{W}
    w::W
end
mutable struct WrapperAccumulator
    a::Float64
end
(f::WrapperAccumulator)(x::Float64) = (f.a += x)
function Mooncake.hand_written_rule_test_cases(rng_ctor, ::Val{:function_wrappers})
    fw = FunctionWrapper{Float64,Tuple{Float64}}(WrapperAccumulator(1.0))
    cases = Any[(false, :none, (;), fw, 2.0)]
    for T in (FunctionWrapper{Float64,Tuple{Any}}, FunctionWrapper{Any,Tuple{Float64}})
        push!(cases, (false, :none, (mode=Mooncake.ForwardMode,), T(WrapperGain(3.0)), 2.0))
    end
    return cases, Any[]
end

@testset "forward wrapper storage" begin
    FW = FunctionWrapper{Float64,Tuple{Float64}}
    call_wrapper(w, x) = w(x)
    for obj in (sin, WrapperGain(3.0))
        fw = FW(obj)
        dfw = Mooncake.zero_tangent(fw)
        if obj isa WrapperGain
            dfw.dobj_ref[] = Mooncake.Tangent((a=2.0, method=Mooncake.NoTangent()))
        end
        expected = obj isa WrapperGain ? (1.5, 4.0) : (sin(0.5), cos(0.5))
        @test Mooncake.tangent_dim(dfw) == (obj isa WrapperGain ? 1 : 0)
        rule = Mooncake.build_frule(call_wrapper, fw, 0.5)
        args = ((call_wrapper, Mooncake.NoTangent()), (fw, dfw), (0.5, 1.0))
        @test Mooncake.value_and_derivative!!(rule, args...) == expected
        @testset "prepared $obj" begin
            cache = Mooncake.prepare_derivative_cache(call_wrapper, fw, 0.5)
            @test Mooncake.value_and_derivative!!(cache, args...) == expected
        end
        @test Mooncake.tangent_to_friendly!!(fw, dfw) === dfw
    end
    fw = FW(WrapperGain(3.0))
    call_twice(t, x) = t[1](x) + t[2](x)
    for w in (fw, FW(WrapperHold(fw))), N in (1, 8)
        cache = Mooncake.prepare_derivative_cache(
            call_twice, (w, w), 0.5; config=Mooncake.Config(chunk_size=N)
        )
        y, g = Mooncake.value_and_gradient!!(cache, call_twice, (w, w), 0.5)
        yr, gr = Mooncake.value_and_gradient!!(
            Mooncake.prepare_gradient_cache(call_twice, (w, w), 0.5),
            call_twice,
            (w, w),
            0.5,
        )
        @test y == yr == 3.0
        @test Mooncake.TestUtils.has_equal_data(g, gr)
        @test g[2][1] === g[2][2]
    end
    t = Mooncake.zero_tangent(fw)
    t.dobj_ref[] = Mooncake.Tangent((a=1.0, method=Mooncake.NoTangent()))
    @test Mooncake.increment!!(t, t).dobj_ref[].fields.a == 1.0
    scaled = Mooncake._scale(2.0, (t, t))
    @test scaled[1] === scaled[2]
    perturbed = Mooncake._add_to_primal((fw, fw), (t, t))
    @test perturbed[1] === perturbed[2]
    # Callable custom tangents can be scalars, with no child identity guard.
    scalar_t = Mooncake.zero_tangent(fw)
    scalar_t.dobj_ref = Ref(2.0)
    @test Mooncake._dot((scalar_t, scalar_t), (scalar_t, scalar_t)) == 4.0
    # Lifting must reuse compiled code, while giving each seed its own storage.
    v = Mooncake.tangent(Mooncake.lift(fw, t))
    @test v.frule === t.frule
    @test v.fwds_wrapper === t.fwds_wrapper
    @testset "native seed writes" begin
        fw = FW(WrapperGain(3.0))
        dfw = Mooncake.zero_tangent(fw)
        @test_throws r"use lift" Mooncake.Lifted{typeof(fw),1}(fw, dfw)
        seed = Mooncake.zero_tangent(fw)
        seed.dobj_ref[] = Mooncake.Tangent((a=1.0, method=Mooncake.NoTangent()))
        for _ in 1:2
            dfw.dobj_ref[] = seed.dobj_ref[]
            @test Mooncake.tangent(
                Mooncake.frule!!(Mooncake.lift(fw, dfw), Mooncake.lift(2.0, 0.0)), 1
            ) == 2
            Mooncake.set_to_zero!!(dfw)
            @test Mooncake.tangent(
                Mooncake.frule!!(Mooncake.lift(fw, dfw), Mooncake.lift(2.0, 0.0)), 1
            ) == 0
            Mooncake.increment!!(dfw, seed)
            @test Mooncake.tangent(
                Mooncake.frule!!(Mooncake.lift(fw, dfw), Mooncake.lift(2.0, 0.0)), 1
            ) == 2
        end
        scaled = Mooncake._scale(2.0, dfw)
        @test Mooncake.tangent(
            Mooncake.frule!!(Mooncake.lift(fw, scaled), Mooncake.lift(2.0, 0.0)), 1
        ) == 4
    end
    @testset "canonical storage" for N in (1, 8)
        fw = FW(WrapperGain(3.0))
        slot = Mooncake.zero_lifted(Val(N), fw)
        v = Mooncake.tangent(slot)
        native = Mooncake.zero_tangent(fw)
        @test_throws r"use lift" Mooncake._check_shared_input_tangents((fw,), (v,))
        @test_throws r"use lift" Mooncake._basis_seed!!(native, (1,), Ref(0), IdDict())
        @test_throws r"use lift" Mooncake.Lifted{typeof(fw),N}(fw, native)
        @test typeof(v) === Mooncake.dual_type(Val(N), typeof(fw))
        @test_throws r"requires a native FunctionWrapper tangent" Mooncake.lift(fw, v)
        different = FW(sin)
        @test_throws r"different callable type" Mooncake.frule!!(
            Mooncake.Lifted{typeof(different),N}(different, v),
            Mooncake.zero_lifted(Val(N), 2.0),
        )
        @test Mooncake._check_shared_input_tangents((slot, slot)) === nothing
        @test_throws ArgumentError Mooncake._scale(2.0, v)
        @test_throws ArgumentError Mooncake.set_to_zero!!(v)
        @test_throws ArgumentError Mooncake.increment!!(v, deepcopy(v))
        @test Mooncake.tangent_dim(v) == 1
        Mooncake.basis_lifted!!(slot, ntuple(_ -> 1, N))
        @test Mooncake.tangent(slot, 1).dobj_ref[].fields.a == 1.0
        other = FW(WrapperGain(7.0))
        rebound = Mooncake.Lifted{typeof(other),N}(other, v)
        x = Mooncake.basis_lifted!!(Mooncake.zero_lifted(Val(N), 2.0), ntuple(_ -> 1, N))
        out = Mooncake.frule!!(rebound, x)
        @test Mooncake.primal(out) == 14.0
        @test Mooncake.tangent(out, 1) == 9.0
        @test Mooncake.tangent(Mooncake.frule!!(slot, x), 1) == 5.0
        rule = Mooncake.build_frule(call_wrapper, fw, 2.0; chunk_size=N)
        out = Mooncake.value_and_derivative!!(
            rule, Mooncake.zero_lifted(Val(N), call_wrapper), slot, x
        )
        @test (Mooncake.primal(out), Mooncake.tangent(out, 1)) == (6.0, 5.0)
    end
    @testset "rebound mutable captures" for N in (1, 8)
        capture(a) = x -> a[1] * x
        for (obj, other) in (
            (WrapperAccumulator(3.0), WrapperAccumulator(7.0)),
            (capture([3.0]), capture([7.0])),
        )
            fw, fw2 = FW(obj), FW(other)
            slot = Mooncake.zero_lifted(Val(N), fw)
            rebound = Mooncake.Lifted{typeof(fw2),N}(fw2, Mooncake.tangent(slot))
            x = Mooncake.basis_lifted!!(
                Mooncake.zero_lifted(Val(N), 2.0), ntuple(_ -> 1, N)
            )
            out = Mooncake.frule!!(rebound, x)
            @test Mooncake.primal(out) == (obj isa WrapperAccumulator ? 9.0 : 14.0)
            @test Mooncake.tangent(out, 1) == (obj isa WrapperAccumulator ? 1.0 : 7.0)
        end
    end
    @testset "abstract signature" for T in (
        FunctionWrapper{Float64,Tuple{Any}}, FunctionWrapper{Any,Tuple{Float64}}
    )
        fw = T(x -> 2.0 * x)
        slot = Mooncake.zero_lifted(Val(1), fw)
        @test typeof(Mooncake.tangent(slot)) === Mooncake.dual_type(Val(1), T)
        for input in (slot, Mooncake.lift(fw, Mooncake.zero_tangent(fw)))
            @test Mooncake.unlift(Mooncake.frule!!(input, Mooncake.lift(3.0, 1.0))) ==
                (6.0, 2.0)
        end
        cache = Mooncake.prepare_derivative_cache(call_wrapper, fw, 3.0)
        @test Mooncake.value_and_gradient!!(cache, call_wrapper, fw, 3.0)[2][3] == 2.0
        t = Mooncake.zero_tangent(T(identity))
        @test Mooncake.TestUtils.has_equal_data(Mooncake._scale(2.0, t), t)
        @test Mooncake.TestUtils.has_equal_data(deepcopy(t), t)
    end
    scaled = Mooncake._scale(2.0, t)
    @test scaled.frule === t.frule
    Mooncake.zero_tangent(fw)
    @test (@allocated Mooncake._scale(2.0, t)) < (@allocated Mooncake.zero_tangent(fw))
    @testset "friendly raw wrapper" for N in (1, 8), metadata in (nothing, Base)
        fw = FW(WrapperGain(3.0, metadata))
        dfw = Mooncake.zero_tangent(fw)
        dfw.dobj_ref[] = Mooncake.Tangent((a=1.0, method=Mooncake.NoTangent()))
        config = Mooncake.Config(; friendly_tangents=true, chunk_size=N)
        cache = Mooncake.prepare_derivative_cache(call_wrapper, fw, 2.0; config)
        @test Mooncake.value_and_derivative!!(
            cache, (call_wrapper, Mooncake.NoTangent()), (fw, dfw), (2.0, 0.0)
        ) == (6.0, 2.0)
        @test_throws r"native tangents for directions" Mooncake.value_and_derivative!!(
            cache,
            (call_wrapper, Mooncake.NoTangent()),
            (fw, FW(WrapperGain(1.0))),
            (2.0, 0.0),
        )
        y, g = Mooncake.value_and_gradient!!(cache, call_wrapper, fw, 2.0)
        yr, gr = Mooncake.value_and_gradient!!(
            Mooncake.prepare_gradient_cache(call_wrapper, fw, 2.0; config),
            call_wrapper,
            fw,
            2.0,
        )
        @test y == yr == 6.0
        # Forward keeps stateless callables primal-shaped; reverse returns NoTangent.
        @test g[1] === call_wrapper
        @test gr[1] === Mooncake.NoTangent()
        @test Mooncake.TestUtils.has_equal_data(Base.tail(g), Base.tail(gr))
    end
    @testset "nested friendly raw wrapper" for N in (1, 8)
        fw = FW(WrapperGain(3.0))
        dfw = Mooncake.zero_tangent(fw)
        dfw.dobj_ref[] = Mooncake.Tangent((a=1.0, method=Mooncake.NoTangent()))
        config = Mooncake.Config(; friendly_tangents=true, chunk_size=N)
        call_held(h, x) = h.w(x)
        for (f, arg, direction, expected) in (
            (call_twice, (fw, fw), (dfw, dfw), (12.0, 4.0)),
            (call_twice, [fw, fw], [dfw, dfw], (12.0, 4.0)),
            (call_twice, Dict(1 => fw, 2 => fw), Dict(1 => dfw, 2 => dfw), (12.0, 4.0)),
            (call_twice, IdDict(1 => fw, 2 => fw), IdDict(1 => dfw, 2 => dfw), (12.0, 4.0)),
            (
                (h, x) -> h.w[1](x),
                WrapperHold(Dict(1 => fw)),
                (w=Dict(1 => dfw),),
                (6.0, 2.0),
            ),
            (
                (h, x) -> h.w[1](x),
                WrapperHold(IdDict(1 => fw)),
                (w=IdDict(1 => dfw),),
                (6.0, 2.0),
            ),
            (call_held, (w=fw,), (w=dfw,), (6.0, 2.0)),
            (call_held, WrapperHold(fw), (w=dfw,), (6.0, 2.0)),
            (call_held, MutableWrapperHold(fw), (w=dfw,), (6.0, 2.0)),
        )
            if arg isa Dict || (arg isa WrapperHold && arg.w isa Dict)
                @test_throws "Prepared caches do not support Dicts whose values can carry derivatives" Mooncake.prepare_derivative_cache(
                    f, arg, 2.0; config
                )
                @test_throws "Prepared caches do not support Dicts whose values can carry derivatives" Mooncake.prepare_gradient_cache(
                    f, arg, 2.0; config
                )
                native = Mooncake._friendly_input_tangent(
                    Mooncake.friendly_tangent_cache(arg),
                    Mooncake.zero_tangent(arg),
                    direction,
                    IdDict(),
                    UInt[],
                    arg,
                    IdDict(),
                )
                @test Mooncake.value_and_derivative!!(
                    Mooncake.build_frule(f, arg, 2.0),
                    (f, Mooncake.NoTangent()),
                    (arg, native),
                    (2.0, 0.0),
                ) == expected
                yr, gr = Mooncake.value_and_gradient!!(
                    Mooncake.build_rrule(f, arg, 2.0), f, arg, 2.0
                )
                @test (yr, Mooncake._dot(gr[2], native)) == expected
                continue
            end
            cache = Mooncake.prepare_derivative_cache(f, arg, 2.0; config)
            y, g = Mooncake.value_and_gradient!!(cache, f, arg, 2.0)
            yr, gr = Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(f, arg, 2.0; config), f, arg, 2.0
            )
            @test y == yr
            @test Mooncake.TestUtils.has_equal_data(Base.tail(g), Base.tail(gr))
            arg isa Union{Tuple,Vector} && @test g[2][1] === g[2][2]
            @test Mooncake.value_and_derivative!!(
                cache, (f, Mooncake.NoTangent()), (arg, direction), (2.0, 0.0)
            ) == expected
        end
    end
    @testset "persistent shared wrapper" begin
        state = Ref(1.0)
        fw = FW(x -> (state[] += x))
        slot = Mooncake.zero_lifted(Val(1), (fw, fw))
        @test Mooncake.tangent(slot)[1] === Mooncake.tangent(slot)[2]
        holder = WrapperHold(fw)
        f(h, x) = h.w(x)
        rule = Mooncake.build_frule(f, holder, 2.0)
        h = Mooncake.zero_lifted(Val(1), holder)
        for expected in (1.0, 2.0)
            out = rule(Mooncake.zero_lifted(Val(1), f), h, Mooncake.lift(2.0, 1.0))
            @test Mooncake.tangent(out, 1) == expected
        end
        twice(w, x) = w(x) + w(x)
        rule = Mooncake.build_frule(twice, fw, 2.0)
        out = rule(
            Mooncake.zero_lifted(Val(1), twice),
            Mooncake.zero_lifted(Val(1), fw),
            Mooncake.lift(2.0, 1.0),
        )
        @test Mooncake.tangent(out, 1) == 3
    end
    # Storage validation must also exclude the primal metadata inside a forward slot.
    fw = FW(WrapperGain(3.0, which(sin, (Float64,))))
    @test_throws r"supplied tangents do not share it" Mooncake._check_shared_input_tangents((
        Mooncake.zero_lifted(Val(1), fw), Mooncake.zero_lifted(Val(1), fw)
    ))
    fw = FW(sin)
    call_two(a, b, x) = a(x) + b(x)
    cache = Mooncake.prepare_derivative_cache(call_two, fw, fw, 2.0)
    out = Mooncake.value_and_derivative!!(
        cache,
        Mooncake.zero_lifted(Val(1), call_two),
        Mooncake.zero_lifted(Val(1), fw),
        Mooncake.zero_lifted(Val(1), fw),
        Mooncake.lift(2.0, 1.0),
    )
    @test Mooncake.primal(out) ≈ 2sin(2.0)
    @test Mooncake.tangent(out, 1) ≈ 2cos(2.0)
    @testset "repeated wrapper containers" begin
        held(a, b, x) = a.w(x) + b.w(x)
        indexed(a, b, x) = a[1](x) + b[1](x)
        referenced(a, b, x) = a[](x) + b[](x)
        for (wrap, f) in
            ((MutableWrapperHold, held), (w -> [w], indexed), (Ref, referenced))
            for obj in (sin, WrapperGain(3.0)), N in (1, 8)
                h = wrap(FW(obj))
                # A cache clamps this one-dimensional stateless case to width 1.
                cache = if N == 1
                    Mooncake.prepare_derivative_cache(f, h, h, 2.0)
                else
                    Mooncake.build_frule(f, h, h, 2.0; chunk_size=N)
                end
                args = (
                    Mooncake.zero_lifted(Val(N), f),
                    Mooncake.zero_lifted(Val(N), h),
                    Mooncake.zero_lifted(Val(N), h),
                    Mooncake.basis_lifted!!(
                        Mooncake.zero_lifted(Val(N), 2.0), ntuple(_ -> 1, N)
                    ),
                )
                if obj === sin
                    out = Mooncake.value_and_derivative!!(cache, args...)
                    @test Mooncake.primal(out) ≈ 2sin(2.0)
                    @test Mooncake.tangent(out, 1) ≈ 2cos(2.0)
                else
                    @test_throws r"supplied tangents do not share it" Mooncake.value_and_derivative!!(
                        cache, args...
                    )
                end
            end
        end
    end
    # A wrapped capture can share derivative storage with another argument.
    a = [3.0]
    fw = FW(x -> a[1] * x)
    dfw, da = Mooncake._zero_tangents((fw, a))
    @test Mooncake.tangent_dim((dfw, da)) == 1
    @test Mooncake._check_shared_input_tangents((fw, a), (dfw, da)) === nothing
    @test_throws ArgumentError Mooncake._check_shared_input_tangents(
        (fw, a), (dfw, copy(da))
    )
    @testset "shared capture" begin
        call_shared(w, a) = w(a[1])
        rule = Mooncake.build_frule(call_shared, fw, a)
        for factory in (
            x -> Mooncake.zero_lifted(Val(1), x),
            x -> Mooncake.lift(x, Mooncake.zero_tangent(x)),
        )
            seed = Mooncake.basis_lifted!!(factory((call_shared, fw, a)), (1,))
            _, tw, ta = Mooncake.tangent(seed, 1)
            @test only(values(tw.dobj_ref[].fields)) === ta
            args = map(
                (p, v) -> Mooncake.Lifted{typeof(p),1}(p, v),
                Mooncake.primal(seed),
                Mooncake.tangent(seed),
            )
            output = rule(args...)
            @test (Mooncake.primal(output), Mooncake.tangent(output, 1)) == (9.0, 6.0)
        end
    end
    # Sweep the callable parameter and argument to form the Jacobian [x a].
    fw = FW(WrapperGain(3.0))
    for N in (1, 2)
        rule = Mooncake.build_frule(call_wrapper, fw, 0.5; chunk_size=N)
        seed = Mooncake.zero_lifted(Val(N), (call_wrapper, fw, 0.5))
        for first_slot in 1:N:2
            slots = ntuple(k -> first_slot + k - 1, N)
            seed = Mooncake.basis_lifted!!(seed, slots)
            args = map(
                (p, v) -> Mooncake.Lifted{typeof(p),N}(p, v),
                Mooncake.primal(seed),
                Mooncake.tangent(seed),
            )
            output = rule(args...)
            @test Mooncake.primal(output) == 1.5
            @test ntuple(k -> Mooncake.tangent(output, k), N) ==
                ntuple(k -> (0.5, 3.0)[slots[k]], N)
        end
    end
end
