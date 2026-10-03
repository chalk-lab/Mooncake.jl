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
    # Storage validation must also exclude the primal metadata inside a forward slot.
    fw = FW(WrapperGain(3.0, which(sin, (Float64,))))
    @test_throws ArgumentError Mooncake._check_shared_input_tangents((
        Mooncake.zero_lifted(Val(1), fw), Mooncake.zero_lifted(Val(1), fw)
    ))
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
