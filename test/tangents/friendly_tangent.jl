@testset "friendly_tangent" begin

    # Symmetric/Hermitian/SymTridiagonal only store part of the matrix, so a single stored
    # gradient number can need to be shown at two positions in the dense result. Both go
    # through arrayify (src/rules/blas.jl), which mirrors the stored value into both
    # positions, matching the standard convention for gradients of a symmetric matrix.
    @testset "Symmetric uplo=$uplo" for (uplo, tx_data) in
                                        ((:U, [1.0 6.0; 0.0 1.0]), (:L, [1.0 0.0; 6.0 1.0]))
        S = LinearAlgebra.Symmetric([1.0 2.0; 999.0 4.0], uplo)
        tx = Mooncake.build_tangent(typeof(S), tx_data, NoTangent())
        cache = Mooncake.friendly_tangent_cache(S)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, S, tx) === dest
        expected = Matrix(LinearAlgebra.Symmetric(tx_data, uplo))
        @test dest == expected
        @test Mooncake.tangent_to_friendly!!(S, tx) == expected
    end

    @testset "Hermitian uplo=$uplo" for (uplo, tx_data) in
                                        ((:U, [2.0 6.0; 0.0 2.0]), (:L, [2.0 0.0; 6.0 2.0]))
        H = LinearAlgebra.Hermitian([1.0 2.0; 999.0 4.0], uplo)
        tx = Mooncake.build_tangent(typeof(H), tx_data, NoTangent())
        cache = Mooncake.friendly_tangent_cache(H)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, H, tx) === dest
        expected = Matrix(LinearAlgebra.Hermitian(tx_data, uplo))
        @test dest == expected
        @test Mooncake.tangent_to_friendly!!(H, tx) == expected
    end

    # A genuinely complex Hermitian, chosen so the mirroring conjugates and the diagonal
    # gets forced real, the same way Hermitian's own indexing treats a primal value. The
    # diagonal entries are given a nonzero imaginary part on purpose, to check that it
    # really gets dropped rather than accidentally passed through.
    @testset "Hermitian (complex) conjugates and keeps the diagonal real" begin
        tx_data = ComplexF64[
            (1.0+0.5im) (2.0+3.0im)
            0.0 (4.0-0.25im)
        ]
        H = LinearAlgebra.Hermitian(randn(ComplexF64, 2, 2), :U)
        tx = Mooncake.build_tangent(typeof(H), tx_data, NoTangent())
        dest = Mooncake.friendly_tangent_cache(H).buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, H, tx) === dest
        @test dest[1, 2] == tx_data[1, 2]
        @test dest[2, 1] == conj(tx_data[1, 2])
        @test dest[1, 1] == real(tx_data[1, 1])
        @test dest[2, 2] == real(tx_data[2, 2])
        @test Mooncake.tangent_to_friendly!!(H, tx) == dest
    end

    @testset "SymTridiagonal" begin
        # dv maps straight to the diagonal. ev is the shared off-diagonal band, so each
        # entry gets mirrored into both (i, i+1) and (i+1, i); everywhere else is 0.
        ST = LinearAlgebra.SymTridiagonal([1.0, 2.0, 3.0], [4.0, 5.0])
        tx = Mooncake.build_tangent(typeof(ST), [1.0, 2.0, 3.0], [6.0, 8.0])
        cache = Mooncake.friendly_tangent_cache(ST)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, ST, tx) === dest
        @test dest[1, 1] == 1.0 && dest[2, 2] == 2.0 && dest[3, 3] == 3.0
        @test dest[1, 2] == 6.0 && dest[2, 1] == 6.0
        @test dest[2, 3] == 8.0 && dest[3, 2] == 8.0
        @test dest[1, 3] == 0.0 && dest[3, 1] == 0.0
        @test Mooncake.tangent_to_friendly!!(ST, tx) == dest
    end

    # Adjoint/Transpose aren't lossy either, just for a different reason: every entry
    # maps to exactly one parent entry, just transposed. The friendly tangent is a plain
    # Matrix{T} shaped like the wrapper itself.
    @testset "Adjoint" begin
        A = LinearAlgebra.Adjoint(randn(3, 2))
        tx_parent = randn(3, 2)  # tangent of the parent field, so matches A.parent's shape
        tx = Mooncake.build_tangent(typeof(A), tx_parent)
        cache = Mooncake.friendly_tangent_cache(A)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test size(dest) == size(A)
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        @test dest == tx_parent'
        @test Mooncake.tangent_to_friendly!!(A, tx) == tx_parent'
    end

    @testset "Transpose" begin
        A = LinearAlgebra.Transpose(randn(3, 2))
        tx_parent = randn(3, 2)  # tangent of the parent field, so matches A.parent's shape
        tx = Mooncake.build_tangent(typeof(A), tx_parent)
        cache = Mooncake.friendly_tangent_cache(A)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test size(dest) == size(A)
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        @test dest == permutedims(tx_parent)
        @test Mooncake.tangent_to_friendly!!(A, tx) == permutedims(tx_parent)
    end

    # Adjoint/Transpose of a Vector give a (1, N) row shape; the parent's tangent is then
    # a Vector too, so the broadcast maps a 1-d tangent into a 2-d buffer.
    @testset "Adjoint of a Vector" begin
        A = LinearAlgebra.Adjoint(randn(3))
        tx_parent = randn(3)
        tx = Mooncake.build_tangent(typeof(A), tx_parent)
        cache = Mooncake.friendly_tangent_cache(A)
        dest = cache.buffer
        @test size(dest) == size(A) == (1, 3)
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        @test dest == tx_parent'
        @test Mooncake.tangent_to_friendly!!(A, tx) == tx_parent'
    end

    @testset "Transpose of a Vector" begin
        A = LinearAlgebra.Transpose(randn(3))
        tx_parent = randn(3)
        tx = Mooncake.build_tangent(typeof(A), tx_parent)
        cache = Mooncake.friendly_tangent_cache(A)
        dest = cache.buffer
        @test size(dest) == size(A) == (1, 3)
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        @test dest == permutedims(tx_parent)
        @test Mooncake.tangent_to_friendly!!(A, tx) == permutedims(tx_parent)
    end

    # Complex correctness checks for Adjoint/Transpose.
    @testset "Adjoint (complex)" begin
        A = LinearAlgebra.Adjoint(randn(ComplexF64, 3, 2))
        tx_parent = randn(ComplexF64, 3, 2)
        tx = Mooncake.build_tangent(typeof(A), tx_parent)
        dest = Mooncake.friendly_tangent_cache(A).buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        @test dest == tx_parent'
        @test Mooncake.tangent_to_friendly!!(A, tx) == tx_parent'
    end

    @testset "Transpose (complex)" begin
        A = LinearAlgebra.Transpose(randn(ComplexF64, 3, 2))
        tx_parent = randn(ComplexF64, 3, 2)
        tx = Mooncake.build_tangent(typeof(A), tx_parent)
        dest = Mooncake.friendly_tangent_cache(A).buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        @test dest == permutedims(tx_parent)
        @test Mooncake.tangent_to_friendly!!(A, tx) == permutedims(tx_parent)
    end

    # Nesting Adjoint/Transpose over a plain array is still isometric (composing two
    # isometric relabellings is still one), so _arrayify_roundtrip_safe recurses through
    # the nesting rather than stopping at the first non-array-tangent-typed parent.
    @testset "nested Adjoint(Transpose(...))" begin
        B = randn(3, 3)
        T = LinearAlgebra.Transpose(B)
        A = LinearAlgebra.Adjoint(T)
        tx_B = randn(3, 3)
        tx_T = Mooncake.build_tangent(typeof(T), tx_B)
        tx = Mooncake.build_tangent(typeof(A), tx_T)
        cache = Mooncake.friendly_tangent_cache(A)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        # Adjoint(Transpose(B)) == B for real B, so the tangent map is the identity too.
        @test dest == tx_B
        @test Mooncake.tangent_to_friendly!!(A, tx) == tx_B
    end

    @testset "nested Transpose(Adjoint(...)) (complex)" begin
        B = randn(ComplexF64, 3, 3)
        Ad = LinearAlgebra.Adjoint(B)
        A = LinearAlgebra.Transpose(Ad)
        tx_B = randn(ComplexF64, 3, 3)
        tx_Ad = Mooncake.build_tangent(typeof(Ad), tx_B)
        tx = Mooncake.build_tangent(typeof(A), tx_Ad)
        cache = Mooncake.friendly_tangent_cache(A)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        # Transpose(Adjoint(B))[i,j] == conj(B[i,j]), so the tangent map conjugates too.
        @test dest == conj.(tx_B)
        @test Mooncake.tangent_to_friendly!!(A, tx) == conj.(tx_B)
    end

    # Isometry check: the friendly presentation must preserve the tangent inner product,
    # so gradients read through it pair correctly with perturbations. Note this cannot
    # catch a missing conj for Adjoint (the error cancels between fx and fy inside the
    # real inner product); the complex value testsets above pin conjugation down.
    @testset "Adjoint/Transpose pairing" begin
        for T in (Float64, ComplexF64),
            W in (LinearAlgebra.Adjoint, LinearAlgebra.Transpose)

            A = W(randn(T, 3, 2))
            tx = Mooncake.build_tangent(typeof(A), randn(T, 3, 2))
            ty = Mooncake.build_tangent(typeof(A), randn(T, 3, 2))
            fx = Mooncake.tangent_to_friendly!!(A, tx)
            fy = Mooncake.tangent_to_friendly!!(A, ty)
            friendly_dot = sum(real.(conj.(fx) .* fy))
            @test Mooncake._dot(tx, ty) ≈ friendly_dot
        end
    end

    @testset "Adjoint of degenerate-length Vectors" begin
        # Boundary conditions for the (1, N) row-shape broadcast path: N=0 and N=1.
        for v in (Float64[], [1.0])
            A = LinearAlgebra.Adjoint(v)
            tx = Mooncake.build_tangent(typeof(A), copy(v))
            dest = Mooncake.friendly_tangent_cache(A).buffer
            @test size(dest) == size(A)
            @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
            @test dest == v'
        end
    end

    # A SubArray with no repeated indices has no aliasing between logical positions and
    # parent storage, so it's exactly as safe as a plain Adjoint/Transpose of an array.
    # Uses a Vector index (not just Colon/UnitRange) to exercise the allunique check, not
    # just the trivially-unique range/colon cases.
    @testset "Adjoint of a SubArray with unique indices" begin
        P = view(randn(4, 3), [1, 3], :)
        A = LinearAlgebra.Adjoint(P)
        tx_parent_data = randn(4, 3)
        # The indices field's own tangent_type isn't just NoTangent here: a Vector index
        # (unlike a Colon/UnitRange one) has a per-element tangent, so it's a Tuple of
        # (Vector{NoTangent}, NoTangent) rather than a single NoTangent for the whole field.
        tx_indices = (fill(NoTangent(), length(P.indices[1])), NoTangent())
        tx_P = Mooncake.build_tangent(
            typeof(P), tx_parent_data, tx_indices, NoTangent(), NoTangent()
        )
        tx = Mooncake.build_tangent(typeof(A), tx_P)
        cache = Mooncake.friendly_tangent_cache(A)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test size(dest) == size(A)
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        expected = (tx_parent_data[P.indices...])'
        @test dest == expected
        @test Mooncake.tangent_to_friendly!!(A, tx) == expected
    end

    # Diagonal's off-diagonal implicit value is 0, which is the correct tangent there (an
    # off-diagonal perturbation isn't a valid direction in Diagonal's parameter space), so
    # this isn't actually lossy the way a triangular wrapper's implicit diagonal is.
    @testset "Transpose of a Diagonal (raw constructor)" begin
        v = randn(3)
        D = LinearAlgebra.Diagonal(v)
        A = LinearAlgebra.Transpose(D)
        tx_diag_data = randn(3)
        tx_D = Mooncake.build_tangent(typeof(D), tx_diag_data)
        tx = Mooncake.build_tangent(typeof(A), tx_D)
        cache = Mooncake.friendly_tangent_cache(A)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        @test dest == LinearAlgebra.Diagonal(tx_diag_data)
        @test Mooncake.tangent_to_friendly!!(A, tx) == LinearAlgebra.Diagonal(tx_diag_data)
    end

    # Symmetric/Hermitian/SymTridiagonal mirror correctly under nesting too: arrayify's
    # recursion doesn't care whether the parent is the top-level differentiated value or
    # sits underneath a Transpose/Adjoint. `expected` is computed by composing the same
    # operations the code under test applies (mirror, then transpose/adjoint), rather than
    # a hand-derived value, so it stays correct regardless of exactly how they interact.
    @testset "Transpose of a Symmetric (raw constructor)" begin
        # transpose(::Symmetric)/adjoint(::Symmetric) simplify to the Symmetric itself in
        # Base, so the raw constructor is needed to get an actual Transpose{...,<:Symmetric}.
        S = LinearAlgebra.Symmetric(randn(3, 3))
        A = LinearAlgebra.Transpose(S)
        tx_S_data = randn(3, 3)
        tx_S = Mooncake.build_tangent(typeof(S), tx_S_data, NoTangent())
        tx = Mooncake.build_tangent(typeof(A), tx_S)
        cache = Mooncake.friendly_tangent_cache(A)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        expected = Matrix(transpose(LinearAlgebra.Symmetric(tx_S_data, Symbol(S.uplo))))
        @test dest == expected
        @test Mooncake.tangent_to_friendly!!(A, tx) == expected
    end

    @testset "Adjoint of a Hermitian (complex, raw constructor)" begin
        # adjoint(::Hermitian) simplifies to the Hermitian itself in Base, so the raw
        # constructor is needed to get an actual Adjoint{...,<:Hermitian}.
        H = LinearAlgebra.Hermitian(randn(ComplexF64, 3, 3), :U)
        A = LinearAlgebra.Adjoint(H)
        tx_H_data = randn(ComplexF64, 3, 3)
        tx_H = Mooncake.build_tangent(typeof(H), tx_H_data, NoTangent())
        tx = Mooncake.build_tangent(typeof(A), tx_H)
        cache = Mooncake.friendly_tangent_cache(A)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        expected = Matrix(adjoint(LinearAlgebra.Hermitian(tx_H_data, Symbol(H.uplo))))
        @test dest == expected
        @test Mooncake.tangent_to_friendly!!(A, tx) == expected
    end

    @testset "Transpose of a SymTridiagonal (raw constructor)" begin
        ST = LinearAlgebra.SymTridiagonal([1.0, 2.0, 3.0], [6.0, 8.0])
        A = LinearAlgebra.Transpose(ST)
        tx_ST = Mooncake.build_tangent(typeof(ST), [1.0, 2.0, 3.0], [6.0, 8.0])
        tx = Mooncake.build_tangent(typeof(A), tx_ST)
        cache = Mooncake.friendly_tangent_cache(A)
        @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        dest = cache.buffer
        @test Mooncake.tangent_to_friendly_internal!!(dest, A, tx) === dest
        expected = Matrix(transpose(LinearAlgebra.SymTridiagonal([1.0, 2.0, 3.0], [6.0, 8.0])))
        @test dest == expected
        @test Mooncake.tangent_to_friendly!!(A, tx) == expected
    end

    @testset "structural parents fall back to AsRaw" begin
        # arrayify (src/rules/blas.jl) can present these parents' tangents as logical
        # arrays, but that presentation may drop/duplicate stored coordinates (a repeated
        # index view) or inject an implicit constant that isn't really part of the tangent
        # (a unit-triangular diagonal). Reading a reverse cotangent through such a
        # presentation isn't generally valid, so these stay AsRaw rather than reusing the
        # Adjoint/Transpose AsCustomised fast path. Nesting Adjoint/Transpose around one of
        # these lossy parents doesn't fix the underlying issue, so it stays AsRaw too
        # ("nested wrapper over a lossy parent" below); nesting around a plain array, a
        # unique-index view, a Diagonal, or a Symmetric/Hermitian/SymTridiagonal is safe
        # instead, and is covered by the testsets above.
        struct MyWeirdArrayForFriendlyTangentTest{T} <: AbstractMatrix{T}
            data::Matrix{T}
        end
        Base.size(x::MyWeirdArrayForFriendlyTangentTest) = size(x.data)

        B = randn(3, 3)
        unique_view = view(B, [1, 2], :)
        repeated_view = view(B, [1, 1], :)
        @test typeof(unique_view) === typeof(repeated_view)

        parents = (
            "view with repeated indices" => repeated_view,
            "unit triangular" => LinearAlgebra.UnitUpperTriangular(randn(3, 3)),
            "nested wrapper over a lossy parent" =>
                LinearAlgebra.Transpose(LinearAlgebra.UnitUpperTriangular(randn(3, 3))),
            "unsupported custom array" => MyWeirdArrayForFriendlyTangentTest(randn(3, 3)),
        )

        @testset "$label / $W" for (label, parent) in parents,
            W in (LinearAlgebra.Adjoint, LinearAlgebra.Transpose)

            A = W(parent)
            cache = Mooncake.friendly_tangent_cache(A)
            @test cache isa Mooncake.FriendlyTangentCache{Mooncake.AsRaw}
            tx = Mooncake.zero_tangent(A)
            @test Mooncake.tangent_to_friendly!!(cache, A, tx, Mooncake.NoCache()) === tx
        end

        # A genuinely unrecognised array type (arrayify has no method beyond its generic,
        # unconditionally-throwing fallback) should still safely fall back to AsRaw, but
        # with a @debug hint pointing at what's missing, unlike a known but structurally
        # lossy type such as UnitUpperTriangular, which logs nothing (arrayify does support
        # it, it's just not safe to present densely, so there's nothing to add).
        @testset "unsupported parent logs a debug hint" begin
            w = MyWeirdArrayForFriendlyTangentTest(randn(3, 3))
            A = LinearAlgebra.Transpose(w)
            @test_logs(
                (:debug, r"could not verify that `arrayify` supports"),
                min_level = Logging.Debug,
                Mooncake.friendly_tangent_cache(A),
            )

            Tri = LinearAlgebra.UnitUpperTriangular(randn(3, 3))
            A_tri = LinearAlgebra.Transpose(Tri)
            @test_logs(min_level = Logging.Debug, Mooncake.friendly_tangent_cache(A_tri))
        end
    end

    @testset "tangent_to_friendly!! routing" begin
        s = 3.0
        S = LinearAlgebra.Symmetric([1.0 2.0; 999.0 4.0])
        tx_S_data = [1.0 6.0; 0.0 1.0]
        tx_S = Mooncake.build_tangent(typeof(S), tx_S_data, NoTangent())
        expected_S = Matrix(LinearAlgebra.Symmetric(tx_S_data, Symbol(S.uplo)))

        # scalar Float64: FriendlyTangentCache{AsRaw} leaf (tangent_type==P), returns tangent directly
        @test Mooncake.tangent_to_friendly!!(s, 7.0) === 7.0

        # AsCustomised cache: delegates to tangent_to_friendly_internal!!, writes in-place
        c_S = Mooncake.friendly_tangent_cache(S)
        result = Mooncake.tangent_to_friendly!!(c_S, S, tx_S, Mooncake.NoCache())
        @test result === c_S.buffer && result == expected_S

        # 2-arg form: equivalent to 4-arg with fresh cache
        @test Mooncake.tangent_to_friendly!!(S, tx_S) == expected_S
    end

    @testset "friendly_tangent_cache recursive struct" begin
        # Struct with a Symmetric field: friendly_tangent_cache should recurse and return
        # a NamedTuple with a FriendlyTangentCache{AsCustomised} for the matrix field.
        struct FooWithSym
            m::LinearAlgebra.Symmetric{Float64,Matrix{Float64}}
            v::Float64
        end
        foo = FooWithSym(LinearAlgebra.Symmetric([1.0 2.0; 3.0 4.0]), 3.14)
        d = Mooncake.friendly_tangent_cache(foo)
        @test d isa NamedTuple{(:m, :v)}
        @test d.m isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        @test d.v isa Mooncake.FriendlyTangentCache{Mooncake.AsRaw}

        # Tangent for FooWithSym
        tx_m_data = [0.5 1.0; 0.0 0.5]
        tx_m = Mooncake.build_tangent(typeof(foo.m), tx_m_data, NoTangent())
        tx_foo = Mooncake.Tangent((; m=tx_m, v=2.0))
        result = Mooncake.tangent_to_friendly!!(foo, tx_foo)
        @test result isa NamedTuple{(:m, :v)}
        @test result.m == Matrix(LinearAlgebra.Symmetric(tx_m_data, Symbol(foo.m.uplo)))
        @test result.v == 2.0
    end

    @testset "friendly_tangent_cache Tuple recursion" begin
        # Tuple dest is built element-wise; each element follows its own cache.
        t = (1.0, LinearAlgebra.Symmetric([1.0 2.0; 3.0 4.0]))
        d = Mooncake.friendly_tangent_cache(t)
        @test d isa Tuple
        @test d[1] isa Mooncake.FriendlyTangentCache{Mooncake.AsRaw}   # Float64: raw == primal
        @test d[2] isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}

        tx_t_data = [1.0 2.0; 0.0 1.0]
        tx_t = (7.0, Mooncake.build_tangent(typeof(t[2]), tx_t_data, NoTangent()))
        result = Mooncake.tangent_to_friendly!!(t, tx_t)
        @test result isa Tuple
        @test result[1] === 7.0
        @test result[2] == Matrix(LinearAlgebra.Symmetric(tx_t_data, Symbol(t[2].uplo)))
    end

    @testset "friendly_tangent_cache NamedTuple recursion" begin
        nt = (; m=LinearAlgebra.Symmetric([1.0 2.0; 3.0 4.0]), v=3.14)
        d = Mooncake.friendly_tangent_cache(nt)
        @test d isa NamedTuple{(:m, :v)}
        @test d.m isa Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}
        @test d.v isa Mooncake.FriendlyTangentCache{Mooncake.AsRaw}

        tx_m_data = [0.5 1.0; 0.0 0.5]
        tx_m = Mooncake.build_tangent(typeof(nt.m), tx_m_data, NoTangent())
        tx_nt = (; m=tx_m, v=2.0)
        result = Mooncake.tangent_to_friendly!!(nt, tx_nt)
        @test result isa NamedTuple{(:m, :v)}
        @test result.m == Matrix(LinearAlgebra.Symmetric(tx_m_data, Symbol(nt.m.uplo)))
        @test result.v == 2.0
    end

    @testset "friendly_tangent_cache mutable struct" begin
        # Mutable structs use AsMutableFields: cache is a sentinel at prepare time,
        # fields are unwrapped to a plain NamedTuple at tangent_to_friendly!! time.
        mutable struct MutFoo
            a::Float64
            b::Vector{Float64}
        end
        x = MutFoo(2.0, [1.0, 2.0])
        d = Mooncake.friendly_tangent_cache(x)
        @test d isa Mooncake.FriendlyTangentCache{Mooncake.AsMutableFields}

        tx = Mooncake.MutableTangent((;
            a=Mooncake.PossiblyUninitTangent(3.0),
            b=Mooncake.PossiblyUninitTangent([0.5, 1.5]),
        ))
        result = Mooncake.tangent_to_friendly!!(x, tx)
        @test result isa NamedTuple{(:a, :b)}
        @test result.a === 3.0
        @test result.b == [0.5, 1.5]
    end

    @testset "friendly_tangent_cache AbstractDict opt-in" begin
        # Dict is mutable: falls through to friendly_tangent_cache_internal via AbstractDict
        # override, returning AsPrimal.
        d_cache = Mooncake.friendly_tangent_cache(Dict("a" => 1.0))
        @test d_cache isa Mooncake.FriendlyTangentCache{Mooncake.AsPrimal}
    end

    @testset "friendly_tangent_cache Vector{Int} returns AsRaw" begin
        A = [1, 2]
        @test Mooncake.friendly_tangent_cache(A) isa
            Mooncake.FriendlyTangentCache{Mooncake.AsRaw}
    end

    @testset "friendly_tangent_cache Transpose{Int} returns AsRaw (#1149)" begin
        A = transpose([1, 2])
        @test Mooncake.friendly_tangent_cache(A) isa
            Mooncake.FriendlyTangentCache{Mooncake.AsRaw}
    end

    @testset "friendly_tangent_cache Adjoint{Int} returns AsRaw" begin
        A = adjoint([1, 2])
        @test Mooncake.friendly_tangent_cache(A) isa
            Mooncake.FriendlyTangentCache{Mooncake.AsRaw}
    end
end
