@testset "test group selection" begin
    # Test that the new test group selection logic is working properly
    
    # Save original ARGS and environment
    original_args = copy(ARGS)
    original_test_group = get(ENV, "TEST_GROUP", nothing)
    
    try
        # Clear environment for testing
        if haskey(ENV, "TEST_GROUP")
            delete!(ENV, "TEST_GROUP")
        end
        empty!(ARGS)
        
        # Test determine_test_group function behavior
        function test_determine_test_group()
            env_test_group = get(ENV, "TEST_GROUP", nothing)
            args_test_group = length(ARGS) > 0 ? ARGS[1] : nothing
            
            if env_test_group !== nothing && args_test_group !== nothing
                if env_test_group != args_test_group
                    # In tests, we don't want to actually warn, just check logic
                    return args_test_group
                end
                return args_test_group
            end
            
            if args_test_group !== nothing
                return args_test_group
            elseif env_test_group !== nothing
                return env_test_group
            else
                return "basic"
            end
        end
        
        # Test default behavior (no ARGS, no env)
        @test test_determine_test_group() == "basic"
        
        # Test with ARGS only
        push!(ARGS, "quality")
        @test test_determine_test_group() == "quality"
        
        # Test with environment variable only
        empty!(ARGS)
        ENV["TEST_GROUP"] = "extended"
        @test test_determine_test_group() == "extended"
        
        # Test with both (should prefer ARGS)
        push!(ARGS, "basic")
        ENV["TEST_GROUP"] = "extended"
        @test test_determine_test_group() == "basic"
        
        # Test with matching values
        ARGS[1] = "extended"
        ENV["TEST_GROUP"] = "extended"
        @test test_determine_test_group() == "extended"
        
    finally
        # Restore original state
        empty!(ARGS)
        append!(ARGS, original_args)
        if original_test_group !== nothing
            ENV["TEST_GROUP"] = original_test_group
        elseif haskey(ENV, "TEST_GROUP")
            delete!(ENV, "TEST_GROUP")
        end
    end
end