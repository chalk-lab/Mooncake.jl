using Mooncake
using DynamicPPL
using BenchmarkTools

model = DynamicPPL.TestUtils.DEMO_MODELS[1]
ldf = DynamicPPL.LogDensityFunction(model)
zt = Mooncake.zero_tangent(ldf)

function set_to_zero_iddict!!(x)
    c = IdDict{Any,Bool}()
    return Mooncake.set_to_zero_internal!!(c, x)
end

function set_to_zero_set!!(x)
    c = Set{UInt}()
    return Mooncake.set_to_zero_internal!!(c, x)
end

function set_to_zero_vector!!(x)
    c = Vector{UInt}()
    return Mooncake.set_to_zero_internal!!(c, x)
end

function set_to_zero_nocache!!(x)
    c = Mooncake.NoCache()
    return Mooncake.set_to_zero_internal!!(c, x)
end

t_iddict = @benchmark set_to_zero_iddict!!($zt)
t_set = @benchmark set_to_zero_set!!($zt)
t_vector = @benchmark set_to_zero_vector!!($zt)
t_nocache = @benchmark set_to_zero_nocache!!($zt)

iddict_time = median(t_iddict.times)
set_time = median(t_set.times)
vector_time = median(t_vector.times)
nocache_time = median(t_nocache.times)

println("\nIdDict{Any,Bool}: $(round(iddict_time, digits=1)) ns")
println("Set{UInt}:        $(round(set_time, digits=1)) ns ($(round(iddict_time/set_time, digits=2))x faster)")
println("Vector{UInt}:     $(round(vector_time, digits=1)) ns ($(round(iddict_time/vector_time, digits=2))x faster)")
println("NoCache:          $(round(nocache_time, digits=1)) ns ($(round(iddict_time/nocache_time, digits=2))x faster, unsafe)")
