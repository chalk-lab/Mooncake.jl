using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Random, Distributions, LinearAlgebra, Test
using Bijectors
using Bijectors: partition, combine, PartitionMask
using Mooncake, ADTypes
import DifferentiationInterface as DI
using Flux

# just define a MLP 
function mlp3(
    input_dim::Int,
    hidden_dims::Int,
    output_dim::Int;
    activation=Flux.leakyrelu,
    paramtype::Type{T}=Float64,
) where {T<:AbstractFloat}
    m = Chain(
        Flux.Dense(input_dim, hidden_dims, activation),
        Flux.Dense(hidden_dims, hidden_dims, activation),
        Flux.Dense(hidden_dims, output_dim),
    )
    return Flux._paramtype(paramtype, m)
end

inputdim = 4
mask_idx = 1:2:inputdim
# creat a masking layer
mask = PartitionMask(inputdim, mask_idx)
cdim = length(mask_idx)

x = randn(inputdim)

t_net = mlp3(cdim, 16, cdim; paramtype=Float64)
ps, st = Optimisers.destructure(t_net)

function loss(ps, st, x, mask)
    t_net = st(ps)
    x₁, x₂, x₃ = partition(mask, x)
    y₁ = x₁ .+ t_net(x₂)
    y = combine(mask, y₁, x₂, x₃)
    # println("y = ", y)
    return sum(abs2, y)
end

loss(ps, st, x, mask)  # return 3.0167880799441793

DI.value_and_gradient(
    loss,
    ADTypes.AutoMooncake(; config=Mooncake.Config()),
    ps,
    DI.Cache(st),
    DI.Constant(x),
    DI.Constant(mask),
)

struct ACL
    mask::Bijectors.PartitionMask
    t::Flux.Chain
end
Flux.@functor ACL (t,)

acl = ACL(mask, t_net)
psacl, stacl = Optimisers.destructure(acl)

function loss_acl(ps, st, x)
    acl = st(ps)
    t_net = acl.t
    mask = acl.mask
    x₁, x₂, x₃ = partition(mask, x)
    y₁ = x₁ .+ t_net(x₂)
    y = combine(mask, y₁, x₂, x₃)
    return sum(abs2, y)
end
loss_acl(psacl, stacl, x) # return 3.0167880799441793

DI.value_and_gradient(
    loss_acl,
    ADTypes.AutoMooncake(; config=Mooncake.Config()),
    psacl,
    DI.Cache(stacl),
    DI.Constant(x),
)
