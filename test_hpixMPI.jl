using Distributions
using LinearAlgebra
using Healpix
using HealpixMPI
using MPI
using Distributed
using Plots
using Random
using BenchmarkTools
using Zygote
using Zygote: @adjoint

include("SHT_MPI.jl")

MPI.Init()

Random.seed!(1123)

nside = 1024
lmax = 2*nside - 1
mmax = lmax

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0
ncore = 2

if crank == root
    H_Map = HealpixMap{Float64, RingOrder}(nside)
    H_Alm = Alm(lmax, lmax, randn(ComplexF64, numberOfAlms(lmax)))
else
    H_Map = nothing
    H_Alm = nothing
end

D_Map = DMap{RR}(comm)
D_Alm = DAlm{RR}(comm)

MPI.Scatter!(H_Map, D_Map, clear=true)
MPI.Scatter!(H_Alm, D_Alm, clear=true)

function sum2_dmap(d_map)
    tot_map = d_map * d_map
    local_s = sum(tot_map.pixels[:,1])
    global_s = MPI.Allreduce(local_s, +, d_map.info.comm)
    return global_s
end
@adjoint function sum2_dmap(d_map)
    s = sum2_dmap(d_map)
    function sum2_dmap_PB(adj_s)
        adj_map = deepcopy(d_map) * (2*adj_s)
        return (adj_map,)
    end
    return s, sum2_dmap_PB
end

function f(alm, map, ncore)
    p = alm2map(alm, map, ncore)
    s = sum2_dmap(p)
    return s
end

withgradient(x->f(x, D_Map, ncore), D_Alm)

MPI.Barrier(comm)

t0 = time()
f(D_Alm, D_Map, ncore)
t1 = time() - t0
println("$t1 seconds")

MPI.Barrier(comm)

t2 = time()
withgradient(x->f(x, D_Map, ncore), D_Alm)
t3 = time() - t2
println("$t3 seconds")

MPI.Finalize()