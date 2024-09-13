function alm2map(d_alm, d_map, nthreads::Integer)

    alm2map!(d_alm, d_map, nthreads=nthreads)
    MPI.Barrier(d_map.info.comm)

    return d_map
end

#=function alm2map(d_alm, nside::Integer, comm::MPI.Comm, nthreads::Integer; root=0)

    if MPI.Comm_rank(comm) == root
        h_map = HealpixMap{Float64, RingOrder}(nside)
    else
        h_map = nothing
    end
    d_map = DMap{RR}(comm)
    HealpixMPI.Scatter!(h_map, d_map)

    MPI.Barrier(comm)
    alm2map!(d_alm, d_map, nthreads=nthreads)
    MPI.Barrier(comm)

    return d_map
end=#

function adjoint_alm2map(d_map, d_alm, nthreads::Integer)

    adjoint_alm2map!(d_map, d_alm, nthreads=nthreads)
    MPI.Barrier(d_map.info.comm)

    return d_alm
end

@adjoint function alm2map(d_alm, d_map, nthreads::Integer)
    p = alm2map(d_alm, d_map, nthreads)
    function alm2map_PB(adj_p)
        adj_a = deepcopy(d_alm)
        adjoint_alm2map!(adj_p, adj_a, nthreads=nthreads)
        return (2*adj_a, nothing, nothing)
    end
    return p, alm2map_PB
end

@adjoint function HealpixMPI.DMap{S,T}(pixels::Matrix{T}, info::GeomInfoMPI) where {S<:Strategy, T<:Real}
    map = DMap{S,T}(pixels, info)
    function DMap_PB(adj_map)
        adj_pix = adj_map.pixels
        adj_info = nothing
        return (adj_pix, adj_info)
    end
    return map, DMap_PB
end

#=function Gather(d_map, h_map)
    HealpixMPI.Gather!(d_map, h_map)
    return h_map
end
function ChainRulesCore.rrule(::typeof(Gather), d_map, h_map)
    m = Gather(d_map, h_map)
    project_dmap = ChainRulesCore.ProjectTo(d_map)
    function Gather_PB(adj_m)
        adj_dm = DAlm{RR}(d_map.info.comm)
        MPI.Scatter!(adj_m, adj_dm, d_map.info.comm)
        return ChainRulesCore.NoTangent(), project_dmap(adj_dm), ChainRulesCore.NoTangent()
    end
    return m, Gather_PB
end

@adjoint function HealpixMPI.DMap{S,T}(pixels::Matrix{T}, info::GeomInfoMPI) where {S<:Strategy, T<:Real}
    map = DMap{S,T}(pixels, info)
    function DMap_PB(adj_map)
        adj_pix = adj_map.pixels
        adj_info = nothing
        return (adj_pix, adj_info)
    end
    return map, DMap_PB
end=#
#=
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

function f(d_alm, nside, NCORE, comm, r)

    d_map = alm2map(d_alm, nside, nthreads=NCORE, comm=comm)
    s = sum2_dmap(d_map)
    return s
end

Random.seed!(1123)

NCORE = 4
NSIDE = 64
lmax = 128

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0

if crank == root
    h_alm = Alm(lmax, lmax, randn(ComplexF64, numberOfAlms(lmax)))
    h_map = HealpixMap{Float64, RingOrder}(NSIDE)
else
    h_alm = nothing
    h_map = nothing
end

d_alm = DAlm{RR}(comm)
MPI.Scatter!(h_alm, d_alm, comm)

MPI.Barrier(comm)

d_s = f(d_alm, NSIDE, NCORE, comm, root)

if crank == root

    map = Healpix.alm2map(h_alm, NSIDE)
    h_s = sum(map.pixels .^ 2)
    println("HEALPix: ")
    println(h_s)

    println("HEALPixMPI: ")
    println(d_s)
end

df = gradient(x->f(x, NSIDE, NCORE, comm, root), d_alm)
if crank == root
    AD_grad = Alm(lmax, lmax, randn(ComplexF64, numberOfAlms(lmax)))
else
    AD_grad = nothing
end
MPI.Gather!(df[1], AD_grad)

if crank == root
    h_eps = Alm(lmax, lmax, zeros(ComplexF64, numberOfAlms(lmax)))
    h_eps.alm[1000] += 0.01
else
    h_eps = nothing
end
d_eps = DAlm{RR}(comm)
MPI.Scatter!(h_eps, d_eps, comm)
FD_grad = (f(d_alm+d_eps, NSIDE, NCORE, comm, root) - f(d_alm-d_eps, NSIDE, NCORE, comm, root))/(0.02)

if crank == root
    println("FD: ", FD_grad, "\n")
    println("AD: ", AD_grad.alm[1000])
end

#println(df[1].alm[1])

MPI.Finalize()=#