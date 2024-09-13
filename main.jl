using Distributions
using LinearAlgebra
using StatsFuns
using StatsBase

using LogDensityProblems
using LogDensityProblemsAD
using AbstractDifferentiation
using MCMCDiagnosticTools
using AdvancedHMC
using MicroCanonicalHMC
using Pathfinder
using MuseInference

using Healpix
using HealpixMPI
using MPI
using Distributed

using Plots
using StatsPlots
using LaTeXStrings

using Random
using ProgressMeter
using BenchmarkTools
using CSV
using DataFrames
using Test

using Zygote: @adjoint
using Zygote
using ChainRules.ChainRulesCore

include("SHT_MPI.jl")
include("DATA_GEN.jl")
include("REPARAM.jl")
include("UTILITIES.jl")
include("INFERENCE_FUNCS.jl")
include("MODELS2.jl")

Random.seed!(1123)

#   RESOLUTION PARAMETERS
nside = 1024
lmax = 2047

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0
ncore = 4

#   REALIZATION MAP
if crank == root
    realiz_Cl, realiz_HAlm, realiz_HMap = Realization("Capse_fiducial_Dl.csv", nside, lmax)
    realiz_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([realiz_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(realiz_Cl))
else
    realiz_Cl, realiz_HAlm, realiz_HMap = nothing, nothing, nothing
    realiz_θ = nothing
end

L = MPIvec_length(realiz_θ, comm, root=0)

#   GENERATED DATA MEASUREMENTS
#   Noise
ϵ=100
N = ϵ*ones(nside2npix(nside))
#   Data Map
if crank == root
    gen_Cl, gen_HAlm, gen_HMap = Measurement(realiz_HMap, N, nside, lmax)
    gen_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([gen_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(gen_Cl))
    invN_HMap = HealpixMap{Float64,RingOrder}(1 ./ N)
else
    gen_Cl, gen_HAlm, gen_HMap = nothing, nothing, nothing
    gen_θ = zeros(L)
    invN_HMap = nothing
end

#   STARTING POINT
if crank == root
    start_Cl, start_HAlm, start_HMap = StartingPoint(gen_Cl, nside)
    start_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([start_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(start_Cl))
else
    start_Cl, start_HAlm, start_HMap = nothing, nothing, nothing
    start_θ = zeros(L)
end

#   PROMOTE HEALPIX.MAP TO HEALIPIXMPI.DMAP
gen_DMap = DMap{RR}(comm)
invN_DMap = DMap{RR}(comm)
HealpixMPI.Scatter!(gen_HMap, gen_DMap, comm, clear=true)
HealpixMPI.Scatter!(invN_HMap, invN_DMap, comm, clear=true)

helper_DMap = deepcopy(gen_DMap)

nlp = nℓπ(start_θ, data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, ncore=ncore, comm=comm, root=root)
if crank == 0
    println("ok")
end
nlp_grad = nℓπ_grad(start_θ, data=gen_DMap,  helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, ncore=ncore, comm=comm, root=root)
if crank == 0
    println("ok")
end

MPI.Barrier(comm)

t0 = time()
nℓπ(start_θ, data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, ncore=ncore, comm=comm, root=root)
t1 = time() - t0
println("$t1 seconds")

MPI.Barrier(comm)

t2 = time()
nℓπ_grad(start_θ, data=gen_DMap,  helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, ncore=ncore, comm=comm, root=root)
t3 = time() - t2
println("$t3 seconds")

#=
function MCHMCℓπ(θ)
    return nℓπ(θ) #ricorda -1
end

function MCHMCℓπ_grad(x)
    f, df = nℓπ(x), nℓπ_grad(x)   #ricorda -1
    return f, df[1]
end

target = CustomTarget(MCHMCℓπ, MCHMCℓπ_grad, start_θ)

n_adapts, n_steps = 1000, 1000
spl = MicroCanonicalHMC.MCHMC(n_adapts, 0.001, integrator="LF", adaptive=true, tune_eps=true, tune_L=false, eps=10.0, tune_sigma=false, L=130., sigma=ones(L))

samples_MCHMC = Sample(spl, target, n_steps, init_params=start_θ, dialog=true, include_latent=false)#, thinning=10) =#

#=n_LF = 10
n_samples, n_adapts = 1_500, 500

metric = DiagEuclideanMetric(L)
ham = Hamiltonian(metric, nℓπ, nℓπ_grad)
initial_ϵ = 0.1
integrator = Leapfrog(initial_ϵ)

kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n_LF)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.75, integrator))

samples_HMC, stats_HMC = sample(ham, kernel, start_θ, n_samples, adaptor, n_adapts; progress=true, verbose=true) #drop_warmup = true
=#



#=function FD_derivat(x, L, i, ε, comm; root=0)
    if MPI.Comm_rank(comm) == root
        e = zeros(L)
        e[i] += ε
        up_x = x + e
        low_x = x - e
    else
        up_x = nothing
        low_x = nothing
    end
    FD_d = (NegLogPosterior(up_x, L, comm, ncore, data = gen_DMap, lmax=lmax, nside=nside, invN = invN_DMap, root = root)-
                    NegLogPosterior(low_x, L, comm, ncore, data = gen_DMap, lmax=lmax, nside=nside, invN = invN_DMap, root = root))/(2*ε)
    return FD_d
end
idx = [1, 2, 3, 4, 5, 6, L-5, L-4, L-3, L-2, L-1, L]
if crank == root
    trials = []
else
    trials = nothing
end
for i in idx
    FD_nlp_grad = FD_derivat(gen_θ, L, i, 0.001, comm, root=root)
    if crank == root
        if isapprox(FD_nlp_grad, AD_nlp_grad[1][i], rtol=0.01)
            push!(trials, 1)
        else
            push!(trials, 0)
        end
    end
end
if crank == root
    c = sum(trials)
    println("PASSED: $c over $(length(idx)) tests \n")
    println("NOT PASSED: $(length(idx)-c) over $(length(idx)) tests \n")
end=#
MPI.Finalize()