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
include("MODELS.jl")

Random.seed!(1123)

#   RESOLUTION PARAMETERS
nside = 8
lmax = 16
ncore = 4


#   REALIZATION MAP
realiz_Cl, realiz_HAlm, realiz_HMap = Realization("Capse_fiducial_Dl.csv", nside, lmax)
realiz_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([realiz_HAlm], lmax, 1), lmax, 1), Cl2Kl(realiz_Cl))

#   GENERATED DATA MEASUREMENTS
#   Noise
ϵ=100
N = ϵ*ones(nside2npix(nside))
#   Data Map
gen_Cl, gen_HAlm, gen_HMap = Measurement(realiz_HMap, N, nside, lmax)
gen_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([gen_HAlm], lmax, 1), lmax, 1), Cl2Kl(gen_Cl))
invN_HMap = HealpixMap{Float64,RingOrder}(1 ./ N)

start_Cl, start_HAlm, start_HMap = StartingPoint(gen_Cl, nside)
start_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([start_HAlm], lmax, 1), lmax, 1), Cl2Kl(start_Cl))

L = length(start_θ)

helper_HMap = deepcopy(start_HMap)

nlp = nℓπ(start_θ)
nlp_grad = nℓπ_grad(start_θ)

println("OK")

#=function MCHMCℓπ(θ)
    return nℓπ(θ) #ricorda -1
end

function MCHMCℓπ_grad(x)
    f, df = nℓπ(x), nℓπ_grad(x)   #ricorda -1
    return f, df[1]
end

target = CustomTarget(MCHMCℓπ, MCHMCℓπ_grad, start_θ)

n_adapts, n_steps = 1000, 1000
spl = MicroCanonicalHMC.MCHMC(n_adapts, 0.001, integrator="LF", adaptive=true, tune_eps=true, tune_L=false, eps=10.0, tune_sigma=false, L=130., sigma=ones(L))

samples_MCHMC = Sample(spl, target, n_steps, init_params=start_θ, dialog=true, include_latent=false)#, thinning=10)
=#