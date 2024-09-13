using Distributions
using LinearAlgebra
using Healpix
using HealpixMPI
using MPI
using Distributed
using Plots
using Random
using ProgressMeter
using BenchmarkTools
using CSV
using DataFrames
using Test
using Ducc0
using .Ducc0.Nufft: u2nu!, nu2u!, make_plan, delete_plan!, u2nu_planned, nu2u_planned

n_threads = Threads.nthreads()
println("Using $n_threads threads")

Random.seed!(1123)

nside = 512
lmax = 2*nside - 1
mmax = lmax

function f(alm, geom_info, nthreads)
    map = ducc_alm2map_spin0(alm, geom_info, nthreads=nthreads)
    return sum(map[:,1].^2)
end

#=alm = Alm(lmax, mmax, rand(Complex{Float64}, numberOfAlms(lmax)))
t = @belapsed f(alm, nside)
println("$t seconds")=#

struct SHT_GeomInfo

    nside::Int
    lmax::Int

    mval::StridedArray{Csize_t,1}
    mstart::StridedArray{Cptrdiff_t,1}
    theta::StridedArray{Cdouble,1}
    nphi::StridedArray{Csize_t,1}
    phi0::StridedArray{Cdouble,1}
    rstart::StridedArray{Csize_t,1}

    function SHT_GeomInfo(nside::Int, lmax::Int)
        
        hmap = Healpix.HealpixMap{Float64, Healpix.RingOrder}(nside)
        rings = [r for r in 1:Healpix.numOfRings(hmap.resolution)]
        theta = Vector{Float64}(undef, length(rings)) #colatitude of every ring
        phi0 = Vector{Cdouble}(undef, length(rings))  #longitude of the first pixel of every ring
        nphi = Vector{Csize_t}(undef, length(rings))  #num of pixels in every ring
        rstart = Vector{Csize_t}(undef, length(rings)) #index of the first pixel in every ring
        rinfo = Healpix.RingInfo(0, 0, 0, 0, 0)               #initialize ring info object
        Threads.@threads for ri in 1:length(rings)
            Healpix.getringinfo!(hmap.resolution, ri, rinfo)  #fill it
            theta[ri] = rinfo.colatitude_rad            #use it to get the necessary data
            phi0[ri] = Healpix.pix2ang(hmap, rinfo.firstPixIdx)[2]
            nphi[ri] = rinfo.numOfPixels
            rstart[ri] = rinfo.firstPixIdx
        end

        mval = Vector{Csize_t}(undef, lmax+1)
        mstart = Vector{Cptrdiff_t}(undef, lmax+1)
        ofss = 1
        for i in 1:lmax+1
            mval[i] = Csize_t(i-1)
            mstart[i] = Cptrdiff_t(ofss)
            ofss += lmax+1-i-1
        end

        new(nside, lmax, mval, mstart, theta, nphi, phi0, rstart)

    end
end

function ducc_alm2map_spin0(alm, geom_info; nthreads=0)

    leg_in = Ducc0.Sht.alm2leg(alm, 0, geom_info.lmax, geom_info.mval, geom_info.mstart, 1, geom_info.theta, nthreads)
    map = Ducc0.Sht.leg2map(leg_in, geom_info.nphi, geom_info.phi0, geom_info.rstart, 1, nthreads)

    return map
end

info = SHT_GeomInfo(nside, lmax)
alms = rand(Complex{Float64}, numberOfAlms(lmax), 1)

t = @belapsed f(alms, info, n_threads)
println("$t seconds")