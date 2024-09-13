function Realization(realiz_Cl_file, nside, lmax)

    realiz_Dl = CSV.read(realiz_Cl_file, DataFrame)[1:lmax-1,1]
    realiz_Cl = dl2cl(realiz_Dl, 2)
    realiz_Cl[1] += 1e-10
    realiz_Cl[2] += 1e-10

    realiz_HMap = synfast(realiz_Cl, nside)
    realiz_HAlm = map2alm(realiz_HMap, lmax=lmax)

    return realiz_Cl, realiz_HAlm, realiz_HMap
end

function Measurement(realiz_map, noise, nside, lmax)

    e = rand(MvNormal(zeros(nside2npix(nside)), Diagonal(noise)))

    gen_HMap = HealpixMap{Float64,RingOrder}(deepcopy(realiz_map) + e)
    gen_HAlm = map2alm(gen_HMap, lmax=lmax)
    gen_Cl = anafast(gen_HMap, lmax=lmax)

    return gen_Cl, gen_HAlm, gen_HMap
end

function StartingPoint(gen_Cl, nside)

    start_HAlm = synalm(gen_Cl)
    start_Cl = alm2cl(start_HAlm)
    start_HMap = Healpix.alm2map(start_HAlm, nside)

    return start_Cl, start_HAlm, start_HMap
end

