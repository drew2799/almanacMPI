function nℓπ(θ; data=gen_HMap, helper_HMap=helper_HMap, lmax=lmax, nside=nside, invN=invN_HMap, ncore=ncore)

    MPI.Init()
    
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    root = 0

    if crank == root
        Dθ = θ
        gen_data = data
        invN_map = invN
        helper = helper_HMap
        L = length(θ)
    else
        Dθ = nothing
        gen_data = nothing
        invN_map = nothing
        helper = nothing
        L = 0
    end

    gen_DMap = DMap{RR}(comm)
    helper_DMap = DMap{RR}(comm)
    invN_DMap = DMap{RR}(comm)
    HealpixMPI.Scatter!(gen_data, gen_DMap, comm)
    HealpixMPI.Scatter!(helper, helper_DMap, comm)
    HealpixMPI.Scatter!(invN_map, invN_DMap, comm)

    MPI.Barrier(comm)

    nlp = NegLogPosterior(Dθ, L, comm, ncore, helper_DMap, data=gen_DMap, lmax=lmax, nside=nside, invN=invN_DMap, root = root)

    MPI.Barrier(comm)

    MPI.Finalize()

    if crank == root
        return nlp
    end
end

function nℓπ_grad(θ; data=gen_HMap, helper_HMap=helper_HMap, lmax=lmax, nside=nside, invN=invN_HMap, ncore=ncore)

    MPI.Init()
    
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    root = 0

    if crank == root
        Dθ = θ
        gen_data = data
        invN_map = invN
        helper = helper_HMap
        L = length(θ)
    else
        Dθ = nothing
        gen_data = nothing
        invN_map = nothing
        helper = nothing
        L = 0
    end

    gen_DMap = DMap{RR}(comm)
    helper_DMap = DMap{RR}(comm)
    invN_DMap = DMap{RR}(comm)
    HealpixMPI.Scatter!(gen_data, gen_DMap, comm)
    HealpixMPI.Scatter!(helper, helper_DMap, comm)
    HealpixMPI.Scatter!(invN_map, invN_DMap, comm)

    MPI.Barrier(comm)

    nlp_grad = gradient(x->NegLogPosterior(x, L, comm, ncore, helper_DMap, data=gen_DMap, lmax=lmax, nside=nside, invN=invN_DMap, root=root), Dθ)

    MPI.Barrier(comm)

    MPI.Finalize()

    if crank == root
        return nlp_grad[1]
    end
end