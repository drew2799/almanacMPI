function nℓπ(θ; data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, ncore=ncore, comm=comm, root=root)

    nlp = NegLogPosterior(θ, L, comm, ncore, helper_DMap, data=data, lmax=lmax, nside=nside, invN=invN, root = root)

    if crank == root
        return nlp
    end
end

function nℓπ_grad(θ; data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, ncore=ncore, comm=comm, root=root)

    nlp_grad = gradient(x->NegLogPosterior(x, L, comm, ncore, helper_DMap, data=data, lmax=lmax, nside=nside, invN=invN, root=root), θ)

    if crank == root
        return nlp_grad
    end
end