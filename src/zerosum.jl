RestrictedBoltzmannMachines.zerosum(rbm::RBM{<:PottsGumbel,<:PottsGumbel}) = _rbm_gumbel(zerosum(_rbm_potts(rbm)))

function RestrictedBoltzmannMachines.zerosum(rbm::RBM{<:PottsGumbel,<:AbstractLayer})
    _rbm = zerosum(_rbm_potts(rbm))
    return RBM(PottsGumbel(_rbm.visible), _rbm.hidden, _rbm.w)
end

function RestrictedBoltzmannMachines.zerosum(rbm::RBM{<:AbstractLayer,<:PottsGumbel})
    _rbm = zerosum(_rbm_potts(rbm))
    return RBM(_rbm.visible, PottsGumbel(_rbm.hidden), _rbm.w)
end

function RestrictedBoltzmannMachines.zerosum!(rbm::RBM{<:PottsGumbel, <:PottsGumbel})
    _rbm_gumbel(zerosum!(_rbm_potts(rbm)))
end

function RestrictedBoltzmannMachines.zerosum!(rbm::RBM{<:PottsGumbel, <:AbstractLayer})
    _rbm = zerosum!(_rbm_potts(rbm))
    return RBM(PottsGumbel(_rbm.visible), _rbm.hidden, _rbm.w)
end

function RestrictedBoltzmannMachines.zerosum!(rbm::RBM{<:AbstractLayer, <:PottsGumbel})
    _rbm = zerosum!(_rbm_potts(rbm))
    return RBM(_rbm.visible, PottsGumbel(_rbm.hidden), _rbm.w)
end

RestrictedBoltzmannMachines.zerosum!(∂::∂RBM, rbm::RBM{<:PottsGumbel,<:PottsGumbel}) = zerosum!(∂, _rbm_potts(rbm))
RestrictedBoltzmannMachines.zerosum!(∂::∂RBM, rbm::RBM{<:AbstractLayer,<:PottsGumbel}) = zerosum!(∂, _rbm_potts(rbm))
RestrictedBoltzmannMachines.zerosum!(∂::∂RBM, rbm::RBM{<:PottsGumbel,<:AbstractLayer}) = zerosum!(∂, _rbm_potts(rbm))

RestrictedBoltzmannMachines.zerosum_weights(weights::AbstractArray, rbm::RBM{<:PottsGumbel,<:PottsGumbel}) = zerosum_weights(weights, _rbm_potts(rbm))
RestrictedBoltzmannMachines.zerosum_weights(weights::AbstractArray, rbm::RBM{<:AbstractLayer,<:PottsGumbel}) = zerosum_weights(weights, _rbm_potts(rbm))
RestrictedBoltzmannMachines.zerosum_weights(weights::AbstractArray, rbm::RBM{<:PottsGumbel,<:AbstractLayer}) = zerosum_weights(weights, _rbm_potts(rbm))
