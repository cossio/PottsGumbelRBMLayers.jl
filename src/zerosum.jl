RestrictedBoltzmannMachines.zerosum(rbm::RBM{<:PottsGumbel,<:PottsGumbel}) = potts_to_gumbel(zerosum(potts_to_gumbel(rbm)))

function RestrictedBoltzmannMachines.zerosum(rbm::RBM{<:PottsGumbel,<:AbstractLayer})
    _rbm = zerosum(potts_to_gumbel(rbm))
    return RBM(PottsGumbel(_rbm.visible), _rbm.hidden, _rbm.w)
end

function RestrictedBoltzmannMachines.zerosum(rbm::RBM{<:AbstractLayer,<:PottsGumbel})
    _rbm = zerosum(potts_to_gumbel(rbm))
    return RBM(_rbm.visible, PottsGumbel(_rbm.hidden), _rbm.w)
end

function RestrictedBoltzmannMachines.zerosum!(rbm::RBM{<:PottsGumbel, <:PottsGumbel})
    potts_to_gumbel(zerosum!(potts_to_gumbel(rbm)))
end

function RestrictedBoltzmannMachines.zerosum!(rbm::RBM{<:PottsGumbel, <:AbstractLayer})
    _rbm = zerosum!(potts_to_gumbel(rbm))
    return RBM(PottsGumbel(_rbm.visible), _rbm.hidden, _rbm.w)
end

function RestrictedBoltzmannMachines.zerosum!(rbm::RBM{<:AbstractLayer, <:PottsGumbel})
    _rbm = zerosum!(potts_to_gumbel(rbm))
    return RBM(_rbm.visible, PottsGumbel(_rbm.hidden), _rbm.w)
end

RestrictedBoltzmannMachines.zerosum!(∂::∂RBM, rbm::RBM{<:PottsGumbel,<:PottsGumbel}) = zerosum!(∂, potts_to_gumbel(rbm))
RestrictedBoltzmannMachines.zerosum!(∂::∂RBM, rbm::RBM{<:AbstractLayer,<:PottsGumbel}) = zerosum!(∂, potts_to_gumbel(rbm))
RestrictedBoltzmannMachines.zerosum!(∂::∂RBM, rbm::RBM{<:PottsGumbel,<:AbstractLayer}) = zerosum!(∂, potts_to_gumbel(rbm))

RestrictedBoltzmannMachines.zerosum_weights(weights::AbstractArray, rbm::RBM{<:PottsGumbel,<:PottsGumbel}) = zerosum_weights(weights, potts_to_gumbel(rbm))
RestrictedBoltzmannMachines.zerosum_weights(weights::AbstractArray, rbm::RBM{<:AbstractLayer,<:PottsGumbel}) = zerosum_weights(weights, potts_to_gumbel(rbm))
RestrictedBoltzmannMachines.zerosum_weights(weights::AbstractArray, rbm::RBM{<:PottsGumbel,<:AbstractLayer}) = zerosum_weights(weights, potts_to_gumbel(rbm))
