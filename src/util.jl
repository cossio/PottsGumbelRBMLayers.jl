"""
    potts_to_gumbel(rbm)

Converts Potts layers to PottsGumbel layers.
"""
function potts_to_gumbel(rbm::RBM)
    visible = potts_to_gumbel(rbm.visible)
    hidden = potts_to_gumbel(rbm.hidden)
    return RBM(visible, hidden, rbm.w)
end

potts_to_gumbel(layer::Potts) = PottsGumbel(layer)
potts_to_gumbel(layer::PottsGumbel) = layer

"""
    gumbel_to_potts(rbm)

Converts PottsGumbel layers to Potts layers.
"""
function gumbel_to_potts(rbm::RBM)
    visible = gumbel_to_potts(rbm.visible)
    hidden = gumbel_to_potts(rbm.hidden)
    return RBM(visible, hidden, rbm.w)
end

gumbel_to_potts(layer::Potts) = layer
gumbel_to_potts(layer::PottsGumbel) = Potts(layer)
