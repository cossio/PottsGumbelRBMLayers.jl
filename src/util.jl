"""
    potts_to_gumbel(rbm)

Converts Potts layers to PottsGumbel layers.
"""
function potts_to_gumbel(rbm::RBM)
    visible = potts_to_gumbel(rbm.visible)
    hidden = potts_to_gumbel(rbm.hidden)
    return RBM(visible, hidden, rbm.w)
end

"""
    gumbel_to_potts(rbm)

Converts PottsGumbel layers to Potts layers.
"""
function gumbel_to_potts(rbm::RBM)
    visible = gumbel_to_potts(rbm.visible)
    hidden = gumbel_to_potts(rbm.hidden)
    return RBM(visible, hidden, rbm.w)
end

function potts_to_gumbel(layer::AbstractLayer)
    if layer isa Potts
        return PottsGumbel(layer)
    else
        return layer
    end
end

function gumbel_to_potts(layer::AbstractLayer)
    if layer isa PottsGumbel
        return Potts(layer)
    else
        return layer
    end
end
