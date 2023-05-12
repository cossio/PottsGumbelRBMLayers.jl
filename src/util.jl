function _rbm_gumbel(rbm::RBM)
    vis = rbm.visible isa Potts ? PottsGumbel(rbm.visible) : rbm.visible
    hid = rbm.hidden isa Potts ? PottsGumbel(rbm.hidden) : rbm.hidden
    return RBM(vis, hid, rbm.w)
end

function _rbm_potts(rbm::RBM)
    vis = rbm.visible isa PottsGumbel ? Potts(rbm.visible) : rbm.visible
    hid = rbm.hidden isa PottsGumbel ? Potts(rbm.hidden) : rbm.hidden
    return RBM(vis, hid, rbm.w)
end
