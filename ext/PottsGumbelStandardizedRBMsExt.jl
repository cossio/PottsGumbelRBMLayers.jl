module PottsGumbelStandardizedRBMsExt

import StandardizedRestrictedBoltzmannMachines
import PottsGumbelRBMLayers
using StandardizedRestrictedBoltzmannMachines: standardize
using StandardizedRestrictedBoltzmannMachines: unstandardize
using StandardizedRestrictedBoltzmannMachines: StandardizedRBM
using PottsGumbelRBMLayers: PottsGumbel
using PottsGumbelRBMLayers: potts_to_gumbel
using PottsGumbelRBMLayers: gumbel_to_potts

function PottsGumbelRBMLayers.potts_to_gumbel(rbm::StandardizedRBM)
    visible = potts_to_gumbel(rbm.visible)
    hidden = potts_to_gumbel(rbm.hidden)
    return StandardizedRBM(visible, hidden, rbm.w, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
end

function PottsGumbelRBMLayers.gumbel_to_potts(rbm::StandardizedRBM)
    visible = gumbel_to_potts(rbm.visible)
    hidden = gumbel_to_potts(rbm.hidden)
    return StandardizedRBM(visible, hidden, rbm.w, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
end

end
