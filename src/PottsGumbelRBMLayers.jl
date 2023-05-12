module PottsGumbelRBMLayers

using RestrictedBoltzmannMachines: AbstractLayer, Potts,
    cgfs, mean_from_inputs, std_from_inputs, var_from_inputs, mode_from_inputs,
    onehot_encode, grad2ave

import RestrictedBoltzmannMachines
import CudaRBMs

include("potts_gumbel.jl")
include("gumbel.jl")

end
