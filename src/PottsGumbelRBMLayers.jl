module PottsGumbelRBMLayers

using RestrictedBoltzmannMachines: AbstractLayer, Potts,
    cgfs, mean_from_inputs, std_from_inputs, var_from_inputs, mode_from_inputs, meanvar_from_inputs,
    onehot_encode, grad2ave, vstack, energies, energy, ∂cgfs, ∂energy_from_moments, moments_from_samples

import RestrictedBoltzmannMachines
import CudaRBMs

include("potts_gumbel.jl")
include("gumbel.jl")

end
