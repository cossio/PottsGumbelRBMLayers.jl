module PottsGumbelRBMLayers

using Random: randexp
using RestrictedBoltzmannMachines: RBM, AbstractLayer, Potts,
    cgfs, mean_from_inputs, std_from_inputs, var_from_inputs, mode_from_inputs, meanvar_from_inputs,
    onehot_encode, grad2ave, vstack, energies, energy, ∂cgfs, ∂energy_from_moments, moments_from_samples,
    zerosum, zerosum!, zerosum_weights, ∂RBM
using EllipsisNotation: (..)

import RestrictedBoltzmannMachines
import RestrictedBoltzmannMachinesHDF5
import CudaRBMs

include("potts_gumbel.jl")
include("gumbel.jl")
include("util.jl")
include("zerosum.jl")
include("cuda.jl")
include("shift_fields.jl")
include("io.jl")

end
