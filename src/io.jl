RestrictedBoltzmannMachinesHDF5.layer_type(::PottsGumbel) = "PottsGumbel"
RestrictedBoltzmannMachinesHDF5.construct_layer(::Val{:PottsGumbel}, par::AbstractArray) = PottsGumbel(par)
