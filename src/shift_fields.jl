RestrictedBoltzmannMachines.shift_fields(l::PottsGumbel, a::AbstractArray) = PottsGumbel(; θ = l.θ .+ a)

function RestrictedBoltzmannMachines.shift_fields!(l::PottsGumbel, a::AbstractArray)
    l.θ .+= a
    return l
end
