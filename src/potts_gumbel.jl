# Like Potts, but uses Gumbel trick which is GPU-friendly
struct PottsGumbel{N,A} <: AbstractLayer{N}
    par::A
    function PottsGumbel{N,A}(par::A) where {N,A<:AbstractArray}
        @assert size(par, 1) == 1 # θ
        @assert ndims(par) == N + 1
        return new(par)
    end
end

PottsGumbel(par::AbstractArray) = PottsGumbel{ndims(par) - 1, typeof(par)}(par)

function PottsGumbel(; θ)
    par = vstack((θ,))
    return PottsGumbel(par)
end

PottsGumbel(::Type{T}, sz::Dims) where {T} = PottsGumbel(; θ = zeros(T, sz))
PottsGumbel(sz::Dims) = PottsGumbel(Float64, sz)

PottsGumbel(layer::Potts) = PottsGumbel(layer.par)
RestrictedBoltzmannMachines.Potts(layer::PottsGumbel) = Potts(layer.par)

RestrictedBoltzmannMachines.cgfs(layer::PottsGumbel, inputs = 0) = cgfs(Potts(layer), inputs)
RestrictedBoltzmannMachines.mean_from_inputs(layer::PottsGumbel, inputs = 0) = mean_from_inputs(Potts(layer), inputs)
RestrictedBoltzmannMachines.mean_abs_from_inputs(layer::PottsGumbel, inputs = 0) = mean_from_inputs(layer, inputs)
RestrictedBoltzmannMachines.std_from_inputs(layer::PottsGumbel, inputs = 0) = std_from_inputs(Potts(layer), inputs)
RestrictedBoltzmannMachines.mode_from_inputs(layer::PottsGumbel, inputs = 0) = mode_from_inputs(Potts(layer), inputs)
RestrictedBoltzmannMachines.var_from_inputs(layer::PottsGumbel, inputs = 0) = var_from_inputs(Potts(layer), inputs)
RestrictedBoltzmannMachines.meanvar_from_inputs(layer::PottsGumbel, inputs = 0) = meanvar_from_inputs(Potts(layer), inputs)

# This is the only change with respect to Potts. Here, we use the Gumbel trick.
function RestrictedBoltzmannMachines.sample_from_inputs(layer::PottsGumbel, inputs = 0)
    c = categorical_sample_from_logits_gumbel(layer.θ .+ inputs)
    return onehot_encode(c, 1:size(layer, 1))
end

CudaRBMs.gpu(layer::PottsGumbel) = PottsGumbel(CudaRBMs.gpu(layer.par))
CudaRBMs.cpu(layer::PottsGumbel) = PottsGumbel(CudaRBMs.cpu(layer.par))

RestrictedBoltzmannMachines.grad2ave(l::PottsGumbel, ∂::AbstractArray) = grad2ave(Potts(l), ∂)
