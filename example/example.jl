import CUDA

using PottsGumbelRBMLayers: PottsGumbel
using CudaRBMs: gpu, cpu
using Random: bitrand
using Statistics: mean
using RestrictedBoltzmannMachines: RBM, Potts, Binary, sample_v_from_v, sample_v_from_h,
    mean_v_from_h, moments_from_samples, zerosum

CUDA.allowscalar(false)

rbm = RBM(PottsGumbel((3, 5)), Binary((2,)), zeros(3,5,2))
rbm = gpu(rbm)

zerosum(rbm)

h = gpu(bitrand(2, 10))
v = sample_v_from_h(rbm, h)
v = sample_v_from_v(rbm, v)

mean_v_from_h(rbm, h)


A = gpu(rand(3, 5, 2))
mean(@views A[1, :, :]; dims=2)
sum(@views A[1, :, :]; dims=2)
mean(A[1, :, :]; dims=2)
sum(A[1, :, :]; dims=2)


A .- sum(A; dims = 1)
A .- mean(A; dims=1)

mean()
