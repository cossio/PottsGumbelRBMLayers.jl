using PottsGumbelRBMLayers: PottsGumbel
using CudaRBMs: gpu, cpu
using Random: bitrand
using RestrictedBoltzmannMachines: RBM, Potts, Binary, sample_v_from_v, sample_v_from_h,
    mean_v_from_h, moments_from_samples

rbm = RBM(PottsGumbel((3, 5)), Binary((2,)), zeros(3,5,2))
rbm = gpu(rbm)

h = gpu(bitrand(2, 10))
v = sample_v_from_h(rbm, h)
v = sample_v_from_v(rbm, v)

mean_v_from_h(rbm, h)
