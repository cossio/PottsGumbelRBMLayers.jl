module PottsGumbelCUDAExt

import CudaRBMs
using CudaRBMs: cpu, gpu
using PottsGumbelRBMLayers: PottsGumbel

CudaRBMs.gpu(layer::PottsGumbel) = PottsGumbel(CudaRBMs.gpu(layer.par))
CudaRBMs.cpu(layer::PottsGumbel) = PottsGumbel(CudaRBMs.cpu(layer.par))

end
