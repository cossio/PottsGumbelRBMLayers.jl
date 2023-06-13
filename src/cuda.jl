CudaRBMs.gpu(layer::PottsGumbel) = PottsGumbel(CudaRBMs.gpu(layer.par))
CudaRBMs.cpu(layer::PottsGumbel) = PottsGumbel(CudaRBMs.cpu(layer.par))
