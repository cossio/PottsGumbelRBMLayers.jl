using Test: @test, @testset
using RestrictedBoltzmannMachines: Potts, xReLU, RBM, zerosum!, zerosum, free_energy, sample_from_inputs
using PottsGumbelRBMLayers: PottsGumbel, potts_to_gumbel
using Statistics: mean

@testset "zerosum (visible Potts)" begin
    hidden = xReLU(; θ = randn(100), γ = rand(100), Δ = rand(100), ξ = rand(100))
    visible = Potts(; θ = randn(5, 108))
    rbm = RBM(visible, hidden, randn(5, 108, 100))
    rbm = potts_to_gumbel(rbm)
    rbm1 = zerosum(rbm)

    v = sample_from_inputs(rbm.visible, zeros(5, 108, 1000))
    F0 = free_energy(rbm, v)
    F1 = free_energy(rbm1, v)
    zerosum!(rbm)
    F2 = free_energy(rbm, v)
    @test all(F0 - F2 .≈ mean(F0 - F2))
    @test F1 ≈ F2
end
