using PottsGumbelRBMLayers: gumbel_to_potts
using PottsGumbelRBMLayers: potts_to_gumbel
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: Potts
using RestrictedBoltzmannMachines: RBM
using StandardizedRestrictedBoltzmannMachines: standardize
using Test: @test
using Test: @testset

@testset "StandardizedRestrictedBoltzmannMachines" begin
    hidden = Binary(; θ = randn(100))
    visible = Potts(; θ = randn(5, 108))
    rbm = standardize(RBM(visible, hidden, randn(5, 108, 100)))
    @test potts_to_gumbel(rbm).visible.θ == visible.θ
    @test gumbel_to_potts(rbm).visible.θ == visible.θ
    @test gumbel_to_potts(potts_to_gumbel(rbm)).visible.θ == visible.θ
end
