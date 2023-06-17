# TODO: move this code to PottsGumbelRBMLayers.jl, after this package is registered

using Test: @testset, @test
using RestrictedBoltzmannMachines: RBM, Binary
using RestrictedBoltzmannMachinesHDF5: save_rbm, load_rbm
using PottsGumbelRBMLayers: PottsGumbel

@testset "I/O" begin
    rbm = RBM(PottsGumbel(; θ=randn(2,3)), Binary(; θ=randn(4)), randn(2,3,4))
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)

    @test loaded_rbm.visible isa PottsGumbel

    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θ == rbm.hidden.θ
end
