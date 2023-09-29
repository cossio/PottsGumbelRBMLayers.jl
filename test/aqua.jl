import Aqua
import PottsGumbelRBMLayers
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(PottsGumbelRBMLayers; ambiguities=false)
end
