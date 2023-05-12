# PottsGumbelRBMLayers.jl

[![Build Status](https://github.com/cossio/PottsGumbelRBMLayers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cossio/PottsGumbelRBMLayers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cossio/PottsGumbelRBMLayers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cossio/PottsGumbelRBMLayers.jl)

Implements a `PottsGumbel` RBM layer, which is equivalent to `Potts` but uses the Gumbel trick to sample from the categorical distribution. This is GPU friendly.