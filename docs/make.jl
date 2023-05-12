using PottsGumbelRBMLayers
using Documenter

DocMeta.setdocmeta!(PottsGumbelRBMLayers, :DocTestSetup, :(using PottsGumbelRBMLayers); recursive=true)

makedocs(;
    modules=[PottsGumbelRBMLayers],
    authors="JFdCD <j.cossio.diaz@gmail.com>",
    repo="https://github.com/cossio/PottsGumbelRBMLayers.jl/blob/{commit}{path}#{line}",
    sitename="PottsGumbelRBMLayers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
