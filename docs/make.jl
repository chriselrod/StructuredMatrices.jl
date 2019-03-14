using Documenter, StructuredMatrices

makedocs(;
    modules=[StructuredMatrices],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/StructuredMatrices.jl/blob/{commit}{path}#L{line}",
    sitename="StructuredMatrices.jl",
    authors="Chris Elrod",
    assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/StructuredMatrices.jl",
)
