module NeuralNetwork

include("network.jl")
include("SGD.jl")

export Chain, Layer, DSGD!, SGD!, GD!, test

end # module