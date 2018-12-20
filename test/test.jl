include("../src/NeuralNetwork.jl")
include("../src/MNIST/src/MNIST.jl")

using .NeuralNetwork
using .MNIST

trd = traindata()
tsd = testdata()

net = read(joinpath(@__DIR__, "net1"), Chain)

test(net, tsd)