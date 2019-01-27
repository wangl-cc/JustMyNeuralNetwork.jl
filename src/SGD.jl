using Random
using ForwardDiff
using LinearAlgebra

const ∇ = ForwardDiff.gradient

function DSGD!(net::Chain, c::Function, traindata::AbstractArray, testdata::AbstractArray, epochs::Integer, minibatchsize::Integer; η_max::Real=50)
    t = test(net, testdata)
    n = length(traindata)
    η = η_max
    for i in 1:epochs
        minibatchs = [shuffle!(traindata)[k-minibatchsize+1:k] for k in minibatchsize:minibatchsize:n]
        for minibatch in minibatchs
            GD!(net, c, minibatch, η=η)
        end
        nt = test(net, testdata)
        println(nt)
        if nt <= t
            η /= 2
        end
        t = nt
    end
    η
end

function test(net::Chain, testdata::AbstractArray)
    count = 0
    for xy in testdata
        py = net(xy[1])
        findmax(xy[2])[2] == findmax(py)[2] && (count+=1)
    end
    return count/length(testdata)
end

function SGD!(net::Chain, c::Function, traindata::AbstractArray, epochs::Integer, minibatchsize::Integer; η::Real=1)
    n = length(traindata)
    for i in 1:epochs
        minibatchs = [shuffle!(traindata)[k-minibatchsize+1:k] for k in minibatchsize:minibatchsize:n]
        for minibatch in minibatchs
            GD!(net, c, minibatch, η=η)
        end
    end
end

function GD!(net::Chain, c::Function, traindata::AbstractArray, epochs::Integer; η::Real=1)
    for i in 1:epochs
        GD!(net, c, traindata, η=η)
    end
end

function GD!(net::Chain, c::Function, traindata::AbstractArray; η::Real=1)
    dW = [zeros(size(l.W)) for l in net.layers]
    db = [zeros(size(l.b)) for l in net.layers]
    for xy in traindata
        ∂b, ∂W = backprop(net, c, xy...)
        for i in eachindex(dW)
            dW[i] += ∂W[i]
            db[i] += ∂b[i]
        end
    end
    for i in eachindex(dW)
        net.layers[i].W .-= dW[i]*η/length(traindata)
        net.layers[i].b .-= db[i]*η/length(traindata)
    end
end

function backprop(net::Chain, c::Function, x::Vector{R}, y::Vector{R})where R<:Real
    a = x
    as = Vector{Vector{R}}()
    zs = Vector{Vector{R}}()
    push!(as, deepcopy(a))
    for l in net.layers
        z = l(a)
        a = l.f.(z)
        push!(zs, z)
        push!(as, deepcopy(a))
    end
    δs = similar(zs)
    δs[end] = ∇(a->cost(c, a, y), as[end]) .* dσ.(zs[end])
    for i in length(net.layers)-1
        δs[end-i] = transpose(net.layers[end-i+1].W) * δs[end-i+1] .* dσ.(zs[end-i])
    end
    ∂W = Vector{Matrix{R}}()
    for k in eachindex(δs)
        push!(∂W, zeros(length(δs[k]), length(as[k])))
        for i in eachindex(δs[k]), j in eachindex(as[k])
            ∂W[k][i, j] = δs[k][i] * as[k][j]
        end
    end
    return δs, ∂W # δs equals to δs
end

function cost(c::Function, a::Vector{R}, y::Vector{R})where R<:Real
    return sum(c, y.-a)/length(a)
end
