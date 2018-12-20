import Base.write, Base.read

struct Layer{T<:Number}
    W::Matrix{T}
    b::Vector{T}
    f::Function
end

Layer(W, b) = Layer(W, b, identity)

Layer(in::Integer, out::Integer, f::Function) = Layer(randn(out, in), randn(out), f)

function (l::Layer)(x::AbstractArray)
    return l.W*x .+ l.b
end

struct Chain
    layers::Vector{<:Layer}
    function Chain(xs)
        for i in 1:length(xs)-1
            size(xs[i].W, 1) != size(xs[i+1].W, 2) && throw(ArgumentError("The former layers's out must equal to the latter's in."))
        end
        new(xs)
    end
end

Chain(xs...) = Chain([xs...])

Chain(ns::Integer...; f::Function=σ) = Chain([Layer(ns[i], ns[i+1], f) for i in 1:length(ns)-1])

Chain(c::Chain, xs...) = Chain(c.layers..., xs...)

Chain(c1::Chain, c2::Chain) = Chain(c1.layers..., c2.layers...)

function (c::Chain)(x::AbstractArray)
    for l in c.layers
        x = l.f.(l(x))
    end
    return x
end

function write(io::IO, c::Chain)
    write(io, UInt8(length(c.layers)))
    write(io, UInt16(size(c.layers[1].W, 2)))
    for l in c.layers
        write(io, UInt16(length(l.b)))
    end
    for l in c.layers
        write(io, vec(l.W))
        write(io, l.b)
    end
end


getlayersize(io::IO, n::Integer) = [read(io, UInt16) for _ in 1:n+1]

function getlayer(io::IO, lin::Integer, lout::Integer, f::Function=σ)
    w = zeros(lout, lin)
    b = zeros(lout)
    for i in 1:lin, j in 1:lout
        w[j, i] = read(io, Float64)
    end
    for i in 1:lout
       b[i] = read(io, Float64)
    end
    Layer(w, b, f)
end

function read(io::IO, ::Type{Chain}; f::Function=σ)
    n = read(io, UInt8)
    lsize = getlayersize(io, n)
    Chain([getlayer(io, lsize[i], lsize[i+1], f) for i in 1:n])
end

σ(x) = 1.0/(1.0+exp(-x))
dσ(x) = σ(x) * (1 - σ(x))
