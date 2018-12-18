abstract type Layer end

struct Chain
    layers::Vector{<:Layer}
    function Chain(xs...)
        for i in 1:length(xs)-1
            size(xs[i].W, 1) != size(xs[i+1].W, 2) && throw(ArgumentError("The former layers's out must equal to the latter's in."))
        end
        new([xs...])
    end
end

function (c::Chain)(x::AbstractArray)
    for l in c.layers
        x = l(x)
    end
    x
end

struct LinearLayer{T<:Number} <: Layer
    W::Matrix{T}
    b::Vector{T}
    F::Function
end

LinearLayer(W, b) = LinearLayer(W, b, identity)

LinearLayer(in::Integer, out::Integer, F::Function) = LinearLayer(randn(out, in), randn(out), F)

(l::LinearLayer)(x::AbstractArray) = l.F.(l.W*x .+ l.b)
