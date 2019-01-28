module MNIST_GPU

using CuArrays

include("getdata.jl")

export trainimages, testimages,
       trainlabels, testlabels,
       traindata,   testdata

function getimage(io::IO)
    img = Array{Int}(undef, NCOLS, NROWS)
    for i in 1:NCOLS, j in 1:NROWS
        img[i, j] = read(io, UInt8)
    end
    return CuArray(img)
end

function getlabel(io::IO)
    i = read(io, UInt8)
    v = zeros(10)
    v[i+1] = 1.
    return CuArray(v)
end

end  # module MNIST_GPU
