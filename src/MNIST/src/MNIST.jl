module MNIST
export trainimages, testimages,
       trainlabels, testlabels

const IMAGEOFFSET = 16
const LABELOFFSET = 8

const NROWS = 28
const NCOLS = 28

const TRAINIMAGES = joinpath(@__DIR__, "..", "data", "train-images.idx3-ubyte")
const TRAINLABELS = joinpath(@__DIR__, "..", "data", "train-labels.idx1-ubyte")
const TESTIMAGES  = joinpath(@__DIR__, "..", "data", "t10k-images.idx3-ubyte")
const TESTLABELS  = joinpath(@__DIR__, "..", "data", "t10k-labels.idx1-ubyte")

function imageheader(io::IO)
    magic_number = bswap(read(io, UInt32))
    total_items = bswap(read(io, UInt32))
    nrows = bswap(read(io, UInt32))
    ncols = bswap(read(io, UInt32))
    return magic_number, Int(total_items), Int(nrows), Int(ncols)
end

function labelheader(io::IO)
    magic_number = bswap(read(io, UInt32))
    total_items = bswap(read(io, UInt32))
    return magic_number, Int(total_items)
end

function getimage(io::IO)
    img = Array{UInt8}(undef, NCOLS, NROWS)
    for i in 1:NCOLS, j in 1:NROWS
        img[i, j] = read(io, UInt8)
    end
    return img
end

function getimage(io::IO, index::Integer)
    seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
    return getimage(io)
end

getlabel(io::IO) = Int(read(io, UInt8))

function getlabel(io::IO, index::Integer)
    seek(io, LABELOFFSET + (index - 1))
    return getlabel(io)
end

function trainimages(i::Integer)
    getimage(IOBuffer(read(TRAINIMAGES)), i)
end

function testimages(i::Integer)
    getimage(IOBuffer(read(TESTIMAGES)), i)
end

function trainlabels(i::Integer)
    getlabel(IOBuffer(read(TRAINLABELS)), i)
end

function testlabels(i::Integer)
    getlabel(IOBuffer(read(TESTLABELS)), i)
end

function trainimages()
    io = IOBuffer(read(TRAINIMAGES))
    _, nimages, nrows, ncols = imageheader(io)
    [getimage(io) for _ in 1:nimages]
end

function testimages()
    io = IOBuffer(read(TESTIMAGES))
    _, nimages, nrows, ncols = imageheader(io)
    [getimage(io) for _ in 1:nimages]
end

function trainlabels()
    io = IOBuffer(read(TRAINLABELS))
    _, nlabels = labelheader(io)
    [getlabel(io) for _ in 1:nlabels]
end

function testlabels()
    io = IOBuffer(read(TESTLABELS))
    _, nlabels = labelheader(io)
    [getlabel(io) for _ in 1:nlabels]
end

end # module