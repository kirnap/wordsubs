using Knet
include("../bilstm.jl")
function test_convert()
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32}) 
    batchsize = 25; hiddens = [280]; embedding = 256; vocabsize = 10000
    m = initmodel(atype, hiddens, embedding, vocabsize)
    mco = convertmodel(m)
    for k in keys(mco)
        @assert(mco[k] == m[k])
    end
    info("Conversion tast passed")
end
!isinteractive() && test_convert()
