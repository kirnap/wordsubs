# To be able to test gpu memory usage
using Knet

include("../bilstm.jl")
include("../infst.jl")
#atype = Array{Float64}
const atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})
#const datafile = "/Users/okirnap/bilstm-in-Knet8/data/ukwac/ukwac.txt"
#const vocabfile = "/Users/okirnap/bilstm-in-Knet8/data/ukwac/ukwac_vocab_100k"
#global stream_f = open(datafile)


function randseq(V,B,T)
    s = Vector{Vector{Int}}()
    #slen = rand(50:T)
    slen = T
    for t in 1:slen
        push!(s, rand(2:V,B))
    end
    return s
end


function test_load()
    batchsize = 25;slen = 30;vocabsize = 20004;hiddens = [300];embedding = 256;
    data = randseq(vocabsize, batchsize, slen)
    m = initmodel(atype, hiddens, embedding, vocabsize)
    s = initstate(atype, hiddens, batchsize)
    opts = oparams(m, Adam; gclip=5.0)
    #gradcheck(bilstm, m, s, data; gcheck=15, verbose=true, atol=0.01)
    if !isinteractive()
        for i=1:5
            #@time gloss = bilstmgrad(m, s, data)
            #bilstm(m, s, data)
            @time train(m, s, data, opts)
        end
    end
    return (m, s, data)
end
!isinteractive() && test_load()

# function main()
#     # regular steps for creating train data
#     vocabulary = create_vocab(vocabfile)
#     sdict = Dict{Int64, Array{Any, 1}}();
#     res = nextbatch(stream_f, sdict, vocabulary, 20)
#     global index_to_word = Array(AbstractString, length(vocabulary))
#     for (k, v) in vocabulary; index_to_word[v] = k;end;
#     return res
# end

# Here are some notes on gpu usage:
# Configuration sequence length : vocabsize : batchsize
# 25 : 100001 : 20 -> OutOfMemory
# 50 :  50001 : 20 -> OutOfMemory
# 

