# For the purpose of infinite stream tests
include("../infst.jl")


function testptb()
    f = open("../data/ptb.train.txt")
    t_vocab = create_vocab("../data/ptb_3k_vocab")
    sdict = Dict{Int64, Array{Any, 1}}();
    batchsize = 10
    index_to_word = Array(AbstractString, length(t_vocab))
    for (k, v) in t_vocab; index_to_word[v] = k;end;
    res = nextbatch(f, sdict, t_vocab, batchsize)
    result = Any[]

    while (res != nothing)
        res = nextbatch(f, sdict, t_vocab, batchsize)
        (res != nothing) && (push!(result, res))
    end
    isinteractive() && (return result) || println("$(length(result)) many sequences in ptb")

end
!isinteractive() && testptb()
