using Knet
include("../infst.jl")
include("../mocommon.jl")


function oldbilstm(model, states, sequence; pdrop=(0, 0))
    total = 0.0
    count = 0
    atype = typeof(AutoGrad.getval(model[:fembed]))

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    sf = copy(states)
    for i=1:length(sequence)-1
        input = convert(atype, sequence[i])
        x = input * model[:fembed]
        h = forward(model[:forw], sf, x)
        fhiddens[i+1] = copy(h)
    end
    fhiddens[1] = convert(atype, zeros(size(fhiddens[2])))

    # backward lstm
    bhiddens = Array(Any, length(sequence))
    sb = copy(states)
    for i=length(sequence):-1:2
        input = convert(atype, sequence[i])
        x = input * model[:bembed]
        bhiddens[i-1] = forward(model[:back], sb, x)
    end
    bhiddens[end] = convert(atype, zeros(size(bhiddens[2])))

    # concatenate layer
    for i=1:length(fhiddens)
        hf = dropout(fhiddens[i], pdrop[1])
        hb = dropout(bhiddens[i], pdrop[2])
        ypred = hcat(hf, hb) * model[:soft][1] .+ model[:soft][2]
        #ypred = hcat(fhiddens[i], bhiddens[i]) * model[:soft][1] .+ model[:soft][2]
        ynorm = logp(ypred, 2)
        ygold = convert(atype, sequence[i])
        count += size(ygold, 1)
        total += sum(ygold .* ynorm)
    end
    return - total / count
end


lg1 = grad(oldbilstm)


function bilstm(model, states, sequence; pdrop=(0, 0))
    total = 0.0
    count = 0
    atype = typeof(AutoGrad.getval(model[:fembed]))

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    sf = copy(states)
    for i=1:length(sequence)-1
        x = model[:fembed][sequence[i], :]
        h = forward(model[:forw], sf, x)
        fhiddens[i+1] = copy(h)
    end
    fhiddens[1] = convert(atype, zeros(size(fhiddens[2])))

    # backward lstm
    bhiddens = Array(Any, length(sequence))
    sb = copy(states)
    for i=length(sequence):-1:2
        x = model[:bembed][sequence[i], :]
        bhiddens[i-1] = forward(model[:back], sb, x)
    end
    bhiddens[end] = convert(atype, zeros(size(bhiddens[2])))

    # concatenate layer
    for i=1:length(fhiddens)
        hf = dropout(fhiddens[i], pdrop[1])
        hb = dropout(bhiddens[i], pdrop[2])
        ypred = hcat(hf, hb) * model[:soft][1] .+ model[:soft][2]
        #ypred = hcat(fhiddens[i], bhiddens[i]) * model[:soft][1] .+ model[:soft][2]
        total += logprob(sequence[i], ypred)
        count += length(sequence[i])
    end
    return - total / count
end


lg2 = grad(bilstm)


function testmodels()
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float64})
    batchsize = 1
    hiddens = [128]
    embedding = 128
    f = open("/mnt/kufs/scratch/okirnap/lexsub/data/testdata.txt")
    global t_vocab = create_vocab("/mnt/kufs/scratch/okirnap/lexsub/data/testvocab")
    sdict = Dict{Int64, Array{Any, 1}}();
    readstream!(f, sdict, t_vocab)
    index_to_word = Array(AbstractString, length(t_vocab))
    for (k, v) in t_vocab; index_to_word[v] = k; end;
   
    ids = nextbatch(f, sdict, t_vocab, batchsize) # ids of the sequences
    println(ibuild_sentence(index_to_word, ids , 1))

    one2sen(d) = map(i-> (find(x->x==true, i)), d)
    
    # create minibatch for old model
    data = [ falses(batchsize, length(index_to_word)) for i=1:length(ids)]
    for i=1:length(ids)
        index = ids[i]
        data[i][index] = true
    end
    sen = map(x->index_to_word[x], one2sen(data))

    srand(12)
    m = initmodel(atype, hiddens, embedding, length(index_to_word))
    s = initstate(atype, hiddens, batchsize)
    @assert(bilstm(m, s, ids) == oldbilstm(m, s, data))
    info("forward test passed")
    g1 = lg1(m, s, data)
    g2 = lg2(m, s, ids)
    for (k,v) in g2
        @assert(g1[k] == v)
    end
    info("backward test passed")
end
!isinteractive() && testmodels()
