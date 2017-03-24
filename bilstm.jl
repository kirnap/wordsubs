# lstm weights initialization
# w[2k-1], w[2k] : weight and bias for kth layer respectively
function initweights(atype, hiddens, embedding, vocab, init=xavier)
    weights = Array(Any, 2length(hiddens))
    input = embedding
    for k = 1:length(hiddens)
        weights[2k-1] = init(input+hiddens[k], 4hiddens[k])
        weights[2k] = zeros(1, 4hiddens[k])
        weights[2k][1:hiddens[k]] = 1 # forget gate bias
        input = hiddens[k]
    end
    return map(w->convert(atype, w), weights)
end


# state initialization
# s[2k-1], s[2k] : hidden and cell respectively
function initstate(atype, hiddens, batchsize)
    state = Array(Any, 2length(hiddens))
    for k=1:length(hiddens)
        state[2k-1] = atype(zeros(batchsize, hiddens[k]))
        state[2k] = atype(zeros(batchsize, hiddens[k]))
    end
    return state
end


function initmodel(atype, hiddens, embedding, vocabsize, init=xavier)
    model = Dict{Symbol, Any}()
    model[:forw] = initweights(atype, hiddens, embedding, vocabsize, init)
    model[:back] = initweights(atype, hiddens, embedding, vocabsize, init)
    model[:fembed] = atype(init(vocabsize, embedding))
    model[:bembed] = atype(init(vocabsize, embedding))
    model[:soft] = [ atype(init(2hiddens[end], vocabsize)), atype(init(1, vocabsize)) ]
    return model
end


function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end


# multilayer lstm forward, returns the final hidden
function forward(weight, states, input)
    x = input
    for i=1:2:length(states)
        (states[i], states[i+1]) = lstm(weight[i], weight[i+1], states[i], states[i+1], x)
        x = states[i]
    end
    return x
end


# dropout layer
function m_dropout(x, p)
    if p > 0
        return x .* (rand!(similar(AutoGrad.getval(x))) .> p) * (1/(1-p))
    else
        return x
    end
end


function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = similar(output)
    @inbounds for i=1:length(output)
        index[i] = i + (output[i]-1)*nrows
    end
    o1 = logp(ypred,2)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end


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
        h  = forward(model[:back], sb, x)
        bhiddens[i-1] = copy(h)
    end
    bhiddens[end] = convert(atype, zeros(size(bhiddens[2])))

    # concatenate layer
    for i=1:length(fhiddens)
        hf = m_dropout(fhiddens[i], pdrop[1])
        hb = m_dropout(bhiddens[i], pdrop[2])
        ypred = hcat(hf, hb) * model[:soft][1] .+ model[:soft][2]
        total += logprob(sequence[i], ypred)
        count += length(sequence[i])
    end
    return - total / count
end


bilstmgrad = grad(bilstm)


oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)


function train(model, state, sequence, opts; pdrop=(0,0))
    gloss = bilstmgrad(model, state, sequence; pdrop=pdrop)
    update!(model, gloss, opts)
end


convertmodel{T<:Number}(x::KnetArray{T}) = convert(Array{T}, x)
convertmodel{T<:Number}(x::Array{T}) = convert(Array{T}, x)
convertmodel(a::Associative)=Dict(k=>convertmodel(v) for (k,v) in a)
convertmodel(a) = map(x->convertmodel(x), a)


function devperp(m, s, dev)
    devloss = 0
    for d in dev
        devloss += bilstm(m, s, d)
    end
    return exp(devloss / length(dev))
end
