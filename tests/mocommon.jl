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
function dropout(x, p)
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
