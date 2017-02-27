# bilstm model implementation, forward and backward lstms have their own embeddings

# same abstraction with:
# https://github.com/denizyuret/Knet.jl/blob/master/examples/charlm.jl#L174
# 1:2*length(layerconfig) -> forward lstm parameters
# 2*length(layerconfig)+1:4*length(layerconfig) -> backward lstm parameters
# final 3 or 4 parameters are embedding and softmax layer parameters, based on embedding config
function initparams(atype, layerconfig, embedsize, vocab, winit)
    parameters = Array(Any, 4*length(layerconfig)+4)

    # Initialize non-Lstm parameters
    parameters[end] = winit * randn(vocab, embedsize) # backward lstm embedding
    parameters[end-1] = winit * randn(vocab, embedsize) # forward lstm embedding
    parameters[end-2] =  winit * randn(layerconfig[end]*2, vocab)# final layer weight
    parameters[end-3] = zeros(1, vocab) # final layer bias

    len = length(layerconfig)
    # forward layer parameters
    input = embedsize
    for k=1:len
        parameters[2k-1] = winit * randn(input+layerconfig[k], 4*layerconfig[k])
        parameters[2k] = zeros(1, 4*layerconfig[k])
        parameters[2k][1:layerconfig[k]] = 1 # forget gate bias
        input = layerconfig[k]
    end

    # bacward layer parameters
    input = embedsize
    for k=1:len
        parameters[2k-1+2len] = winit * randn(input+layerconfig[k], 4*layerconfig[k])
        parameters[2k+2len] = zeros(1, 4*layerconfig[k])
        parameters[2k+2len][1:layerconfig[k]] = 1 # forget gate bias
        input = layerconfig[k]
    end
    return map(p->convert(atype, p), parameters)
end


# One initialization 2 different copies of the same state initialization
function initstate(atype, layerconfig, batchsize)
    state = Array(Any, 2*length(layerconfig))
    for k=1:length(layerconfig)
        state[2k-1] = zeros(batchsize, layerconfig[k])
        state[2k] = zeros(batchsize, layerconfig[k])
    end
    return map(s->convert(atype, s), state)
end


function lstm(weight, bias, hidden, cell, input)
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


# multi layer lstm forward function
function forward(parameters, states, input)
    x = input
    for i=1:2:length(states)
        (states[i], states[i+1]) = lstm(parameters[i], parameters[i+1], states[i], states[i+1], x)
        x = states[i]
    end
    return x
end


function loss(parameters, states, sequence, values=[])
    total = 0.0
    count = 0
    atype = typeof(AutoGrad.getval(parameters[1]))
    hidden = length(states) / 2
    hidden = convert(Int, hidden)

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    sf =  copy(states)
    for i=1:length(sequence)-1
        input = convert(atype, sequence[i])
        x = input * parameters[end-1]
        hf = forward(parameters[1:2*hidden], sf, x)
        fhiddens[i+1] = copy(hf)
    end
    fhiddens[1] = convert(atype, zeros(size(fhiddens[2])))

    # backward lstm
    bhiddens = Array(Any, length(sequence))
    sb = copy(states)
    for i=length(sequence):-1:2
        input = convert(atype, sequence[i])
        x = input * parameters[end]
        bh = forward(parameters[2hidden+1:4hidden], sb, x)
        bhiddens[i-1] = copy(bh)
    end
    bhiddens[end] = convert(atype, zeros(size(bhiddens[2])))

    # merge layer
    for i=1:length(sequence)
        count += sum(sequence[i])
        ypred = hcat(fhiddens[i], bhiddens[i]) * parameters[end-2] .+ parameters[end-3]
        ynorm = logp(ypred, 2)
        ygold = convert(atype, sequence[i])
        total += sum(ygold .* ynorm)
    end
    push!(values, AutoGrad.getval((-total/count)))
    return - (total/count)
end

lossgradient = grad(loss)
