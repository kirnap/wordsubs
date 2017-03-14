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


function bilstm(model, states, sequence; pdrop=(0, 0))
    total = 0.0
    count = 0
    atype = typeof(AutoGrad.getval(model[:fembed]))

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    sf = copy(states)
    for i=1:length(sequence)-1
        # input = convert(atype, sequence[i])
        # x = input * model[:fembed]
        x = model[:fembed][sequence[i], :]
        h = forward(model[:forw], sf, x)
        fhiddens[i+1] = copy(h)
    end
    fhiddens[1] = convert(atype, zeros(size(fhiddens[2])))

    # backward lstm
    bhiddens = Array(Any, length(sequence))
    sb = copy(states)
    for i=length(sequence):-1:2
        #input = convert(atype, sequence[i])
        #x = input * model[:bembed]
        x = model[:bembed][sequence[i], :]
        bhiddens[i-1] = forward(model[:back], sb, x)
    end
    bhiddens[end] = convert(atype, zeros(size(bhiddens[2])))

    # concatenate layer
    for i=1:length(fhiddens)
        hf = dropout(fhiddens[i], pdrop[1])
        hb = dropout(bhiddens[i], pdrop[2])
        ypred = hcat(hf, hb) * model[:soft][1] .+ model[:soft][2]
        ynorm = logp(ypred, 2)
        ygold = convert(atype, sequence[i])
        count += size(ygold, 1)
        total += sum(ygold .* ynorm)
    end
    return - total / count
end


lossgradient = grad(bilstm)


function oparams{T<:Number}(::Array{T}; o...)
    o = Dict(o)
    if haskey(o, :gclip)
        return Sgd(;lr=o[:gclip])
    else
        return Sgd()
    end
end


function oparams{T<:Number}(::KnetArray{T}; o...)
    o = Dict(o)
    if haskey(o, :gclip)
        return Sgd(;gclip=o[:gclip])
    else
        return Sgd()
    end
end


oparams(a::Associative; o...)=Dict(k=>oparams(v;o...) for (k,v) in a)
oparams(a; o...)=map(x->oparams(x;o...), a)
initopts(model; o...) = oparams(model; o...)


function train(model, state, sequence, opts; pdrop=(0,0))
    gloss = lossgradient(model, state, sequence;pdrop=pdrop)
    # for k in keys(gloss)
    #     if k == :forw || k == :back || k == :soft
    #         axpy!(2, gloss[k][1], model[k][1])
    #         axpy!(2, gloss[k][2], model[k][2])
    #     elseif k == :fembed || k == :bembed
    #         axpy!(2, gloss[k], model[k])
    #     end
    # end
    update!(model, gloss, opts)
end
