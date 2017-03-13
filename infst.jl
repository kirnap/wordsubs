# Infinite stream processing
SOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


function create_vocab(vocabfile::AbstractString)
    result = Dict{AbstractString, Int}(SOS=>1, EOS=>2, UNK=>3)
    open(vocabfile) do f
        for line in eachline(f)
            words= split(line)
            @assert(length(words) == 1, "The vocabulary file seems broken")
            word = words[1]
            get!(result, word, 1+length(result))
        end
    end
    return result
end


""" Reads from infinite stream modifies the s paramater in the light of sequence lengths """
function readstream!(f::IO, s::Dict{Int64, Array{Any, 1}}, vocab::Dict{AbstractString, Int64}, maxlines::Int=100, llimit::Int=3, ulimit::Int=900)
    k = 0
    while (k != maxlines)
        eof(f) && return false
        words = split(readline(f))
        ((length(words) < llimit) || (length(words) > ulimit)) && continue
        seq = Int32[]
        push!(seq, vocab[SOS])
        for word in words
            index = get(vocab, word, vocab[UNK])
            push!(seq, index)
        end
        push!(seq, vocab[EOS])
        skey = length(seq)
        (!haskey(s, skey)) && (s[skey] = Any[])
        push!(s[skey], seq)
        k += 1
    end
end


function mbatch(sequences::Array{Any, 1}, batchsize::Int)
    seqlen = length(sequences[1])
    data = Array(Any, seqlen) #data = [ falses(batchsize, vocabsize) for i=1:seqlen]
    for cursor=1:seqlen
        d = Array(Int32, batchsize)
        for i=1:batchsize
            d[i] = sequences[i][cursor]
        end
        data[cursor] = d
    end
    return data
end


# Collects the sentences from the stream and modifies sdict based on that gives a random length minibatch
function nextbatch(f::IO, sdict::Dict{Int64, Array{Any, 1}}, vocab::Dict{AbstractString, Int64}, batchsize::Int)
    slens = collect(filter(x->length(sdict[x])>=batchsize, keys(sdict)))
    if length(slens) < 10
        readstream!(f, sdict, vocab, 1000)
        slens = collect(filter(x->length(sdict[x])>=batchsize, keys(sdict)))
        if length(slens) == 0
            return nothing
        end
    end
    slen = rand(slens)
    sequence = sdict[slen][1:batchsize]
    deleteat!(sdict[slen] ,1:batchsize)
    return mbatch(sequence, batchsize)
end


function ibuild_sentence(index_to_word::Array{AbstractString,1}, sequence::Array{Any,1}, kth::Int)
    sentence = Any[]
    for i=1:length(sequence)
        z = d[i][kth]
        @assert(length(z) == 1)
        push!(sentence, index_to_word[z])
    end
    return sentence
end

