# Preprocessing for the big text files

const SOS = "<s>"
const EOS = "</s>"
const UNK = "<unk>"
const LLIMIT = 3
const ULIMIT = 300

type Data
    word_to_index::Dict{AbstractString, Int}
    index_to_word::Vector{AbstractString}
    batchsize::Int
    sequences::Array
end


function Data(datafile; word_to_index=nothing, vocabfile=nothing, batchsize=20)
    
    vocab_exists = (word_to_index != nothing)
    if !vocab_exists
        word_to_index = Dict{AbstractString, Int}(SOS=>1, EOS=>2, UNK=>3)
        if vocabfile != nothing
            info("Working with provided vocabfile: $vocabfile")
            V = vocab_from_file(vocabfile)
        end
    end
    
    stream = open(datafile)
    sequences = Array[]
    for line in eachline(stream)
        words = Int32[]
        push!(words, word_to_index[SOS])
        for word in split(line)
            if !vocab_exists && vocabfile != nothing && !(word in V)
                word = UNK
            end
            if vocab_exists
                index = get(word_to_index, word, word_to_index[UNK])
            else
                index = get!(word_to_index, word, 1+length(word_to_index))
            end
            push!(words, index)
        end
        push!(words, word_to_index[EOS])
        (length(words) < ULIMIT) && (length(words)>LLIMIT) && push!(sequences, words)
    end
    close(stream)
    vocabsize = length(word_to_index)
    index_to_word = Array(AbstractString, vocabsize)
    for (word, index) in word_to_index
        index_to_word[index] = word
    end
    sort!(sequences, by=x->length(x), rev=true)
    Data(word_to_index, index_to_word, batchsize, sequences)
end


""" Creates a set that contains all the words in that file, vocab file given as each vocab in a single line
    sorted_counted represents pure create_vocab.sh output
"""
function vocab_from_file(vocabfile::AbstractString)
    V = Set{AbstractString}()
    open(vocabfile) do file
        for line in eachline(file)
            words= split(line)
            @assert(length(words) == 1, "The vocabulary file seems broken")
            push!(V, words[1])
        end
    end
    return V
end


""" To be able to understand the sequence structure of the words """
function create_metadata(d::Data)
    sequences = d.sequences
    metadata = Dict{Int32, Int32}()
    for item in sequences
        if length(item) in keys(metadata)
            metadata[length(item)] += 1
        else
            metadata[length(item)] = 1
        end
    end
    return metadata
end


function batch_info(metadata::Dict{Int32, Int32}, batchsize::Int)
    c1 = 0
    c2 = 0
    info("There are $(length(metadata)) different sequence lengths")
    for item in keys(metadata)
        if metadata[item]< batchsize
            c1 += 1
        end
        if div(batchsize,2) < metadata[item] < batchsize
            c2 += 1
        end
    end
    info("There are $c1 sequences less than batchsize")
    info("There are $c2 sequences bigger than half batchsize and less than batchsize")
end


function create_offset(d::Data)
    offsets = Int32[]
    s = d.sequences
    curr = prev = length(s[1])
    push!(offsets, 1)
    for i=2:length(s)
        curr = length(s[i])
        if curr != prev
            push!(offsets, i)
            prev = curr
        end
    end
    (offsets[end] != Int32(length(s))) && push!(offsets, length(s))
    return offsets
end


function modify_offset(of::Array{Int32, 1}, batchsize::Int)
    result = Int32[]
    for i = 1:length(of) - 1 # TODO take care of the last entry
        nit = of[i]
        push!(result, nit)
        nit += batchsize
        while (nit < of[i+1])
            push!(result, nit)
            nit += batchsize
        end
    end
    push!(result, of[end])
    return result
end


function create_minibatch(sequences::Array{Array,1}, batchsize::Int, st::Int, en::Int, vocabsize::Int)
    st = Int64(st)
    seqlen = length(sequences[st])
    data = [ falses(batchsize, vocabsize) for i=1:seqlen]
    sentences = sequences[st:en]
    noi = length(sentences)
    for cursor=1:seqlen
        for row=1:noi
            index = sentences[row][cursor]
            data[cursor][row, index] = 1
        end
    end
    return data
end


function ibuild_sentence(index_to_word::Array{AbstractString,1}, sequence::Array{BitArray{2}, 1}, kth::Int)
    sentence = Any[]
    for i=1:length(sequence)
        z = find(x->x==true, sequence[i][kth, :])
        append!(sentence, index_to_word[z])
    end
    return sentence
end


function next_seq(sequences::Array{Array, 1}, mof::Array{Int32, 1}, index::Int, vocabsize::Int, batchsize::Int)
    st = Int64(mof[index])
    en = mof[index+1]-1
    seq = create_minibatch(sequences, batchsize, st, en, vocabsize)
    return seq
end



# sample usage:
# @time d = Data("../bilstm-in-Knet8/data/nounCompound/google_sets/noun_google_data";vocabfile ="../bilstm-in-Knet8/data/nounCompound/google_sets/google_noun10k.vocab");

