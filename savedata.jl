using ArgParse,JLD
include("preprocess.jl")


function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--trainfile"; required=true; help="To save the metadata about trainfile")
        ("--vocabfile"; required=true; help="To be able to use with different vocabulary sizes")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    for (k, v) in o
        println("$k => $v")
    end

    @time d = Data(o[:trainfile]; vocabfile=o[:vocabfile])
    ofs = create_offset(d)
    vocabsize = length(d.word_to_index)
    fname = split(o[:trainfile], '/')[end]
    savefile = string(fname, "_", vocabsize, "k", ".jld")
    info("saving to $savefile")
    save(savefile, "filename", o[:trainfile], "sequences", d.sequences, "word_to_index", d.word_to_index ,"offsets", ofs, "index_to_word", d.index_to_word)
end
!isinteractive() && main(ARGS)
