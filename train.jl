using Knet, ArgParse, JLD
include("bilstm.jl")
include("infst.jl")

function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--trainfile"; required=true; help="Infinite stream training file")
        ("--devfile"; required=true; help="dev data to test perplexity")
        ("--vocabfile"; required=true; help="Vocabulary file to train a model")
        ("--savefile"; help="To save the julia model")
        ("--atype"; default=(gpu() >= 0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--hiddens"; arg_type=Int; nargs='+'; default=[300]; help="hidden layer configuration")
        ("--embedding"; arg_type=Int; default=256)
        ("--batchsize"; arg_type=Int; default=23)
        ("--gclip"; arg_type=Float64; default=5.0)
        
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    atype = eval(parse(o[:atype]))
    for (k, v) in o
        println("$k => $v")
    end



    # prepare train data infinite stream
    ulimit = 31
    maxlines = 500
    fdir = open(o[:trainfile])
    vocab = create_vocab(o[:vocabfile])
    sdict = Dict{Int64, Array{Any, 1}}();
    readstream!(fdir, sdict, vocab; maxlines=1000, ulimit=ulimit)

    # initialize model
    vsize = length(vocab)
    m = initmodel(atype, o[:hiddens], o[:embedding], vsize)
    s = initstate(atype, o[:hiddens], o[:batchsize])
    opts = oparams(m, Adam; gclip=o[:gclip])


    # prepare test data
    dev = create_testdata(o[:devfile], vocab, 5)
    sdev = initstate(atype, o[:hiddens], 5)

    dloss = devperp(m, sdev, dev)
    println("Initial dev perplexity is $dloss")

    # training loop
    ids = nextbatch(fdir, sdict, vocab, o[:batchsize]; ulimit=ulimit, maxlines=maxlines)
    checkpoint = 0
    info("Training started")
    while ids != nothing
        train(m, s, ids, opts)
        ids = nextbatch(fdir, sdict, vocab, o[:batchsize]; ulimit=ulimit, maxlines=maxlines)
        if checkpoint % 8000 == 0
            checkpoint = 1
            dloss = devperp(m, sdev, dev)
            println("Dev perp is $dloss")
            moc = convertmodel(m)
            save(o[:savefile], "model", moc)
        end
        checkpoint += 1
    end
end
!isinteractive() && main(ARGS)
