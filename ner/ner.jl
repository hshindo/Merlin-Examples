using ProgressMeter

mutable struct NER
    worddict::Dict
    chardict::Dict
    tagset
    model
end

function NER()
    words = h5read(wordembeds_file, "key")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict("UNKNOWN" => 1)
    for word in words
        for c in Vector{Char}(word)
            get!(chardict, string(c), length(chardict)+1)
            get!(chardict, string(uppercase(c)), length(chardict)+1)
        end
    end
    NER(worddict, chardict, BIOES(), nothing)
end

function encode_word(ner::NER, words::Vector{String})
    worddict = ner.worddict
    unkword = worddict["UNKNOWN"]
    ids = map(words) do w
        w = lowercase(w)
        # w = replace(word, r"[0-9]", '0')
        get(worddict, w, unkword)
    end
    Var(ids)
end

function encode_char(ner::NER, words::Vector{String})
    chardict = ner.chardict
    unkchar = chardict["UNKNOWN"]
    batchdims = Int[]
    ids = Int[]
    for w in words
        # w = replace(word, r"[0-9]", '0')
        chars = Vector{Char}(w)
        push!(batchdims, length(chars))
        for c in chars
            push!(ids, get(chardict,string(c),unkchar))
        end
    end
    Var(ids, batchdims)
end

function readdata!(ner::NER, path::String)
    data = Tuple[]
    words, tags = String[], String[]
    lines = open(readlines, path)
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line) || i == length(lines)
            isempty(words) && continue
            w = encode_word(ner, words)
            c = encode_char(ner, words)
            t = Var(encode(ner.tagset,tags))
            push!(data, (w,c,t))
            empty!(words)
            empty!(tags)
        else
            items = split(line, "\t")
            push!(words, String(items[1]))
            push!(tags, String(items[2]))
        end
    end
    data
end

function makebatch(batchsize::Int, data::Vector)
    batches = []
    for i = 1:batchsize:length(data)
        j = min(i+batchsize-1, length(data))
        batch = data[i:j]
        b = ntuple(length(data[1])) do k
            x = cat(1, map(x -> x[k].data, batch)...)
            batchdims = cat(1, map(x -> x[k].batchdims, batch)...)
            Var(x, batchdims)
        end
        push!(batches, b)
    end
    batches
end

function train(ner::NER, traindata::Vector, testdata::Vector)
    info("# Training sentences:\t$(length(traindata))")
    info("# Testing sentences:\t$(length(testdata))")
    info("# Words:\t$(length(ner.worddict))")
    info("# Chars:\t$(length(ner.chardict))")
    info("# Tags:\t$(length(ner.tagset.tag2id))")
    testdata = makebatch(100, testdata)

    wordembeds = embeddings(h5read(wordembeds_file,"value"))
    charembeds = embeddings(Float32, length(ner.chardict), 20)
    ner.model = Model(wordembeds, charembeds, length(ner.tagset.tag2id))
    batchsize = 10
    opt = SGD()
    for epoch = 1:50
        println("Epoch:\t$epoch")
        opt.rate = 0.0005 * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("opt rate: $(opt.rate)")
        #opt.rate = 0.00075

        Merlin.config.train = true
        shuffle!(traindata)
        batches = makebatch(batchsize, traindata)
        prog = Progress(length(batches))
        loss = 0.0
        for i in 1:length(batches)
            w, c, t = batches[i]
            h = ner.model(w, c)
            y = softmax_crossentropy(t, h)
            loss += sum(y.data)
            params = gradient!(y)
            foreach(p -> update!(p,opt), params)
            next!(prog)
        end
        loss /= length(batches)
        println("Loss:\t$loss")

        # test
        println("Testing...")
        Merlin.config.train = false
        pred = Int[]
        gold = Int[]
        for (w,c,t) in testdata
            y = ner.model(w, c)
            append!(pred, vec(argmax(y.data,1)))
            append!(gold, t.data)
        end
        length(pred) == length(gold) || throw("Length mismatch: $(length(pred)), $(length(gold))")

        ranges_p = decode(ner.tagset, pred)
        ranges_g = decode(ner.tagset, gold)
        fscore(ranges_g, ranges_p)
        println()
    end
end
