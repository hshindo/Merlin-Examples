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
    map(w -> get(worddict,lowercase(w),unkword), words)
end

function encode_char(ner::NER, words::Vector{String})
    chardict = ner.chardict
    unkchar = chardict["UNKNOWN"]
    map(words) do w
        map(c -> get(chardict,string(c),unkchar), Vector{Char}(w))
    end
end

function readdata!(ner::NER, path::String)
    data = []
    words, tags = String[], String[]
    lines = open(readlines, path)
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line) || i == length(lines)
            isempty(words) && continue
            w = encode_word(ner, words)
            c = encode_char(ner, words)
            t = encode(ner.tagset, tags)
            push!(data, (w,c,t))
            empty!(words)
            empty!(tags)
        else
            items = split(line, "\t")
            push!(words, String(items[1]))
            push!(tags, String(items[2]))
            #word = replace(word, r"[0-9]", '0')
        end
    end
    data
end

function train(ner::NER, traindata::Vector, testdata::Vector)
    info("# Training sentences:\t$(length(traindata))")
    info("# Testing sentences:\t$(length(testdata))")
    info("# Words:\t$(length(ner.worddict))")
    info("# Chars:\t$(length(ner.chardict))")
    info("# Tags:\t$(length(ner.tagset))")
    _testdata = []
    for i = 1:200:length(testdata)
        j = min(i+200-1, length(testdata))
        data = testdata[i:j]
        w = map(x -> x[1], data)
        batchsize_w = Var(map(length,w))
        w = Var(cat(1,w...))
        c = Vector{Int}[]
        foreach(x -> append!(c,x[2]), data)
        batchsize_c = Var(map(length,c))
        c = Var(cat(1,c...))
        t = map(x -> x[3], data)
        t = Var(cat(1,t...))
        push!(_testdata, (w,batchsize_w,c,batchsize_c,t))
    end
    testdata = _testdata

    wordembeds = h5read(wordembeds_file, "value")
    charembeds = randn(Float32, 20, length(ner.chardict)) * 0.223f0
    ner.model = Model(wordembeds, charembeds, length(ner.tagset))
    opt = SGD()
    for epoch = 1:50
        println("Epoch:\t$epoch")
        opt.rate = 0.001 / (1 + 0.05*(epoch-1))
        #opt.rate = 0.00075

        shuffle!(traindata)
        batchsize = 16
        _traindata = []
        for i = 1:batchsize:length(traindata)
            j = min(i+batchsize-1, length(traindata))
            data = traindata[i:j]
            w = map(x -> x[1], data)
            batchsize_w = Var(map(length,w))
            w = Var(cat(1,w...))
            c = Vector{Int}[]
            foreach(x -> append!(c,x[2]), data)
            batchsize_c = Var(map(length,c))
            c = Var(cat(1,c...))
            t = map(x -> x[3], data)
            t = Var(cat(1,t...))
            push!(_traindata, (w,batchsize_w,c,batchsize_c,t))
        end

        Merlin.config.train = true
        loss = minimize!(ner.model, opt, _traindata)
        println("Loss:\t$loss")

        # test
        println("Testing...")
        Merlin.config.train = false
        pred = cat(1, map(ner.model, testdata)...)
        gold = cat(1, map(x -> x[end].data, testdata)...)
        length(pred) == length(gold) || throw("Length mismatch: $(length(pred)), $(length(gold))")

        ranges_p = decode(ner.tagset, pred)
        ranges_g = decode(ner.tagset, gold)
        fscore(ranges_g, ranges_p)
        println()
    end
end

function fscore{T}(golds::Vector{T}, preds::Vector{T})
    set = intersect(Set(golds), Set(preds))
    count = length(set)
    prec = round(count/length(preds), 5)
    recall = round(count/length(golds), 5)
    fval = round(2*recall*prec/(recall+prec), 5)
    println("Prec:\t$prec")
    println("Recall:\t$recall")
    println("Fscore:\t$fval")
end
