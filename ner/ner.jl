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

function encode(ner::NER, words::Vector{String})
    worddict = ner.worddict
    chardict = ner.chardict
    unkword = worddict["UNKNOWN"]
    unkchar = chardict["UNKNOWN"]
    w = map(w -> get(worddict,lowercase(w),unkword), words)
    cs = map(words) do w
        map(c -> get(chardict,string(c),unkchar), Vector{Char}(w))
    end
    w, cs
end

function readdata!(ner::NER, path::String)
    data_w, data_c, data_t = Var[], Var[], Var[]
    words, tags = String[], String[]
    lines = open(readlines, path)
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line) || i == length(lines)
            isempty(words) && continue
            w, cs = encode(ner, words)
            t = encode(ner.tagset, tags)
            push!(data_w, Var(w))
            batchdims = map(length, cs)
            c = cat(1, cs...)
            push!(data_c, Var(c,batchdims))
            push!(data_t, Var(t))
            empty!(words)
            empty!(tags)
        else
            items = split(line, "\t")
            push!(words, String(items[1]))
            push!(tags, String(items[2]))
            #word = replace(word, r"[0-9]", '0')
        end
    end
    data_w, data_c, data_t
end

function train(ner::NER, trainfile::String, testfile::String)
    train_w, train_c, train_t = readdata!(ner, trainfile)
    test_w, test_c, test_t = readdata!(ner, testfile)
    info("# Training sentences:\t$(length(train_w))")
    info("# Testing sentences:\t$(length(test_w))")
    info("# Words:\t$(length(ner.worddict))")
    info("# Chars:\t$(length(ner.chardict))")
    info("# Tags:\t$(length(ner.tagset))")

    wordembeds = h5read(wordembeds_file, "value")
    charembeds = randn(Float32, 20, length(ner.chardict)) * 0.223f0
    ner.model = Model(wordembeds, charembeds, length(ner.tagset))
    opt = SGD()
    for epoch = 1:50
        println("Epoch:\t$epoch")
        opt.rate = 0.001 / (1 + 0.05*(epoch-1))
        #opt.rate = 0.00075

        train_data = makebatch(16, train_w, train_c, train_t)
        Merlin.config.train = true
        function train_f(data::Tuple)
            w, c, t = data
            y = ner.model(w, c)
            softmax_crossentropy(t, y)
        end
        loss = minimize!(train_f, opt, collect(zip(train_data...)))
        println("Loss:\t$loss")

        # test
        println("Testing...")
        Merlin.config.train = false
        function test_f(data::Tuple)
            w, c = data
            y = ner.model(w, c)
            vec(argmax(y.data,1))
        end
        test_data = collect(zip(test_w, test_c))
        pred = cat(1, map(test_f, test_data)...)
        gold = cat(1, map(t -> t.data, test_t)...)
        length(pred) == length(gold) || throw("Length mismatch.")

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
