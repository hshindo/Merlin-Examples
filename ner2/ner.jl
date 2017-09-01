mutable struct NER
    worddict::Dict
    chardict::Dict
    tagset
    model
end

function NER()
    words = h5read(wordembeds_file, "key")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict{String,Int}()
    NER(worddict, chardict, BIO2(), nothing)
end

function encode(ner::NER, words::Vector{String})
    worddict = ner.worddict
    chardict = ner.chardict
    unkword = worddict["UNKNOWN"]
    w = map(w -> get!(worddict,lowercase(w),unkword), words)
    cs = map(words) do w
        chars = Vector{Char}(w)
        map(c -> get!(chardict,string(c),length(chardict)+1), chars)
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
    #train_w, train_c, train_t = train_w[5:5], train_c[5:5], train_t[5:5]
    #test_w, test_c, test_t = train_w, train_c, train_t
    info("# Training sentences:\t$(length(train_w))")
    info("# Testing sentences:\t$(length(test_w))")
    info("# Words:\t$(length(ner.worddict))")
    info("# Chars:\t$(length(ner.chardict))")
    info("# Tags:\t$(length(ner.tagset))")

    wordembeds = h5read(wordembeds_file, "value")
    charembeds = randn(Float32, 20, length(ner.chardict)) * 0.01f0
    ner.model = Model(wordembeds, charembeds, length(ner.tagset))
    opt = SGD()
    epoch = 1
    ispretrain = true
    f = train_f_pre
    while epoch <= 50
        println("Epoch:\t$epoch")
        opt.rate = 0.001 / (1 + 0.05*(epoch-1))
        #opt.rate = 0.00075

        if ispretrain && epoch == 11
            println("Refinement...")
            c1, c2 = 0, 0
            for i = 1:length(train_w)
                y = ner.model(train_w[i], train_c[i], false)
                t = train_t[i]
                for j = 1:size(y.data,2)
                    tt = t.data[j]
                    if y.data[tt,j] >= y.data[tt+1,j]
                        c1 += 1
                    else
                        t.data[j] += 1
                        c2 += 1
                    end
                end
            end
            epoch = 1
            wordembeds = h5read(wordembeds_file, "value")
            charembeds = randn(Float32, 20, length(ner.chardict)) * 0.01f0
            ner.model = Model(wordembeds, charembeds, length(ner.tagset))
            ispretrain = false
            f = train_f
            continue
        end

        train_data = makebatch(16, train_w, train_c, train_t)
        loss = minimize!(x -> f(ner,x), opt, collect(zip(train_data...)))
        println("Loss:\t$loss")

        # test
        println("Testing...")
        test_data = collect(zip(test_w, test_c))
        pred = cat(1, map(x -> test_f(ner,x), test_data)...)
        gold = cat(1, map(t -> t.data, test_t)...)
        length(pred) == length(gold) || throw("Length mismatch.")

        ranges_p = decode(ner.tagset, pred)
        ranges_g = decode(ner.tagset, gold)
        fscore(ranges_g, ranges_p)
        println()
        epoch += 1
    end
end

function train_f_pre(ner, data::Tuple)
    w, c, t = data
    y = ner.model(w, c, true)
    tt = zeros(Float32, size(y.data))
    for i = 1:length(t.data)
        tt[t.data[i],i] = 0.5
        tt[t.data[i]+1,i] = 0.5
    end
    softmax_crossentropy(Var(tt,t.batchdims), y)
end

function train_f(ner, data::Tuple)
    w, c, t = data
    y = ner.model(w, c, true)
    softmax_crossentropy(t, y)
end

function test_f(ner, data::Tuple)
    w, c = data
    y = ner.model(w, c, false)
    vec(argmax(y.data,1))
end

function fscore(golds::Vector{T}, preds::Vector{T}) where T
    set = intersect(Set(golds), Set(preds))
    count = length(set)
    prec = round(count/length(preds), 5)
    recall = round(count/length(golds), 5)
    fval = round(2*recall*prec/(recall+prec), 5)
    println("Prec:\t$prec")
    println("Recall:\t$recall")
    println("Fscore:\t$fval")
end
