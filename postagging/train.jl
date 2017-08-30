using Merlin
using HDF5

const wordembeds_file = "wordembeds_nyt100.h5"

function train()
    words = h5read(wordembeds_file, "s")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict{Char,Int}(' ' => 1)
    tagdict = Dict{String,Int}()
    #traindoc = UD_English.traindata()
    #testdoc = UD_English.testdata()
    train_w, train_c, train_t = readdata(".data/wsj_00-18.conll", worddict, chardict, tagdict)
    test_w, test_c, test_t = readdata(".data/wsj_22-24.conll", worddict, chardict, tagdict)
    info("# Training sentences:\t$(length(train_w))")
    info("# Testing sentences:\t$(length(test_w))")
    info("# Words:\t$(length(worddict))")
    info("# Chars:\t$(length(chardict))")
    info("# Tags:\t$(length(tagdict))")

    wordembeds = h5read(wordembeds_file, "v")
    T = eltype(wordembeds)
    charembeds = rand(T, 10, length(chardict))
    charembeds = charembeds * T(0.002) - T(0.001)
    nn = Model(wordembeds, charembeds, length(tagdict))
    opt = SGD(momentum=0.95, nesterov=true)

    batchsize = 8
    train_data = makebatch(batchsize, train_w, train_c, train_t)
    train_data = collect(zip(train_data...))
    test_data = makebatch(100, test_w, test_c, test_t)
    test_data = collect(zip(test_data...))

    for epoch = 1:10
        println("Epoch: $epoch")
        opt.rate = 0.001 / batchsize / epoch
        #opt.rate = 0.0075 / (1 + 0.05*(epoch-1))

        loss = 0
        for i in randperm(length(train_data))
            w, c, t = train_data[i]
            y = nn(w, c)
            y = softmax_crossentropy(t, y)
            loss += sum(y.data)
            gradient!(y)
            update!(nn.fw, opt)
            update!(nn.fc, opt)
            update!(nn.fs, opt)
        end
        loss /= batchsize * length(train_data)
        # loss = minimize!(train_f, opt, train_data)
        println("Loss: $loss")

        # test
        ys = Int[]
        zs = Int[]
        for (w,c,t) in test_data
            append!(ys, t.data)
            y = nn(w, c)
            z = vec(argmax(y.data,1))
            append!(zs, z)
        end
        length(ys) == length(zs) || throw("Length mismatch.")

        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        acc = round(acc, 5)
        println("Test acc.: $acc")
        println()
    end
    # Merlin.save("postagger.h5", nn)
end

include("data.jl")
include("model.jl")
train()
