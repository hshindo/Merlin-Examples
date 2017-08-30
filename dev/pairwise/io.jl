using MLDatasets
using HDF5

type Dataset
    traindata
    testdata
    worddict
    chardict
    tagdict
end

type Token
    word::Int
    chars::Vector{Int}
end

function setup_data()
    h5file = joinpath(dirname(@__FILE__), "wordembeds_nyt100.h5")
    isfile(h5file) || download("https://cl.naist.jp/~shindo/wordembeds_nyt100.h5", h5file)
    words = h5read(h5file, "s")
    #wordembeds = Lookup(h5read(h5file,"v"))
    #charembeds = Lookup(Float32,100,10)

    worddict = IntDict(words)
    chardict = IntDict{String}()
    tagdict = IntDict{String}()

    traindata = UD_English.traindata()
    testdata = UD_English.testdata()
    #traindata = CoNLL.read(".data/wsj_00-18.conll")
    #testdata = CoNLL.read(".data/wsj_22-24.conll")
    info("# sentences of train data: $(length(traindata))")
    info("# sentences of test data: $(length(testdata))")

    traindata = encode(traindata, worddict, chardict, tagdict, true)
    testdata = encode(testdata, worddict, chardict, tagdict, false)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    info("# tags: $(length(tagdict))")

    Dataset(traindata, testdata, worddict, chardict, tagdict)
end

function encode(data::Vector, worddict, chardict, tagdict, append::Bool)
    data_x, data_y = Vector{Token}[], Vector{Int}[]
    unkword = worddict["UNKNOWN"]
    for sent in data
        push!(data_x, Token[])
        push!(data_y, Int[])
        for items in sent
            word, tag = items[2], items[5]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)

            chars = Vector{Char}(word0)
            if append
                charids = map(c -> push!(chardict,string(c)), chars)
            else
                charids = map(c -> get(chardict,string(c),0), chars)
            end
            tagid = push!(tagdict, tag)
            token = Token(wordid, charids)
            push!(data_x[end], token)
            push!(data_y[end], tagid)
        end
    end
    data_x, data_y
end
