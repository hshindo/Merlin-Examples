mutable struct Parser
    worddict::Dict
    chardict::Dict
    model
end

function Parser()
    words = h5read(wordembeds_file, "key")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict("UNKNOWN" => 1)

    for word in words
        for c in Vector{Char}(word)
            get!(chardict, string(c), length(chardict)+1)
            get!(chardict, string(uppercase(c)), length(chardict)+1)
        end
    end
    Parser(worddict, chardict, nothing)
end

function encode(p::DepParser, words::Vector{String})
    worddict = p.worddict
    chardict = p.chardict
    unkword = worddict["UNKNOWN"]
    unkchar = chardict["UNKNOWN"]
    w = map(w -> get(worddict,lowercase(w),unkword), words)
    #cs = map(words) do w
    #    map(c -> get(chardict,string(c),unkchar), Vector{Char}(w))
    #end
    w
end

function readdata!(path::String, p::DepParser)
    datasets = Dataset[]
    words, tags = String[], String[]
    lines = open(readlines, path)
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line) || i == length(lines)
            isempty(words) && continue
            w, cs = encode(ner, words)
            t = encode(ner.tagset, tags)
            push!(datasets, Dataset(w,cs,t))
            empty!(words)
            empty!(tags)
        else
            items = split(line, "\t")
            push!(words, String(items[1]))
            push!(tags, String(items[2]))
            #word = replace(word, r"[0-9]", '0')
        end
    end
    datasets
end

function setup_data(doc::Vector, worddict, chardict)
    data = []
    unkword = worddict["UNKNOWN"]
    root = worddict["PADDING"]
    for sent in doc
        w = Int[root]
        cs = Var[Var([0])]
        h = Int[0]
        for items in sent
            word = items[2]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)

            chars = Vector{Char}(word0)
            charids = map(c -> get!(chardict,c,length(chardict)+1), chars)

            head = parse(Int, items[7]) + 1
            push!(w, wordid)
            push!(cs, Var(charids))
            push!(h, head)
        end
        push!(data, (Var(w),cs,Var(h)))
    end
    data
end
