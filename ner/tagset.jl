struct BIO
    dict::Dict
    tags::Vector
end
BIO() = BIO(Dict("O"=>1), ["O"])

Base.length(tagset::BIO) = length(tagset.tags)

function encode(tagset::BIO, tags::Vector{String})
    basetag = ""
    map(tags) do tag
        tag == "O" && return 1
        if tag == "_"
            tag = "I-" * basetag
        else
            basetag = tag
            tag = "B-" * basetag
        end
        get!(tagset.dict, tag) do
            id = length(tagset.dict) + 1
            push!(tagset.tags, tag)
            id
        end
    end
end

function decode(tagset::BIO, ids::Vector{Int})
    spans = Tuple{Int,Int,String}[]
    bpos = 0
    for i = 1:length(ids)
        tag = tagset.tags[ids[i]]
        tag == "O" && continue
        startswith(tag, "B-") && (bpos = i)
        nexttag = i == length(ids) ? "O" : tagset.tags[ids[i+1]]
        if !startswith(nexttag,"I-") && bpos > 0
            basetag = tagset.tags[ids[bpos]][3:end]
            push!(spans, (bpos,i,basetag))
            bpos = 0
        end
    end
    spans
end

struct BIOES
    dict::Dict
    tags::Vector
end
BIOES() = BIOES(Dict("O"=>1), ["O"])

Base.length(tagset::BIOES) = length(tagset.tags)

function encode(tagset::BIOES, tags::Vector{String})
    basetag = ""
    ids = Int[]
    for i = 1:length(tags)
        tag = tags[i]
        if tag == "O"
            push!(ids, 1)
            continue
        end
        if tag == "_"
            tag = i == length(tags) || tags[i+1] == "O" ? "E-$basetag" : "I-$basetag"
        else
            basetag = tag
            tag = i == length(tags) || tags[i+1] != "_" ? "S-$(basetag)" : "B-$(basetag)"
        end
        id = get!(tagset.dict, tag) do
            id = length(tagset.dict) + 1
            push!(tagset.tags, tag)
            id
        end
        push!(ids, id)
    end
    ids
end

function decode(tagset::BIOES, ids::Vector{Int})
    spans = Tuple{Int,Int,String}[]
    bpos = 0
    for i = 1:length(ids)
        tag = tagset.tags[ids[i]]
        tag == "O" && continue
        startswith(tag, "B-") && (bpos = i)
        startswith(tag, "S-") && (bpos = i)
        nexttag = i == length(ids) ? "O" : tagset.tags[ids[i+1]]
        if (startswith(tag,"S-") || startswith(tag,"E-")) && bpos > 0
            basetag = tagset.tags[ids[bpos]][3:end]
            push!(spans, (bpos,i,basetag))
            bpos = 0
        end
    end
    spans
end

struct BIO2
    dict::Dict
    tags::Vector
end
BIO2() = BIO2(Dict("O"=>1), String["O"])

Base.length(tagset::BIO2) = length(tagset.tags)

function encode(tagset::BIO2, tags::Vector{String})
    basetag = ""
    ids = map(tags) do tag
        tag == "O" && return 1
        if tag == "_"
            tag = "I-" * basetag
        else
            basetag = tag
            tag = "B-" * basetag
        end
        get!(tagset.dict, tag) do
            id = length(tagset.dict) + 1
            push!(tagset.tags, tag)
            id
        end
    end
    res = Int[]
    for i = 1:length(ids)
        prev = i == 1 ? "BOS" : tagset.tags[ids[i-1]]
        next = i == length(ids) ? "EOS" : tagset.tags[ids[i+1]]
        tag = tagset.tags[ids[i]]
        tag = "$(tag):$(prev):$(next)"
        id = get!(tagset.dict, tag) do
            id = length(tagset.dict) + 1
            push!(tagset.tags, tag)
            id
        end
        push!(res, id)
    end
    res
end

function decode(tagset::BIO2, ids::Vector{Int})
    spans = Tuple{Int,Int,String}[]
    bpos = 0
    for i = 1:length(ids)
        tag = tagset.tags[ids[i]]
        startswith(tag, "O") && continue
        startswith(tag, "B-") && (bpos = i)
        nexttag = i == length(ids) ? "O" : tagset.tags[ids[i+1]]
        if !startswith(nexttag,"I-") && bpos > 0
            strs = Vector{String}(split(tagset.tags[ids[bpos]], ":"))
            basetag = strs[1][3:end]
            push!(spans, (bpos,i,basetag))
            bpos = 0
        end
    end
    spans
end
