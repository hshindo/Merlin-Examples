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

struct BIO2
    dict::Dict
    tags::Vector
end
BIO2() = BIO2(Dict(), String[])

Base.length(tagset::BIO2) = length(tagset.tags)

function encode(tagset::BIO2, tags::Vector{String})
    basetag = ""
    ids = Int[]
    for tag in tags
        if tag == "O"
        elseif tag == "_"
            tag = "I-" * basetag
        else
            basetag = tag
            tag = "B-" * basetag
        end
        for suffix in ("-1","-2")
            t = tag * suffix
            id = get!(tagset.dict, t) do
                id = length(tagset.dict) + 1
                push!(tagset.tags, t)
                id
            end
            suffix == "-1" && push!(ids,id)
        end
    end
    ids
end

function decode(tagset::BIO2, ids::Vector{Int})
    spans = Tuple{Int,Int,String}[]
    bpos = 0
    for i = 1:length(ids)
        tag = tagset.tags[ids[i]]
        startswith(tag, "O-") && continue
        startswith(tag, "B-") && (bpos = i)
        nexttag = i == length(ids) ? "O" : tagset.tags[ids[i+1]]
        if !startswith(nexttag,"I-") && bpos > 0
            strs = Vector{String}(split(tagset.tags[ids[bpos]], "-"))
            basetag = strs[2]
            push!(spans, (bpos,i,basetag))
            bpos = 0
        end
    end
    spans
end
