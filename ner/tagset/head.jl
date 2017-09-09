struct Head
end

Base.length(tagset::Head) = length(tagset.tag2id)

function encode(tagset::Head, tags::Vector{String})
    ids = [0, 0, 0] # special tag ids for ROOT-B, ROOT-I and ROOT-O
    bpos = 0
    for i = 1:length(tags)
        tag = tags[i]
        if startswith(tag, "B-")
            push!(ids, 1)
            bpos = i
        elseif startswith(tag, "I-")
            push!(ids, 2)
        elseif tag == "O"
            push!(ids, 3)
        elseif startswith(tag, "E-")
            push!(ids, bpos+3)
            bpos = 0
        end
    end

    ids = Int[]
    for i = 1:length(tags)
        tag = tags[i]
        if tag == "O"
            push!(ids, 1)
        elseif tag == "_"
            push!(ids, 2)
        else

        end
    end
    map(tags) do tag
        tag == "O" && return 1
        tag == "_" && return 2
        id = get!(tagset.dict, tag) do
            id = length(tagset.dict) + 1
            push!(tagset.tags, tag)
            id
        end

            tag = 2
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
