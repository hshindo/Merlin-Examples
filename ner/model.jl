struct Model
    nn
end

function Model(wordembeds::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = eltype(wordembeds[1].data)
    nn = @graph (w,c) begin
        w = lookup(wordembeds, w)
        c = lookup(charembeds, c)
        d = size(charembeds[1], 1)
        c = Conv1D(T,5,d,5d,2,1)(c)
        c = max(c, 2)
        c = resize(c, Node(batchsize,w))

        x = concat(1, w, c)
        d = size(wordembeds[1],1) + 5size(charembeds[1],1)
        dh = 300
        x = Conv1D(T,5,d,dh,2,1)(x)
        x = relu(x)
        x = dropout(x, 0.3)
        x = Conv1D(T,5,dh,dh,2,1)(x)
        x = relu(x)
        Linear(T,dh,ntags)(x)
    end
    Model(nn)
end

(m::Model)(w, c) = m.nn(w, c)

function setup_posembeds{T}(::Type{T}, dim::Int, len::Int)
    embeds = zeros(T, dim, len)
    for p = 1:len
        for i = 1:2:dim
            x = p / 10000^((i-1)/dim)
            embeds[i,p] = sin(x)
            embeds[i+1,p] = cos(x)
        end
    end
    embeds
end
