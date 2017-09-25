struct Model
    nn
end

function Model2(wordembeds::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
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

        x1 = x
        x1 = dropout(x1, 0.3)
        x1 = Conv1D(T,5,dh,dh,2,1)(x1)
        x1 = relu(x1)
        #x1 = Standardize(T,(dh,10))(x1)

        x2 = concat(1, x, x1)
        x2 = dropout(x2, 0.3)
        x2 = Conv1D(T,5,2dh,dh,2,1)(x2)
        x2 = relu(x2)
        #x2 = Standardize(T,(2dh,10))(x2)

        x3 = concat(1, x, x1, x2)
        x3 = dropout(x3, 0.3)
        x3 = Conv1D(T,5,3dh,dh,2,1)(x3)
        x3 = relu(x3)
        x = x3

        Linear(T,dh,ntags)(x)
    end
    Model(nn)
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
        x = crelu(x)

        x = dropout(x, 0.3)
        x = Conv1D(T,5,2dh,dh,2,1)(x)
        x = crelu(x)
        Linear(T,2dh,ntags)(x)
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
