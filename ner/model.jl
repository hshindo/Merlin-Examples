struct Model
    g
end

function Model(wordembeds::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = eltype(wordembeds[1])
    xw = Node()
    w = Node(lookup, wordembeds, xw)

    xc = Node()
    c = Node(lookup, charembeds, xc)
    d = size(charembeds[1], 1)
    c = Node(Conv1D(T,5,d,5d,2,1), c)
    c = Node(max, c, 2)
    c = Node(resize, c, Node(batchsize,w))

    x = Node(concat, 1, w, c)
    d = size(wordembeds[1],1) + 5size(charembeds[1],1)
    dh = 300
    x = Node(Conv1D(T,5,d,dh,2,1), x)
    x = Node(relu, x)

    x = Node(dropout, x, 0.3)
    x = Node(Conv1D(T,5,dh,dh,2,1), x)
    x = Node(relu, x)
    x = Node(Linear(T,dh,ntags), x)
    g = Graph(input=(xw,xc), output=x)
    Model(g)
end

(m::Model)(w, c) = m.g(w, c)

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
