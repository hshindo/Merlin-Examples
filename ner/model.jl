function setup_model(wordembeds::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = eltype(wordembeds[1])
    w = Node()
    hw = lookup(wordembeds, w)

    c = Node()
    hc = lookup(charembeds, c)
    d = size(charembeds[1], 1)
    hc = Conv1D(T,5,d,5d,2,1)(hc)
    hc = max(hc, 2)
    hc = resize(hc, batchsize(w))

    h = concat(1, hw, hc)
    d = size(wordembeds[1],1) + 5size(charembeds[1],1)
    dh = 300
    h = Conv1D(T,5,d,dh,2,1)(h)
    h = relu(h)

    #x = Node(dropout, x, 0.3)
    #x = Node(Conv1D(T,5,dh,dh,2,1), x)
    #x = Node(relu, x)
    #x = Node(Linear(T,dh,ntags), x)
    Graph(input=(w,c), output=h)
end

#=
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
=#
