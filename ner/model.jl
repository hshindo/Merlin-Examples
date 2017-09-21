struct Model
    f
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    f = @graph (w,c) begin
        w = lookup(Embedding(Var(wordembeds)), w)
        c = lookup(Embedding(Var(charembeds)), c)
        d = size(charembeds, 1)
        c = Conv1D(T,5,d,5d,2,1)(c)
        c = max(c, 2)
        c = split(c, batchsize(w))

        x = concat(1, w, c)
        d = size(wordembeds,1) + size(charembeds,1)*5
        dh = 300
        x = Conv1D(T,5,d,dh,2,1)(x)
        x = relu(x)
        x = dropout(x, 0.3)
        x = Conv1D(T,5,dh,dh,2,1)(x)
        x = relu(x)
        Linear(T,dh,ntags)(x)
    end
    Model(f)
end

function (m::Model)(data::Tuple)
    w, c, t = data

    #ps = Matrix{Float32}[]
    #for i in batchsize_w.data
    #    push!(ps, setup_posembeds(Float32,50,i))
    #end
    #p = Var(cat(2, ps...))

    #w = m.fw(w)
    #c = m.fc(c)
    #c.batchdims = w.batchdims
    #x = cat(1, w, c)
    #y = m.fs(x)
    y = m.f(w, c)
    if Merlin.config.train
        softmax_crossentropy(t, y)
    else
        vec(argmax(y.data,1))
    end
end

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
