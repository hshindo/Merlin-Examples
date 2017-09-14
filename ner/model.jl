struct Model
    fw
    fc
    fs
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    fw = @graph x begin
        Lookup(wordembeds)(x)
    end

    fc = @graph x begin
        d = size(charembeds, 1)
        x = Lookup(charembeds)(x)
        x = Conv1D(T,5,d,5d,2,1)(x)
        max(x, 2)
    end

    fs = @graph x begin
        d = size(wordembeds,1) + size(charembeds,1)*5
        dh = 300
        x = Conv1D(T,5,d,dh,2,1)(x)
        x = relu(x)

        x1 = dropout(x, 0.3)
        x1 = Conv1D(T,5,dh,dh,2,1)(x1)
        x1 = relu(x1)
        x1 = Standardize(T,(dh,10))(x1)
        x += x1

        x2 = dropout(x1, 0.3)
        x2 = Conv1D(T,5,dh,dh,2,1)(x2)
        x2 = relu(x2)
        x2 = Standardize(T,(dh,10))(x2)
        x += x2

        Linear(T,dh,ntags)(x)
    end
    Model(fw, fc, fs)
end

function (m::Model)(data::Tuple)
    w, c, t = data

    #ps = Matrix{Float32}[]
    #for i in batchsize_w.data
    #    push!(ps, setup_posembeds(Float32,50,i))
    #end
    #p = Var(cat(2, ps...))

    w = m.fw(w)
    c = m.fc(c)
    c.batchdims = w.batchdims
    x = cat(1, w, c)
    y = m.fs(x)
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
