struct Model
    fw
    fc
    fs
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    fw = @graph x begin
        Lookup(wordembeds)(x)
    end

    fc = @graph (x,dims) begin
        d = size(charembeds, 1)
        x = Lookup(charembeds)(x)
        x = Conv1D(T,5,d,5d,2,1)(x, dims)
        max_batch(x, dims)
    end

    fs = @graph (x,dims) begin
        d = size(wordembeds,1) + size(charembeds,1)*5
        dh = 300
        x = Conv1D(T,5,d,dh,2,1)(x, dims)
        x = relu(x)

        x1 = dropout(x, 0.3)
        x1 = Conv1D(T,5,dh,dh,2,1)(x1, dims)
        x1 = relu(x1)
        x1 = Standardize(T,(dh,10))(x1)
        x += x1

        # biaffine
        #h1 = Linear(T,dh,dh)(x)
        #h1 = relu(h1)
        #h2 = Linear(T,dh,dh)(x)
        #h2 = relu(h2)
        #u = Linear(T,dh,dh)(h1)
        #x = BLAS.gemm('T', 'N', 1, h2, u)

        Linear(T,dh,ntags)(x)
    end
    Model(fw, fc, fs)
end

function (m::Model)(data::Tuple)
    w, batchsize_w, c, batchsize_c, t = data
    w = m.fw(w)
    c = m.fc(c, batchsize_c)
    x = concat(1, w, c)
    y = m.fs(x, batchsize_w)
    if Merlin.config.train
        softmax_crossentropy(t, y)
    else
        vec(argmax(y.data,1))
    end
end
