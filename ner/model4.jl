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

    d = size(wordembeds,1) + size(charembeds,1)*5
    fs = @graph x begin
        dh = 300

        #c = Linear(T,d,dh)(x)
        #c = relu(c)
        #c = Standardize(T,(dh,10))(c)
        #f = Linear(T,d,dh)(x)
        #f = sigmoid(f)
        #x = Linear(T,d,dh)(x)
        #x = f .* c + (1-f) .* x

        #x1 = dropout(x, 0.3)
        #x1 = Conv1D(T,5,dh,dh,2,1)(x1)
        #x1 = sigmoid(x1)
        # x1 = Standardize(T,(dh,10))(x1)
        #x = x1 .* x
        x += x1

        Linear(T,dh,ntags)(x)
    end
    Model(fw, fc, fs)
end

function (m::Model)(data::Tuple)
    w, c, t = data
    w = m.fw(w)
    c = m.fc(c)
    c.batchdims = w.batchdims
    x = cat(1, w, c)
    for b = 1:length(w.batchdims)
        n = w.batchdims[b]
        for i = 1:n
            
        end
    end

    y = m.fs(x)
    if Merlin.config.train
        softmax_crossentropy(t, y)
    else
        vec(argmax(y.data,1))
    end
end
