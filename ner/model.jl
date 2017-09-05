struct Model
    fw
    fc
    fs
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    fw = @graph x begin
        Lookup(wordembeds)(x)
    end

    d = size(charembeds, 1)
    fc = @graph x begin
        x = Lookup(charembeds)(x)
        x = Conv1D(T,5,d,5d,2,1)(x)
        max(x, 2)
    end

    d = size(wordembeds,1) + size(charembeds,1)*5
    dh = 300
    fs = @graph x begin
        x = Conv1D(T,5,d,dh,2,1)(x)
        x = relu(x)
        #x = Standardize(T,(dh,10))(x)

        x1 = dropout(x, 0.3)
        x1 = Conv1D(T,5,dh,dh,2,1)(x1)
        x1 = relu(x1)
        x1 = Standardize(T,(dh,10))(x1)
        x += x1

        x1 = dropout(x, 0.3)
        x1 = Conv1D(T,5,dh,dh,2,1)(x1)
        x1 = relu(x1)
        x1 = Standardize(T,(dh,10))(x1)
        x += x1
        #x = Standardize(T,(dh,10))(x)

        x1 = dropout(x, 0.3)
        x1 = Conv1D(T,5,dh,dh,2,1)(x1)
        x1 = relu(x1)
        x1 = Standardize(T,(dh,10))(x1)
        x += x1

        x1 = dropout(x, 0.3)
        x1 = Conv1D(T,5,dh,dh,2,1)(x1)
        x1 = relu(x1)
        x1 = Standardize(T,(dh,10))(x1)
        x += x1

        Linear(T,dh,ntags)(x)
    end
    Model(fw, fc, fs)
end

function (m::Model)(word::Var, char::Var)
    #word.batchdims = [size(word.data, ndims(word.data))]
    w = m.fw(word)
    c = m.fc(char)
    c.batchdims = w.batchdims
    x = cat(1, w, c)
    m.fs(x)
end
