struct Model
    fw
    fc
    fx
    fy
    fout
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, tagembeds::Matrix{T})
    fw = @graph x begin
        Lookup(wordembeds)(x)
    end

    dc = size(charembeds, 1)
    fc = @graph x begin
        x = Lookup(charembeds)(x)
        x = Conv1D(T,5dc,5dc,2dc,dc)(x)
        max(x, 2)
    end

    dx = size(wordembeds,1) + 5dc
    fx = @graph x begin
        x = Conv1D(T,5dx,2dx,2dx,dx)(x)
        x = relu(x)
        Linear(T,2dx,2dx)(x)
    end

    dy = size(tagembeds, 1)
    fy = @graph x begin
        x = Lookup(tagembeds)(x)
        x = Conv1D(T,5dy,2dy,2dy,dy)(x)
        x = relu(x)
        Linear(T,2dy,2dy)(x)
    end

    dout = 2dx + 2dy
    fout = @graph x begin
        Linear(T,dout,1)(x)
    end
    Model(fw, fc, fx, fy, fout)
end

function (m::Model)(word::Var, char::Var, tag::Var)
    w = m.fw(word)
    c = m.fc(char)
    c.batchdims = w.batchdims
    hx = m.fx(cat(1,w,c))
    hy = m.fy(tag)
    o = m.fout(cat(1,hx,hy))
    o
end

function (m::Model)(word::Var, char::Var, tag::Var, istrain::Bool)
    w = m.fw(word)
    c = m.fc(char)
    c.batchdims = w.batchdims
    hx = m.fx(cat(1,w,c))
    hy = m.fy(tag)
    o = m.fout(cat(1,hx,hy))
    o
end
