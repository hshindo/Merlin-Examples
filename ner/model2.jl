struct Model
    fw
    fc
    fs
    O
    q0
    q1
    Y1
    Y2
    #conv
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int, Y1::Var)
    fw = @graph x begin
        Lookup(wordembeds)(x)
    end

    fc = @graph x begin
        d = size(charembeds, 1)
        x = Lookup(charembeds)(x)
        x = Conv1D(T,5,d,5d,2,1)(x)
        max(x, 2)
    end

    dh = 300
    fs = @graph x begin
        d = size(wordembeds,1) + size(charembeds,1)*5
        x = Conv1D(T,5,d,dh,2,1)(x)
        x = relu(x)

        x1 = dropout(x, 0.3)
        x1 = Conv1D(T,5,dh,dh,2,1)(x1)
        x1 = relu(x1)
        x1 = Standardize(T,(dh,10))(x1)
        x += x1

        x
    end

    O = Linear(T,dh,ntags)
    q0 = zerograd(rand(T,ntags,1))
    q1 = zerograd(rand(T,ntags,1))
    #Y1 = zerograd(ones(T,ntags,ntags)*T(1/ntags))
    Y2 = zerograd(ones(T,ntags,ntags)*T(1/ntags))
    #conv = Conv1D(T,6d,ntags,2d,2d)
    Model(fw, fc, fs, O, q0, q1, Y1, Y2)
end

function (m::Model)(data::Tuple)
    w, c, t = data
    w = m.fw(w)
    c = m.fc(c)
    c.batchdims = w.batchdims
    x = cat(1, w, c)
    h = m.fs(x)
    u = m.O(h)
    Q = softmax(-u)

    for i = 1:5
        L = m.Y1 * Q
        #R = m.Y2 * Q
        s = 1
        vars = Var[]
        for n in w.batchdims
            l = cat(2, m.q0, L[:,s:s+n-2])
            l.batchdims = [size(l.data,2)]
            #r = cat(2, R[:,s+1:s+n-1], m.q1)
            #r.batchdims = l.batchdims
            push!(vars, l)
            s += n
        end
        u += cat(2, vars...)
        Q = softmax(-u)
    end

    #n = size(Q.data, 2)
    #for i = 1:5
    #    L = m.Y1 * concat(2, m.q0, Q)[:,1:n]
    #    R = m.Y2 * concat(2, Q, m.q1)[:,2:n+1]
    #    QQ = L + R
    #    u += QQ
    #    Q = softmax(-u)
    #end
    y = Q

    if Merlin.config.train
        crossentropy(t, y)
    else
        vec(argmax(y.data,1))
    end
end

#=
function (m::Model)(word::Var, chars::Vector{Var})
    w = m.fw(word)
    cs = Var[]
    for i = 1:length(chars)
        push!(cs, m.fc(chars[i]))
    end
    c = cat(2, cs...)
    s = cat(1, w, c)
    h = m.fs(s)
    u = m.O(h)
    #F = m.conv(h)
    Q = softmax(-u)
    n = length(chars)
    for i = 1:5
        L = m.Y1 * cat(2, m.q0, Q)[:,1:n]
        R = m.Y2 * cat(2, Q, m.q1)[:,2:n+1]
        QQ = L + R
        u += QQ
        Q = softmax(-u)
    end
    Q
end
=#
