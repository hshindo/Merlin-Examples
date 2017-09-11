using HDF5
using Merlin

include("eval.jl")
include("tagset/bioes.jl")
include("ner.jl")
include("model2.jl")

const wordembeds_file = ".data/glove.6B.100d.h5"
const wordembeds_file2 = ".data/word2vec_nyt100d.h5"
#const datapath = joinpath(dirname(@__FILE__), ".data")

# training
ner = NER()
traindata = readdata!(ner, ".data/eng.train.BIOES")
testdata = readdata!(ner, ".data/eng.testb.BIOES")
train(ner, traindata, testdata)
#save("NER.jld2", Dict("a"=>seg))

# decoding
#seg = load("NER.jld2")
#println(seg["a"].char2id)
#seg = Merlin.load("NER.merlin")
