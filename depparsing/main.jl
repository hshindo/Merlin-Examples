using HDF5
using Merlin

include("parser.jl")
include("model.jl")

const wordembeds_file = ".data/glove.6B.100d.h5"

#traindoc = CoNLL.read(".data/wsj_00-18.conll")
#testdoc = CoNLL.read(".data/wsj_22-24.conll")
#info("# sentences of train doc: $(length(traindoc))")
#info("# sentences of test doc: $(length(testdoc))")

#traindata = setup_data(traindoc, worddict, chardict)
#testdata = setup_data(testdoc, worddict, chardict)
#info("# words: $(length(worddict))")
#info("# chars: $(length(chardict))")
#traindata, testdata, worddict, chardict
