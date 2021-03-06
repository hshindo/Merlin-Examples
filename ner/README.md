# Named Entity Recognition
This is an example of Named Entity Recognition (NER) with neural networks.  
make sure you have installed `Merlin.jl`.  

## Installation
```
julia> Pkg.add("HDF5")
julia> Pkg.add("ProgressMeter")
julia> Pkg.add("JLD2")
```

## Training
First, download [pre-trained word embeddings](https://cl.naist.jp/~shindo/glove.6B.100d.h5) and put it in `.data/`.  
Then, run the script:
```
julia ner.jl -train <training data> <test data>
```

## Testing
```
julia ner.jl -test <model> <test data>
```
