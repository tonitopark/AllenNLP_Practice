
# Writing Contextual representation of Elmo to disk -- elmo command

"""
    echo "The cryptocurrency space is now figuring out to have the highest search on Google globally ." > sentences.txt
    echo "Bitcoin alone has a sixty percent share of global search ." >> sentences.txt
    allennlp elmo sentences.txt elmo_layers.hdf5 --all
"""

import h5py
h5py_file = h5py.File("elmo_layers.hdf5",'r')
embedding = h5py_file.get("0")
print(embedding)
assert(len(embedding)==3) #one layer for each vector
print(len(embedding[0]))
assert(len(embedding[0])==16) # one entry for each word in the source sentence

# Using Elmo as PyTorch Module to train a new model

from allennlp.modules.elmo import Elmo, batch_to_ids

# This code snippet computes two layers of representations(SNLI and SQUAD models)

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# compute two different representation for each token.
# Each is a linear combination for the 3 layers in ELMO

elmo = Elmo(options_file,weight_file,2,dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence','.'],['Another','.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)
print(embeddings)

print(embeddings.keys())

print('First list \n')
print(embeddings['elmo_representations'][0])
print('\nThe shape of the representation is {}'.format(
    embeddings['elmo_representations'][0].size()))

print('\n')
print('Second list \n')
print(embeddings['elmo_representations'][0])



#Using ELMo with existing allennlp models





