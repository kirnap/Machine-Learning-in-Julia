include("helper_functions.jl")


function fprop(input_batch,
              word_embedding_weights,
              embed_to_hid_weights,
              hid_to_output_weigths,
              hid_bias,
              output_bias)
#=
To forward propagate through the neural network.
---
Inputs:
  input_batch :
  The input data as a matrix of size numwords X batchsize where,
  numwords is the number of words, batchsize is the number of data points.
  So, if input_batch(i, j) = k then the ith word in data point j is word
  index k of the vocabulary.
=#

# Compute the state of the word embedding layer
# Here simply look up the word_embedding_weights and choose the corresponding weights
numwords, batchsize = size(input_batch)
vocab_size, numofhid1units = size(word_embedding_weights)
numofhid2units = size(embed_to_hid_weights, 2)

# Compute the stae of the word embedding layer
# Simple look up in inputs word indices in the word_embedding_weights matrix.
embedding_layer_state = modified_reshape(
                                  word_embedding_weights[
                                    vec(modified_reshape(input_batch, row_number=1)), :]',
                                        row_number=numofhid1units * numwords)

end
