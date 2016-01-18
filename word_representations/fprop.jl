using Devectorize
include("helper_functions.jl")


function fprop(input_batch,
              word_embedding_weights,
              embed_to_hid_weights,
              hid_to_output_weights,
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

# Compute the stae of the WORD EMBEDDING LAYER
# Simple look up in inputs word indices in the word_embedding_weights matrix.
embedding_layer_state = modified_reshape(
                                  word_embedding_weights[
                                    vec(modified_reshape(input_batch, row_number=1)), :]',
                                        row_number=numofhid1units * numwords)
# Compute the state of the HIDDEN LAYER

# First Compute inputs to hidden units
inputs_to_hidden_units = embed_to_hid_weights' * embedding_layer_state + repmat(hid_bias, 1, batchsize)

# Next step is to apply the logistic function to find activations
@devec hidden_layer_state = 1./ (1 + exp(-inputs_to_hidden_units))


# Compute the state of the OUTPUT LAYER

# First we need to compute the inputs for softmax units
inputs_to_softmax = hid_to_output_weights' * hidden_layer_state + repmat(output_bias, 1, batchsize)

# This trick is taken from Hinton's homeworks
inputs_to_softmax  = inputs_to_softmax - repmat(findmaxcols(inputs_to_softmax), vocab_size, 1)

# Apply the softmax approach
@devec output_layer_state = exp(inputs_to_softmax)

# Normalize the output_layer_state
output_layer_state = output_layer_state ./ repmat(sum_columns(output_layer_state), vocab_size, 1)



return (embedding_layer_state, hidden_layer_state, output_layer_state)
end
