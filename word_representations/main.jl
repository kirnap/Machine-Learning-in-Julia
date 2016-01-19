include("fprop.jl")
include("load_data.jl")
include("helper_functions.jl")
include("cost.jl")
using Devectorize


function train(epochs)
#=
Inputs:
  epochs: Number of epochs to run
Output:
  model: A structure containing the learned weights and biases and vocabulary
=#

# SET HYPERPARAMETERS
batchsize = 100
learning_rate = 0.1
momentum = 0.9
numhid1 = 50
numhid2 = 200
init_wt = 0.01  # This is the standard deviation of the normal distributios

# LOAD DATA
(train_input, train_target, valid_input, valid_target, test_input, test_target,
vocab) = load_data(batchsize)
numwords, batchsize, numbatches = size(train_input)
vocab_size = size(vocab, 2)

# Get initial weights and deltas
(word_embedding_weights,
 embed_to_hid_weights,
 hid_to_output_weights,
 hid_bias,
 output_bias,
 word_embedding_weights_delta,
 word_embedding_weights_gradient,
 embed_to_hid_weights_delta,
 hid_to_output_weights_delta,
 hid_bias_delta,
 output_bias_delta) = network_initializer(init_wt, numhid1, numhid2, numwords, vocab_size)

# Create expansion matrix for output layer error calculation
expansion_matrix = eye(vocab_size)

# TRAIN
for epoch = 1:epochs
  println("Epoch: ",epoch)
  for m = 1:numbatches
    input_batch = train_input[:, :, m]
    target_batch = train_target[:, :, m]

    # FORWARD PROPAGATE
    (embedding_layer_state, hidden_layer_state, output_layer_state) =
      fprop(input_batch,
            word_embedding_weights, embed_to_hid_weights,
            hid_to_output_weights, hid_bias, output_bias)
    # Compute the error derivative of the output layer
    expanded_target_batch = output_mapper(expansion_matrix, target_batch)
    @devec error_deriv = output_layer_state - expanded_target_batch

    # Measure the loss function
    CE = cost(expanded_target_batch, output_layer_state, batchsize)

    # BACK PROPAGATION

    # OUTPUT LAYER
    hid_to_output_weights_gradient = hidden_layer_state * error_deriv'
    output_bias_gradient = sum_rows(error_deriv)
    back_propogated_deriv_1 = hid_to_output_weights * error_deriv
    @devec back_propogated_deriv_1 = back_propogated_deriv_1 .* hidden_layer_state .* (1 - hidden_layer_state)

    # HIDDEN LAYER
    embed_to_hid_weights_gradient = embedding_layer_state * back_propogated_deriv_1'
    hid_bias_gradient = sum_rows(back_propogated_deriv_1)
    back_propogated_deriv_2 = embed_to_hid_weights * back_propogated_deriv_1

    word_embedding_weights_gradient[:, :] = 0
    # EMBEDDING LAYER
    for w = 1:numwords
      word_embedding_weights_gradient +=
        expansion_matrix[:, vec(input_batch[w, :])] * back_propogated_deriv_2[1 + (w-1) * numhid1: w * numhid1,:]'
    end

    # UPDATE NETWORK WEIGHTS AND BIASES
    @devec word_embedding_weights_delta = momentum .* word_embedding_weights_delta + word_embedding_weights_gradient ./ batchsize
    @devec word_embedding_weights = word_embedding_weights - learning_rate .* word_embedding_weights_delta

    @devec embed_to_hid_weights_delta = momentum .* embed_to_hid_weights_delta + embed_to_hid_weights_gradient ./ batchsize
    @devec embed_to_hid_weights = embed_to_hid_weights- learning_rate .* embed_to_hid_weights_delta

    @devec hid_to_output_weights_delta = momentum .* hid_to_output_weights_delta + hid_to_output_weights_gradient ./ batchsize
    @devec hid_to_output_weights = hid_to_output_weights - learning_rate .* hid_to_output_weights_delta;

    @devec hid_bias_delta = momentum .* hid_bias_delta + hid_bias_gradient ./ batchsize
    @devec hid_bias = hid_bias - learning_rate .* hid_bias_delta;

    @devec output_bias_delta = momentum .* output_bias_delta + output_bias_gradient ./ batchsize
    @devec output_bias = output_bias - learning_rate .* output_bias_delta
  end
    (embedding_layer_state, hidden_layer_state, output_layer_state) =
    fprop(valid_input,
          word_embedding_weights, embed_to_hid_weights,
          hid_to_output_weights, hid_bias, output_bias)
    datasetsize = size(valid_input, 2)
    expanded_valid_target = output_mapper(expansion_matrix, valid_target)
    CE = cost(expanded_valid_target, output_layer_state, datasetsize)
    println("Validation error after epoch: ", epoch, "is ", CE)
end

end
train(5)
