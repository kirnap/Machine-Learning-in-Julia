using Devectorize
function cost(expanded_target_batch, output_layer_state, batchsize)
@devec CE = -sum(expanded_target_batch .* log(output_layer_state + exp(-30))) ./ batchsize
return CE
end
