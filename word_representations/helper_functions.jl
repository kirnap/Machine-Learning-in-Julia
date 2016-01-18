#=
To provide some helping functions which are more MATLAB like style
=#

function modified_reshape(matrix; row_number=0, column_number=0)
  #=
  This function allows you not to specify all the dimension of a reshape matrix
  Simply write the dimension that you want to change and let the remaining part
  is done by function itself
  =#
  row, column = size(matrix)
  number_of_entries = row * column

  if row_number == 0 && column_number == 0
    return matrix

  elseif row_number != 0 && column_number == 0
    if number_of_entries % row_number != 0
      throw(DimensionMismatch("Dimensions must be consistent with input matrix"))
    end
    return reshape(matrix, row_number, convert(Int64, number_of_entries / row_number))

  elseif row_number == 0 && column_number != 0
    if number_of_entries % column_number != 0
      throw(DimensionMismatch("Dimensions must be consistent with input matrix"))
    end
    return reshape(matrix, convert(Int64, number_of_entries / column_number), column_number)

  else
    if row_number * column_number != number_of_entries
      throw(DimensionMismatch("Dimensions must be consistent with input matrix"))
    end
      return reshape(matrix, row_number, column_number)
  end
end


function network_initializer(init_wt, numhid1, numhid2, numwords, vocab_size)
  #=
  This function is designed to create initial weights and delta matrices
  for backpropagation algorithm

  =#
  word_embedding_weights = init_wt * randn(vocab_size, numhid1)
  embed_to_hid_weights = init_wt * randn(numwords * numhid1, numhid2)
  hid_to_output_weigths = init_wt * randn(numhid2, vocab_size)
  hid_bias = zeros(numhid2, 1)
  output_bias = zeros(vocab_size, 1)

  word_embedding_weights_delta = zeros(vocab_size, numhid1)
  word_embedding_weights_gradient = zeros(vocab_size, numhid1)

  embed_to_hid_weights_delta = zeros(numwords * numhid1, numhid2)

  hid_to_output_weights_delta = zeros(numhid2, vocab_size)

  hid_bias_delta = zeros(numhid2, 1)

  output_bias_delta = zeros(vocab_size, 1)

return (word_embedding_weights,
        embed_to_hid_weights,
        hid_to_output_weigths,
        hid_bias,
        output_bias,
        word_embedding_weights_delta,
        word_embedding_weights_gradient,
        embed_to_hid_weights_delta,
        hid_to_output_weights_delta,
        hid_bias_delta,
        output_bias_delta)

end

function findmaxcols(matrix)
  #=
  Input: matrix by size of M X N
  returns : a row vector containing maximum of each column in rows.

  =#
  result = zeros(1, size(matrix,2))
  for i=1:size(matrix, 2)
    result[1,i] = maximum(matrix[:,i])
  end
  return result
end


function sum_columns(matrix)
  #=
  MATLAB like summer to sum up the columns of a matrix

  =#
  result = zeros(1, size(matrix,2))
  for i=1:size(matrix, 2)
    result[1, i] = sum(matrix[:, i])
  end
  return result
end


function sum_rows(matrix)
  #=
  MATLAB like summer to sum up the rows of a matrix
  returns a column vector in where the each column is corresponding sum.
  =#
  result = zeros(size(matrix, 1), 1)
  for i=1:size(matrix, 1)
    result[i, 1] = sum(matrix[i, :])
  end
  return result
end


function output_mapper(expansion_matrix, target_batch)
#=
Since Julia suggests devoctarizes implementations,
this function creates a mapping matrix which maps the target values a vector of
1 and 0s
=#
  batchsize = size(target_batch)[2]
  result_matrix = zeros(size(expansion_matrix)[1], batchsize)
  for i=1:batchsize
    result_matrix[:, i] = expansion_matrix[:, target_batch[i]]
  end
  return result_matrix
end
