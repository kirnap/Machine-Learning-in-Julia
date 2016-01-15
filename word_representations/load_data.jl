using MAT


function load_data(N::Int64)
  #=
  This function loads the training, validation and test sets.
  Further, it divides the training set into mini-batches.
  Finally, it returns the needed data for every case.
  --------------
  Input N : Size of mini-batch
  --------------
  Outputs:
  train_input: An array of size D X N X M, where
              D: number of words in the input(in this dataset it is 3)
              N: size of each mini-batch
              M: number of mini-batches
  train_target: Ab array of size 1 X N X M.
  valid_input: An array of size D X number of points in the validation set.
  test: An array of size D X number of points in the test set.
  vocab: Vocabulary of containing index to word mapping.

  =#

  file = matopen("data.mat")
  data = read(file, "data")
  numdims = size(data["trainData"], 1)
  D = numdims - 1
  M = convert(Int64, floor(size(data["trainData"], 2) / N))
  train_input = reshape(data["trainData"][1:D, 1:N * M], D, N, M)
  train_target = reshape(data["trainData"][D + 1, 1:N * M], 1, N, M)
  valid_input = data["validData"][1:D, :]
  valid_target = data["validData"][D+1, :]
  test_input = data["testData"][1:D, :]
  test_target = data["testData"][D + 1, :]
  vocab = data["vocab"]
  
  return (train_input,
          train_target,
          valid_input,
          valid_target,
          test_input,
          test_target)
end

load_data(3)
