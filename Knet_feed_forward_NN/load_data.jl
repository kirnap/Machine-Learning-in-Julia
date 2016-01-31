# To load a matlab data
using MAT

function load_data()
  #=
  This function loads the training, validation and test datas.
  Finally it returns all the needed data for traing and test datas
  Here are the dimensions:

  train_input = <number_of_futures:256> by <number of training cases>
  train_target = sparse vector for each training case <10> by <number of training cases>

  test_input = <number_of_futures:256> by <number of test cases>
  test_target = <10> by <number of training cases>

  ----
  =#

file = matopen("data.mat")
data = read(file, "data")

# Load the training data
train_input = data["training"]["inputs"]
train_target = data["training"]["targets"]

# Load the validation data
valid_input = data["validation"]["inputs"]
valid_target = data["validation"]["targets"]

# Load the test data
test_input = data["test"]["inputs"]
test_target = data["test"]["targets"]

return (train_input,
        train_target,
        valid_input,
        valid_target,
        test_input,
        test_target)
end
