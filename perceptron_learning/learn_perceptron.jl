#=
Please read and understand the basics of perceptron learning procedure before
further looking the code here are a couple of good ways to look through resources
https://class.coursera.org/neuralnets-2012-001/lecture/10

=#
using DataArrays


function learn_perceptron(train_data_biased, target, weights)
  #=
  To evaluate the perceptron based on a calculation of x' * w (dot product)
  target is a vector containing -1 for negative examples and 1 for positive
  examples
  =#
  for i in range(1, size(train_data_biased)[1])
    activation = train_data_biased[i,:] * weights
    if !check_output(activation, target[i])
      weights = weights + target[i] * train_data_biased[i,:]'  # Update the weight vector
    end
  end
  return weights
end


function check_output(activation, real_value)
  #=
  2 correct cases:
  - activation>0 and real_value=1 ==> positive examples
  - activation<0 and real_value=-1 ==> negative examples
  Please notice that in both cases the multiplication of activation and real_value
  is positive number
  Otherwise negative number
  =#
  if activation[1] * real_value > 0
    return true
  end
    return false
end


function add_bias(train_data)
  #=
  To add bias unit to a given data set
  =#
  number_of_columns = size(train_data)[1]
  return [@data(ones(number_of_columns)) train_data]
end


function eval_perceptron(train_data_biased, weights, target)
  #=
  Count the number of errors in a given weight set and return the number of
  errors
  =#
  number_of_errors = 0
  for i in range(1, size(train_data_biased)[1])
    activation = train_data_biased[i,:] * weights
    if !check_output(activation, target[i])
      number_of_errors += 1
    end
  end
  return number_of_errors
end
