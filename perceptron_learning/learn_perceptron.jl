using DataArrays

function learn_perceptron(train_data, target, weights)
  #=
  To evaluate the perceptron based on a calculation of x' * w (dot product)
  =#
  for i in range(1, size(train_data)[1])
    activation = train_data[i,:] * weights
    if !check_output(activation, target[i])
      weights = weights + train_data[i,:]'
end


function check_output(activation, real_value)
  #=
  2 correct cases:
  - activation>0 and real_value=1
  - activation<0 and real_value=-1
  Please notice that in both cases the multiplication of activation and real_value
  is positive number
  Otherwise negative number
  =#
  if activation * real_value > 0
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
