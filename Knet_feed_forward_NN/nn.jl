include("load_data.jls")


using Knet

#=
This network consists of 3 layers:

input layer consisting of linear units (taking the input matrix)
hidden layer including user defined number of logistic units
Output layer with 10 units all of which are a softmax group
=#

@knet function forward_net(train_data; hidden=100)
  input_to_hidden_weights = par(init=Gaussian(0, 0.1), dims = (hidden,256))
  input_to_hidden = input_to_hidden_weights * train_data

  hidden_activation = sigm(input_to_hidden)
  hidden_to_output_weights = par(init=Gaussian(0, 0.1), dims= (10,hidden))

  input_to_output = hidden_to_output_weights * hidden_activation
  return soft(input_to_output)
end

function train(f, data, loss)
    for (x,y) in data
        forw(f, x)
        back(f, y, loss)
        update!(f)
    end
end

function test(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in data
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    sumloss / numloss
end
