include("load_data.jl")

using Knet

#=
Network consists of 3 layers:

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

function main(numofepochs = 100)
   # Load the data
   (train_input,
    train_target,
    valid_input,
    valid_target,
    test_input,
    test_target) = load_data()

    # Split data into minibatches
    batchsize = 100
    train_data = minibatch(train_input, train_target, batchsize)
    test_data = minibatch(test_input, test_target, batchsize)

   net = compile(:forward_net)
   setp(net; lr=0.1)

   println("Test error before training: ", test(net, test_data, zeroone))

   for i=1:numofepochs
     train(net, train_data, softloss)
     println("Test error after epoch ", i, ": ",test(net, test_data, zeroone))
   end

end
main()
