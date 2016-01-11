include("learn_perceptron.jl")
using DataArrays


# create initial weight vector
w_init = randn(3,1);

# Create initial train data
train_data = @data([0.80857 0.83721;
                    0.35714 0.8505;
                    -0.75143 -0.7309;
                    -0.3 0.12625;
                    0.64286 -0.54485]);

target = @data([-1.0; -1.0; -1.0; -1.0;-1.0]);

train_data_biased = add_bias(train_data);

number_of_errors = eval_perceptron(train_data_biased, w_init, target)
println("The errors before starting training: ", number_of_errors)
while (number_of_errors > 0)
  w_init = learn_perceptron(train_data_biased, target, w_init)
  number_of_errors = eval_perceptron(train_data_biased, w_init, target)
end
println("Desired weight vector for a given train set: ")
println(w_init)
