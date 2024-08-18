from layer import Layer
import numpy as np


# inherit from base class Layer
class FCLayer(Layer):
    """
    FC stands for fully connected, in this case each neuron of the previous layer connects to each of the neuron of the  next, hence the term "fully connected"

    input_size = number of input neurons
    output_size = number of output neurons
    """

    def __init__(self, input_size, output_size, weight, bias):
        super().__init__()

        if weight.size == 0:
            self.weights = np.random.rand(input_size, output_size) - 0.5
            self.weights = np.array([[round(num, 3) for num in self.weights[i]] for i in range(len(self.weights))])
        else:
            self.weights = weight
        # print(f"weight: {self.weights}")
        if bias.size == -0:
            self.bias = np.random.rand(1, output_size) - 0.5
            self.bias = np.array([[round(num, 3) for num in self.bias[i]] for i in range(len(self.bias))])
        else:
            self.bias = bias
        # print(f"bias: {self.bias}")

    # return output for a given input
    def forward_propagation(self, input_data):
        # print("weights:", self.weights)
        # print("bias:", self.bias)

        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias

        print(self.output)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # print(f"output_error: {output_error} weights.T: {self.weights.T}")
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # print(f"sel.input.t: {self.input.T}")
        # print(weights_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        # print(f"self.weights: {self.weights} self.bias: {self.bias}")

        return input_error
