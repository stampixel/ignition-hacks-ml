class Network:
    """
    Class which handles everything that has to do with the neural network

    Used to define a neural network, used to add layers, ability to change activation function.
    Ability to rerun the network and predict outcome of specific input.
    Training the neural network itself
    """
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []
        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    # training
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                    # print(f"output: {output}")
                # print("pass")
                # compute loss
                err += self.loss(y_train[j], output)

                # print(f"true: {y_train[j]} output: {output}")
                # backward propagation
                error = self.loss_prime(y_train[j], output)
                # print(error)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            # Averaging error for all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
            # print(f"{err},")
