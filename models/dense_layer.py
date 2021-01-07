import numpy as np;

class Layer_Dense:

    # Layer's init
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights (fc layer) randomly because of lack of pre-trained model and biases to be zeros
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # From inputs, weights and biases calculate layer's output
        self.output = np.dot(inputs, self.weights) + self.biases

