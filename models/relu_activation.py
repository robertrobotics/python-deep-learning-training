import numpy as np;

class ReLU_Activation:

    # Default initialization of a class
    def __init__(self):
        pass

    # Activation function - ReLU is a simplest activation function allowing to meet non-linear problems with neural network
    def activate(self, input_vector):
        self.output = np.maximum(0, input_vector)

