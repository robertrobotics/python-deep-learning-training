import numpy as np

class Softmax_Activation:

    def __init__(self):
        pass

    def activate(self, input_vector):
        # Get unnormalized probabilities 
        exp_values = np.exp(input_vector - np.max(input_vector, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities