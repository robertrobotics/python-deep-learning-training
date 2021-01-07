from models.relu_activation import ReLU_Activation
from models.dense_layer import Layer_Dense
import nnfs;
from nnfs.datasets import spiral_data

nnfs.init()

# Get the examplary spiral data as a dataset
X, y = spiral_data(samples=300, classes=3)

# Create an instance of activation ReLU helper class
activation_fn = ReLU_Activation()

# Initialize one dense layer of the neural network
dense_layer1 = Layer_Dense(2,3)

# Forward the input data through the dense layer
dense_layer1.forward(X)

# Take the output from the dense layer and pass it via activation function
activation_fn.activate(dense_layer1.output)

# Print few samples of the network's output
print(activation_fn.output[:5])