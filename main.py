from models.softmax_activation import Softmax_Activation
from models.relu_activation import ReLU_Activation
from models.dense_layer import Layer_Dense
import nnfs;
from nnfs.datasets import spiral_data

nnfs.init()

# Get the examplary spiral data as a dataset
X, y = spiral_data(samples=300, classes=3)

# Initialize input layer of the neural network
dense_layer1 = Layer_Dense(2,3)

# Create an instance of activation ReLU helper class
activation_fn = ReLU_Activation()

# Initialize another dense layer of the neural network
dense_layer2 = Layer_Dense(3,3)

# Create an instance of softmax activation helper class
activation_fn2 = Softmax_Activation() 

# Forward the input data through the input layer
dense_layer1.forward(X)

# Takes the output from the input layer and pass it via activation function
activation_fn.activate(dense_layer1.output)

# Forward the input data through the dense layer
dense_layer2.forward(activation_fn.output)

# Take the output from the input layer and pass it via activation function
activation_fn2.activate(dense_layer2.output)

# Print few samples of the network's output
print(activation_fn2.output[:5])