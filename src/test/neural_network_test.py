import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pytest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from src.neural_network import NeuralNetwork


def test_neural_network():
    # color map for better visualization
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

    # Generating 1000 observations with 10 labels
    data, labels = make_blobs(n_samples=1000, centers=2, n_features=4, random_state=0)
    print(data.shape, labels.shape)

    # plot
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
    plt.show()

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.2, random_state=0)

    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Create a neural network instance
    layer_sizes = [4, 5, 6, 1]
    nn = NeuralNetwork(input_size=4, hidden_layer_sizes=layer_sizes, output_size=1)

    # Train the neural network and make predictions on training data
    nn.train(X_train, y_train, epochs=1000)
    predictions = nn.get_prediction(X_train)

    assert predictions.shape == y_train.shape

    # Make predictions on testing data
    predictions = nn.get_prediction(X_test)

    assert predictions.shape == y_test.shape


def test_forward_propagation():
    # Create a neural network instance and input data
    nn = NeuralNetwork(input_size=2, hidden_layer_sizes=[3], output_size=2)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Forward propagation
    activations = nn.forward(X)

    assert len(activations) == 3  # Number of layers including input and output
    assert activations[-1].shape == (4, 2)  # Output layer should have 2 neurons for binary classification



def test_backward_propagation():
    # Create a neural network instance
    nn = NeuralNetwork(input_size=2, hidden_layer_sizes=[3], output_size=2)

    # Input data and labels
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # Dummy one-hot encoded labels for binary classification

    # Forward
    activations = nn.forward(X)

    # Backward
    nn.backward(y, activations)

    # Check if weights and biases are updated
    for i in range(len(nn.weights)):
        assert not np.allclose(nn.weights[i], np.zeros_like(nn.weights[i]))
        assert not np.allclose(nn.biases[i], np.zeros_like(nn.biases[i]))
