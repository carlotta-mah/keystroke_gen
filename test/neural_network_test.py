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

    # Create a neural network instance
    layer_structure = [X_train.shape[1], 10, 10, 1]
    nn = NeuralNetwork(layer_structure, 0.0002, 'sigmoid')

    # Train the neural network and make predictions on training data
    nn.train(X_train, y_train, epochs=1000)
    predictions = nn.get_prediction(X_train)

    assert predictions.shape == y_train.shape

    # Make predictions on testing data
    predictions = nn.get_prediction(X_test)

    assert predictions.shape == y_test.shape


def test_forward_propagation():
    # Create a neural network instance and input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    layer_structure = [X.shape[1], 10, 10, 1]
    nn = NeuralNetwork(layer_structure, 0.0002, 'sigmoid')

    # Forward propagation
    pred, hidden = nn.forward(X)

    assert len(hidden) == 4  # Number of layers
    assert pred.shape == (4, 1)  # Output layer should have 1 neuron for binary classification



def test_backward_propagation():
    # Input data and labels
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1], [0], [0], [1]])  # Dummy one-hot encoded labels for binary classification

    # Create a neural network instance
    layer_structure = [X.shape[1], 10, 10, 1]
    nn = NeuralNetwork(layer_structure, 0.0002, 'sigmoid')

    # Forward
    _, hidden = nn.forward(X)

    # Backward
    nn.backward(hidden, y)

    # Check if weights and biases are updated
    for i in range(len(nn.weights)):
        assert not np.allclose(nn.weights[i], np.zeros_like(nn.weights[i]))
        assert not np.allclose(nn.biases[i], np.zeros_like(nn.biases[i]))
