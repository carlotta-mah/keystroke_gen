import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pytest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from src.neural_network import NeuralNetwork
from src.neural_network import softmax
from src.neural_network import sigmoid
from itertools import chain


def test_neural_network():
    # color map for better visualization
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

    # Generating 1000 observations with 10 labels
    data, labels = make_blobs(n_samples=1000, centers=3, n_features=4, random_state=0)
    # one hot encoding
    labels = np.eye(3)[labels]

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.2, random_state=0)

    # Create a neural network instance
    layer_structure = [X_train.shape[1], 2, 10, 3]
    nn = NeuralNetwork(file=None, layer_structure=layer_structure, learning_rate=0.0002, activation='sigmoid')

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
    nn = NeuralNetwork(file=None, layer_structure=layer_structure, learning_rate=0.0002, activation='sigmoid')

    # Forward propagation
    pred, hidden = nn.forward(X)

    assert len(hidden) == 4  # Number of layers
    assert pred.shape == (4, 1)  # Output layer should have 1 neuron for binary classification


def test_backward_propagation_shapes():
    # Input data and labels
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1], [0], [0], [1]])  # Dummy one-hot encoded labels for binary classification

    # Create a neural network instance
    layer_structure = [X.shape[1], 10, 10, 1]
    nn = NeuralNetwork(file=None, layer_structure=layer_structure, learning_rate=0.0002, activation='sigmoid')

    # Forward
    _, hidden = nn.forward(X)

    # Backward
    nn.backward(hidden, y)

    # Check if weights and biases are updated
    for i in range(len(nn.weights)):
        assert not np.allclose(nn.weights[i], np.zeros_like(nn.weights[i]))
        assert not np.allclose(nn.biases[i], np.zeros_like(nn.biases[i]))


def test_backward_propagation():
    nn = NeuralNetwork(file=None, layer_structure=[2, 2, 2], learning_rate=0.0002, activation='sigmoid')
    X1 = np.array([[0.1, 0.5]])
    y1 = np.array([[0.05, 0.95]])
    nn.weights = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5, 0.7], [0.6, 0.8]])]
    nn.biases = [np.array([0.25, 0.25]), np.array([0.35, 0.35])]
    pred1, hidden1 = nn.forward(X1)
    loss = nn.calculate_abs_loss(y1, pred1)

    nn.backward(hidden1, loss)
    expected_weights = np.array([[0.09933517, 0.19919586],
                                 [0.29667583, 0.39597928], [0.45187813, 0.71056392],
                                 [0.55073363, 0.81081516]])
    weights = list(chain.from_iterable(nn.weights))
    assert (0.00001 > np.abs(weights - expected_weights)).all


def test_save_load_model():
    # Create a neural network instance
    layer_structure = [4, 10, 10, 1]
    nn = NeuralNetwork(file=None, layer_structure=layer_structure, learning_rate=0.0002, activation='sigmoid')

    # Train the neural network
    X = np.random.rand(10, 4)
    y = np.random.rand(10, 1)
    nn.train(X, y, epochs=10)
    train_loss = nn._losses['train']
    validation_loss = nn._losses['validation']
    # Save the model
    nn.save_model('model')

    # Load the model
    nn.load_model('model.npy')

    # Check if the model was loaded correctly
    assert nn.layer_sizes == layer_structure
    assert len(nn.weights) == len(layer_structure) - 1
    assert len(nn.biases) == len(layer_structure) - 1
    assert nn.activation == 'sigmoid'
    assert nn.learning_rate == 0.0002
    assert nn._batch_size == 32
    assert nn._validation_split == 0.2
    assert nn._losses == {'train': train_loss, 'validation': validation_loss}


def test_softmax():
    # test data from https://stackoverflow.com/questions/47372685/softmax-function-in-neural-network-python
    test_array = np.array([[0.101, 0.202, 0.303],
                           [0.404, 0.505, 0.606]])
    test_output = [[0.30028906, 0.33220277, 0.36750817],
                   [0.30028906, 0.33220277, 0.36750817]]
    assert (np.allclose(softmax(test_array), test_output))


def test_sigmoid():
    test_array = np.array([0.26894142, 0.73105858])
    test_output = [0.566833, 0.675038]
    print(sigmoid(test_array))
    assert (np.allclose(sigmoid(test_array), test_output))


def test_cross_entropy_loss():
    # test data from https://neuralthreads.medium.com/categorical-cross-entropy-loss-the-most-important-loss-function-d3792151d05b
    nn = NeuralNetwork(file=None, layer_structure=[4, 10, 10, 1], learning_rate=0.0002, activation='sigmoid')
    y = np.array([[0, 1, 0, 0]])
    pred = np.array([[0.05, 0.85, 0.1, 0.1]])
    loss = nn.cross_entropy(y, pred)
    expected_loss = 0.16251892949777494
    assert (np.allclose(loss, expected_loss))


def test_cross_entropy_grad():
    # test data from https://neuralthreads.medium.com/categorical-cross-entropy-loss-the-most-important-loss-function-d3792151d05b
    nn = NeuralNetwork(file=None, layer_structure=[4, 10, 10, 1], learning_rate=0.0002, activation='sigmoid')
    y = np.array([[0], [1], [0], [0]])
    pred = np.array([[0.05], [0.85], [0.1], [0.1]])
    grad = nn.cross_entropy_grad(y, pred)
    expected_array = np.array([[0], [-1.17647059], [0], [0]])
    assert (np.allclose(grad, expected_array))


def test_macro_precision():
    nn = NeuralNetwork(file=None, layer_structure=[4, 10, 10, 1], learning_rate=0.0002, activation='sigmoid')
    y = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pred = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
    precision = nn.calculate_macro_precision(y, pred)
    assert precision == 0.5
