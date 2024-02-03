import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pytest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from src.neural_network import NeuralNetwork

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Accuracy score.
    """
    # Ensure y_true and y_pred have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    # Calculate accuracy
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions

    return accuracy

def precision_score(y_true, y_pred):
    """
    Calculate the precision score.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Precision score.
    """
    # Ensure y_true and y_pred have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    # Calculate precision
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    return precision

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
    print(X_train.shape, y_train.shape)

    # Define input, output, and hidden layer sizes
    layer_sizes = [4, 5, 6, 1]

    # Create a neural network
    nn = NeuralNetwork(layer_sizes)

    # Train the neural network

    nn.train(X_train, y_train, epochs=1000)
    # Make predictions
    predictions = nn.get_prediction(X_train)
    assert predictions.shape == y_train.shape

    # Make predictions
    predictions = nn.get_prediction(X_test)

    assert predictions.shape == y_test.shape
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Calculate precision
    precision = precision_score(y_test, predictions)

    # Assert statements to check if accuracy and precision meet certain criteria
    assert accuracy >= 0.8, "Accuracy should be at least 80%"
    assert precision >= 0.8, "Precision should be at least 80%"

    print("Accuracy:", accuracy)
    print("Precision:", precision)

    # Check if predictions are close to the actual values
    # assert np.allclose(predictions, y, atol=0.1)