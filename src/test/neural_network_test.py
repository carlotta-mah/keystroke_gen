import numpy as np
import pytest
from src.neural_network import NeuralNetwork

def test_neural_network():
    # Define input, output, and hidden layer sizes
    input_size = 2
    hidden_size = 3
    output_size = 1

    # Create a neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Dummy data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the neural network
    nn.train(X, y, epochs=1000)

    # Make predictions
    predictions = nn.forward(X)

    # Check if predictions are close to the actual values
    assert np.allclose(predictions, y, atol=0.1)