""" Class implementing a neural network.
 With one hidden layer using the sigmoid activation function.
 We initialize random weights and biases.
 Then it trains the network using backpropagation with gradient descent.
 Finally, it makes predictions using the trained neural network."""

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass from input to hidden layer
        self.hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)

        # Forward pass from hidden to output layer
        self.predictions = self.sigmoid(
            np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)

        return self.predictions

    def backward(self, X, y):
        # Compute error
        error = y - self.predictions

        # Compute gradients for the output layer
        delta_output = error * self.sigmoid_derivative(self.predictions)

        # Compute gradients for the hidden layer
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * self.learning_rate
        self.bias_hidden_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(delta_hidden) * self.learning_rate
        self.bias_input_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):

            predictions = self.forward(X)
            self.backward(X, y)

            if epoch % 100 == 0:
                loss = np.mean(np.square(y - predictions))
                print(f'Epoch {epoch}, Loss: {loss}')


if __name__ == "__main__":
    # Example usage
    # Define input, output, and hidden layer sizes
    input_size = 2
    hidden_size = 3
    output_size = 1

    # Create a neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Generate some dummy data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the neural network
    nn.train(X, y, epochs=1000)

    # Make predictions
    predictions = nn.forward(X)
    print("Predictions:")
    print(predictions)