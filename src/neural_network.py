""" Class implementing a neural network.
 With one hidden layer using the sigmoid activation function.
 We initialize random weights and biases.
 Then it trains the network using backpropagation with gradient descent.
 Finally, it makes predictions using the trained neural network."""

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.layer_count = len(layer_sizes)

        # zero bias for each layer
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(self.layer_count - 1)]
        # random weights for each layer
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(self.layer_count - 1)]
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Propagate the input data through the neural network, compute predictions
    # X is the input matrix
    # returns all activations
    def forward(self, X):
        # list containing the activations of each layer
        activations = [X]
        # Compute the weighted sum of inputs for each layer except output layer
        for i in range(self.layer_count - 1):
            # multiply activation of previous layer with weights, add bias
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            # apply activation function
            activations.append(self.sigmoid(z))

        return activations

    # Backpropagation: figure out how bad hypothesis (each assigned weight) is -> derive E
    def backward(self, X, y, activations):

        # deltas: Gradients of the loss function
        deltas = [None] * (self.layer_count - 1)

        # Initialize backpropagation mechanism with derivative of last layer activation
        last_activation = activations[-1]
        deltas[-1] = (last_activation - y) * self.sigmoid_derivative(last_activation)

        for i in reversed(range(1, self.layer_count - 1)):
            deltas[i - 1] = np.dot(deltas[i], self.weights[i].T) * self.sigmoid_derivative(activations[i])

        # Update weights and biases
        for i in range(self.layer_count - 1):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0)

    def train(self, X, y, epochs):
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError("Number of features in X does not match the size of the input layer.")

        for epoch in range(epochs):
            try:
                activations = self.forward(X)
                self.backward(X, y, activations)

                # Every 100 epochs, print loss.
                if epoch % 100 == 0:
                    loss = np.mean(np.square(y - activations[-1]))
                    print(f'Epoch {epoch}, Loss: {loss}')
            except ValueError as e:
                print("error!", epoch)
                raise ValueError("failed here: ", epoch, e)

    def get_prediction(self, X):
        activations = self.forward(X)
        return activations[-1]


if __name__ == "__main__":
    # Example usage
    # Number of neurons in each layer
    layers_size = [4, 5, 3, 2]

    # Create a neural network
    nn = NeuralNetwork(layers_size)

    # Generate some dummy data
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, (100, 1))

    # Train the neural network
    nn.train(X, y, epochs=1000)

    # Make predictions
    prediction = nn.get_prediction(X)
    print("Predictions:")
    print(prediction)
