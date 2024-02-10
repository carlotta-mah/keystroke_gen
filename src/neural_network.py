""" Class implementing a neural network.
 With layers using the sigmoid activation function.
 We initialize random weights and biases.
 Then it trains the network using backpropagation with gradient descent.
 Finally, it makes predictions using the trained neural network."""

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_sizes, output_size=1, learning_rate=0.01, activation='sigmoid'):
        self.layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.layer_count = len(hidden_layer_sizes) + 2
        self._initialize_weights_and_biases()
        self.learning_rate = learning_rate
        self.activation = activation

    def _initialize_weights_and_biases(self):
        self.biases = [np.zeros((1, size)) for size in self.layer_sizes[1:]]
        self.weights = [np.random.randn(prev, current) for prev, current in
                        zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Invalid activation function. Choose from 'sigmoid', 'relu', or 'tanh'.")

    def _activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'tanh':
            return 1 - np.power(x, 2)

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
            activations.append(self._activation_function(z))

        return activations

    # Backpropagation: figure out how bad hypothesis (each assigned weight) is -> derive E
    def backward(self, y, activations):

        # deltas: Gradients of the loss function
        deltas = [None] * (self.layer_count - 1)

        # Initialize backpropagation mechanism with derivative of last layer activation
        last_activation = activations[-1]
        deltas[-1] = (last_activation - y) * self._activation_derivative(last_activation)

        for i in reversed(range(1, self.layer_count - 1)):
            deltas[i - 1] = np.dot(deltas[i], self.weights[i].T) * self._activation_derivative(activations[i])

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
                self.backward(y, activations)

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
    # Generate synthetic keystroke data
    num_samples = 1000
    num_features = 10
    num_users = 5

    # Generate random keystroke data (binary features)
    X_train = np.random.randint(0, 2, size=(num_samples, num_features))

    # Generate random user labels
    y_train = np.random.randint(0, num_users, size=num_samples)

    # Convert labels to one-hot encoding
    def one_hot_encode(labels, num_classes):
        encoded_labels = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            encoded_labels[i, label] = 1
        return encoded_labels

    y_train_one_hot = one_hot_encode(y_train, num_users)

    # Define dimensions
    input_size = num_features
    hidden_size = 64
    output_size = num_users

    # Create an instance of the NeuralNetwork class
    model = NeuralNetwork(input_size, [hidden_size], output_size, learning_rate=0.01)

    # Train the model
    epochs = 1000
    model.train(X_train, y_train_one_hot, epochs)

