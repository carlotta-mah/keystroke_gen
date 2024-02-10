""" Class implementing a neural network.
 With layers using the sigmoid activation function.
 We initialize random weights and biases.
 Then it trains the network using backpropagation with gradient descent.
 Finally, it makes predictions using the trained neural network."""

import numpy as np
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(self,
                 layer_structure,
                 learning_rate=0.01,
                 activation='sigmoid',
                 validation_split=0.2):
        self.layer_sizes: List[int] = layer_structure
        self.layer_count = len(layer_structure)
        self._initialize_weights_and_biases()
        self.learning_rate = learning_rate
        self.activation = activation
        self._losses: Dict[str, list] = {"train": [], "validation": []}
        self._validation_split = validation_split

    def _initialize_weights_and_biases(self):
        self.biases = [np.zeros((1, size)) for size in self.layer_sizes[1:]]
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(self.layer_count - 1)]
    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Invalid activation function. Choose from 'sigmoid', 'relu', or 'tanh'.")

    def _activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'tanh':
            return 1 - np.power(x, 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Propagate the input data through the neural network, compute predictions
    # X is the input matrix
    # returns all activations
    def forward(self, X):
        # list containing the activations of each layer
        hidden = [X.copy()]
        # Compute the weighted sum of inputs for each layer except output layer
        for i in range(self.layer_count - 1):
            # multiply activation of previous layer with weights, add bias
            X = np.matmul(X, self.weights[i]) + self.biases[i]
            if i < self.layer_count - 1:
                X = np.maximum(X, 0)
            # multiply activation of previous layer with weights, add bias
            # z = np.dot(X[-1], self.weights[i]) + self.biases[i]
            # apply activation function
            hidden.append(self._activation_function(X))
        return X, hidden

    # Backpropagation: figure out how bad hypothesis (each assigned weight) is -> derive E
    def backward(self, hidden, deltas):

        # deltas: Gradients of the loss function
        # deltas: np.ndarray = np.zeros(self.layer_count - 1)

        for i in reversed(range(1, self.layer_count - 1)):
            if i != self.layer_count - 2:
                deltas = np.multiply(deltas, np.heaviside(hidden[i + 1], 0))

            # deltas[i - 1] = np.dot(deltas[i], self.weights[i].T) * self._activation_derivative(activations[i])

            weights_grad = hidden[i].T @ deltas
            self.weights[i] -= self.learning_rate * weights_grad
            self.biases[i] -= self.learning_rate * np.mean(deltas, axis=0)
            deltas = deltas @ self.weights[i].T

        # Update weights and biases
        for i in range(self.layer_count - 1):
            pass

    def calculate_loss(self, actual: np.ndarray, predicted: np.ndarray):
        return predicted - actual
    def calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return (actual - predicted) ** 2
    def train(self, X, y, epochs):
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError("Number of features in X does not match the size of the input layer.")
        # validation split
        X, X_v, y, y_v = train_test_split(X, y, test_size=self._validation_split, random_state=42)
        for epoch in range(epochs):
            try:
                pred, hidden = self.forward(X)
                train_loss = self.calculate_loss(y, pred)
                self.backward(hidden, train_loss)
                val_predictions, _ = self.forward(X_v)
                val_loss = np.mean(self.calculate_mse(val_predictions, y_v))
                self._losses["train"].append(train_loss)
                self._losses["validation"].append(val_loss)

                # Every 100 epochs, print loss.
                if epoch % 100 == 0:
                    loss = np.mean(np.square(y - pred))
                    print(f'Epoch {epoch}, Loss: {loss}')
            except ValueError as e:
                raise ValueError("failed in epoch: ", epoch, " with error: ", e)

    def get_prediction(self, X):
        activations = self.forward(X)
        return activations[-1]


def generate_data():
    # Define correlation values
    corr_a = 0.8
    corr_b = 0.4
    corr_c = -0.2

    # Generate independent features
    a = np.random.normal(0, 1, size=100000)
    b = np.random.normal(0, 1, size=100000)
    c = np.random.normal(0, 1, size=100000)
    d = np.random.randint(0, 4, size=100000)
    e = np.random.binomial(1, 0.5, size=100000)

    # Generate target feature based on independent features
    target = 50 + corr_a * a + corr_b * b + corr_c * c + d * 10 + 20 * e + np.random.normal(0, 10, size=100000)

    # Create DataFrame with all features
    df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'target': target})
    return df



if __name__ == "__main__":
    df = generate_data()

    # Separate the features and target
    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    # Create an instance of the NeuralNetwork class
    layer_structure = [X_train.shape[1], 10, 10, 1]
    model = NeuralNetwork( layer_structure, 0.2)

    # Train the model
    epochs = 1000
    model.train(X_train, y_train, epochs)

