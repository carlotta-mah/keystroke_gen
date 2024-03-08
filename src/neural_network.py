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
from sklearn.metrics import accuracy_score


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self,
                 file=None,
                 layer_structure=None,
                 learning_rate=0.01,
                 activation='relu',
                 validation_split=0.2,
                 ):
        self._batch_size = 32
        if file is not None:
            self.load_model(file)
        else:
            if layer_structure is None:
                raise ValueError("Layer structure is required.")
            self.learning_rate = learning_rate
            self.layer_sizes: List[int] = layer_structure
            self.layer_count = len(layer_structure)
            self._initialize_weights_and_biases()
            self.activation = activation
            self._losses: Dict[str, list] = {"train": [], "validation": []}
            self._validation_split = validation_split

    def _initialize_weights_and_biases(self):
        self.biases = [np.ones((1, size)) for size in self.layer_sizes[1:]]
        self.weights = [np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i]) for i in
                        range(1, self.layer_count)]

    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Invalid activation function. Choose from 'sigmoid', 'relu', or 'tanh'.")

    def _activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return sigmoid(x) * (1 - sigmoid(x))
            # return x * (1-x)
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'tanh':
            return 1 - np.power(x, 2)
        else:
            raise ValueError("Invalid activation function. Choose from 'sigmoid', 'relu', or 'tanh'.")


    # Propagate the input data through the neural network, compute predictions
    # X is the input matrix
    # returns all activations
    def forward(self, batch: np.ndarray):
        activations = [batch.copy()]
        for i in range(self.layer_count - 1):
            sums = np.matmul(batch, self.weights[i])
            batch = sums + self.biases[i]
            if i == self.layer_count - 2:
                batch = softmax(batch)
            else:
                batch = self._activation_function(batch)

            # Store the forward pass hidden values for use in backprop
            activations.append(batch.copy())
        return batch, activations

    # Backpropagation: figure out how bad hypothesis (each assigned weight) is -> derive E
    def backward(self, activations, error):
        # compute for output layer

        # init deltas
        deltas = np.multiply(error, (activations[-1]))
        for i in reversed(range(self.layer_count - 1)):
            if i < self.layer_count - 2:
                deltas = np.multiply(deltas, self._activation_derivative(activations[i + 1]))

            w_pre = activations[i].T @ deltas
            b_pre = np.mean(deltas)

            deltas = deltas @ self.weights[i].T

            self.weights[i] -= self.learning_rate * w_pre
            self.biases[i] -= self.learning_rate * b_pre

        return

    def calculate_abs_loss(self, actual: np.ndarray, predicted: np.ndarray):
        return predicted - actual

    def calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return (actual - predicted) ** 2

    def train(self, X, y, epochs):
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError("Number of features in X does not match the size of the input layer.")
        # validation split
        X, X_v, y, y_v = train_test_split(X, y, test_size=self._validation_split, random_state=12)

        # indices = np.random.permutation(len(X))
        # X_shuffled, y_shuffled = np.array(X[indices]), np.array(y[indices])

        for epoch in range(epochs):
            # Shuffle the data and divide into mini-batches
            epoch_losses = []
            for i in range(0, len(X), self._batch_size):
                try:
                    # Select mini-batch
                    x_batch, y_batch = (X[i:(i + self._batch_size)],
                                        y[i:(i + self._batch_size)])
                    # Forward pass
                    pred, activations = self.forward(x_batch)
                    # Calculate loss
                    batch_loss = self.cross_entropy(y_batch, pred) / len(x_batch)
                    # batch_loss = self.categorical_cross_entropy(y_batch, pred)
                    epoch_losses.append(batch_loss)
                    # Backward pass
                    self.backward(activations, self.calculate_abs_loss(y_batch, pred))
                except ValueError as e:
                    raise ValueError("failed in epoch: ", epoch, " with error: ", e)

            train_loss = np.mean(epoch_losses)
            self._losses["train"].append(train_loss)

            validation_pred, _ = self.forward(X_v)
            validation_loss = self.cross_entropy(y_v, validation_pred) / len(X_v)
            self._losses["validation"].append(validation_loss)

            # Every 100 epochs, print loss.
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {train_loss}, Validation Loss: {np.mean(validation_loss)}')

    def get_prediction(self, X):
        pred, _ = self.forward(X)
        # set the highest probability to 1 and the rest to 0
        m = np.zeros_like(
            pred)  # alternatively, with multiple 1:  pred = (pred == pred.max(axis=1)[:,None]).astype(int)
        m[np.arange(len(pred)), pred.argmax(1)] = 1
        return m

    def save_model(self, filename):
        model = {'weights': self.weights, 'biases': self.biases, 'activation': self.activation,
                 'layer_sizes': self.layer_sizes,
                 'losses': self._losses, 'learning_rate': self.learning_rate, 'batch_size': self._batch_size,
                 'validation_split': self._validation_split}
        np.save(filename, model)

    def load_model(self, filename):
        model = np.load(filename, allow_pickle=True)
        self.layer_sizes = model.item().get('layer_sizes')
        self.layer_count = len(self.layer_sizes)
        self.weights = model.item().get('weights')
        self.biases = model.item().get('biases')
        self.activation = model.item().get('activation')
        self._losses = model.item().get('losses')
        self.learning_rate = model.item().get('learning_rate')
        self._batch_size = model.item().get('batch_size')
        self._validation_split = model.item().get('validation_split')
        return

    def plot_learning(self):
        import matplotlib.pyplot as plt
        plt.plot(self._losses["train"], label="Train Loss")
        plt.plot(self._losses["validation"], label="Validation Loss")
        plt.legend()
        plt.show()
        return

    def calculate_accuracy(self, y_true, predictions):
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(y_true, axis=1))

    def calculate_macro_precision(self, y_true, y_pred):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        classes = list(np.unique(y_true))
        precision = 0
        for c in classes:
            c_true = np.array([1 if p == c else 0 for p in y_true])
            c_pred = np.array([1 if p == c else 0 for p in y_pred])
            true_positive = np.sum(np.dot(c_true, c_pred))
            false_positive = np.sum(np.dot((1 - c_true), c_pred))
            if true_positive + false_positive == 0:
                continue
            precision += true_positive / (true_positive + false_positive)
        return precision / len(classes)

    def cross_entropy(self, y_true, y_pred):
        # add epsilon to prevent log(0)
        epsilon = 10 ** -100
        return -np.sum(y_true * np.log(y_pred + epsilon))

    def cross_entropy_grad(self, y_true, y_pred):
        # add epsilon to prevent log(0)
        epsilon = 10 ** -100
        return -y_true / (y_pred + epsilon)
