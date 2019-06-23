#!/usr/bin/env python
"""NeuralNetwork.py
The module defines a feed-forward artificial neural network based on a
multi-layer perceptron model. Training through backpropagation algorithm used
in conjunction with the stochastic gradient descent optimisation method.
"""

__author__  = "David Palma"
__license__ = "MIT license"
__version__ = "1.0.0"

import numpy as np

# Activation function: hyperbolic tangent
def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1.0 - np.tanh(x)**2


# Activation function: sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


# Output activation function: softmax function
def softmax(x):
    scores = np.exp(x)
    probs  = scores / np.sum(scores, axis=1, keepdims=True)
    return probs


class MLP:
    """
    Artificial neural network based on MLP model.
    """

    def __init__(self, layers, activation="hyperbolic tangent"):
        """
        Constructor that sets the activation function, the number of neurons
        in each layer. All weights are initially set to a weighted random
        number from a normal distribution, whilst the biases are set to zero.
        """

        assert (len(layers) == 3), "Number of layers supported is 3."

        self.n_activation = activation  # activation function
        self.n_output     = "softmax"   # output activation function
        self.num_layers   = len(layers) # layers (must be 3)
        self.num_inputs   = layers[0]   # n. inputs
        self.num_hiddens  = layers[1]   # n. hidden neurons
        self.num_outputs  = layers[2]   # n. classes
        self.Loss         = []          # array of loss values

        # set activation function
        if activation == "hyperbolic tangent":
            self.activation   = tanh
            self.d_activation = d_tanh
        elif activation == "sigmoid":
            self.activation   = sigmoid
            self.d_activation = d_sigmoid

        # set output activation
        self.output_activation = softmax

        # initialise parameters
        np.random.seed(0)
        self.W1 = np.random.randn(layers[0], layers[1]) / np.sqrt(layers[0])
        self.B1 = np.zeros((1, layers[1]))
        self.W2 = np.random.randn(layers[1], layers[2]) / np.sqrt(layers[1])
        self.B2 = np.zeros((1, layers[2]))

        return None

    def train(self, X, y, eps=1e-3, reg=1e-3, epochs=20000, stop=1e-5):
        """
        Training the neural network using the stochastic gradient descent
        method on a given dataset X and output y.
        """

        assert (eps > 0
                and epochs > 0), "Eps and epochs must be greater than 0."

        # conversion to numpy array
        X = np.asarray(X)
        y = np.asarray(y)

        num_samples = X.shape[0]

        # gradient descent
        for k in range(epochs):

            # forward propagation
            Z1    = X.dot(self.W1) + self.B1
            A1    = self.activation(Z1)
            Z2    = A1.dot(self.W2) + self.B2
            probs = self.output_activation(Z2)

            # margin of error
            delta = probs
            delta[range(num_samples), y] -= 1

            # backpropagte the gradient to the parameters
            dW2 = np.dot(A1.T, delta)
            dB2 = np.sum(delta, axis=0, keepdims=True)

            delta = np.dot(delta, self.W2.T)
            delta = delta * self.d_activation(Z1)

            dW1 = np.dot(X.T, delta)
            dB1 = np.sum(delta, axis=0, keepdims=True)

            # add regularisation gradient contribution
            # note: biases don't have regularisation
            dW2 += reg * self.W2
            dW1 += reg * self.W1

            # perform a parameter update
            self.W1 += -eps * dW1
            self.B1 += -eps * dB1
            self.W2 += -eps * dW2
            self.B2 += -eps * dB2

            # compute the loss every 100 epoch
            if (k % 100 == 0):
                self.Loss.append(self.compute_loss(X, y, reg))

            if ((k > 100) and (np.abs(self.Loss[-1] - self.Loss[-2]) < stop)):
                break

        print("Finished at epoch %d" % k)

        return None

    def compute_loss(self, X, y, reg_lambda=0):
        """
        Compute the loss: average cross-entropy loss and regularisation.
        Note: loss value is low when the correct class probability is high.
        """

        # conversion to numpy array
        X = np.asarray(X)
        y = np.asarray(y)

        # forward propagation
        Z1    = X.dot(self.W1) + self.B1
        A1    = self.activation(Z1)
        Z2    = A1.dot(self.W2) + self.B2
        probs = self.output_activation(Z2)

        # compute the loss
        data_loss = np.sum(-np.log(probs[range(len(X)), y]))

        # add regularisation term to loss
        reg_loss  = reg_lambda/2 * np.sum(np.square(self.W1)) + \
                    reg_lambda/2 * np.sum(np.square(self.W2))

        return 1. / len(X) * (data_loss + reg_loss)

    def predict(self, X):
        """
        Prediction of the output, which is similar to the forward pass part
        of backpropagation but returns the class with the highest probability.
        """

        # conversion to numpy array
        X = np.asarray(X)

        # forward propagation
        Z1    = X.dot(self.W1) + self.B1
        A1    = self.activation(Z1)
        Z2    = A1.dot(self.W2) + self.B2
        probs = self.output_activation(Z2)

        return np.argmax(probs, axis=1)
