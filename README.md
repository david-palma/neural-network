# Implementation of a neural network from scratch in Python

This repository contains a python implementation of a feed forward artificial neural network (ANN) based on multi-layer perceptron (MLP) model.

**NOTE**: the present tutorial is not a complete and comprehensive guide to neural network, rather it is intended to build some basic skills and get familiar with the concepts.

## Model

ANN is a collection of interconnected neurons that incrementally learn from their environment (data) to capture essential linear and nonlinear trends in complex data, so that it provides reliable predictions for new situations containing even partial and noisy information.

The model adopted in this implementation refers to the multi-layer perceptron model with a single hidden layer (three-layer neural network), which is represented in the following figure.

<p align="center"><img src="./figures/mlp.png" width="700px"></img><p>

A three-layer neural network corresponds to a function <!-- $f: \mathbb{R}^N \to \mathbb{R}^M$ --> <img style="transform: translateY(0.25em);" src="svg\xxqYDzOvRn.svg"/> with <!-- $x \in \mathbb{R}^N$ --> <img style="transform: translateY(0.25em);" src="svg\Cl6d7WLCGv.svg"/> and <!-- $y \in \mathbb{R}^M$ --> <img style="transform: translateY(0.25em);" src="svg\L7kZwgStzO.svg"/>.

### Feed-forward propagation
At the very beginning, all weights are initially set to a weighted random number from a normal distribution (i.e., <!-- $\sim N(0,1)$ --> <img style="transform: translateY(0.25em);" src="svg\Qk55eCTqak.svg"/>), whilst the biases are set to zero.

Then, it is possible to compute the propagation forward through the network to generate the output value(s).

The hidden node values are given by:

<!-- $$
\begin{aligned}
a_1 &= s \left(b_1^{(1)} + w_{11}^{(1)}x_1 + w_{12}^{(1)}x_2 \ldots w_{1n}^{(1)}x_n\right) \\
a_2 &= s \left(b_2^{(1)} + w_{21}^{(1)}x_1 + w_{22}^{(1)}x_2 \ldots w_{2n}^{(1)}x_n\right) \\
& \ \; \vdots \\
a_h &= s \left(b_h^{(1)} + w_{h1}^{(1)}x_1 + w_{h2}^{(1)}x_2 \ldots w_{hn}^{(1)}x_n\right)
\end{aligned}
$$ -->

<div align="center"><img src="svg\PBUZSRwuNb.svg"/></div>

Then, using this result it is possible to compute the output node values:

<!-- $$
\begin{aligned}
y_1 &= G \left(b_1^{(2)} + w_{11}^{(2)}a_1 + w_{12}^{(2)}a_2 \ldots w_{1h}^{(2)}a_h\right) \\
y_2 &= G \left(b_2^{(2)} + w_{21}^{(2)}a_1 + w_{22}^{(2)}a_2 \ldots w_{2h}^{(2)}a_h\right) \\
& \ \; \vdots \\
y_m &= G \left(b_m^{(2)} + w_{m1}^{(2)}a_1 + w_{m2}^{(2)}a_2 \ldots w_{mh}^{(2)}a_h\right)
\end{aligned}
$$ -->

<div align="center"><img src="svg\tC06uM4cxG.svg"/></div>

However, it is also possible to use a matrix notation:

<!-- $$f(x) = G \left(b^{(2)} + W^{(2)} \left(s \left(b^{(1)} + W^{(1)}x\right)\right)\right)$$ -->

<div align="center"><img src="svg\YogX5yZw4V.svg"/></div>

where:
* <!-- $b^{(1)} \in \mathbb{R}^H,\; b^{(2)} \in \mathbb{R}^M$ --> <img style="transform: translateY(0.25em);" src="svg\18HRTWvBrn.svg"/> are the bias vectors;
* <!-- $W^{(1)} \in \mathbb{R}^{N\times H},\; W^{(2)} \in \mathbb{R}^{H\times M}$ --> <img style="transform: translateY(0.25em);" src="svg\MDd3BBo683.svg"/> are the weight matrices;
* <!-- $G,\; s$ --> <img style="transform: translateY(0.25em);" src="svg\i6SzNXG8fr.svg"/> are the activation functions.

### Activation function

For the activation function it has been used the hyperbolic tangent, but you can choose the sigmoid function as well.

<p align="center"><img src="./figures/eq11.gif"></img></p>

<p align="center"><img src="./figures/tanh.gif"></img></p>

### Classification

Classification is done by projecting an input vector onto a set of hyperplanes, each of which corresponds to a class.

The distance from the input to a hyperplane reflects the probability that the input is a member of the corresponding class.

The probability that an input vector <img src="./figures/eq12.gif"></img> is a member of a class <img src="./figures/eq13.gif"></img> is:

<p align="center"><img src="./figures/eq14.gif"></img></p>

The model's prediction <img src="./figures/eq15.gif"></img> is the class whose probability is maximal:

<p align="center"><img src="./figures/eq16.gif"></img></p>

### Training and backpropagation

The training of the network takes place through the backpropagation algorithm used in conjunction with the stochastic gradient descent optimisation method.

The total error (loss function) is given by the sum of squared errors of prediction, which is the  sum of the squares of residuals (deviations predicted from actual empirical values of data):

<p align="center"><img src="./figures/eq17.gif"></img></p>

The goal in this step is to find the gradient of each weight with respect to the output:

<p align="center"><img src="./figures/eq18.gif"></img></p>

where <img src="./figures/eq19.gif"></img> is the learning rate, which should be tuned to ensure a fast convergence of the weights to a response, without oscillations.


Now it is possible to apply the chain rule to back propagate the error in order to update the weight matrix and bias vector (the math part is not reported here).

## Usage

To import the module type the following command

```python
# import the module
import NeuralNetwork
```

Then you can define your own neural network with custom number of inputs, hidden neurons, and output neurons

```python
# define the number of neurons in each layer
no_inputs  = 2
no_hiddens = 7
no_outputs = 2

# constructor
ann = NeuralNetwork.MLP([no_inputs,no_hiddens,no_outputs])
```

where the activation function is set by default to be the hyperbolic tangent.

Then, given a training dataset `xt, yt` and a test dataset `x`

```python
# train the neural network
ann.train(xt, yt)

# output prediction
y_pred = ann.predict(x)
```

### Test #1: logical exclusive OR (XOR)

This is a classical non–linearly separable problem for logical XOR with noisy inputs.

The truth table of the logical exclusive OR (XOR) shows that it outputs true whenever the inputs differ:

| <img src="./figures/eq20.gif"></img> | <img src="./figures/eq21.gif"></img> | <img src="./figures/eq22.gif"></img> |
|:-:|:-:|:-:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

* input: <img src="./figures/eq23.gif"></img>
* training set: <img src="./figures/eq24.gif"></img>
* the output consists of 2 classes

The test dataset is a dataset that is independent of the training dataset, but that follows the same probability distribution as the training dataset.

The model is initially fit on the training dataset, so we can take a look at the loss per epoch graph. The figure below shows that the loss monotonically decreasing towards a minimum, which is consistent with the gradient descent optimisation algorithm.

<p align="center"><img src="./figures/xor_cost.gif" width="700px"></img><p>

Let's look at the final prediction (output) using the implemented Artificial Neural Network with 7 neurons in hidden layer <img src="./figures/eq26.gif"></img>.

<p align="center"><img src="./figures/xor_output.gif" width="700px"></img><p>

As you can see, the neural network has been able to find a decision boundary that successfully separates the classes.

### Test #2: multiple classes prediction

This is another non-linearly separable problem where the dataset consists of four (noisy) spirals rotated by a fixed angle <img src="./figures/eq26.gif"></img> between them.

* input: <img src="./figures/eq23.gif"></img>
* training set: <img src="./figures/eq24.gif"></img>
* the output consists of 4 classes

The figure below shows that the loss monotonically decreasing towards a minimum, which is consistent with the gradient descent optimisation algorithm.

<p align="center"><img src="./figures/moon_cost.gif" width="700px"></img><p>

Let's look at the final prediction (output) using the implemented Artificial Neural Network with 15 neurons in hidden layer <img src="./figures/eq26.gif"></img>.

<p align="center"><img src="./figures/moon_output.gif" width="700px"></img><p>

As you can see, the neural network has been able to find a decision boundary that successfully separates the classes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
