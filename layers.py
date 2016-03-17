# Copyright (c) 2016, Diego Alonso Cortez, diego@mathisart.org
# Code released under the BSD 2-Clause License
# Inspired by: Michael Nielsen, Mocha
#   http://neuralnetworksanddeeplearning.com
#   https://github.com/pluskid/Mocha.jl

import numpy as np
import abc  # Define "skeleton" classes that do nothing but provide a blueprint to create classes
from mathisart import abstractstaticmethod

from activation_functions import *
from cost_functions import *

"""The Unicors module with all the layers! Each activation function is implemented
as a class with two static methods: the function itself and its derivative.
"""


class Layer(metaclass=abc.ABCMeta):  # TODO!
    """A neural net is a finite sequence of layers. Stack various layers on top of each other to
    build different architectures. Each layer has a weight matrix and a bias vector. If we
    stack together all the weight matrices of every layer on the net, the result is a 3D tensor.

    The number of neurons on layer *l* is denoted *s_l*.
    The weight matrix on layer *l* is denoted *W^(l)*.
    The bias vector on layer *l* is denoted *b^(l)*.
    The weighted inputs on layer *l* is denoted *z_l*.
    The activation on layer *l* is denoted *a_l*.

    The layer is completely unaware of what happens in the outside world. Two important procedures
    need to be defined to implement a layer:
        Feed-forward: given the inputs, compute the outputs
        Back-propagate: given the errors propagated from upper layers, compute the gradient

    Building a network architecture means defining a sequence of layers, and connecting them.
    """

    def __init__(self, layer, neurons, weight, bias):
        self.layer = layer  # Number of the layer (in the sequence of layers that make up the net)
        self.neurons = neurons  # Amount of neurons
        self.weight = weight  # Weight matrix
        self.bias = bias  # Bias vector


class InputLayer(Layer):
    pass


class PerceptronLayer(Layer):
    """The bread-and-butter layer of Multilayer Perceptrons, the classical universal
    approximators. Usually this layer is fully connected (ie. "dense"), but it also admits
    dropout/dropconnect.

    A single one of these layers (with a nonlinear activation function) is enough to approximate
    any function to any accuracy.

    The input to this layer is a 1-array. If this layer is used as an input layer, ensure the
    input has zero mean and unit variance!
    """

    def __init__(self, neurons=100, dropconn=None, activation=ReLU):
        # Net params
        self.position = 0
        self.eta = 0
        self.lmbda = 0
        self.mbatch_size = 0
        self.n_train = 0
        self.n_params = 0

        # Layer params
        self.neurons = neurons
        self.dropconn = dropconn
        self.activation = activation
        self.dtype = np.float32
        self.weight = 0            # Weight matrix
        self.bias = 0              # Bias vector
        self.x = 0                 # Input vector
        self.z = 0                 # Weighted input vector
        self.a = 0                 # Activation vector
        self.delta = 0             # Delta vector
        self.nabla_w = 0           # Changes in weight matrix
        self.nabla_b = 0           # Changes in bias vector
        self.leakage = 0.25        # TODO: Leakage vector for PReLUs!
        self.nabla_leakage = 0.25  # TODO

    def forward_propagation(self, x):
        self.x = x
        if self.dropconn is not None:
            weight = self.weight * np.random.binomial(1, 1-self.dropconn, size=self.weight.shape)
        else:
            weight = self.weight
        self.z = np.dot(self.x, weight) + self.bias
        self.a = self.activation.f(z=self.z)

    def update_nabla(self, delta):
        self.nabla_w = np.dot(self.x.T, delta)
        self.nabla_b = np.sum(delta, axis=0)

    def backpropagate_error(self, delta):
        self.delta = np.dot(delta, self.weight.T) * self.activation.D(z=self.x)

    def gradient_descent(self):
        self.weight *= (1 - self.eta * self.lmbda/self.n_train)
        self.weight -= self.eta * self.nabla_w/self.mbatch_size
        self.bias -= self.eta * self.nabla_b/self.mbatch_size


class ConvolutionLayer(Layer):
    """The bread-and-butter of Convolutional Neural Networks. Unlike perceptron layers,
    convolutional layers admit multidimensional inputs and can detect local patterns.

    The input to this layer is a 4-array of shape (num, channels, height, width).

    The filter tensor is a 1-array of n x n filters. Eg. if the input image has shape (1, 32, 32)
    and the filter tensor has shape (6, 3, 3), the convolution of the input with the filter tensor
    will yield 6 feature maps of shape (28, 28), which is just a feature tensor of shape
    (6, 28, 28)
    """

    def __init__(self, neurons=100, filter_shape=(6, 3, 3), stride=1, pad=0, activation=ReLU):
        self.position = 0

        filter_std = math.sqrt(2 / (filter_shape[1] * filter_shape[2]))
        self.neurons = neurons
        self.filter = np.random.normal(scale=filter_std, size=filter_shape).astype(np.float32)
        self.stride = stride
        self.pad = pad
        self.activation = activation

    def forward_propagation(self):
        return

    def backpropagation(self):
        return


class PoolingLayer(Layer):
    """A subsamplying layer that does its pooling using the max function.
    """

    def __init__(self, neurons, stride=(2, 2)):
        pass


class SoftmaxLossLayer(Layer):
    """This layer is usually used as the output layer.

    TODO: finish implementation!
    """

    def __init__(self, neurons=10, cost=CrossEntropy):
        self.neurons = neurons
        self.cost = cost
        self.delta = 0

    def forward_propagation(self):
        return

    def backpropagate_error(self, z, activation, target):
        self.delta = self.cost.D(z=z, activation=activation, target=target)


class AutoencoderLayer(Layer):
    """The simplest form of an autoencoder is a feedforward, non-recurrent neural net which is
    very similar to the multilayer perceptron (MLP): an input layer, an output layer, and one
    or more hidden layers connecting them. The differences between autoencoders and MLPs, though,
    are that in an autoencoder, the output layer has the same number of nodes as the input layer,
    and the autoencoder is indended to *reconstruct* the input.

    An *autoencoder* takes an input x in R**p, maps it to a *latent representation* y in R**q
    (encoding), and then maps it back to some z in R**p (decoding). The mapping are usually linear
    maps followed by an elementwise nonlinearity.

    Often we constrain the weights in the *decoder* to be the transpose of the weights in the
    *encoder*. This is referred to as *tied weights*.

    A *denoising auto-encoder* is an auto-encoder with noise corruptions.

    After training, we can take the weights/bias of the *encoder* layer in a (denoising)
    auto-encoder as an initialization of an hidden (perceptron) layer in some net. When there are
    multiple hidden layers, layer-wise pre-training of stacked (denoising) auto-encoders can be
    used to obtain initializations for all the hidden layers.

    Layer-wise pre-training of stacked auto-encoders consists of the following steps:
        1. Train the bottommost auto-encoder.
        2. Remove the decoder layer, construct a new auto-encoder by taking the latent
            representation of the previous auto-encoder as input.
        3. Train the new auto-encoder. Note the weights/bias of the encoder from the
            previously trained auto-encoders are fixed when training the new auto-encoder.
        4. Repeat step 2 and 3 until enough layers are pre-trained.
    """

    def __init__(self, neurons=10, activation=ReLU):
        self.neurons = neurons

    def forward_propagation(self):
        return

    def backpropagation(self):
        return


class _ArgmaxLayer(Layer):
    """Compute the arg-max along the “channel” dimension. This layer is only used in the test
    network to produce predicted classes. It can't do backpropagation.
    """
    def argmax(a, axis=None, out=None):
        return np.argmax(a=a, axis=axis, out=out)


class _ResultsLayer(Layer):
    """Stack this on top of a trained net to have it read results! This is a "read-only" layer; it
    has no backpropagation method or any other features for training!
    """

    def forward_propagation(self):
        return
