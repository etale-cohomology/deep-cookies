# Copyright (c) 2016, Diego Alonso Cortez, diego@mathisart.org
# Code released under the BSD 2-Clause License
# Inspired by: Michael Nielsen, Mocha
#   http://neuralnetworksanddeeplearning.com
#   https://github.com/pluskid/Mocha.jl

import numpy as np
import abc  # Define "skeleton" classes that do nothing but provide a blueprint to create classes
from mathisart import abstractstaticmethod

"""The Unicors module with all the activations functions! Each activation function is implemented
as a class with two static methods: the function itself and its derivative.
"""


class ActivationFunction(metaclass=abc.ABCMeta):
    """A blueprint for the *Activation Function* classes! These interface with the Layer classes.
    An activation function need only be differentiable (or so it was thought!), because
    differentiability will be our basis for learning.

    Regardless of the type of layer, activation functions are be applied (elementwise) on the
    so-called *weighted inputs* (denoted *z*). On a given layer *l*, the image of the activation
    function (on a given *z_l*) is called the *activation* and is denoted *a_l*.

    The activation of a layer will be what we consider its "output".
    """

    @abstractstaticmethod
    def f(z):
        """The activation function itself!"""

    @abstractstaticmethod
    def D(z):
        """The derivative of the activation function!"""


class Sigmoid(ActivationFunction):
    """The good 'ol sigmoid (aka. logistic) activation function. It maps the real line to the
    (open) unit interval bijectively, and it's symmetric across x=0 over y=1/2. It's continuous
    everywhere and differentiable everywhere. It satisfies a first-order nonlinear differential
    equation (f' = f - f**2), leading to a very simple derivative. It saturates very easily,
    slowing down learning to a virtual halt. "Saturation" refers to places where its derivative is
    near 0.

    The sigmoid function is homotopically equivalent to the tanh function.

    Initially loved, the sigmoid has been superseeded by the more computationally efficient and
    biologically plausible ReLU.
    """

    def f(z):
        return (1 + np.exp(z)**-1)**-1

    def D(z):
        return Sigmoid.f(z) * (1 - Sigmoid.f(z))


class TanH(ActivationFunction):
    """The hyperbolic tangent activation function. It maps the real line to the interval (-1; 1)
    bijectively, and it's symmetric across x=0 over y=0. It's continuous everywhere and
    differentiable everywhere. It satisfies a first-order nonlinear differential
    equation (f' = 1 - f**2), leading to a very simple derivative. It saturates very easily,
    slowing down learning to a virtual halt. "Saturation" refers to places where its derivative is
    near 0.

    The tanh function is homotopically equivalent to the sigmoid function.
    """

    def f(z):
        return np.tanh(z)

    def D(z):
        return 1 - TanH.f(z)**2


class ReLU(ActivationFunction):
    """The AWESOME Rectified Linear Unit. Efficient, biological. What more could you want? It maps
    the real line to the positive real line. It's continuous everywhere, but nondifferentiable at
    x=0.

    It has been shown that piecewise linear units, such as ReLU, can compute highly complex and
    structured functions (Montufar et al., 2014).
    """

    def f(z):
        return z * (z > 0)
        # return np.maximum(0, z)

    def D(z):
        return z > 0
        return np.array(z > 0, dtype=np.float32)
        # return 1 * (z > 0)
        # return 1.0 * (z > 0)
        # return (np.sign(z) + 1) / 2


class SoftPlus(ActivationFunction):
    """The everywhere-differentiable version of the ReLU. Its derivative is, surprisingly, the
    sigmoid function.
    """

    def f(z):
        return np.log(1 + np.exp(z))

    def D(z):
        return Sigmoid.f(z)


class NoisyReLU(ActivationFunction):
    """A ReLU with Gaussian noise!
    """

    def f(z):
        return np.maximum(0, z + np.random.randn(*z.shape))  # Some activations below zero!
        return np.maximum(0, z + np.random.randn(*z.shape) * (z > 0))  # No activations below zero!

    def D(z):
        return ReLU.D(z)


class LReLU(ActivationFunction):
    """Leaky ReLU! A generalization of ReLUs with a small nonzero gradient when the unit is no
    active.

    Leaky ReLU is defined as:

    a = max(0, y) + alpha * min(0, y)

    When alpha = 0, we get ReLU. When alpha = 1, we get the Identity activation.
    """
    alpha = 0.1  # Leakage parameter!

    def f(z):
        lrelu = LReLU.alpha * z * (z < 0)
        lrelu += ReLU.f(z)
        # leakyrelu = ReLU.f(z)
        # leakyrelu[np.where(leakyrelu == 0)] = LeakyReLU.alpha * z
        return lrelu

    def D(z):
        lrelu = LReLU.alpha * (z <= 0)
        lrelu += z > 0
        # leakyrelu = ReLU.D(z)
        # leakyrelu[np.where(leakyrelu == 0)] = LeakyReLU.alpha
        return lrelu


class PReLU(ActivationFunction):  # TODO
    """Parametric ReLU! A generalization of leaky ReLUs where the leakage parameter is *learned*
    and can vary across channels! This last remark means the leakage parameter is a vector
    corresponding to each input of the activation function!

    PReLU can be trained by backpropagation and optimized simultaneously with the other layers.

    Source:
        http://arxiv.org/pdf/1502.01852v1.pdf
    """
    def f(z, alpha):
        prelu = alpha * z * (z < 0)  # z and alpha have the same length!
        prelu += ReLU.f(z)
        return prelu

    def D(z, alpha):
        prelu = alpha * (z <= 0)  # z and alpha have the same length!
        prelu += z > 0
        return prelu


class Identity(ActivationFunction):
    """The identity activation function! Even more computationally efficient than the ReLU, but
    maybe less biologically plausible?

    "Why would one want to do use an identity activation function? After all, a multi-layered
    network with linear activations at each layer can be equally-formulated as a single-layered
    linear network. It turns out that the identity activation function is surprisingly useful. For
    example, a multi-layer network that has nonlinear activation functions amongst the hidden
    units and an output layer that uses the identity activation function implements a powerful
    form of nonlinear regression. Specifically, the network can predict continuous target values
    using a linear combination of signals that arise from one or more layers of nonlinear
    transformations of the input."
    """

    def f(z):
        return z

    def D(z):
        return 1


class Kumaraswamy(ActivationFunction):
    """The Kumaraswamy unit (Tomczak), as seen on TV (well, the Arxiv):
    http://arxiv.org/pdf/1505.02581.pdf

    The Kumaraswamy unit follows from modeling a bunch of copies of the same neuron using the
    generalized Kumaraswamy distribution, and it's closely related to the ReLU.

    When a = b = 1, we recover the sigmoid function!
    """
    a, b = 8, 30  # Or 5, 6

    def f(z):
        return 1 - (1 - Sigmoid.f(z)**Kumaraswamy.a)**Kumaraswamy.b

    def D(z):
        return  # TODO
        # return -Kumaraswamy.b * (1 - Sigmoid.f(z)**Kumaraswamy.a)**(Kumaraswamy.b - 1)
