# Copyright (c) 2016, Diego Alonso Cortez, diego@mathisart.org
# Code released under the BSD 2-Clause License
# Inspired by: Michael Nielsen, Mocha
#   http://neuralnetworksanddeeplearning.com
#   https://github.com/pluskid/Mocha.jl

import numpy as np
import abc  # Define "skeleton" classes that do nothing but provide a blueprint to create classes
from mathisart import abstractstaticmethod

from activation_functions import *

"""The Unicors module with all the cos functions! Each cost function is implemented
as a class with two static methods: the function itself and its derivative.
"""


class CostFunction(metaclass=abc.ABCMeta):
    """A blueprint for the *Cost Function* classes! These interface with the Layer classes.

    The cost (loss, error) measures how well a net's output (activation) matches the supervised
    target. A net's output is the output from the output layer. During training, only the
    derivative is needed.

    The cost function is denoted *J*. The cost function is all-important to neural networks,
    because *learning* for them means minimizing *J* as a function of *W* and *b*. Learning,
    therefore, is simply a minimization problem, and the function to be minized is simply *J*.
    """

    @abstractstaticmethod
    def f(activation, target):
        """The cost function itself!"""

    @abstractstaticmethod
    def D(z, activation, target):
        """The derivative of the cost function (ie. the error delta)!"""


class CrossEntropy(CostFunction):
    """A cool cost function. Efficient derivative!
    """

    @staticmethod
    def f(activation, target):
        cross_entropy = -target * np.log(activation) - (1 - target) * np.log(1 - activation)
        # nan_to_num helps numerical stability for logs of numbers close to 0
        return np.sum(np.nan_to_num(cross_entropy))

    @staticmethod
    def D(z, activation, target, cost=None):
        return activation - target


class SquaredNorm(CostFunction):
    """A not-so-cool cost function. It's half the squared norm of the elementwise differences. Use
    for comparisons with the cross entropy cost.
    """

    @staticmethod
    def f(activation, target):
        return 0.5 * np.linalg.norm(activation - target)**2

    @staticmethod
    def D(z, activation, target, act=ReLU):
        return (activation - target) * act.D(z)
