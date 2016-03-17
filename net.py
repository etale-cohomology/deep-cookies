# Copyright (c) 2016, Diego Alonso Cortez, diego@mathisart.org
# Code released under the BSD 2-Clause License
# Inspired by: Michael Nielsen, Mocha
#   http://neuralnetworksanddeeplearning.com
#   https://github.com/pluskid/Mocha.jl

"""This it the main document in the Unicorns library!

Files in the Unicorns library:

    net.py                    # Main class that glues everything together
    layers.py                 # Classes that implement layers
    activation_functions.py   # Classes that implement activation functions
    cost_functions.py         # Classes that implement cost functions
    mnist.py                  # MNIST data manipulation/augmentation
    mathisart.py              # Utilities!
"""

import numpy as np
from numpy.random import uniform
import math
from mathisart import printz, save_ndarray, load_ndarray, timeit

# Unicorns' modules!
from layers import *
from activation_functions import *
from cost_functions import *


###################################################################################################
class Net:
    """Our main neural network class! It brings together the Layer classes, Activation Function
    classes, and Cost Function classes! The main input is a *list* of layer instances (assuming
    suitable topology).

    Examples:
        # Before running this snippet, download and unzip mnist.7z

        # Load digits and set data type
        mnist_imgs, mnist_labels = load_ndarray('mnist.npy')

        # Initialize layers
        fc0 = PerceptronLayer(neurons=784)
        fc1 = PerceptronLayer(neurons=50)
        loss = SoftmaxLossLayer(neurons=10)
        layers = [fc0, fc1, loss]

        # Initialize net
        net = Net(layers=layers)

        # Train net
        net.train(imgs=mnist_imgs, labels=mnist_labels)

    Notes
    -----

    A feedforward neural net is a finite directed acyclic graph going in one direction. This means
    we can group the nodes in "layers". Each vertex is a *neuron*. Each edge is a *weight*, and
    has an origin and a destination.

    Neuron coordinates are (layer, position within layer), yielding topological coordinates.

    A neural net is a *graph*. Each weight is an *edge*, with a beginning and an end. Each neuron
    is a *vertex*. We represent each layer in this graph by a single matrix. In this matrix, each
    row is a *beginning* neuron, and each entry of the row is a path.

    At first it's unintuitive to think of something atomic like a neuron/vertex be represented
    by something non-atomic like a *row*. A *row* isn't even a number, but a collection of
    numbers. But matrices like these turn out to be very convenient graph representations.

    We process digit examples (ie. inputs) not one at a time, but in groups called *minibatches*.
    Each digit example has shape (784,).

    Example: Let the minibatch size be 100. Let *imgs* be a minibatch, meaning it has shape
    (100, 784). Let W be the weight matrix at layer 0 (ie. the input layer) with shape (784, 30).
    Now np.dot(imgs, W) has shape (100, 30), corresponding to there being 100

    A weight will learn (ie. change) slowly if the *input neuron* is low-activation or if the
    *output neuron* is low- or high-activation.

    WARNING: In the literature one usually finds that each *column* is a beginning neuron,
    which results in *imgs* having shape (784, 100) and W having shape (30, 784), so that one has
    np.dot(W, imgs), which has shape (30, 100).

    From http://mochajl.readthedocs.org/:
    "The abstraction and separation of layers from the architecture is important. The library
    implementation can focus on each layer type independently, and does not need to worry about
    how those layers are going to be connected with each other. On the other hand, the network
    designer can focus on the architecture, and does not need to worry about the internal
    computations of layers. This enables us to compose layers almost arbitrarily to create very
    deep / complicated networks. The network could be carrying out highly sophisticated
    computations when viewed as a whole, yet all the complexities are nicely decomposed into
    manageable pieces."

    From other sources:

    "DropConnect is even slower to converge than Dropout, but yields a lower test error in the
     end."

    "Pooling layers are usually used immediately after convolutional layers. The intuition is that
    once a feature has been found, its exact location isn't as important as its rough location
    relative to other features."

    "Regularization is any modification we make to a learning algorithm that is intended to reduce
    its generalization error but not its training error."

    """

    def __init__(self, layers, epochs=10, mbatch_size=128, eta=0.5, dtype=np.float32):
        self.layers = layers
        self.n_layers = len(layers)
        self.epochs = epochs
        self.mbatch_size = mbatch_size
        self.eta = eta
        self.lmbda = self.eta**-1
        self.dtype = dtype

        # Initialize parameters in each non-output layer
        print('\nInitializing net...')
        shapes = np.empty(shape=self.n_layers, dtype=object)
        for i, layer in enumerate(self.layers[:-1]):
            shapes[i+1] = (layer.neurons, self.layers[i+1].neurons)

            # Xavier initialization?
            layer.weight = uniform(-1, 1, size=shapes[i+1]).astype(self.dtype) / layer.neurons
            layer.bias = uniform(-1, 1, size=shapes[i+1][1]).astype(self.dtype) / layer.neurons

            # Other initialization
            # scale = np.sqrt(2/shapes[i+1][0])
            # layer.weight = np.random.normal(scale=scale, size=shapes[i+1]).astype(self.dtype)
            # layer.bias = np.zeros(shape=shapes[i+1][1], dtype=self.dtype)

            layer.position = i
            layer.mbatch_size = self.mbatch_size
            layer.eta = self.eta
            layer.lmbda = self.lmbda
            layer.n_params = layer.weight.size + layer.bias.size

        self.n_params = np.sum([layer.n_params for layer in self.layers[:-1]])
        print('Layers: %d\nParameters: %d\nLearning rate: %.3f\nL2 regularization: %.1f\n'
              'Precision: %s\n' % (self.n_layers, self.n_params, self.eta, self.lmbda, self.dtype))

    # ---------------------------------
    def train(self, imgs, labels, keep=True, print_errors=False, visualize=False):
        """Train the net by *minibatch SGD*. Evaluate the net against the test data, each epoch.

        Each image is a 1-array of shape (m,) from [-1, 1]**m, where *m* is the amount of pixels
        Each label is a 1-array of shape (p,) from {0, 1}**p, where *p* is the amount of labels

        Args:
            imgs (ndarray): An (n, m)-array of n flattened images
            labels (ndarray: An (n, p)-array of n labels
            keep (bool, optional): Save the weights/biases
            print_errors (bool, optional): Print the training, validation, and test errors
            visualize (bool, optional): Visualize how weights change using the vispy library (GPU)

        Returns:
            None
        """
        n_examples = imgs.shape[0]              # Total number of examples!
        n_train = int(5/7 * n_examples)        # Examples for training (weight/bias tuning)
        n_valid = (n_examples - n_train) // 2   # Examples for validation (hyperparameter tuning)
        n_test = (n_examples - n_train) // 2    # Examples for testing
        mbatches = n_train // self.mbatch_size  # Total number of minibatches!
        print('Epochs: %d\nExamples: %d\nTraining: %d\nValidation: %d\nTest: %d\nMinibatches: %d\n'
              % (self.epochs, n_examples, n_train, n_valid, n_test, mbatches))

        imgs = imgs.astype(self.dtype)
        labels = labels.astype(self.dtype)
        train_imgs = imgs[0:n_train]
        train_labels = labels[0:n_train]
        valid_imgs = imgs[n_train:n_train+n_valid]
        valid_labels = labels[n_train:n_train+n_valid]
        test_imgs = imgs[n_train+n_valid:n_train+n_valid+n_test]
        test_labels = labels[n_train+n_valid:n_train+n_valid+n_test]

        for layer in self.layers[:-1]:
            layer.n_train = n_train

        if visualize:
            self.visualize()

        for epoch in range(self.epochs):
            train_imgs, train_labels = shuffle_dataset(imgs=train_imgs, labels=train_labels)

            for i in range(0, n_train, self.mbatch_size):
                imgs = train_imgs[i+0:i+self.mbatch_size]
                labels = train_labels[i+0:i+self.mbatch_size]

                # The juicy stuff happens here!
                self.forward_propagation(imgs=imgs)
                self.backward_propagation(labels=labels)
                self.gradient_descent()

            # self.update_eta(epoch=epoch)

            # Predictions of our network on the various sets! Not part of training
            if print_errors:
                self.print_error(imgs=valid_imgs, labels=valid_labels, epoch=epoch, verbose=False)
                # self.print_error(imgs=test_imgs, labels=test_labels, epoch=epoch, verbose=False)
            else:
                print('Epoch %d.' % epoch)

        if not print_errors:
            self.print_error(imgs=valid_imgs, labels=valid_labels, epoch=self.epochs-1)
        print('Network trained for %d epoch(s)!' % (self.epochs))

        if keep:
            for i, layer in enumerate(self.layers[:-1]):
                save_ndarray(layer.weight, 'weight' + str(i))
                save_ndarray(layer.bias, 'bias' + str(i))

    def forward_propagation(self, imgs):
        """Call the forward_propagation method on each computational non-output layer!"""
        # Always first layer
        self.layers[0].forward_propagation(x=imgs)

        # Hidden layers!
        for i in range(len(self.layers) - 2):
            self.layers[i+1].forward_propagation(x=self.layers[i].a)

    def backward_propagation(self, labels):
        """In the last layer, compute the errors using the cost function and backpropagate them.
        In the intermediate (aka. "hidden") layers, update the nablas and compute the errors.
        In the first layer (aka. zeroth layer, input layer), update the nablas.

        Notes
        -----

        Backpropagation is a O(n) algorithm to compute the gradient of the cost function C_x. The
        gradient of C_x depends on every bias and weight, so naively computing it is O(n**2).

        This gradient encodes information about how changing the biases and weights in a network
        changes the output of C_x. The goal of backpropagation is to compute the list of partial
        derivatives of C_x with respect to every bias and weight in the net!

        A function F of n variables has n partial derivatives. The n-list of all these partial
        derivatives is called the *gradient of F*, or *nabla F*.
        """
        # Always last layer
        self.layers[-1].backpropagate_error(z=self.layers[-2].z, activation=self.layers[-2].a,
                                            target=labels)

        # All intermediate (aka. hidden) layers
        for i in range(len(self.layers)-1, 1, -1):
            self.layers[i-1].update_nabla(delta=self.layers[i].delta)
            self.layers[i-1].backpropagate_error(delta=self.layers[i].delta)

        # Always input layer
        self.layers[0].update_nabla(delta=self.layers[1].delta)

    def gradient_descent(self):
        """Apply (minibatch) gradient descent to every non-output layer!"""
        for layer in reversed(self.layers[:-1]):
            layer.gradient_descent()

    # ---------------------------------
    def update_eta(self, epoch):
        """Modify the learning rate *eta* after enough epochs have gone by."""
        quarter_epochs = self.epochs // 4
        if quarter_epochs >= 25 and (epoch+1) % quarter_epochs == 0:
            self.eta *= 0.5
            self.lmbda = self.eta**-1
            print('Update! New eta and lambda: %.6f, %.6f' % (self.eta, self.lmbda))

    def results(self, x):
        """Main function to *read* new inputs! *Reading* means performing a full forward pass with
        pre-trained weights/biases! Return the *excitation* (ie. activation) of each neuron (in
        the output layer) when "reading" an input. The most excited of the output neurons will
        correspond to the input the net thinks it's looking at.

        Args:
            x (ndarray): Either a single (flattened, of course) input (1-array), or a collection
                of inputs (2-array).

        Returns:
            ndarray: Binary 1-array of with length equal to the amount of output neurons
        """
        activations = [x]
        for i in range(len(self.layers) - 1):
            z = np.dot(activations[i], self.layers[i].weight) + self.layers[i].bias
            a = self.layers[i].activation.f(z)
            activations.append(a)
        return activations[-1]

    def print_error(self, imgs, labels, epoch, verbose=False):
        """Print the net's errors when reading a set of images/labels, by counting the amount of
        correctly read inputs. The "interpreted input" is the index of the neuron in the final
        layer with the highest activation."""
        n_imgs = imgs.shape[0]
        test_results = [(np.argmax(self.results(img)), np.argmax(label))
                        for img, label in zip(imgs, labels)]
        successes = np.sum(int(x == y) for x, y in test_results)

        # Print detailed data regarding each evaluation!!!
        if verbose:
            for img, label in zip(imgs, labels):
                excitation = self.results(img)
                global_excitation = np.sum(excitation)
                predicted_input = np.argmax(excitation)
                confidence = 100 * excitation[predicted_input] / global_excitation
                print(excitation,
                      '%.1f' % global_excitation,
                      '%d (%.1f%%)' % (predicted_input, confidence),
                      np.argmax(label), sep='  |  ')

        error = 100 * (1 - successes / n_imgs)
        print('Epoch %d. %0.1f%% error, %d/%d' % (epoch + 1, error, successes, n_imgs))
        return error  # Return error for documentation purposes!

    def visualize(self):
        """Visualize the evolution of weights using the vispy GPU-accerelated library!"""
        from vispy import app
        from visualization import Canvas, vertex_shader, fragment_shader, process_weight_matrix
        # Window
        W = 500
        SIZE = np.array([W, W])
        pos = [0, 0]

        data = np.random.randn(784, 100)
        data = process_weight_matrix(data=data, window_size=SIZE)

        position = np.array([[-1, -1], [-1, 0], [0, -1], [0, 0]], dtype=np.float32)
        textcoord = np.array([[0, 1], [0, 0], [1, 1], [1, 0]], dtype=np.float32)

        canvas = Canvas(size=SIZE, position=pos, keys='interactive')
        canvas.show()
        app.run()


def shuffle_dataset(imgs, labels):
    """Shuffle a dataset made up of images and labels, in a way that preserves their original
    relative ordering. The shuffling does *not* happen in place! Arrays are shuffled across axis
    0, not directly (which is slow), but using shuffled indices.

    Args:
        imgs (ndarray): An n-array interpreted as a list of (n-1)-arrays
        labels (ndarray): An n-array interpreted as a list of (n-1)-arrays

    Returns:
        ndarray: An n-array, shuffled across axis 0
        ndarray: An n-array, shuffled across axis 0
    """
    shuffled_imgs, shuffled_labels = np.zeros_like(imgs), np.zeros_like(labels)
    indices = np.arange(imgs.shape[0])
    np.random.shuffle(indices)

    for i, index in enumerate(indices):
        shuffled_imgs[i] = imgs[index]
        shuffled_labels[i] = labels[index]

    return shuffled_imgs, shuffled_labels


def _filter_img(img):
    """Pre-process image (turn to filter, de-noise, detect edges, etc.)!"""
    import cv2
    gaussian_kernel = np.array([[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                                [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                                [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                                [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                                [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]])

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    img = 255 - img  # Invert color!
    img = cv2.filter2D(img, -1, gaussian_kernel)
    return img


###################################################################################################
"""Tests!"""

if __name__ == '__main__':
    # Load digits and set data type
    mnist_imgs, mnist_labels = load_ndarray('mnist1.npy')

    # Initialize layers
    fc0 = PerceptronLayer(neurons=784, activation=ReLU)
    fc1 = PerceptronLayer(neurons=200, activation=ReLU)
    # fc2 = PerceptronLayer(neurons=30)
    loss = SoftmaxLossLayer(neurons=10)
    layers = [fc0, fc1, loss]

    # Initialize net
    net = Net(layers=layers, epochs=20, eta=0.5, mbatch_size=100)

    # Train net
    net.train(imgs=mnist_imgs, labels=mnist_labels, keep=False, print_errors=True, visualize=True)
