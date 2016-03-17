# Unicorns v0.0.1

**Unicorns** is a Python implementation of various **deep learning** algorithms. It can build neural networks that are modular in *layers*, *activation functions*, and *cost functions*. It is intended to illustrate the main ideas from first principles (its sole "black box" being matrix multiplication), as a companion to a series of upcoming tutorials. After having a firm grasp of the primitives, thoough, people should move on to the highly optimized libraries (eg. [Theano](https://github.com/Theano/Theano), [Caffe](https://github.com/BVLC/caffe), [cuDNN](https://github.com/hannes-brt/cudnn-python-wrappers)).

Currently Unicorns only admits a CPU backend. Future versions are expected to support GPUs, through the PyCUDA interface.

Inspired by: [Michael Nielsen](http://neuralnetworksanddeeplearning.com) [Mocha](https://github.com/pluskid/Mocha.jl)

### Performance

**MNIST:** Unicorns gets 3% error in 10 epochs (6 seconds on a 14-core CPU @ 2.2 GHz) with 50 hidden units (1 hidden layer), and 1.5% error in 20 epochs (20 seconds) with 200 hidden units (1 hidden layer).

**CIFAR-10:** Coming soon

### Supported functions/layers

**Layers**

1. Perceptron layer (aka. inner product layer, fully connected layer)
2. Softmax loss layer
3. Convolution layer — COMING SOON!
4. Polling layer — COMING SOON!
4. Autoencoder layer — COMING SOON!

**Activation functions**

1. ReLU
2. Leaky ReLU
3. Parametric ReLU — COMING SOON!
4. Softplus
5. Sigmoid
6. Tanh
7. Kumaraswamy — COMING SOON!

**Cost functions**

1. Cross-entropy
2. Mean squares

### Examples

**Minimal example: 2-layer fully connected feedforward net**

    # Before running this snippet, download and unzip mnist.7z to the same folder as your script

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



### Requirements

Python 3.x
NumPy
Matplotlib (optional, visualization)
Vispy (optional, GPU-accelerated visualization)

### Source style

Code uses 99-character lines to allow for expressive variable names  
Each line of code is intended to contain a single atomic idea (the Raymond Hettinger rule!)

### License

BSD 2-Clause License


## Tutorial for Feedforward Nets:

We're going to build our neural net like we build legos. Image we want to build a lego robot. Our lego robot has arms, and legs, and a head, and navigation system and a weapons system. If we had a team of engineers, we could assign one engineer to work on the arms, one to work on the legs, one to work on the weapons system, etc. And then we would join all of these modules together.

Modularity is good because it allows us to build very complicated things using simple constituents. Each basic constituent can be very, very simple (like a lego block), but, together, they can build something as complex as we want.

Our neural net will use three types of blocks:

1. Activation functions
2. Cost functions
3. Layers

Each activation function will have two functions attached to it.
Each cost function will have two functions attached to it.
Each layer will have one activation function and one cost function attached to it.
Each neural net will have as many layers as we want.

There's many types of activation functions, there's many types of cost functions, and there's many types of layers. Using these three degrees of freedom we can build many types of layers.

A neural net is —you guessed it— a **sequence of layers**. So, using different types of layers we can build neural nets of various architectures. This turns out to be insanely convenient, because by swapping parts around we can build nets that are vastly different and do all sorts of cool things, all using the same code.

Neural nets are classified by their architecture, and these are some common ones:

1. Feedforward nets
2. Convolutional nets
3. Recurrent nets
4. Deep belief nets

[]

In Python, this sort of modularity is achieved using three tools:

1. Functions
2. Classes
3. Metaclasses

A function is a collection of operations.
A class is a collection of functions (so, a collection of collections of operations).
A metaclass is a collection of classes (so, a collection of collections of collections of
operations).

A class is a blueprint for (Python) objects. A class can be seen an factory to mass-produce objects. Classes can be used define objects of arbitrary complexity, but, since objects will be our lego blocks, I think it's better to keep them as simple as possible. The complexity should stem from the way we stack various objects together, not from the objects themselves.
