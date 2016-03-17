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
