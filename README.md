![DeepUnicorns](deepunicorns.png)

**DeepUnicorns** is a Python implementation of various **deep learning** algorithms. It can build [neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) that are [modular](https://en.wikipedia.org/wiki/Modularity) in _layers_, _activation functions_, and _cost functions_ (aka _loss functions_).

It's __goals__ are:

1. To illustrate the main ideas from __first principles__. The sole "black box" is [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication), but most (all?) NumPy backends use the `O**3` algorithm anyways!
1. To be modular
1. To be barebones
1. To be hackable
1. To be fast (rather, as fast as possible on a CPU!)

After having a firm grasp of the primitives, people should move on to the highly optimized libraries (eg. [Theano](https://github.com/Theano/Theano), [Caffe](https://github.com/BVLC/caffe), [cuDNN](https://github.com/hannes-brt/cudnn-python-wrappers)).

Unicorns only admits a CPU backend.

Inspired by: [Michael Nielsen](http://neuralnetworksanddeeplearning.com), [Mocha](https://github.com/pluskid/Mocha.jl)


### Performance

**MNIST:** Unicorns gets 3% error in 10 epochs (6 seconds on a 14-core CPU @ 2.2 GHz) with 50 hidden units (1 hidden layer), and 1.5% error in 20 epochs (20 seconds) with 200 hidden units (1 hidden layer).

**CIFAR-10:** TODO!


### Supported functions/layers

**Layers**

1. Perceptron layer (aka. inner product layer, fully connected layer)
1. Softmax loss layer
1. Convolution layer — TODO!
1. Pooling layer — TODO!
1. Autoencoder layer — TODO!

**Activation functions**

1. ReLU
1. Leaky ReLU
1. Softplus
1. Sigmoid
1. Tanh

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
NumPy (even better if it's compiled against [MKL](https://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl)!)


### Source style

Code uses 99-character lines to allow for expressive variable names  
Each line of code is intended to contain a single atomic idea (the Raymond Hettinger rule!)
