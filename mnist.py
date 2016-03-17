# Copyright (c) 2016, Diego Alonso Cortez, diego@mathisart.org
# Code released under the BSD 2-Clause License
# Inspired by:
#   Michael Nielsen
#       http://neuralnetworksanddeeplearning.com
#   Mocha
#       https://github.com/pluskid/Mocha.jl

from net import Net, Net2
from mathisart import printz, properties, save_ndarray, load_ndarray, timeit

import numpy as np
import cv2
import os
import subprocess

import scipy.signal
import itertools
from joblib import Parallel, delayed

###################################################################################################
"""Helper functions!"""


def _img_grid(imgs, grid_x=2, grid_y=2):
    """Arrange a list of (small) images on a grid, ready for display/save!

    Args:
        imgs (ndarray): A 3-array interpreted as a 1-array of 2-arrays. Eg. a 3-array of shape
        (10000, 500, 500) would be interpreted as a list of ten thousand 500x500 images
        grid_x: The width of the image grid, in number of images
        grid_y: The height of the image grid, in number of images

    Returns:
        ndarray: A numpy 2-array that can be read as a single image.
    """
    rows = []
    for i in range(grid_y):
        row = [*imgs[i*grid_x:(i+1)*grid_x]]
        row = np.concatenate(row, axis=1)
        rows.append(row)

    return np.concatenate(rows, axis=0)


def _vectorize(label):
    """Turn a training label into a 10-dimensional vector!"""
    vector = np.zeros(shape=10, dtype=np.uint8)
    vector[label] = 1
    return vector


def _create_2d_gaussian(dim, sigma):
    """Create a 2D gaussian kernel with standard deviation *sigma* and shape (dim, dim).

    Code by:

    Args:
        dim (int): Height of the gaussian kernel
        sigma (float): Standard deviation of the gaussian kernel
        :type sigma: float

    Returns:
        ndarray: A 2-array
    """
    kernel = np.zeros((dim, dim), dtype=np.float32)
    center = dim / 2
    variance = sigma**2
    coeff = 1 / (2 * variance)  # Normalization coefficient

    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val**2 + y_val**2
            denom = 2 * variance
            kernel[x, y] = coeff * np.exp(numerator/denom)**-1

    return kernel / np.sum(kernel)


def _elastic_transform(img, kernel_dim=13, sigma=6, alpha=36, negated=False):
    """Elasticly deform a 2D image by convolving with a gaussian kernel. Image should be square.

    Code by:

    Args:
        img (ndarray): Input image in 2-array (not 1-array!) form
        kernel (int): Height of the gaussian kernel
        sigma (float): Kernel's standard deviation
        alpha (float): A multiplicative factor post-convolution

    Returns:
        ndarray

    TODO: Try to vectorize operations with NumPy
    """
    import math
    result = np.zeros(img.shape)

    # Random displacement fields
    disp_field_x = np.array([[np.random.random_integers(-1, 1) for x in range(img.shape[0])]
                             for y in range(img.shape[1])]) * alpha
    disp_field_y = np.array([[np.random.random_integers(-1, 1) for x in range(img.shape[0])]
                             for y in range(img.shape[1])]) * alpha

    # Gaussian kernel
    kernel = _create_2d_gaussian(dim=kernel_dim, sigma=sigma)

    # Convolve the fields with the gaussian kernel
    disp_field_x = scipy.signal.convolve2d(disp_field_x, kernel)
    disp_field_y = scipy.signal.convolve2d(disp_field_y, kernel)

    # Deform by averaging each pixel with the adjacent 4 pixels based on displacement fields
    for row in range(img.shape[1]):
        for col in range(img.shape[0]):
            low_ii = row + math.floor(disp_field_x[row, col])
            high_ii = row + math.ceil(disp_field_x[row, col])

            low_jj = col + math.floor(disp_field_y[row, col])
            high_jj = col + math.ceil(disp_field_y[row, col])

            if low_ii < 0 or low_jj < 0 or high_ii >= img.shape[1]-1 or high_jj >= img.shape[0]-1:
                continue

            res = img[low_ii, low_jj]/4 + img[low_ii, high_jj]/4 + \
                img[high_ii, low_jj]/4 + img[high_ii, high_jj]/4

            result[row, col] = res

    return result


def _deskew(img, y_displacement=0):
    """Deskwew an image using moments."""
    moments = cv2.moments(img)  # Moments!
    skew = moments['mu11'] / moments['mu02']
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew], [0, 1, y_displacement]])
    img = cv2.warpAffine(img, M, img.shape, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# ---------------------------------
def deform_save(img, kernel_dim, sigma, alpha, i):
    """Do elastic deformation and save! Useful for embarassing paralellization with joblib."""
    distorted_img = _elastic_transform(img=img, kernel_dim=kernel_dim, sigma=sigma, alpha=alpha)
    cv2.imwrite('./images/z_elastic' + str(i) + '.png', distorted_img)


def _elastic_transform2(img, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in Simard, Steinkraus and Platt ("Best Practices
    for Convolutional Neural Networks applied to Visual Document Analysis", in Proc. of the
    International Conference on Document Analysis and Recognition, 2003").

    Code by: fmder
        https://gist.github.com/fmder/
    """
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = img.shape
    dx = gaussian_filter((random_state.rand(*shape)*2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape)*2 - 1), sigma, mode='constant', cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(img, indices, order=1).reshape(shape)


###################################################################################################
"""Loding functions!"""


def open_mnist(keep=False):
    """Decompress, unpickle, and load the MNIST dataset, which is a 3-list of 2-lists of n-lists,
    with n = 50,000 or 10,000. Then, convert it into a more manageable data type."""
    import gzip
    import pickle

    print('Loading original MNIST dataset...')

    # Reading gz is faster than reading 7z!
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        mnist_old = pickle.load(f, encoding='latin1')

    # Turns tuple into list, so we can edit
    mnist_old = [list(dataset) for dataset in mnist_old]

    # Vectorize the labels of all three sets!
    for i in range(3):
        mnist_old[i][1] = [_vectorize(label) for label in mnist_old[i][1]]

    train_data, valid_data, test_data = mnist_old
    n_train = len(train_data[0])
    n_valid = len(valid_data[0])
    n_test = len(test_data[0])
    sizes = np.cumsum([0, n_train, n_valid, n_test])

    # Our MNIST data type!
    n_mnist = n_train + n_valid + n_test  # 50,000 + 10,000 + 10,000
    mnist_imgs = np.zeros(shape=(n_mnist, 784), dtype=np.float32)
    mnist_labels = np.zeros(shape=(n_mnist, 10), dtype=np.uint8)

    for i, size in enumerate(sizes[:-1]):
        mnist_imgs[size:sizes[i+1]] = mnist_old[i][0]
        mnist_labels[size:sizes[i+1]] = mnist_old[i][1]

    if keep:
        mnist = np.empty(shape=(2,), dtype=object)
        mnist[0] = mnist_imgs
        mnist[1] = mnist_labels
        save_ndarray(mnist, 'mnist1')  # Size reading is buggy, but array is fine...!

    print('Done!')
    return mnist_imgs, mnist_labels


def load_mnist1(keep=True):
    """Load a pre-processes MNIST dataset, containing images and labels."""
    if 'mnist1.npy' not in os.listdir():
        subprocess.Popen('7z e mnist1.7z -mmt', shell=True).wait()
        print()
    mnist_imgs, mnist_labels = load_ndarray('mnist1')
    if not keep:
        os.remove('mnist1.npy')
    print('Digit examples loaded: %d' % mnist_imgs.shape[0])
    return mnist_imgs, mnist_labels


def load_mnist2(keep=True):
    """Load the MNIST 2, which contains 70,000 orginal digits plus expansions."""
    if 'mnist2.npy' not in os.listdir():
        subprocess.Popen('7z e mnist2.7z -mmt', shell=True).wait()
        print()
    mnist_imgs, mnist_labels = load_ndarray('mnist2')
    if not keep:
        os.remove('mnist2.npy')
    print('Digit examples loaded: %d' % mnist_imgs.shape[0])
    return mnist_imgs, mnist_labels


def expand_mnist(imgs, labels, expand_to=1e8):
    """Expand MNIST examples up to arbitraty sizes.

    Args:
        imgs:
        labels:
        expand_to (int, optional):

    Returns:
        ndarray: Old and new images
        ndarray: Labels for every image
    """
    expand_to = int(expand_to)
    print('Expanding MNIST... From %d examples to %d examples!' % (imgs.shape[0], expand_to))

    imgs = imgs.reshape(imgs.shape[0], 28, 28)

    # Helper lists
    elastic_kernel_dims = [11, 12, 13, 14, 15, 16, 17]
    elastic_sigmas = [3, 3.5, 4, 4.5, 5, 5.5, 6.5, 7, 7.5, 8, 8.5]
    elastic_alphas = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    dilation_x = [1, 2, 3, 4, 5, 6]
    dilation_y = [1, 2, 3, 4, 5, 6]
    erosion_x = [1, 2, 3]
    erosion_y = [1, 2, 3]

    # Elastic deformation kernel (continuous Gaussian kernel)
    elastic_ker = list(itertools.product(elastic_kernel_dims, elastic_sigmas, elastic_alphas))

    # Dilation kernel
    dilation_ker = [np.ones((i, j), np.uint8)
                    for i, j in itertools.product(dilation_x, dilation_y)]

    # Erosion kernel
    erosion_ker = [np.ones((i, j), np.uint8) for i, j in itertools.product(erosion_x, erosion_y)]

    imgs = _elastic_transform(imgs, *elastic_ker[0])
    # img = cv2.dilate(imgs[0], dilation_ker[0])
    # img = cv2.erode(imgs[0], erosion_ker[0])

    # Rotation
    pass

    # Zoom
    pass

    # Noise: Gauss!
    # noise_gauss = np.random.randn(*img.shape).astype(np.uint8)
    # img += noise_gauss

    # Noise: Poisson!
    # noise_poisson = np.random.poisson(50, size=img.shape).astype(np.uint8)
    # img += noise_poisson

    # Skew
    pass

    # Projection: Mercator
    pass

    # Projection: Others
    pass

    # Blur!
    gaussian_kernel = np.array([[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                                [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                                [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                                [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                                [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]])
    # img = cv2.filter2D(img, -1, gaussian_kernel)

    # ---------------------------------
    # product = itertools.product(elastic_kernel_dims, elastic_sigmas, elastic_alphas)
    # Parallel(n_jobs=14)(delayed(deform_save)(img, kernel_dim, sigma, alpha, i)
    #                     for i, (kernel_dim, sigma, alpha)
    #                     in enumerate(itertools.product(kernel_dims, sigmas, alphas)))
    # ---------------------------------

    return imgs, labels

# np.set_printoptions(linewidth=210, edgeitems=14, precision=2)
# mnist_imgs, mnist_labels = load_mnist2(keep=True)
# mnist_imgs, mnist_labels = expand_mnist(imgs=mnist_imgs[:2], labels=mnist_labels[:2])
# printz(mnist_imgs, mnist_labels)


###################################################################################################
"""Output functions!"""


def mnist_to_console(imgs, labels):
    """Print the MNIST metadata to the console!"""
    print('Images: %s  %s  %s' % (type(imgs), imgs.shape, imgs.dtype))
    print('Labels: %s  %s  %s' % (type(labels), labels.shape, labels.dtype))


def mnist_to_image(imgs, grid_x=80, grid_y=50):
    """Save the MNIST examples as image files on disk! Each image file will be a "grid" of digits,
    with grid_x * grid_y digits. Try to make this product a divisor of the total number of images.

    Args:
        imgs (ndarray): A 2-array of flattened digits! Eg. (50000, 784,)
        grid_x (int): The width of the output image grid, in number of images
        grid_y (int): The height of the output image grid, in number of images

    Returns:
        None
    """
    print('Saving %d digits... %d x %d digits per image' % (imgs.shape[0], grid_x, grid_y))
    grid = grid_x * grid_y  # Number of digit images per grid!

    # Unflatten images from (n, 784,) to (n, 28, 28)! (Recall 784 equals 28*28)
    n_imgs = imgs.shape[0]
    imgs = imgs.reshape(n_imgs, digit_x, digit_y)

    n_grids = int(n_imgs / grid)

    for i in range(n_grids):
        print('Saving image %d/%d' % (i+1, n_grids))
        file = _img_grid(imgs=imgs[i*grid:(i+1)*grid], grid_x=grid_x, grid_y=grid_y)
        filename = 'digit_images/mnist_' + str(i+1) + '.png'  # .png images will be smaller...
        cv2.imwrite(filename, 255 * file)


###################################################################################################
np.set_printoptions(precision=2, edgeitems=20, suppress=True, linewidth=210)  # 280

# Neurons in the net!
digit_x, digit_y = 28, 28  # Dimensions of each MNIST digit!
input_layer = digit_x * digit_y
hidden_layers = [1000]
output_layer = 10  # One for each digit!
neurons = [input_layer, *hidden_layers, output_layer]

# Net hyperparameters!
epochs = 50
minibatch = 50    # Minibatch size!
eta = 0.5          # 0.5 achieves 1% accuracy in 20 epochs (with 500 hidden neurons)
dropconnect = 0.0  # The proportion of *weights* that will "fall asleep" each epoch!

# Initialize net!
net = Net2(neurons=neurons, epochs=epochs, mbatch=minibatch, eta=eta, dropconn=dropconnect)

# Load original MNIST (mnist1)!
# mnist_imgs, mnist_labels = open_mnist(keep=True)  # 2s / 3s
mnist_imgs, mnist_labels = load_mnist1(keep=True)  # 2.5s / 2.5s (1s if already unzipped!)

# Load expanded MNIST (mnist2)!
# mnist_imgs, mnist_labels = load_mnist2(keep=True)

# Expand MNIST from tens of thousands to tens of millions!
# mnist_imgs, mnist_labels = expand_mnist(imgs=mnist_imgs, labels=mnist_labels)

# Train net!
net.train(imgs=mnist_imgs, labels=mnist_labels, keep=False, print_errors=True)


###################################################################################################
"""Use net!"""

# img = valid_imgs[2].reshape(28, 28)
# net.read_digits(img=img, verbose=True, keep=True)
# printz(img)

# Scan John's image!
# img = cv2.imread('images/eight.jpg', 0)  # 0 means grayscale! Not strictly needed
# net.read_digits(img=img, verbose=True, keep=True)
# printz(img)

# img = cv2.imread('images/z_0-28_0-28.jpg', 0)
# img2 = img / np.max(img)
# img3 = img2 * np.max(img)
# img3 = img3.astype(np.uint8)
# printz(img.shape, img2.shape, img3.shape)


###################################################################################################
"""Print MNIST to console (as array) and save to disk (as image)!"""

# mnist_to_console(imgs=mnist_imgs, labels=mnist_labels)
# mnist_to_image(mnist_imgs)


###################################################################################################
"""Elastic transform 2!!"""

# imgs, labels = load_mnist2()
# imgs = imgs.reshape(imgs.shape[0], 28, 28)[:40]

# for i, img in enumerate(imgs):
#     # Try alpha 30-100, sigma 3-7
#     img_B = _elastic_transform2(img=img, alpha=30, sigma=3)
#     cv2.imwrite('digit_images/img_' + str(1) + '_A.png', 255*img)
#     cv2.imwrite('digit_images/img_' + str(1) + '_B.png', 255*img_B)


# img = cv2.imread('digit_images/digits1.png', 0)
# img_e = _elastic_transform2(img=img, alpha=30, sigma=3)
# cv2.imwrite('digit_images/digits1_' + str(1) + 'e.png', 255*img_e)


###################################################################################################
# imgs, labels = load_mnist2()
# imgs = imgs.reshape(imgs.shape[0], 28, 28)[:40]

# for i, img in enumerate(imgs):
#     img_B = elastic_transform(img=img, alpha=36, sigma=20)
#     img = cv2.resize(img, (50, 50))
#     img_B = cv2.resize(img_B, (50, 50))
#     cv2.imwrite('digit_images/img_' + str(i) + '_A.png', 255*img)
#     cv2.imwrite('digit_images/img_' + str(i) + '_B.png', 255*img_B)
