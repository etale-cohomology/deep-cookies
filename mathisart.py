# Copyright (c) 2016, Diego Alonso Cortez, diego@mathisart.org
# Code released under the BSD 2-Clause License

"""# mathIsART utilities!

**mathisart.util v0.0.5**

Welcome to the mathIsART utilities library! Just some helper functions, nothing fancy.

## Index of functions:

**Name**                **Short description**
printz                  A printing format I use way too much...
properties              See basic metadata of objects, specially arrays/lists
timeit                  Time execution of code. Call using `with`
silence                 A bazooka that silences all errors inside a `with` block
cartesian               Cartesian product of 1-arrays. Efficient implmementation
add_row                 NumPy's equivalent of: 'Add a row to your spreadsheet!'
add_col                 NumPy's equivalent of: 'Add a column to your spreadsheet!'
save                    Save a NumPy array as .npy or .7z
load                    Load a NumPy array, either from .npy or .7z
imshow                  Show images (using OpenCV)
palette                 A bunch of colors I like
hex_to_rgb              Convert a hex string to an RGB 3-tuple
rgb_to_hex              Convert an RGB 3-tuple to a hex string

## Index of classes:

Infix                   Ugly hack to get (non-Pythonic) infix operators
Bunch                   A "list" indexed by an arbitrary set (not just the natural numbers)
abstractstaticmethod    Call on abstract methods to force their concretizations to be static
singleton               Singleton design pattern, decorator form
Singleton               Singleton design pattern, metaclass form

## Definitions/notation:

Let *n* be a **nonnegative integer**. Then:

- An **n-list** is a **sequence with n elements**, indexed from 0 to n-1. Equivalently, it's an
n-dimensional vector, ie. an element of an n-dimensional vector space. In Python, an n-list may
be implemented as a `list()` or as a `numpy.ndarray` of shape `(n,)`
- An **n-array** is an **n-dimensional array**, meaning its **SHAPE** is an *n-dimensional vector*
- A **matrix** is a **2-array**

**Remarks**:

    - An n-list is a 1-array
    - Notice that an array of shape `(n,)` is *not* an n-array, but rather a 1-array. An
    n-dimensional array corresponds usually to a vector of much higher dimension. Eg. a 1080 x 1920
    matrix (corresponding to a Full HD grayscale image) is a 2-array because its *shape* is
    `(1080, 1920)` and so its *shape* is a 2-dimensional vector, but the matrix itself is a
    2,073,600-dimensional vector
"""

import numpy as np     # none(4x4, 2.5s), blosc(4x4,3.4s)
import inspect
import contextlib      # AWESOME with statements!
import shutil          # To check if something is in the PATH
import time            # For the timeit() function!
import os
import functools
import sys
import subprocess

# Try to import non-standard libraries
with contextlib.suppress(ImportError):
    import cv2  # mathisart.util.imshow uses this

np.set_printoptions(precision=2)


###################################################################################################
def printz(*args, sep=False):
    """Print each arg in a new line.

    Args:
        *args: Stuff to be printed
        sep (bool): Set to True to print a full-line divider between args!

    Returns:
        None
    """
    end = '\n' + 79*'-' + '\n\n'
    if sep:
        print(*args, sep='\n' + 79*'-' + '\n', end=end)
    else:
        print(*args, sep='\n', end=end)


def properties(*objects, show=False):
    """A catch-all attribute-printing function. Input any object!

    Args:
        objects: Comma-separated objects
        show (bool, optional): Set to False to not print the object itself, nor its base!

    Returns:
        None

    TODO: make this less ugly, while keeping independent error-checking of each statement
    """
    for obj in objects:

        print('PROPERTIES of object:')
        print(type(obj))

        with silence():
            print('Shape:\t\t', obj.shape)
        with silence():
            print('dtype:\t\t', obj.dtype)
        with silence():
            print('Size:\t\t', obj.size)
        with silence():
            print('Bytes:\t\t', obj.nbytes)
        with silence():
            print('Sys bytes:\t', sys.getsizeof(obj))
        with silence():
            print('Dimension:\t', obj.ndim)
        with silence():
            print('Itemsize:\t', obj.itemsize)
        with silence():
            print('Strides:\t', obj.strides)
        if show:
            with silence():
                print('Base:\t\t', obj.base)
        with silence():
            print('Data:\t\t', obj.data)
        with silence():
            print('CPU ID:\t\t', id(obj))
        with silence():
            print('GPU ID:\t\t', obj.ptr)
        with silence():
            print('dbuffer:\t', obj.__array_interface__['data'][0])
        with silence():
            print('C_contig:\t', obj.flags.c_contiguous)
        with silence():
            print('F_contig:\t', obj.flags.f_contiguous)

        with silence():
            print('Signature:\t', inspect.signature(obj))
        with silence():
            print('MRO:\t\t', obj.__mro__)

        with silence():
            print('Length:\t\t', len(obj))
        with silence():
            print('Length:\t\t', obj.length())
        with silence():
            print('Length:\t\t', obj.length)

        if show:
            print('Object:', obj)
        print(79*'-', '\n')


@contextlib.contextmanager  # Create factory function for *with* context managers
def timeit():
    """Time execution of code, based on time.clock(). Call using a *with* statement!

    Examples:
        with timeit():
            a @ a
    """
    start = time.clock()
    yield  # The decorated func must return a generator-iterator
    print('Operation took: %0.6f seconds!' % (time.clock() - start))


def silence(error=Exception):
    """Wrapper for contextlib.suppress(). Call using a *with* statement!

    Examples:
        with supress():
            2.shape
    """
    return contextlib.suppress(error)


def cartesian(*arrays, out=None):
    """Generate the n-fold Cartesian product of n 1-arrays. Efficient implementation.

    Args:
        *arrays: Comma-separated 1-arrays

    Returns:
        ndarray: A 2-array of shape (s1 * s2 * ... * s3, n)

    Source:
        http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def add_row(grid, row):
    """Expand a 2-array across axis 0 (eg. from shape (100, 15) to shape (101, 15)). Excel lingo:
    'add a new row to your spreadsheet'.

    Args:
        grid (ndarray): The original grid to which you want to add a row
        row (ndarray): Row to be appended

    Returns:
        ndarray
    """
    if grid.ndim != 2:
        print('Base grid is not a 2-array!')
        return
    elif row.ndim > 2:  # Notice 1 or 2 dimensions can work
        print('New row is too high-dimensional!')
        return

    row = row.row(1, row.size)  # From shape (n, ) to (1, n)
    return np.vstack((grid, row))


def add_col(grid, col):
    """Expand a 2-array across axis 1 —eg. from shape (100, 15) to shape (100, 16)—. Excel lingo:
    'add a new column to your spreadsheet'

    Args:
        grid (ndarray): The original grid to which you want to add a column
        col (ndarray): The column to be appended

    Returns:
        ndarray
    """
    if grid.ndim != 2:
        print('Base grid is not a 2-array!')
        return
    elif col.ndim > 2:  # Notice 1 or 2 dimensions can work
        print('New column is too high-dimensional!')
        return

    col = col.reshape(col.size, 1)  # From shape (n, ) to (n, 1)
    return np.hstack((grid, col))


def save_ndarray(ndarray, filename, compress=False):
    """Store a single NumPy array on an .npy binary file. Tries to keep allow_pickle at False.
    Optional multithreaded compression using 7-zip (needs to be in the PATH).

    Args:
        ndarray (ndarray): Array to be stored on disk.
        filename (str): The file's name on disk. No need for .npy extension
        compress (bool, optional): Set to True to save array as .npy and then compress it to .7z

    Returns:
        None
    """
    if filename.endswith('.npy'):
        file_sans = filename[:-4]
    else:
        file_sans = filename  # String assignments are always copies!
        filename += '.npy'
    print('Saving {0!s}-array ({1:,} bytes) to disk...' .format(ndarray.shape, ndarray.nbytes))

    if compress:
        status_compression = 'compressed'
        if shutil.which('7z') is None:
            raise FileNotFoundError('7z not found on the PATH!')

        try:
            np.save(filename, ndarray, allow_pickle=False)
            status_pickle = 'unpickled'
        except ValueError:
            np.save(filename, ndarray, allow_pickle=True)
            status_pickle = 'pickled'

        subprocess.Popen('7z a ' + file_sans + '.7z ' + file_sans + '.npy -mmt', shell=True).wait()
        os.remove(filename)
        filename = file_sans + '.7z '
        print()

    else:
        status_compression = 'uncompressed'
        try:
            np.save(filename, ndarray, allow_pickle=False)
            status_pickle = 'unpickled'
        except ValueError:
            np.save(filename, ndarray, allow_pickle=True)
            status_pickle = 'pickled'

    print('Successfully saved {0!s}-array ({1:,} bytes) as [{2!s}] in [{3}], [{4}] form!'
          .format(ndarray.shape, ndarray.nbytes, filename, status_compression, status_pickle))


def load_ndarray(filename):
    """Load a NumPy array from disk into memory, with extension .npy or .7z. If no extension is
    included in the argument, first assume it's .npy, then .7z.

    Args:
        filename (str): Name of the NumPy array (in disk).

    Returns:
        ndarray
    """
    if filename.endswith('.npy'):
        compress = False
        status = 'uncompressed'  # Everything OK!
    elif filename.endswith('.7z'):
        compress = True
        status = 'compressed'
    else:
        file_npy = filename + '.npy'
        if file_npy in os.listdir():
            filename = file_npy
            compress = False
            status = 'uncompressed'
        else:
            file_7z = filename + '.7z'
            if file_7z in os.listdir():
                filename = file_7z
                compress = True
                status = 'compressed'
            else:
                raise FileNotFoundError

    # ---------------------------------
    size = os.stat(filename).st_size
    print('Loading {0:,} [{1}] bytes from disk... File: {2}'
          .format(size, status, filename))

    if compress:
        if shutil.which('7z') is None:
            raise FileNotFoundError('7z not found on the PATH!')
        subprocess.Popen('7z e ' + filename + ' -mmt', shell=True).wait()
        ndarray = np.load(filename[:-3] + '.npy')
    else:
        ndarray = np.load(filename)
        print('Succesfully loaded {0!s}-array ({1:,} bytes) from {2}!'
              .format(ndarray.shape, size, filename))

    return ndarray


def imshow(*images):
    """Show images! Uses OpenCV."""

    print('Showing %d image(s)...' % (len(images)))

    for i, img in enumerate(images):
        print('Image %d of shape %s, dtype %s.' % (i + 1, img.shape, img.dtype))
        print(img, '\n')
        cv2.imshow('Image ' + str(i + 1), img)
    print('Press Q to close image(s)...')
    cv2.waitKey(0)


def palette(color=None, index=0, space='rgba'):
    """A collection of colors! Input color name, return color RGBA (default). Supported color
    spaces: RGB, RGBA, hex.

    Examples:
        palette()  # Random color from main collection
        palette('red')  # Red color, main collection
        palette('mono2', 2)  # third color, mono2 collection

    TODO: Add support for more color spaces.
    """

    # Main collection, indexed by name!
    single = Bunch(
        main='#2196f3',
        alt='#0088cb',
        black='#222',
        red='#e74c3c',
        green='#2ecc71',
        blue='#3498db',
        purple='#9b59b6',
        darkblue='#34495e',
        orange='#e67e22',
        yellow='#f1c40f')

    # Other collections, indexed by number!
    multi = Bunch(
        mono1=['#00557f', '#006698', '#0077b2', '#0088cb', '#0099e5',
               '#00aafe', '#19b3ff'],
        mono2=['#00567f', '#4cc5ff', '#00Acff', '#26627f'],
        analogous=['#0dafff', '#0c9fe8', '#00abff', '#0c9fe8'],
        darker1=['#007ab6', '#006ca2', '#005f8e', '#005179', '#004465',
                 '#003651', '#00283c', '#001b28', '#000d14'],
        darker2=['#0c83e1', '#0b79ce', '#0a6dbb', '#0962a9', '#085796',
                 '#074c84', '#064271', '#05375e'],
        lighter1=['#1993d0', '#329fd5', '#4cabda', '#66b7df', '#7fc3e5',
                  '#99cfea', '#b2dbef', '#cce7f4', '#e5f3f9'],
        lighter2=['#eef7fe', '#dbeefd', '#c8e5fc', '#b6dcfb', '#a3d4fa',
                  '#91cbf9', '#7ec2f8', '#6bb9f7', '#59b0f6'])

    if color is None:  # If you don't know what you want, anything is good
        color = np.random.choice(list(single.values()))
    else:
        try:  # First look on the first palette
            color = single[color]
        except KeyError:  # Then look on the second one
            color = multi[color][index]

    space = space.lower()  # Make string a tad safer
    if space == 'rgba':
        return hex_to_rgba(color)
    elif space == 'hex':
        return color
    elif space == 'rgb':
        return hex_to_rgb(color)
    elif space in {'hsv', 'hsl', 'lab'}:
        raise NotImplementedError  # TODO
    else:
        raise AttributeError('Invalid color space!')


def hex_to_rgb(color):
    """Hex string to RGB 3-list in one fell swoop! Like Macduff."""
    hx = color.lstrip('#')  # Remove the hash symbol!

    if len(hx) == 3:  # If hex is in shortform, expand it to fullform
        hx = hx[0]*2 + hx[1]*2 + hx[2]*2

    color = [int(hx[i:i+2], 16) for i in (0, 2, 4)]
    return np.array(color)


def hex_to_rgba(hx):
    """Hex string to RGB 4-list in one fell swoop!"""
    return np.append(hex_to_rgb(hx), 1)


def rgb_to_hex(color):
    """RGB 3-list or RGBA 4-list to hex string! But ignore alpha channel."""
    color = tuple((color[0], color[1], color[2]))  # remove alpha channel!
    return '#%02x%02x%02x' % color


###################################################################################################
"""Experimental functions!"""


def cache(function):
    """Cache results of static functions (ie. function that always return the same value)! Call as
    a decorator!

    Source: Raymond Hettinger
        https://www.youtube.com/watch?v=OSGv2VnC0go

    Example:
        @cache
        def web_lookup(url):
            return urllib.urlopen(url).read()

    TODO: make it work
    """
    saved = {}

    # functools.wraps allows decorated functions to keep their own docstrings!
    @functools.wraps(function)
    def wrapper(*args):
        if args in saved:
            return wrapper(*args)
        result = function(*args)
        saved[args] = result
        return result
    return wrapper


def os_filenames(path, extension='jpg'):
    """Returns a list of filenames of a given extension in a directory."""
    ext = extension.lower()  # Sanitize a bit
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def bytes_from_file(filename, chunksize=8192):
    """Read binary files efficiently!

    Source:
        http://stackoverflow.com/questions/1035340/reading-binary-file-in-python-and-looping-over-each-byte
    """
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break


def invert_color(img):
    return 255 - img


def mapz(function, list_of_args):
    """Just the *map* function with the *list* function applied afterwards. Ah, GwR..."""
    return list(map(function, list_of_args))


###################################################################################################
class Infix:
    """Definition of an infix operator class. Call as a decorator. The new operator can't be a
    Python operator. '|' is the operator with the lowest priority still practical to use.

    Examples:

        # 2-fold product
        @Infix
        def x(a, b):
            return a * b
        c = 2 |x| 4
        print(c)
        COUT: 8

        # Functional programming
        def curry(f,x):
            def curried_function(*args, **kw):
                return f(*((x,)+args),**kw)
            return curried_function
        curry=Infix(curry)

        # More stuff
        add5= operator.add |curry| 5
        print add5(6)
        COUT: 11

    Source:
        http://code.activestate.com/recipes/384122/
    """

    def __init__(self, function):
        self.function = function

    def __or__(self, other):
        return self.function(other)

    def __ror__(self, other):
        # partial() turns a func of n args into func of n-k args
        return Infix(functools.partial(self.function, other))

    def __call__(self, value1, value2):
        return self.function(value1, value2)


class Bunch(dict):
    """A dict allowing access using method notation and dict notation! Sexier printing for the
    whole dict.

    Examples:
        data = Bunch(bananas=10, unicorns=1337)
        print(data.bananas)

    Source:
        http://code.activestate.com/recipes/52308/

    TODO: accept zip argument, like a standard dict. Why doesn't it already...?
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)  # Access a Bunch object like a dict!
        self.__dict__ = self

    def __str__(self):
        """When we print self, we get a nice string representation!"""
        state = ["%s=%r" % (attribute, value)
                 for (attribute, value)
                 in self.__dict__.items()]
        return '\n'.join(state)


class abstractstaticmethod(staticmethod):
    """Use this decorator on methods of an abstract base class to force them to be static on every
    subclass! Now there's *no* need to *also* call the @staticmethod decorator on them!

    Examples:
        class MyAbstractClass(metaclass=abc.ABCMeta):
            @abstractstaticmethod
            def foo():

        class Child(MyAbstractClass):
            # Now this method will be static even *without* the @staticmethod decorator!
            def foo():
                print(5)

        child = Child
        child.foo()

    Source:
        http://stackoverflow.com/questions/4474395/staticmethod-and-abc-abstractmethod-will-it-blend
    """
    __slots__ = ()

    def __init__(self, function):
        super(abstractstaticmethod, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


def singleton(cls):
    """The singleton decorator! Decorate your classes with this to make them singletons.

    Examples:
        @singleton
        class IWantThisClassToBeASingleton:
            pass

        x = IWantToBeASingleton()
        y = IWantToBeASingleton()
        print(id(x) == id(y))  # Prints True!

    Source:
        http://intermediatepythonista.com/metaclasses-abc-class-decorators
    """
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance


class Singleton(type):
    """The Singleton metaclass! Inherit from this class to become a singleton!

    Examples:
        class IWantToBeASingleton(metaclass=Singleton):
            pass

        x = IWantToBeASingleton()
        y = IWantToBeASingleton()
        print(id(x) == id(y))  # Prints True!
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


###################################################################################################
"""# mathisart.triangles v0.0.1

Welcome to the mathIsART Triangles module! Some helper functions for computation triangles.
Initially intended for Euclidean 2-space, eventually it should include general n-dimensional
inner product spaces over (non)Archimedean fields.

"""

import math


def long_side(diagonal, aspect_ratio_long=16, aspect_ratio_short=9):
    numerator = diagonal**2 * aspect_ratio_long**2
    denominator = aspect_ratio_short**2 + aspect_ratio_long**2
    return math.sqrt(numerator / denominator)


def short_side(diagonal, aspect_ratio_long=16, aspect_ratio_short=9):
    numerator = diagonal**2 * aspect_ratio_short**2
    denominator = aspect_ratio_short**2 + aspect_ratio_long**2
    return math.sqrt(numerator / denominator)


###################################################################################################
"""Unit tests!!"""

# if __name__ == '__main__':
#     a = np.arange(100)
#     save(a, 'lala', compress=False)
#     save(a, 'lala.npy', compress=False)
#     save(a, 'lala', compress=True)
#     save(a, 'lala.npy', compress=True)
#     a = load('lala')
#     a = load('lala.npy')
#     a = load('lala.7z')
#     print(a)
