import numpy as np
import random
import pickle, json, sys

""" 
Contents:
    Activation Functions
        -Sigmoid
        -ReLU
        -PReLU
        -ELU
        -SoftMax
        -Identity
    Cost Functions
        -Cross Entropy
        -Quadratic
    Layers
        -Input
        -Fully Connected
        -Convolution
        -Pooling
    Network


Written by Randy Zhang, July __, 2018
"""

###############################################################################################################
###############################################################################################################

'''
Begin: ACTIVATION FUNCTIONS
    :def f(z): activation function
    :def f_prime(z): derivative of activation function
'''


class Sigmoid(object):

    @staticmethod
    def f(z):
        """
        Sigmoid function
        :param z: input numpy array
        :return: sigmoid function applied to input
        """
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def f_prime(z):
        """
        Sigmoid Prime - the derivative of the sigmoid function
        :param z: input numpy array
        :return: derivative of sigmoid function applied to input
        """
        return Sigmoid.f(z) * (1 - Sigmoid.f(z))


class ReLU(object):
    # Rectified Linear Unit
    @staticmethod
    def f(z):
        return max(0, z)

    @staticmethod
    def f_prime(z):
        for i in range(len(z)):
            if z[i] < 0:
                z[i] = 0
            else:
                z[i] = 1

        return z


class PReLU(object):
    # Parametric ReLU
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def f(self, z):
        return max(self.alpha * z, z)

    def f_prime(self, z):
        for i in range(len(z)):
            if z[i] < 0:
                z[i] = self.alpha
            else:
                z[i] = 1

        return z


class ELU(object):
    # Exponential Linear Unit
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def f(self, z):
        return max(self.alpha * (np.exp(z) - 1), z)

    def f_prime(self, z):
        for i in range(len(z)):
            if z[i] < 0:
                z[i] = self.alpha * np.exp(z)
            else:
                z[i] = 1

        return z


class SoftMax(object):

    @staticmethod
    def f(z):
        return np.exp(z) / sum(np.exp(z))

    @staticmethod
    def f_prime(z):
        return SoftMax.f(z) - SoftMax.f(z) ** 2


class Identity(object):

    @staticmethod
    def f(z):
        return z

    @staticmethod
    def f_prime(z):
        return np.array([1] * len(z))


'''
Begin: COST FUNCTIONS
    :def fn(a, y): cost function with output a, expected output y
    :def delta(z, a, y): derivative of the cost function with 
                         respect to z (dCdz) given parameters
                         output a, expected output y, and input
                         to final layer activation function z.
    :def dCda(a, y): derivative of the cost function with respect
                     to a, the output of the final layer
'''


class CrossEntropyCost(object):
    """
    Removes the sigmoid prime term from the gradient of the first layer
    when used with the sigmoid activation function, thus increasing the
    speed of learning
    """
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        # :param z: unused, included for consistency with other deltas
        # activation function must be sigmoid for this derivative to hold
        return a - y

    @staticmethod
    def dCda(a, y):
        return (y - a) / (a**2 - a)


class QuadraticCost(object):
    # Also called the L2 cost function
    @staticmethod
    def fn(a, y):
        return np.linalg.norm(a - y) ** 2 / 2

    @staticmethod
    def delta(z, a, y, activation_f=Sigmoid):
        return (a - y) * activation_f.f_prime(z)

    @staticmethod
    def dCda(a, y):
        return a - y


'''
Begin: LAYERS
    :def connect_to(layer): connects the layer to the 
        previous layers in the network, generating weights
        for the connections between them
    :def forward_pass(a): computes the activation of this
        layer given the activation 'a' of the previous layer
    :def backward_pass(dCdz): given the partial dCdz from 
        the subsequent layer, returns 3 things - dCdW, dCdb
        for this layer, and dCd(z-1) to be used in the 
        computation for the previous layer
    :def reweight(dCdW, dCdb, eta, m, L1dn, L2dn): re-weights 
        the weights and biases given dCdW, dCdb, the 
        learning rate eta, the number of examples used in
        this mini_batch m, and the constants of L1 and L2
        regularization L1dn and L2dn 
'''


class InputLayer(object):
    """
    Input layer has no 'connect_to' function as there
    is no previous layer to connect to. It also has
    no 'backward_pass' function for the same reason.
    As this layer has no modifiable parameters, the
    'reweight' function has been removed as well
    """
    def __init__(self, size):
        self.size = size
        self.a = None

    def forward_pass(self, a):
        self.a = a
        return a


class FullyConnectedLayer(object):
    """
    Standard dense fully connected layer
    """
    def __init__(self, size, activation_f=Sigmoid):
        self.activation_f = activation_f
        self.size = size
        self.biases = np.random.randn(size, 1)
        self.weights = None
        self.a = None              # f(z)
        self.z = None              # a0*w + b
        self.a0 = None             # previous layer's activation

    def connect_to(self, layer):
        self.weights = np.random.randn(self.size, layer.size) / np.sqrt(layer.size)

    def forward_pass(self, a):
        self.a0 = a
        self.z = np.dot(self.weights, a) + self.biases
        self.a = self.activation_f.f(self.z)
        return self.a

    def backward_pass(self, dCdz):
        dCdb = dCdz * self.activation_f.f_prime(self.z)
        dCdW = np.dot(dCdb, self.a0.transpose())
        dCdz = np.dot(np.transpose(self.weights), dCdz)
        return dCdz, dCdW, dCdb

    def reweight(self, dCdW, dCdb, eta, m, L1dn=0.0, L2dn=0.0):
        self.biases = self.biases - eta/m * dCdb
        self.weights = (1 - eta * L2dn) * self.weights - eta/m * dCdW - eta * L1dn


# ------------------------------------------------------------
# IMAGE PROCESSING LAYERS
# ------------------------------------------------------------


def convolve(img, img_size, filter_in, filter_size, stride, padding='same'):
    """
    Basic convolution operation on an image img with filter filter_in
    :param img: the image to be convolved, passed as a
                vector of pixel values
    :param img_size: tuple containing (length, width) of
                     the image
    :param filter_in: the filter used in the convolution
                      passed as a vector of weights
    :param filter_size: tuple containing (length, width)
                        of the filter
    :param stride: the step size of the filter
    :param padding: used to determine a tuple (l_pad,
                    w_pad) used to pad the edges of the
                    image with 0-pixels in order to fix
                    the size of the output convolution
    :return: the convolution of img with filter_in
    """
    convolution = []
    if padding == 'same':
        padding = filter_size
    elif padding == 'none':
        padding = (0, 0)
    elif padding == 'full':
        padding = (int(filter_size[0]*3/2), int(filter_size[1]*3/2))

    for i in range(int((filter_size[0] - padding[0])/2),
                   img_size[0] - int((filter_size[0] - padding[0] - 1)/2), stride):
        for j in range(int((filter_size[1] - padding[1])/2),
                       img_size[1] - int((filter_size[1] - padding[1] - 1)/2), stride):
            img_block = []
            for k1 in range(int(-filter_size[0]/2), int((filter_size[0] + 1)/2)):
                for k2 in range(int(-filter_size[1]/2), int((filter_size[1] + 1)/2)):
                    # if out of bounds, pad with 0-pixels
                    if i + k1 < 0 or i + k1 >= img_size[0] or j + k2 < 0 or j + k2 >= img_size[1]:
                        img_block.append(0)
                    else:
                        img_block.append(img[(i + k1) * img_size[1] + j + k2])
            img_block = np.array(img_block).reshape(1, -1)
            out_point = np.dot(img_block, filter_in)
            convolution.append(out_point)

    conv_size = (img_size[0] - int((filter_size[0] - padding[0] - 1)/2) - int((filter_size[0] - padding[0])/2),
                 img_size[1] - int((filter_size[1] - padding[1] - 1)/2) - int((filter_size[1] - padding[1])/2))

    return np.array(convolution).reshape(-1, 1), conv_size


class ConvolutionLayer(object):

    def __init__(self, depth, img_size, filter_size=(3, 3),
                 stride=1, padding='same', activation_f=Sigmoid):
        """
        For stride != 1, use padding='none'
        :param depth: number of filters per image in the
                      previous layer
        :param img_size: tuple (length, width)
        :param filter_size: tuple (length, width)
        :param stride: number of pixels the filter moves
                       when creating the feature map
                       mathematically, stride = t
        :param padding: the type of padding to be used in
                        the convolutions - 'same' or 'none'
        :param activation_f: the activation function to be
                             applied to the convolution
        """
        self.depth = depth
        self.img_size = img_size
        self.filter_size = filter_size
        self.stride = stride
        self.a = None
        self.z = None
        self.a0 = None
        self.prev_layer = None
        self.conv_size = None
        self.padding = padding
        self.activation_f = activation_f

        self.size = (int((img_size[0] - filter_size[0])/stride) + 1) * (
                     int((img_size[1] - filter_size[1])/stride) + 1) * depth

        # one bias for each filter
        self.biases = np.random.randn(depth, 1)
        # one set of weights for each filter
        self.weights = [np.random.randn(filter_size[0] * filter_size[1], 1) /
                        np.sqrt(filter_size[0] * filter_size[1]) for i in range(depth)]

    def connect_to(self, layer):
        # Nothing to do here
        return

    def forward_pass(self, a):
        self.a0 = a
        img_out = []
        for filtr in range(self.depth):
            first = True

            # convolve over all input images and output the sum
            for i in range(0, len(a), self.img_size[0] * self.img_size[1]):
                convolution_i, conv_size = convolve(a[i:i + self.img_size[0] * self.img_size[1]], self.img_size,
                                                    self.weights[filtr], self.filter_size, self.stride,
                                                    padding=self.padding)
                if first:
                    convolution = convolution_i
                else:
                    convolution += convolution_i

            img_out.append(convolution)
        self.conv_size = conv_size

        self.z = np.array(img_out).reshape(-1, 1) + self.biases
        self.a = self.activation_f.f(self.z)
        return self.a

    def backward_pass(self, dCdz):
        """
        The following link explains CNN backpropagation in more depth.
        https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
        """
        dCdb = dCdz * self.activation_f.f_prime(self.z)

        # dCdW can be computed by convolving the input image with dCdz such that
        # we are essentially doing an inverse convolution to get the gradients
        # with respect to the filter
        dCdW = []
        if self.padding == 'same':
            padding = (self.filter_size[0] - 1, self.filter_size[1] - 1)
        else:
            padding = self.padding
        for i in range(0, len(dCdz), self.conv_size[0] * self.conv_size[1]):
            convolution = np.zeros((self.filter_size[0] * self.filter_size[1], 1))
            for j in range(0, len(self.a0), self.img_size[0] * self.img_size[1]):
                convolution_i, _ = convolve(self.a0[j:j + self.img_size[0] * self.img_size[1]], self.img_size,
                                            dCdz[i:i + self.conv_size[0] * self.conv_size[1]], self.conv_size,
                                            self.stride, padding=padding)
                convolution += convolution_i
            dCdW.append(convolution)

        # dCd(z-1) can be computed by a full convolution of dCdz with the filter used
        dz1 = []
        for filtr in range(self.depth):
            first = True
            for i in range(0, len(dCdz), self.conv_size[0] * self.conv_size[1]):
                convolution_i, _ = convolve(dCdz[i:i + self.conv_size[0] * self.conv_size[1]], self.conv_size,
                                            self.weights[filtr], self.filter_size, self.stride,
                                            padding='full')
                if first:
                    convolution = convolution_i
                else:
                    convolution += convolution_i
            dz1.append(convolution)
        dCdz = np.array(dz1).reshape(-1, 1)

        return dCdz, np.array(dCdW), np.array(dCdb)

    def reweight(self, dCdW, dCdb, eta, m, L1dn=0.0, L2dn=0.0):
        self.biases = self.biases - eta/m * dCdb
        self.weights = (1 - eta * L2dn) * np.array(self.weights) - eta / m * np.array(dCdW) - eta * L1dn


class PoolingLayer(object):
    # Should only connect to image processing layers
    def __init__(self, img_size, pool_size, stride, pool_type='max'):
        self.img_size = img_size
        self.pool_size = pool_size
        self.stride = stride
        self.pool_type = pool_type
        self.a0 = None
        self.a = None
        self.depth = None

        self.size = None

        # static weights, makes computation easier for mean pooling with convolve()
        self.weights = np.array([1/(pool_size[0]*pool_size[1])] *
                                pool_size[0] * pool_size[1]).reshape(pool_size[0], pool_size[1])

    def connect_to(self, layer):
        self.depth = layer.depth
        self.size = (int((self.img_size[0] - self.pool_size[0])/self.stride) + 1) * (
                     int((self.img_size[1] - self.pool_size[1])/self.stride) + 1) * self.depth
        return

    def forward_pass(self, a):
        self.a0 = a
        pool = []
        if self.pool_type == 'max':
            '''
            The code here is pretty much the same as in convolve,
            but instead of taking a dot product, we are taking
            max(img_block)
            '''
            for image in range(0, len(a), self.img_size[0] * self.img_size[1]):
                img = a[image:image + self.img_size[0] * self.img_size[1]]
                for i in range(int((self.pool_size[0])/2),
                               self.img_size[0] - int((self.pool_size[0] - 1)/2), self.stride):
                    for j in range(int((self.pool_size[1])/2),
                                        self.img_size[1] - int((self.pool_size[1] - 1)/2), self.stride):
                        img_block = []
                        for k1 in range(int(-self.pool_size[0] / 2), int((self.pool_size[0] + 1) / 2)):
                            for k2 in range(int(-self.pool_size[1] / 2), int((self.pool_size[1] + 1) / 2)):
                                if i + k1 < 0 or i + k1 >= self.img_size[0] or j + k2 < 0 or j + k2 > self.img_size[1]:
                                    img_block.append(0)
                                else:
                                    img_block.append(img[(i + k1) * self.img_size[1] + j + k2])
                        pool.append(max(img_block))
        elif self.pool_type == 'mean':
            for i in range(0, len(a), self.img_size[0] * self.img_size[1]):
                pool.append(convolve(a[i:i + self.img_size[0] * self.img_size[1]],
                                     self.img_size, self.weights, self.pool_size,
                                     self.stride, padding='none')[0])

        pool = np.array(pool).reshape(-1, 1)
        self.a = pool
        return self.a

    def backward_pass(self, dCdz):
        # no weights and biases
        # just modify dCdz for the right dimensions
        if self.pool_type == 'max':
            dz1 = np.zeros(self.a0.shape)
            for image in range(0, len(dz1), self.img_size[0] * self.img_size[1]):
                dzidx = 0
                dimg = dz1[image:image + self.img_size[0] * self.img_size[1]]
                img = self.a0[image:image + self.img_size[0] * self.img_size[1]]
                for i in range(int((self.pool_size[0])/2),
                               self.img_size[0] - int((self.pool_size[0] - 1)/2), self.stride):
                    for j in range(int((self.pool_size[1])/2),
                                        self.img_size[1] - int((self.pool_size[1] - 1)/2), self.stride):
                        img_block = []
                        for k1 in range(int(-self.pool_size[0] / 2), int((self.pool_size[0] + 1) / 2)):
                            for k2 in range(int(-self.pool_size[1] / 2), int((self.pool_size[1] + 1) / 2)):
                                if i + k1 < 0 or i + k1 >= self.img_size[0] or j + k2 < 0 or j + k2 > self.img_size[1]:
                                    img_block.append(0)
                                else:
                                    img_block.append(img[(i + k1) * self.img_size[1] + j + k2])
                        k1 = int(img_block.index(max(img_block))/self.pool_size[0]) - int(self.pool_size[0]/2)
                        k2 = img_block.index(max(img_block)) - \
                             self.pool_size[0] * int(img_block.index(max(img_block))/self.pool_size[0]) - \
                             int(self.pool_size[1]/2)
                        dimg[(i + k1) * self.img_size[1] + j + k2] += dCdz[dzidx]
                        dzidx += 1
                dz1[image:image + self.img_size[0] * self.img_size[1]] += dimg
            dCdz = dz1.reshape(-1, 1)
        elif self.pool_type == 'mean':
            dCdz = np.array([di / (self.pool_size[0] * self.pool_size[1])] * self.pool_size[0] * self.pool_size[1]
                            for di in dCdz)
            dCdz.reshape(-1, 1)

        return dCdz, np.array([]), np.array([])

    def reweight(self, dCdW, dCdb, eta, m, L1dn=0.0, L2dn=0.0):
        # nothing to reweight
        return


'''
Begin: NEURAL NETWORK CLASS
    :add_layer(layer) : adds a layer to the network and
                        connects it to the previous layer
                        of the network
    :feed_forward(a) : computes the output activation of
                       the network given an input 'a'
    :back_propagation(x, y) : calls feed_forward on 'x'
                              and computes the gradients
                              to the cost function given
                              expected output 'y' which
                              are used to update the 
                              parameters in the network 
                              on the backward pass
    :SGD(training_data, epochs, mini_batch_size, eta, 
         test_data=None, L1=0.0, L2=0.0, 
         early_stopping=False, variable_eta=False) :
            performs stochastic gradient descent on the
            network with learning rate 'eta' given the 
            'training_data' over the number of 'epochs' 
            using a batch size of 'mini_batch_size'.
            L1 and L2 regularization parameters are
            provided as well and support for early
            stopping and a gradual decrease of the 
            learning rate has also been implemented.
    :update_mini_batch(eta, mini_batch, n, L1, L2) :
            helper function for SGD
'''


class Network(object):

    def __init__(self, cost_f=CrossEntropyCost):
        self.cost_f = cost_f
        self.layers = []

    def add_layer(self, layer):
        if self.layers:
            layer.connect_to(self.layers[-1])

        self.layers.append(layer)

    def feed_forward(self, a):
        for layer in self.layers:
            a = layer.forward_pass(a)

        return a

    def back_propagation(self, x, y):
        self.feed_forward(x)

        # reverse order of layers
        dCdW = []
        dCdb = []

        # NOTE -
        # The following code is janky - it forces us to precompute
        # the gradients to the first backward pass. It can be made more
        # universal by using an implementation with cost_f.dCda(a, y)
        z = self.layers[-1].z
        a = self.layers[-1].a
        delta = self.cost_f.delta(z, a, y)

        dCdb.append(delta)
        dCdW.append(np.dot(delta, self.layers[-2].a.transpose()))
        dCdz = np.dot(np.transpose(self.layers[-1].weights), delta)

        for layer in range(2, len(self.layers)):
            dCdz, dCdWi, dCdbi = self.layers[-layer].backward_pass(dCdz)
            dCdW.append(dCdWi)
            dCdb.append(dCdbi)

        return dCdW, dCdb

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None,
            L1=0.0, L2=0.0, early_stopping=False, variable_eta=False):
        n = len(training_data)
        epoch_count = 0
        stop_count = 0
        max_eval = 0
        eta_count = 1
        stop = False

        while not stop:
            epoch_count += 1

            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(eta, mini_batch, n, L1, L2)

            evaluation_result = self.evaluate(test_data)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch_count, evaluation_result, len(test_data)))
            else:
                print("Epoch {0} complete".format(epoch_count))

            if early_stopping:
                if evaluation_result > max_eval:
                    max_eval = evaluation_result
                    stop_count = 0
                else:
                    stop_count += 1
                    if stop_count == epochs:
                        if not variable_eta:
                            stop = True
                        elif eta_count == 7:
                            stop = True
                        else:
                            stop_count = 0
                            eta = eta / eta_count
                            eta_count += 1
            else:
                if epoch_count == epochs:
                    stop = True

    def update_mini_batch(self, eta, mini_batch, n, L1, L2):
        m = len(mini_batch)
        dCdW = []
        dCdb = []
        L1dn = L1 / n
        L2dn = L2 / n

        for x, y in mini_batch:
            dCdWi, dCdbi = self.back_propagation(x, y)
            if not dCdW:
                dCdW = dCdWi
                dCdb = dCdbi
            else:
                dCdW = [d + di for d, di in zip(dCdW, dCdWi)]
                dCdb = [d + di for d, di in zip(dCdb, dCdbi)]

        for l in range(1, len(self.layers)):
            self.layers[-l].reweight(dCdW[l-1], dCdb[l-1], eta, m, L1dn, L2dn)

    # function evaluate(self, test_data) must be rewritten to match each set of data
    # CURRENT DATASET: MNIST
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        total_correct = sum(int(x == y) for x, y in test_results)
        return total_correct

# --------------------------------------------------------------------------------------------------------------
#       INCOMPLETE SAVE / LOAD
# --------------------------------------------------------------------------------------------------------------
    def save(self, fname):
        data = {'layers': [str(layer.__name__) for layer in self.layers],
                'weights': [layer.weights.tolist() for layer in self.layers],
                'biases': [layer.biases.tolist() for layer in self.layers],
                'cost': str(self.cost_f.__name__)}

        f = open(fname, 'w')
        json.dump(data, f)
        f.close()


# Load a trained neural net
def load(fname):
    f = open(fname, 'r')
    data = json.load(f)
    f.close()

    cost = getattr(sys.modules[__name__], data['cost'])
    net = Network(cost_f=cost)
    for layer in data['layers']:
        layer_name = getattr(sys.modules[__name__], layer)
