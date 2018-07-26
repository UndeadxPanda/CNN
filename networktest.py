import network
import mnist_data_loader

training_data, validation_data, test_data = mnist_data_loader.load_data_wrapper('mnist.pkl.gz')

'''
net = network.Network()
net.add_layer(network.InputLayer(784))
net.add_layer(network.FullyConnectedLayer(30))
net.add_layer(network.FullyConnectedLayer(10))
net.SGD(training_data, 30, 10, 0.1, test_data=validation_data, L2=5.0)
test_eval = net.evaluate(test_data=test_data)
print('Test evaluation: {0} / {1}'.format(test_eval, len(test_data)))
'''

cnet = network.Network()
cnet.add_layer(network.InputLayer(784))
cnet.add_layer(network.ConvolutionLayer(20, (28, 28), (5, 5), padding='none'))
cnet.add_layer(network.PoolingLayer((24, 24), (2, 2), 2))
cnet.add_layer(network.FullyConnectedLayer(100))
cnet.add_layer(network.FullyConnectedLayer(10, activation_f=network.SoftMax))
cnet.SGD(training_data, 1, 10, 0.1, test_data=validation_data, L2=5.0)
test_eval = cnet.evaluate(test_data=test_data)
print('Test evaluation: {0} / {1}'.format(test_eval, len(test_data)))
