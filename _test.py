from numpy import array
from random import randint

from _general.network import Network


inp = array([[[1], [2], [3], [4]],
             [[0], [0], [0], [0]],
             [[-1],[-2],[-3],[-4]],
             [[-1], [1],[-1], [1]]
             ])
print(inp.reshape(inp.shape[:2]))
layer_data = [
        ["convolutional", [[4, 4, 1], [2, 2, 2, 1], 1, 0, "relu", 1]],
        ]
network = Network(layer_data, "squared")
network.save("test")
print(network.layer[0].weight)
print(network.layer[0].bias)
out = network.evaluate(inp)
print(out)
#print(out.reshape(out.shape[:2]))
