from numpy import array

from _general.network import Network
from _general.layer_pooling import Layer_pooling


image = array([
    [[1, 1], [2, 2], [3, 2], [4, 1]],
    [[2, 0], [3, 0], [4, 0], [5, 0]],
    [[3, 0], [4, 0], [5, 0], [6, 0]],
    [[4, 1], [5, 2], [6, 2], [7, 1]],
    ])

gradient = array([
    [[1, 5], [2, 5]],
    [[3, 5], [4, 5]],
    ])

layer = Layer_pooling([4, 4, 2], 2)
output = layer.evaluate(image)
print(output)
backprop = layer.backpropagate(image, gradient)
print(backprop)
