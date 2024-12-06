from numpy import array

from _general.network import Network
from _general.layer_pooling import Layer_pooling

image = array([
    [[1], [2], [3], [4]],
    [[2], [4], [6], [8]],
    [[3], [6], [9], [12]],
    [[4], [8], [12], [16]],
    ])

gradient = array([
    [[1], [-1]],
    [[-1], [1]],
    ])

layer = Layer_pooling(image.shape, 2)
print(layer.evaluate(image))
print(layer.backpropagate(image, gradient))
