from numpy import array

from _general.network import Network
from _general.layer_convolutional import Layer_convolutional

image = array([
    [[1, 1], [2, 1], [3, 1], [4, 1]],
    [[2, 1], [4, 1], [6, 1], [8, 1]],
    [[3, 1], [6, 1], [9, 1], [12,1]],
    [[4, 1], [8, 1], [12,1], [16,1]],
    ])

filt = array([
    [
    [[1, 2], [3, 4]],
    [[3, 4], [5, 6]],
    ],
    [
    [[1, -1], [1, -1]],
    [[1, -1], [1, -1]],
    ],
    [
    [[0, 0], [0, 0]],
    [[0, 0], [0, 0]],
    ]
    ])

gradient = array([
    [[1], [-1]],
    [[-1], [1]],
    ])

layer = Layer_convolutional(image.shape, filt.shape, 1, 0, "relu", 1)
out1 = layer.evaluate(image)
out2 = layer.evaluate_2(image)
print(f"out1 == out2:\n{out1 == out2}\n")
print(f"out1:\n{out1}\n")
print(f"out2:\n{out2}\n")
print(f"out1 - out2:\n{out1 - out2}\n")
