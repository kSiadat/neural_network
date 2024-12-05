from numpy import array
from random import randint

from _general.network import Network
from _general.layer_normal import Layer


inp = array([[[1], [2], [3], [4]],
             [[0], [0], [0], [0]],
             [[-1],[-2],[-3],[-4]],
             [[-1], [1],[-1], [1]]
             ])
"""
inp = array([[[1], [2], [3]],
             [[0], [0], [0]],
             [[-1],[-2],[-3]],
             ])
"""
print(f"input:\n{inp.reshape(inp.shape[:2])}")
layer_data = [
        ["convolutional", [[4, 4, 1], [2, 3, 3, 1], 1, 1, "relu", 1]],
        ]
network = Network(layer_data, "squared")
network.save("test")
#print(f"weights {network.layer[0].weight.shape}:\n{network.layer[0].weight}")
#print(f"biases:\n{network.layer[0].bias}")
out = network.evaluate(inp)
#print(out)
print(f"output:\n{out}")#.reshape(out.shape[:2])}")

gradient = array([[[1, 1], [5, 1], [1, 1], [5, 1]],
                  [[1,-1], [5,-1], [1, 1], [5, 1]],
                  [[1, 1], [5, 1], [1, 1], [5, 1]],
                  [[1,-1], [5,-1], [1, 1], [5, 1]],
                  ])
inp_grad = network.layer[0].backpropagate(inp, gradient)
print(f"gradient:\n{gradient}")#.reshape(gradient.shape[:2])}")
print(f"d_weight:\n{network.layer[0].d_weight}")
print(f"d_bias:\n{network.layer[0].d_bias}")
print(f"input gradient:\n{inp_grad}")#.reshape(inp_grad.shape[:2])}")

