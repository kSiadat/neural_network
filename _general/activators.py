from math import e
from numpy import exp


def relu_old(z):
    if z > 0:
        return z
    return 0

def d_relu_old(a):
    if a > 0:
        return 1
    return 0

def leaky_relu_old(z):
    if z > 0:
        return z
    return z * 0.01

def d_leaky_relu_old(a):
    if a > 0:
        return 1
    return 0.01

def sigmoid_old(z):
    if z < -700:
        return 0
    return 1 / (1 + e**-z)

def d_sigmoid_old(a):
    return a * (1 - a)

def nothing(z):
    return z

def d_nothing(a):
    return 1

def relu(arr):
    return arr * (arr > 0)

def d_relu(arr):
    return 1 * (arr > 0)

def leaky_relu(arr):
    return arr * (((arr < 0) * 0.01) + ((arr > 0) * 1))

def d_leaky_relu(arr):
    return (((arr <= 0) * 0.01) + ((arr > 0) * 1))

def sigmoid(arr):
    cap = arr * (arr > -700)
    return 1 / (1 + exp(-cap))

def d_sigmoid(arr):
    return arr * (1 - arr)


name_to_func = {
        "relu":       [relu,       d_relu],
        "leaky_relu": [leaky_relu, d_leaky_relu],
        "sigmoid":    [sigmoid,    d_sigmoid],
        "nothing":    [nothing,    d_nothing],
        }

func_to_name = {
        relu:       "relu",
        leaky_relu: "leaky_relu",
        sigmoid:    "sigmoid",
        nothing:    "nothing",
        }

def lookup_activator(name):
    return name_to_func[name][0], name_to_func[name][1]

def lookup_name(activator):
    return func_to_name[activator]
