from math import e


def relu(n):
    if n > 0:
        return n
    return 0

def d_relu(n):
    if n > 0:
        return 1
    return 0

def leaky_relu(n):
    if n > 0:
        return n
    return n * 0.01

def d_leaky_relu(n):
    if n > 0:
        return 1
    return 0.01

def sigmoid(n):
    if n < -700:
        return 0
    return 1 / (1 + e**-n)

def d_sigmoid(n):
    s = sigmoid(n)
    return s * (1 - s)

activator_lookup = [
    ["relu", relu, d_relu],
    ["leaky_relu", leaky_relu, d_leaky_relu],
    ["sigmoid", sigmoid, d_sigmoid],
]
