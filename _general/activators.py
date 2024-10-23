from math import e


# activators take z as input
# d_activators take a as input, essentially a = activator(z)


def relu(z):
    if z > 0:
        return z
    return 0

def d_relu(a):
    if a > 0:
        return 1
    return 0

def leaky_relu(z):
    if z > 0:
        return z
    return z * 0.01

def d_leaky_relu(a):
    if a > 0:
        return 1
    return 0.01

def sigmoid(z):
    if z < -700:
        return 0
    return 1 / (1 + e**-z)

def d_sigmoid(a):
    return a * (1 - a)

def nothing(z):
    return z

def d_nothing(a):
    return 1


activator_lookup = [
    ["relu", relu, d_relu],
    ["leaky_relu", leaky_relu, d_leaky_relu],
    ["sigmoid", sigmoid, d_sigmoid],
    ["nothing", nothing, d_nothing],
]
