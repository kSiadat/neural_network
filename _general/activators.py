from math import e


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
