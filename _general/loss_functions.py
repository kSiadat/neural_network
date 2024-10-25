from numpy import exp, log, square


def lookup_function(name):
    lookup = {
        "squared": [squared, d_squared],
        "softmax": [softmax, d_softmax],
        }
    return lookup[name][0], lookup[name][1]

def lookup_name(function):
    lookup = {
        squared: "squared",
        softmax: "softmax",
        }
    return lookup[function]


def squared(target, guess):
    return 0.5 * square(guess - target)

def d_squared(target, guess):
    return guess - target

def softmax(target, guess):
    exp_arr = exp(guess)
    exp_arr = exp_arr / exp_arr.sum()
    return -target * log(exp_arr)

def d_softmax(target, guess):
    exp_arr = exp(guess)
    exp_arr = exp_arr / exp_arr.sum()
    return exp_arr - target
