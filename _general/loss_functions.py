from numpy import exp, log, square


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


name_to_func = {
        "squared": [squared, d_squared],
        "softmax": [softmax, d_softmax],
        }

func_to_name = {
        squared: "squared",
        softmax: "softmax",
        }

def lookup_function(name):
    return name_to_func[name][0], name_to_func[name][1]

def lookup_name(function):
    return func_to_name[function]
