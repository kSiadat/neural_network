from numpy import empty, exp, log, square


def squared(target, guess):
    return 0.5 * square(guess - target)

def d_squared(target, guess):
    return guess - target

def sub_softmax(x):
    exp_arr = exp(x)
    if len(x.shape) == 1:
        return exp_arr / exp_arr.sum()
    elif len(x.shape) == 2:
        new = empty(x.shape)
        for x in range(len(x)):
            new[x] = exp_arr[x] / exp_arr[x].sum()
        return new

def softmax(target, guess):
    return -target * log(sub_softmax(guess))

def d_softmax(target, guess):
    return sub_softmax(guess) - target


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
