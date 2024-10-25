from numpy import array

from _general.loss_functions import *


func, d_func = lookup_function("softmax")
name = lookup_name(func)
print(name)
