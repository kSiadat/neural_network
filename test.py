from numpy import array

from _general.loss_functions import *


target = array([0, 1, 0])
guess = array([0.1, 0.4, 0.5])
print(softmax(target, guess))
print(d_softmax(target, guess))
