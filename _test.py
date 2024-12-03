from numpy import array
from random import randint

from _general.layer_converter import Layer_converter
from _general.layer_convolutional import Layer_convolutional


layer = Layer_convolutional("relu", [8, 8, 1], [4, 3, 3, 1], 2, 0, 1)
text = layer.save()
print(text)
layer.load(text)
layer.display()
text = layer.save()
print(text)
