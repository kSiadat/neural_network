from numpy import array
from random import randint

from _general.layer_converter import Layer_converter


layer = Layer_converter([16], [4, 4, 1])
text = layer.save()
print(text)
layer.load(text)
layer.display()
text = layer.save()
print(text)
