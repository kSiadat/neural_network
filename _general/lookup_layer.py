from .layer_convolutional import Layer_convolutional
from .layer_normal import Layer

name_to_class = {
        "normal":           Layer,
        "convolutional":    Layer_convolutional, 
        }

class_to_name = {
        Layer:                  "normal",
        Layer_convolutional:    "convolutional",
        }

def lookup_class(name):
    return name_to_class[name]

def lookup_name(clas):
    return class_to_name[clas]