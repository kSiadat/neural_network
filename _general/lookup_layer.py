from .layer_converter import Layer_converter
from .layer_convolutional import Layer_convolutional
from .layer_normal import Layer

name_to_class = {
        "normal":           Layer,
        "converter":        Layer_converter,
        "convolutional":    Layer_convolutional, 
        }

class_to_name = {
        Layer:                  "normal",
        Layer_converter:        "converter",
        Layer_convolutional:    "convolutional",
        }

def lookup_class(name):
    return name_to_class[name]

def lookup_name(clas):
    return class_to_name[clas]

def text_init(name, text):
    if lookup_class(name) is Layer:
        return Layer(None, None, None, None, text)
    elif lookup_class(name) is Layer_converter:
        return Layer_converter(None, None, text)
    elif lookup_class(name) is Layer_convolutional:
        return Layer_convolutional(None, None, None, None, None, None, text)
