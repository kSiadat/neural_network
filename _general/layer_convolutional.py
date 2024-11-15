from numpy import random, zeros

from .activators import lookup_activator, lookup_name
#from .layer_lookup import lookup_name as lookup_class_name
from .layer_normal import Layer


class Layer_convolutional(Layer):
    def __init__(self, inp_shape, filter_param, activator, random_range, load_text=None):
        if load_text is None:
            self.shape    = filter_param[0]
            self.stride   = filter_param[1]
            self.padding  = filter_param[2]
            self.quantity = filter_param[3]

            self.activator, self.d_activator = lookup_activator(activator)
            generator = random.default_rng()
            self.weight = generator.uniform(-random_range, random_range, [self.quantity, self.shape[0], self.shape[1], self.shape[2]])
            self.bias   = generator.uniform(-random_range, random_range, [self.quantity])

            self.inp_shape = inp_shape
            self.out_shape = [
                ((inp_shape[0] - self.shape[0] + (2 * self.padding)) / self.stride) + 1,
                ((inp_shape[1] - self.shape[1] + (2 * self.padding)) / self.stride) + 1,
                self.quantity,
                ]
        else:
            self.load(load_text)
        self.reset_d()

    def reset_d(self):
        self.d_weight = zeros(self.weight.shape)
        self.d_bias = zeros(self.bias.shape)

    #def load(self, text):
    #    pass

    def save(self):
        #text_l = lookup_class_name(Layer_convolutional)
        text_a = lookup_name(self.activator)
        text_s = ",".join([str(X)  for X in self.weight.shape])
        text_w = ",".join([str(X)  for X in self.weight.flatten()])
        text_b = ",".join([str(X)  for X in self.bias.copy()]) # remove .copy() ? It's also in layer_normal
        return f"{text_a}|{text_s}|{text_w}|{text_b}"

    def display(self):
        pass

    def evaluate(self, inp):
        pass

    def backpropagate(self, inp, gradient):
        pass

    def get_all_gradients(self):
        pass

    def adjust(self, rate):
        pass
