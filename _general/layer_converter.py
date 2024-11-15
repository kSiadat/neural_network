from numpy import array

from .layer_normal import Layer

class Layer_converter(Layer):
    def __init__(self, inp_shape, out_shape, load_text=None):
        if load_text is None:
            self.inp_shape = array(inp_shape)
            self.out_shape = array(out_shape)
        else:
            self.load(load_text)

    def reset_d(self):
        pass

    def load(self, text):
        pass

    def save(self):
        pass

    def display(self):
        print(f"{self.inp_shape} -> {self.out_shape}")

    def evaluate(self, inp):
        return inp.reshape(self.out_shape)

    def backpropagate(self, inp, gradient):
        pass#### think about how gradients are reshaped

    def get_all_gradients(self):
        pass##### same as above

    def adjust(self, rate):
        pass
