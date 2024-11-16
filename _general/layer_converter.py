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
        return None

    def load(self, text):
        text = [[int(Y)  for Y in X.split(",")]  for X in text.split("|")]
        self.inp_shape = array(text[0])
        self.out_shape = array(text[1])

    def save(self):
        text_i = ",".join([str(X)  for X in self.inp_shape])
        text_o = ",".join([str(X)  for X in self.out_shape])
        return f"{text_i}|{text_o}"

    def display(self):
        print(f"{self.inp_shape} -> {self.out_shape}")

    def evaluate(self, inp):
        return inp.reshape(self.out_shape)

    def backpropagate(self, inp, gradient):
        return gradient.reshape(inp.shape)

    def get_all_gradients(self):
        return None

    def adjust(self, rate):
        return None
