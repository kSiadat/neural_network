from numpy import array, empty, random, zeros

from .activators import lookup_activator, lookup_name


class Layer:
    def __init__(self, inp_size, size, activator, random_range, load_text=None):
        if load_text is None:
            generator = random.default_rng()
            self.weight = generator.uniform(-random_range, random_range, [size, inp_size])
            self.bias   = generator.uniform(-random_range, random_range, [size])
            self.activator, self.d_activator = lookup_activator(activator)
        else:
            self.load(load_text)
        self.reset_d()

    def reset_d(self):
        self.d_weight = zeros(self.weight.shape)
        self.d_bias = zeros(self.bias.shape)

    def load(self, text):
        text = [X.split(",")  for X in text.split("|")]
        data_s = [int(X)  for X in text[1]]
        data_w = array([float(X)  for X in text[2]])
        data_b = array([float(X)  for X in text[3]])
        self.weight = data_w.reshape(data_s)
        self.bias = data_b
        self.activator, self.d_activator = lookup_activator(text[0][0])

    def save(self):
        text_a = lookup_name(self.activator)
        text_s = ",".join([str(X)  for X in self.weight.shape])
        text_w = ",".join([str(X)  for X in self.weight.flatten()])
        text_b = ",".join([str(X)  for X in self.bias])
        return f"{text_a}|{text_s}|{text_w}|{text_b}"

    def display(self, meta=True, main=False, output=False, d=False):
        text = ""
        if meta:
            text += f"weight shape: {self.weight.shape}\nbias shape: {self.bias.shape}\nactivator: {lookup_name(self.activator)}\n"
        if main:
            text += f"weights:\n{self.weight}\nbiases:\n{self.bias}\n"
        if output:
            text += f"output:\n{self.output}\n"
        if d:
            text += f"weight gradients:\n{self.d_weight}\nbias gradients:\n{self.d_bias}\n"
        return text

    def evaluate(self, inp):
        self.output = self.weight * inp
        self.output = self.activator(self.output.sum(axis=1) + self.bias)
        return self.output

    def backpropagate(self, inp, gradient):
        d_z = self.d_activator(self.output)
        d_z = d_z * gradient
        d_z_wide = d_z.repeat(self.weight.shape[1]).reshape(self.weight.shape)
        self.d_weight = self.d_weight + (d_z_wide * inp)
        self.d_bias = self.d_bias + d_z
        inp_gradient = (d_z_wide * self.weight).sum(axis=0)
        return inp_gradient

    def adjust(self, rate):
        self.weight = self.weight - (self.d_weight * rate)
        self.bias = self.bias - (self.d_bias * rate)
        self.reset_d()
