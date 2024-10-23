from numpy import array, empty, random, zeros

from .activators import activator_lookup


class Layer:
    def __init__(self, size, inp_size, activator, random_range, load_text=None):
        if load_text is None:
            generator = random.default_rng()
            self.weight = generator.uniform(random_range[0], random_range[1], [size, inp_size])
            self.bias = generator.uniform(random_range[0], random_range[1], [size])
            self.set_activator(activator)
        else:
            self.load(load_text)
        self.reset_d()
        
    def set_activator(self, name):
        for X in activator_lookup:
            if name == X[0]:
                self.activator = X[1]
                self.d_activator = X[2]
                break

    def get_activator_text(self):
        for X in activator_lookup:
            if self.activator is X[1]:
                return X[0]

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
        self.set_activator(text[0][0])

    def save(self):
        text_a = self.get_activator_text()
        text_s = ",".join([str(X)  for X in self.weight.shape])
        text_w = ",".join([str(X)  for X in self.weight.flatten()])
        text_b = ",".join([str(X)  for X in self.bias.copy()])
        return f"{text_a}|{text_s}|{text_w}|{text_b}"

    def display(self, meta=True, main=False, output=False, d=False):
        text = ""
        if meta:
            text += f"weight shape: {self.weight.shape}\nbias shape: {self.bias.shape}\nactivator: {self.get_activator_text()}\n"
        if main:
            text += f"weights:\n{self.weight}\nbiases:\n{self.bias}\n"
        if output:
            text += f"output:\n{self.output}\n"
        if d:
            text += f"weight gradients:\n{self.d_weight}\nbias gradients:\n{self.d_bias}\n"
        return text

    def evaluate(self, inp):
        self.output = self.weight * inp
        self.output = self.output.sum(axis=1) + self.bias
        for x in range(len(self.output)):
            self.output[x] = self.activator(self.output[x])
        return self.output

    def backpropagate(self, inp, gradient):
        d_z = empty(self.output.shape)
        for x in range(len(d_z)):
            d_z[x] = self.d_activator(self.output[x])
        d_z = d_z * gradient
        d_z_weight = d_z.repeat(self.weight.shape[1]).reshape(self.weight.shape)
        self.d_weight = self.d_weight + (d_z_weight * inp)
        self.d_bias = self.d_bias + d_z
        return self.get_all_gradients()

    def get_all_gradients(self):
        return self.d_weight.sum(axis=0)

    def adjust(self, rate):
        self.weight = self.weight - (self.d_weight * rate)
        self.bias = self.bias - (self.d_bias * rate)
        self.reset_d()
