from numpy import array, random, transpose, zeros
from random import uniform


from .activators import activator_lookup


class Node:
    def __init__(self, size, activator, load_text=None):
        if load_text is None:
            self.weight = random.uniform(-1, 1, [size])
            self.bias = uniform(-1, 1)
            self.set_activator(activator)
        else:
            self.load(load_text)
        self.value = 0
        self.reset_d()

    def set_activator(self, name):
        for X in activator_lookup:
            if name == X[0]:
                self.activator = X[1]
                self.d_activator = X[2]

    def calculate(self, inp):
        self.value = self.activator(sum(self.weight * inp) + self.bias)

    def backpropagate(self, inp, gradient):
        d_node = gradient * self.d_activator(self.value)
        self.d_bias += d_node
        self.d_weight = self.d_weight + (d_node * inp)

    def reset_d(self):
        self.d_bias = 0
        self.d_weight = zeros(len(self.weight))

    def adjust(self, rate):
        self.bias -= self.d_bias * rate
        self.weight = self.weight - (self.d_weight * rate)
        self.reset_d()

    def display(self):
        print("w:", self.weight, "\nb:", self.bias, "\nv:", self.value, "\nd_b:", self.d_bias, "\nd_w:", self.d_weight)

    def save(self):
        for X in activator_lookup:
            if self.activator is X[1]:
                name = X[0]
        return f"{name};" + ",".join([str(X)  for X in self.weight]) + f";{self.bias}"

    def load(self, text):
        text = text.split(";")
        text[1] = text[1].split(",")
        self.set_activator(text[0])
        self.weight = array([float(X)  for X in text[1]])
        self.bias = float(text[2])


class Layer:
    def __init__(self, size, inp, typ, load_text=None):
        if load_text is None:
            self.node = [Node(inp, typ)  for y in range(size)]
        else:
            self.load(load_text)

    def get_all_values(self):
        return array([X.value  for X in self.node])

    def get_all_gradients(self):
        weights = array([X.d_weight  for X in self.node])
        return weights.sum(axis=0)

    def calculate(self, inp):
        for x in range(len(self.node)):
            self.node[x].calculate(inp)
        return self.get_all_values()

    def backpropagate(self, inp, gradients):
        for x in range(len(self.node)):
            self.node[x].backpropagate(inp, gradients[x])
        return self.get_all_gradients()

    def adjust(self, rate):
        for x in range(len(self.node)):
            self.node[x].adjust(rate)

    def display(self):
        for X in self.node:
            X.display()
            print()

    def save(self):
        return "|".join([X.save()  for X in self.node])

    def load(self, text):
        text = text.split("|")
        self.node = [Node(None, None, X)  for X in text]
