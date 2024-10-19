from math import floor
from random import uniform


from .activators import activator_lookup
from .layer_normal import Layer, Node


class Filter(Node):
    def __init__(self, size, depth, activator, load_text=None):
        if load_text is None:
            self.weight = [[[uniform(-1, 1)  for z in range(depth)]  for y in range(size)]  for x in range(size)]
            self.bias = uniform(-1, 1)
            self.size = size
            self.depth = depth
            self.set_activator(activator)
        else:
            pass
        self.output = None
        self.d_weight = [0  for x in range(len(self.weight))]
        self.d_bias = 0

    def calculate(self, image, stride):##### do something about padding
        w = floor(((len(image)-self.size) / stride) + 1)
        h = floor(((len(image[0])-self.size ) / stride) + 1)
        self.output = []
        for x in range(0, w, stride):
            self.output.append([])
            for y in range(0, h, stride):
                total = 0
                for a in range(self.size):
                    for b in range(self.size):
                        for c in range(self.depth):
                            total += self.weight[a][b][c] * image[x+a][y+b][c]
                total = self.activator(total + self.bias)
                self.output[-1].append(total)
        return self.output

    def backpropagate(self, image, gradient):
        self.d_bias = 0#gradient * self.d_activator(self.value)
        self.d_weight = [[[0  for Z in Y]  for Y in X]  for X in self.weight]
        

    def adjust(self, rate):
        pass

    def display(self):
        for X in self.weight:
            print(X)
        print(self.bias)


if __name__ == "__main__":
    f = Filter(3, 1, "relu")
    f.display()
    n = f.calculate([[[0], [1], [2], [3]], [[0], [1], [2], [3]], [[0], [1], [2], [3]], [[0], [1], [2], [3]]], 1)
    print(n)
