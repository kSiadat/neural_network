from numpy import array, empty

from .layer_normal import Layer

class Layer_pooling(Layer):
    def __init__(self, inp_shape, size, load_text=None):
        if load_text is None:
            self.size = size
            self.out_shape = array([inp_shape[0] // size,
                                    inp_shape[1] // size,
                                    inp_shape[2]
                                    ])
        else:
            self.load(load_text)

    def reset_d(self):
        return None

    def load(self, text):
        self.factor = int(text)

    def save(self):
        return str(self.factor)

    def display(self):
        print(self.factor)

    def evaluate(self, inp):
        self.output = empty(self.out_shape)
        for x in range(self.out_shape[0]):
            for y in range(self.out_shape[1]):
                i = [x * self.size, y * self.size]
                for z in range(self.out_shape[2]):
                    self.output[x][y][z] = inp[i[0]:i[0]+self.size, i[1]:i[1]+self.size, z].max()
        return self.output

    def backpropagate(self, inp, gradient):
        return gradient.repeat(self.size, axis=0).repeat(self.size, axis=1)

    def adjust(self, rate):
        return None
