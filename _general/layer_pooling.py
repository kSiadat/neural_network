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
        text = text.split("|")
        self.size = int(text[0])
        self.out_shape = array([int(X)  for X in text[1].split(",")])

    def save(self):
        text_s = str(self.size)
        text_o = ",".join([str(X)  for X in self.out_shape])
        return f"{text_s}|{text_o}"

    def display(self):
        print(f"kernel size:\n{self.size}")
        print(f"output shape:\n{self.out_shape}")

    def evaluate(self, inp):
        self.output = empty(self.out_shape)
        self.mask = empty(inp.shape)
        for x in range(self.out_shape[0]):
            for y in range(self.out_shape[1]):
                i = [x * self.size, y * self.size]
                for z in range(self.out_shape[2]):
                    num = inp[i[0]:i[0]+self.size, i[1]:i[1]+self.size, z].max()
                    sub_mask = num == inp[i[0]:i[0]+self.size, i[1]:i[1]+self.size, z]
                    self.mask[i[0]:i[0]+self.size, i[1]:i[1]+self.size, z] = sub_mask
                    self.output[x][y][z] = num
        return self.output

    def backpropagate(self, inp, gradient):
        out = self.mask * gradient.repeat(self.size, axis=0).repeat(self.size, axis=1)
        return out

    def adjust(self, rate):
        return None
