from numpy import array, random, zeros

from .activators import lookup_activator, lookup_name
#from .layer_lookup import lookup_name as lookup_class_name
from .layer_normal import Layer


class Layer_convolutional(Layer):
    def __init__(self, inp_shape, filter_shape, stride, padding, activator, random_range, load_text=None):
        if load_text is None:
            self.shape    = array(filter_shape) # format: [quantity, x, y, z]
            self.stride   = stride
            self.padding  = padding

            self.activator, self.d_activator = lookup_activator(activator)
            generator = random.default_rng()
            self.weight = generator.uniform(-random_range, random_range, self.shape)
            self.bias   = generator.uniform(-random_range, random_range, [self.shape[0]])

            #self.inp_shape = inp_shape
            self.out_shape = array([
                int(((inp_shape[0] - self.shape[1] + (2 * padding)) / stride) + 1),
                int(((inp_shape[1] - self.shape[2] + (2 * padding)) / stride) + 1),
                self.shape[0],
                ])
        else:
            self.load(load_text)
        self.reset_d()

    def reset_d(self):
        self.d_weight = zeros(self.weight.shape)
        self.d_bias = zeros(self.bias.shape)

    def load(self, text):
        text = [X.split(",")  for X in text.split("|")]
        data_s = array([int(X)  for X in text[1]])
        data_t = int(text[2][0])
        data_p = int(text[3][0])
        data_o = array([int(X)  for X in text[4]])
        data_w = array([float(X)  for X in text[5]])
        data_b = array([float(X)  for X in text[6]])
        
        self.activator, self.d_activator = lookup_activator(text[0][0])
        self.shape     = data_s
        self.stride    = data_t
        self.padding   = data_p
        self.out_shape = data_o
        self.weight    = data_w.reshape(data_s)
        self.bias      = data_b

    def save(self):
        text_a = lookup_name(self.activator)
        text_s = ",".join([str(X)  for X in self.shape])
        text_t = str(self.stride)
        text_p = str(self.padding)
        text_o = ",".join([str(X)  for X in self.out_shape])
        text_w = ",".join([str(X)  for X in self.weight.flatten()])
        text_b = ",".join([str(X)  for X in self.bias])
        return f"{text_a}|{text_s}|{text_t}|{text_p}|{text_o}|{text_w}|{text_b}"

    def display(self):
        pass

    def evaluate(self, inp):
        output = zeros(self.out_shape)
        for f in range(self.shape[0]):
            for x in range(self.out_shape[0]):
                for y in range(self.out_shape[1]):
                    corner = [x * self.stride, y * self.stride]
                    sub_inp = inp[corner[0]:corner[0]+self.shape[1], corner[1]:corner[1]+self.shape[2]]
                    multiplied = sub_inp * self.weight[f]
                    output[x][y][f] = self.activator(multiplied.sum() + self.bias[f])
        return output

    def backpropagate(self, inp, gradient):
        pass

    def get_all_gradients(self):
        pass

    def adjust(self, rate):
        pass
