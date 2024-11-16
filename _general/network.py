from numpy import empty, random

from .layer_normal import Layer
from .lookup_layer import lookup_class, lookup_name as lookup_class_name, text_init
from .loss_functions import lookup_function, lookup_name


class Network:
    def __init__(self, layer_data, loss_func, load_path=None):
        if load_path is None:
            self.layer = [lookup_class(X[0])(*X[1])  for X in layer_data]
            self.loss_function, self.d_loss_function = lookup_function(loss_func)
        else:
            self.load(load_path)

    """
    def __init__1(self, shape, activator, random_range, loss_func, load_path=None):
        if load_path is None:
            self.layer = [Layer(shape[x], shape[x-1], activator[x-1], random_range[x-1])  for x in range(1, len(shape))]
            self.loss_function, self.d_loss_function = lookup_function(loss_func)
        else:
            self.load(load_path)
    """

    def save(self, path):
        loss_text = lookup_name(self.loss_function) + "\n"
        text = loss_text + "\n".join([f"{lookup_class_name(type(X))}/{X.save()}"  for X in self.layer])
        with open(f"{path}.txt", "w") as file:
            file.write(text)

    def load(self, path):
        with open(f"{path}.txt", "r") as file:
            text = file.read()
        text = text.split("\n")
        self.loss_function, self.d_loss_function = lookup_function(text[0])
        text = [X.split("/")  for X in text[1:]]
        self.layer = [text_init(*X)  for X in text]

    def display(self, meta=True, main=False, output=False, d=False):
        for x in range(len(self.layer)):
            print(f"===== layer {x+1}:\n{self.layer[x].display(meta, main, output, d)}")

    def evaluate(self, inp):
        for x in range(len(self.layer)):
            inp = self.layer[x].evaluate(inp)
        return inp

    def backpropagate(self, inp, gradient):
        for x in range(1, len(self.layer)):
            gradient = self.layer[-x].backpropagate(self.layer[-x-1].output, gradient)
        self.layer[0].backpropagate(inp, gradient)

    def get_output(self):
        return self.layer[-1].output

    def adjust(self, rate):
        for x in range(len(self.layer)):
            self.layer[x].adjust(rate)

    def epoch(self, data, label, rate):
        size = len(data)
        answer = empty([size, len(self.layer[-1].bias)])
        for x in range(size):
            self.evaluate(data[x])
            answer[x] = self.get_output()
            d_loss = self.d_loss_function(label[x], answer[x])
            self.backpropagate(data[x], d_loss)
        self.adjust(rate / size)
        return answer

    def epoch_test(self, data):
        size = len(data)
        answer = empty([size, len(self.layer[-1].bias)])
        for x in range(size):
            self.evaluate(data[x])
            answer[x] = self.get_output()
        return answer

    def online_epoch(self, data, label, rate):
        size = len(data)
        answer = empty([size, len(self.layer[-1].bias)])
        for x in range(size):
            self.evaluate(data[x])
            answer[x] = self.get_output()
            d_loss = self.d_loss_function(label[x], answer[x])
            self.backpropagate(data[x], d_loss)
            self.adjust(rate)
        return answer

    @staticmethod
    def sample_indexes(data_size, sample_size):
        return random.randint(0, data_size, [sample_size])

    def batch_epoch(self, data, label, rate, sample_size):
        indexes = Network.sample_indexes(len(data), sample_size)
        return self.epoch(data[indexes], label[indexes], rate), indexes

    def batch_epoch_test(self, data, sample_size):
        indexes = Network.sample_indexes(len(data), sample_size)
        return self.epoch_test(data[indexes]), indexes
