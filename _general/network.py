from numpy import empty, random

from _general_2.layer_normal import Layer


class Network:
    def __init__(self, shape, activator, random_range, load_path=None):
        if load_path is None:
            self.layer = [Layer(shape[x], shape[x-1], activator[x-1], random_range[x-1])  for x in range(1, len(shape))]
        else:
            self.load(load_path)

    def save(self, path):
        text = "\n".join([X.save()  for X in self.layer])
        with open(f"{path}.txt", "w") as file:
            file.write(text)

    def load(self, path):
        with open(f"{path}.txt", "r") as file:
            text = file.read()
        text = text.split("\n")
        self.layer = [Layer(None, None, None, None, X)  for X in text]

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
            self.backpropagate(data[x], self.get_output() - label[x])
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
            self.backpropagate(data[x], answer[x] - label[x])
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
