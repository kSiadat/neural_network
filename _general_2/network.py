from numpy import empty, random

from _general_2.layer_normal import Layer


class Network:
    def __init__(self, shape, activator, random_range, load_path=None):
        if load_path is None:
            self.layer = [Layer(shape[x], shape[x-1], activator[x-1], random_range)  for x in range(1, len(shape))]
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
    def get_sample(data, label, size, do_label=True):
        indexes = random.randint(0, size, [size])
        data_sample = data[indexes]
        if do_label:
            label_sample = label[indexes]
            return data_sample, label_sample
        return data_sample

    def batch_epoch(self, data, label, sample_size, rate):
        data_sample, label_sample = Network.get_sample(data, label, sample_size)
        return self.epoch(data_sample, label_sample, rate)

    def batch_epoch_test(self, data, sample_size):
        data_sample = Network.get_sample(data, None, sample_size, False)
        return self.epoch_test(data_sample)
