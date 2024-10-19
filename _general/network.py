from math import e
from numpy import array, random
from random import randint, uniform


from .layer_normal import Layer


class Network:
    def __init__(self, size=None, typ=None, path=None):
        if path is None:
            self.layer = [Layer(size[x], size[x-1], typ[x-1])  for x in range(1, len(size))]
        else:
            self.load(path)

    def get_output(self):
        return self.layer[-1].get_all_values()

    def calculate(self, inp):
        out = self.layer[0].calculate(inp)
        for x in range(1, len(self.layer)):
            out = self.layer[x].calculate(out)
        return out

    def backpropagate(self, inp, target):
        out = self.layer[-1].get_all_values()
        gradients = out - target
        for x in range(1, len(self.layer)):
            gradients = self.layer[-x].backpropagate(self.layer[-x-1].get_all_values(), gradients)
        self.layer[0].backpropagate(inp, gradients)

    def adjust(self, rate):
        for x in range(len(self.layer)):
            self.layer[x].adjust(rate)

    def full_pass(self, inp, target, rate):
        answer = self.calculate(inp)
        self.backpropagate(inp, target)
        self.adjust(rate)
        return answer

    def epoch(self, data, labels, rate):
        answers = []
        for x in range(len(data)):
            answers.append(self.full_pass(data[x], labels[x], rate))
        return array(answers)

    def epoch_test(self, data):
        answers = []
        for x in range(len(data)):
            answers.append(self.calculate(data[x]))
        return array(answers)

    def batch_pass(self, inp, target):
        answer = self.calculate(inp)
        self.backpropagate(inp, target)
        return answer

    def batch_epoch(self, data, labels, rate, sample_size):
        indexes = random.randint(0, len(data), sample_size)
        data_sample = data[indexes]
        label_sample = labels[indexes]
        answers = []
        for x in range(len(data_sample)):
            answers.append(self.batch_pass(data_sample[x], label_sample[x]))
        self.adjust(rate / sample_size)
        return array(answers), indexes

    def full_batch_epoch(self, data, labels, rate):
        answers = []
        for x in range(len(data)):
            answers.append(self.batch_pass(data[x], labels[x]))
        self.adjust(rate / len(data))
        return array(answers)

    def display(self):
        for x in range(len(self.layer)):
            print(f"layer {x}:")
            self.layer[x].display()
            print()

    def save(self, path):
        text = "\n".join([X.save()  for X in self.layer])
        with open(f"{path}.txt", "w") as file:
            file.write(text)

    def load(self, path):
        with open(f"{path}.txt", "r") as file:
            text = file.read()
        text = text.split("\n")
        self.layer = [Layer(None, None, None, X)  for X in text]
