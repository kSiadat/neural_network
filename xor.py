from numpy import absolute, array

from _general.network import Network


def evaluate_2(answers, labels):
        loss = absolute(answers - labels)
        pass


def evaluate(answers, labels):
        l = len(labels)
        clean = []
        messy = []
        for x in range(l):
            loss = abs(answers[x][0] - labels[x][0])
            messy.append(loss)
            if loss < 0.5:
                clean.append(1)
            else:
                clean.append(0)
        return sum(clean) / l, round(sum(messy), 5) / l, clean, [round(X, 3)  for X in messy]

#net = Network([2, 2, 1], ["leaky_relu", "sigmoid"])
net = Network(path="xor")
data = array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = array([[0.1], [0.9], [0.9], [0.1]])
labels2 = array([[0], [1], [1], [0]])
training = True

if training:
    rate = 0.00005
    epoch_count = 5000000
    interval = 10000
    answers = net.epoch_test(data)
    clean_avg, messy_avg, clean_ind, messy_ind = evaluate(answers, labels2)
    print("0\t", clean_avg, "\t", messy_avg, "\t", clean_ind, "\t", messy_ind)
    for x in range(epoch_count):
        answers = net.epoch(data, labels, rate)
        if ((x + 1) % interval == 0):
            clean_avg, messy_avg, clean_ind, messy_ind = evaluate(answers, labels2)
            print(x + 1, "\t", clean_avg, "\t", messy_avg, "\t", clean_ind, "\t", messy_ind)
else:
    answers = net.epoch_test(data)
    clean_avg, messy_avg, clean_ind, messy_ind = evaluate(answers, labels2)
    print(clean_avg, "\t", messy_avg, "\t", clean_ind, "\t", messy_ind)

net.save("auto_save")
