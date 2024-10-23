from numpy import absolute, array

from _general.network import Network


data = array([[0, 0], [0, 1], [1, 0], [1, 1]])
label = array([[0.1], [0.9], [0.9], [0.1]])
human_label = array([[0], [1], [1], [0]])
rate = 0.01

epochs = 10000
interval = 1000

def print_test(x, data, label):
    answer = network.epoch_test(data)
    individual = absolute(answer - human_label)
    average = individual.sum() / len(data)
    print(f"{x + 1}:\t{round(average, 4)}\t{[round(X[0], 4)  for X in individual]}")

if False: # xor
    network = Network([2, 2, 1], ["relu", "sigmoid"], [0.5, 0.5])
    #network = Network(None, None, None, "test_4")

    print_test(-1, data, human_label)
    for x in range(epochs):
        network.online_epoch(data, label, rate)
        if (x + 1) % interval == 0:
            print_test(x, data, human_label)
    network.save("test_4")

network = Network([2, 2, 1], ["relu", "relu"], [1, 1])
network.batch_epoch(data, label, rate, 2)
