from numpy import absolute, array

from _general_2.network import Network


data = array([[0, 0], [0, 1], [1, 0], [1, 1]])
label = array([[0.1], [0.9], [0.9], [0.1]])
human_label = array([[0], [1], [1], [0]])

rate = 0.01
epochs = 10000
interval = 1000

load = False
load_path = "xor_auto-save"
save = True
save_path = load_path

def print_test(x, data, label):
    answer = network.epoch_test(data)
    individual = absolute(answer - human_label)
    average = individual.sum() / len(data)
    print(f"{x + 1}:\t{round(average, 4)}\t{[round(X[0], 4)  for X in individual]}")

if load:
    network = Network(None, None, None, in_path)
else:
    network = Network([2, 2, 1], ["relu", "sigmoid"], [0.5, 0.5])

print_test(-1, data, human_label)
for x in range(epochs):
    network.online_epoch(data, label, rate)
    if (x + 1) % interval == 0:
        print_test(x, data, human_label)

network.save(save_path)
