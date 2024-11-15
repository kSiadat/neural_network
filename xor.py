from numpy import absolute, array

from _general.network import Network


data = array([[0, 0], [0, 1], [1, 0], [1, 1]])
label = array([[0.1], [0.9], [0.9], [0.1]])
human_label = array([[0], [1], [1], [0]])

rate = 0.01
epochs = 10000
interval = 1000

load = False
load_path = "xor"
save = True
save_path = "xor_auto-save"

def print_test(x, data, label):
    answer = network.epoch_test(data)
    individual = absolute(answer - human_label)
    average = individual.sum() / len(data)
    print(f"{x + 1}:\t{round(average, 4)}\t{[round(X[0], 4)  for X in individual]}")

if load:
    network = Network(None, None, load_path)
    #network = Network(None, None, None, None, load_path)
else:
    layer_data = [
        ["normal", [2, 2, "relu",    0.5]],
        ["normal", [2, 1, "sigmoid", 0.5]],
        ]
    network = Network(layer_data, "squared")
    #network = Network([2, 2, 1], ["relu", "sigmoid"], [0.5, 0.5], "squared")

print_test(-1, data, human_label)
for x in range(epochs):
    network.online_epoch(data, label, rate)
    if (x + 1) % interval == 0:
        print_test(x, data, human_label)

if save:
    network.save(save_path)
