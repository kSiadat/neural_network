from numpy import absolute, array, concatenate, logical_and
from ucimlrepo import fetch_ucirepo

from _general.network import Network


text_class = {
    "Iris-setosa":     [1, 0, 0],
    "Iris-versicolor": [0, 1, 0],
    "Iris-virginica":  [0, 0, 1],
    }
iris = fetch_ucirepo(id=53) # will fail without internet
all_data = array(iris.data.features)
all_label = array([text_class[X[0]]  for X in array(iris.data.targets)])

train_data =  concatenate((all_data[:40],  all_data[50:90],  all_data[100:140]))
train_label = concatenate((all_label[:40], all_label[50:90], all_label[100:140]))
test_data =  concatenate((all_data[40:50],  all_data[90:100],  all_data[140:150]))
test_label = concatenate((all_label[40:50], all_label[90:100], all_label[140:150]))


rate = 1 
epochs = 200
interval = 10

load = False
load_path = "iris_auto-save"
save = False
save_path = "iris_auto-save"

if load:
    network = Network(None, None, None, load_path)
else:
    network = Network([4, 6, 3], ["relu", "sigmoid"], [0.25, 0.2])

answer = network.epoch_test(test_data)
test_loss = absolute(answer - test_label).mean(axis=0)
print(f"0:\t{test_loss.mean().round(4)}\t{test_loss.round(4)}")

for x in range(epochs):
    answer = network.epoch(train_data, train_label, rate)
    if (x + 1) % interval == 0:
        train_loss = absolute(answer - train_label).mean(axis=0)
        answer = network.epoch_test(test_data)
        test_loss = absolute(answer - test_label).mean(axis=0)
        print(f"{x + 1}:\t{train_loss.mean().round(4)}\t{train_loss.round(4)}\t{test_loss.mean().round(4)}\t{test_loss.round(4)}")

if save:
    network.save(save_path)
