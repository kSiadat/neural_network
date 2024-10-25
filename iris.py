from numpy import absolute, array, concatenate
from ucimlrepo import fetch_ucirepo

from _general.network import Network
from _general.utils import error_rate


def save_mean_std(path, data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    text_mean = ",".join([str(X)  for X in mean])
    text_std  = ",".join([str(X)  for X in std])
    with open(f"{path}.txt", "w") as file:
        file.write(text_mean + "\n" + text_std)

def load_mean_std(path):
    with open(f"{path}.txt", "r") as file:
        text = file.read()
    text = text.split("\n")
    mean = [float(X)  for X in text[0].split(",")]
    std  = [float(X)  for X in text[1].split(",")]
    return mean, std


mean_std_path = "iris//mean_std"
text_class = {
    "Iris-setosa":     [1, 0, 0],
    "Iris-versicolor": [0, 1, 0],
    "Iris-virginica":  [0, 0, 1],
    }

iris = fetch_ucirepo(id=53) # will fail without internet
all_data = array(iris.data.features)
all_label = array([text_class[X[0]]  for X in array(iris.data.targets)])
mean, std = load_mean_std(mean_std_path)
all_data = (all_data - mean) / std

train_data =  concatenate((all_data[:40],  all_data[50:90],  all_data[100:140]))
train_label = concatenate((all_label[:40], all_label[50:90], all_label[100:140]))
test_data =  concatenate((all_data[40:50],  all_data[90:100],  all_data[140:150]))
test_label = concatenate((all_label[40:50], all_label[90:100], all_label[140:150]))


rate = 0.05
epochs = 10000
interval = 1000

load = True
load_path = "iris"
save = True
save_path = "iris_auto-save"

if load:
    network = Network(None, None, None, None, load_path)
else:
    network = Network([4, 6, 3], ["relu", "nothing"], [0.25, 0.2], "softmax")

answer = network.epoch_test(test_data)
test_loss = network.loss_function(test_label, answer).mean(axis=0)
test_error = error_rate(answer, test_label)
print(f"0:\t{test_error.round(2)}\t{test_loss.mean().round(4)}\t{test_loss.round(4)}")

for x in range(epochs):
    answer = network.epoch(train_data, train_label, rate)
    if (x + 1) % interval == 0:
        train_loss = network.loss_function(train_label, answer).mean(axis=0)
        train_error = error_rate(answer, train_label)
        answer = network.epoch_test(test_data)
        test_loss = network.loss_function(test_label, answer).mean(axis=0)
        test_error = error_rate(answer, test_label)
        print(f"{x + 1}:\t{train_error.round(3)}\t{train_loss.mean().round(4)}\t{train_loss.round(4)}\t{test_error.round(2)}\t{test_loss.mean().round(4)}\t{test_loss.round(4)}")

if save:
    network.save(save_path)
