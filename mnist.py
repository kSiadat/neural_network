from numpy import logical_and

from mnist.mnist_reader import Mnist_reader
from _general.network import Network


def error_rate(answers, labels):
    is_max = answers == answers.max(axis=1)[:, None]
    single_max = is_max.sum(axis=1) == 1
    correct = logical_and(labels, is_max).sum(axis=1)
    perfect = logical_and(correct, single_max)
    return (len(labels) - sum(perfect)) / len(labels)


main_path = "mnist\\data"
train_image_path = f"{main_path}\\train-images.idx3-ubyte"
train_label_path = f"{main_path}\\train-labels.idx1-ubyte"
test_image_path = f"{main_path}\\t10k-images.idx3-ubyte"
test_label_path = f"{main_path}\\t10k-labels.idx1-ubyte"
reader = Mnist_reader(train_image_path, train_label_path, test_image_path, test_label_path)
train_img, train_lab, test_img, test_lab = reader.read_all()

rate = 0.0001
epochs = 10000
interval = 100
test_interval = 100
sample_size = 20

load = True
load_path = "auto_save"
save = True
save_path = "mnist_auto-save"

if load:
    network = Network(None, None, None, load_path)
else:
    network = Network([784, 350, 10], ["leaky_relu", "sigmoid"], [0.001, 0.002])

for x in range(epochs):
    train_answers, indexes = network.batch_epoch(train_img, train_lab, rate, sample_size)
    if (x + 1) % interval == 0:
        train_error = error_rate(train_answers, train_lab[indexes])
        if (x + 1) % test_interval == 0:
            test_answers, indexes = network.batch_epoch_test(test_img, 100)
            test_error = error_rate(test_answers, test_lab[indexes])
            print(f"{x+1}: \t {train_error} \t {test_error}")
        else:
            print(f"{x+1}: \t {train_error}")

if save:
    network.save(save_path)
