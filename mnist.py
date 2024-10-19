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
train_img = train_img
train_lab = train_lab
test_img = test_img[:100]
test_lab = test_lab[:100]

#network = Network([784, 350, 10], ["leaky_relu", "sigmoid"])
network = Network(path="auto_save")
rate = 0.0001
epochs = 10000
interval = 10
test_interval = 100
sample_size = 20

for x in range(epochs):
    train_answers, indexes = network.batch_epoch(train_img, train_lab, rate, sample_size)
    if (x + 1) % interval == 0:
        train_error = error_rate(train_answers, train_lab[indexes])
        if (x + 1) % test_interval == 0:
            test_answers = network.epoch_test(test_img)
            test_error = error_rate(test_answers, test_lab)
            print(f"{x+1}: \t {train_error} \t {test_error}")
        else:
            print(f"{x+1}: \t {train_error}")
        
network.save("auto_save")
