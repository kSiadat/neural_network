from mnist.mnist_reader import Mnist_reader
from _general.network import Network
from _general.utils import error_rate


def full_test():
    answer = network.epoch_test(test_img)
    loss = network.loss_function(test_lab, answer)
    error = error_rate(test_lab, answer)
    return loss, error

def part_test():
    answer, indexes = network.batch_epoch_test(test_img, test_sample_size)
    loss = network.loss_function(test_lab[indexes], answer)
    error = error_rate(test_lab[indexes], answer)
    return loss, error


# settings
train = True

rate = 0.001
epochs = 1000
sample_size = 20

interval = 100
test_interval = 100
test_sample_size = 0

load = False
load_path = "mnist_conv_4-6"
save = True
save_path = "mnist_auto-save"

convolutional = False
# end settings


main_path = "mnist\\data"
reader = Mnist_reader(main_path)
if convolutional:
    dimensions = 3
else:
    dimensions = 1
train_img, train_lab, test_img, test_lab = reader.read_all(dimensions)

if load:
    network = Network(None, None, load_path)
else:
    if not convolutional:
        layer_data = [
            ["normal", [784, 350, "relu", 1]],
            ["normal", [350, 10, "nothing", 1]],
            ]
    else:
        layer_data = [
            ["convolutional", [[28, 28, 1], [4, 5, 5, 1], 1, 0, "relu", 1]],
            ["pooling", [[24, 24, 4], 2]],
            ["convolutional", [[12, 12, 4], [6, 3, 3, 4], 1, 0, "relu", 1]],
            ["pooling", [[10, 10, 6], 2]],
            ["converter", [[5, 5, 6], [150]]],
            ["normal", [150, 10, "nothing", 1]],
            ]
    network = Network(layer_data, "softmax")

if train:
    if test_sample_size > 0:
        test_loss, test_error = part_test()
    else:
        test_loss, test_error = full_test()
    print("epoch\terror\tloss\terror\tloss\tloss")
    print(f"0:\t\t\t{test_error}\t{test_loss.mean().round(4)}")

    for x in range(epochs):
        train_answer, indexes = network.batch_epoch(train_img, train_lab, rate, sample_size)
        #train_answer = network.online_epoch(train_img, train_lab, rate)
        #train_answer = network.epoch(train_img, train_lab, rate)
        if (x + 1) % interval == 0:
            train_loss = network.loss_function(train_lab[indexes], train_answer)
            train_error = error_rate(train_lab[indexes], train_answer)
            #train_loss = network.loss_function(train_lab, train_answer)
            #train_error = error_rate(train_lab, train_answer)
            if (x + 1) % test_interval == 0:
                if test_sample_size > 0:
                    test_loss, test_error = part_test()
                else:
                    test_loss, test_error = full_test()
                print(f"{x+1}:\t{train_error.round(4)}\t{train_loss.mean().round(4)}\t{test_error}\t{test_loss.mean().round(4)}\t{test_loss.mean(axis=0).round(4)}")
            else:
                print(f"{x+1}:\t{train_error}\t{train_loss.mean().round(4)}")

    if save:
        network.save(save_path)

else:
    loss, error = full_test()
    print("error\tloss\tloss")
    print(f"{error}\t{loss.mean().round(4)}\t{loss.mean(axis=0).round(4)}")
