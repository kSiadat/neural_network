from mnist.mnist_reader import Mnist_reader
from _general.network import Network
from _general.utils import error_rate


main_path = "mnist\\data"
reader = Mnist_reader(main_path)
train_img, train_lab, test_img, test_lab = reader.read_all()

rate = 0.001
epochs = 10000
interval = 1000
test_interval = 1000
sample_size = 20

load = True
load_path = "mnist_auto-save"
save = True
save_path = "mnist_auto-save"

if load:
    network = Network(None, None, None, None, load_path)
else:
    network = Network([784, 350, 10], ["relu", "nothing"], [1, 1], "softmax")

test_answer, indexes = network.batch_epoch_test(test_img, 1000)
test_loss = network.loss_function(test_lab[indexes], test_answer)
test_error = error_rate(test_answer, test_lab[indexes])
print(f"0:\t{test_loss.mean().round(4)}\t{test_error}")

for x in range(epochs):
    train_answer, indexes = network.batch_epoch(train_img, train_lab, rate, sample_size)
    if (x + 1) % interval == 0:
        train_loss = network.loss_function(train_lab[indexes], train_answer)
        train_error = error_rate(train_answer, train_lab[indexes])
        if (x + 1) % test_interval == 0:
            test_answer, indexes = network.batch_epoch_test(test_img, 1000)
            test_loss = network.loss_function(test_lab[indexes], test_answer)
            test_error = error_rate(test_answer, test_lab[indexes])
            print(f"{x+1}:\t{train_loss.mean().round(4)}\t{train_error}\t{test_loss.mean().round(4)}\t{test_error}\t{test_loss.mean(axis=0).round(4)}")
        else:
            print(f"{x+1}:\t{train_loss.mean().round(4)}\t{train_error}")

if save:
    network.save(save_path)
