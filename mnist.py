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


main_path = "mnist\\data"
reader = Mnist_reader(main_path)
train_img, train_lab, test_img, test_lab = reader.read_all()

# settings
train = False

rate = 0.001
epochs = 5
sample_size = 20

interval = 1
test_interval = 1
test_sample_size = 0

load = True
load_path = "mnist"
save = False
save_path = "mnist_auto-save"
# end settings

if load:
    network = Network(None, None, load_path)
else:
    layer_data = [
        ["normal", [784, 350, "relu", 1]],
        ["normal", [350, 10, "nothing", 1]],
        ]
    network = Network(layer_data, "softmax")

if train:
    if test_sample_size > 0:
        test_loss, test_error = part_test()
    else:
        test_loss, test_error = full_test()
    print(f"0:\t\t\t{test_loss.mean().round(4)}\t{test_error}")

    for x in range(epochs):
        #train_answer, indexes = network.batch_epoch(train_img, train_lab, rate, sample_size)
        #train_answer = network.online_epoch(train_img, train_lab, rate)
        train_answer = network.epoch(train_img, train_lab, rate)
        if (x + 1) % interval == 0:
            #train_loss = network.loss_function(train_lab[indexes], train_answer)
            #train_error = error_rate(train_lab[indexes], train_answer)
            train_loss = network.loss_function(train_lab, train_answer)
            train_error = error_rate(train_lab, train_answer)
            if (x + 1) % test_interval == 0:
                if test_sample_size > 0:
                    test_loss, test_error = part_test()
                else:
                    test_loss, test_error = full_test()
                print(f"{x+1}:\t{train_loss.mean().round(4)}\t{train_error.round(4)}\t{test_loss.mean().round(4)}\t{test_error}\t{test_loss.mean(axis=0).round(4)}")
            else:
                print(f"{x+1}:\t{train_loss.mean().round(4)}\t{train_error}")

    if save:
        network.save(save_path)

else:
    loss, error = full_test()
    print(f"{loss.mean().round(4)}\t{error}\t{loss.mean(axis=0).round(4)}")
