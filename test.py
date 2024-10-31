from numpy import array
from random import randint

from _general.network import Network
from mnist.mnist_reader import Mnist_reader


main_path = "mnist\\data"
reader = Mnist_reader(main_path)
train_img, train_lab, test_img, test_lab = reader.read_all()


i = randint(1, len(train_img))
network = Network([784, 350, 100, 10], ["relu", "relu", "nothing"], [1, 1, 1], "softmax")
answer = network.online_epoch(train_img[i-1:i], train_lab[i-1:i], 0.01)
print(answer)
