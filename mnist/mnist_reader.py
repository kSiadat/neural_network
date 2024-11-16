# Most of the code in this file taken from https://www.kaggle.com/code/hojjatk/read-mnist-dataset

from array import array as py_array
from matplotlib import pyplot as plt
from numpy import array, asarray, reshape, zeros
from random import randint
from struct import unpack


class Mnist_reader:
    def __init__(self, main_path):
        self.train_image_path = f"{main_path}\\train-images.idx3-ubyte"
        self.train_label_path = f"{main_path}\\train-labels.idx1-ubyte"
        self.test_image_path = f"{main_path}\\t10k-images.idx3-ubyte"
        self.test_label_path = f"{main_path}\\t10k-labels.idx1-ubyte"

    def read_pair(self, image_path, label_path, flat=True):
        labels = []
        with open(label_path, "rb") as file:
            magic, size, = unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049 got {magic}")
            label_data = asarray(py_array("B", file.read()))
        labels = zeros([len(label_data), 10])
        for x in range(len(label_data)):
            labels[x][label_data[x]] = 1

        with open (image_path, "rb") as file:
            magic, size, rows, cols = unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051 got {magic}")
            image_data = asarray(py_array("B", file.read()))
        if flat:
            images = image_data.reshape(len(image_data) // 784, 784)
        else:
            images = image_data.reshape([len(image_data) // 784, 28, 28])
        images = images / 255
        
        return images, labels

    def read_all(self, flat=True):
        train_image, train_label = self.read_pair(self.train_image_path, self.train_label_path, flat)
        test_image, test_label = self.read_pair(self.test_image_path, self.test_label_path, flat)
        return train_image, train_label, test_image, test_label


def show_random_images(amount, image_set, label_set):
    images = []
    labels = []
    for x in range(amount):
        i = randint(0, len(image_set))
        images.append(image_set[i])
        labels.append(f"img: {i + 1}, value: {label_set[i]}")

    cols = 5
    rows = int((amount / cols) + 1)
    plt.figure(figsize=(30, 20))
    for x in range(amount):
        plt.subplot(rows, cols, x + 1)
        plt.imshow(images[x], cmap=plt.cm.gray)
        plt.title(labels[x], fontsize = 15)
    plt.show()
        

if __name__ == "__main__":
    main_path = "data"
    reader = Mnist_reader(main_path)
    train_img, train_lab, test_img, test_lab = reader.read_all(False)
    show_random_images(10, train_img, train_lab)
