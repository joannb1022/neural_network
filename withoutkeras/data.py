import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2

class Data():
    def __init__(self):
        self.categories = ["square", "circle", "triangle", "star"]
        self.dir_name_train = "./train"
        self.dir_name_test = "./test"
        self.dir_name_rec = "./paint_data"
        self.img_size = 32
        self.train_data = create_data(self.categories, self.img_size, self.dir_name_train)
        self.train_data_with_labels = create_data_with_labels(self.train_data, self.img_size)
        self.test_data = create_data(self.categories, self.img_size, self.dir_name_test)
        self.test_data_with_labels = create_data_with_labels(self.test_data, self.img_size)
        self.rec_data = create_data(self.categories, self.img_size, self.dir_name_rec)
        self.recognize_data_with_labels = create_data_with_labels(self.rec_data, self.img_size)
        self.next = False


def one_hot_encode(vec):
    n = len(vec)
    out = np.zeros((n, 4))
    for i in range(n):
        out[i, vec[i]] = 1

    return out


def create_data(categories, img_size, dir_name):
    data = []
    for category in categories:
        path = os.path.join(dir_name, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            data.append([new_array, class_num])

    return data


def create_data_with_labels(data, img_size):
    random.shuffle(data)
    x = []
    y = []
    for feature, label in data:
        x.append(feature)
        y.append(label)

    x = np.array(x).reshape(-1, img_size,  img_size,  1)
    y = np.array(y)
    x = x / 255.0
    y = one_hot_encode(y)

    return x,y
