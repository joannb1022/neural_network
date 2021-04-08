import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2


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
    return x,y


def show_random_shapes(x, y, number, p = None):
    indices = np.random.choice(range(0, x.shape[0]), number)
    y = np.argmax(y, axis=1)
    if p is None:
        p = y
    plt.figure(figsize=(10, 5))
    for i, index in enumerate(indices):
        plt.subplot(2, 5, i+1)
        print(x[index].size)
        plt.imshow(x[index].reshape((28, 28)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
        if y[index] == p[index]:
            col = 'g'
        else:
            col = 'r'
        plt.xlabel(str(p[index]), color=col)
    return plt

# # Normalize the data
# x_train = np.array(x_train) / 255
# x_val = np.array(x_val) / 255
