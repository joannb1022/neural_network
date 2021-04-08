import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import preparedata
import random
from preparedata import create_data, create_data_with_labels, show_random_shapes


class Data:
    def __init__(self):
        self.categories = ["square", "circle", "triangle", "star"]
        self.dir_name_train = "C:/Users/chleb/Desktop/neural_network-master/train"
        self.dir_name_test = "C:/Users/chleb/Desktop/neural_network-master/test"
        self.img_size = 32
        self.train_data = create_data(self.categories, self.img_size, self.dir_name_train)
        self.train_data_with_labels = create_data_with_labels(self.train_data, self.img_size)
        self.test_data = create_data(self.categories, self.img_size, self.dir_name_test)
        self.test_data_with_labels = create_data_with_labels(self.test_data, self.img_size)
        self.model = None
        self.next = False
