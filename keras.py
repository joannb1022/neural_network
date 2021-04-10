import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import preparedata
import random
from preparedata import create_data, create_data_with_labels, show_random_shapes

class Model():
    def __init__(self):
        self.categories = [ "circle", "square","triangle", "star"]
        self.dir_name_train = "./withoutkeras/train"
        self.dir_name_test = "./withoutkeras/test"
        self.img_size = 28
        self.train_data = create_data(self.categories, 28, self.dir_name_train)
        self.train_data_with_labels = create_data_with_labels(self.train_data, self.img_size)
        self.test_data = create_data(self.categories, 28, self.dir_name_test)
        self.test_data_with_labels = create_data_with_labels(self.test_data, self.img_size)
        self.model = None
        self.next = False
    def train(self):

        x,y = self.train_data_with_labels
        x = tf.keras.utils.normalize(x, axis = 1)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape = x.shape[1:]))
        model.add(tf.keras.layers.Dense(units = 128, activation=tf.nn.relu))
        #model.add(tf.keras.layers.Dense(units = 128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units = len(self.categories), activation=tf.nn.softmax))

        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x, y, epochs = 5)

        model.save("network")
        self.model = model

    def predict(self, number):
        fig = plt.figure()
        x, y = self.test_data_with_labels
        pred = self.model.predict(x)
        pred = np.argmax(pred, axis=1)
        print(pred)
        for i in range(number):
            print(i)
            index = random.randint(0,x.shape[0]-1)
            print(index)
            if y[index] == pred[index]:
                col = 'g'
            else:
                col = 'r'
            fig.canvas.mpl_connect('close_event', self.handle_close)
            fig.canvas.set_window_title("{} {}".format("Image", i+1))
            while True:
                #plt.imshow(x[n].reshape((28, 28)))
                plt.imshow(x[index].reshape((28, 28)) ,cmap='binary')
                plt.xlabel(self.categories[pred[index]], color=col)
                plt.xticks([])
                plt.yticks([])
                plt.draw()
                if self.waitforbuttonpress():
                    break
            plt.cla()


    def waitforbuttonpress(self):
        while plt.waitforbuttonpress(0.2) is None:
            if not self.next:
                return False
        return True

    def handle_close(self, evt):
        self.next = True
