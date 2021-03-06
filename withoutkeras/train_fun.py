import tensorflow as tf
from data import Data
from tensorflow_model import my_model
import numpy as np
import matplotlib.pyplot as plt
import window
from helper import Helper
import random

helper = Helper() #pomocnicze funkcje znajduja sie w jednym pliku

def get_next_batch(batch_size, i, images, labels):
    x = images[i:i + batch_size]
    y = labels[i:i + batch_size]
    return x, y


def train_step(model, x_input, y_true, epoch, test_images, test_labels):

    avg_loss = None

    """
    tf.GradientTape oblicza gradient w odniesieniu do danych wejściowych (zwykle tf.Variables)
    i zapisuje na "taśmie". Potem używana jest ona do obliczania straty w stosunku do zmiennych modelu
    (w tym przypdaku wag i bias)
    """

    with tf.GradientTape() as tape:
        preds = model.run(x_input)
        loss = helper.loss_function(preds, y_true)
        grads = tape.gradient(loss, model.trainable_variables())
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables()))

        pred = model.run(test_images)
        avg_loss = tf.reduce_mean(loss)  #oblicza srednia elementow w tensorze

        print(" On epoch {}".format(epoch))
        tf.print( "Loss: ", avg_loss)
        print("\n")

        return avg_loss

def train(model, plots):
    data = Data()
    train_images, train_labels = data.train_data_with_labels
    test_images, test_labels = data.test_data_with_labels

    for epoch in range(model.epochs):
        batch_data, batch_labels = get_next_batch(model.batch_size, epoch, train_images, train_labels)
        avg_loss = train_step(model, batch_data, batch_labels, epoch, test_images, test_labels)
        model.train_loss.append(avg_loss)

    if plots:
        plt.plot(model.train_loss)
        plt.title('Loss')
        plt.show()

def recognize(model):
    data = Data()
    #x, y = data.recognize_data_with_labels #paint_data
    x,y = data.test_data_with_labels
    pred = model.run(x)
    pred =np.argmax(pred, axis = 1)
    plt.figure(figsize=(10, 5))
    num = 10
    for i in range(num):
        plt.subplot(2, 5, i+1)
        index = random.randint(0,x.shape[0]-1)
        plt.imshow(x[index].reshape((32, 32)) ,cmap='binary')
        plt.xlabel(data.categories[pred[index]])
        plt.xticks([])
        plt.yticks([])
        plt.draw()
    return plt
