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

    epoch_loss_avg = None

    with tf.GradientTape() as tape:
        preds = model.run(x_input)
        loss = helper.loss_function(preds, y_true)
        grads = tape.gradient(loss, model.trainable_variables())
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables()))


        y_pred = model.run(test_images)
        matches = tf.equal(tf.math.argmax(y_pred, 1), tf.math.argmax(test_labels, 1))

        epoch_loss_avg = tf.reduce_mean(loss)

        print(" On epoch {}".format(epoch))
        tf.print( "Loss: ", epoch_loss_avg)
        print("\n")

        return  epoch_loss_avg

def train(model, plots):
    data = Data()
    train_images, train_labels = data.train_data_with_labels
    test_images, test_labels = data.test_data_with_labels

    for epoch in range(model.epochs):
        batch_data, batch_labels = get_next_batch(model.batch_size, epoch, train_images, train_labels)
        epoch_loss_avg = train_step(model, batch_data, batch_labels, epoch, test_images, test_labels)
        model.train_loss.append(epoch_loss_avg)

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
        # if y[index]==pred[index]:
        #     col = 'g'
        # else:
        #     col = 'r'
        plt.imshow(x[index].reshape((32, 32)) ,cmap='binary')
        plt.xlabel(data.categories[pred[index]])
        # plt.xlabel(pred[index], col=col)
        plt.xticks([])
        plt.yticks([])
        plt.draw()
    return plt
