import tensorflow as tf


class Helper():

    def __init__(self):
        self.name = "helper"

    def conv_layer(self, input_x, w, b):
        y = tf.nn.conv2d(input=input_x, filters=w, strides=[1, 1, 1, 1], padding='SAME') + b
        return y


    def maxPool_layer(self, x, poolSize):
        return tf.nn.max_pool2d(input=x, ksize=[1, poolSize, poolSize, 1], strides=[1, poolSize, poolSize, 1],
                                padding="SAME")


    def fullyConnected_layer(self, input_layer, w, b):
        y = tf.matmul(input_layer, w) + b
        return y

    def get_tfVariable(self, shape, name):
        return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name=name, trainable=True, dtype=tf.float32)

    def loss_function(self, y_pred, y_true):
        return tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true), logits=y_pred)

    def loss_function(self,y_pred, y_true):
        return tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true), logits=y_pred)
