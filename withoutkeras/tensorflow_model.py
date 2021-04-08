import tensorflow as tf
from helper import Helper #w tym pliku znajduja sie funkcje pomocnicze

class my_model():

    def __init__(self):

        self.pool_size = 2
        self.dropout = 0.5
        self.categories = 4
        self.helper = Helper()
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 32 #to tak z dupy, mozesz zmieniac
        self.epochs = 50
        self.train_loss= []

        self.shapes = [
            [5, 5, 1, 32],
            [5, 5, 32, 64],
            [8 * 8 * 64, 512],
            [512, self.categories]
        ]

        self.weights = []
        for i in range(len(self.shapes)):
            self.weights.append(self.helper.get_tfVariable(self.shapes[i], 'weight{}'.format(i)))

        self.bias = []
        for i in range(len(self.shapes)):
            self.bias.append(self.helper.get_tfVariable([1, self.shapes[i][-1]], 'bias{}'.format(i)))

    def run(self, x_input):

        conv1 = self.helper.conv_layer(x_input, self.weights[0], self.bias[0])
        conv1 = tf.nn.relu(conv1)
        pool1 = self.helper.maxPool_layer(conv1, poolSize=self.pool_size)

        conv2 = self.helper.conv_layer(pool1, self.weights[1], self.bias[1])
        conv2 = tf.nn.relu(conv2)
        pool2 = self.helper.maxPool_layer(conv2, poolSize=self.pool_size)

        flat1 = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])

        fully1 = tf.nn.relu(self.helper.fullyConnected_layer(flat1, self.weights[2], self.bias[2]))

        fully1_dropout = tf.nn.dropout(fully1, rate=self.dropout)

        y_pred = self.helper.fullyConnected_layer(fully1_dropout, self.weights[3], self.bias[3])

        return y_pred

    def trainable_variables(self):

        return self.weights + self.bias
