import tensorflow as tf
from helper import Helper #w tym pliku znajduja sie funkcje pomocnicze

class my_model():

    def __init__(self, epoch):

        self.dropout = 0.5
        self.categories = 4
        self.helper = Helper()
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 32
        self.epochs = epoch
        self.train_loss= []

        self.shapes = [
            [3, 3, 1, 32],   #[filter_h, filter_w, in, out]
            [5, 5, 32, 64],
            [8 * 8 * 64, 512],
            [512, self.categories]
        ]

        # A 3×3 kernel with a dilation rate of 2
        #will have the same view field of a 5×5 kernel.
        #This will increase our field of perception but not
        #increase our computational cost.

        self.weights = []
        for i in range(len(self.shapes)):
            self.weights.append(self.helper.get_tfVariable(self.shapes[i], 'weight{}'.format(i)))

        self.bias = []
        for i in range(len(self.shapes)):
            self.bias.append(self.helper.get_tfVariable([1, self.shapes[i][-1]], 'bias{}'.format(i)))

    def run(self, images_input):

        conv1 = self.helper.conv_layer(images_input, self.weights[0], self.bias[0])
        conv1 = tf.nn.relu(conv1)  #Rectified Linear Unit
        pool1 = self.helper.max_pool_layer(conv1)

        conv2 = self.helper.conv_layer(pool1, self.weights[1], self.bias[1])
        conv2 = tf.nn.relu(conv2)
        pool2 = self.helper.max_pool_layer(conv2)

        # print(pool2.shape[2], pool2.shape[1], pool2.shape[3] ) -> 8,8,64

        flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])

        fully = tf.nn.relu(self.helper.fully_connected_layer(flat, self.weights[2], self.bias[2]))

        fully_dropout = tf.nn.dropout(fully, rate=self.dropout)

        pred = self.helper.fully_connected_layer(fully_dropout, self.weights[3], self.bias[3])

        #print(pred)

        return pred

    def trainable_variables(self):

        return self.weights + self.bias
