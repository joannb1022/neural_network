import tensorflow as tf


class Helper():

    def __init__(self):
        self.name = "helper"

    def conv_layer(self, input_x, filters, b):
        y = tf.nn.conv2d(input = input_x, filters = filters, strides=[1, 1, 1, 1], padding='SAME') + b
        return y

    def max_pool_layer(self, x):
        return tf.nn.max_pool2d(input = x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def fully_connected_layer(self, input_layer, w, b):
        y = tf.matmul(input_layer, w) + b
        return y

    def get_tfVariable(self, shape, name):

        # Trainable = True -> GradientTapes automatically watch uses of this variable
        return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name=name, trainable=True, dtype=tf.float32)  #Initializer that generates tensors with a normal distribution.

    def loss_function(self, pred, target):

        # Measures the probability error in discrete classification tasks
        #in which the classes are mutually exclusive (each entry is in exactly one class).

        return tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target), logits=pred)


    def loss_function(self,  pred , target ):
        return tf.losses.categorical_crossentropy( target , pred )


"""
OPISY WARSTW:

Convolutional Layer - bierze wszystkie piksele w obrazie a poprzez zastosowane filtry powstają reprezentacje
poszczgólnych części obrazów (tz. "feature maps"). Przy tworzeniu tej warstwy ustalamy rozmiar filtra -
najczęściej jest to 3. Ważne jest, aby "głębokość" filtra i obrazu zgadzały się. Kolejnym parametrem jest "stride", który
informuje, o jak dużo pikseli filtr "porusza się" po obrazie. Ostatnim parametrem jest "padding" decydujący o wyjściowym rozmiarze obrazu.
Wartosć 'SAME' oznacza, że rozmiar nie zmieni się, co wydaje się łatwiejszą opcją do zarządzania.

Pooling Layer - ta warstwa przetwarza informacje i kompresuje je. Koncentruje się na najważniejszych częściach obrazu. Kryterium na podstawie,
którego są one wybierane najczęści opiera się na tzw. max pooling. Polega to na pobieraniu makymalnej wartości piksela z obrębu jednego filtra.

Flattening - aby ostatnie warstwy modelu były w stanie przetworzyć dane, należy je "spłaszczyć". 


"""
