import tensorflow as tf


class my_model():

  def __init__(self):

    self.pool_size = 2
    self.dropout = 0.5
    self.nclasses = 4

    self.shapes = [
    [5, 5, 1, 32],
    [5, 5, 32, 64],
    [8*8*64,512],
    [512, self.nclasses]
    ]

    self.weights = []
    for i in range(len(self.shapes)):
      self.weights.append( get_tfVariable(self.shapes[i] , 'weight{}'.format( i ) ) )

    self.bias = []
    for i in range(len(self.shapes)):
      self.bias.append( get_tfVariable([1,self.shapes[i][-1]] , 'bias{}'.format( i ) ) )



  def run(self, x_input):

    conv1 = conv_layer(x_input,self.weights[0],self.bias[0])
    pool1 = maxPool_layer(conv1,poolSize=self.pool_size)

    conv2 = conv_layer(pool1,self.weights[1],self.bias[1])
    pool2 = maxPool_layer(conv2,poolSize=self.pool_size)

    flat1 = tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])

    fully1 = tf.nn.relu(fullyConnected_layer(flat1,self.weights[2],self.bias[2]))

    fully1_dropout = tf.nn.dropout(fully1,rate=self.dropout)

    y_pred = fullyConnected_layer(fully1_dropout,self.weights[3],self.bias[3])

    #print(conv1.shape,pool1.shape,conv2.shape,pool2.shape,flat1.shape,fully1.shape,y_pred.shape)

    return y_pred

  def trainable_variables(self):

    return self.weights + self.bias




def conv_layer(input_x,w,b):

  # input_x -> [batch,H,W,Channels]
  # filter_shape -> [filters H, filters W, Channels In, Channels Out]

  y = tf.nn.conv2d(input=input_x,filters=w,strides=[1,1,1,1],padding='SAME') + b

  y = tf.nn.relu(y)

  return y


def maxPool_layer(x,poolSize):
  # x -> [batch,H,W,Channels]

  return tf.nn.max_pool2d(input=x,ksize=[1,poolSize,poolSize,1],strides=[1,poolSize,poolSize,1],padding="SAME")


def fullyConnected_layer(input_layer,w,b):

  y = tf.matmul(input_layer,w) + b

  return y




def get_tfVariable(shape, name):

  return tf.Variable(tf.random.truncated_normal(shape,stddev=0.1), name=name, trainable=True, dtype=tf.float32)
