import tensorflow as tf
from data import Data
from tensorflow_model import my_model
import numpy as np

def one_hot_encode(vec):

  n = len(vec)
  out = np.zeros((n, 4))
  for i in range(n):
    out[i, vec[i]] = 1

  return out

data = Data()
train_images, train_labels = data.train_data_with_labels
test_images, test_labels = data.test_data_with_labels

train_images = train_images / 255.0
train_labels = one_hot_encode(train_labels)

test_images = test_images / 255.0
test_labels = one_hot_encode(test_labels)



model = my_model()

def loss_function(y_pred,y_true):

    return tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true),logits=y_pred)


def get_next_batch(batch_size, i, images, labels):
  x = images[i:i+batch_size]
  y = labels[i:i+batch_size]
  #self.i = (self.i + batch_size) % len(self.training_images)
  return x, y

optimizer = tf.optimizers.Adam(learning_rate=0.001)

def train_step( model, x_input , y_true, epoch):

  epoch_accuracy = None
  epoch_loss_avg = None

  with tf.GradientTape() as tape:

    # Get the predictions
    preds = model.run(x_input)

    # Calc the loss
    current_loss = loss_function(preds, y_true)

    # Get the gradients
    grads = tape.gradient(current_loss, model.trainable_variables())

    # Update the weights
    optimizer.apply_gradients(zip(grads, model.trainable_variables()))

    if epoch%100 == 0:

      y_pred = model.run(test_images)
      matches  = tf.equal(tf.math.argmax(y_pred,1), tf.math.argmax(test_labels,1))

      epoch_accuracy = tf.reduce_mean(tf.cast(matches,tf.float32))
      epoch_loss_avg = tf.reduce_mean(current_loss)

      print("--- On epoch {} ---".format(epoch))
      tf.print("Accuracy: ", epoch_accuracy, "| Loss: ",epoch_loss_avg)
      print("\n")

    return epoch_accuracy,epoch_loss_avg



num_epochs = 5000
batch_size = 100

train_loss_results = []
train_accuracy_results = []

for epoch in range(num_epochs):

  # Get next batch
  batch_x, batch_y = get_next_batch( batch_size, epoch, train_images, train_labels)

  # Train the model
  epoch_accuracy, epoch_loss_avg = train_step(model, batch_x, batch_y, epoch)

  if(epoch_loss_avg is not None):
    train_loss_results.append(epoch_loss_avg)
    train_accuracy_results.append(epoch_accuracy)

plt.plot(train_loss_results)
plt.title('Loss')
plt.show()
plt.title('Accuracy')
plt.plot(train_accuracy_results)
plt.show()
