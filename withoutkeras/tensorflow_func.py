import tensorflow as tf
from data import Data

data = Data()
x, y = data.test_data_with_labels
print(x)


def train_step( model, x_input , y_true, epoch):

  epoch_accuracy = None
  epoch_loss_avg = None

  with tf.GradientTape() as tape:

    # Get the predictions
    preds = model.run(x_input)

    # Calc the loss
    current_loss = loss_function(preds,y_true)

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
  batch_x, batch_y = next_batch(epoch, batch_size)

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
