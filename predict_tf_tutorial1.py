# https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html

# from time import clock
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Read training data
train_frame = pd.read_csv('data/train.csv')

# Make random train and validation sets
from sklearn.cross_validation import train_test_split
train_frame, valid_frame = train_test_split(train_frame, test_size = 0.2)

train_labels = train_frame['label'].values
train_dataset = train_frame.iloc[:,1:].values

valid_labels = valid_frame['label'].values
valid_dataset = valid_frame.iloc[:,1:].values

# Read test data 
test_frame = pd.read_csv('data/test.csv')
test_dataset = test_frame.values

def reformat(dataset, labels = None):
  image_size = 28
  num_labels = 10
  # num_channels = 1 # grayscale
  # dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # dataset = dataset.astype(np.float32)
  if labels is not None:
      labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
      # labels = (np.arange(num_labels) == labels[:,None]).astype(np.int32)
      return dataset, labels
  else:
      return dataset

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset = reformat(test_dataset)

# Need normalization
train_dataset = train_dataset / 255
valid_dataset = valid_dataset / 255
test_dataset = test_dataset / 255

class next_batch():

    last = 0
    epochs = 0

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        from random import shuffle
        self.max_index = len(images)
        self.indices = list(range(self.max_index))
        shuffle(self.indices)

    def __call__(self, sample_size = 50):
        if self.last > self.max_index:
            self.epochs += 1
            self.last = 0

        start = self.last 
        end = self.last = self.last + sample_size 
        indices = self.indices[start:end]

        return (self.images[indices], self.labels[indices])

# def next_batch(sample_size = 50, indices = shuffle(range(1,len(train_dataset)+1)):
#     indices = np.random.choice(len(train_dataset), sample_size, replace = False)
#     return (train_dataset[indices], train_labels[indices])

next_batch = next_batch(train_dataset, train_labels)
images = valid_dataset
labels = valid_labels

# first_batch = next_batch(50)
# print(first_batch[1])
# print('Training set', first_batch[0].shape, first_batch[1].shape)
print('Validation set', images.shape, labels.shape)
print('Test set', test_dataset.shape)

# Tensor flow includes MNIST

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# next_batch = next_batch(mnist.train.images, mnist.train.labels.astype(np.float32))
# next_batch = mnist.train.next_batch
# images = mnist.test.images
# labels = mnist.test.labels

# next_batch = next_batch(mnist.train.images[:len(train_labels)], train_labels)

first_batch = next_batch(50)
print('Training set', first_batch[0].shape, first_batch[1].shape)
print('Validation set', images.shape, labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
  batch = next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print(tf.argmax(y,1).eval(feed_dict={x: images}))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: images, y_: labels}))

# Save predictions
test_frame['ImageId'] = range(1, len(test_dataset)+1)
test_frame['Label'] = tf.argmax(y,1).eval(feed_dict={x: test_dataset})
test_frame.to_csv('predict.csv', columns = ('ImageId', 'Label'), index = None)
