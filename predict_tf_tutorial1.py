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
  # image_size = 28
  num_labels = 10
  # num_channels = 1 # grayscale
  # dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  if labels is not None:
      labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
      return dataset, labels
  else:
      return dataset

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset = reformat(test_dataset)

def next_batch(sample_size = 50):
    indices = np.random.choice(len(train_dataset), sample_size)
    return (train_dataset[indices], train_labels[indices])

next_batch = next_batch
images = valid_dataset
labels = valid_labels

# Tensor flow includes MNIST

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# next_batch = mnist.train.next_batch
# images = mnist.test.images
# labels = mnist.test.labels

# print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', images.shape, labels.shape)
# print('Test set', test_dataset.shape)

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

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: images, y_: labels}))
