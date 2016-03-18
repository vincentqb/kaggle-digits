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

class next_batch():

    last = 0
    epochs = 0

    def __init__(self):
        from random import shuffle
        self.max_index = len(train_dataset)
        self.indices = list(range(self.max_index))
        shuffle(self.indices)

    def __call__(self, sample_size = 50):
        if self.last > self.max_index:
            self.epochs += 1
            self.last = 0

        start = self.last 
        end = self.last = self.last + sample_size 
        indices = self.indices[start:end]

        return (train_dataset[indices], train_labels[indices])

# def next_batch(sample_size = 50, indices = shuffle(range(1,len(train_dataset)+1)):
#     indices = np.random.choice(len(train_dataset), sample_size, replace = False)
#     return (train_dataset[indices], train_labels[indices])

next_batch = next_batch()
images = valid_dataset
labels = valid_labels

first_batch = next_batch(50)
print('Training set', first_batch[0].shape, first_batch[1].shape)
print('Validation set', images.shape, labels.shape)
print('Test set', test_dataset.shape)

# Tensor flow includes MNIST

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 
# next_batch = mnist.train.next_batch
# images = mnist.test.images
# labels = mnist.test.labels
# 
# first_batch = next_batch(50)
# print('Training set', first_batch[0].shape, first_batch[1].shape)
# print('Validation set', images.shape, labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(20000):
  batch = next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print(tf.argmax(y,1).eval(feed_dict={x: images}))
print("test accuracy %g"%accuracy.eval(feed_dict={x: images, y_: labels, keep_prob: 1.0}))

# Save predictions
test_frame['ImageId'] = range(1, len(test_dataset)+1)
test_frame['Label'] = tf.argmax(y_conv,1).eval(feed_dict={x: test_dataset})
test_frame.to_csv('predict.csv', columns = ('ImageId', 'Label'), index = None)
