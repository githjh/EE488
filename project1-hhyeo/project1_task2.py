# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import numpy as np

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 9])
y_10 = tf.placeholder(tf.float32, shape=[None, 10])

#TODO: remove digit 10 from dataset
#TODO: transfer learning by adding output neuron

# Convolutional layer
x_image = tf.reshape(x, [-1,28,28,1])
W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 30, 500], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# 9 digit - Output layer
W_fc2 = tf.Variable(tf.truncated_normal([500, 9], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[9]))
y_hat=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# 10 digit - Output layer
W_fc2_10 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2_10 = tf.Variable(tf.constant(0.1, shape=[10]))
y_hat_10 = tf.nn.softmax(tf.matmul(h_fc1, W_fc2_10) + b_fc2_10)

# Train and Evaluate the Model
cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cross_entropy_10 = - tf.reduce_sum(y_10*tf.log(y_hat_10))
train_step_10 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_10, var_list = [W_fc2_10, b_fc2_10])
correct_prediction_10 = tf.equal(tf.argmax(y_hat_10,1), tf.argmax(y_10,1))
accuracy_10 = tf.reduce_mean(tf.cast(correct_prediction_10, tf.float32))

removed_integer = 9
# Remove 10 digit inside a batch
def filter_batch(batch_data, batch_label):
    filter_idx = []
    for i in range(len(batch_label)):
        if batch_label[i][removed_integer] == 1:
            filter_idx.append(i)

    #print(filter_idx)

    #delete row for 10 digit
    batch_data = np.delete(batch_data, filter_idx, 0)
    batch_label = np.delete(batch_label, filter_idx, 0)

    #delete column for 10 digit
    #batch_data = np.delete(batch_data, removed_integer, 1)
    batch_label = np.delete(batch_label, removed_integer, 1)

    return batch_data, batch_label

#Train 9 digit
sess.run(tf.initialize_all_variables())
print("=================================")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(5):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        batch_data, batch_label = filter_batch(batch[0], batch[1])

        train_step.run(feed_dict={x: batch_data, y_: batch_label})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x: batch_data, y_: batch_label})
            #TODO: validation set accuracy
            #val_accuracy = accuracy.eval(feed_dict=\
            #    {x: mnist.validation.images, y_:mnist.validation.labels})
            #print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
            print("[9 digit] |%d\t|%d\t|%.4f\t"%(j+1, i+1, train_accuracy))
            #print('[9 digit] train_accuracy: {}'.format(train_accuracy))

# Train 10 digit (transfer learning)
# sess.run(tf.initialize_all_variables())
print("=================================")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(5):
    for i in range(550):
        batch = mnist.train.next_batch(100)

        train_step_10.run(feed_dict={x: batch[0], y_10: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy_10.eval(feed_dict={x: batch[0], y_10: batch[1]})
            val_accuracy = accuracy_10.eval(feed_dict=\
                {x: mnist.validation.images, y_10:mnist.validation.labels})
            print("[10 digit] |%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))

print("|===============================|")
test_accuracy = accuracy_10.eval(feed_dict=\
    {x: mnist.test.images, y_10:mnist.test.labels})
print("[10 digit] test accuracy=%.4f"%(test_accuracy))
