# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

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

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_hat=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
print("=================================")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
#TODO: change to 5
for j in range(1):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            val_accuracy = accuracy.eval(feed_dict=\
                {x: mnist.validation.images, y_:mnist.validation.labels})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
print("|===============================|")
test_accuracy = accuracy.eval(feed_dict=\
    {x: mnist.test.images, y_:mnist.test.labels})
print("test accuracy=%.4f"%(test_accuracy))

#task3
test_output = h_fc1.eval(feed_dict=\
    {x: mnist.test.images, y_:mnist.test.labels})

#subtract mean
mean_vector = np.mean(test_output, 1)
mean_matrix = np.tile(mean_vector, (500, 1))
mean_matrix = np.transpose(mean_matrix)

test_output = test_output - mean_matrix

#apply SVD
phi_output = np.matmul(np.transpose(test_output), test_output)
U, s, V = np.linalg.svd(phi_output, full_matrices=True)
z_output = np.matmul(test_output, U)

#task 3-1
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_xlabel('1st column')
ax.set_ylabel('2nd column')
ax.set_title('Z dot graph')
ax.plot(z_output[:,0], z_output[:,1], 'ro')
fig.savefig('task3_1.png')
plt.close(fig)

#task 3-2
row_list = []
test_output = h_fc1.eval(feed_dict=\
    {x: mnist.test.images, y_:mnist.test.labels})

for i in range(10):
    count = 0
    for j in range(np.size(mnist.test.labels, 0)):
        if mnist.test.labels[j, i] == 1:
            #print(mnist.test.labels[j])
            row_list.append(test_output[j])
            count = count + 1

        if count == 10:
            break

row_arr = np.array(row_list)

#subtract mean
mean_vector = np.mean(row_arr, 1)
mean_matrix = np.tile(mean_vector, (500, 1))
mean_matrix = np.transpose(mean_matrix)

row_arr = row_arr - mean_matrix

#apply SVD
phi_output = np.matmul(np.transpose(row_arr), row_arr)
U, s, V = np.linalg.svd(phi_output, full_matrices=True)
z_output = np.matmul(row_arr, U)

#task 3-1
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_xlabel('1st column')
ax.set_ylabel('2nd column')
ax.set_title('Z dot graph')
ax.plot(z_output[0:10,0], z_output[0:10,1], 'C1o')
ax.plot(z_output[10:20,0], z_output[10:20,1], 'C2o')
ax.plot(z_output[20:30,0], z_output[20:30,1], 'C3o')
ax.plot(z_output[30:40,0], z_output[30:40,1], 'C4o')
ax.plot(z_output[40:50,0], z_output[40:50,1], 'C5o')
ax.plot(z_output[50:60,0], z_output[50:60,1], 'C6o')
ax.plot(z_output[60:70,0], z_output[60:70,1], 'C7o')
ax.plot(z_output[70:80,0], z_output[70:80,1], 'C8o')
ax.plot(z_output[80:90,0], z_output[80:90,1], 'C9o')
ax.plot(z_output[90:100,0], z_output[90:100,1], 'C0o')
fig.savefig('task3_2.png')
plt.close(fig)





