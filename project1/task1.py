# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

num_label_0 = 25000
num_label_1 = 25000

num_hidden = 20
num_class = 2

num_sample = num_label_0 + num_label_1

num_validation = 1000
num_test = 1000
num_train = num_sample - num_validation - num_test

batch_size = 100

# r - lable(0 or 1)
r0 = np.random.normal(0,1,num_label_0)
r1 = np.random.normal(0,1,num_label_1)

# t - label(0 or 1)
t0 = np.random.uniform(0, 2*np.pi, num_label_0)
t1 = np.random.uniform(0, 2*np.pi, num_label_1)

# x - lable(0 or 1) - coordinate (0 or 1)
x01 = r0 * np.cos(t0)
x02 = r0 * np.sin(t0)

x11 = (r1 + 5) * np.cos(t1)
x12 = (r1 + 5) * np.sin(t1)

label_0 = np.zeros(num_label_0 * num_class )
label_1 = np.zeros(num_label_1 * num_class )

label_0 = label_0.reshape(num_label_0, num_class)
label_0[:,0] = 1

label_1 = label_1.reshape(num_label_1, num_class)
label_1[:,1] = 1

x1 = np.concatenate((x01,x11), axis=0)
x2 = np.concatenate((x02,x12), axis=0)

print (x1,x2)
x1 = x1.reshape(num_sample,1)
x2 = x2.reshape(num_sample,1)

x_data = np.concatenate((x1,x2), axis=1)
label = np.concatenate((label_0,label_1), axis=0)

#for draw a classification line 
#x1_draw = (np.arange(-7,7,0.01)).reshape(1400,1)
#x2_draw = (np.arange(-7,7,0.01)).reshape(1400,1)
#x_draw = np.concatenate((x1_draw,x2_draw), axis=1)


x_draw = np.zeros(1400*1400*2).reshape(1400*1400,2)

for i in range(1400):
    for j in range(1400):
        x_draw[1400*i + j][0] = -7 + 0.01*i
        x_draw[1400*i + j][1] = -7 + 0.01*j

def shuffle_data( p_x_data, p_label):
	train_set = np.concatenate((p_x_data, p_label), axis=1)
	train_set_shuffled = np.random.permutation(train_set)
	x_shuffled = train_set_shuffled [:, 0:x_data.shape[1]]
	label_shuffled = train_set_shuffled [:,x_data.shape[1]:]
	return (x_shuffled, label_shuffled)

x_data_set, label_set = shuffle_data(x_data, label)

x_train = x_data_set[:num_train, :]
x_val = x_data_set[num_train:num_train+num_validation, :]
x_test = x_data_set[num_train+num_validation:,:]

label_train = label_set[:num_train, :]
label_val = label_set[num_train: num_train+num_validation, :]
label_test = label_set[num_train+num_validation:, :]

'''
plt.scatter(x01,x02)
plt.scatter(x11,x12)
plt.axes().set_aspect('equal')
plt.show()
'''

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([2, num_hidden], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([num_hidden, 2], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[2]))
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

total_batch = num_train/batch_size

for j in range(10):
	index_counter = 0
	x_shuffled, label_shuffled = shuffle_data(x_train, label_train)
	for i in range(total_batch):
		start_index = index_counter * batch_size
		end_index = (index_counter + 1) * batch_size
		batch_x = x_shuffled [start_index : end_index]
		batch_y = label_shuffled [start_index : end_index]
		#print ("batch_x", batch_x)
		#print ("batch_y", batch_y)
		index_counter = index_counter + 1
		if (index_counter >= total_batch):
			index_counter = 0
		#batch = mnist.train.next_batch(100)
		train_step.run(feed_dict={x: batch_x, y_: batch_y})
		if i%10 == 9:
			train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
			val_accuracy = accuracy.eval(feed_dict=\
			{x: x_val, y_: label_val})
			print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
print("|===============================|")

output  = tf.argmax(y_hat,1).eval(feed_dict = {x:x_draw})
line_x = np.array([])
line_y = np.array([])
filp_check = output[0]
for i in range(len(output)):
    if(output[i] != filp_check):
		line_x = np.append(line_x, x_draw[i,0])
		line_y = np.append(line_y, x_draw[i,1])
		filp_check = output[i]
		#print ("filp", i)
#print (line_x, line_y)

#print (output)

#test_accuracy = accuracy.eval(feed_dict=\
#    {x: mnist.test.images, y_:mnist.test.labels})

test_accuracy = accuracy.eval(feed_dict=\
    {x: x_test, y_:label_test})
print("test accuracy=%.4f"%(test_accuracy))

plt.scatter(x01[:500],x02[:500])
plt.scatter(x11[:500],x12[:500])
plt.scatter(line_x,line_y)
plt.plot(line_x,line_y)
plt.axes().set_aspect('equal')
plt.show()



