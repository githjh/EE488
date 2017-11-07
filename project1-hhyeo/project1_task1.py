import tensorflow as tf
import numpy as np
import argparse
import random
import args
import sys

FLAGS = args.get()

def generate_dataset(num):
    train_data_list = []
    data_list = []
    label_list = []
    for i in range(num):
        r = np.random.normal()
        t = np.random.uniform(0, np.pi * 2)

        data = np.array([[r*np.cos(t), r*np.sin(t)]])
        label = np.array([[1, 0]])
        data_list.append(data)
        label_list.append(label)

        data = np.array([[(r+5)*np.cos(t), (r+5)*np.sin(t)]])
        label = np.array([[0, 1]])
        data_list.append(data)
        label_list.append(label)

    train_data_list = [data_list, label_list]

    return train_data_list

def shuffle_dataset(dataset_list, label_list):
    l = list(zip(dataset_list, label_list))
    random.shuffle(l)
    dataset_list_ , label_list_ = zip(*l)

    return dataset_list_, label_list_

def build_model():
    x = tf.placeholder(tf.float32, shape=[None, 2])
    y_label = tf.placeholder(tf.float32, shape=[None, 2])

    w_0 = tf.Variable(tf.truncated_normal([2, 20], stddev=0.01))
    b_0 = tf.Variable(tf.constant(0.1, shape=[20]))
    h_0 = tf.nn.relu(tf.matmul(x, w_0) + b_0)

    w_1 = tf.Variable(tf.truncated_normal([20, 2], stddev=0.01))
    b_1 = tf.Variable(tf.constant(0.1, shape=[1]))
    h_1 = tf.nn.softmax(tf.matmul(h_0, w_1) + b_1)

    y = h_1
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_label, logits = y)

    return x, y, y_label, cross_entropy

def get_batch(train_dataset, index, batch_num):
    #if len(train_dataset) < index * args.batch_num
    margin = (index + 1) * batch_num - len(train_dataset[0])
    if margin > 0:
        batch_data = train_dataset[0][index * batch_num : len(train_dataset)]
        batch_data.append(train_dataset[0][0:margin])
        a = np.array(batch_data)
        #batch_data.concatenate(batch_data, train_dataset[0][0: margin])
        batch_label = train_dataset[1][index * batch_num : len(train_dataset)]
        batch_data.append(train_dataset[1][0:margin])
    else:
        batch_data = train_dataset[0][index * batch_num : (index + 1) * FLAGS.batch_num]
        batch_label = train_dataset[1][index * batch_num : (index + 1) * FLAGS.batch_num]

        batch_data_arr = np.squeeze(np.array(batch_data), axis = 1)
        batch_label_arr = np.squeeze(np.array(batch_label), axis = 1)

        #print(batch_data_arr.shape, batch_label_arr.shape)

    return batch_data_arr, batch_label_arr

    #return [np.array(batch_data).squeeze(reshape(len(batch_data), 2), np.array(batch_label).reshape(len(batch_data), 2)]

def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    #dataset generation
    train_dataset = generate_dataset(FLAGS.train_dataset_num)
    valid_dataset = generate_dataset(FLAGS.valid_dataset_num)
    test_dataset = generate_dataset(FLAGS.test_dataset_num)

    #print(len(train_dataset[0]))
    #sys.exit(-1)

    x, y, y_label, cross_entropy = build_model()
    optimizer = tf.train.AdamOptimizer(FLAGS.initial_lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(FLAGS.epoch):
            shuffle_dataset(train_dataset[0], train_dataset[1])

            for j in range(int(len(train_dataset[0]) / FLAGS.batch_num)):
                batch = get_batch(train_dataset, j, FLAGS.batch_num)
                _, accuracy_ = sess.run([optimizer, accuracy], feed_dict={x: batch[0], y_label: batch[1]})

                #print(batch[0].shape)

            print("[train] epoch: %d, step: %d, accuracy: %.3f"%(i, j, accuracy_))

            #accuracy = sess.run([accuracy], feed_dict={x: valid_dataset[0], y_label: valid_dataset[1]})
            #print("[valid] epoch: %d, accuracy: %.3f"%(i, accuracy))

    #TODO: validation set accuracy(task2 example code), hyper-parameter setting, test case plot

if __name__ == '__main__':
    tf.app.run()













