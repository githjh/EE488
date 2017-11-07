import sys
import tensorflow as tf

flags = tf.app.flags
FLAGS= flags.FLAGS

flags.DEFINE_integer('train_dataset_num', 10000, '')
flags.DEFINE_integer('valid_dataset_num', 100, '')
flags.DEFINE_integer('test_dataset_num', 10000, '')
flags.DEFINE_integer('epoch', 1, '')
flags.DEFINE_integer('batch_num', 128, '')
flags.DEFINE_float('initial_lr', 1e-4, '')

def get():
    return FLAGS

