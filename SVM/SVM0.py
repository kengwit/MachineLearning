import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()
iris = datasets.load_iris()
x_vals = np.array( [ [x[0],x[3]] for x in iris.data ]  )
y_vals = np.array( [ 1 if y==0 else -1 for y in iris.target ])

#
# split the dataset into train and test sets
# training = 120 (80% of 150)
# test     = 30  (remaining)
train_indices = np.random.choice( len(x_vals), round( len(x_vals)*0.8 ), replace=False )
test_indices  = np.array( list(set(range(len(x_vals))) - set(train_indices)) )
x_vals_train = x_vals[train_indices]
x_vals_test  = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test  = y_vals[test_indices]

batch_size = 100
x_data   = tf.placeholder( shape=[None, 2], dtype=tf.float32 )
y_target = tf.placeholder( shape=[None, 1], dtype=tf.float32 )

# -----------------------------------------------------
# variables
# -----------------------------------------------------
A = tf.Variable( tf.random_normal(shape=[2,1]) )
b = tf.Variable( tf.random_normal(shape=[1,1]) )
model_output = tf.subtract(tf.matmul(x_data,A),b)
l2_norm = tf.reduce_sum( tf.square(A) )
alpha = tf.constant([0.1])
