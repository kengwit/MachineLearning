import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

# generate data
(x_vals,y_vals) = datasets.make_circles(n_samples=500,factor=0.5,noise=0.1)

y_vals = np.array([1 if y==1 else -1 for y in y_vals])

class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]

class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

batch_size = 250
x_data = tf.placeholder(shape=[None,2],dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None,2],dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1,batch_size]))

# create Gaussian kernel
# distance is a vector having components:
# x_vals[0][0]**2+x_vals[0][1]**2
# x_vals[1][0]**2+x_vals[1][1]**2
#   ....
# x_vals[n][0]**2+x_vals[n][1]**2
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data),1)
dist = tf.reshape(dist,[-1,1])

# checking
#temp1 = sess.run(dist,feed_dict={x_data: x_vals})


# plot
plt.plot(class1_x,class1_y,'ro',label='Class 1')
plt.plot(class2_x,class2_y,'kx',label='Class 2')
plt.legend(loc='lower right')

plt.ylim([-1.5,1.5])
plt.xlim([-1.5,1.5])
plt.show()









