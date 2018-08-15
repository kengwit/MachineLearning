import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess=tf.Session()

# seed
tf.set_random_seed(5)
np.random.seed(42)

batch_size = 50
a1 = tf.Variable( tf.random_normal(shape=[1,1]))
b1 = tf.Variable( tf.random_normal(shape=[1,1]))
a2 = tf.Variable( tf.random_normal(shape=[1,1]))
b2 = tf.Variable( tf.random_normal(shape=[1,1]))

x = np.random.normal(2,0.1,500)
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)

# two models
sigmoid_activation = tf.sigmoid( tf.add( tf.matmul(x_data,a1), b1))
relu_activation = tf.nn.relu( tf.add( tf.matmul(x_data,a2), b2))

# loss functions
loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation,0.75)))
loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation,0.75)))

# declare optimization algoritm and initialize variables
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step_sigmoid = my_opt.minimize(loss1)
train_step_relu = my_opt.minimize(loss2)

init = tf.global_variables_initializer()
sess.run(init)

# loop
loss_vec_sigmoid = []
loss_vec_relu = []
activation_sigmoid = []
activation_relu = []
for i in range(750):
    rand_indices = np.random.choice( len(x), size = batch_size )
    x_vals = np.transpose( [ x[rand_indices] ] )
    
    fd = {x_data: x_vals}
    
    sess.run( train_step_sigmoid, fd)
    sess.run( train_step_relu, fd)
    
    # store losses
    temp_loss_sigmoid = sess.run( loss1, fd )
    temp_loss_relu = sess.run( loss2, fd )
    
    loss_vec_sigmoid.append(temp_loss_sigmoid)
    loss_vec_relu.append(temp_loss_relu)
    
    # store activation values
    temp_sigmoid_activation = np.mean( sess.run( sigmoid_activation, fd ) )
    temp_relu_activation = np.mean( sess.run( relu_activation, fd ) )
    
    activation_sigmoid.append(temp_sigmoid_activation)
    activation_relu.append(temp_relu_activation)
    
    
    
# plot
plt.plot(activation_sigmoid,'k-',label='Sigmoid Activation')
plt.plot(activation_relu,'r--',label='Relu Activation')
plt.ylim([0.0,1.0])
plt.title('Activation Outputs')
plt.xlabel('Generation')
plt.ylabel('Outputs')
plt.legend(loc='upper right')
plt.show()

plt.plot(loss_vec_sigmoid,'k-',label='Sigmoid Activation')
plt.plot(loss_vec_relu,'r--',label='Relu Activation')
plt.ylim([0.0,1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()






