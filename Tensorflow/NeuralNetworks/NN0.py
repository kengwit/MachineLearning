import tensorflow as tf
sess=tf.Session()

# a*x = 50

a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

multiplication = tf.multiply(a,x_data)

loss = tf.square(tf.subtract(multiplication,50.))

init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

for i in range(22):
    fd = {x_data: x_val}
    sess.run(train_step,fd)
    a_val = sess.run(a)
    mult_output = sess.run(multiplication,fd)
    print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))
    


# a*x + b = 50
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()

a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

two_gate = tf.add( tf.multiply(a,x_data), b )

loss = tf.square(tf.subtract(two_gate,50.))

init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)


for i in range(22):
    fd = {x_data: x_val}
    sess.run(train_step,fd)
    a_val = sess.run(a)
    b_val = sess.run(b)
    two_gate_output = sess.run(two_gate,fd)
    print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_output))
    
