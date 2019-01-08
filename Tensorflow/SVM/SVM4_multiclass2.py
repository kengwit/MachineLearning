import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops 
ops.reset_default_graph() 

sess = tf.Session()

# load data
iris = datasets.load_iris()
x_vals = np.array([[x[0],x[3]] for x in iris.data])

y_vals1 = np.array([1 if y==0 else -1 for y in iris.target])
y_vals2 = np.array([1 if y==1 else -1 for y in iris.target])
y_vals3 = np.array([1 if y==2 else -1 for y in iris.target])
y_vals = np.array([y_vals1,y_vals2,y_vals3])


class1_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==0]
class1_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==0]

class2_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==1]
class2_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==1]

class3_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==2]
class3_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==2]


batch_size = 50
# -------------------
# change 1 below
# -------------------
x_data = tf.placeholder(shape=[None,2],dtype=tf.float32)
y_target = tf.placeholder(shape=[3,None],dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None,2],dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[3,batch_size])) # note: b here are Lagrange multipliers not the intercept

gamma = tf.constant(-10.0)
dist = tf.reduce_sum(tf.square(x_data),1)
dist = tf.reshape(dist,[-1,1])
sq_dists = tf.add( tf.subtract(dist, tf.multiply(2.,tf.matmul(x_data,tf.transpose(x_data)))), tf.transpose(dist))
# note: my_kernel is a square matrix of size ( batch_size * batch_size )
my_kernel = tf.exp(tf.multiply(gamma,tf.abs(sq_dists)))


# -------------------
# change 2 below
# -------------------
def reshape_matmul(mat):
    v1 = tf.expand_dims(mat,1)
    v2 = tf.reshape(v1,[3,batch_size,1])
    return(tf.matmul(v2,v1))
    
# declare dual problem
# note: 
# b has dimensions 3 * batch_size 
# y_target has dimensions 3 * batch_size 
#    
model_output = tf.matmul(b,my_kernel)
first_term   = tf.reduce_sum(b,1) # should have 3 numbers
# note: 
# b_vec_cross is a square matrix of size ( batch_size * batch_size )
# y_target_cross is a matrix of size ( 3 * batch_size )
b_vec_cross  = tf.matmul(tf.transpose(b),b)
y_target_cross = reshape_matmul(y_target)
# -------------------
# change 3 below
# note: temp below has dimensions 3 * batch_size * batch_size
# i.e. 3 square matrices, each with size ( batch_size * batch_size )
# the dimensions/indices to reduce are therefore 1 and 2
# -------------------
#temp = tf.multiply(my_kernel,tf.multiply(b_vec_cross,y_target_cross))
#second_term = tf.reduce_sum(temp,[1,2]) 
#loss = tf.reduce_sum( tf.negative(tf.subtract(first_term,second_term)) )


# alternatively, use batch multiplication 
yb_vec = tf.multiply(y_target,b) # 3 * batch_size
K_times_yb = tf.matmul(my_kernel,tf.transpose(yb_vec)) # batch_size * 3
ybT_Kb_yb = tf.matmul( yb_vec, K_times_yb )
second_term = tf.reduce_sum(ybT_Kb_yb)
loss = tf.reduce_sum( tf.negative(tf.subtract(first_term,second_term)) )

#num1 = tf.matmul(yb_vec[0],tf.reshape(K_times_yb[:,0],[3,50])

# create prediction and accuracy functions
# prediction = we have the kernel of the points with the prediction data
rA = tf.reshape(tf.reduce_sum(tf.square(x_data),axis=1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid),axis=1),[-1,1])
pred_sq_dist = tf.add(tf.subtract(rA,tf.multiply(2.,tf.matmul(x_data,tf.transpose(prediction_grid)))),tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma,tf.abs(pred_sq_dist)))
prediction_output = tf.matmul(tf.multiply(y_target,b),pred_kernel)
#prediction_output = tf.expand_dims(tf.reduce_sum(prediction_output,axis=1),1)
#note: the subtraction of reduce_mean below enforces values between -1 and 1
po_mean = tf.expand_dims(tf.reduce_mean(prediction_output,axis=1),1)
prediction = tf.argmax(prediction_output-po_mean,0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(y_target,0)),tf.float32))

# Declare optimizer 
my_opt = tf.train.GradientDescentOptimizer(0.01) 
train_step = my_opt.minimize(loss) 


# Initialize variables 
init = tf.global_variables_initializer() 
sess.run(init) 

 
# Training loop 
loss_vec = [] 
batch_accuracy = [] 
for i in range(400): 
    rand_index = np.random.choice(len(x_vals), size=batch_size) 
    rand_x = x_vals[rand_index] 
    rand_y = y_vals[:, rand_index] 
    fd = feed_dict={x_data: rand_x, y_target: rand_y}
    
    sess.run(train_step, fd) 
     
    temp_loss = sess.run(loss, fd) 
    loss_vec.append(temp_loss) 
     
    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, 
                                             y_target: rand_y, 
                                             prediction_grid: rand_x}) 
    batch_accuracy.append(acc_temp) 
     
    if (i + 1) % 25 == 0: 
        print('Step #' + str(i+1)) 
        print('Loss = ' + str(temp_loss)) 


# Create a mesh to plot points in 
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1 
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), 
                     np.arange(y_min, y_max, 0.02)) 
grid_points = np.c_[xx.ravel(), yy.ravel()] 

fd2 = feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: grid_points}
grid_predictions = sess.run(prediction, fd2) 
grid_predictions = grid_predictions.reshape(xx.shape) 


# Plot points and grid 
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8) 
plt.plot(class1_x, class1_y, 'ro', label='I. setosa') 
plt.plot(class2_x, class2_y, 'kx', label='I. versicolor') 
plt.plot(class3_x, class3_y, 'gv', label='I. virginica') 
plt.title('Gaussian SVM Results on Iris Data') 
plt.xlabel('Pedal Length') 
plt.ylabel('Sepal Width') 
plt.legend(loc='lower right') 
plt.ylim([-0.5, 3.0]) 
plt.xlim([3.5, 8.5]) 
plt.show() 


# Plot batch accuracy 
plt.plot(batch_accuracy, 'k-', label='Accuracy') 
plt.title('Batch Accuracy') 
plt.xlabel('Generation') 
plt.ylabel('Accuracy') 
plt.legend(loc='lower right') 
plt.show() 


# Plot loss over time 
plt.plot(loss_vec, 'k-') 
plt.title('Loss per Generation') 
plt.xlabel('Generation') 
plt.ylabel('Loss') 
plt.show() 
