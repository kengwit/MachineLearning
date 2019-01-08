import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from sklearn import datasets 
from tensorflow.python.framework import ops 
ops.reset_default_graph() 

def normalize(array):
    return (array - array.mean()) / array.std()

def invert_normalize(array,mean,std): 
    return array*std+mean


# Create graph 
sess = tf.Session() 

# Regularization param
RegC=0.1

# Generate data 
iris = datasets.load_iris()
x_vals = np.array( [ [x[0],x[3]] for x in iris.data ]  )
# standardize
x_vals[:,0] = (x_vals[:,0] - x_vals[:,0].mean()) / x_vals[:,0].std()
x_vals[:,1] = (x_vals[:,1] - x_vals[:,1].mean()) / x_vals[:,1].std()

y_vals = np.array( [ 1 if y==0 else -1 for y in iris.target ])

class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1] 
class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1] 
class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1] 
class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1] 


# Declare batch size 
batch_size = 100 


# Initialize placeholders 
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32) 
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32) 
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32) 


# Create variables for svm 
b = tf.Variable(tf.random_normal(shape=[1,batch_size])) 
b = tf.clip_by_value(b,0.0,RegC)
intercept = tf.Variable(tf.random_normal(shape=[1,1])) 

# Apply kernel 
# Linear Kernel 
my_kernel = tf.matmul(x_data, tf.transpose(x_data)) 


# Compute SVM Model 
first_term = tf.reduce_sum(b) 
b_vec_cross = tf.matmul(tf.transpose(b), b) 
y_target_cross = tf.matmul(y_target, tf.transpose(y_target)) 
second_term = tf.multiply(tf.constant(0.5),tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross))) )
loss = tf.negative(tf.subtract(first_term, second_term)) 


# Create Prediction Kernel 
# Linear prediction kernel 
pred_kernel = tf.matmul(x_data, tf.transpose(prediction_grid)) 


wvec = tf.matmul( tf.multiply(tf.transpose(y_target),b), x_data )

indices_pos = tf.where(tf.equal(y_target,1))[:,0]
indices_min = tf.where(tf.equal(y_target,-1))[:,0]
xpos = tf.gather(x_data,indices_pos)
xmin = tf.gather(x_data,indices_min)

wxmin = tf.reduce_min(tf.matmul( wvec,tf.transpose(xpos) ))
wxmax = tf.reduce_max(tf.matmul( wvec,tf.transpose(xmin) ))


#intercept = tf.multiply(tf.constant(-0.5),tf.add( wxmin,wxmax )) 
prediction_output = tf.add( tf.matmul(tf.multiply(tf.transpose(y_target),b), pred_kernel), [0.0]) 
prediction = tf.sign(prediction_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32)) 


# Declare optimizer 
my_opt = tf.train.GradientDescentOptimizer(0.005) 
train_step = my_opt.minimize(loss) 


# Initialize variables 
init = tf.global_variables_initializer() 
sess.run(init) 


# Training loop 
loss_vec = [] 
batch_accuracy = [] 
for i in range(6000): 
    rand_index = np.random.choice(len(x_vals), size=batch_size) 
    rand_x = x_vals[rand_index] 
    rand_y = np.transpose([y_vals[rand_index]]) 
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y}) 
     
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y}) 
    loss_vec.append(temp_loss) 
    
    temp_intercept = sess.run(intercept, feed_dict={x_data: rand_x, y_target: rand_y}) 
    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, 
                                             y_target: rand_y, 
                                             prediction_grid:rand_x}) 
    batch_accuracy.append(acc_temp) 
     
    if (i+1)%1000==0: 
         print('Step #' + str(i+1)) 
         print('Loss = ' + str(temp_loss)) 
         print('Intercept = ' + str(temp_intercept))

 # Create a mesh to plot points in 
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1 
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), 
                      np.arange(y_min, y_max, 0.02)) 
grid_points = np.c_[xx.ravel(), yy.ravel()] 
[grid_predictions] = sess.run(prediction, feed_dict={x_data: rand_x, 
                                                    y_target: rand_y, 
                                                    prediction_grid: grid_points}) 
grid_predictions = grid_predictions.reshape(xx.shape) 
 

# Plot points and grid 
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8) 
plt.plot(class1_x, class1_y, 'ro', label='Class 1') 
plt.plot(class2_x, class2_y, 'kx', label='Class -1') 
plt.title('Gaussian SVM Results') 
plt.xlabel('x') 
plt.ylabel('y') 
plt.legend(loc='lower right') 
plt.ylim([-1.5, 1.5]) 
plt.xlim([-1.5, 1.5]) 
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
 


