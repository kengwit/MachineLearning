import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

def normalize(array):
    return (array - array.mean()) / array.std()

def invert_normalize(array,mean,std): 
    return array*std+mean

sess = tf.Session()
iris = datasets.load_iris()
#x_vals0 = np.array( [ x[0] for x in iris.data ]  )
#x_vals1 = np.array( [ x[3] for x in iris.data ]  )
#x_vals = np.stack( (x_vals0, x_vals1),axis=1)
    
x_vals = np.array( [[x[0],x[3]] for x in iris.data ]  )
y_vals = np.array( [ 1 if y==0 else -1 for y in iris.target ])
class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]
class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

#
# split the dataset into train and test sets
# training = 120 (80% of 150)
# test     = 30  (remaining)
train_indices = np.random.choice( len(x_vals), round( len(x_vals)*0.8 ), replace=False )
test_indices  = np.array( list(set(range(len(x_vals))) - set(train_indices)) )
x_vals_train = x_vals[train_indices]
x_vals_test  = x_vals[test_indices]

y_vals_train = x_vals[train_indices]
y_vals_test  = x_vals[test_indices]



batch_size = 100
x_data = tf.placeholder(shape=[None,2],dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None,2],dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1,batch_size])) # note: b here are Lagrange multipliers not the intercept

# create kernel
my_kernel = tf.matmul(x_data,tf.transpose(x_data))

# declare dual problem
first_term   = tf.reduce_sum(b)
b_vec_cross  = tf.matmul(tf.transpose(b),b)
y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
second_term = tf.multiply(1.0,tf.reduce_sum(tf.multiply(my_kernel,tf.multiply(b_vec_cross,y_target_cross))))
loss = tf.negative(tf.subtract(first_term,second_term))

# create prediction and accuracy functions
# prediction = we have the kernel of the points with the prediction data
#pred_kernel = tf.matmul(x_data,tf.transpose(prediction_grid))
#prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target),b),pred_kernel)

#note: the subtraction of reduce_mean below enforces values between -1 and 1
#prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
#accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction),tf.squeeze(y_target)),tf.float32))

# create optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# training loop
loss_vec = []
batch_accuracy = []
for i in range(3000):

    rand_index = np.random.choice(len(x_vals_train),size=batch_size)
    rand_x     = x_vals_train[rand_index]
    rand_y     = np.transpose([y_vals_train[rand_index]])
    
    sess.run(train_step,feed_dict={x_data: rand_x, y_target: rand_y})
    
    #term1 = sess.run(first_term,feed_dict={x_data: rand_x, y_target: rand_y})
    #term2 = sess.run(second_term,feed_dict={x_data: rand_x, y_target: rand_y})
    #print(term2)
    
    temp_loss = sess.run( loss,feed_dict={x_data: rand_x, y_target: rand_y})
    #print('check=%e temp_loss=%e'%(-(term1-term2),temp_loss))
    loss_vec.append(temp_loss)
    
    #acc_temp = sess.run(accuracy,feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x} )
    #batch_accuracy.append(acc_temp)
    
    if ( i+1 )%100 == 0:
        print('Step #' + str(i+1) )
        print('Loss = ' + str(temp_loss) )
              
setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i] == 1]

not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i] == -1]

plt.plot(setosa_x,setosa_y,'o',label='I. setosa')
plt.plot(not_setosa_x,not_setosa_y,'x',label='Non-I. setosa')
#plt.plot(x1_vals,best_fit,'r-',label='Linear Separator',linewidth=3)
plt.ylim([0,10])
plt.legend(loc='lower right')

plt.show()



## Create a mesh to plot points in
#x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
#y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                     np.arange(y_min, y_max, 0.02))
#grid_points = np.c_[xx.ravel(), yy.ravel()]
#[grid_predictions] = sess.run(prediction, feed_dict={x_data: rand_x,
#                                                   y_target: rand_y,
#                                                   prediction_grid: grid_points})
#grid_predictions = grid_predictions.reshape(xx.shape)
#
## Plot points and grid
#plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
#plt.plot(class1_x, class1_y, 'ro', label='Class 1')
#plt.plot(class2_x, class2_y, 'kx', label='Class -1')
#plt.title('Gaussian SVM Results')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.legend(loc='lower right')
#plt.ylim([-1.5, 1.5])
#plt.xlim([-1.5, 1.5])
#plt.show()
#
## Plot batch accuracy
#plt.plot(batch_accuracy, 'k-', label='Accuracy')
#plt.title('Batch Accuracy')
#plt.xlabel('Generation')
#plt.ylabel('Accuracy')
#plt.legend(loc='lower right')
#plt.show()
#
## Plot loss over time
#plt.plot(loss_vec, 'k-')
#plt.title('Loss per Generation')
#plt.xlabel('Generation')
#plt.ylabel('Loss')
#plt.show()
#
### Evaluate on new/unseen data points
### New data points:
##new_points = np.array([(-0.75, -0.75),
##                       (-0.5, -0.5),
##                       (-0.25, -0.25),
##                       (0.25, 0.25),
##                       (0.5, 0.5),
##                       (0.75, 0.75)])
##
##[evaluations] = sess.run(prediction, feed_dict={x_data: x_vals,
##                                                y_target: np.transpose([y_vals]),
##                                                prediction_grid: new_points})
##
##for ix, p in enumerate(new_points):
##    print('{} : class={}'.format(p, evaluations[ix]))
##
#
#
#
#
#
#
