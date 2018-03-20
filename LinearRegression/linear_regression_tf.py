import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
#import seaborn as sns

#from sklearn.preprocessing import scale
#import sklearn.linear_model as skl_lm
#from sklearn.metrics import mean_squared_error, r2_score
#import statsmodels.api as sm
#import statsmodels.formula.api as smf

# using tensorflow
#import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn import datasets
rng = np.random

# get advertising data: sales (in thousands of units) as a function of 
# advertising budgets (in thousands of dollars) for TV, radio, newspaper media
advertising = pd.read_csv('./../Data/Advertising.csv',usecols=[1,2,3,4])

x_vals = np.array([x for x in advertising.TV])
y_vals = np.array([x for x in advertising.sales])

# =================================================================
def normalize(array): 
    return (array - array.mean()) / array.std()

def invert_normalize(array,mean,std): 
    return array*std+mean


x_vals_n = normalize(x_vals)
y_vals_n = normalize(y_vals)

n_samples = x_vals_n.size

# TF graph input
#X = tf.placeholder("float")
#Y = tf.placeholder("float")
#W = tf.Variable(np.random.randn(), name="weight")
#b = tf.Variable(np.random.randn(), name="bias")

X = tf.placeholder(shape=[None,1],dtype=tf.float32)
Y = tf.placeholder(shape=[None,1],dtype=tf.float32)
W = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Set parameters
learning_rate = 0.1
training_iteration = 200
display_step = 20
batch_size = n_samples

# Construct a linear model
model = tf.add(tf.multiply(X, W), b)

# Minimize squared errors
#loss = tf.reduce_sum(tf.square(Y-model)) # this is the residual sum-of-squares (RSS), but can be unstable 
#loss = tf.reduce_sum(tf.pow(Y-model, 2))/(2 * n_samples) 
loss = 0.5*tf.reduce_mean(tf.square(Y-model)) # same as second one above, but without divide by 2
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) #Gradient descent

# Initialize variables
init = tf.initialize_all_variables()

# Launch a graph
sess=tf.Session()
sess.run(init)

# Fit all training data
loss_vec=[]
for iteration in range(training_iteration):
    #for (x, y) in zip(x_vals_n, y_vals_n):
    #    print(x)
    #    fd = {X: x, Y: y}    
    #    sess.run(optimizer, feed_dict=fd)
    #rand_index = np.random.choice(n_samples,size=batch_size)
    #x = np.transpose([x_vals_n[rand_index]])
    #y = np.transpose([y_vals_n[rand_index]])
    x = np.transpose([x_vals_n])
    y = np.transpose([y_vals_n])
    fd = {X: x, Y: y}
    sess.run(optimizer, feed_dict=fd)
    loss_val = sess.run(loss,feed_dict=fd)
    loss_vec.append(loss_val)
    # Display logs per iteration step
    if iteration % display_step == 0:
        print("Iteration:", '%04d' % (iteration + 1), "loss=", "{:.9f}".format(loss_val),\
        "W=", sess.run(W).item(0), "b=", sess.run(b).item(0))
        

# Validate a tuning model
#
## Display a plot
#plt.figure()
#plt.plot(x_vals_n, y_vals_n, 'ro', label='Normalized samples')
#plt.plot(x_vals_n, sess.run(W).item(0) * x_vals_n + sess.run(b).item(0), label='Fitted line')
#plt.legend()
#
#plt.show()


best_fit_n = sess.run(W).item(0) * x_vals_n + sess.run(b).item(0)    
plt.plot(x_vals_n,y_vals_n,'o')
plt.plot(x_vals_n,best_fit_n,'r-')
plt.show()


best_fit = invert_normalize(best_fit_n,y_vals.mean(),y_vals.std())
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x=x_vals,y=best_fit)
plt.plot(x_vals,y_vals,'o')
plt.plot(x_vals,best_fit,'r-')
plt.show()

plt.plot(loss_vec,'k-')
plt.show()