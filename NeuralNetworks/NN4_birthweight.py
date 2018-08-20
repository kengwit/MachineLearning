import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import csv
import os
import numpy as np
from tensorflow.python.framework import ops

## load data
## name of data file 
birth_weight_file = 'birth_weight.csv' 
#birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat' 
#
## Download data and create data file if file does not exist in current directory 
#os.remove(birth_weight_file)
#if not os.path.exists(birth_weight_file): 
#    birth_file = requests.get(birthdata_url) 
#    birth_data = birth_file.text.split('\r\n')
#    
#    
#    birth_header = birth_data[0].split('\t') 
#    birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] 
#                    for y in birth_data[1:] if len(y) >= 1] 
#    with open(birth_weight_file, "w") as f: 
#        writer = csv.writer(f) 
#        writer.writerows([birth_header]) 
#        writer.writerows(birth_data) 
#        f.close() 

 
 
# read birth weight data into memory 
birth_data = [] 
with open(birth_weight_file, newline='') as csvfile: 
    csv_reader = csv.reader(csvfile) 
    birth_header = next(csv_reader) 
    for row in csv_reader: 
        birth_data.append(row) 
 

birth_data = [[float(x) for x in row] for row in birth_data] 
 
# Extract y-target (birth weight) 
y_vals = np.array([x[8] for x in birth_data]) 
 
# Filter for features of interest 
# note: here, we are interested in SEVEN (7) features
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI'] 
x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data]) 
        
# Reset the graph for new run 
ops.reset_default_graph() 
sess = tf.Session()


# 
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)
batch_size = 100

# split data 80-20
train_indices = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
test_indices = np.array(  list(  set(range(len(x_vals))) - set(train_indices)  )   )
x_vals_train = x_vals[train_indices] 
x_vals_test = x_vals[test_indices] 
y_vals_train = y_vals[train_indices] 
y_vals_test = y_vals[test_indices] 


# Record TRAINING column max and min for scaling of non-training data 
train_max = np.max(x_vals_train, axis=0) 
train_min = np.min(x_vals_train, axis=0) 

# Normalize by column (min-max norm to be between 0 and 1) 
def normalize_cols(mat, max_vals, min_vals): 
    return (mat - min_vals) / (max_vals - min_vals) 

 
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train, train_max, train_min)) 
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test, train_max, train_min)) 

# Define Variable Functions (weights and bias) 
def init_weight(shape, st_dev): 
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev)) 
    return weight 
 
 
def init_bias(shape, st_dev): 
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev)) 
    return bias 


# Create Placeholders 
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32) 
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32) 
 
# Create a fully connected layer: 
def fully_connected(input_layer, weights, biases): 
    #print('before')
    #print(tf.shape(input_layer))
    #print(tf.shape(biases))
    layer = tf.add(tf.matmul(input_layer, weights), biases) 
    #layer = tf.matmul(input_layer, weights)
    #print(tf.shape(layer))
    
    #print('after')
    return tf.nn.relu(layer) 
 
# -------Create the first layer (50 hidden nodes)-------- 
weight_1 = init_weight(shape=[7, 25], st_dev=10.0) 
bias_1 = init_bias(shape=[25], st_dev=10.0) 
layer_1 = fully_connected(x_data, weight_1, bias_1) 
 
# -------Create second layer (25 hidden nodes)-------- 
weight_2 = init_weight(shape=[25, 10], st_dev=10.0) 
bias_2 = init_bias(shape=[10], st_dev=10.0) 
layer_2 = fully_connected(layer_1, weight_2, bias_2) 
 
# -------Create third layer (5 hidden nodes)-------- 
weight_3 = init_weight(shape=[10, 3], st_dev=10.0) 
bias_3 = init_bias(shape=[3], st_dev=10.0) 
layer_3 = fully_connected(layer_2, weight_3, bias_3) 
 
# -------Create output layer (1 output value)-------- 
weight_4 = init_weight(shape=[3, 1], st_dev=10.0) 
bias_4 = init_bias(shape=[1], st_dev=10.0) 
final_output = fully_connected(layer_3, weight_4, bias_4) 
 
# Declare loss function (L1) 
loss = tf.reduce_mean(tf.abs(y_target - final_output)) 
 
# Declare optimizer 
my_opt = tf.train.AdamOptimizer(0.02) 
train_step = my_opt.minimize(loss) 

# Initialize Variables 
init = tf.global_variables_initializer() 
sess.run(init) 

# Training loop 
loss_vec = [] 
test_loss = [] 
for i in range(1200): 
    rand_index = np.random.choice(len(x_vals_train), size=batch_size) 
    rand_x     = x_vals_train[rand_index] # size 100x7
    rand_y     = np.transpose([y_vals_train[rand_index]]) # size 100x1
    #sess.run(layer_1,feed_dict = {x_data: rand_x, y_target: rand_y})
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y}) 
    
    fd_train = {x_data: rand_x, y_target: rand_y}
    temp_loss = sess.run(loss, feed_dict=fd_train) 
    loss_vec.append(temp_loss) 

    fd_test = {x_data: x_vals_test, y_target: np.transpose([y_vals_test])}
    test_temp_loss = sess.run(loss, feed_dict=fd_test) 
    test_loss.append(test_temp_loss) 
    if (i+1) % 25 == 0: 
        #print('Generation: ' + str(i+1)) 
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss)) 
 
# Plot loss (MSE) over time 
plt.plot(loss_vec, 'k-', label='Train Loss') 
plt.plot(test_loss, 'r--', label='Test Loss') 
plt.title('Loss (MSE) per Generation') 
plt.legend(loc='upper right') 
plt.xlabel('Generation') 
plt.ylabel('Loss') 
plt.show() 
 
# Model Accuracy 
# using TRAINED NEURAL NET
actuals = np.array([x[0] for x in birth_data]) 
test_actuals = actuals[test_indices] 
train_actuals = actuals[train_indices] 
test_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_test})] 
train_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_train})] 
test_preds = np.array([1.0 if x < 2500.0 else 0.0 for x in test_preds]) 
train_preds = np.array([1.0 if x < 2500.0 else 0.0 for x in train_preds]) 

# Print out accuracies 
test_acc = np.mean([x == y for x, y in zip(test_preds, test_actuals)]) 
train_acc = np.mean([x == y for x, y in zip(train_preds, train_actuals)]) 
print('On predicting the category of low birthweight from regression output (<2500g):') 
print('Test Accuracy: {}'.format(test_acc)) 
print('Train Accuracy: {}'.format(train_acc)) 
 
# Evaluate new points on the model 
# Need vectors of 'AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI' 
new_data = np.array([[35, 185, 1., 0., 0., 0., 1.], 
                     [18, 160, 0., 1., 0., 0., 1.]]) 
new_data_scaled = np.nan_to_num(normalize_cols(new_data, train_max, train_min)) 
new_logits = [x[0] for x in sess.run(final_output, feed_dict={x_data: new_data_scaled})] 
new_preds = np.array([1.0 if x < 2500.0 else 0.0 for x in new_logits]) 
 
print('New Data Predictions: {}'.format(new_preds)) 
