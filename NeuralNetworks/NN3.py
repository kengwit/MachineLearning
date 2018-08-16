import tensorflow as tf
import numpy as np
sess = tf.Session()
data_size = 25
data_1d = np.random.normal(size=data_size)
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])


def conv_layer_1d(input_1d, my_filter):
    
    # Make 1d input into 4d
    #  input_1d is of size [data_size]
    input_2d = tf.expand_dims(input_1d, 0) # [width=1,data_size] 
    input_3d = tf.expand_dims(input_2d, 0) # [batch_size=1,width=1,data_size]
    input_4d = tf.expand_dims(input_3d, 3) # [batch_size=1,width=1,data_size,channel=1]
    
    # Perform convolution
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter,
                                      strides=[1,1,1,1], padding="VALID")
    # Now drop extra dimensions
    conv_output_1d = tf.squeeze(convolution_output)
    return(conv_output_1d)
    
    
my_filter = tf.Variable(tf.random_normal(shape=[1,5,1,1]))
my_convolution_output = conv_layer_1d(x_input_1d, my_filter)
