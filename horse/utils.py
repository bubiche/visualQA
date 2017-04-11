import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier_conv
from tensorflow import random_normal_initializer as gaussian
from tensorflow import constant_initializer as constant

def cosine_sim(mem, ref):
	mem_norm = tf.norm(mem, axis = -1, keep_dims = True)
	ref_norm = tf.norm(ref)
	dot_prod = tf.matmul(mem, tf.transpose(ref))
	cosine_sim = dot_prod / (mem_norm * ref_norm)
	return cosine_sim

def _last_dim(tensor):
	return tensor.get_shape().as_list()[-1]

def const_var(name, val, shape):
	return tf.get_variable(name = name,
		shape = shape, initializer = constant(val))

def xavier_var(name, shape):
	return tf.get_variable(name = name, 
		shape = shape, initializer = xavier())

def gaussian_var(name, mean, std, shape):
    return tf.get_variable(name = name,
        shape = shape, initializer = gaussian(mean, std))
        
def xavier_var_conv(name, shape):
	return tf.get_variable(name = name,
		shape = shape, initializer = xavier_conv())

def _sharp_gate(x):
	last_dim = _last_dim(x)
	linear = tf.matmul(x, xavier_var('gatew', (last_dim, 1)))
	linear += const_var('gateb', 0.0, (1,))
	return tf.nn.softplus(linear)

def sharpen(x):
	power = tf.pow(x, 1. + _sharp_gate(x))
	power = tf.div(
		tf.sub(power, tf.reduce_min(power, -1)),
		tf.sub(tf.reduce_max(power, -1), 
			   tf.reduce_min(power, -1)))
	#summation = tf.reduce_sum(power, -1, keep_dims = True)
	return power

def tanh_gate(x, feat_in, feat_out):
	linear = tf.matmul(x, xavier_var('tanhw', (feat_in, feat_out)))
	linear += xavier_var('tanhb', (feat_out,))
	return tf.tanh(linear)

def conv_pool_leak(x, feat_in, feat_out):
	# conv
	padding = [[1, 1]] * 2
	temp = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
	temp = tf.nn.conv2d(temp, 
		xavier_var_conv('convw', [3, 3, feat_in, feat_out]), 
		padding = 'VALID', strides = [1, 1, 1, 1])
	conved = tf.nn.bias_add(
		temp, const_var('convb', 0.0, (feat_out,)))

	# pool
	pooled = tf.nn.max_pool(
		conved, padding = 'SAME',
		ksize = [1, 2, 2, 1],
		strides = [1, 2, 2, 1])

	# leaky
	return tf.maximum(0.1 * pooled, pooled)