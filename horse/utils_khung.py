from .utils import cosine_sim, gate, conv_pool_act, gaussian_var
import tensorflow as tf

def _identity(x):
	return x

def _leak(x):
	return tf.maximum(0.1 * x, x)

def ref(x):
	print('ref')
	x_flat = tf.reshape(x, [-1, 1024])
	with tf.variable_scope('tanh_gate'):
		tanh_vol = gate(x_flat, 1024, 512, tf.tanh)

	reference = gaussian_var(
			'ref', 0.00204, 0.0462, [1, 1024])
	with tf.variable_scope('tanh_gate', reuse = True):
		tanh_ref = gate(reference, 1024, 512, tf.tanh)


	similar = cosine_sim(tanh_vol, tanh_ref)
	return tf.reshape(similar, [-1, 49])

def no_ref(x):
	print('noref')
	x_flat = tf.reshape(x, [-1, 1024])
	with tf.variable_scope('gate1'):
		leaked = gate(x_flat, 1024, 32, _leak)
	with tf.variable_scope('gate2'):
		tanhed = gate(leaked, 32, 1, tf.tanh)
	return tf.reshape(tanhed, [-1, 49])

def no_sharp(x):
	print('nosharp')
	return _identity(x)

def softmax(x):
	print('softmax')
	boped = tf.nn.softmax(x * 100)
	return tf.reshape(boped, [-1, 7, 7, 1])

def power(x):
	print('power')
	sign = tf.sign(x)
	t = sign * tf.pow(sign * x, 1./3.)
	t = (t + 1.) / 2.
	return tf.reshape(t, [-1, 7, 7, 1])

def count(x):
	print('count')
	conv1 = conv_pool_act(x, 1024, 64, _leak, 'conv1')
	conv2 = conv_pool_act(conv1, 64, 5, tf.nn.sigmoid, 'conv2')
	return tf.reduce_sum(conv2, [1, 2, 3])