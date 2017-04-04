import tensorflow as tf
from .utils import xavier_var

EPS = 1e-3

def _softmax(x, mask):
	x = x - tf.reduce_min(x, -1) + EPS
	x = tf.multiply(x, mask)
	x = tf.exp(x) - 1.
	x = x / tf.reduce_sum(x, -1)
	return x

def tanh_attend(inp_dim, hid_dim, # outdimlstm, mdim
				vecs, memory, step, mask = None): # vt/ut, m(k-1)
	Wm = xavier_var(
		'Wm_{}'.format(step), [hid_dim, hid_dim])
	Wf = xavier_var(
		'Wu_{}'.format(step), [inp_dim, hid_dim])

	Wh = xavier_var(
		'Wh_{}'.format(step), [hid_dim, hid_dim])
	
	transformed_f = tf.einsum('aij,jk->aik', vecs, Wf)
	transformed_m = tf.matmul(memory, Wm)
	
	crossed = tf.einsum('aik,ak->ai',
		transformed_f, transformed_m)

	normalizer = tf.nn.softmax
	args = [crossed]
	if mask is not None:
		normalizer = _softmax
		args.append(mask)
	attention = normalizer(*args)
	
	return tf.einsum('aij,ai->aj', vecs, attention)