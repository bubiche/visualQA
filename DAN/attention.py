import tensorflow as tf
from .utils import xavier_var

def img_attend(self, memory, step):
	Wvm = xavier_var(
		'Wvm_{}'.format(step), [self._TXT_DIM, self._HID_DIM])
	Wv = xavier_var(
		'Wv_{}'.format(step), [self._IMG_DIM, self._HID_DIM])
	Wvh = xavier_var(
		'Wvh_{}'.format(step), [self._HID_DIM, self._HID_DIM])
	P = xavier_var(
		'P_{}'.format(step), [self._IMG_DIM, self._TXT_DIM])

	transformed_v = tf.einsum('aij,jk->aik', self.vn, Wv)
	transformed_m = tf.matmul(memory, Wvm)
	
	crossed = tf.einsum('aik,ak->ai',
		transformed_v, transformed_m)
	transcross = tf.matmul(crossed, Wvh)
	attention = tf.nn.softmax(transcrosse)
	combined = tf.einsum('aij,ai->aj', self.vn, attention)

	return tf.tanh(tf.matmul(combined, P))

def txt_attend(self, memory, step):
	Wum = xavier_var(
		'Wum_{}'.format(step), [self._TXT_DIM, self._HID_DIM])
	Wu = xavier_var(
		'Wu_{}'.format(step), [self._TXT_DIM, self._HID_DIM])
	Wuh = xavier_var(
		'Wuh_{}'.format(step), [self._HID_DIM, self._HID_DIM])
	
	transformed_u = tf.einsum('aij,jk->aik', self.ut, Wu)
	transformed_m = tf.matmul(memory, Wum)
	
	crossed = tf.einsum('aik,ak->ai',
		transformed_u, transformed_m)
	transcross = tf.matmul(crossed, Wuh)
	attention = tf.nn.softmax(transcrosse)
	
	return tf.einsum('aij,ai->aj', self.ut, attention)