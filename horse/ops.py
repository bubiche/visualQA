import tensorflow as tf

class op(object):
	def __init__(self, inp):
		self.inp = inp

	def build(self, loader, *args):
		self._get_weights(loader, *args)
		self._build_tf(*args)

	def _get_weights(self, loader, *args):
		pass

	def _build_tf(self, *args):
		pass


class crop(op):
	def _build_tf(self, *args):
		self.out = self.inp * 2. - 1.


class conv(op):
	def _get_weights(self, loader, ksize, c_in, c_out,
			  		stride, pad, batch_norm, act):
		bias_size = c_out
		weight_size = ksize ** 2 * c_in * c_out
		self.bias = loader.walk(bias_size)
		weight = loader.walk(weight_size)
		weight = weight.reshape([c_out, c_in, ksize, ksize])
		self.weight = weight.transpose([2,3,1,0])

	def _build_tf(self, ksize, c_in, c_out,
			  	stride, pad, batch_norm, act):
		padding = [[pad, pad]] * 2
		temp = tf.pad(self.inp, [[0, 0]] + padding + [[0, 0]])
		temp = tf.nn.conv2d(temp, self.weight, 
			padding = 'VALID', strides = [1] + [stride] * 2 + [1])
		self.out = tf.nn.bias_add(temp, self.bias)


class conn(op):
	def _get_weights(self, loader, inp_size, out_size, act):
		bias_size = out_size
		weight_size = inp_size * out_size
		self.bias = loader.walk(bias_size)
		weight = loader.walk(weight_size)
		if not loader.transpose:
			weight = weight.reshape([out_size, inp_size])
			weight = weight.transpose([1, 0])
		else:
			weight = weight.reshape([inp_size, out_size])
		self.weight = weight

	def _build_tf(self, inp_size, out_size, act):
		self.out = tf.nn.xw_plus_b(
			self.inp, self.weight, self.bias)


class maxpool(op):
	def _build_tf(self, size, stride, pad):
		self.out = tf.nn.max_pool(
			self.inp, padding = 'SAME',
			ksize = [1] + [size] * 2 + [1],
			strides = [1] + [stride] * 2 + [1])


class leaky(op):
	def _build_tf(self):
		self.out = tf.maximum(.1 * self.inp, self.inp)


import tensorflow.contrib.slim as slim


class flatten(op):
	def _build_tf(self):
		trans = tf.transpose(self.inp, [0,3,1,2])
		self.out = slim.flatten(trans)


op_dict = {
	'crop': crop,
	'convolutional': conv,
	'maxpool': maxpool,
	'connected': conn,
	'leaky': leaky,
	'flatten': flatten
}