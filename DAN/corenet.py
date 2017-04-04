'''
1. As described in the paper, 
	(m, h, u_t, u^(k), v^(k)) is 512, 

	while the pretrained image feature vector (v_n) has 2048 dimensions. 

	Note that the alpha^(k)_v,n in Eq. 5-6 and alpha^(k)_u,t in Eq. 9-10 is a scalar value. 

2. You can explore data samples at http://visualqa.org/browser/
'''

import tensorflow
from .utils import xavier_var
from . import attention

class CoreNet(object):
	def __init__(self, imglstm, txtlstm, FLAGS):
		self._ut = txtlstm.out
		self._vt = imglstm.out
		self._txt_len = tf.cast(txtlstm.len, tf.float32)

		self._mask = tf.placeholder(
			tf.float32, [None, FLAGS.max_txt_len])
		self._build_net(FLAGS)

	def _build_step0(self):
		v0 = tf.reduce_mean(self._vt, 1) # batch x 150 x img_out_dim
		u0 = tf.reduce_sum(self._ut, 1) / self._txt_len
		m0 = tf.multiply(v0, m0)
		self._vk, self._uk, self._mk = [v0], [u0], [m0]


	def _build_step(self, step, FLAGS):
		memory = self._mk[-1]
		
		v_step = attention.tanh_attend(
			FLAGS.img_out_dim, FLAGS.hid_dim, 
			self._vk[-1], memory, step)
		
		u_step = attention.tanh_attend(
			FLAGS.txt_out_dim, FLAGS.hid_dim, 
			self._uk[-1], memory, step, mask)
		
		self._vk.append(v_step)
		self._uk.append(u_step)

		return tf.multiply(v_step, u_step)

	def _build_net(self, FLAGS):
		self._build_step0()
		for step in range(FLAGS.nstep):
			m_next = self._build_step(step, FLAGS) 
			self._mk.append(m_next)
		self._out = self._mk[-1]

	@property
	def out(self):
		return self._out
	