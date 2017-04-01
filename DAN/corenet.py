'''
1. As described in the paper, 
	(m, h, u_t, u^(k), v^(k)) is 512, 

	while the pretrained image feature vector (v_n) has 2048 dimensions. 

	Note that the alpha^(k)_v,n in Eq. 5-6 and alpha^(k)_u,t in Eq. 9-10 is a scalar value. 

2. You can explore data samples at http://visualqa.org/browser/
'''

import tensorflow
from .utils import xavier_var
from .attention import img_attend, txt_attend

class CoreNet(object):
	
	_IMG_NFT = 150
	_IMG_DIM = 512
	_HID_DIM = 512

	def __init__(self, bilstm, nstep):
		self.ut = bilstm.out
		self._len = bilstm.len
		self._len = tf.cast(self._len, tf.float32)
		self._TXT_DIM = bilstm.out_dim

		self.vn = tf.placeholder(tf.float32, 
			[None, self._IMG_NFT, self._IMG_DIM])

		self._build_net(nstep)

	def _build_step0(self):
		self._P0 = xavier_var(
			'P_0', [self._IMG_DIM, self._TXT_DIM])
		v0 = tf.tanh(tf.matmul(
			tf.reduce_mean(self.vn, 1), self._P0))
		u0 = tf.reduce_sum(self._ut, 1) / self._len
		m0 = tf.multiply(v0, m0)

		self._vk, self._uk, self._mk = [v0], [u0], [m0]

	_img_attend = img_attend
	_txt_attend = txt_attend

	def _build_step(self, memory, step):
		v_step = self._img_attend(memory, step)
		u_step = self._txt_attend(memory, step)
		self._vk.append(v_step)
		self._uk.append(u_step)

		return tf.multiply(v_step, u_step)

	def _build_net(self, nstep):
		self._build_step0()
		for step in range(nstep):
			m_next = self._build_step(self._mk[-1], step) 
			self._mk.append(m_next)
		self._out = self._mk[-1]

	@property
	def out(self):
		return self._out
	