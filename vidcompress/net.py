import tensorflow as tf
from DAN.bilstm import biLSTM

class CompressNet(object):
	def __init__(self, FLAGS):
		self._flags = FLAGS
		self._img_encode = biLSTM(
			FLAGS.vid_len, FLAGS.frame_dim, FLAGS.out_dim)
		self._skip_thought = SkipThought()

		# Placeholders
		self._frames = self._img_encode.vec
		self._vid_len = self._img_encode.len
		self._sentences = tf.placeholder(
			tf.float32, [None, self._flags.skip_thought_len])

		# 

	def _build_net(self):