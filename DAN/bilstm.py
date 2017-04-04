import tensorflow as tf

class biLSTM(object):

	def __init__(self, max_len, vec_dim, out_dim):
		self._max_len = max_len # 150 / 40
		self._vec_dim = vec_dim # 512 / 300
		self._out_dim = out_dim # 512 / 512

		self._build_placeholders()
		self._build_bilstm()

	def _build_placeholders(self):
		self.vec = tf.placeholder(tf.float32,
			[None, self._max_len, self._vec_dim])
		self.len = tf.placeholder(tf.int32, [None])

	def _build_bilstm(self):
		cell_fw = tf.contrib.rnn.BasicLSTMCell(
			num_units = self._out_dim, 
			input_size = self._vec_dim)
		cell_bw = tf.contrib.rnn.BasicLSTMCell(
			num_units = self._out_dim, 
			input_size = self._vec_dim)

		outputs, _ = tf.nn.bidirectional_dynamic_rnn(
			cell_fw, cell_bw, self.vec, self.len)
		out_fw, out_bw = outputs
		self.out = out_fw + out_bw