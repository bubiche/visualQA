import tensorflow as tf

class biLSTM(object):

	_MAX_LEN = 40
	_VEC_DIM = 300
	_OUT_DIM = 512

	def __init__(self):
		self._out_dim = _OUT_DIM
		self._build_placeholders()
		self._build_bilstm()

	def _build_placeholders(self):
		self.vec = tf.placeholder(tf.float32,
			[None, self._MAX_LEN, self._VEC_DIM])
		self.len = tf.placeholder(tf.int32, [None])

	def _build_bilstm(self):
		cell_fw = tf.contrib.rnn.BasicLSTMCell(
			num_units = self._OUT_DIM, 
			input_size = self._VEC_DIM)
		cell_bw = tf.contrib.rnn.BasicLSTMCell(
			num_units = self._OUT_DIM, 
			input_size = self._VEC_DIM)

		outputs, _ = tf.nn.bidirectional_dynamic_rnn(
			cell_fw, cell_bw, self._vec, self._len)
		out_fw, out_bw = outputs
		self._out = out_fw + out_bw

	@property
	def out(self):
		return self._out

	@property
	def out_dim(self):
		return self._out_dim
	
	@property
	def len(self):
		return self._len