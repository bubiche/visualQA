import tensorflow as tf
from .bilstm import biLSTM
from .corenet import CoreNet
from .utils import _log, xavier

def _last_dim(tensor):
	shape = tensor.get_shape()
	shape_list = shape.as_list()
	return shape_list[-1]

class DAN(object):

	_TRAINER = dict({
		'rmsprop': tf.train.RMSPropOptimizer,
		'adadelta': tf.train.AdadeltaOptimizer,
		'adagrad': tf.train.AdagradOptimizer,
		'adagradDA': tf.train.AdagradDAOptimizer,
		'momentum': tf.train.MomentumOptimizer,
		'adam': tf.train.AdamOptimizer,
		'ftrl': tf.train.FtrlOptimizer,
	})

	def __init__(self, FLAGS):
		self.FLAGS = FLAGS
		self._txtlstm = biLSTM(
			FLAGS.max_txt_len, FLAGS.txt_vec_len, FLAGS.txt_out_dim)
		self._imglstm = biLSTM(
			FLAGS.max_img_len, FLAGS.img_vec_len, FLAGS.img_out_dim)
		self._corenet = CoreNet(
			self._txtlstm, self._imglstm, FLAGS)

		# core net's out is feature extracted.
		self._feature = self._corenet.out # [batch x m_dim]
		self._inp_dim = _last_dim(self._feature) # m_dim

		# place holders
		self._img_vec = self._imglstm.vec
		self._img_len = self._imglstm.len
		self._txt_vec = self._txtlstm.vec
		self._txt_len = self._txtlstm.len


		if not FLAGS.train: return
		self._build_loss()
		self._build_trainer()

	def _img_len(self):
		return np.ones([FLAGS.batch_size]) * FLAGS.max_img_len

	def train(self, batch_yielder):

		def _create_mask(txt_len):
			return np.arange(FLAG.max_txt_len) < txt_len[:, None]

		loss_mva = None
		batches = enumerate(batch_yielder.next_batch())
		for step, ((img_vec, txt_vec, txt_len), target) in batches:
			loss = self.sess.run([self._train_op, self._loss], {
				self._img_vec: img_vec,
				self._img_len: self._img_len(),
				self._txt_vec: txt_vec,
				self._txt_len: txt_len,
				self._corenet._mask: _create_mask(txt_len)
				self._target: target
			})

			loss_mva = loss if loss_mva is None else
				loss_mva * .9 + loss * .1
			_log('step = {}, loss = {}, loss_mva = {}'.format(
				step, loss, loss_mva))

	def _build_loss(self):
		pass

	def _build_trainer(self):
	    optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
	    gradients = optimizer.compute_gradients(self._loss)
	    self._train_op = optimizer.apply_gradients(gradients)


class RegressDAN(DAN):
	def _build_loss(self):
		W = xavier_var('W', [self._inp_dim])
		self._out = tf.matmul(self._feature, W)
		self._target = tf.placeholder(tf.float32, [None])
		self._loss = tf.nn.l2_loss(self._out - self._target)


class ClassifyDAN(DAN):
	def _build_loss(self):
		nclass = self.FLAGS.nclass
		W = xavier_var('W', [self._inp_dim, nclass])
		self._out = tf.matmul(self._feature, W)
		self._target = tf.placeholder(tf.float32, [None, nclass])
		self._loss = tf.nn.softmax_cross_entropy_with_logits(
			self._target, self._out)
