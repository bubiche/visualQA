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
		self._bilstm = biLSTM()
		self._corenet = CoreNet(
			self._bilstm, FLAGS.nstep)

		self._vn = self._corenet.vn
		self._vec = self._bilstm.vec
		self._len = self._bilstm.len
		self._feature = self._corenet.out
		self._inp_dim = _last_dim(self._feature)

		if not FLAGS.train: return
		self._build_loss()
		self._build_trainer()


	def train(self, datian):
		loss_mva = None

		batches = datian.shuffle()
		for step, (img_vec, txt_vec, ), target in enumerate(batches):
			loss = self.sess.run([self.train_op, self.loss], {
				self._vn: img_vec,
				self._vec: txt_vec,
				self._len: txt_len,
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
	    self.train_op = optimizer.apply_gradients(gradients)


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
