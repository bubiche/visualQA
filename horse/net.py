import tensorflow as tf
from yolo import YOLO
from batch_yielder.batch_yielder import BatchYielder
import cv2
from .utils import cosine_sim, sharpen
from .utils import conv_pool_leak, xavier_var
from .ops import op_dict

def _log(*msgs):
	for msg in list(msgs):
		print(msg)

class HorseNet(object):

	_TRAINER = dict({
		'sgd': tf.train.GradientDescentOptimizer,
		'rmsprop': tf.train.RMSPropOptimizer,
		'adadelta': tf.train.AdadeltaOptimizer,
		'adagrad': tf.train.AdagradOptimizer,
		'adagradDA': tf.train.AdagradDAOptimizer,
		'momentum': tf.train.MomentumOptimizer,
		'adam': tf.train.AdamOptimizer,
		'ftrl': tf.train.FtrlOptimizer,
	})

	def __init__(self, FLAGS):
		self._flags = FLAGS
		self._yolo = YOLO(FLAGS.cfg, FLAGS.weight, 28)
		self.batch_yielder = BatchYielder(
			FLAGS.batch_size, FLAGS.epoch,
			FLAGS.vec_path, FLAGS.count_path, FLAGS.n_use)

		self._build_placeholder()
		self._build_net()

	def _build_placeholder(self):
		self._volume = tf.placeholder(
			tf.float32, [None, 7, 7, 1024])

	def _build_net(self):
		volume_flat = tf.reshape(self._volume, [-1, 1024])
		reference = tf.reshape(self._yolo.out, [1024])

		similar = cosine_sim(volume_flat, reference)
		similar = tf.reshape(simiar, [-1, 49])

		sharped = sharpen(similar)
		attention = tf.reshape(sharped, [-1, 7, 7, 1])
		focused = self._volume * attention

		conved = conv_pool_leak(self._focused, 1024, 2048)
		feat = tf.reduce_sum(conved, [0, 1, 2])

		out = tf.matmul(feat, xavier_var([2048, 1]))
		out += xavier_var([1,])
		self._out = tf.nn.relu(out)

		if self._flags.train:
			self._build_loss()
			self._build_trainer()

		self._sess = tf.Session()
		self._sess.run(tf.global_variables_initializer())

	def _build_loss(self):
		self._target = tf.placeholder(tf.float32, [None])
		self._loss = tf.nn.l2_loss(self._target - self._out)

	def _build_trainer(self):
	    optimizer = self._TRAINER[self._flags.trainer](self._flags.lr)
	    gradients = optimizer.compute_gradients(self._loss)
	    self._train_op = optimizer.apply_gradients(gradients)

	def train(self):
		loss_mva = None
		batches = enumerate(self.batch_yielder.next_batch())
		for step, (feature, target) in batches:
			_, loss = self._sess.run([self._train_op, self._loss], {
				self._volume: feature,
				self._target: target
			})

			loss_mva = loss if loss_mva is None else
				loss_mva * .9 + loss * .1
			_log('step = {}, loss = {}, loss_mva = {}'.format(
				step, loss, loss_mva))

	def predict(self, img_list):
		def _preprocess(img_path):
			im = cv2.imread(img_path)
			h, w, c = self.meta['inp_size']
			imsz = cv2.resize(im, (w, h))
			imsz = imsz / 255.
			imsz = imsz[:,:,::-1]
			return imsz

		preprocessed = list()
		for img_path in img_list:
			img_tensor = _preprocess(img_path)
			preprocessed.append(img_tensor)

		return self._sess.run(
			self._out, {self._volume : preprocessed})



