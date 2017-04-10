import tensorflow as tf
from .yolo import YOLO
from batch_yielder.batch_yielder import BatchYielder
import cv2
import os
from .utils import cosine_sim, sharpen, tanh_gate
from .utils import conv_pool_leak, xavier_var, const_var
from .ops import op_dict

def _log(*msgs):
	for msg in list(msgs):
		print(msg)

def _mult(a, b):
	return (a + 1) % b == 0

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
		self._batch_yielder = BatchYielder(FLAGS)
		self._build_placeholder()
		self._build_net()

	def _build_placeholder(self):
		self._volume = tf.placeholder(
			tf.float32, [None, 7, 7, 1024])

	def _build_net(self):
		self._fetches = list()
		volume_flat = tf.reshape(self._volume, [-1, 1024])
		reference = tf.reshape(self._yolo.out, [1, 1024])

		with tf.variable_scope('tanh_gate'):
			tanh_vol = tanh_gate(volume_flat, 1024, 512)

		with tf.variable_scope('tanh_gate', reuse = True):
			tanh_ref = tanh_gate(reference, 1024, 512)

		similar = cosine_sim(tanh_vol, tanh_ref)
		similar = tf.reshape(similar, [-1, 49])
		similar = (similar + 1.) / 2.

		sharped = tf.nn.softmax(similar)
		#self._fetches += [self._yolo._inp]
		attention = tf.reshape(sharped, [-1, 7, 7, 1])
		focused = self._volume * attention

		conved = conv_pool_leak(focused, 1024, 2048)
		feat = tf.reduce_sum(conved, [1, 2])

		out = tf.matmul(feat, xavier_var('fcw', [2048, 1]))
		out += const_var('fcb', 0.0, [1,])
		self._out = tf.nn.softplus(out)

		if self._flags.train:
			self._build_loss()
			self._build_trainer()

		self._sess = tf.Session()
		self._sess.run(tf.global_variables_initializer())

	def _build_loss(self):
		self._target = tf.placeholder(tf.float32, [None])
		self._loss = tf.nn.l2_loss(self._target - self._out)
		int_target = tf.cast(self._target, tf.int32)
		int_out = tf.cast(self._out, tf.int32)
		correct = tf.equal(int_target, int_out)
		correct = tf.cast(correct, tf.float32)
		self._accuracy = tf.reduce_mean(correct)

	def _build_trainer(self):
		optimizer = self._TRAINER[self._flags.trainer](self._flags.lr)
		gradients = optimizer.compute_gradients(self._loss)
		self._train_op = optimizer.apply_gradients(gradients)
		self._saver = tf.train.Saver(tf.global_variables(),
			max_to_keep = self._flags.keep)

	def load_from_ckpt(self):
		load_name = 'horse-{}'.format(self._flags.load)
		load_path = os.path.join(self._flags.backup, load_name)
		print('Loading from {}'.format(load_path))
		self._saver.restore(self._sess, load_path)

	def _save_ckpt(self, step):
		file_name = 'horse-{}'.format(step)
		path = os.path.join(self._flags.backup, file_name)
		if os.path.isfile(path): return
		print('Saving ckpt at step {}'.format(step))
		self._saver.save(self._sess, path)

	def train(self):
		loss_mva = None
		batches = enumerate(self._batch_yielder.next_batch())
		fetches = [self._train_op, self._loss, self._accuracy]
		fetches = fetches + self._fetches

		once = (None, None)
		for step, (feature, target) in batches:
			if step > 0:
				feature, target = save
			fetched = self._sess.run(fetches, {
				self._volume: feature,
				self._target: target})
			_, loss, accuracy = fetched
			#print(sharp)
			accuracy = int(accuracy * 100)
			loss_mva = loss if loss_mva is None else \
				loss_mva * .9 + loss * .1
			message = '{}. loss {} mva {} acc {}% '.format(
				step, loss, loss_mva, accuracy)

			if step == 0:
				save = (feature, target)
				continue

			# if _mult(step, self._flags.valid_every):
			# 	valid_accuracy = self._accuracy_data(
			# 		self._batch_yielder.validation_set())
			# 	valid_accuracy = int(valid_accuracy * 100)
			# 	message += 'valid acc {}% '.format(valid_accuracy)

			# if _mult(step, self._flags.test_every):
			# 	test_accuracy = self._accuracy_data(
			# 		self._batch_yielder.test_set())
			# 	test_accuracy = int(test_accuracy * 100)
			# 	message += 'test acc {}%'.format(test_accuracy)

			_log(message)
			
			# if _mult(step, self._flags.save_every):
			# 	self._save_ckpt(step)

		self._save_ckpt(step)

	def _accuracy_data(self, data):
		volume_feed, target_feed = data
		return self._sess.run(self._accuracy, {
				self._volume: volume_feed,
				self._target: target_feed
			})

	def predict_img(self):
		def _preprocess(img_path):
			im = cv2.imread(img_path)
			h, w, c = self.meta['inp_size']
			imsz = cv2.resize(im, (w, h))
			imsz = imsz / 255.
			imsz = imsz[:,:,::-1]
			return imsz

		preprocessed = list()
		for img_path in os.listdir(self._flags.test_imgs):
			if not os.path.isfile(img_path):
				continue
			img_tensor = _preprocess(img_path)
			preprocessed.append(img_tensor)

		return self._sess.run(
			self._out, {self._volume : preprocessed})