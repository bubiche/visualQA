import tensorflow as tf
from .yolo import YOLO
from batch_yielder.batch_yielder import BatchYielder
from batch_yielder.batch_yielder_binhyen import BatchYielderBinhYen
import cv2
import numpy as np
import os
from .utils_khung import ref, no_ref, no_sharp, softmax, power, count
from .ops import op_dict
import pickle
import time

def _log(*msgs):
	for msg in list(msgs):
		print(msg)

def _mult(a, b):
	return (a + 1) % b == 0

ref_list = [no_ref, ref]
sharp_list = [no_sharp, softmax, power]

config_dict = dict({
	0: (None, 'no_attention'),
	1: (0, 0, 'noref_nosharp'),
	2: (1, 0, 'ref_nosharp'),
	3: (0, 1, 'noref_softmax'),
	4: (1, 1, 'ref_softmax'),
	5: (0, 2, 'noref_power'),
	6: (1, 2, 'ref_power')
})

labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]

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

		self._name = '{}_{}'.format(
			config_dict[FLAGS.config][-1],	FLAGS.cls)

		self._build_placeholder()
		self._build_net()
		self._batch_yielder = BatchYielderBinhYen(FLAGS)

	def _build_placeholder(self):
		self._volume = tf.placeholder(
			tf.float32, [None, 7, 7, 1024])

	def _build_net(self):
		config = config_dict[self._flags.config]
		if config[0] is None:
			self._attention = tf.constant(
				np.ones([self._flags.batch_size, 7, 7, 1]), dtype = np.float32)
		else:
			ref_fun = ref_list[config[0]]
			attention = ref_fun(self._volume)
			sharp_fun = sharp_list[config[1]]
			attention = sharp_fun(attention)
			self._attention = tf.reshape(
				attention, [-1, 7, 7, 1])

		attended = self._volume * self._attention
		self._out = count(attended)

		if self._flags.train:
			self._build_loss()
			self._build_trainer()

		self._sess = tf.Session()
		self._sess.run(tf.global_variables_initializer())

	def _build_loss(self):
		self._target = tf.placeholder(tf.float32, [None])
		self._loss = tf.nn.l2_loss(self._target - self._out)
		int_target = tf.cast(tf.round(self._target), tf.int32)
		int_out = tf.cast(tf.round(self._out), tf.int32)
		correct = tf.equal(int_target, int_out)
		correct = tf.cast(correct, tf.float32)
		self._accuracy = tf.reduce_mean(correct)
		deviation = tf.abs(int_target - int_out)
		self._deviation = tf.reduce_mean(deviation)

	def _build_trainer(self):
		optimizer = self._TRAINER[self._flags.trainer](self._flags.lr)
		gradients = optimizer.compute_gradients(self._loss)
		self._train_op = optimizer.apply_gradients(gradients)
		self._saver = tf.train.Saver(tf.global_variables(),
			max_to_keep = self._flags.keep)

	def load_from_ckpt(self):
		load_name = '{}-{}'.format(self._name, self._flags.load)
		load_path = os.path.join(self._flags.backup, load_name)
		print('Loading from {}'.format(load_path))
		self._saver.restore(self._sess, load_path)

	def _save_ckpt(self, step, log):
		file_name = '{}-{}'.format(self._name, step)
		path = os.path.join(self._flags.backup, file_name)
		if os.path.isfile(path): return

		print('Saving ckpt at step {}'.format(step))
		self._saver.save(self._sess, path)

		msg = '{} {}'.format(self._name, log)
		print('loging {}'.format(msg))
		with open('accuracy_log', 'a') as f:
			f.write(msg + '\n')


	def train(self):
		loss_mva = None
		batches = enumerate(self._batch_yielder.next_batch())
		fetches = [self._train_op, self._loss]
		fetches += [self._accuracy, self._deviation]

		start = time.time()
		for step, (feature, target) in batches:
			fetched = self._sess.run(fetches, {
				self._volume: feature,
				self._target: target})
			_, loss, accuracy, deviation = fetched

			loss_mva = loss if loss_mva is None else \
				loss_mva * .9 + loss * .1
			message = '{} {}. loss {0:.3f} mva {0:.3f} acc {0:.3f}%, dev {0:.3f} '.format(
				self._name, step, loss, loss_mva, accuracy * 100, deviation)

			if _mult(step, self._flags.test_every):
				print('test table:')
				test_accuracy, test_dev = self._accuracy_data(
					self._batch_yielder.test_set())
				message += 'test acc {0:.3f}%, dev {0:.3f} '.format(
					test_accuracy * 100, test_dev)

			fps = (time.time()-start) / step
			message += '{}fps'.format(fps)
			_log(message)
				# img_name = 'horseref/horseref-{}.jpg'.format(step)
				# img_uint = (horse * 255.).astype(np.uint8)[0]
				# print(img_uint.shape)
				# cv2.imwrite(img_name, img_uint)

		self._save_ckpt(step, log = 'acc {} dev {}'.format(
			test_accuracy, test_dev))

	def _accuracy_data(self, data):
		volume_feed, target_feed = data
		dev, acc, pred = self._sess.run(
			[self._deviation, self._accuracy, self._out], {
				self._volume: volume_feed,
				self._target: target_feed
			})
		pred = np.round(pred).astype(np.int32)
		target_feed = target_feed.astype(np.int32)
		print('doin table')
		confusion_table(target_feed, pred)
		return acc, dev

	def get_attention(self, vec):
		att, pred = self._sess.run(
			[self._attention, self._out], {
			self._volume: vec
			}).reshape([-1, 7, 7])
		print(att,'\n')
		pred = np.round(pred).astype(np.int32)
		return att, pred