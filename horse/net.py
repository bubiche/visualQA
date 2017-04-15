import tensorflow as tf
from .yolo import YOLO
from batch_yielder.batch_yielder import BatchYielder
import cv2
import numpy as np
import os
from .utils import cosine_sim, sharpen, tanh_gate, confusion_table
from .utils import conv_pool_act, xavier_var, const_var, gaussian_var
from .utils import conv_flat, conv_act
from .ops import op_dict
import pickle

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
		# self._ref = gaussian_var('ref', 0.0, 0.2, [1, 128])

		self._ref = gaussian_var(
			'ref', 0.0, 1.0, [1, 1024])
		self._build_placeholder()
		self._build_net()
		self._batch_yielder = BatchYielder(FLAGS)

	def _build_placeholder(self):
		self._volume = tf.placeholder(
			tf.float32, [None, 7, 7, 1024])

	def _build_net(self):
		self._fetches = []
		#volume_flat = tf.reshape(self._volume, [-1, 1024])

		def _leak(tensor):
			return tf.maximum(0.1 * tensor, tensor)

		tanh_vol = conv_act(self._volume, 1024, 1024, _leak, 'att_conv')
		tanh_vol = tf.reshape(tanh_vol, [-1, 1024])
		tanh_ref = tf.tanh(self._ref)

		similar = cosine_sim(tanh_vol, tanh_ref)
		similar = tf.nn.softmax(similar * 100)# * 2. - 1.
		# sign = tf.sign(similar)
		# similar = sign * tf.pow(sign * similar, 1./3.)
		# attention = (similar + 1.) / 2.
		# similar = tf.reshape(similar, [-1, 49])
		# similar = sharpen(similar)

		self._attention = tf.reshape(similar, [-1, 7, 7, 1])
		# convx = tf.reshape(convx, [-1, 49])
		# sharped = sharpen(convx)
		# sharped = tf.reshape(sharped, [-1, 49])
		# self._out = tf.reduce_sum(sharped, -1)
		#self._fetches += [self._yolo._inp]
		# focused = self._volume * self._attention

		# att1 = conv_act(self._volume, 1024, 512, _leak, 'att1')
		# att2 = conv_act(att1, 512, 256, _leak, 'att2')
		# att3 = conv_act(att2, 256, 128, tf.tanh, 'att3')
		# att3 = tf.reshape(att3, [-1, 128])
		# tanh_ref = tf.tanh(self._ref)

		# similar = cosine_sim(att3, tanh_ref)
		# similar = sharpen(similar)
		# similar = (similar + 1.)/2.

		# self._attention = tf.reshape(similar, [-1, 7, 7, 1])

		attended = self._volume * self._attention
		conv1 = conv_act(attended, 1024, 256, _leak, 'conv1')
		conv2 = conv_act(conv1, 256, 64, _leak, 'conv2')
		conv3 = conv_act(conv2, 64, 5, tf.nn.sigmoid, 'conv3')

		self._out = tf.reduce_sum(conv3,[1,2,3])

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
		fetches = [self._train_op, self._loss]
		fetches += [self._accuracy, self._attention]
		fetches = fetches + self._fetches

		for step, (feature, target) in batches:
			fetched = self._sess.run(fetches, {
				self._volume: feature,
				self._target: target})
			_, loss, accuracy, attention = fetched

			accuracy = int(accuracy * 100)
			loss_mva = loss if loss_mva is None else \
				loss_mva * .9 + loss * .1
			message = '{}. loss {} mva {} acc {}% '.format(
				step, loss, loss_mva, accuracy)


			if _mult(step, self._flags.valid_every):
				print('valid table:')
				valid_accuracy = self._accuracy_data(
					self._batch_yielder.validation_set())
				valid_accuracy = int(valid_accuracy * 100)
				message += 'valid acc {}% '.format(valid_accuracy)

			if _mult(step, self._flags.test_every):
				print('test table:')
				test_accuracy = self._accuracy_data(
					self._batch_yielder.test_set())
				test_accuracy = int(test_accuracy * 100)
				message += 'test acc {}%'.format(test_accuracy)
			
			if _mult(step, self._flags.save_every):
				print('train table:')
				train_accuracy = self._accuracy_data(
					self._batch_yielder.next_epoch())
				train_accuracy = int(train_accuracy * 100)
				message += 'train acc {}%'.format(train_accuracy)
				print('\n', attention)
				self._save_ckpt(step)

			_log(message)
				# img_name = 'horseref/horseref-{}.jpg'.format(step)
				# img_uint = (horse * 255.).astype(np.uint8)[0]
				# print(img_uint.shape)
				# cv2.imwrite(img_name, img_uint)

		self._save_ckpt(step)

	def _accuracy_data(self, data):
		volume_feed, target_feed = data
		acc, pred = self._sess.run([self._accuracy, self._out], {
				self._volume: volume_feed,
				self._target: target_feed
			})
		pred = np.round(pred).astype(np.int32)
		target_feed = target_feed.astype(np.int32)
		confusion_table(target_feed, pred)
		return acc

	def get_attention(self, vec):
		att = self._sess.run(self._attention, {
			self._volume: vec
			}).reshape([-1, 7, 7])
		print(att,'\n')
		return att

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