from .cfg_parser import cfg_yielder
from .weight_parser import weights_parser
from .ops import op_dict
from .utils import xavier_var, gaussian_var
import tensorflow as tf
import numpy as np
import cv2


labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
	"train", "tvmonitor"]


class YOLOtop(object):
	def __init__(self, cfg_path, weight_path, from_layer):
		self._weight_yielder = weights_parser(weight_path)
		self._build_net_from(cfg_path, from_layer)

	def _build_net_from(self, cfg_path, from_layer):
		self.layers = list()
		flag  = False
		for i, layer_cfg in enumerate(cfg_yielder(cfg_path)):
			if i == 0:
				self._meta = layer_cfg
				self._inp = tf.placeholder(
					tf.float32, [None, 448, 448, 3])
				current = self._inp
				continue

			layer_type, index = layer_cfg[0], layer_cfg[1]
			if not flag and index == from_layer + 1:
				print(current.get_shape(), 'now a placeholder')
				current = tf.placeholder(
					tf.float32, current.get_shape().as_list())
				self._inp = current
				flag = True

			layer = op_dict[layer_type](current)
			layer.build(self._weight_yielder, *layer_cfg[2:])
			self.layers.append(layer)
			current = layer.out
			print(index, layer_type, current.get_shape().as_list())

		self.out = current
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def process_box(self, b, threshold):
		max_indx = np.argmax(b.probs)
		max_prob = b.probs[max_indx]
		label = labels20[max_indx]
		if max_prob > threshold:
			return label
		return None

	def postprocess(self, net_out):
		"""
		Takes net output, draw predictions, save to disk
		"""
		boxes = yolo_box_constructor(self._meta, net_out, 0.2)

		count = 0
		for b in boxes:
			if self.process_box(b, threshold) == 'horse':
				count += 1
		return count

	def forward(self, vecs):
		outs = self.sess.run(self.out, {
			self._inp: vecs
		})
		res = list()
		for out in outs:
			res.append(self.postprocess(out))
		return np.array(res)
