from .cfg_parser import cfg_yielder
from .weight_parser import weights_parser
from .ops import op_dict
from horse.utils import gaussian_var
import tensorflow as tf
import numpy as np
import pickle
import cv2

class YOLO(object):
    def __init__(self, FLAGS):#,cfg_path, weight_path, up_to):
        weight_path = FLAGS.wgt
        cfg_path = FLAGS.cfg
        up_to = FLAGS.up_to
        print('Doing {}'.format(FLAGS.target))

        self._flags = FLAGS
        self._weight_yielder = weights_parser(weight_path)
        self._build_net_up_to(cfg_path, up_to)
        self._load_target()
        self._train_and_dump()

    def _build_net_up_to(self, cfg_path, up_to):
        self.layers = list()
        for i, layer_cfg in enumerate(cfg_yielder(cfg_path)):
            if i == 0:
                self.meta = layer_cfg
                self._inp = gaussian_var('inp', 
                    0.5, 0.05, [1, 64, 64, 3])
                current = self._inp
                continue

            layer_type, index = layer_cfg[0], layer_cfg[1]
            if index > up_to: break

            layer = op_dict[layer_type](current)
            layer.build(self._weight_yielder, *layer_cfg[2:])
            self.layers.append(layer)
            current = layer.out
            #print(index, layer_type, current.get_shape().as_list())

        self._out = tf.reshape(current, [1,1024])
        self._build_loss()
        self._build_trainer()

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def _build_loss(self):
        self._target = tf.placeholder(
            tf.float32, self._out.get_shape())
        self._loss = tf.nn.l2_loss(self._target - self._out)

    def _build_trainer(self):
        optimizer = tf.train.AdamOptimizer(self._flags.lr)
        gradients = optimizer.compute_gradients(self._loss)
        self._train_op = optimizer.apply_gradients(gradients)

    def _load_target(self):
        file = self._flags.target
        with open(file, 'rb') as f:
            self._target_val = pickle.load(f)

    def _train_and_dump(self):
        for step in range(self._flags.epoch):
            _, loss = self._sess.run(
                [self._train_op, self._loss], {
                self._target: self._target_val
            })
            print('step {} loss {}'.format(step+1, loss))

        img_ref = self._sess.run(self._inp)
        img_ref = img_ref.reshape([64,64,3]) * 255.
        img_ref = img_ref.astype(np.uint8)
        cv2.imwrite('{}.jpg'.format(
            self._flags.target), img_ref)