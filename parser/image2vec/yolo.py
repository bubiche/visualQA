from .cfg_parser import cfg_yielder
from .weight_parser import weights_parser
from .ops import op_dict
import tensorflow as tf
import cv2

class YOLO(object):
    def __init__(self, cfg_path, weight_path, up_to):
        self._weight_yielder = weights_parser(weight_path)
        self._build_net_up_to(cfg_path, up_to)

    def _build_net_up_to(self, cfg_path, up_to):
        self.layers = list()
        for i, layer_cfg in enumerate(cfg_yielder(cfg_path)):
            if i == 0:
                self.meta = layer_cfg
                self.inp = tf.placeholder(
                    tf.float32, [None] + self.meta['inp_size'])
                current = self.inp
                continue

            layer_type, index = layer_cfg[0], layer_cfg[1]
            if index > up_to: break

            layer = op_dict[layer_type](current)
            layer.build(self._weight_yielder, *layer_cfg[2:])
            self.layers.append(layer)
            current = layer.out
            #print(index, layer_type, current.get_shape().as_list())

        self.out = current
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def forward(self, img_list):
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

        return self.sess.run(
            self.out, {self.inp : preprocessed})
            
    def forward_frame(self, frame):
        def _preprocess(frame):
            h, w, c = self.meta['inp_size']
            imsz = cv2.resize(frame, (w, h))
            imsz = imsz / 255.
            #imsz = imsz[:,:,::-1]
            return imsz
        preprocessed = list()
        img_tensor = _preprocess(frame)
        preprocessed.append(img_tensor)

        return self.sess.run(
            self.out, {self.inp : preprocessed})

