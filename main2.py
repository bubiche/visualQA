from tensorflow import flags
import tensorflow as tf
from image2vec.yolo import YOLO
import os

flags.DEFINE_integer('epoch', 1000, 'no. epoch')
flags.DEFINE_string('cfg', 
	'./parser/image2vec/yolo-full.cfg', 'where is the config')
flags.DEFINE_string('wgt', 
	'./parser/image2vec/yolo-full.weights', 'where is the weights')
flags.DEFINE_integer('up_to', 28, 'up to layer')
flags.DEFINE_float('lr', 10, 'learning rate')
flags.DEFINE_string('target', 'whatever', 'just a placeholder')
FLAGS = flags.FLAGS


all_targets = os.listdir('trained_refs')
for target in all_targets:
	FLAGS.target = './trained_refs/' + target
	tf.reset_default_graph()
	net = YOLO(FLAGS)
