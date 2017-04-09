from horse.net import HorseNet
from tensorflow import flags
import os

flags.DEFINE_integer("batch_size", 128, "size of each batch")
flags.DEFINE_integer("epoch", 1000, "number of epoch")
flags.DEFINE_string("vec_path", 'parser/full_vec.hdf5', "path of vec file")
flags.DEFINE_string("count_path", 'parser/full_count.hdf5', "path of count file")
flags.DEFINE_integer("n_use", -1, "number of training example to be used")
flags.DEFINE_boolean("train", True, "train the whole net")
flags.DEFINE_float("lr", 1e-5, "learning rate")
flags.DEFINE_string("trainer", "rmsprop", "training algorithm")
flags.DEFINE_boolean("savepb", False, "save net and weight to a .pb file")
flags.DEFINE_string("cfg", "parser/image2vec/yolo-full.cfg", "path to YOLO cfg file")
flags.DEFINE_string("weight", "parser/image2vec/yolo-full.weights", "path to YOLO weight file")
FLAGS = flags.FLAGS

horse_net = HorseNet(FLAGS)

if FLAGS.train:
    print('Enter training ...'); horse_net.train()
    if not FLAGS.savepb: exit('Training finished')
    
flags.DEFINE_boolean("savepb", False, "save net and weight to a .pb file")

horse_net.predict()