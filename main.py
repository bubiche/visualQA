from horse.net import HorseNet
from tensorflow import flags
import os

flags.DEFINE_integer('batch_size', 128, 'size of each batch')
flags.DEFINE_integer('epoch', 1000, 'number of epoch')
flags.DEFINE_string('vec_path', 'parser/full_vec.hdf5', 'path of vec file')
flags.DEFINE_string('count_path', 'parser/full_count.hdf5', 'path of count file')
flags.DEFINE_integer('n_use', -1, 'number of training example to be used')
flags.DEFINE_boolean('train', True, 'train the whole net')
flags.DEFINE_float('lr', 1e-5, 'learning rate')
flags.DEFINE_string('trainer', 'adam', 'training algorithm')
flags.DEFINE_string('cfg', 'horse/yolo-full.cfg', 'path to YOLO cfg file')
flags.DEFINE_string('weight', 'horse/yolo-full.weights', 'path to YOLO weight file')
flags.DEFINE_string('backup', 'backup', 'path to ckpt folder')
flags.DEFINE_integer('save_every', 100, 'ckpt every x step')
flags.DEFINE_integer('test_every', 100, 'test every x step')
flags.DEFINE_integer('valid_every', 20, 'validate every x step')
flags.DEFINE_integer('keep', 20, 'maximum ckpt to keep')
flags.DEFINE_integer('load', 0, 'load from ckpt x?')
FLAGS = flags.FLAGS

def get_dir(dirs):
    for d in dirs:
        this = os.path.abspath(os.path.join(os.path.curdir, d))
        if not os.path.exists(this): os.makedirs(this)
get_dir([FLAGS.backup])

horse_net = HorseNet(FLAGS)

if FLAGS.load:
	horse_net.load_from_ckpt()

if FLAGS.train:
    print('Enter training ...')
    horse_net.train()
    print('Training finished.')
    
#horse_net.predict()