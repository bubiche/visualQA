from horse.net import HorseNet
from tensorflow import flags
from parser.horse_net_user import Visualizer
import os

flags.DEFINE_integer('batch_size', 1024, 'size of each batch')
flags.DEFINE_integer('epoch', 1000, 'number of epoch')
flags.DEFINE_string('vec_path', 'parser/train_vec.hdf5', 'path of train vec file')
flags.DEFINE_string('count_path', 'parser/train_count.hdf5', 'path of train count file')
flags.DEFINE_string('val_vec_path', 'parser/val_vec.hdf5', 'path of val vec file')
flags.DEFINE_string('val_count_path', 'parser/val_count.hdf5', 'path of val count file')
flags.DEFINE_string('test_vec_path', 'parser/test_vec.hdf5', 'path of test vec file')
flags.DEFINE_string('test_count_path', 'parser/test_count.hdf5', 'path of test count file')
flags.DEFINE_integer('n_use', -1, 'total number of data to be used')
flags.DEFINE_integer('val_ratio', 15, 'ratio of data to be used for validation')
flags.DEFINE_integer('test_ratio', 15, 'ratio of data to be used for test')
flags.DEFINE_boolean('train', True, 'train the whole net')
flags.DEFINE_float('lr', 1e-5, 'learning rate')
flags.DEFINE_string('trainer', 'adam', 'training algorithm')
flags.DEFINE_string('cfg', 'horse/yolo-full.cfg', 'path to YOLO cfg file')
flags.DEFINE_string('weight', 'horse/yolo-full.weights', 'path to YOLO weight file')
flags.DEFINE_string('backup', 'backup', 'path to ckpt folder')
flags.DEFINE_integer('save_every', 100, 'ckpt every x step')
flags.DEFINE_integer('test_every', 100, 'test every x step')
flags.DEFINE_integer('valid_every', 20, 'validate every x step')
flags.DEFINE_integer('epoch_every', 100, 'epoch every x step')
flags.DEFINE_integer('keep', 20, 'maximum ckpt to keep')
flags.DEFINE_integer('load', 0, 'load from ckpt x?')
flags.DEFINE_string('see', '', 'name of the image')
flags.DEFINE_string('full_vec_path', 'parser/full_vec.hdf5', 'path to full vec')
flags.DEFINE_string('full_name_path', 'parser/full_name.hdf5', 'path to full name')
flags.DEFINE_string('full_count_path', 'parser/full_count.hdf5', 'path to full count')
flags.DEFINE_string('see_path', 'seer_input/', 'path to images to get attention')
flags.DEFINE_boolean('equal', False, 'make number of images with horse and without horse equal each epoch')
flags.DEFINE_string('data_path', 'parser/', 'path to all image data')
flags.DEFINE_boolean('noise', False, 'train with noise added to images')
flags.DEFINE_string('cls', 'horse', 'the class to be trained with')
flags.DEFINE_string('voc_vec_path', 'parser/full_vec_voc.hdf5', 'path to vec file')
flags.DEFINE_string('voc_count_path', 'parser/full_count_voc.hdf5', 'path to count file')
flags.DEFINE_string('split_path', 'parser/full_split_voc.hdf5', 'path to split file')
flags.DEFINE_string('voc_name_path', 'parser/full_name_voc.hdf5', 'path to name file')
flags.DEFINE_boolean('see_test', False, 'visualize attention in the test set')
flags.DEFINE_boolean('see_wrong', False, 'visualize attention in wrong images of the test set')
flags.DEFINE_integer('config', 0, 'configuration to be used')

FLAGS = flags.FLAGS

def get_dir(dirs):
    for d in dirs:
        this = os.path.abspath(os.path.join(os.path.curdir, d))
        if not os.path.exists(this): os.makedirs(this)
get_dir([FLAGS.backup, 'horseref'])

horse_net = HorseNet(FLAGS)

if FLAGS.load:
    horse_net.load_from_ckpt()
    
if FLAGS.see != '':
    seer = Visualizer(FLAGS, horse_net)
    vecs = seer.get_vecs()
    ret_vec = horse_net.get_attention(vecs)
    seer.visualize_multiple(ret_vec)
    exit()

if FLAGS.see_test:
    seer = Visualizer(FLAGS, horse_net)
    seer.visualize_test_set()
    exit()
    
if FLAGS.see_wrong:
    if FLAGS.config == 0: exit()
    seer = Visualizer(FLAGS, horse_net)
    seer.visualize_wrong_test()
    exit()
    
if FLAGS.train:
    print('Enter training ...')
    horse_net.train()
    print('Training finished.')
    
#horse_net.predict()