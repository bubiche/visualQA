import json
import skvideo.io
import os
import h5py
import shutil
from image2vec import yolo
import skipthoughts

EXTRACT_FRAME_COUNT = 150
BASE_VID_PATH = '/home/npnguyen/ugvideo/train_videos/'

# load models
print('Load YOLO...')
yolo_net = yolo.YOLO(
    'image2vec/yolo-small.cfg', 
    'image2vec/yolo-small.weights',
    up_to = 29)

#print('Load Skipthoughts...')
#txt_model = skipthoughts.load_model()
#txt_encoder = skipthoughts.Encoder(txt_model)

def get_vid_path_from_vid_id(vid_id):
    path_list = [BASE_VID_PATH, vid_id, '.mp4']
    return ''.join(path_list)

def txt_to_vec(sentence):
    return (txt_encoder.encode([sentence]))[0]

def extract_images(cap, n, directory, step, skip, verbose = True):
    count_save = 0
    i = skips

    while i < n:
        skvideo.io.vwrite("%s/frame%d.bmp" % (directory, count_save), cap[i])
        count_save += 1    
            
        if count_save == EXTRACT_FRAME_COUNT:
            return
        
        i += step

def vid_vec_from_dir(directory):
    img_list = [f for f in os.listdir(directory) if x.endswith('.bmp')]
    print('Frame count: %d' % (len(img_list)))
    vec = yolo_net.forward(img_list)
    shutil.rmtree(directory, ignore_errors=True)
    return vec
        
def vid_to_vec(filename):
    cap = skvideo.io.vread(filename)
    metadata = skvideo.io.ffprobe(filename)
    frame_count = int(metadata['video']['@nb_frames'])
        
    step = 0
    skip = 0
    if frame_count < EXTRACT_FRAME_COUNT:
        step = 1
    else:
        step = int(frame_count/EXTRACT_FRAME_COUNT)
        skip = int((frame_count - step * EXTRACT_FRAME_COUNT)/2) - 1
            
    directory = (os.path.splitext(filename)[0])
    if not os.path.exists(directory):
        os.makedirs(directory)

    extract_images(cap, frame_count, directory, step, skip, True)
    return vid_vec_from_dir(directory)

# create hdf5 files
vid_vec_file = h5py.File('vid_vec.hdf5', 'w')
#txt_vec_file = h5py.File('txt_vec.hdf5', 'w')

with open('train_meta_v2.json') as data_file:    
    data = json.load(data_file)

# create datasets
video_list = data['meta']
data_size = len(video_list)
vid_dset = vid_vec_file.create_dataset('vid_vec', (data_size, 150, 512), dtype='f')
#txt_dset = txt_vec_file.create_dataset('txt_vec', (data_size, 4800), dtype='f')

# parse meta data
i = 0
for vid in video_list:
    filename = get_vid_path_from_vid_id(vid['video_id'])
    print('Working on %s' % (filename))
    vid_vec = vid_to_vec(str(filename))
    #txt_vec = txt_to_vec(vid['title'])
    vid_dset[i] = vid_vec
    #txt_dset[i] = txt_vec
    i += 1
    
vid_vec_file.close()
#txt_vec_file.close()