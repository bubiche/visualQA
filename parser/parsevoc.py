import json
from image2vec import yolo
import h5py
import os
import operator
import datetime

print('Read json')
with open('voc.json') as data_file:    
    data = json.load(data_file)

sorted_data = sorted(data.items(), key=operator.itemgetter(1), reverse=True)
data_size = len(sorted_data)

print('create hdf5 file')
img_vec_file = h5py.File('voc_vec_23.hdf5', 'w')
count_file = h5py.File('voc_count_23.hdf5', 'w')
img_dset = img_vec_file.create_dataset('vec', (data_size, 14, 14, 512), dtype='f')
count_dset = count_file.create_dataset('count', (data_size,), dtype='i')

print('Load YOLO...')
net = yolo.YOLO(
    'image2vec/yolo-full.cfg', 
    'image2vec/yolo-full.weights',
    up_to = 23)

i = 0
for (key, value) in sorted_data:
    vec = net.forward([key])
    img_dset[i] = vec[0]
    count_dset[i] = value
    i += 1
    if i % 100 == 0:
        print(i)
        print(datetime.datetime.now())
        
img_vec_file.close()
count_file.close()
    