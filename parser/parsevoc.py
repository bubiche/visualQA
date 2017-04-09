import json
from image2vec import yolo
import h5py
import os
import operator

print('Read json')
with open('voc.json') as data_file:    
    data = json.load(data_file)

print('create hdf5 file')
img_vec_file = h5py.File('voc_vec.hdf5', 'w')
count_file = h5py.File('voc_count.hdf5', 'w')
img_dset = img_vec_file.create_dataset('vec', (2198, 14, 14, 512), dtype='f')
count_dset = count_file.create_dataset('count', (2198,), dtype='i')

print('Load YOLO...')
net = yolo.YOLO(
    'image2vec/yolo-full.cfg', 
    'image2vec/yolo-full.weights',
    up_to = 23)

sorted_data = sorted(data.items(), key=operator.itemgetter(1), reverse=True)
i = 0
for (key, value) in sorted_data:
    vec = net.forward([key])
    img_dset[i] = vec[0]
    img_dset[i] = value
    i += 1
    if i % 100 == 0:
        print(i)
    if i = 2198:
        break
        
img_vec_file.close()
count_file.close()
    