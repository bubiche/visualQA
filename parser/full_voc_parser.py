import json
from image2vec import yolo
import h5py
import os
import operator
import datetime

print('Read json')
with open('voc_full.json') as data_file:    
    data = json.load(data_file)
    
data_size = len(data)
print(data_size)

check = {}
check['aeroplane'] = 0
check['bicycle'] = 1
check['bird'] = 2
check['boat'] = 3
check['bottle'] = 4
check['bus'] = 5
check['car'] = 6
check['cat'] = 7
check['chair'] = 8
check['cow'] = 9
check['diningtable'] = 10
check['dog'] = 11
check['horse'] = 12
check['motorbike'] = 13
check['person'] = 14
check['pottedplant'] = 15
check['sheep'] = 16
check['sofa'] = 17
check['train'] = 18
check['tvmonitor'] = 19

print('create hdf5 file')
img_vec_file = h5py.File('full_vec_voc.hdf5', 'w')
count_file = h5py.File('full_count_voc.hdf5', 'w')
name_file = h5py.File('full_name_voc.hdf5', 'w')
img_dset = img_vec_file.create_dataset('vec', (data_size, 7, 7, 1024), dtype='f')
count_dset = {}
for key, value in check.items():
    count_dset[key] = count_file.create_dataset(key, (data_size,), dtype='i')
dt = h5py.special_dtype(vlen=bytes)
name_dset = name_file.create_dataset('name', (data_size,), dtype=dt)
    
print('Load YOLO...')
net = yolo.YOLO(
    'image2vec/yolo-full.cfg', 
    'image2vec/yolo-full.weights',
    up_to = 28)
    
i = 0
for key, value in data.items:
    vec = net.forward([key])
    img_dset[i] = vec[0]
    name_dset[i] = key
    for key, value in check.items():
        count_dset[key][i] = value[key]
    i += 1
    if i % 100 == 0:
        print(i)
        print(datetime.datetime.now())
        
print(name_dset[0])
for key, value in check.items():
    print(key)
    print(count_dset[key][0])
