import h5py
import os
import json
import operator

def store_horse():
    img_list = [f.encode("utf8") for f in os.listdir('.') if f.endswith('.jpg')]
    name_file = h5py.File('horses_name.hdf5', 'w')
    dt = h5py.special_dtype(vlen=bytes)
    name_dset = name_file.create_dataset('name', data=img_list, dtype=dt)
    print(name_dset.shape)
    print(name_dset[0])
    name_file.close()
    
def store_voc():
    print('Read json')
    with open('voc.json') as data_file:    
        data = json.load(data_file)
        
    sorted_data = sorted(data.items(), key=operator.itemgetter(1), reverse=True)
    data_size = len(sorted_data)

    print('create hdf5 file')
    name_file = h5py.File('voc_name.hdf5', 'w')
    dt = h5py.special_dtype(vlen=bytes)
    name_dset = name_file.create_dataset('name', (data_size,), dtype=dt)
    
    i = 0
    for (key, value) in sorted_data:
        name_dset[i] = key
        i += 1
        
    print(name_dset.shape)
    print(name_dset[0])
    name_file.close()
    
def merge():
    dset_name = 'name'
    input_file_1 = h5py.File('horses_name.hdf5', 'r')
    input_file_2 = h5py.File('voc_name.hdf5', 'r')
    output_file = h5py.File('full_name.hdf5', 'w')
    
    dset1 = input_file_1[dset_name]
    dset2 = input_file_2[dset_name]
    
    total_size = dset1.shape[0] + dset2.shape[0]
    dt = h5py.special_dtype(vlen=bytes)
    out_dset = output_file.create_dataset(dset_name, (total_size,), dtype=dt)
    
    i = 0
    for dat in dset1:
        out_dset[i] = dat
        i += 1
        
    for dat in dset2:
        out_dset[i] = dat
        i += 1
        
    output_file.close()
    input_file_1.close()
    input_file_2.close()
    