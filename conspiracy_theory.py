import json
import h5py
import shutil

data = [None] * 7

for i in range(7):
    file_name = '{}.json'.format(i)
    with open(file_name, 'r') as fp:
        data[i] = json.load(fp)
        
count_file = h5py.File('parser/full_count_voc.hdf5', 'r')
name_file = h5py.File('parser/full_name_voc.hdf5', 'r')
split_idx_file = h5py.File('parser/full_split_voc.hdf5', 'r')

count_dset = count_file['person']
name_dset = name_file['name']
test_idx_dset = split_idx_file['test']

img_id = 0
for idx in test_idx_dset:
    file_name = name_dset[idx].decode()
    true_count = count_dset[idx]
    
    count_list = [0] * 8
    for i in range(7):
        count_list[i] = data[i][file_name]    
    count_list[7] = true_count
    
    src_file = 'parser/{}'.format(file_name)
    dst_file = '{}-{}_{}_{}_{}_{}_{}_{}_{}'.format(img_id, count[0], count[1], count[2], count[3]
                                                    , count[4], count[5], count[6], count[7])
    shutil.copy(src_file, dst_file)
    img_id += 1

count_file.close()
name_file.close()
split_idx_file.close()

