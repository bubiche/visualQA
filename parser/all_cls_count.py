import h5py

count_file = h5py.File('full_count_voc.hdf5', 'a')

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

count_dset = []

for key, value in check.items():
    count_dset.append(count_file[key])
    
all_count_dset = count_file.create_dataset('all', (27088,), dtype='i')

for i in range(27088):
    tmp = 0
    for j in range(20):
        tmp += count_dset[j][i]
        
    all_count_dset[i] = tmp
    print(all_count_dset[i])
    
count_file.close()

