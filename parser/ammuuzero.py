import h5py

def get_zero_percent(dset):
    size = dset.shape[0]
    print(size)
    count = 0.0
    for i in range(size):
        if dset[i] == 0:
            count += 1.0
            
    return count/size

count_file = h5py.File('full_count_voc.hdf5', 'r')
split_file = h5py.File('full_split_voc.hdf5', 'r')

test_idx_dset = split_file['test']
check = {}
'''
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
'''
check['furniture'] = 0
check['animal'] = 0
check['vehicle'] = 0
count_dset = {}
for key, value in check.items():
    count_dset[key] = count_file[key]
    
zero_count = {}
'''
zero_count['aeroplane'] = 0
zero_count['bicycle'] = 0
zero_count['bird'] = 0
zero_count['boat'] = 0
zero_count['bottle'] = 0
zero_count['bus'] = 0
zero_count['car'] = 0
zero_count['cat'] = 0
zero_count['chair'] = 0
zero_count['cow'] = 0
zero_count['diningtable'] = 0
zero_count['dog'] = 0
zero_count['horse'] = 0
zero_count['motorbike'] = 0
zero_count['person'] = 0
zero_count['pottedplant'] = 0
zero_count['sheep'] = 0
zero_count['sofa'] = 0
zero_count['train'] = 0
zero_count['tvmonitor'] = 0
'''
zero_count['furniture'] = 0
zero_count['animal'] = 0
zero_count['vehicle'] = 0

for idx in test_idx_dset:
    for key, value in zero_count.items():
        if count_dset[key][idx] == 0: zero_count[key] += 1

for key, value in zero_count.items():
    print(key, value/float(test_idx_dset.shape[0]))
        
count_file.close()
split_file.close()