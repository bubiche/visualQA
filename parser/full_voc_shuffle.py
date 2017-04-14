import numpy as np
import h5py

name_file = h5py.File('full_name_voc.hdf5', 'r')
split_file = h5py.File('full_split_voc.hdf5', 'w')

name_dset = name_file['name']
data_size = name_dset.shape[0]
n_val = int(data_size * 15 / 100)
n_test = n_val
n_train = data_size - n_test - n_val

train_dset = split_file.create_dataset('train', (n_train,), dtype='i')
val_dset = split_file.create_dataset('val', (n_val,), dtype='i')
test_dset = split_file.create_dataset('test', (n_test,), dtype='i')

shuffle_idx = np.random.permutation(self.data_size)

i = 0
while i < n_train:
    train_dset[i] = shuffle_idx[i]
    i += 1

print(i)

while i < n_train + n_val:
    val_dset[i] = shuffle_idx[i]
    i += 1

print(i)
    
while i < n_train + n_val + n_test:
    test_dset[i] = shuffle_idx[i]
    i += 1
    
print(i)
    
