import numpy as np
import h5py

COUNT = 2538
class Splitter(object):

    def __init__(self):
        self.vec_file = h5py.File('full_vec.hdf5', 'r')
        self.count_file = h5py.File('full_count.hdf5', 'r')
        self.vec_dset = self.vec_file['vec']
        self.count_dset = self.count_file['count']
        tmp_vec = np.zeros((COUNT, 7, 7, 1024))
        tmp_count = np.zeros((COUNT,))
        
        print('Get all horse images')
        tmp_vec[0:1439] = np.array(self.vec_dset)[0:1439]
        tmp_count[0:1439] = np.array(self.count_dset)[0:1439]
        
        print('Get random')
        i = 1439
        while i < COUNT:
            random_idx = np.random.choice(list(range(1439, 27428)))
            tmp_vec[i] = np.array(self.vec_dset)[random_idx]
            tmp_count[i] = np.array(self.count_dset)[random_idx]
            i += 1
            
        self.vec_dset = tmp_vec
        self.count_dset = tmp_count
        
        self.data_size = self.get_data_size()
        self.n_use = -1
        if self.n_use < 1 or self.n_use > self.data_size: self.n_use = self.data_size
        self.n_val = int(self.n_use * 15 / 100)
        self.n_test = int(self.n_use * 15 / 100)
        self.train_size = self.n_use - self.n_val - self.n_test
        self.shuffle_data()
        self.create_val_test_files()
        
    def create_val_test_files(self):
        self.val_vec = h5py.File('val_vec.hdf5', 'w')
        self.val_count = h5py.File('val_count.hdf5', 'w')
        self.test_vec = h5py.File('test_vec.hdf5', 'w')
        self.test_count = h5py.File('test_count.hdf5', 'w')
        self.train_vec = h5py.File('train_vec.hdf5', 'w')
        self.train_count = h5py.File('train_count.hdf5', 'w')
        
        self.val_vec_dset = self.val_vec.create_dataset('vec', (self.n_val, 7, 7, 1024), dtype='f')
        self.val_count_dset = self.val_count.create_dataset('count', (self.n_val,), dtype='i')
        self.test_vec_dset = self.test_vec.create_dataset('vec', (self.n_test, 7, 7, 1024), dtype='f')
        self.test_count_dset = self.test_count.create_dataset('count', (self.n_test,), dtype='i')
        self.train_vec_dset = self.train_vec.create_dataset('vec', (self.train_size, 7, 7, 1024), dtype='f')
        self.train_count_dset = self.train_count.create_dataset('count', (self.train_size,), dtype='i')
        
        print('Dump train')
        i = 0
        while i < self.train_size:
            self.train_vec_dset[i] = self.get_x_at_index(self.shuffle_idx[i])
            self.train_count_dset[i] = self.get_annotation_at_index(self.shuffle_idx[i])
            i += 1
        
        print(self.train_vec_dset.shape)
        print(self.train_count_dset.shape)
        
        print('Dump val')
        i = 0
        while i < self.n_val:
            self.val_vec_dset[i] = self.get_x_at_index(self.shuffle_idx[i + self.train_size])
            self.val_count_dset[i] = self.get_annotation_at_index(self.shuffle_idx[i + self.train_size])
            i += 1
        
        print(self.val_vec_dset.shape)
        print(self.val_count_dset.shape)
        
        print('Dump test')
        i = 0
        while i < self.n_test:
            self.test_vec_dset[i] = self.get_x_at_index(self.shuffle_idx[i + self.train_size + self.n_val])
            self.test_count_dset[i] = self.get_annotation_at_index(self.shuffle_idx[i + self.train_size + self.n_val])
            i += 1
         
        print(self.test_vec_dset.shape)
        print(self.testcount_dset.shape)
        
        self.val_vec.close()
        self.val_count.close()
        self.test_vec.close()
        self.test_count.close()
        self.train_vec.close()
        self.train_count.close()
        
    def get_data_size(self):
        return self.vec_dset.shape[0]

    def shuffle_data(self):
        self.shuffle_idx = np.random.permutation(self.data_size)
        
    def shuffle_train(self):
        self.shuffle_idx_train = np.random.permutation(self.train_size)

    def get_x_at_index(self, idx):
        return self.vec_dset[idx]

    def get_annotation_at_index(self, idx):
        return self.count_dset[idx]
        
splitter = Splitter()